import asyncio
from asyncio import Queue
from asyncio import coroutines
from typing import List

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics.engine.results import Results

from utils.util import plot_crowds, plot_bboxes, plot_skeletons, set_yolo_model
from crowd_detection.crowd_detector import CrowdDetector
from active_gestures_detection.active_gestures_detector import ActiveGesturesDetector
from raised_hands_detection.raised_hands_detector import RaisedHandsDetector
from squat_detection.squat_detector import SquatDetector


class Main:

    def __init__(self):
        # название детектора, текст, координаты, вкл/выкл
        self.detectors_data = {
            'crowd': ['Скопления\nлюдей', (96, 320), False],
            'gestures': ['Активная\nжестикуляция', (75, 468), False],
            'hands': ['Поднятые\nруки', (106, 620), False],
            'squat': ['Человек на\nкорточках', (92, 770), False],
        }
        self.font = ImageFont.truetype('../resources/fonts/Gilroy-Regular.ttf', 30)
        self.background_img = cv2.imread('../resources/images/background.png')
        self.triggers_queue: Queue[str] = Queue(20)

        self.crowd_detector = CrowdDetector()
        self.gestures_detector = ActiveGesturesDetector()
        self.hands_detector = RaisedHandsDetector()
        self.squat_detector = SquatDetector()

    def click_event(self, event, x, y, flags, params) -> None:
        """Кликер для определения координат"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if 35 <= x <= 309:  # смотрим, на какой детектор было нажатие
                if 297 <= y <= 407:  # и инвертируем значение вкл/выкл
                    self.detectors_data['crowd'][-1] = not self.detectors_data['crowd'][-1]
                elif 447 <= y <= 557:
                    self.detectors_data['gestures'][-1] = not self.detectors_data['gestures'][-1]
                elif 597 <= y <= 707:
                    self.detectors_data['hands'][-1] = not self.detectors_data['hands'][-1]
                elif 747 <= y <= 857:
                    self.detectors_data['squat'][-1] = not self.detectors_data['squat'][-1]

    def plot_detectors_data(self, frame: np.array) -> np.array:
        """
        Отрисовка включенных/выключенных детекторов на фоновой картинке.
        :param frame: Кадр, на который будут отрисованы данные по детекторам.
        :return: Фоновая картинка с отрисованными детекторами.
        """
        image = Image.fromarray(frame).convert("RGBA")
        txt_placeholder = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        for data in self.detectors_data.values():
            color = (255, 255, 255, 0) if data[-1] else (128, 128, 128, 0)
            draw.text(data[1], data[0], font=self.font, fill=color, align='center', spacing=10)
        combined = Image.alpha_composite(image, txt_placeholder)
        return np.array(combined)

    async def run_detectors(self, frame: np.array, detections: Results) -> np.array:
        """
        Запустить обработку в тех детекторах, которые включены, с отрисовкой на текущем кадре.
        :param frame: Текущий кадр из видеопотока.
        :param detections: YOLO detections.
        :return: Кадр с отрисованными детекциями.
        """
        if self.detectors_data['crowd'][-1]:
            crowds = await self.crowd_detector.detect_(detections)
            frame = await plot_crowds(frame, crowds)

        running: List[coroutines] = []
        if self.detectors_data['gestures'][-1]:
            running.append(self.gestures_detector.detect_(detections))
        if self.detectors_data['hands'][-1]:
            running.append(self.hands_detector.detect_(detections))
        if self.detectors_data['squat'][-1]:
            running.append(self.squat_detector.detect_(detections))
        results = await asyncio.gather(*running)
        if results and (filtered_results := [result for result in results if result.size != 0]):
            frame = await plot_bboxes(frame, np.concatenate(filtered_results)[:, 1:])
        return frame

    async def main(self, stream_source):
        """
        Запуск детекторов.
        :param stream_source:
        :return:
        """
        cap = cv2.VideoCapture(stream_source)
        cv2.namedWindow('main', cv2.WND_PROP_FULLSCREEN)
        cv2.setMouseCallback('main', self.click_event)
        cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        _, frame = cap.read()
        self.crowd_detector.set_frame_shape(frame.shape)
        reshape = tuple((np.array(frame.shape[:-1][::-1]) / 2).astype(int))
        yolo_detector = set_yolo_model('n', 'pose', 'pose')
        for detections in yolo_detector.track(
                stream_source, classes=[0], stream=True, conf=0.5, verbose=False
        ):
            frame = await plot_skeletons(detections.orig_img, detections)
            frame = await self.run_detectors(frame, detections)

            show_frame = self.background_img.copy()
            show_frame[270:1030, 460:1804] = cv2.resize(frame, reshape)
            cv2.imshow('main', self.plot_detectors_data(show_frame))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = Main()
    asyncio.run(main.main('rtsp://admin:Qwer123@192.168.0.108?subtype=1'))
