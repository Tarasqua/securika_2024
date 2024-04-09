"""
@tarasqua

Запуск демонстрации.
"""

import asyncio
import time
from asyncio import Queue
from asyncio import coroutines
from typing import List
from collections import deque
from pathlib import Path

from loguru import logger
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER

from utils.util import plot_crowds, plot_bboxes, plot_skeletons, set_yolo_model, UltralyticsLogsFilter
from crowd_detection.crowd_detector import CrowdDetector
from active_gestures_detection.active_gestures_detector import ActiveGesturesDetector
from raised_hands_detection.raised_hands_detector import RaisedHandsDetector
from squat_detection.squat_detector import SquatDetector


LOGGER.addFilter(UltralyticsLogsFilter())  # фильтруем забагованные логи


class Main:

    def __init__(self, font_path: str, bg_image_path: str, ):
        """
        Запуск демонстрации.
        :param font_path: Путь до шрифта.
        :param bg_image_path: Путь до изображения с фоном.
        """
        # название детектора, текст, координаты, вкл/выкл
        self.detectors_data = {
            'crowd': ['Скопления\nлюдей', (96, 320), False],
            'gestures': ['Активная\nжестикуляция', (75, 468), False],
            'hands': ['Поднятые\nруки', (106, 620), False],
            'squat': ['Человек на\nкорточках', (92, 770), False],
        }
        self.triggers_tl_positions = ((19, 460), (19, 925), (19, 1389))  # верхние левые координаты для сработок
        self.trigger_frame_shape = (0, 0)  # размер изображения с видеопотока после сжатия (нужно для отрисовки)
        self.font = ImageFont.truetype(font_path, 30)  # кнопки
        self.triggers_font = ImageFont.truetype(font_path, 20)  # сработки
        self.background_img = cv2.imread(bg_image_path)

        self.triggers_queue: Queue[str] = Queue(20)  # очередь для оповещения о сработке
        self.triggers_frames: deque[np.array] = deque(maxlen=3)  # кадры сработок
        self.triggers_data: deque[str] = deque(maxlen=3)  # информация о них
        self.detections_frame = np.array([])  # кадр с отрисованными детекциями
        self.fullscreen_mode = False  # полноэкранный режим для стрима
        self.fullscreen_triggers_mode = False, 0  # полноэкранный режим для сработок (вкл/выкл, какой)

        self.crowd_detector = CrowdDetector(self.triggers_queue)
        self.gestures_detector = ActiveGesturesDetector(self.triggers_queue)
        self.hands_detector = RaisedHandsDetector(self.triggers_queue)
        self.squat_detector = SquatDetector(self.triggers_queue)

    def click_event(self, event, x, y, flags, params) -> None:
        """
        Callback на кликер для определения координат, куда было совершено нажатие
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.fullscreen_mode and not self.fullscreen_triggers_mode[0]:
                if 35 <= x <= 309:  # смотрим, на какой детектор было нажатие
                    if 297 <= y <= 407:  # и инвертируем значение вкл/выкл
                        self.detectors_data['crowd'][-1] = not self.detectors_data['crowd'][-1]
                    elif 447 <= y <= 557:
                        self.detectors_data['gestures'][-1] = not self.detectors_data['gestures'][-1]
                    elif 597 <= y <= 707:
                        self.detectors_data['hands'][-1] = not self.detectors_data['hands'][-1]
                    elif 747 <= y <= 857:
                        self.detectors_data['squat'][-1] = not self.detectors_data['squat'][-1]
                elif 300 <= y <= 1060 and 460 <= x <= 1804:  # полноэкранный режим для стрима
                    self.fullscreen_mode = True
                elif 19 <= y <= 432:  # полноэкранный режим для сработок
                    if 460 <= x <= 693 and len(self.triggers_frames) > 0:
                        self.fullscreen_triggers_mode = True, 0
                    elif 925 <= x <= 1158 and len(self.triggers_frames) > 1:
                        self.fullscreen_triggers_mode = True, 1
                    elif 1389 <= x <= 1622 and len(self.triggers_frames) > 2:
                        self.fullscreen_triggers_mode = True, 2
            else:
                self.fullscreen_mode = False
                self.fullscreen_triggers_mode = False, 0

    def plot_additional_data(self, frame: np.array, triggers_reshape: tuple) -> np.array:
        """
        Отрисовка включенных/выключенных детекторов на фоновой картинке, а также сработки и подписи под ними.
        :param frame: Кадр, на который будут отрисованы данные по детекторам.
        :param triggers_reshape: Размеры, до которых ужимать триггерную картинку.
        :return: Фоновая картинка с отрисованными детекторами.
        """
        # отрисовываем сработки, если они есть
        for det_frame, (y, x) in zip(self.triggers_frames, self.triggers_tl_positions):
            resized_det = cv2.resize(det_frame, triggers_reshape)
            frame[y:y + self.trigger_frame_shape[0], x:x + self.trigger_frame_shape[1]] = resized_det
        # добавляем текст на кнопки и под сработки
        image = Image.fromarray(frame).convert("RGBA")
        txt_placeholder = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        for data, (y, x) in zip(self.triggers_data, self.triggers_tl_positions):  # под сработки
            draw.text((x, y + self.trigger_frame_shape[0] + 10), data,
                      font=self.triggers_font, fill=(255, 255, 255, 0))
        for data in self.detectors_data.values():  # на кнопки
            color = (255, 255, 255, 0) if data[-1] else (128, 128, 128, 0)
            draw.text(data[1], data[0], font=self.font, fill=color, align='center', spacing=10)
        combined = Image.alpha_composite(image, txt_placeholder)
        return cv2.cvtColor(np.array(combined), cv2.COLOR_BGRA2BGR)

    async def update_triggers(self) -> None:
        """
        Callback на сработку какого-либо детектора.
        :return: None.
        """
        while True:
            detector: str = await self.triggers_queue.get()  # ожидаем сработки
            await asyncio.sleep(0.5)  # искусственная задержка, чтобы данные успели отрисоваться
            self.triggers_frames.appendleft(self.detections_frame)  # добавляем картинку
            self.triggers_data.appendleft(  # какой детектор и время
                time.strftime("%H:%M:%S") + ' ' + self.detectors_data[detector][0].replace('\n', ' ')
            )
            logger.warning(f'{detector.capitalize()} TRIGGERED')  # отписываем в логи
            self.triggers_queue.task_done()  # помечаем таску выполненной
            await asyncio.sleep(0)  # освобождаем поток

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
        trigger_task = asyncio.create_task(self.update_triggers())
        cap = cv2.VideoCapture(stream_source)
        _, frame = cap.read()
        self.crowd_detector.set_frame_shape(frame.shape)
        reshape_main_frame = tuple((np.array(frame.shape[:-1][::-1]) / 2).astype(int))
        reshape_trigger = tuple((np.array(frame.shape[:-1][::-1]) / 6.5).astype(int))
        reshape_full_screen = tuple((np.array(self.background_img.shape[:-1][::-1])).astype(int))
        self.trigger_frame_shape = tuple(cv2.resize(frame, reshape_trigger).shape[:-1])

        cv2.namedWindow('main', cv2.WND_PROP_FULLSCREEN)
        cv2.setMouseCallback('main', self.click_event)
        cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        yolo_detector = set_yolo_model('n', 'pose', 'pose')
        for detections in yolo_detector.track(
                stream_source, classes=[0], stream=True, conf=0.5, verbose=False
        ):
            frame = await plot_skeletons(detections.orig_img, detections)
            self.detections_frame = await self.run_detectors(frame, detections)
            if self.fullscreen_mode:
                cv2.imshow('main', cv2.resize(self.detections_frame, reshape_full_screen))
            elif self.fullscreen_triggers_mode[0]:
                cv2.imshow('main',
                           cv2.resize(self.triggers_frames[self.fullscreen_triggers_mode[1]], reshape_full_screen))
            else:
                show_frame = self.background_img.copy()
                show_frame[300:reshape_main_frame[1] + 300, 460:reshape_main_frame[0] + 460] = (
                    cv2.resize(self.detections_frame, reshape_main_frame))
                cv2.imshow('main',
                           self.plot_additional_data(show_frame, reshape_trigger))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        trigger_task.cancel()


if __name__ == '__main__':
    main = Main((Path.cwd().parents[0] / 'resources' / 'fonts' / 'Gilroy-Regular.ttf').as_posix(),
                (Path.cwd().parents[0] / 'resources' / 'images' / 'background.png').as_posix())
    asyncio.run(main.main('rtsp://admin:Qwer123@192.168.0.108?subtype=1'))
