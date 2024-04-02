import os
import asyncio
from pathlib import Path

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO

from crowd_detection.crowd_detector import CrowdDetector


def set_yolo_model(yolo_model: str, yolo_class: str, task: str = 'detect') -> YOLO:
    """
    Выполняет проверку путей и наличие модели:
        Если директория отсутствует, создает ее, а также скачивает в нее необходимую модель
    :param yolo_model: n (nano), m (medium), etc.
    :param yolo_class: seg, pose, boxes
    :param task: detect, segment, classify, pose
    :return: Объект YOLO-pose
    """
    yolo_class = f'-{yolo_class}' if yolo_class != 'boxes' else ''
    yolo_models_path = Path.cwd().parents[1] / 'resources' / 'models' / 'yolo_models'
    if not os.path.exists(yolo_models_path):
        Path(yolo_models_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(yolo_models_path, f'yolov8{yolo_model}{yolo_class}')
    if not os.path.exists(f'{model_path}.onnx'):
        YOLO(model_path).export(format='onnx')
    return YOLO(f'{model_path}.onnx', task=task)


def get_writer(file_name: str, im_shape: tuple[int, int]) -> cv2.VideoWriter:
    """
    Запись демо работы детектора.
    :param file_name: Наименование файла.
    :param im_shape: Разрешение входного видеопотока.
    :return: Объект класса VideWriter.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(f'{file_name}', fourcc, 30.0, im_shape)


async def plot_bboxes(frame: np.array, bboxes: np.array) -> np.array:
    """
    Отрисовка ббоксов.
    :param frame: Кадр для рисования.
    :param bboxes: Ббоксы в формате [[x1, y1, x2, y2], [...]].
    :return: Кадр с отрисованными ббоксами.
    """
    def plot_(bbox: np.array):
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

    tasks = [asyncio.to_thread(plot_, bbox) for bbox in bboxes]
    await asyncio.gather(*tasks)
    return frame


async def main(stream_source):
    """
    Запуск детекторов.
    :param stream_source:
    :return:
    """
    cap = cv2.VideoCapture(stream_source)
    _, frame = cap.read()
    crowd_ = CrowdDetector(frame.shape)
    reshape = tuple((np.array(frame.shape[:-1][::-1]) / 2).astype(int))
    # writer = get_writer('handrail demo 2.mp4', frame.shape[:-1][::-1])
    yolo_detector = set_yolo_model('n', 'pose', 'pose')
    for detections in yolo_detector.track(
            stream_source, classes=[0], stream=True, conf=0.1, verbose=False
    ):
        # frame = plot_detection(detections) if len(detections) else detections.orig_img
        # writer.write(frame)
        bboxes: np.array = await crowd_.detect_(detections)
        frame = await plot_bboxes(detections.plot().copy(), bboxes)
        cv2.imshow('main', cv2.resize(frame, reshape))
        if cv2.waitKey(1) & 0xFF == 27:
            break
    # writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    asyncio.run(main('pedestrians.mp4'))
