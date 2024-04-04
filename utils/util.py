import asyncio
import os
import fnmatch
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import ops
from ultralytics import YOLO
from ultralytics.engine.results import Results


def find_file(pattern: str, path: str) -> str:
    """
    Поиск файла по паттерну (первое вхождение).
    :param pattern: Паттерн для поиска.
    :param path: Корневой путь для поиска.
    :return: Путь из корневой папки до найденного файла (если файл не найден, возвращается пустая строка).
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return ''


def iou(bbox1: np.array, bbox2: np.array) -> np.float32:
    """
    Вычисляет IOU двух входных bbox'ов.
    :param bbox1: Координаты 1 bbox'a в формате np.array([x1, y1, x2, y2]).
    :param bbox2: Координаты 2 bbox'a в формате np.array([x1, y1, x2, y2]).
    :return: IOU.
    """
    return ops.box_iou(
        torch.from_numpy(np.array([bbox1])),
        torch.from_numpy(np.array([bbox2])),
    ).numpy()[0][0]


def angle_degrees_2d(p1: np.array, p2: np.array, p3: np.array) -> float:
    """
    Нахождение угла между тремя точками (в формате [x, y]) на плоскости в градусах.
    :param p1: Координаты первой точки на плоскости.
    :param p2: Координаты второй точки на плоскости.
    :param p3: Координаты третьей точки на плоскости.
    :return: Угол между тремя точками в градусах.
    """
    if np.array([None, None]) in np.array([p1, p2, p3]):
        return 0
    v1, v2 = p1 - p2, p3 - p2
    return float(np.degrees(np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )))


async def get_hands_angles(detections: Results, kpts_confidence: float = 0.5) -> np.array:
    """
    Расчет углов (в градусах) в локтях и плечах затреченных людей.
    :param detections: YOLO detections.
    :param kpts_confidence: Confidence для ключевых точек.
    :return: Углы в порядке: [[id, левый локоть, левое плечо, правое плечо, правый локоть], [...]]
    """
    def angles_(detection: Results) -> np.array:
        """
        Поиск углов по одному человеку
        :param detection: YOLO detection.
        :return: Углы в порядке: [id, левый локоть, левое плечо, правое плечо, правый локоть]
        """
        if (id_data := detection.boxes.id) is None:
            return np.array([])
        id_ = id_data.numpy()[0]
        key_points = [kp[:-1] if kp[-1] > kpts_confidence
                      else np.array([None, None])  # если conf ниже порогового
                      for kp in detection.keypoints.data.numpy()[0]]  # ключевые точки с порогом
        left_elbow = angle_degrees_2d(key_points[5], key_points[7], key_points[9])  # левый локоть
        right_elbow = angle_degrees_2d(key_points[6], key_points[8], key_points[10])  # правый локоть
        left_shoulder = angle_degrees_2d(key_points[11], key_points[5], key_points[7])  # левое плечо
        right_shoulder = angle_degrees_2d(key_points[12], key_points[6], key_points[8])  # правое плечо
        return np.array([id_, left_elbow, left_shoulder, right_shoulder, right_elbow])

    angles_tasks = [asyncio.to_thread(angles_, detection) for detection in detections]
    people_data = await asyncio.gather(*angles_tasks)
    return np.array([data for data in people_data if data.size != 0])  # убираем тех, у кого не нашли id


async def get_legs_angles(detections: Results, kpts_confidence: float = 0.5) -> np.array:
    """
    Расчет углов (в градусах) в коленях затреченных людей.
    :param detections: YOLO detections.
    :param kpts_confidence: Confidence для ключевых точек.
    :return: Углы в порядке: [[id, левое колено, левое бедро, правое бедро, правое колено], [...]]
    """
    def angles_(detection: Results) -> np.array:
        """
        Поиск углов по одному человеку
        :param detection: YOLO detection.
        :return: Углы в порядке: [id, левый локоть, левое плечо, правое плечо, правый локоть]
        """
        if (id_data := detection.boxes.id) is None:
            return np.array([])
        id_ = id_data.numpy()[0]
        key_points = [kp[:-1] if kp[-1] > kpts_confidence
                      else np.array([None, None])  # если conf ниже порогового
                      for kp in detection.keypoints.data.numpy()[0]]  # ключевые точки с порогом
        left_knee = angle_degrees_2d(key_points[11], key_points[13], key_points[15])  # левое колено
        right_knee = angle_degrees_2d(key_points[12], key_points[14], key_points[16])  # правое колено
        left_hip = angle_degrees_2d(key_points[5], key_points[11], key_points[13])  # левое бедро
        right_hip = angle_degrees_2d(key_points[6], key_points[12], key_points[14])  # правое бедро
        return np.array([id_, left_knee, left_hip, right_hip, right_knee])

    angles_tasks = [asyncio.to_thread(angles_, detection) for detection in detections]
    people_data = await asyncio.gather(*angles_tasks)
    return np.array([data for data in people_data if data.size != 0])  # убираем тех, у кого не нашли id


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
    return YOLO(f'{model_path}.onnx', task=task, verbose=False)


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
