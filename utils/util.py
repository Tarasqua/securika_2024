import asyncio
import os
import fnmatch
from pathlib import Path
from typing import List

import cv2
import torch
import numpy as np
from torchvision import ops
from ultralytics import YOLO
from ultralytics.engine.results import Results

KEY_POINTS = [  # наименования ключевых точек для YOLO по порядку
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
    'right_ankle'
]
LIMBS = (  # конечностей, заключенные между ключевых точек
    (KEY_POINTS.index('right_eye'), KEY_POINTS.index('nose')),
    (KEY_POINTS.index('right_eye'), KEY_POINTS.index('right_ear')),
    (KEY_POINTS.index('left_eye'), KEY_POINTS.index('nose')),
    (KEY_POINTS.index('left_eye'), KEY_POINTS.index('left_ear')),
    (KEY_POINTS.index('right_shoulder'), KEY_POINTS.index('right_elbow')),
    (KEY_POINTS.index('right_elbow'), KEY_POINTS.index('right_wrist')),
    (KEY_POINTS.index('left_shoulder'), KEY_POINTS.index('left_elbow')),
    (KEY_POINTS.index('left_elbow'), KEY_POINTS.index('left_wrist')),
    (KEY_POINTS.index('right_hip'), KEY_POINTS.index('right_knee')),
    (KEY_POINTS.index('right_knee'), KEY_POINTS.index('right_ankle')),
    (KEY_POINTS.index('left_hip'), KEY_POINTS.index('left_knee')),
    (KEY_POINTS.index('left_knee'), KEY_POINTS.index('left_ankle')),
    (KEY_POINTS.index('right_shoulder'), KEY_POINTS.index('left_shoulder')),
    (KEY_POINTS.index('right_hip'), KEY_POINTS.index('left_hip')),
    (KEY_POINTS.index('right_shoulder'), KEY_POINTS.index('right_hip')),
    (KEY_POINTS.index('left_shoulder'), KEY_POINTS.index('left_hip'))
)
PALETTE = np.array([  # цветовая палитра для ключевых точек и конечностей
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255], [153, 204, 255], [255, 102, 255],
    [255, 51, 255], [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153],
    [102, 255, 102], [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]])
LIMBS_COLORS = PALETTE[[16, 16, 16, 16, 9, 9, 9, 9, 0, 0, 0, 0, 7, 7, 7, 7]]  # цвета для конечностей
KPTS_COLORS = PALETTE[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]  # цвета для ключевых точек


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


async def plot_crowds(frame: np.array, crowds: np.array) -> np.array:
    """
    Отрисовка скоплений людей с заливкой ббокса.
    :param frame: Кадр для отрисовки.
    :param crowds: Данные по скоплениям людей в формате [[x1, y1, x2, y2], [...]].
    :return: Кадр с отрисованными скоплениями людей.
    """

    async def make_crowds_overlay(crowds_data: np.array) -> np.array:
        """
        Формирование полупрозрачного слоя с ббоксами скоплений для наложения.
        :param crowds_data: Данные по скоплениям людей.
        :return: Полупрозрачный слой.
        """
        overlay_ = frame.copy()
        crowds_tasks = [asyncio.to_thread(
            cv2.rectangle, overlay_, crowd[:2].astype(int), crowd[2:].astype(int), (0, 0, 255), -1, 8, 0
        ) for crowd in crowds_data]
        await asyncio.gather(*crowds_tasks)
        return overlay_

    if crowds.size != 0:
        overlay = await make_crowds_overlay(crowds)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    return frame


async def plot_bboxes(frame: np.array, bboxes: np.array) -> np.array:
    """
    Отрисовка ббоксов людей, на которых сработали детекторы.
    :param frame: Кадр для отрисовки.
    :param bboxes: Ббоксы людей в формате [[x1, y1, x2, y2], [...]].
    :return: Кадр с отрисованными людьми.
    """
    if bboxes.size != 0:
        bboxes_tasks = [asyncio.to_thread(
            cv2.rectangle, frame, bbox[:2].astype(int), bbox[2:].astype(int), (0, 0, 255), 4, 8, 0
        ) for bbox in bboxes]
        await asyncio.gather(*bboxes_tasks)
    return frame


async def plot_skeletons(frame: np.array, detections: Results, conf: float = 0.5) -> np.array:
    """
    Отрисовка скелетов людей в текущем кадре.
    :param frame: Кадр для отрисовки.
    :param detections: YOLO detections.
    :param conf: Порог по confidence.
    :return: Кадр c отрисованными скелетами людей.
    """

    async def plot_kpts(points: np.array) -> None:
        """
        Отрисовка ключевых точек человека на кадре.
        :param points: Пронумерованные координаты точек в формате [[i, x, y], [...]].
        :return: None.
        """
        circle_tasks = [asyncio.to_thread(  # отрисовываем точки
            cv2.circle, frame, (x, y), 5, tuple(map(int, KPTS_COLORS[i])), -1, 8, 0)
            for i, x, y in points.astype(int)]
        await asyncio.gather(*circle_tasks)

    async def plot_limbs(points: np.array) -> None:
        """
        Отрисовка конечностей человека на кадре.
        :param points: Пронумерованные координаты точек в формате [[i, x, y], [...]].
        :return: None.
        """
        # берем только те конечности, точки которых прошли фильтрацию по conf
        filtered_limbs = [limb for limb in LIMBS if np.all(np.in1d(limb, points[:, 0]))]
        limbs_tasks = [asyncio.to_thread(  # отрисовываем конечности
            cv2.line, frame,
            points[:, 1:][points[:, 0] == p1].astype(int)[0], points[:, 1:][points[:, 0] == p2].astype(int)[0],
            tuple(map(int, LIMBS_COLORS[i])), 2, 8, 0
        ) for i, (p1, p2) in enumerate(filtered_limbs)]
        await asyncio.gather(*limbs_tasks)

    if len(detections) == 0:
        return frame
    # номеруем и фильтруем по conf ключевые точки (нумерация нужна, чтобы после фильтрации не потерять порядок)
    people_kpts = [(points := np.c_[np.arange(17), kpt])[points[:, 3] >= conf][:, :-1]  # берем только i и точки
                   for kpt in detections.keypoints.data.numpy()]
    for kpts in people_kpts:  # отрисовываем по каждому человеку
        await asyncio.gather(plot_kpts(kpts), plot_limbs(kpts))
    return frame
