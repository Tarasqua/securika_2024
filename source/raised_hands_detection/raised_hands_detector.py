"""
@tarasqua

Детектор поднятых рук.
Выставочная реализация, используя лишь пороги.
"""

import numpy as np
from ultralytics.engine.results import Results

from source.config_loader import Config
from utils.util import get_hands_angles


class RaisedHandsDetector:
    """
    Детектор поднятых рук.
    """

    def __init__(self):
        self.config_ = Config()
        self.config_.initialize('raised_hands')

    def analyze_people(self, angles_data: np.array, detections: Results):
        """
        Проверка углов рук людей и выявление тех, у кого руки подняты.
        :param angles_data: Данные по углам на текущем кадре в формате:
            [[id, левый локоть, левое плечо, правое плечо, правый локоть], [...]].
        :param detections: YOLO detections.
        :return: Список людей с поднятыми руками и их ббоксы в формате: [[id, x1, y1, x2, y2], [...]]
            (пустой список, если ничего не нашли).
        """
        # проверка на то, что-либо руки вытянуты вверх в плечах, либо человек как бы сдается
        if (filtered_angles_data := angles_data[
            (np.any(angles_data[:, 2:4] > self.config_.get('SHOULDERS_ANGLE_THRESHOLD'))) |  # вытянуты
            (np.any(angles_data[:, [1, -1]] < self.config_.get('ELBOW_BENT_ANGLE_THRESHOLD')) and np.any(  # сдается
                angles_data[:, 2:4] > self.config_.get('SHOULDERS_BENT_ANGLE_THRESHOLD')))
        ]).size == 0:
            return np.array([])
        raised_hands_ids = filtered_angles_data[0][:, 0]  # айдишники тех, у кого руки подняты
        return np.array([[id_, *det.boxes.xyxy.numpy()[0].astype(int)]  # и берем их ббоксы
                         for det in detections
                         if (id_data := det.boxes.id) is not None  # на всякий случай перепроверяем
                         and (id_ := id_data.numpy()[0]) in raised_hands_ids])

    async def detect_(self, detections: Results) -> np.array:
        """
        Обработка YOLO-pose-треков для нахождения людей с поднятыми руками в кадре.
        :param detections: YOLO detections.
        :return: Список людей с поднятыми руками и их ббоксы в формате: [[id, x1, y1, x2, y2], [...]]
            (пустой список, если ничего не нашли).
        """
        if (len(detections) != 0 and
                (angles := await get_hands_angles(detections, self.config_.get('KEY_POINTS_CONFIDENCE'))).size != 0):
            return self.analyze_people(angles, detections)
        return np.array([])
