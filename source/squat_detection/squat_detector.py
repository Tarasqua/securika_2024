"""
@tarasqua

Детектор человека на корточках.
Выставочная реализация, используя лишь пороги.
"""

import numpy as np
from ultralytics.engine.results import Results

from source.config_loader import Config
from utils.util import get_legs_angles


class SquatDetector:
    """
    Детектор людей на корточках.
    """

    def __init__(self):
        self.config_ = Config()
        self.config_.initialize('squat')

    def analyze_people(self, angles_data: np.array, detections: Results):
        """
        Проверка углов рук людей и выявление тех, у кого руки подняты.
        :param angles_data: Данные по углам на текущем кадре в формате:
            [[id, левый локоть, левое плечо, правое плечо, правый локоть], [...]].
        :param detections: YOLO detections.
        :return: Список людей на корточках и их ббоксы в формате: [[id, x1, y1, x2, y2], [...]]
            (пустой список, если ничего не нашли).
        """
        if (filtered_angles_data := angles_data[  # проверка на то что ноги видны и что человек сидит
            np.all((angles_data[:, 1:] < self.config_.get('ANGLES_THRESHOLD')) & (angles_data[:, 1:] != 0))]
        ).size == 0:
            return np.array([])
        squat_ids = filtered_angles_data[0][:, 0]  # айдишники тех, кто на корточках
        return np.array([[id_, *det.boxes.xyxy.numpy()[0].astype(int)]  # и берем их ббоксы
                         for det in detections
                         if (id_data := det.boxes.id) is not None  # на всякий случай перепроверяем
                         and (id_ := id_data.numpy()[0]) in squat_ids])

    async def detect_(self, detections: Results) -> np.array:
        """
        Обработка YOLO-pose-треков для нахождения людей на корточках в кадре.
        :param detections: YOLO detections.
        :return: Список людей на корточках и их ббоксы в формате: [[id, x1, y1, x2, y2], [...]]
            (пустой список, если ничего не нашли).
        """
        if (len(detections) != 0 and
                (angles := await get_legs_angles(detections, self.config_.get('KEY_POINTS_CONFIDENCE'))).size != 0):
            return self.analyze_people(angles, detections)
        return np.array([])