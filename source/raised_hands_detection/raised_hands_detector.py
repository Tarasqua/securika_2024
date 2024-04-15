"""
@tarasqua

Детектор поднятых рук.
Выставочная реализация, используя лишь пороги.
"""

from asyncio import Queue

from loguru import logger
import numpy as np
from ultralytics.engine.results import Results

from utils.config_loader import Config
from utils.util import get_hands_angles


class RaisedHandsDetector:
    """
    Детектор поднятых рук.
    """

    def __init__(self, triggers_queue: Queue):
        self.triggers_queue: Queue[str] = triggers_queue  # очередь для оповещения о сработке
        self.config_ = Config()
        self.config_.initialize('raised_hands')
        self.prev_ids = np.array([])  # id людей с поднятыми руками на предыдущем кадре
        logger.success('Raised hands detector successfully initialized')

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
            (((angles_data[:, 2] > self.config_.get('SHOULDERS_BENT_ANGLE_THRESHOLD')) |
             (angles_data[:, 3] > self.config_.get('SHOULDERS_BENT_ANGLE_THRESHOLD')))
            & ((angles_data[:, 1] < self.config_.get('ELBOW_BENT_ANGLE_THRESHOLD')) |
               (angles_data[:, -1] < self.config_.get('ELBOW_BENT_ANGLE_THRESHOLD')))) |
            (((angles_data[:, 2] > self.config_.get('SHOULDERS_ANGLE_THRESHOLD')) |
             (angles_data[:, 3] > self.config_.get('SHOULDERS_ANGLE_THRESHOLD'))))
        ]).size == 0:
            return np.array([])
        raised_hands_ids = filtered_angles_data[:, 0]  # айдишники тех, у кого руки подняты
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
        if (len(detections) != 0 and  # находим углы в руках
                (angles := await get_hands_angles(detections, self.config_.get('KEY_POINTS_CONFIDENCE'))).size != 0):
            # смотрим, у кого в кадре подняты руки
            if (raising_hands := self.analyze_people(angles, detections)).size == 0:
                return np.array([])
            # смотрим, есть ли в жестикулирующих новые id и, если да, считаем это как новую сработку
            if raising_hands[:, 0][~np.in1d(raising_hands[:, 0], self.prev_ids)].size != 0:
                await self.triggers_queue.put('hands')
            self.prev_ids = raising_hands[:, 0]
            return raising_hands
        self.prev_ids = np.array([])  # если ничего не найдено на текущем кадре
        return np.array([])
