"""
@tarasqua

Детектор человека на корточках.
Выставочная реализация, используя лишь пороги.
"""

from asyncio import Queue

from loguru import logger
import numpy as np
from ultralytics.engine.results import Results

from utils.config_loader import Config
from utils.util import get_legs_angles


class SquatDetector:
    """
    Детектор людей на корточках.
    """

    def __init__(self, triggers_queue: Queue):
        self.triggers_queue: Queue[str] = triggers_queue  # очередь для оповещения о сработке
        self.config_ = Config()
        self.config_.initialize('squat')
        self.prev_ids = np.array([])  # id людей на корточках на предыдущем кадре
        logger.success('Squat detector successfully initialized')

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
        if (len(detections) != 0 and  # находим углы в ногах
                (angles := await get_legs_angles(detections, self.config_.get('KEY_POINTS_CONFIDENCE'))).size != 0):
            # находим тех, кто на корточках
            if (squatting := self.analyze_people(angles, detections)).size == 0:
                return np.array([])
            # смотрим, есть ли в людях на корточках новые id и, если да, считаем это как новую сработку
            if squatting[:, 0][~np.in1d(squatting[:, 0], self.prev_ids)].size != 0:
                await self.triggers_queue.put('squat')
            self.prev_ids = squatting[:, 0]
            return squatting
        self.prev_ids = np.array([])  # если ничего не найдено на текущем кадре
        return np.array([])
