"""
@tarasqua

Детектор активной жестикуляции.
Выставочная реализация, без использования таймера и сбора нормальной статистики по углам (лишь пороги).
"""

from asyncio import Queue

from loguru import logger
import numpy as np
from ultralytics.engine.results import Results

from utils.config_loader import Config
from utils.util import get_hands_angles


class ActiveGesturesDetector:
    """
    Детектор активной жестикуляции.
    """

    def __init__(self, triggers_queue: Queue):
        self.triggers_queue: Queue[str] = triggers_queue  # очередь для оповещения о сработке
        self.config_ = Config()
        self.config_.initialize('active_gestures')
        # id, mean left angle, mean right angle, active gestures count, obs time
        self.people_data: np.array = np.array([], dtype=np.int64).reshape(0, 5)
        self.prev_ids = np.array([])  # id людей, которые активно жестикулировали на предыдущем кадре
        logger.success('Active gestures detector successfully initialized')

    def filter_people_data(self, detections: Results) -> None:
        """
        Отфильтровываем тех, кто:
            1. не нашелся в текущем кадре;
            2. тех, за которыми давно следим (для удобства слежения в определенном интервале времени и
                сбрасывания детекции).
        :param detections: YOLO detections.
        :return: None.
        """
        cur_frame_ids = [id_.numpy()[0] for det in detections if (id_ := det.boxes.id) is not None]
        self.people_data = self.people_data[  # оставляем только те данные по людям,
            np.in1d(self.people_data[:, 0], cur_frame_ids) &  # что нашлись в данном кадре
            (self.people_data[:, -1] < self.config_.get('MAX_OBSERVATION_INTERVAL'))]  # и мало наблюдаем

    def update_people_data(self, new_angles_data: np.array) -> None:
        """
        Обновление данных по наблюдаемым людям с проверкой на резкое изменение углов в руках.
        :param new_angles_data: Данные по углам на текущем кадре в формате:
            [[id, левый локоть, левое плечо, правое плечо, правый локоть], [...]].
        :return: None.
        """
        for data in new_angles_data:
            if (id_ := data[0]) not in self.people_data[:, 0]:  # если человек с данным id еще не наблюдался
                self.people_data = np.vstack([  # добавляем дефолтную информацию по нему
                    self.people_data, np.array([id_, np.mean(data[1:3]), np.mean(data[3:]), 0, 1])])
                continue
            diff = np.absolute(  # иначе же находим изменение угла в руках по нему
                (old_data := self.people_data[self.people_data[:, 0] == id_][0])[1:3] -
                np.array([(new_left := np.mean(data[1:3])), (new_right := np.mean(data[3:]))])
            )
            if any(diff > self.config_.get('DELTA_ANGLE_THRESHOLD')):  # делаем проверку по порогу
                old_data[-2] += 1  # меняем количество активных жестикуляций
            self.people_data[self.people_data[:, 0] == id_] = (  # обновляем данные по человеку
                np.array([id_, new_left, new_right, old_data[-2], old_data[-1] + 1]))

    def get_actively_gesturing(self, detections: Results) -> np.array:
        """
        Находим активно жестикулирующих людей.
        :param detections: YOLO detections.
        :return: Список активно жестикулирующих людей и их ббоксы в формате: [[id, x1, y1, x2, y2], [...]]
            (пустой список, если ничего не нашли).
        """
        # находим айдишники людей, которые активно жестикулируют
        act_gest_ids = self.people_data[self.people_data[:, -2] > self.config_.get('MAX_ACTIVE_GESTURES')][:, 0]
        return np.array([[id_, *det.boxes.xyxy.numpy()[0].astype(int)]  # и берем их ббоксы
                         for det in detections
                         if (id_data := det.boxes.id) is not None  # на всякий случай перепроверяем
                         and (id_ := id_data.numpy()[0]) in act_gest_ids])

    async def detect_(self, detections: Results) -> np.array:
        """
        Обработка YOLO-pose-треков для нахождения активно жестикулирующих людей в видеопотоке.
        :param detections: YOLO detections.
        :return: Список активно жестикулирующих людей и их ббоксы в формате: [[id, x1, y1, x2, y2], [...]]
            (пустой список, если ничего не нашли).
        """
        if len(detections) != 0:
            self.filter_people_data(detections)  # фильтруем сработки
            curr_angles = await get_hands_angles(  # находим углы в руках
                detections, self.config_.get('KEY_POINTS_CONFIDENCE'))
            self.update_people_data(curr_angles)  # обновляем данные по людям
            # находим тех, кто активно жестикулирует
            if (actively_gesturing := self.get_actively_gesturing(detections)).size == 0:
                return np.array([])
            # смотрим, есть ли в жестикулирующих новые id и, если да, считаем это как новую сработку
            if actively_gesturing[:, 0][~np.in1d(actively_gesturing[:, 0], self.prev_ids)].size != 0:
                await self.triggers_queue.put('gestures')
            self.prev_ids = actively_gesturing[:, 0]
            return actively_gesturing
        self.prev_ids = np.array([])  # если ничего не найдено на текущем кадре
        return np.array([])
