"""
@tarasqua

Детектор скопления людей.
Выставочная реализация, без использования трехмерных сеток, а используя лишь KMeans и YOLO-pose.
"""

from typing import Tuple, List
from itertools import combinations, chain

import numpy as np
from ultralytics.engine.results import Results

from source.config_loader import Config
from source.crowd_detection.kmeans_separator import kmeans_fit
from utils.util import iou


class CrowdDetector:
    """
    Детектор скопления людей.
    """

    def __init__(self, frame_shape: Tuple[int, ...]):
        self.frame_shape = frame_shape[:-1][::-1]
        self.config_ = Config()
        self.config_.initialize('crowd')
        self.kmeans_ = kmeans_fit(None, self.config_.get('SCENE_SIZE'))
        self.prev_ids = np.array([])

    def get_bbox_centroid(self, data: np.array, to_absolute: bool = False) -> Tuple[np.array, np.array]:
        """
        Нахождение ббокса и центроида.
        :param data: Данные в формате [group, human_id, x1, y1, x2, y2, c_x, c_y].
        :param to_absolute: Переводить обратно в абсолютные координаты или нет.
        :return: Ббокс [x1, y1, x2, y2] и центроид [x, y].
        """
        bbox_ = np.array([
            [min(data[:, 2]), min(data[:, 3])],  # x1, y1
            [max(data[:, 4]), max(data[:, 5])],  # x2, y2
        ])
        bbox_ = bbox_ * self.frame_shape if to_absolute else bbox_
        return np.concatenate(bbox_), bbox_.sum(axis=0) / 2

    def get_kpts_centroid(self, detection: Results) -> np.array:
        """
        Возвращает центроид человека относительно видимости ключевых точек.
        :param detection: YOLO detection.
        :return: Id + координаты центроида в относительных координатах [id, c_x, c_y].
        """
        kpts = detection.keypoints.data.numpy()[0]
        points = np.array([kpts[5], kpts[6], kpts[11], kpts[12]])
        person_id = detection.boxes.id.numpy().astype(int)[0]
        if all(points[:, -1] >= self.config_.get('KEY_POINTS_CONFIDENCE')):  # если видно все тело
            return np.concatenate([[person_id], (points[:, :-1] / self.frame_shape).sum(axis=0) / 4])
        elif all(points[:, -1][:-2] >= self.config_.get('KEY_POINTS_CONFIDENCE')):  # видно только плечи
            return np.concatenate([[person_id], (points[:, :-1][:-2] / self.frame_shape).sum(axis=0) / 2])
        else:  # не видно тела
            xyxy = np.array([(bbox := detection.boxes.xyxy.numpy()[0])[:-2], bbox[-2:]]) / self.frame_shape
            return np.concatenate([[person_id], xyxy.sum(axis=0) / 2])

    def get_kmeans_grouped(self, detections: Results, people_centroids: np.array) -> Tuple[set, np.array]:
        """
        Разделение людей по группам (кластеризация) с помощью KMeans.
        :param detections: YOLO detections.
        :param people_centroids: Id + координаты центроида человека [id, c_x, c_y].
        :return: Set из групп людей + данные по людям в формате [[group, human_id, x1, y1, x2, y2, c_x, c_y], [...]]
        """
        prediction = self.kmeans_.predict(people_centroids[:, 1:])
        grouped_data = []
        for i in range(len(prediction)):
            x1, y1, x2, y2 = np.concatenate(np.array(
                [[(bbox := detections[i].boxes.xyxy.numpy()[0])[0], bbox[1]], [bbox[2], bbox[3]]]) / self.frame_shape)
            grouped_data.append(
                [prediction[i],  # group
                 people_centroids[i][0],  # id
                 x1, y1, x2, y2,  # bbox
                 people_centroids[i][1], people_centroids[i][2]]  # centroid
            )
        return set(prediction), np.array(grouped_data)

    def check_groups_inside(self, groups: set, people_data: np.array) -> List[np.array]:
        """
        Проверка расстояний и IOU внутри каждой из групп с заменой группы неподходящих людей на -1.
        :param groups: Set из групп, полученных из KMeans.
        :param people_data: Данные по людям в данной группе.
        :return: Список из проверенных данных по всем группам.
        """

        def check_group(data: np.array) -> List[np.array]:
            """
            Проверка расстояний и IOU в конкретной группе.
            :param data: Данные по конкретной группе.
            :return: Список из проверенных данных по группе.
            """
            grouped = dict()
            for data1, data2 in combinations(data, 2):  # каждый с каждым в группе
                if (np.linalg.norm(data1[-2:] - data2[-2:]) < self.config_.get('MIN_DISTANCE')  # проверяем расстояние
                        or iou(data1[2:-2], data2[2:-2]) > self.config_.get('IOU_THRESHOLD')):  # и IOU
                    grouped[data1[1]] = data1
                    grouped[data2[1]] = data2
            # мержим списки и меняем группу на -1 у тех, что находятся далеко друг от друга
            return list(chain.from_iterable(
                [list(grouped.values()),
                 [np.concatenate([[-1], d[1:]]) for d in data if d[1] not in grouped]]
            ))

        checked = []
        for group in groups:
            if len(data_by_group := people_data[people_data[:, 0] == group]) == 1:
                checked.append(data_by_group)  # если он один,
                continue  # то расстояния сравнивать не имеет смысла
            checked.append(check_group(data_by_group))
        return np.array(list(chain.from_iterable(checked)))  # мержим списки

    def check_groups(self, groups: set, people_data: np.array) -> np.array:
        """
        Проверка IOU между группами людей и их объединение, путем изменения группы в общем списке.
        :param groups: Set из групп, полученных из KMeans.
        :param people_data: Приближение по группировкам людей.
        :return: Данные по людям в формате [[group, human_id, x1, y1, x2, y2, c_x, c_y], [...]].
        """
        # находим ббоксы и центроиды по группам
        groups_bboxes = np.array([
            np.concatenate([[group], *self.get_bbox_centroid(people_data[people_data[:, 0] == group])])
            for group in groups if group in people_data[:, 0]
        ])
        # смотрим IOU между группами и меняем у людей группы, если они удовлетворяют условию
        for group1, group2 in combinations(groups_bboxes, 2):
            # смотрим расстояния и IOU между группами
            if iou(group1[1:-2], group2[1:-2]) > 2 * self.config_.get('IOU_THRESHOLD'):
                # и делаем одну группу для обоих, изменяя группы у людей
                people_data[:, 0][people_data[:, 0] == group2[0]] = group1[0]
        return people_data

    def check_ungrouped_people(self, people_data: np.array) -> np.array:
        """
        Проверка людей, не вошедших ни в одну из групп: 
            либо добавляем их к существующим, либо образуем новые, если ни с какой группой не сопоставились.
        :param people_data: Приближение по группировкам людей [[group, human_id, x1, y1, x2, y2, c_x, c_y], [...]].
        :return: Данные по людям в формате [[group, human_id, x1, y1, x2, y2, c_x, c_y], [...]]
        """
        # добавляем в группы людей, которые не вошли ни в одну группу
        for group in np.unique(people_data[:, 0][people_data[:, 0] != -1]):  # бежим по группам
            group_bbox, group_centroid = self.get_bbox_centroid(people_data[people_data[:, 0] == group])
            for person in people_data[people_data[:, 0] == -1]:  # проверяем, подходит ли человек данной группе
                if (np.linalg.norm(group_centroid - person[-2:]) < self.config_.get('MIN_DISTANCE')
                        or iou(group_bbox, person[2:-2]) > self.config_.get('IOU_THRESHOLD')):
                    people_data[:, 0][people_data[:, 1] == person[1]] = group
        # смотрим, образовывают ли люди, не вошедшие ни в одну группу, свои собственные
        max_group = np.max(people_data[:, 0])  # для добавления нового id группы
        changed = set()  # чтобы отслеживать, сматчился ли уже id
        for person1, person2 in combinations(people_data[people_data[:, 0] == -1], 2):
            if (np.linalg.norm(person1[-2:] - person2[-2:]) < self.config_.get('MIN_DISTANCE')
                    or iou(person1[2:6], person2[2:6]) > self.config_.get('IOU_THRESHOLD')):
                if person1[1] not in changed and person2[1] not in changed:  # если оба еще ни с кем не сматчились
                    max_group += 1  # образовываем новую группу
                    people_data[:, 0][people_data[:, 1] == person1[1]] = max_group  # добавляем в нее людей
                    people_data[:, 0][people_data[:, 1] == person2[1]] = max_group
                    changed.add(person1[1])  # помечаем их, как просмотренных
                    changed.add(person2[1])
                elif person1[1] not in changed:  # если сопоставился только person2
                    people_data[:, 0][people_data[:, 1] == person1[1]] \
                        = people_data[:, 0][people_data[:, 1] == person2[1]][0]  # меняем у 1 группу на группу 2
                    changed.add(person1[1])  # помечаем его, как уже просмотренный
                elif person2[1] not in changed:  # наоборот, если сопоставился только person1
                    people_data[:, 0][people_data[:, 1] == person2[1]] \
                        = people_data[:, 0][people_data[:, 1] == person1[1]][0]  # меняем у 2 группу на группу 1
                    changed.add(person2[1])
        return people_data

    def clusterize_people(self, groups: set, people_data: np.array) -> np.array:
        """
        Нахождение скоплений людей, отталкиваясь от приближения, сделанного с помощью KMeans.
        :param groups: Группы, на которые были поделены люди после кластеризации.
        :param people_data: Приближение в формате [[group, human_id, x1, y1, x2, y2, c_x, c_y], [...]].
        :return: Люди, находящиеся в группах в формате [[group, human_id, x1, y1, x2, y2, c_x, c_y], [...]].
        """
        # делаем проверку расстояний и IOU внутри каждой группы
        checked_inner = self.check_groups_inside(groups, people_data)
        # делаем проверку между группами
        checked_between = self.check_groups(groups, checked_inner)
        # делаем проверку для людей, которые не вошли ни в одну группу
        checked_ungrouped = self.check_ungrouped_people(checked_between)
        # фильтруем людей, которые не вошли ни в какую группу 
        filtered_data = np.array(
            [data for data in checked_ungrouped
             if data[0] != -1 and  # фильтруем людей, которые не вошли ни в какую группу
             # и тех, что образуют группу с количеством людей, меньше порогового
             len(people_data[checked_ungrouped[:, 0] == data[0]]) >= self.config_.get('MIN_CROWD_NUM_OF_PEOPLE')])
        return filtered_data

    def check_trigger(self, curr_ids: np.array) -> bool:
        """
        Нахождение новых сработок, путем нахождения новых id в списке групп людей.
        :param curr_ids: Текущие id людей в группах.
        :return: Найдены новые группы или нет, или же новые люди вошли в существующие группы.
        """
        triggered = curr_ids[~np.in1d(curr_ids, self.prev_ids)].size != 0
        self.prev_ids = curr_ids
        return triggered

    async def detect_(self, detections: Results) -> np.array:
        """
        Обработка YOLO-pose-треков для обнаружения скоплений людей в кадре.
        :param detections: YOLO detections.
        :return: Ббоксы скоплений людей в формате [[x1, y1, x2, y2], [...]] (пустой, если ничего не обнаружено).
        """
        if len(detections) >= self.config_.get('MIN_CROWD_NUM_OF_PEOPLE'):
            people_centroids = np.array([self.get_kpts_centroid(detection) for detection in detections
                                         if detection.boxes.id is not None])
            if people_centroids.size != 0:
                kmeans_groups, kmeans_grouped_people = self.get_kmeans_grouped(detections, people_centroids)
                grouped_people = self.clusterize_people(kmeans_groups, kmeans_grouped_people)
                if grouped_people.size != 0:
                    grouped_bboxes = np.array(
                        [self.get_bbox_centroid(grouped_people[grouped_people[:, 0] == group], True)[0]
                         for group in np.unique(grouped_people[:, 0])])
                    # return grouped_bboxes, self.check_trigger(grouped_people[:, 1])
                    return grouped_bboxes
        self.prev_ids = np.array([])
        # return np.array([]), False
        return np.array([])
