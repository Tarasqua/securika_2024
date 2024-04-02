import os
import fnmatch

import torch
import numpy as np
from torchvision import ops


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
