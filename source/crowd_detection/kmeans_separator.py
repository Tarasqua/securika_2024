import os
import fnmatch
from pathlib import Path
import joblib

from loguru import logger
import torch
from sklearn.cluster import KMeans


class TrainDataNotTransmittedException(Exception):
    """Ошибка, возникающая при отсутствии данных для обучения KMeans."""
    def __init__(self):
        logger.exception('Training data has not been transmitted')

    def __str__(self):
        return ""


class ModelNotFoundException(Exception):
    """Ошибка, возникающая при отсутствии kmeans модели в директории с моделями."""
    def __init__(self):
        logger.exception("Couldn't find kmeans model file")

    def __str__(self):
        return ""


def kmeans_fit(train_data_path: str | None, scene_size: str) -> KMeans:
    """
    Предобучение KMeans модели.
    :param train_data_path: Путь до файла для обучения.
    :param scene_size: Размер сцены детекции.
    :return: Объект sklearn KMeans.
    """

    def find_train_data() -> Path:
        """Поиск данных для предобучения модели."""
        '/home/tarasqua/work/exhibition/resources/train_data/kmeans'
        for file in os.listdir(train_path := Path.cwd().parents[0] / 'resources' / 'train_data' / 'kmeans'):
            if fnmatch.fnmatch(file, '*.pt'):
                return train_path / file
        raise ModelNotFoundException()

    def select_n_clusters() -> int:
        """Подбор количества кластеров на основе размера сцены."""
        match scene_size:
            case 'small':
                return 2
            case 'medium':
                return 5
            case 'large':
                return 10
            case _:
                raise

    train_data = torch.load(train_data_path) if train_data_path is not None else torch.load(find_train_data())
    kmeans_model = KMeans(n_clusters=(clusters := select_n_clusters()),
                          init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans_model.fit(train_data)
    logger.success(f'KMeans model trained successfully on training data "{train_data_path}" with {clusters} clusters')
    return kmeans_model


def kmeans_save(kmeans_model, model_name: str) -> None:
    """
    Сохранение предобученной модели.
    :param kmeans_model: Предобученная модель sklearn KMeans.
    :param model_name: Наименование модели для сохранения.
    :return: None.
    """
    if not (new_dir := Path.cwd().parents[1] / 'resources' / 'models' / 'kmeans').exists():
        new_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_model, f'{new_dir / model_name}.joblib')


def kmeans_load(model_name: str) -> KMeans:
    """Подгрузка модели из директории с моделями"""
    if not (kmeans_model_path := (
            Path.cwd().parents[1] / 'resources' / 'models' / 'kmeans' / f'{model_name}.joblib')).exists():
        raise ModelNotFoundException
    model = joblib.load(kmeans_model_path)
    logger.success(f'KMeans model "{kmeans_model_path}" loaded successfully')
    return model
