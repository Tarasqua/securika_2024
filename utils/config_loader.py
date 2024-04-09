"""
@tarasqua

Подгрузка конфига из resources/config/config.yml
"""

import yaml
import json
from functools import reduce
from pathlib import Path
from typing import Any

from loguru import logger

from utils.util import find_file


class ConfigNotFoundException(Exception):
    """Обработка не нахождения конфиг-файла ни локально, ни на Redis."""

    def __init__(self, entity: str):
        logger.critical(f'The config-file for entity "{entity}" could not be found either locally or on Redis')

    def __str__(self):
        return ""


class Config:
    """Подгрузка конфига"""

    def __init__(self):
        self.config_data = dict()

    def initialize(self, entity: str) -> None:
        """
        Инициализация конфига.
        :param entity: Сущность, для которой нужно подтянуть конфиг.
        :return: None.
        """
        # пытаемся найти его локально
        if (config_path := find_file(f'{entity}*.yml', Path.cwd().parents[1].as_posix())) == '':
            raise ConfigNotFoundException(entity)
        logger.info(f'Config-file was found locally: {config_path}')
        # подгружаем
        with open(config_path, 'r') as f:
            self.config_data: dict = yaml.load(f, Loader=yaml.FullLoader)
        logger.success(f'Config for entity "{entity}" successfully set up')

    def get(self, *setting_name) -> Any:
        """
        Геттер конфига.
        :param setting_name: Путь до определенного параметра в конфиг-файле вида *args.
        :return: Запрпашиваемый параметр из конфига.
        """
        return reduce(dict.get, setting_name, self.config_data)
