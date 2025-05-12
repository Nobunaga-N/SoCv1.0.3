"""
Модуль для настройки системы логирования.
"""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(log_dir='logs', log_level=logging.INFO):
    """
    Настройка системы логирования.

    Args:
        log_dir: директория для хранения логов
        log_level: уровень логирования

    Returns:
        logging.Logger: настроенный логгер
    """
    # Создание директории для логов, если она не существует
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Формирование имени файла лога с датой и временем
    log_filename = log_path / f'sea_conquest_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    # Настройка форматирования
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Настройка файлового обработчика
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Настройка консольного обработчика
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Настройка корневого логгера
    logger = logging.getLogger('sea_conquest_bot')
    logger.setLevel(log_level)

    # Очистка существующих обработчиков
    logger.handlers = []

    # Добавление обработчиков к логгеру
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Логирование настроено. Уровень: {logging.getLevelName(log_level)}")
    logger.info(f"Лог сохраняется в: {log_filename}")

    return logger