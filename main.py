"""
Главный модуль бота для прохождения обучения в игре Sea of Conquest.
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path

from logger import setup_logger
from adb_controller import ADBController
from image_handler import ImageHandler
from game_bot import GameBot

def parse_arguments():
    """
    Парсинг аргументов командной строки.
    """
    parser = argparse.ArgumentParser(description='Бот для прохождения обучения в игре Sea of Conquest')

    parser.add_argument(
        '-c', '--cycles',
        type=int,
        default=1,
        help='Количество циклов обучения (по умолчанию: 1)'
    )

    parser.add_argument(
        '-d', '--device',
        type=str,
        default=None,
        help='Имя устройства ADB (по умолчанию: первое доступное)'
    )

    parser.add_argument(
        '-H', '--host',
        type=str,
        default='127.0.0.1',
        help='Хост ADB сервера (по умолчанию: 127.0.0.1)'
    )

    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5037,
        help='Порт ADB сервера (по умолчанию: 5037)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Включение подробного логирования'
    )

    # Аргументы --start-server и --end-server удалены, так как теперь запрашиваются интерактивно

    return parser.parse_args()

def check_environment():
    """
    Проверка окружения перед запуском бота.

    Returns:
        bool: True если все проверки пройдены, False иначе
    """
    logger = logging.getLogger('sea_conquest_bot.main')

    # Проверка наличия ADB
    logger.info("Проверка наличия ADB...")
    try:
        import subprocess
        result = subprocess.run(['adb', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ADB не найден или не работает корректно")
            return False
        logger.info(f"Обнаружен ADB: {result.stdout.splitlines()[0]}")
    except Exception as e:
        logger.error(f"Ошибка при проверке ADB: {e}")
        return False

    # Проверка наличия директории с изображениями
    logger.info("Проверка наличия директории с изображениями...")
    from config import IMAGES_DIR, IMAGE_PATHS

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR, exist_ok=True)
        logger.warning(f"Директория с изображениями создана: {IMAGES_DIR}")

    # Проверка наличия необходимых изображений
    missing_images = []
    for image_key, image_path in IMAGE_PATHS.items():
        if not os.path.exists(image_path):
            missing_images.append(f"{image_key}: {image_path}")

    if missing_images:
        logger.warning("Отсутствуют следующие изображения (они будут нужны во время выполнения):")
        for missing in missing_images:
            logger.warning(f"  - {missing}")
        logger.warning("Пожалуйста, добавьте недостающие изображения в папку images перед запуском бота")
        # Возвращаем True, так как согласно новому ТЗ мы в основном используем распознавание текста
        # Можно продолжить работу, но предупредить пользователя

    else:
        logger.info("Все необходимые изображения найдены")

    # Проверка наличия Tesseract OCR
    try:
        import pytesseract
        logger.info("OCR (Tesseract) доступен для использования")
    except ImportError:
        logger.warning("OCR (Tesseract) не доступен. Рекомендуется установить для лучшего распознавания текста")
        # Продолжаем работу, но предупреждаем пользователя

    # Проверка наличия необходимых библиотек
    try:
        import cv2
        import numpy
        import subprocess
    except ImportError as e:
        logger.error(f"Отсутствует необходимая библиотека: {e}")
        logger.error("Установите необходимые библиотеки: pip install -r requirements.txt")
        return False

    logger.info("Все проверки пройдены успешно")
    return True


def main():
    """Главная функция запуска бота."""
    # Парсинг аргументов командной строки (оставляем для других параметров)
    args = parse_arguments()

    # Настройка логирования
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(log_level=log_level)
    logger.info("Запуск бота для прохождения обучения в игре Sea of Conquest")

    # Проверка окружения
    if not check_environment():
        logger.error("Проверка окружения не пройдена. Выход.")
        sys.exit(1)

    try:
        # Запрос диапазона серверов у пользователя через консоль
        while True:
            try:
                print("\n=== Настройка диапазона серверов ===")
                start_server = int(input("Введите начальный сервер (по умолчанию 619): ") or "619")
                end_server = int(input("Введите конечный сервер (по умолчанию 1): ") or "1")

                if start_server < end_server:
                    print("Ошибка: Начальный сервер должен быть больше или равен конечному.")
                    continue

                break
            except ValueError:
                print("Ошибка: Введите корректные числовые значения.")

        # Запрос начального шага для первого сервера
        start_step = 1
        try:
            print("\n=== Настройка начального шага (только для первого сервера) ===")
            start_step = int(input("Введите начальный шаг для первого сервера (по умолчанию 1): ") or "1")

            if start_step < 1 or start_step > 97:  # 97 - последний шаг обучения
                print(f"Предупреждение: Введен шаг {start_step} вне диапазона 1-97. Будет использован шаг 1.")
                start_step = 1
        except ValueError:
            print("Ошибка при вводе начального шага. Будет использован шаг 1.")
            start_step = 1

        # Инициализация компонентов
        logger.info("Инициализация компонентов...")

        # Создание контроллера ADB
        adb_controller = ADBController(
            host=args.host,
            port=args.port,
            device_name=args.device
        )

        # Создание обработчика изображений
        image_handler = ImageHandler(adb_controller)

        # Создание бота
        game_bot = GameBot(adb_controller, image_handler)

        # Запуск бота на выполнение заданного количества циклов
        logger.info(f"Запуск бота на {args.cycles} циклов с серверами от {start_server} до {end_server}")
        logger.info(f"Начальный шаг для первого сервера: {start_step}")
        game_bot.run_bot(cycles=args.cycles,
                         start_server=start_server,
                         end_server=end_server,
                         first_server_start_step=start_step)

    except KeyboardInterrupt:
        logger.info("Работа бота прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка при работе бота: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Работа бота завершена")

if __name__ == "__main__":
    main()