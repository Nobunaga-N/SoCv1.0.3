"""
Модуль для взаимодействия с устройством через ADB.
"""
import time
import numpy as np
import cv2
import random
import logging
import subprocess
import os
import tempfile
from io import BytesIO

from config import DEFAULT_TIMEOUT, LOADING_TIMEOUT, GAME_PACKAGE, GAME_ACTIVITY

class ADBController:
    """Класс для взаимодействия с устройством через ADB."""

    def __init__(self, host='127.0.0.1', port=5037, device_name=None):
        """
        Инициализация контроллера ADB.

        Args:
            host: хост ADB сервера
            port: порт ADB сервера
            device_name: имя устройства (если None, будет выбрано первое доступное)
        """
        self.logger = logging.getLogger('sea_conquest_bot.adb')
        self.host = host
        self.port = port
        self.device_name = device_name

        # Попытка подключения к устройству
        self.logger.info("Поиск подключенных устройств...")

        # Подключение через subprocess - более надежный метод
        try:
            # Проверка наличия устройств
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            devices_output = result.stdout.strip().split('\n')[1:]

            # Фильтрация пустых строк
            devices = [d.split('\t')[0] for d in devices_output if d.strip() and '\t' in d]

            if not devices:
                self.logger.error("Не найдено подключенных устройств")
                raise ConnectionError("Не найдено подключенных устройств")

            # Выбор устройства
            if device_name:
                if device_name not in devices:
                    self.logger.error(f"Устройство {device_name} не найдено")
                    raise ValueError(f"Устройство {device_name} не найдено")
                self.device_serial = device_name
            else:
                self.device_serial = devices[0]

            self.logger.info(f"Подключено к устройству: {self.device_serial}")

            # Проверка соединения
            self.execute_adb_command('shell', 'echo', 'Connected')

        except Exception as e:
            self.logger.error(f"Ошибка при подключении к устройству: {e}")
            raise

    def execute_adb_command(self, *args, binary_output=False):
        """
        Выполнение команды ADB через subprocess.

        Args:
            *args: аргументы команды ADB
            binary_output: если True, вернуть бинарные данные (для команд, возвращающих бинарные данные)

        Returns:
            str или bytes: вывод команды
        """
        cmd = ['adb']

        # Добавление опции -s для указания устройства, если оно задано
        if self.device_serial:
            cmd.extend(['-s', self.device_serial])

        # Добавление аргументов команды
        cmd.extend(args)

        # Выполнение команды
        try:
            if binary_output:
                result = subprocess.run(cmd, capture_output=True, check=True)
                return result.stdout
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка при выполнении команды ADB: {e}")
            self.logger.error(f"Stderr: {e.stderr}")
            raise

    def tap(self, x, y):
        """
        Выполнение клика по координатам.

        Args:
            x: координата x
            y: координата y
        """
        self.logger.debug(f"Клик по координатам: ({x}, {y})")
        self.execute_adb_command('shell', 'input', 'tap', str(x), str(y))
        # Удаляем задержку time.sleep(DEFAULT_TIMEOUT)

    def tap_random(self, center_x, center_y, radius=50):
        """
        Выполнение клика по случайным координатам в заданной области.

        Args:
            center_x: центр области по x
            center_y: центр области по y
            radius: радиус области
        """
        random_x = center_x + random.randint(-radius, radius)
        random_y = center_y + random.randint(-radius, radius)
        self.logger.debug(
            f"Случайный клик в области ({center_x}±{radius}, {center_y}±{radius}): ({random_x}, {random_y})")
        self.tap(random_x, random_y)
        # Удаляем задержку time.sleep(DEFAULT_TIMEOUT)

    def swipe(self, start_x, start_y, end_x, end_y, duration=1000):
        """
        Выполнение свайпа от одной точки к другой.

        Args:
            start_x: начальная координата x
            start_y: начальная координата y
            end_x: конечная координата x
            end_y: конечная координата y
            duration: продолжительность свайпа в миллисекундах
        """
        self.logger.debug(f"Свайп от ({start_x}, {start_y}) к ({end_x}, {end_y})")
        self.execute_adb_command('shell', 'input', 'swipe',
                                 str(start_x), str(start_y),
                                 str(end_x), str(end_y),
                                 str(duration))
        # Удаляем задержку time.sleep(DEFAULT_TIMEOUT)

    def complex_swipe(self, points, total_duration=2000):
        """
        Выполнение сложного свайпа через несколько точек.

        Args:
            points: список точек [(x1, y1), (x2, y2), ...]
            total_duration: общая продолжительность свайпа в миллисекундах
        """
        if len(points) < 2:
            self.logger.error("Для свайпа необходимо минимум 2 точки")
            return

        self.logger.debug(f"Сложный свайп через точки: {points}")

        # Разделение общей продолжительности свайпа на сегменты
        segment_duration = total_duration // (len(points) - 1)

        for i in range(len(points) - 1):
            start_x, start_y = points[i]
            end_x, end_y = points[i + 1]
            self.swipe(start_x, start_y, end_x, end_y, segment_duration)

    def key_event(self, key_code):
        """
        Отправка события клавиши.

        Args:
            key_code: код клавиши
        """
        self.logger.debug(f"Отправка события клавиши: {key_code}")
        self.execute_adb_command('shell', 'input', 'keyevent', str(key_code))
        # Удаляем задержку time.sleep(DEFAULT_TIMEOUT)

    def press_esc(self):
        """Нажатие клавиши ESC (BACK)."""
        self.logger.debug("Нажатие ESC")
        self.key_event(4)  # KEYCODE_BACK

    def screenshot(self):
        """
        Получение скриншота экрана.

        Returns:
            numpy.ndarray: изображение в формате OpenCV (BGR)
        """
        try:
            # Использование shell команды screencap для получения скриншота в бинарном формате
            self.logger.debug("Получение скриншота экрана")

            # Метод 1: Через exec-out (более быстрый метод, но может не работать на некоторых устройствах)
            try:
                # Используем binary_output=True, т.к. screencap возвращает бинарные данные
                screenshot_binary = self.execute_adb_command('exec-out', 'screencap', '-p', binary_output=True)

                # Преобразование бинарных данных PNG в изображение формата OpenCV
                image = cv2.imdecode(np.frombuffer(screenshot_binary, np.uint8), cv2.IMREAD_COLOR)

                if image is not None:
                    return image
            except Exception as e:
                self.logger.warning(f"Ошибка при получении скриншота через exec-out: {e}")
                # Продолжаем выполнение, попробуем альтернативный метод

            # Метод 2: Через временный файл (более надежный метод)
            # Создаем временный файл для скриншота
            temp_file = 'temp_screenshot.png'

            # Сохраняем скриншот во временный файл на устройстве
            self.execute_adb_command('shell', 'screencap', '-p', '/sdcard/' + temp_file)

            # Получаем файл с устройства
            self.execute_adb_command('pull', '/sdcard/' + temp_file, temp_file)

            # Читаем изображение с помощью OpenCV
            image = cv2.imread(temp_file)

            # Удаляем временные файлы
            import os
            try:
                os.remove(temp_file)
                self.execute_adb_command('shell', 'rm', '/sdcard/' + temp_file)
            except Exception as e:
                self.logger.warning(f"Ошибка при удалении временных файлов: {e}")

            if image is None:
                raise ValueError("Не удалось получить скриншот")

            return image

        except Exception as e:
            self.logger.error(f"Ошибка при получении скриншота: {e}", exc_info=True)

            # В случае критической ошибки возвращаем пустое изображение
            # чтобы не вызывать падение программы
            return np.zeros((720, 1280, 3), dtype=np.uint8)

    def start_app(self, package_name=GAME_PACKAGE, activity_name=GAME_ACTIVITY):
        """
        Запуск приложения.

        Args:
            package_name: имя пакета приложения
            activity_name: имя активности
        """
        self.logger.info(f"Запуск приложения: {package_name}")

        if activity_name:
            cmd = f"am start -n {package_name}/{activity_name}"
            self.execute_adb_command('shell', cmd)
        else:
            self.execute_adb_command('shell', 'monkey', '-p', package_name,
                                    '-c', 'android.intent.category.LAUNCHER', '1')

        time.sleep(LOADING_TIMEOUT)

    def stop_app(self, package_name=GAME_PACKAGE):
        """
        Остановка приложения.

        Args:
            package_name: имя пакета приложения
        """
        self.logger.info(f"Остановка приложения: {package_name}")
        self.execute_adb_command('shell', 'am', 'force-stop', package_name)
        time.sleep(DEFAULT_TIMEOUT)