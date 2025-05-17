"""
Модуль для обработки изображений и поиска шаблонов на экране.
"""
import cv2
import numpy as np
import time
import logging

from config import TEMPLATE_MATCHING_THRESHOLD, SCREENSHOT_TIMEOUT

class ImageHandler:
    """Класс для обработки изображений и поиска шаблонов на экране."""

    def __init__(self, adb_controller):
        """
        Инициализация обработчика изображений.

        Args:
            adb_controller: экземпляр класса ADBController
        """
        self.logger = logging.getLogger('sea_conquest_bot.image')
        self.adb = adb_controller
        self.templates = {}  # Кэш шаблонов изображений

    def load_template(self, template_path):
        """
        Загрузка шаблона изображения.

        Args:
            template_path: путь к файлу шаблона

        Returns:
            numpy.ndarray: изображение шаблона
        """
        # Проверка наличия шаблона в кэше
        if template_path in self.templates:
            return self.templates[template_path]

        # Загрузка шаблона
        template = cv2.imread(template_path)
        if template is None:
            self.logger.error(f"Не удалось загрузить шаблон: {template_path}")
            raise FileNotFoundError(f"Не удалось загрузить шаблон: {template_path}")

        # Сохранение шаблона в кэше
        self.templates[template_path] = template
        return template

    def find_template(self, screenshot, template_path, threshold=TEMPLATE_MATCHING_THRESHOLD):
        """
        Поиск шаблона на скриншоте.

        Args:
            screenshot: скриншот (numpy.ndarray)
            template_path: путь к файлу шаблона
            threshold: порог соответствия (0-1)

        Returns:
            tuple: (x, y, w, h) координаты и размеры найденного шаблона или None
        """
        # Загрузка шаблона
        template = self.load_template(template_path)

        # Поиск шаблона
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            # Получение размеров шаблона
            h, w = template.shape[:2]

            # Координаты центра найденного шаблона
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2

            return (center_x, center_y, w, h)

        return None

    def wait_for_template(self, template_path, timeout=30, threshold=TEMPLATE_MATCHING_THRESHOLD):
        """
        Ожидание появления шаблона на экране.

        Args:
            template_path: путь к файлу шаблона
            timeout: максимальное время ожидания в секундах
            threshold: порог соответствия (0-1)

        Returns:
            tuple: (x, y, w, h) координаты и размеры найденного шаблона или None
        """
        self.logger.debug(f"Ожидание появления шаблона: {template_path}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Получение скриншота
                screenshot = self.adb.screenshot()

                if screenshot is None or screenshot.size == 0:
                    self.logger.warning("Получен пустой скриншот")
                    time.sleep(SCREENSHOT_TIMEOUT)
                    continue

                # Поиск шаблона
                result = self.find_template(screenshot, template_path, threshold)

                if result:
                    self.logger.info(f"Шаблон {template_path} найден на координатах: {result[:2]}")
                    return result

            except Exception as e:
                self.logger.error(f"Ошибка при ожидании шаблона {template_path}: {e}", exc_info=True)
                time.sleep(SCREENSHOT_TIMEOUT * 2)  # Увеличенная пауза при ошибке

            # Пауза перед следующей попыткой
            time.sleep(SCREENSHOT_TIMEOUT)

        self.logger.warning(f"Шаблон {template_path} не найден за {timeout} сек")
        return None

    def tap_on_template(self, template_path, timeout=30, threshold=TEMPLATE_MATCHING_THRESHOLD):
        """
        Поиск шаблона на экране и клик по нему.

        Args:
            template_path: путь к файлу шаблона
            timeout: максимальное время ожидания в секундах
            threshold: порог соответствия (0-1)

        Returns:
            bool: True если шаблон найден и клик выполнен, False иначе
        """
        try:
            result = self.wait_for_template(template_path, timeout, threshold)

            if result:
                x, y, _, _ = result
                self.adb.tap(x, y)
                return True

        except Exception as e:
            self.logger.error(f"Ошибка при поиске и клике по шаблону {template_path}: {e}", exc_info=True)

        return False

    def is_template_on_screen(self, template_path, timeout=1, threshold=TEMPLATE_MATCHING_THRESHOLD):
        """
        Проверка наличия шаблона на экране.

        Args:
            template_path: путь к файлу шаблона
            timeout: максимальное время ожидания в секундах
            threshold: порог соответствия (0-1)

        Returns:
            bool: True если шаблон найден, False иначе
        """
        return self.wait_for_template(template_path, timeout, threshold) is not None