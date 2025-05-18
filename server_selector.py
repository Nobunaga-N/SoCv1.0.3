"""
Оптимизированный модуль для выбора серверов с улучшенным OCR.
Урезанная версия с основной функциональностью.
"""
import cv2
import numpy as np
import re
import logging
import time
from typing import Optional, List, Tuple, Dict

from config import SEASONS, COORDINATES, PAUSE_SETTINGS, OCR_REGIONS


class OptimizedServerSelector:
    """
    Оптимизированный класс для выбора серверов с точным определением координат.
    """

    def __init__(self, adb_controller, ocr_available=True):
        """
        Инициализация селектора серверов.

        Args:
            adb_controller: контроллер ADB
            ocr_available: доступность OCR
        """
        self.logger = logging.getLogger('sea_conquest_bot.server_selector')
        self.adb = adb_controller
        self.ocr_available = ocr_available

    def select_season(self, season_id: str) -> bool:
        """
        Выбор сезона с оптимизированной логикой.

        Args:
            season_id: идентификатор сезона (S1, S2, S3, S4, S5, X1, X2, X3, X4)

        Returns:
            bool: True если сезон выбран успешно
        """
        self.logger.info(f"Выбор сезона: {season_id}")

        if season_id not in SEASONS:
            self.logger.error(f"Неизвестный сезон: {season_id}")
            return False

        # Проверяем, нужен ли скроллинг для нижних сезонов
        if season_id in ['X2', 'X3', 'X4']:
            self._scroll_to_lower_seasons()

        # Клик по сезону
        return self._click_season(season_id)

    def _scroll_to_lower_seasons(self):
        """Скроллинг для показа нижних сезонов."""
        self.logger.info("Скроллинг для отображения нижних сезонов")

        start_x, start_y = COORDINATES['season_scroll_start']
        end_x, end_y = COORDINATES['season_scroll_end']

        self.adb.swipe(start_x, start_y, end_x, end_y, duration=1000)
        time.sleep(PAUSE_SETTINGS['after_season_scroll'])

    def _click_season(self, season_id: str) -> bool:
        """Клик по сезону."""
        season_coords = COORDINATES['seasons']

        if season_id not in season_coords:
            self.logger.error(f"Координаты для сезона {season_id} не найдены")
            return False

        x, y = season_coords[season_id]

        time.sleep(PAUSE_SETTINGS['before_season_click'])
        self.adb.tap(x, y)
        time.sleep(PAUSE_SETTINGS['after_season_click'])

        return True

    def get_servers_with_coordinates(self, debug_save=False) -> Dict[int, Tuple[int, int]]:
        """
        Получение видимых серверов с точными координатами через OCR.

        Args:
            debug_save: сохранять ли отладочные изображения

        Returns:
            dict: словарь {server_id: (click_x, click_y)}
        """
        if not self.ocr_available:
            self.logger.warning("OCR не доступен")
            return {}

        try:
            import pytesseract

            screenshot = self.adb.screenshot()
            x, y, w, h = OCR_REGIONS['servers']
            roi = screenshot[y:y + h, x:x + w]

            # Отладочное сохранение
            if debug_save:
                self._save_debug_image(roi, "server_roi")

            # Обработка изображения
            servers_with_coords = {}
            processed_images = self._preprocess_image(roi, w, h)

            for method_name, img, scale in processed_images:
                if debug_save:
                    self._save_debug_image(img, f"processed_{method_name}")

                # OCR анализ
                data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT,
                    lang='rus+eng', config='--psm 6'
                )

                # Поиск серверов
                self._extract_servers_from_ocr_data(
                    data, servers_with_coords, x, y, scale
                )

            sorted_servers = dict(sorted(servers_with_coords.items(), reverse=True))

            if sorted_servers:
                self.logger.info(f"Найдены сервера: {list(sorted_servers.keys())}")

            return sorted_servers

        except Exception as e:
            self.logger.error(f"Ошибка получения координат серверов: {e}")
            return {}

    def _preprocess_image(self, roi, w, h) -> List[Tuple[str, np.ndarray, int]]:
        """Предобработка изображения для OCR."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        processed = []

        # Стандартная бинаризация
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        processed.append(("binary", binary, 1))

        # Адаптивная бинаризация
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        processed.append(("adaptive", binary_adaptive, 1))

        # Увеличенное изображение
        scale_factor = 2
        resized = cv2.resize(gray, (w * scale_factor, h * scale_factor),
                           interpolation=cv2.INTER_CUBIC)
        _, binary_resized = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY_INV)
        processed.append(("resized", binary_resized, scale_factor))

        return processed

    def _extract_servers_from_ocr_data(self, data, servers_dict, offset_x, offset_y, scale):
        """Извлечение серверов из данных OCR."""
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = int(data['conf'][i])

            if confidence < 30:
                continue

            server_numbers = self._parse_server_numbers(text)

            for server_id in server_numbers:
                if 100 <= server_id <= 619 and server_id not in servers_dict:
                    # Вычисляем координаты центра текста
                    text_x = data['left'][i] // scale
                    text_y = data['top'][i] // scale
                    text_w = data['width'][i] // scale
                    text_h = data['height'][i] // scale

                    abs_x = offset_x + text_x + text_w // 2
                    abs_y = offset_y + text_y + text_h // 2

                    servers_dict[server_id] = (abs_x, abs_y)

    def _parse_server_numbers(self, text: str) -> List[int]:
        """Парсинг номеров серверов из текста."""
        patterns = [
            r"Море\s*#(\d{1,3})",
            r"#(\d{1,3})",
            r"\b(\d{3})\b",
            r"\b(\d{1,3})\b"
        ]

        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    num = int(match)
                    if 1 <= num <= 619:
                        numbers.append(num)
                except ValueError:
                    continue

        return numbers

    def _save_debug_image(self, image, prefix):
        """Сохранение отладочного изображения."""
        import os
        debug_dir = "debug_ocr"
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = int(time.time())
        cv2.imwrite(f"{debug_dir}/{prefix}_{timestamp}.png", image)

    def find_server_coordinates(self, server_id: int) -> Optional[Tuple[int, int]]:
        """
        Поиск координат конкретного сервера.

        Args:
            server_id: номер сервера

        Returns:
            tuple: (x, y) координаты сервера или None
        """
        servers_dict = self.get_servers_with_coordinates(debug_save=True)

        if server_id in servers_dict:
            return servers_dict[server_id]

        # Поиск ближайшего сервера
        if servers_dict:
            closest_server = min(servers_dict.keys(),
                               key=lambda s: abs(s - server_id))
            if abs(closest_server - server_id) <= 5:
                self.logger.info(f"Используем близкий сервер {closest_server} вместо {server_id}")
                return servers_dict[closest_server]

        return None