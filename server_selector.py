"""
Оптимизированный модуль для выбора серверов с улучшенным OCR и логикой скроллинга.
Исправления: точный скроллинг, лучшая фильтрация OCR, адаптивный поиск.
"""
import cv2
import numpy as np
import re
import logging
import time
from typing import Optional, List, Tuple, Dict

from config import SEASONS, COORDINATES, PAUSE_SETTINGS, OCR_REGIONS, SERVER_RECOGNITION_SETTINGS


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
        self.last_servers = []  # История последних найденных серверов для отслеживания движения
        self.current_season = None  # Текущий выбранный сезон
        self.cached_servers = {}  # Кеш для результатов OCR
        self.last_screenshot_time = 0  # Время последнего скриншота
        self.cache_timeout = 1.0  # Таймаут кеша в секундах

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

        # Сохраняем текущий сезон для валидации
        self.current_season = season_id
        self.cached_servers = {}  # Очищаем кеш при смене сезона

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

    def get_servers_with_coordinates(self, force_refresh=False) -> Dict[int, Tuple[int, int]]:
        """
        Получение видимых серверов с точными координатами через OCR.
        Добавлено кеширование для уменьшения количества вызовов OCR.

        Args:
            force_refresh: принудительно обновить кеш

        Returns:
            dict: словарь {server_id: (click_x, click_y)}
        """
        current_time = time.time()

        # Проверяем кеш
        if not force_refresh and current_time - self.last_screenshot_time < self.cache_timeout:
            if self.cached_servers:
                return self.cached_servers

        if not self.ocr_available:
            self.logger.warning("OCR не доступен")
            return {}

        try:
            import pytesseract

            screenshot = self.adb.screenshot()
            if screenshot is None or screenshot.size == 0:
                self.logger.warning("Получен пустой скриншот")
                return {}

            x, y, w, h = OCR_REGIONS['servers']
            roi = screenshot[y:y + h, x:x + w]

            # Обработка изображения
            servers_with_coords = {}
            processed_images = self._preprocess_image(roi, w, h)

            for method_name, img, scale in processed_images:

                # OCR анализ
                data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT,
                    lang='rus+eng', config='--psm 6'
                )

                # Поиск серверов
                self._extract_servers_from_ocr_data(
                    data, servers_with_coords, x, y, scale
                )

            # Фильтрация и валидация результатов с учетом текущего сезона
            validated_servers = self._validate_servers_with_season(servers_with_coords)
            sorted_servers = dict(sorted(validated_servers.items(), reverse=True))

            # Обновляем кеш и время
            self.cached_servers = sorted_servers
            self.last_screenshot_time = current_time

            if sorted_servers:
                # Логируем только если результат отличается от предыдущего
                if list(sorted_servers.keys()) != self.last_servers:
                    self.logger.info(f"Найдены валидные сервера: {list(sorted_servers.keys())}")
                    self.last_servers = list(sorted_servers.keys())
            else:
                self.logger.warning("Не найдено валидных серверов")

            return sorted_servers

        except Exception as e:
            self.logger.error(f"Ошибка получения координат серверов: {e}")
            return {}

    def _validate_servers_with_season(self, servers_dict: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        Валидация найденных серверов с учетом текущего сезона.

        Args:
            servers_dict: словарь серверов {server_id: (x, y)}

        Returns:
            dict: отфильтрованный словарь валидных серверов
        """
        validated = {}

        # Получаем диапазон серверов для текущего сезона
        if self.current_season and self.current_season in SEASONS:
            season_data = SEASONS[self.current_season]
            season_min = min(season_data['min_server'], season_data['max_server'])
            season_max = max(season_data['min_server'], season_data['max_server'])
        else:
            # Если сезон не установлен, используем полный диапазон
            season_min = 1
            season_max = 619

        for server_id, coords in servers_dict.items():
            # Базовая проверка диапазона
            if not (1 <= server_id <= 619):
                self.logger.debug(f"Сервер {server_id} вне общего диапазона")
                continue

            # Проверка принадлежности к текущему сезону
            if not (season_min <= server_id <= season_max):
                self.logger.debug(f"Сервер {server_id} не принадлежит сезону {self.current_season} ({season_min}-{season_max})")
                continue

            # Проверка координат
            x, y = coords
            roi_x, roi_y, roi_w, roi_h = OCR_REGIONS['servers']
            if not (roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h):
                self.logger.debug(f"Сервер {server_id} имеет координаты вне области поиска")
                continue

            # Проверка логичности последовательности (более мягкая для серверов в пределах сезона)
            if self.last_servers:
                min_last = min(self.last_servers)
                max_last = max(self.last_servers)

                # Для серверов в пределах сезона используем более мягкий критерий
                reasonable_range = 50
                if server_id < min_last - reasonable_range or server_id > max_last + reasonable_range:
                    self.logger.debug(f"Сервер {server_id} далеко от предыдущих результатов ({min_last}-{max_last})")
                    continue

            validated[server_id] = coords

        # Если найдено мало валидных серверов, логируем для отладки
        if len(validated) < 3:
            self.logger.debug(f"Мало валидных серверов ({len(validated)}), сезон: {self.current_season}")

        return validated

    def _preprocess_image(self, roi, w, h) -> List[Tuple[str, np.ndarray, int]]:
        """Предобработка изображения для OCR с улучшенной фильтрацией."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        processed = []

        # Применение размытия для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Стандартная бинаризация с предварительным размытием
        _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
        processed.append(("binary_blur", binary, 1))

        # Адаптивная бинаризация
        binary_adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        processed.append(("adaptive_blur", binary_adaptive, 1))

        # Увеличенное изображение для лучшего распознавания мелких символов
        scale_factor = 2
        resized = cv2.resize(blurred, (w * scale_factor, h * scale_factor),
                           interpolation=cv2.INTER_CUBIC)
        _, binary_resized = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY_INV)
        processed.append(("resized_blur", binary_resized, scale_factor))

        return processed

    def _extract_servers_from_ocr_data(self, data, servers_dict, offset_x, offset_y, scale):
        """Извлечение серверов из данных OCR с улучшенной фильтрацией."""
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = int(data['conf'][i])

            # Повышаем минимальную уверенность для лучшей фильтрации
            if confidence < 40:
                continue

            server_numbers = self._parse_server_numbers(text)

            for server_id in server_numbers:
                # Дополнительная проверка на разумность номера сервера
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
        """Парсинг номеров серверов из текста с улучшенной логикой."""
        # Очистка текста от лишних символов
        clean_text = re.sub(r'[^\w\s#№:]', ' ', text)

        patterns = [
            r"Море\s*[#№]\s*(\d{3})",  # "Море #504"
            r"[#№]\s*(\d{3})",          # "#504"
            r"\b(\d{3})\b",             # Трехзначное число отдельно
        ]

        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, clean_text)
            for match in matches:
                try:
                    num = int(match)
                    # Более строгая проверка диапазона
                    if 100 <= num <= 619:
                        numbers.append(num)
                except ValueError:
                    continue

        # Убираем дубликаты, сохраняя порядок
        unique_numbers = []
        for num in numbers:
            if num not in unique_numbers:
                unique_numbers.append(num)

        return unique_numbers

    # Убираем метод _save_debug_image так как он больше не нужен

    def find_server_coordinates(self, server_id: int, attempts: int = 1) -> Optional[Tuple[int, int]]:
        """
        Поиск координат конкретного сервера с уменьшенным количеством попыток.

        Args:
            server_id: номер сервера
            attempts: количество попыток поиска (уменьшено с 3 до 1)

        Returns:
            tuple: (x, y) координаты сервера или None
        """
        for attempt in range(attempts):
            # Принудительно обновляем данные только на первой попытке
            servers_dict = self.get_servers_with_coordinates(force_refresh=(attempt == 0))

            # Прямой поиск целевого сервера
            if server_id in servers_dict:
                self.logger.info(f"Сервер {server_id} найден точно на попытке {attempt + 1}")
                return servers_dict[server_id]

            # Небольшая пауза между попытками
            if attempt < attempts - 1:
                time.sleep(0.3)

        return None

    def scroll_to_server_range(self, target_server: int, current_servers: List[int]) -> str:
        """
        Интеллектуальный скроллинг к диапазону сервера.

        Args:
            target_server: целевой сервер
            current_servers: текущие видимые сервера

        Returns:
            str: 'found' если цель видна, 'small' если выполнен мелкий скроллинг,
                 'regular' если обычный, 'none' если скроллинг не нужен
        """
        if not current_servers:
            self.logger.warning("Нет текущих серверов для определения направления скроллинга")
            self._perform_regular_scroll(target_server, [])
            return 'regular'

        min_visible = min(current_servers)
        max_visible = max(current_servers)

        self.logger.debug(f"Целевой сервер: {target_server}, видимые: {min_visible}-{max_visible}")

        # Если целевой сервер уже видимый
        if min_visible <= target_server <= max_visible:
            return 'found'

        # Определяем расстояние до цели
        distance_to_min = abs(target_server - min_visible)
        distance_to_max = abs(target_server - max_visible)
        min_distance = min(distance_to_min, distance_to_max)

        # Если очень близко (1-3 сервера), используем мелкий скроллинг
        if min_distance <= 3:
            self._perform_small_scroll(target_server, current_servers)
            return 'small'
        # Если близко (4-8 серверов), пробуем мелкий скроллинг
        elif min_distance <= 8:
            self._perform_small_scroll(target_server, current_servers)
            return 'small'
        # Если далеко, используем обычный скроллинг
        else:
            self._perform_regular_scroll(target_server, current_servers)
            return 'regular'

    def _perform_small_scroll(self, target_server: int, current_servers: List[int]) -> bool:
        """Выполнение мелкого скроллинга с улучшенной точностью."""
        self.logger.debug(f"Выполняем мелкий скроллинг к серверу {target_server}")

        if not current_servers:
            # Если нет данных о текущих серверах, делаем скроллинг вниз
            scroll_down = True
        else:
            min_visible = min(current_servers)
            max_visible = max(current_servers)
            # Определяем направление более точно
            if target_server < min_visible:
                scroll_down = True
            elif target_server > max_visible:
                scroll_down = False
            else:
                # Цель в диапазоне, но не найдена - возможно нужно небольшой сдвиг
                # Определяем к какому краю диапазона ближе
                scroll_down = abs(target_server - min_visible) < abs(target_server - max_visible)

        # Используем более мелкий шаг скроллинга
        if scroll_down:
            start_coords = COORDINATES['server_small_scroll_start']
            end_coords = COORDINATES['server_small_scroll_end']
        else:
            start_coords = COORDINATES['server_small_scroll_end']
            end_coords = COORDINATES['server_small_scroll_start']

        # Выполняем еще более мелкий скроллинг
        small_duration = SERVER_RECOGNITION_SETTINGS['small_scroll_duration'] // 2  # Половина обычного времени
        self.adb.swipe(*start_coords, *end_coords, duration=small_duration)
        time.sleep(PAUSE_SETTINGS['after_server_scroll'])

        # Очищаем кеш после скроллинга
        self.cached_servers = {}
        return True

    def _perform_regular_scroll(self, target_server: int, current_servers: List[int]) -> bool:
        """Выполнение обычного скроллинга."""
        min_visible = min(current_servers) if current_servers else 500
        scroll_down = target_server < min_visible

        self.logger.debug(f"Выполняем {'вниз' if scroll_down else 'вверх'} скроллинг к серверу {target_server}")

        if scroll_down:
            start_coords = COORDINATES['server_scroll_start']
            end_coords = COORDINATES['server_scroll_end']
        else:
            start_coords = COORDINATES['server_scroll_end']
            end_coords = COORDINATES['server_scroll_start']

        self.adb.swipe(*start_coords, *end_coords,
                      duration=SERVER_RECOGNITION_SETTINGS['scroll_duration'])
        time.sleep(PAUSE_SETTINGS['after_server_scroll'])

        # Очищаем кеш после скроллинга
        self.cached_servers = {}
        return True