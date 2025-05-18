"""
Полностью переписанный game_bot.py с улучшенным определением координат серверов
и подробным логированием всех шагов обучения.
"""
import time
import logging
import cv2
import numpy as np
from typing import Optional, Tuple, List

from config import (
    IMAGE_PATHS, COORDINATES, SEASONS, DEFAULT_TIMEOUT, LOADING_TIMEOUT,
    GAME_PACKAGE, GAME_ACTIVITY, TEMPLATE_MATCHING_THRESHOLD,
    OCR_REGIONS, PAUSE_SETTINGS, SERVER_RECOGNITION_SETTINGS
)
from server_selector import OptimizedServerSelector


class OptimizedGameBot:
    """Оптимизированный класс бота для прохождения обучения в игре Sea of Conquest."""

    def __init__(self, adb_controller, image_handler):
        """
        Инициализация бота.

        Args:
            adb_controller: контроллер ADB
            image_handler: обработчик изображений
        """
        self.logger = logging.getLogger('sea_conquest_bot.game')
        self.adb = adb_controller
        self.image = image_handler
        self.ocr_available = self._check_ocr_availability()
        self.server_selector = OptimizedServerSelector(adb_controller, self.ocr_available)

    def _check_ocr_availability(self) -> bool:
        """Проверка доступности OCR."""
        try:
            import pytesseract
            return True
        except ImportError:
            self.logger.warning("OCR не доступен")
            return False

    # Основные методы управления игрой
    def start_game(self):
        """Запуск игры."""
        self.logger.info("Запуск игры")
        self.adb.start_app(GAME_PACKAGE, GAME_ACTIVITY)
        time.sleep(LOADING_TIMEOUT)

    def stop_game(self):
        """Остановка игры."""
        self.logger.info("Остановка игры")
        self.adb.stop_app(GAME_PACKAGE)

    # Методы взаимодействия с интерфейсом
    def click_coord(self, x: int, y: int):
        """Клик по координатам."""
        self.logger.debug(f"Клик по координатам ({x}, {y})")
        self.adb.tap(x, y)

    def click_image(self, image_key: str, timeout: int = 30) -> bool:
        """Поиск и клик по изображению."""
        if image_key not in IMAGE_PATHS:
            self.logger.error(f"Изображение '{image_key}' не найдено")
            return False
        return self.image.tap_on_template(IMAGE_PATHS[image_key], timeout)

    def wait_for_image(self, image_key: str, timeout: int = 30) -> Optional[Tuple[int, int, int, int]]:
        """Ожидание появления изображения."""
        if image_key not in IMAGE_PATHS:
            self.logger.error(f"Изображение '{image_key}' не найдено")
            return None
        return self.image.wait_for_template(IMAGE_PATHS[image_key], timeout)

    # Методы работы с текстом
    def find_text_on_screen(self, text: str, region: Optional[Tuple[int, int, int, int]] = None,
                           timeout: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        Поиск текста на экране с использованием OCR.

        Args:
            text: искомый текст
            region: область поиска (x, y, w, h)
            timeout: время ожидания

        Returns:
            tuple: координаты найденного текста
        """
        if not self.ocr_available:
            return None

        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            screenshot = self.adb.screenshot()
            if screenshot is None:
                time.sleep(0.5)
                continue

            # Определяем область поиска
            if region:
                x, y, w, h = region
                roi = screenshot[y:y + h, x:x + w]
                offset_x, offset_y = x, y
            else:
                roi = screenshot
                offset_x, offset_y = 0, 0

            # Поиск текста
            if self._find_text_in_image(roi, text):
                center_x = offset_x + roi.shape[1] // 2
                center_y = offset_y + roi.shape[0] // 2
                return (center_x, center_y, roi.shape[1], roi.shape[0])

            time.sleep(0.5)

        return None

    def _find_text_in_image(self, image: np.ndarray, target_text: str) -> bool:
        """Поиск текста в изображении."""
        try:
            import pytesseract

            # Предобработка
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Несколько методов обработки
            methods = [
                cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1],
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
            ]

            # Проверяем каждый метод
            target_lower = target_text.lower()
            for processed in methods:
                result = pytesseract.image_to_string(processed, lang='rus+eng')
                if target_lower in result.lower():
                    return True

            return False
        except Exception as e:
            self.logger.error(f"Ошибка поиска текста: {e}")
            return False

    def find_and_click_text(self, text: str, region: Optional[Tuple[int, int, int, int]] = None,
                          timeout: Optional[int] = None) -> bool:
        """Поиск и клик по тексту."""
        result = self.find_text_on_screen(text, region, timeout)
        if result:
            x, y, _, _ = result
            self.click_coord(x, y)
            return True
        return False

    # Методы работы с серверами
    def determine_season_for_server(self, server_id: int) -> Optional[str]:
        """Определение сезона для сервера."""
        for season_id, season_data in SEASONS.items():
            if season_data['min_server'] >= server_id >= season_data['max_server']:
                return season_id
        return None

    def select_server(self, server_id: int) -> bool:
        """
        Улучшенный выбор сервера с точными координатами.

        Args:
            server_id: номер сервера

        Returns:
            bool: успех операции
        """
        self.logger.info(f"Выбор сервера {server_id}")

        # Определяем сезон
        season_id = self.determine_season_for_server(server_id)
        if not season_id:
            self.logger.error(f"Сезон для сервера {server_id} не найден")
            return False

        # Выбираем сезон
        if not self.server_selector.select_season(season_id):
            return False

        # Ищем сервер
        return self._find_and_click_server(server_id)

    def _find_and_click_server(self, server_id: int) -> bool:
        """Поиск и клик по серверу."""
        # Попытка найти без скроллинга
        coords = self.server_selector.find_server_coordinates(server_id)
        if coords:
            self._click_server_at_coordinates(coords)
            return True

        # Скроллинг и поиск
        return self._scroll_and_find_server(server_id)

    def _scroll_and_find_server(self, server_id: int) -> bool:
        """Оптимизированный скроллинг и поиск сервера."""
        self.logger.info(f"Поиск сервера {server_id}")

        # Получаем текущие видимые сервера ОДИН раз
        current_servers = self.server_selector.get_servers_with_coordinates(force_refresh=True)
        if not current_servers:
            self.logger.warning("Не удалось получить сервера, используем резервный метод")
            return self._fallback_scroll_search(server_id)

        current_servers_list = list(current_servers.keys())

        # Быстрая проверка наличия сервера
        coords = self.server_selector.find_server_coordinates(server_id)
        if coords:
            self._click_server_at_coordinates(coords)
            return True

        # Основной цикл скроллинга - увеличено количество попыток
        max_attempts = 10  # Увеличено до 10 для лучшего поиска

        for attempt in range(max_attempts):
            self.logger.info(f"Попытка скроллинга {attempt + 1}/{max_attempts}")

            # Определяем тип скроллинга
            scroll_result = self.server_selector.scroll_to_server_range(server_id, current_servers_list)

            if scroll_result == 'found':
                # Цель должна быть видна
                coords = self.server_selector.find_server_coordinates(server_id)
                if coords:
                    self._click_server_at_coordinates(coords)
                    return True

            # Получаем новый список серверов после скроллинга ОДИН раз с небольшой задержкой
            time.sleep(0.5)  # Даем время на анимацию
            new_servers = self.server_selector.get_servers_with_coordinates(force_refresh=True)
            if new_servers:
                current_servers_list = list(new_servers.keys())

                # Проверяем, нашли ли целевой сервер
                if server_id in new_servers:
                    self.logger.info(f"Найден сервер {server_id} после скроллинга!")
                    coords = new_servers[server_id]
                    self._click_server_at_coordinates(coords)
                    return True

            current_servers = new_servers

        # Финальный поиск
        return self._final_server_search(server_id, current_servers_list)

    def _check_scroll_progress(self, old_servers: dict, new_servers: dict) -> bool:
        """Упрощенная проверка прогресса скроллинга."""
        if not old_servers or not new_servers:
            return True

        old_set = set(old_servers.keys()) if isinstance(old_servers, dict) else set(old_servers)
        new_set = set(new_servers.keys()) if isinstance(new_servers, dict) else set(new_servers)

        # Если есть хотя бы 50% новых серверов, считаем что есть прогресс
        intersection = old_set & new_set
        return len(intersection) < min(len(old_set), len(new_set)) * 0.5

    def _perform_opposite_scroll(self, server_id: int, current_servers: List[int]) -> bool:
        """Упрощенное выполнение скроллинга в противоположном направлении."""
        if not current_servers:
            return False

        self.logger.info("Скроллинг в противоположном направлении")

        # Определяем направление
        min_visible = min(current_servers)
        scroll_up = server_id > min_visible

        if scroll_up:
            start_coords = COORDINATES['server_scroll_end']
            end_coords = COORDINATES['server_scroll_start']
        else:
            start_coords = COORDINATES['server_scroll_start']
            end_coords = COORDINATES['server_scroll_end']

        self.adb.swipe(*start_coords, *end_coords, duration=800)
        time.sleep(1.0)
        return True

    def _final_server_search(self, server_id: int, current_servers: List[int]) -> bool:
        """Упрощенный финальный поиск сервера."""
        self.logger.info(f"Финальный поиск сервера {server_id}")

        # Ищем ближайший сервер в пределах сезона
        if current_servers:
            # Определяем сезон для целевого сервера
            target_season = self.determine_season_for_server(server_id)
            if target_season and target_season in SEASONS:
                season_data = SEASONS[target_season]
                season_min = min(season_data['min_server'], season_data['max_server'])
                season_max = max(season_data['min_server'], season_data['max_server'])

                # Фильтруем сервера по сезону
                season_servers = [s for s in current_servers if season_min <= s <= season_max]

                if season_servers:
                    closest = min(season_servers, key=lambda s: abs(s - server_id))
                    difference = abs(closest - server_id)

                    # Используем найденный сервер если он достаточно близко
                    if difference <= 3:  # Уменьшено с 5 до 3
                        self.logger.info(f"Выбираем ближайший сервер {closest} (разница: {difference})")
                        final_servers = self.server_selector.get_servers_with_coordinates()
                        if closest in final_servers:
                            coords = final_servers[closest]
                            self._click_server_at_coordinates(coords)
                            return True

        self.logger.error(f"Не удалось найти подходящий сервер для {server_id}")
        return False

    def _check_overshoot_improved(self, current_servers: List[int], target_server: int) -> bool:
        """Улучшенная проверка перескроллинга."""
        if not current_servers:
            return False

        min_visible = min(current_servers)
        max_visible = max(current_servers)

        # Проверяем, проскролили ли мы слишком далеко в любую сторону
        overshoot_threshold = SERVER_RECOGNITION_SETTINGS['overshoot_threshold']

        if target_server > max_visible + overshoot_threshold:
            self.logger.info(f"Перескроллинг вверх: цель {target_server}, видимые {min_visible}-{max_visible}")
            return True
        elif target_server < min_visible - overshoot_threshold:
            self.logger.info(f"Перескроллинг вниз: цель {target_server}, видимые {min_visible}-{max_visible}")
            return True

        return False

    def _fine_tune_scroll(self, target_server: int, current_servers: List[int]) -> bool:
        """Точная корректировка скроллинга при обнаружении перескроллинга."""
        self.logger.info("Выполняем точную корректировку скроллинга")

        max_fine_attempts = 3
        for attempt in range(max_fine_attempts):
            # Определяем направление корректировки
            min_visible = min(current_servers)
            max_visible = max(current_servers)

            if target_server > max_visible:
                # Нужно скроллить вверх (показать большие номера)
                direction = "up"
            elif target_server < min_visible:
                # Нужно скроллить вниз (показать меньшие номера)
                direction = "down"
            else:
                # Цель уже в видимом диапазоне
                return True

            # Выполняем мелкий скроллинг в нужном направлении
            if direction == "down":
                start_coords = COORDINATES['server_small_scroll_start']
                end_coords = COORDINATES['server_small_scroll_end']
            else:
                start_coords = COORDINATES['server_small_scroll_end']
                end_coords = COORDINATES['server_small_scroll_start']

            self.adb.swipe(*start_coords, *end_coords,
                           duration=SERVER_RECOGNITION_SETTINGS['small_scroll_duration'])
            time.sleep(PAUSE_SETTINGS['after_server_scroll'] * 0.5)  # Меньшая пауза для точной корректировки

            # Проверяем результат
            new_servers = self.server_selector.get_servers_with_coordinates()
            if new_servers:
                current_servers = list(new_servers.keys())
                coords = self.server_selector.find_server_coordinates(target_server)
                if coords:
                    return True

        return False

    def _try_nearest_server_improved(self, server_id: int, current_servers: List[int]) -> bool:
        """Улучшенная попытка выбрать ближайший сервер с дополнительным поиском."""
        if not current_servers:
            self.logger.error("Нет доступных серверов для выбора ближайшего")
            return False

        closest = min(current_servers, key=lambda s: abs(s - server_id))
        difference = abs(closest - server_id)

        # Если разница очень маленькая (1-2), пробуем еще немного поискать
        if difference <= 2:
            self.logger.info(f"Цель {server_id} очень близко к {closest}, пробуем дополнительный мелкий скроллинг")

            # Определяем направление для поиска точного сервера
            if server_id < closest:
                # Нужно скроллить вниз чуть-чуть
                start_coords = COORDINATES['server_small_scroll_start']
                end_coords = COORDINATES['server_small_scroll_end']
            else:
                # Нужно скроллить вверх чуть-чуть
                start_coords = COORDINATES['server_small_scroll_end']
                end_coords = COORDINATES['server_small_scroll_start']

            # Очень мелкий скроллинг
            extra_small_duration = 150  # Очень короткий скроллинг
            self.adb.swipe(*start_coords, *end_coords, duration=extra_small_duration)
            time.sleep(PAUSE_SETTINGS['after_server_scroll'] * 0.5)

            # Проверяем, появился ли нужный сервер
            coords = self.server_selector.find_server_coordinates(server_id)
            if coords:
                self.logger.info(f"Найден точный сервер {server_id} после дополнительного поиска")
                self._click_server_at_coordinates(coords)
                return True

        # Используем более строгий критерий для выбора ближайшего сервера
        max_difference = SERVER_RECOGNITION_SETTINGS.get('max_server_difference', 5)

        if difference <= max_difference:
            self.logger.info(f"Выбираем ближайший доступный сервер {closest} (разница: {difference})")
            servers_dict = self.server_selector.get_servers_with_coordinates()
            if closest in servers_dict:
                coords = servers_dict[closest]
                self._click_server_at_coordinates(coords)
                return True

        self.logger.error(f"Ближайший сервер {closest} слишком далеко от целевого {server_id} (разница: {difference})")
        return False

    def _fallback_scroll_search(self, server_id: int) -> bool:
        """Резервный метод поиска сервера при сбое основного алгоритма."""
        self.logger.info("Применяем резервный метод поиска сервера")

        # Выполняем несколько скроллингов в обоих направлениях
        directions = [
            ('down', COORDINATES['server_scroll_start'], COORDINATES['server_scroll_end']),
            ('up', COORDINATES['server_scroll_end'], COORDINATES['server_scroll_start'])
        ]

        for direction_name, start_coords, end_coords in directions:
            self.logger.info(f"Резервный поиск: скроллинг {direction_name}")

            for _ in range(3):  # Увеличено обратно до 3 попыток в каждом направлении
                self.adb.swipe(*start_coords, *end_coords,
                               duration=SERVER_RECOGNITION_SETTINGS['scroll_duration'])
                time.sleep(PAUSE_SETTINGS['after_server_scroll'])

                # Проверяем, появился ли нужный сервер
                servers = self.server_selector.get_servers_with_coordinates(force_refresh=True)
                if server_id in servers:
                    coords = servers[server_id]
                    self._click_server_at_coordinates(coords)
                    return True

        return False

    def _click_server_at_coordinates(self, coords: Tuple[int, int]):
        """Клик по серверу с паузами."""
        time.sleep(PAUSE_SETTINGS['before_server_click'])
        self.click_coord(coords[0], coords[1])
        time.sleep(PAUSE_SETTINGS['after_server_click'])

    # Специализированные методы для обучения
    def find_skip_button_infinite(self) -> bool:
        """
        Бесконечный поиск кнопки ПРОПУСТИТЬ только через OCR с разными методами обработки.
        Не останавливается пока кнопка не будет найдена и нажата.

        Returns:
            bool: всегда True (когда кнопка найдена)
        """
        self.logger.info("Начинаем бесконечный поиск кнопки ПРОПУСТИТЬ через OCR")

        skip_variants = [
            "ПРОПУСТИТЬ", "ПРОПУСТИTЬ", "ПРОNYСТИТЬ", "ПPОПУСТИТЬ",
            "ПРОПУCTИТЬ", "ПРOПУСТИТЬ", "ПPOПУСТИТЬ", "SKIP",
            "ПРОПYСТИТЬ", "ПРОПУСТИTЬ", "ПРОПУСТИТЬ", "ПРОПУСТИTь",
            "ПРОПУСТИТЬ >>", "ПРОПУСТИТЬ>", "ПРОПУСТИТЬ >", ">>",
            "ПРОПУCTИТЬ >>", "ПРOПУСТИТЬ >>", "ПРОПУСТИTЬ >>"
        ]

        region = OCR_REGIONS['skip_button']
        attempt = 0
        log_interval = 20  # Логируем каждые 20 попыток

        while True:
            attempt += 1

            # Логируем прогресс каждые N попыток
            if attempt % log_interval == 1:
                self.logger.info(f"Попытка поиска ПРОПУСТИТЬ #{attempt}")

            try:
                screenshot = self.adb.screenshot()
                if screenshot is None or screenshot.size == 0:
                    self.logger.debug("Пустой скриншот, повторяем")
                    time.sleep(0.5)
                    continue

                # Получаем область поиска
                x, y, w, h = region
                roi = screenshot[y:y + h, x:x + w]

                # Метод 1: Стандартная бинаризация (белый текст на темном фоне)
                result = self._ocr_method_binary_white_text(roi, skip_variants, attempt)
                if result:
                    coords = (x + result[0], y + result[1])
                    self.click_coord(coords[0], coords[1])
                    self.logger.info(f"ПРОПУСТИТЬ найден методом 1 (бинаризация белого текста) на попытке #{attempt}")
                    return True

                # Метод 2: Инвертированная бинаризация (темный текст на светлом фоне)
                result = self._ocr_method_binary_dark_text(roi, skip_variants, attempt)
                if result:
                    coords = (x + result[0], y + result[1])
                    self.click_coord(coords[0], coords[1])
                    self.logger.info(f"ПРОПУСТИТЬ найден методом 2 (бинаризация темного текста) на попытке #{attempt}")
                    return True

                # Метод 3: Специальный метод для полупрозрачных кнопок
                result = self._ocr_method_transparent_buttons(roi, skip_variants, attempt)
                if result:
                    coords = (x + result[0], y + result[1])
                    self.click_coord(coords[0], coords[1])
                    self.logger.info(f"ПРОПУСТИТЬ найден методом 3 (полупрозрачные кнопки) на попытке #{attempt}")
                    return True

                # Метод 4: Поиск по цвету (выделение определенных цветовых диапазонов)
                result = self._ocr_method_color_filtering(roi, skip_variants, attempt)
                if result:
                    coords = (x + result[0], y + result[1])
                    self.click_coord(coords[0], coords[1])
                    self.logger.info(f"ПРОПУСТИТЬ найден методом 4 (цветовая фильтрация) на попытке #{attempt}")
                    return True

                # Метод 5: Контрастное усиление
                result = self._ocr_method_contrast_enhancement(roi, skip_variants, attempt)
                if result:
                    coords = (x + result[0], y + result[1])
                    self.click_coord(coords[0], coords[1])
                    self.logger.info(f"ПРОПУСТИТЬ найден методом 5 (контрастное усиление) на попытке #{attempt}")
                    return True

                # Метод 6: Двойная проверка (комбинированный метод)
                result = self._ocr_method_double_check(roi, skip_variants, attempt)
                if result:
                    coords = (x + result[0], y + result[1])
                    self.click_coord(coords[0], coords[1])
                    self.logger.info(f"ПРОПУСТИТЬ найден методом 6 (двойная проверка) на попытке #{attempt}")
                    return True

                # Метод 7: Расширенная область поиска (каждые 10 попыток)
                if attempt % 10 == 0:
                    expanded_region = (max(0, x - 50), max(0, y - 20),
                                       min(screenshot.shape[1] - x, w + 100),
                                       min(screenshot.shape[0] - y, h + 40))
                    exp_x, exp_y, exp_w, exp_h = expanded_region
                    expanded_roi = screenshot[exp_y:exp_y + exp_h, exp_x:exp_x + exp_w]

                    result = self._ocr_method_binary_white_text(expanded_roi, skip_variants, attempt)
                    if result:
                        coords = (exp_x + result[0], exp_y + result[1])
                        self.click_coord(coords[0], coords[1])
                        self.logger.info(f"ПРОПУСТИТЬ найден методом 7 (расширенная область) на попытке #{attempt}")
                        return True

            except Exception as e:
                self.logger.debug(f"Ошибка в попытке #{attempt}: {e}")

            # Пауза между попытками
            time.sleep(0.3)

    def _ocr_method_binary_white_text(self, roi, skip_variants, attempt):
        """OCR метод: бинаризация для белого текста на темном фоне."""
        try:
            import pytesseract
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT,
                                             lang='rus+eng', config='--psm 6')

            return self._parse_ocr_data_with_validation(data, skip_variants)
        except:
            return None

    def _ocr_method_binary_dark_text(self, roi, skip_variants, attempt):
        """OCR метод: инвертированная бинаризация для темного текста на светлом фоне."""
        try:
            import pytesseract
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT,
                                             lang='rus+eng', config='--psm 6')

            return self._parse_ocr_data_with_validation(data, skip_variants)
        except:
            return None

    def _ocr_method_double_check(self, roi, skip_variants, attempt):
        """OCR метод: двойная проверка - комбинирует несколько надежных методов."""
        try:
            import pytesseract
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Метод 1: Стандартная бинаризация
            _, binary1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            data1 = pytesseract.image_to_data(binary1, output_type=pytesseract.Output.DICT,
                                              lang='rus+eng', config='--psm 6')
            result1 = self._parse_ocr_data_with_validation(data1, skip_variants)

            # Метод 2: Инвертированная бинаризация
            _, binary2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            data2 = pytesseract.image_to_data(binary2, output_type=pytesseract.Output.DICT,
                                              lang='rus+eng', config='--psm 6')
            result2 = self._parse_ocr_data_with_validation(data2, skip_variants)

            # Если оба метода нашли текст в похожих местах - это надежный результат
            if result1 and result2:
                # Проверяем, что координаты близко (в пределах 50 пикселей)
                if abs(result1[0] - result2[0]) <= 50 and abs(result1[1] - result2[1]) <= 50:
                    # Возвращаем среднее значение координат
                    avg_x = (result1[0] + result2[0]) // 2
                    avg_y = (result1[1] + result2[1]) // 2
                    return (avg_x, avg_y)

            # Если только один метод нашел - возвращаем его (но с меньшей уверенностью)
            if result1:
                return result1
            if result2:
                return result2

            return None
        except:
            return None

    def _parse_ocr_data_with_validation(self, data, skip_variants):
        """Парсинг данных OCR с дополнительной валидацией."""
        for i in range(len(data['text'])):
            text = data['text'][i].strip().upper()
            confidence = int(data['conf'][i])
            width = data['width'][i]
            height = data['height'][i]

            # Повышенные требования к уверенности
            if confidence < 50:
                continue

            # Проверяем размер найденного текста (кнопка не может быть слишком маленькой)
            if width < 30 or height < 10:
                continue

            # Проверяем каждый вариант
            for variant in skip_variants:
                variant_upper = variant.upper()

                # Точное совпадение (самый надежный)
                if text == variant_upper:
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    return (x, y)

                # Частичное совпадение (только если высокая уверенность)
                if confidence >= 70 and (variant_upper in text or text in variant_upper):
                    # Дополнительная проверка: длина найденного текста должна быть разумной
                    if len(text) >= 4 and len(text) <= 20:
                        x = data['left'][i] + data['width'][i] // 2
                        y = data['top'][i] + data['height'][i] // 2
                        return (x, y)

        return None

    def _ocr_method_transparent_buttons(self, roi, skip_variants, attempt):
        """OCR метод: специально для полупрозрачных кнопок с темным фоном."""
        try:
            import pytesseract

            # Преобразуем в градации серого
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Метод 1: Выделение светлого текста на темном фоне с низким порогом
            _, binary1 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

            # Метод 2: Выделение с более высоким порогом (для очень светлого текста)
            _, binary2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

            # Метод 3: Автоматический порог по методу Отцу
            _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Метод 4: Комбинированный - объединяем результаты
            combined = cv2.bitwise_or(cv2.bitwise_or(binary1, binary2), binary3)

            # Morfологическая обработка для очистки от шумов
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            # Увеличиваем изображение для лучшего распознавания
            h, w = cleaned.shape
            resized = cv2.resize(cleaned, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

            # OCR с более мягким PSM
            data = pytesseract.image_to_data(resized, output_type=pytesseract.Output.DICT,
                                             lang='rus+eng', config='--psm 8')

            # Парсим результаты (с учетом масштабирования)
            for i in range(len(data['text'])):
                text = data['text'][i].strip().upper()
                confidence = int(data['conf'][i])

                # Для полупрозрачных кнопок используем более низкий порог уверенности
                if confidence < 35:
                    continue

                # Проверяем каждый вариант
                for variant in skip_variants:
                    variant_upper = variant.upper()

                    # Очень точная проверка для коротких вариантов типа ">>"
                    if variant_upper == ">>" and ">>" in text:
                        # Возвращаем координаты с учетом масштабирования
                        x = (data['left'][i] + data['width'][i] // 2) // 2
                        y = (data['top'][i] + data['height'][i] // 2) // 2
                        return (x, y)

                    # Обычная проверка
                    if variant_upper in text or text in variant_upper:
                        # Дополнительная проверка для избежания ложных срабатываний
                        if len(text) >= 2:  # Минимум 2 символа
                            # Возвращаем координаты с учетом масштабирования
                            x = (data['left'][i] + data['width'][i] // 2) // 2
                            y = (data['top'][i] + data['height'][i] // 2) // 2
                            return (x, y)

            return None
        except:
            return None
        """OCR метод: фильтрация по цвету."""
        try:
            import pytesseract
            # Преобразуем в HSV для лучшей работы с цветами
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Диапазон для белого/светлого текста
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([255, 30, 255])

            # Диапазон для желтого текста (часто используется для кнопок)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])

            # Создаем маски
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_combined = cv2.bitwise_or(mask_white, mask_yellow)

            data = pytesseract.image_to_data(mask_combined, output_type=pytesseract.Output.DICT,
                                             lang='rus+eng', config='--psm 6')

            return self._parse_ocr_data(data, skip_variants)
        except:
            return None

    def _ocr_method_contrast_enhancement(self, roi, skip_variants, attempt):
        """OCR метод: улучшение контраста."""
        try:
            import pytesseract
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Улучшение контраста с помощью CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT,
                                             lang='rus+eng', config='--psm 6')

            return self._parse_ocr_data(data, skip_variants)
        except:
            return None

    def _parse_ocr_data(self, data, skip_variants):
        """Парсинг данных OCR и поиск текста ПРОПУСТИТЬ с базовой валидацией."""
        for i in range(len(data['text'])):
            text = data['text'][i].strip().upper()
            confidence = int(data['conf'][i])

            # Повышенные требования к уверенности
            if confidence < 40:
                continue

            # Проверяем каждый вариант
            for variant in skip_variants:
                variant_upper = variant.upper()

                # Точное совпадение
                if text == variant_upper:
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    return (x, y)

                # Частичное совпадение (только для высокой уверенности)
                if confidence >= 60 and variant_upper in text:
                    x = data['left'][i] + data['width'][i] // 2
                    y = data['top'][i] + data['height'][i] // 2
                    return (x, y)

        return None

    # Основные методы выполнения обучения
    def perform_tutorial(self, server_id: int, start_step: int = 1) -> bool:
        """
        Выполнение обучения на сервере.

        Args:
            server_id: номер сервера
            start_step: начальный шаг

        Returns:
            bool: успех выполнения
        """
        self.logger.info(f"Начало обучения на сервере {server_id} с шага {start_step}")

        try:
            if start_step <= 6:
                if not self._execute_initial_steps(server_id, start_step):
                    return False

            if start_step <= 97:
                if not self._execute_tutorial_steps(max(start_step, 7)):
                    return False

            self.logger.info(f"Обучение на сервере {server_id} завершено успешно")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка выполнения обучения: {e}", exc_info=True)
            return False

    def _execute_initial_steps(self, server_id: int, start_step: int) -> bool:
        """Выполнение начальных шагов (1-6)."""
        # Шаг 1: Открываем профиль (первый шаг без задержки по ТЗ)
        if start_step <= 1:
            self.logger.info("Шаг 1: Клик по координатам (52, 50) - открываем профиль")
            self.click_coord(52, 50)

        # Шаг 2: Открываем настройки - добавляем задержку до действия
        if start_step <= 2:
            self.logger.info("Шаг 2: Ждем 1.5 сек и открываем настройки")
            time.sleep(1.5)
            self.click_coord(1076, 31)

        # Шаг 3: Открываем вкладку персонажей - добавляем задержку до действия
        if start_step <= 3:
            self.logger.info("Шаг 3: Ждем 1.5 сек и открываем вкладку персонажей")
            time.sleep(1.5)
            self.click_coord(643, 319)

        # Шаг 4: Создаем персонажа на новом сервере - добавляем задержку до действия
        if start_step <= 4:
            self.logger.info("Шаг 4: Ждем 1.5 сек и создаем персонажа на новом сервере")
            time.sleep(1.5)
            self.click_coord(271, 181)

        # Шаг 5: Выбор сервера
        if start_step <= 5:
            self.logger.info(f"Шаг 5: Выбор сервера {server_id}")
            if not self.select_server(server_id):
                self.logger.error(f"Не удалось выбрать сервер {server_id}")
                return False

        # Шаг 6: Подтверждаем создание персонажа
        if start_step <= 6:
            self.logger.info("Шаг 6: Ждем 2.5 сек и подтверждаем создание персонажа")
            time.sleep(2.5)
            self.click_coord(787, 499)

            # Добавляем дополнительную задержку в 17 секунд после шага 6 для загрузки
            self.logger.info("Ожидание 17 секунд после подтверждения создания персонажа...")
            time.sleep(17)

        return True

    def _execute_tutorial_steps(self, start_step: int) -> bool:
        """
        Выполнение оставшихся шагов обучения с подробным логированием.
        Бесконечное ожидание кнопки ПРОПУСТИТЬ через разные OCR методы.

        Args:
            start_step: начальный шаг для выполнения

        Returns:
            bool: True если шаги выполнены успешно, False иначе
        """
        self.logger.info(f"Выполнение оставшихся шагов обучения, начиная с шага {start_step}")

        try:
            # Шаг 7: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 7:
                self.logger.info("Шаг 7: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 7: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                time.sleep(1)

            # Шаг 8: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 8:
                self.logger.info("Шаг 8: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 8: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                time.sleep(1)

            # Шаг 9: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 9:
                self.logger.info("Шаг 9: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 9: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                time.sleep(1)

            # Шаг 10: Активируем бой нажатием на пушку
            if start_step <= 10:
                self.logger.info("Шаг 10: Планируем активировать бой - ищем cannon_is_ready.png и нажимаем (718, 438)")
                if self.wait_for_image('cannon_is_ready', timeout=20):
                    self.logger.info("Шаг 10: Изображение cannon_is_ready.png найдено, выполняем клик по (718, 438)")
                    self.click_coord(718, 438)
                    self.logger.info("Шаг 10: ВЫПОЛНЕН - клик по пушке выполнен (через изображение)")
                else:
                    self.logger.warning("Шаг 10: Изображение cannon_is_ready.png не найдено, выполняем клик по координатам")
                    self.click_coord(718, 438)
                    self.logger.info("Шаг 10: ВЫПОЛНЕН - клик по пушке выполнен (по координатам)")

            # Шаг 11: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 11:
                self.logger.info("Шаг 11: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 11: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 12: Ждем появления hell_henry.png и нажимаем ПРОПУСТИТЬ
            if start_step <= 12:
                self.logger.info('Шаг 12: Планируем дождаться hell_henry.png и нажать ПРОПУСТИТЬ')
                if self.wait_for_image('hell_henry', timeout=15):
                    self.logger.info("Шаг 12: Изображение hell_henry.png найдено, ищем ПРОПУСТИТЬ")
                    self.find_skip_button_infinite()
                    self.logger.info("Шаг 12: ВЫПОЛНЕН - hell_henry найден, ПРОПУСТИТЬ нажат")
                else:
                    self.logger.warning("Шаг 12: Изображение hell_henry.png не найдено, но ищем ПРОПУСТИТЬ")
                    self.find_skip_button_infinite()
                    self.logger.info("Шаг 12: ВЫПОЛНЕН - ПРОПУСТИТЬ нажат (без hell_henry)")

            # Шаг 13: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 13:
                self.logger.info("Шаг 13: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 13: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 14: Жмем на иконку кораблика
            if start_step <= 14:
                self.logger.info("Шаг 14: Планируем нажать на иконку кораблика (58, 654) через 3 секунды")
                time.sleep(3)
                self.logger.info("Шаг 14: Выполняем клик по координатам (58, 654)")
                self.click_coord(58, 654)
                self.logger.info("Шаг 14: ВЫПОЛНЕН - клик по иконке кораблика выполнен")

            # Шаг 15: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 15:
                self.logger.info("Шаг 15: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 15: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 16: Отстраиваем нижнюю палубу
            if start_step <= 16:
                self.logger.info("Шаг 16: Планируем отстроить нижнюю палубу - клик по (638, 403) через 2 секунды")
                time.sleep(2)
                self.logger.info("Шаг 16: Выполняем клик по координатам (638, 403)")
                self.click_coord(638, 403)
                self.logger.info("Шаг 16: ВЫПОЛНЕН - постройка нижней палубы инициирована")

            # Шаг 17: Отстраиваем паб в нижней палубе
            if start_step <= 17:
                self.logger.info("Шаг 17: Планируем отстроить паб в нижней палубе - клик по (635, 373) через 2.5 секунды")
                time.sleep(2.5)
                self.logger.info("Шаг 17: Выполняем клик по координатам (635, 373)")
                self.click_coord(635, 373)
                self.logger.info("Шаг 17: ВЫПОЛНЕН - постройка паба в нижней палубе инициирована")

            # Шаг 18: Латаем дыры в складе на нижней палубе
            if start_step <= 18:
                self.logger.info("Шаг 18: Планируем залатать дыры в складе - клик по (635, 373) через 2.5 секунды")
                time.sleep(2.5)
                self.logger.info("Шаг 18: Выполняем клик по координатам (635, 373)")
                self.click_coord(635, 373)
                self.logger.info("Шаг 18: ВЫПОЛНЕН - ремонт склада инициирован")

            # Шаг 19: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 19:
                self.logger.info("Шаг 19: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 19: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 20: Отстраиваем верхнюю палубу
            if start_step <= 20:
                self.logger.info("Шаг 20: Планируем отстроить верхнюю палубу - клик по (345, 386) через 2.5 секунды")
                time.sleep(2.5)
                self.logger.info("Шаг 20: Выполняем клик по координатам (345, 386)")
                self.click_coord(345, 386)
                self.logger.info("Шаг 20: ВЫПОЛНЕН - постройка верхней палубы инициирована")

            # Шаг 21: Выбираем пушку
            if start_step <= 21:
                self.logger.info("Шаг 21: Планируем выбрать пушку - клик по (77, 276) через 1.5 секунды")
                time.sleep(1.5)
                self.logger.info("Шаг 21: Выполняем клик по координатам (77, 276)")
                self.click_coord(77, 276)
                self.logger.info("Шаг 21: ВЫПОЛНЕН - пушка выбрана")

            # Шаг 22: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 22:
                self.logger.info("Шаг 22: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 22: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 23: Ищем collect_items.png и нажимаем на координаты
            if start_step <= 23:
                self.logger.info('Шаг 23: Планируем собрать предметы - ищем collect_items.png и нажимаем (741, 145)')
                if self.wait_for_image('collect_items', timeout=15):
                    self.logger.info("Шаг 23: Изображение collect_items.png найдено, выполняем клик по (741, 145)")
                    self.click_coord(741, 145)
                    self.logger.info("Шаг 23: ВЫПОЛНЕН - предметы собраны (через изображение)")
                else:
                    self.logger.warning("Шаг 23: Изображение collect_items.png не найдено, выполняем клик по координатам")
                    self.click_coord(741, 145)
                    self.logger.info("Шаг 23: ВЫПОЛНЕН - клик для сбора предметов выполнен (по координатам)")

            # Шаг 24: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 24:
                self.logger.info("Шаг 24: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 24: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                time.sleep(0.5)

            # Шаг 25: Нажимаем на квест "Старый соперник"
            if start_step <= 25:
                self.logger.info('Шаг 25: Планируем начать квест "Старый соперник" - клик по (93, 285) через 1.5 секунды')
                time.sleep(1.5)
                self.logger.info("Шаг 25: Выполняем клик по координатам (93, 285)")
                self.click_coord(93, 285)
                self.logger.info('Шаг 25: ВЫПОЛНЕН - квест "Старый соперник" активирован')

            # Шаг 26: Ждем и нажимаем кнопку закрытия диалога
            if start_step <= 26:
                self.logger.info("Шаг 26: Планируем закрыть диалог - клик по (1142, 42) через 8 секунд")
                time.sleep(8)
                self.logger.info("Шаг 26: Выполняем клик по координатам (1142, 42)")
                self.click_coord(1142, 42)
                self.logger.info("Шаг 26: ВЫПОЛНЕН - диалог закрыт")

            # Шаг 27: Нажимаем на квест "Старый соперник" повторно
            if start_step <= 27:
                self.logger.info('Шаг 27: Планируем повторно активировать квест "Старый соперник" - клик по (93, 285) через 1.5 секунды')
                time.sleep(1.5)
                self.logger.info("Шаг 27: Выполняем клик по координатам (93, 285)")
                self.click_coord(93, 285)
                self.logger.info('Шаг 27: ВЫПОЛНЕН - квест "Старый соперник" повторно активирован')

            # Шаг 28: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 28:
                self.logger.info("Шаг 28: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 28: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                time.sleep(1)

            # Шаг 29: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 29:
                self.logger.info("Шаг 29: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 29: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                time.sleep(3)

            # Шаг 30: Жмем на любую часть экрана чтобы продолжить после победы
            if start_step <= 30:
                self.logger.info("Шаг 30: Планируем продолжить после победы - клик по центру экрана (630, 413)")
                self.click_coord(630, 413)
                self.logger.info("Шаг 30: ВЫПОЛНЕН - клик для продолжения после победы выполнен")

            # Шаг 31: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 31:
                self.logger.info("Шаг 31: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 31: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                time.sleep(0.5)

            # Шаг 32: Ждем появления gold_compas.png и жмем на компас
            if start_step <= 32:
                self.logger.info("Шаг 32: Планируем активировать компас - ожидаем gold_compas.png и нажимаем (1074, 88)")
                if self.wait_for_image('gold_compas', timeout=15):
                    self.logger.info("Шаг 32: Изображение gold_compas.png найдено, выполняем клик по (1074, 88)")
                    time.sleep(0.5)
                    self.click_coord(1074, 88)
                    self.logger.info("Шаг 32: ВЫПОЛНЕН - компас активирован (через изображение)")
                else:
                    self.logger.warning("Шаг 32: Изображение gold_compas.png не найдено, выполняем клик по координатам")
                    time.sleep(0.5)
                    self.click_coord(1074, 88)
                    self.logger.info("Шаг 32: ВЫПОЛНЕН - клик по компасу выполнен (по координатам)")

            # Шаг 33: Еще раз жмем на компас
            if start_step <= 33:
                self.logger.info("Шаг 33: Планируем повторно активировать компас - клик по (701, 258) через 1.5 секунды")
                time.sleep(1.5)
                self.logger.info("Шаг 33: Выполняем клик по координатам (701, 258)")
                self.click_coord(701, 258)
                self.logger.info("Шаг 33: ВЫПОЛНЕН - компас повторно активирован")

            # Шаг 34: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 34:
                self.logger.info("Шаг 34: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 34: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 35: Жмем назад чтобы выйти из вкладки компаса
            if start_step <= 35:
                self.logger.info("Шаг 35: Планируем выйти из вкладки компаса - клик по кнопке назад (145, 25)")
                self.click_coord(145, 25)
                self.logger.info("Шаг 35: ВЫПОЛНЕН - выход из вкладки компаса выполнен")

            # Шаг 36: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 36:
                self.logger.info("Шаг 36: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 36: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 37: Ожидаем изображение long_song.png и нажимаем на квест "Далекая песня"
            if start_step <= 37:
                self.logger.info('Шаг 37: Планируем активировать квест "Далекая песня" - ожидаем long_song.png и нажимаем (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.logger.info("Шаг 37: Изображение long_song.png найдено, выполняем клик по (93, 285)")
                    self.click_coord(93, 285)
                    self.logger.info('Шаг 37: ВЫПОЛНЕН - квест "Далекая песня" активирован (через изображение)')
                else:
                    self.logger.warning("Шаг 37: Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)
                    self.logger.info('Шаг 37: ВЫПОЛНЕН - клик на квест "Далекая песня" выполнен (по координатам)')

            # Шаг 38: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 38:
                self.logger.info("Шаг 38: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 38: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 39: Ожидаем изображение long_song.png и еще раз нажимаем на квест "Далекая песня"
            if start_step <= 39:
                self.logger.info('Шаг 39: Планируем повторно активировать квест "Далекая песня" - ожидаем long_song.png и нажимаем (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.logger.info("Шаг 39: Изображение long_song.png найдено, выполняем клик по (93, 285)")
                    self.click_coord(93, 285)
                    self.logger.info('Шаг 39: ВЫПОЛНЕН - квест "Далекая песня" повторно активирован (через изображение)')
                else:
                    self.logger.warning("Шаг 39: Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)
                    self.logger.info('Шаг 39: ВЫПОЛНЕН - клик на квест "Далекая песня" повторно выполнен (по координатам)')

            # Шаг 40: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 40:
                self.logger.info("Шаг 40: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 40: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 41: Ожидаем изображение confirm_trade.png и нажимаем "Согласиться на обмен"
            if start_step <= 41:
                self.logger.info('Шаг 41: Планируем согласиться на обмен - ожидаем confirm_trade.png и нажимаем (151, 349)')
                if self.wait_for_image('confirm_trade', timeout=15):
                    self.logger.info("Шаг 41: Изображение confirm_trade.png найдено, выполняем клик по (151, 349)")
                    self.click_coord(151, 349)
                    self.logger.info("Шаг 41: ВЫПОЛНЕН - согласие на обмен подтверждено (через изображение)")
                else:
                    self.logger.warning("Шаг 41: Изображение confirm_trade.png не найдено, выполняем клик по координатам")
                    self.click_coord(151, 349)
                    self.logger.info("Шаг 41: ВЫПОЛНЕН - клик для согласия на обмен выполнен (по координатам)")

            # Шаг 42: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 42:
                self.logger.info("Шаг 42: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 42: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 43: Ожидаем изображение long_song.png и нажимаем на квест "Исследовать залив Мертвецов"
            if start_step <= 43:
                self.logger.info('Шаг 43: Планируем начать исследование залива Мертвецов - ожидаем long_song.png и нажимаем (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.logger.info("Шаг 43: Изображение long_song.png найдено, выполняем клик по (93, 285)")
                    self.click_coord(93, 285)
                    self.logger.info('Шаг 43: ВЫПОЛНЕН - квест "Исследовать залив Мертвецов" активирован (через изображение)')
                else:
                    self.logger.warning("Шаг 43: Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)
                    self.logger.info('Шаг 43: ВЫПОЛНЕН - клик на квест "Исследовать залив Мертвецов" выполнен (по координатам)')

            # Шаг 44: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 44:
                self.logger.info("Шаг 44: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 44: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 45: Ожидаем изображение prepare_for_battle.png и нажимаем на координаты
            if start_step <= 45:
                self.logger.info('Шаг 45: Планируем подготовиться к битве - ожидаем prepare_for_battle.png и нажимаем (85, 634)')
                if self.wait_for_image('prepare_for_battle', timeout=15):
                    self.logger.info("Шаг 45: Изображение prepare_for_battle.png найдено, выполняем клик по (85, 634)")
                    self.click_coord(85, 634)
                    self.logger.info("Шаг 45: ВЫПОЛНЕН - подготовка к битве активирована (через изображение)")
                else:
                    self.logger.warning("Шаг 45: Изображение prepare_for_battle.png не найдено, выполняем клик по координатам")
                    self.click_coord(85, 634)
                    self.logger.info("Шаг 45: ВЫПОЛНЕН - клик для подготовки к битве выполнен (по координатам)")

            # Шаг 46: Нажимаем начать битву
            if start_step <= 46:
                self.logger.info("Шаг 46: Планируем начать битву - клик по (1157, 604) через 0.5 секунды")
                time.sleep(0.5)
                self.logger.info("Шаг 46: Выполняем клик по координатам (1157, 604)")
                self.click_coord(1157, 604)
                self.logger.info("Шаг 46: ВЫПОЛНЕН - битва запущена")
                time.sleep(3)

            # Шаг 47: Нажимаем каждые 1.5 секунды по координатам, пока не появится start_battle.png
            if start_step <= 47:
                self.logger.info("Шаг 47: Планируем дождаться готовности к битве - ищем start_battle.png")
                found = False
                for attempt in range(20):  # Максимум 20 попыток
                    self.logger.debug(f"Шаг 47: Попытка {attempt + 1}/20 - ищем start_battle.png")
                    if self.click_image("start_battle", timeout=1):
                        self.logger.info(f"Шаг 47: ВЫПОЛНЕН - start_battle.png найден и нажат на попытке {attempt + 1}")
                        found = True
                        break
                    self.logger.debug("Шаг 47: start_battle.png не найден, кликаем по центру экрана")
                    self.click_coord(642, 334)
                    time.sleep(1.5)

                if not found:
                    self.logger.warning("Шаг 47: start_battle.png не найден за 20 попыток, но продолжаем")

            # Шаг 48: Нажимаем каждые 1.5 секунды, пока не найдем изображение ship_waiting_zaliz.png
            if start_step <= 48:
                self.logger.info("Шаг 48: Планируем дождаться корабля - ищем ship_waiting_zaliz.png")
                found = False
                for attempt in range(20):  # Максимум 20 попыток
                    self.logger.debug(f"Шаг 48: Попытка {attempt + 1}/20 - ищем ship_waiting_zaliz.png")
                    if self.wait_for_image('ship_waiting_zaliz', timeout=1):
                        self.logger.info(f"Шаг 48: ship_waiting_zaliz.png найден на попытке {attempt + 1}, кликаем по (93, 285)")
                        self.click_coord(93, 285)
                        self.logger.info("Шаг 48: ВЫПОЛНЕН - корабль найден и активирован")
                        found = True
                        break
                    self.logger.debug("Шаг 48: ship_waiting_zaliz.png не найден, кликаем по центру экрана")
                    self.click_coord(642, 334)
                    time.sleep(1.5)

                if not found:
                    self.logger.warning("Шаг 48: ship_waiting_zaliz.png не найден за 20 попыток, кликаем по квесту")
                    self.click_coord(93, 285)
                    self.logger.info("Шаг 48: ЗАВЕРШЕН - клик по квесту выполнен как резерв")

            # Шаг 49: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 49:
                self.logger.info("Шаг 49: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 49: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 50: Ожидаем изображение long_song_2.png и нажимаем на квест
            if start_step <= 50:
                self.logger.info('Шаг 50: Планируем активировать следующий этап квеста - ожидаем long_song_2.png и нажимаем (93, 285)')
                if self.wait_for_image('long_song_2', timeout=15):
                    self.logger.info("Шаг 50: Изображение long_song_2.png найдено, выполняем клик по (93, 285)")
                    self.click_coord(93, 285)
                    self.logger.info("Шаг 50: ВЫПОЛНЕН - следующий этап квеста активирован (через изображение)")
                else:
                    self.logger.warning("Шаг 50: Изображение long_song_2.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)
                    self.logger.info("Шаг 50: ВЫПОЛНЕН - клик на квест выполнен (по координатам)")

            # Шаг 51: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 51:
                self.logger.info("Шаг 51: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 51: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 52: Нажимаем на череп в заливе мертвецов
            if start_step <= 52:
                self.logger.info("Шаг 52: Планируем активировать череп в заливе мертвецов - клик по (653, 403) через 2 секунды")
                time.sleep(2)
                self.logger.info("Шаг 52: Выполняем клик по координатам (653, 403)")
                self.click_coord(653, 403)
                self.logger.info("Шаг 52: ВЫПОЛНЕН - череп в заливе мертвецов активирован")

            # Шаги 53-57: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(53, 58):
                if start_step <= step:
                    self.logger.info(f"Шаг {step}: Планируем найти и нажать ПРОПУСТИТЬ")
                    self.find_skip_button_infinite()
                    self.logger.info(f"Шаг {step}: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                    # Добавляем задержку 0.5 секунд для шагов с 53 по 56 включительно
                    if 53 <= step <= 56:
                        time.sleep(0.5)

            # Шаг 58: Ожидаем изображение long_song_3.png и нажимаем на квест
            if start_step <= 58:
                self.logger.info('Шаг 58: Планируем продолжить квест - ожидаем long_song_3.png и нажимаем (93, 285)')
                if self.wait_for_image('long_song_3', timeout=15):
                    self.logger.info("Шаг 58: Изображение long_song_3.png найдено, выполняем клик по (93, 285)")
                    self.click_coord(93, 285)
                    self.logger.info("Шаг 58: ВЫПОЛНЕН - квест продолжен (через изображение)")
                else:
                    self.logger.warning("Шаг 58: Изображение long_song_3.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)
                    self.logger.info("Шаг 58: ВЫПОЛНЕН - клик на квест выполнен (по координатам)")

            # Шаг 59: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 59:
                self.logger.info("Шаг 59: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 59: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 60: Ожидаем изображение long_song.png и нажимаем на квест
            if start_step <= 60:
                self.logger.info('Шаг 60: Планируем вернуться к исходному квесту - ожидаем long_song.png и нажимаем (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.logger.info("Шаг 60: Изображение long_song.png найдено, выполняем клик по (93, 285)")
                    self.click_coord(93, 285)
                    self.logger.info("Шаг 60: ВЫПОЛНЕН - возврат к исходному квесту (через изображение)")
                else:
                    self.logger.warning("Шаг 60: Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)
                    self.logger.info("Шаг 60: ВЫПОЛНЕН - клик на квест выполнен (по координатам)")

            # Шаг 61: Ожидаем изображение long_song_4.png и нажимаем на иконку молоточка
            if start_step <= 61:
                self.logger.info('Шаг 61: Планируем открыть меню строительства - ожидаем long_song_4.png и нажимаем (43, 481)')
                if self.wait_for_image('long_song_4', timeout=15):
                    self.logger.info("Шаг 61: Изображение long_song_4.png найдено, выполняем клик по (43, 481)")
                    self.click_coord(43, 481)
                    self.logger.info("Шаг 61: ВЫПОЛНЕН - меню строительства открыто (через изображение)")
                else:
                    self.logger.warning("Шаг 61: Изображение long_song_4.png не найдено, выполняем клик по координатам")
                    self.click_coord(43, 481)
                    self.logger.info("Шаг 61: ВЫПОЛНЕН - клик на иконку молоточка выполнен (по координатам)")

            # Шаг 62: Ожидаем изображение long_song_5.png и выбираем корабль для улучшения
            if start_step <= 62:
                self.logger.info('Шаг 62: Планируем выбрать корабль для улучшения - ожидаем long_song_5.png и нажимаем (127, 216)')
                if self.wait_for_image('long_song_5', timeout=15):
                    self.logger.info("Шаг 62: Изображение long_song_5.png найдено, выполняем клик по (127, 216)")
                    self.click_coord(127, 216)
                    self.logger.info("Шаг 62: ВЫПОЛНЕН - корабль для улучшения выбран (через изображение)")
                else:
                    self.logger.warning("Шаг 62: Изображение long_song_5.png не найдено, выполняем клик по координатам")
                    self.click_coord(127, 216)
                    self.logger.info("Шаг 62: ВЫПОЛНЕН - клик для выбора корабля выполнен (по координатам)")

            # Шаг 63: Дожидаемся надписи "УЛУЧШИТЬ" и кликаем на нее
            if start_step <= 63:
                self.logger.info('Шаг 63: Планируем улучшить корабль - ищем текст "УЛУЧШИТЬ"')
                if not self.find_and_click_text("УЛУЧШИТЬ", region=(983, 588, 200, 100), timeout=5):
                    self.logger.warning('Шаг 63: Текст "УЛУЧШИТЬ" не найден, кликаем по примерным координатам')
                    self.click_coord(1083, 638)
                    self.logger.info("Шаг 63: ВЫПОЛНЕН - клик для улучшения выполнен (по координатам)")
                else:
                    self.logger.info('Шаг 63: ВЫПОЛНЕН - кнопка "УЛУЧШИТЬ" найдена и нажата')

            # Шаг 64: Жмем назад чтобы выйти из вкладки корабля
            if start_step <= 64:
                self.logger.info("Шаг 64: Планируем выйти из вкладки корабля - клик по кнопке назад (145, 25) через 2 секунды")
                time.sleep(3)
                self.logger.info("Шаг 64: Выполняем клик по координатам (145, 25)")
                self.click_coord(145, 25)
                self.logger.info("Шаг 64: ВЫПОЛНЕН - выход из вкладки корабля выполнен")

            # Шаг 65: Жмем на кнопку постройки
            if start_step <= 65:
                self.logger.info("Шаг 65: Планируем открыть меню постройки - клик по (639, 603) через 1.5 секунды")
                time.sleep(1.5)
                self.logger.info("Шаг 65: Выполняем клик по координатам (639, 603)")
                self.click_coord(639, 603)
                self.logger.info("Шаг 65: ВЫПОЛНЕН - меню постройки открыто")

            # Шаг 66: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 66:
                self.logger.info("Шаг 66: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 66: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 67: Жмем на иконку компаса
            if start_step <= 67:
                self.logger.info("Шаг 67: Планируем открыть компас - клик по (1072, 87) через 1.5 секунды")
                time.sleep(1.5)
                self.logger.info("Шаг 67: Выполняем клик по координатам (1072, 87)")
                self.click_coord(1072, 87)
                self.logger.info("Шаг 67: ВЫПОЛНЕН - компас открыт")
                time.sleep(5)

            # Шаг 68: Ожидаем изображение long_song_6.png и нажимаем на квест
            if start_step <= 68:
                self.logger.info('Шаг 68: Планируем активировать новый квест - ожидаем long_song_6.png и нажимаем (101, 284)')
                if self.wait_for_image('long_song_6', timeout=25):
                    self.logger.info("Шаг 68: Изображение long_song_6.png найдено, выполняем клик по (101, 284) через 1.5 секунды")
                    time.sleep(1.5)
                    self.click_coord(101, 284)
                    self.logger.info("Шаг 68: ВЫПОЛНЕН - новый квест активирован (через изображение)")
                else:
                    self.logger.warning("Шаг 68: Изображение long_song_6.png не найдено, выполняем клик по координатам через 1.5 секунды")
                    time.sleep(1.5)
                    self.click_coord(101, 284)
                    self.logger.info("Шаг 68: ВЫПОЛНЕН - клик на квест выполнен (по координатам)")

            # Шаг 70: Нажимаем на иконку молоточка
            if start_step <= 70:
                self.logger.info("Шаг 70: Планируем открыть меню строительства - клик по (43, 481) через 2.5 секунды")
                time.sleep(2.5)
                self.logger.info("Шаг 70: Выполняем клик по координатам (43, 481)")
                self.click_coord(43, 481)
                self.logger.info("Шаг 70: ВЫПОЛНЕН - меню строительства открыто")

            # Шаг 71: Ожидаем изображение cannon_long.png и выбираем каюту гребцов
            if start_step <= 71:
                self.logger.info('Шаг 71: Планируем выбрать каюту гребцов - ожидаем cannon_long.png и нажимаем (968, 436)')
                if self.wait_for_image('cannon_long', timeout=15):
                    self.logger.info("Шаг 71: Изображение cannon_long.png найдено, выполняем клик по (968, 436) через 2.5 секунды")
                    time.sleep(2.5)
                    self.click_coord(968, 436)
                    self.logger.info("Шаг 71: ВЫПОЛНЕН - каюта гребцов выбрана (через изображение)")
                else:
                    self.logger.warning("Шаг 71: Изображение cannon_long.png не найдено, выполняем клик по координатам через 2.5 секунды")
                    time.sleep(2.5)
                    self.click_coord(968, 436)
                    self.logger.info("Шаг 71: ВЫПОЛНЕН - клик для выбора каюты гребцов выполнен (по координатам)")

            # Шаг 72: Ожидаем изображение long_song_6.png и подтверждаем постройку
            if start_step <= 72:
                self.logger.info('Шаг 72: Планируем подтвердить постройку каюты гребцов - ожидаем long_song_6.png и нажимаем (676, 580)')
                if self.wait_for_image('long_song_6', timeout=15):
                    self.logger.info("Шаг 72: Изображение long_song_6.png найдено, выполняем клик по (676, 580) через 2.5 секунды")
                    time.sleep(2.5)
                    self.click_coord(676, 580)
                    self.logger.info("Шаг 72: ВЫПОЛНЕН - постройка каюты гребцов подтверждена (через изображение)")
                else:
                    self.logger.warning("Шаг 72: Изображение long_song_6.png не найдено, выполняем клик по координатам через 2.5 секунды")
                    time.sleep(2.5)
                    self.click_coord(676, 580)
                    self.logger.info("Шаг 72: ВЫПОЛНЕН - клик для подтверждения постройки выполнен (по координатам)")

            # Шаг 74: Нажимаем на квест "Заполучи кают гребцов: 1"
            if start_step <= 74:
                self.logger.info('Шаг 74: Планируем активировать квест "Заполучи кают гребцов: 1" - клик по (123, 280) через 5 секунд')
                time.sleep(5)
                self.logger.info("Шаг 74: Выполняем клик по координатам (123, 280)")
                self.click_coord(123, 280)
                self.logger.info('Шаг 74: ВЫПОЛНЕН - квест "Заполучи кают гребцов: 1" активирован')

            # Шаг 75: Нажимаем на квест "Заполучи орудийных палуб: 1"
            if start_step <= 75:
                self.logger.info('Шаг 75: Планируем активировать квест "Заполучи орудийных палуб: 1" - клик по (123, 280) через 3.5 секунды')
                time.sleep(3.5)
                self.logger.info("Шаг 75: Выполняем клик по координатам (123, 280)")
                self.click_coord(123, 280)
                self.logger.info('Шаг 75: ВЫПОЛНЕН - квест "Заполучи орудийных палуб: 1" активирован')

            # Шаг 76: Нажимаем на иконку молоточка
            if start_step <= 76:
                self.logger.info('Шаг 76: Планируем открыть меню строительства - клик по (43, 481) через 2.5 секунды')
                time.sleep(2.5)
                self.logger.info("Шаг 76: Выполняем клик по координатам (43, 481)")
                self.click_coord(43, 481)
                self.logger.info("Шаг 76: ВЫПОЛНЕН - меню строительства открыто")

            # Шаг 77: Ожидаем изображение cannon_long.png и выбираем орудийную палубу
            if start_step <= 77:
                self.logger.info('Шаг 77: Планируем выбрать орудийную палубу - ожидаем cannon_long.png и нажимаем (687, 514)')
                if self.wait_for_image('cannon_long', timeout=15):
                    self.logger.info("Шаг 77: Изображение cannon_long.png найдено, выполняем клик по (687, 514)")
                    self.click_coord(687, 514)
                    self.logger.info("Шаг 77: ВЫПОЛНЕН - орудийная палуба выбрана (через изображение)")
                    time.sleep(2.5)
                else:
                    self.logger.warning("Шаг 77: Изображение cannon_long.png не найдено, выполняем клик по координатам")
                    self.click_coord(687, 514)
                    self.logger.info("Шаг 77: ВЫПОЛНЕН - клик для выбора орудийной палубы выполнен (по координатам)")
                    time.sleep(2.5)

            # Шаг 78: Ожидаем изображение long_song_6.png и подтверждаем постройку
            if start_step <= 78:
                self.logger.info('Шаг 78: Планируем подтвердить постройку орудийной палубы - ожидаем long_song_6.png и нажимаем (679, 581)')
                if self.wait_for_image('long_song_6', timeout=15):
                    self.logger.info("Шаг 78: Изображение long_song_6.png найдено, выполняем клик по (679, 581)")
                    self.click_coord(679, 581)
                    self.logger.info("Шаг 78: ВЫПОЛНЕН - постройка орудийной палубы подтверждена (через изображение)")
                    time.sleep(2.5)
                else:
                    self.logger.warning("Шаг 78: Изображение long_song_6.png не найдено, выполняем клик по координатам")
                    self.click_coord(679, 581)
                    self.logger.info("Шаг 78: ВЫПОЛНЕН - клик для подтверждения постройки выполнен (по координатам)")
                    time.sleep(2.5)

            # Шаг 80: Нажимаем на квест "Заполучи орудийных палуб: 1"
            if start_step <= 80:
                self.logger.info('Шаг 80: Планируем завершить квест орудийных палуб - клик по (134, 280) через 3 секунды')
                time.sleep(3)
                self.logger.info("Шаг 80: Выполняем клик по координатам (134, 280)")
                self.click_coord(134, 280)
                self.logger.info('Шаг 80: ВЫПОЛНЕН - квест орудийных палуб завершен')

            # Шаг 82: Жмем на иконку компаса
            if start_step <= 82:
                self.logger.info('Шаг 82: Планируем открыть компас - клик по (1072, 87) через 1.5 секунды')
                time.sleep(1.5)
                self.logger.info("Шаг 82: Выполняем клик по координатам (1072, 87)")
                self.click_coord(1072, 87)
                self.logger.info("Шаг 82: ВЫПОЛНЕН - компас открыт")

            # Шаг 83: Жмем на указатель на экране
            if start_step <= 83:
                self.logger.info('Шаг 83: Планируем активировать указатель - клик по (698, 273) через 2.5 секунды')
                time.sleep(2.5)
                self.logger.info("Шаг 83: Выполняем клик по координатам (698, 273)")
                self.click_coord(698, 273)
                self.logger.info("Шаг 83: ВЫПОЛНЕН - указатель активирован")

            # Шаг 84: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 84:
                self.logger.info("Шаг 84: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 84: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 86: Ожидаем изображение ship_song.png и нажимаем на иконку компаса над кораблем
            if start_step <= 86:
                self.logger.info('Шаг 86: Планируем активировать компас над кораблем - ожидаем ship_song.png и нажимаем (652, 214)')
                if self.wait_for_image('ship_song', timeout=5):
                    self.logger.info("Шаг 86: Изображение ship_song.png найдено, выполняем клик по (652, 214)")
                    self.click_coord(652, 214)
                    self.logger.info("Шаг 86: ВЫПОЛНЕН - компас над кораблем активирован (через изображение)")
                else:
                    self.logger.warning("Шаг 86: Изображение ship_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(652, 214)
                    self.logger.info("Шаг 86: ВЫПОЛНЕН - клик на компас над кораблем выполнен (по координатам)")

            # Шаги 87-89: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(87, 90):
                if start_step <= step:
                    self.logger.info(f"Шаг {step}: Планируем найти и нажать ПРОПУСТИТЬ")
                    self.find_skip_button_infinite()
                    self.logger.info(f"Шаг {step}: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")
                    # Добавляем задержку 1 секунду для шагов с 87 по 89 включительно
                    if 87 <= step <= 89:
                        time.sleep(1)

            # Шаг 90: Нажимаем на квест "Богатая добыча"
            if start_step <= 90:
                self.logger.info('Шаг 90: Планируем активировать квест "Богатая добыча" - клик по (89, 280) через 2.5 секунды')
                time.sleep(2.5)
                self.logger.info("Шаг 90: Выполняем клик по координатам (89, 280)")
                self.click_coord(89, 280)
                self.logger.info('Шаг 90: ВЫПОЛНЕН - квест "Богатая добыча" активирован')

            # Шаг 91: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 91:
                self.logger.info("Шаг 91: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 91: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 92: Ищем изображение griffin.png и нажимаем на координаты
            if start_step <= 92:
                self.logger.info("Шаг 92: Планируем закрыть диалог с грифоном - ищем griffin.png и нажимаем (1142, 42)")
                if self.wait_for_image('griffin', timeout=30):
                    self.logger.info("Шаг 92: Изображение griffin.png найдено, выполняем клик по (1142, 42)")
                    self.click_coord(1142, 42)
                    self.logger.info("Шаг 92: ВЫПОЛНЕН - диалог с грифоном закрыт (через изображение)")
                else:
                    self.logger.warning("Шаг 92: Изображение griffin.png не найдено, выполняем клик по координатам")
                    self.click_coord(1142, 42)
                    self.logger.info("Шаг 92: ВЫПОЛНЕН - клик для закрытия диалога выполнен (по координатам)")

            # Шаг 93: Ждем 7 секунд и ищем изображение molly.png
            if start_step <= 93:
                self.logger.info("Шаг 93: Планируем дождаться Молли - ждем 7 секунд, затем ищем molly.png и нажимаем (1142, 42)")
                time.sleep(7)
                if self.wait_for_image('molly', timeout=40):
                    self.logger.info("Шаг 93: Изображение molly.png найдено, выполняем клик по (1142, 42)")
                    self.click_coord(1142, 42)
                    self.logger.info("Шаг 93: ВЫПОЛНЕН - диалог с Молли закрыт (через изображение)")
                    time.sleep(3)
                else:
                    self.logger.warning("Шаг 93: Изображение molly.png не найдено, выполняем клик по координатам")
                    self.click_coord(1142, 42)
                    self.logger.info("Шаг 93: ВЫПОЛНЕН - клик для закрытия диалога выполнен (по координатам)")
                    time.sleep(3)

            # Шаг 94: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 94:
                self.logger.info("Шаг 94: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 94: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 95: Ищем картинку coins.png и нажимаем на координаты
            if start_step <= 95:
                self.logger.info("Шаг 95: Планируем собрать монеты - ищем coins.png и нажимаем (931, 620)")
                if self.click_image("coins", timeout=25):
                    self.logger.info("Шаг 95: ВЫПОЛНЕН - картинка coins.png найдена и нажата")
                else:
                    self.logger.warning("Шаг 95: Картинка coins.png не найдена, выполняем клик по координатам")
                    self.click_coord(931, 620)
                    self.logger.info("Шаг 95: ВЫПОЛНЕН - клик для сбора монет выполнен (по координатам)")

            # Шаг 96: Ищем слово ПРОПУСТИТЬ бесконечно
            if start_step <= 96:
                self.logger.info("Шаг 96: Планируем найти и нажать ПРОПУСТИТЬ")
                self.find_skip_button_infinite()
                self.logger.info("Шаг 96: ВЫПОЛНЕН - кнопка ПРОПУСТИТЬ найдена и нажата")

            # Шаг 97: Проверяем наличие баннера ПРОПУСТИТЬ, если нет - нажимаем на квест
            if start_step <= 97:
                self.logger.info('Шаг 97: Планируем завершить обучение - ждем 6 секунд, проверяем ПРОПУСТИТЬ, затем активируем квест')
                time.sleep(6)

                # Проверяем наличие кнопки ПРОПУСТИТЬ с небольшим таймаутом
                if self.find_skip_button_with_timeout(timeout=5):
                    self.logger.info('Шаг 97: ПРОПУСТИТЬ найден и нажат, ждем 4 секунды')
                    time.sleep(4)
                    self.logger.info('Шаг 97: Активируем финальный квест "На волосок от смерти"')
                    self.click_coord(89, 280)
                    self.logger.info('Шаг 97: ВЫПОЛНЕН - финальный квест активирован (после ПРОПУСТИТЬ)')
                else:
                    self.logger.info('Шаг 97: ПРОПУСТИТЬ не найден, сразу активируем финальный квест')
                    self.click_coord(89, 280)
                    self.logger.info('Шаг 97: ВЫПОЛНЕН - финальный квест активирован (без ПРОПУСТИТЬ)')

            self.logger.info("ВСЕ ШАГИ ОБУЧЕНИЯ УСПЕШНО ВЫПОЛНЕНЫ!")
            return True

        except Exception as e:
            self.logger.error(f"КРИТИЧЕСКАЯ ОШИБКА при выполнении шагов обучения: {e}", exc_info=True)
            return False

    # Основной метод запуска
    def run_bot(self, cycles: int = 1, start_server: int = 619,
               end_server: int = 1, first_server_start_step: int = 1) -> int:
        """
        Запуск бота на выполнение циклов обучения.

        Args:
            cycles: количество циклов
            start_server: начальный сервер
            end_server: конечный сервер
            first_server_start_step: начальный шаг для первого сервера

        Returns:
            int: количество успешно выполненных циклов
        """
        self.logger.info(f"Запуск бота: {cycles} циклов, сервера {start_server}-{end_server}")

        if start_server < end_server:
            self.logger.error("Начальный сервер должен быть больше конечного")
            return 0

        successful_cycles = 0
        current_server = start_server
        servers_to_process = min(cycles, start_server - end_server + 1)

        for cycle in range(1, servers_to_process + 1):
            self.logger.info(f"=== Цикл {cycle}/{servers_to_process}, сервер {current_server} ===")

            try:
                current_step = first_server_start_step if cycle == 1 else 1

                if self.perform_tutorial(current_server, start_step=current_step):
                    successful_cycles += 1
                    self.logger.info(f"Цикл {cycle} завершен успешно")
                else:
                    self.logger.error(f"Ошибка в цикле {cycle}")

                # Переход к следующему серверу
                if cycle < servers_to_process:
                    current_server -= 1
                    if current_server < end_server:
                        break
                    time.sleep(DEFAULT_TIMEOUT * 4)

            except Exception as e:
                self.logger.error(f"Критическая ошибка в цикле {cycle}: {e}", exc_info=True)

        self.logger.info(f"Завершено {successful_cycles}/{servers_to_process} циклов")
        return successful_cycles