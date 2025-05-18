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

        # Получаем текущие видимые сервера
        current_servers = self.server_selector.get_servers_with_coordinates()
        if not current_servers:
            self.logger.warning("Не удалось получить сервера, используем резервный метод")
            return self._fallback_scroll_search(server_id)

        current_servers_list = list(current_servers.keys())

        # Быстрая проверка наличия сервера
        coords = self.server_selector.find_server_coordinates(server_id)
        if coords:
            self._click_server_at_coordinates(coords)
            return True

        # Основной цикл скроллинга
        max_attempts = 6  # Уменьшено с 10 до 6

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

            # Получаем новый список серверов после скроллинга
            new_servers = self.server_selector.get_servers_with_coordinates()
            if new_servers:
                current_servers_list = list(new_servers.keys())

                # Проверяем, нашли ли целевой сервер
                coords = self.server_selector.find_server_coordinates(server_id)
                if coords:
                    self.logger.info(f"Найден сервер {server_id} после скроллинга!")
                    self._click_server_at_coordinates(coords)
                    return True

            # Если нет прогресса несколько раз подряд, меняем стратегию
            if attempt >= 3 and not self._check_scroll_progress(current_servers, new_servers):
                self.logger.info("Меняем стратегию скроллинга")
                self._perform_opposite_scroll(server_id, current_servers_list)

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

        # Последняя попытка найти точный сервер
        coords = self.server_selector.find_server_coordinates(server_id, attempts=2)
        if coords:
            self.logger.info(f"Найден сервер {server_id} в финальном поиске!")
            self._click_server_at_coordinates(coords)
            return True

        # Ищем ближайший сервер
        if current_servers:
            closest = min(current_servers, key=lambda s: abs(s - server_id))
            difference = abs(closest - server_id)

            # Для финального поиска разрешаем большую разность
            if difference <= 5:  # Уменьшено с 8 до 5
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

            for _ in range(3):  # 3 попытки в каждом направлении
                self.adb.swipe(*start_coords, *end_coords,
                               duration=SERVER_RECOGNITION_SETTINGS['scroll_duration'])
                time.sleep(PAUSE_SETTINGS['after_server_scroll'])

                # Проверяем, появился ли нужный сервер
                coords = self.server_selector.find_server_coordinates(server_id)
                if coords:
                    self._click_server_at_coordinates(coords)
                    return True

        return False

    def _click_server_at_coordinates(self, coords: Tuple[int, int]):
        """Клик по серверу с паузами."""
        time.sleep(PAUSE_SETTINGS['before_server_click'])
        self.click_coord(coords[0], coords[1])
        time.sleep(PAUSE_SETTINGS['after_server_click'])

    # Специализированные методы для обучения
    def find_skip_button(self, max_attempts: Optional[int] = None) -> bool:
        """Поиск и клик по кнопке ПРОПУСТИТЬ."""
        self.logger.info("Поиск кнопки ПРОПУСТИТЬ")

        skip_variants = [
            "ПРОПУСТИТЬ", "ПРОПУСТИTЬ", "ПРОNYСТИТЬ", "ПPОПУСТИТЬ",
            "ПРОПУCTИТЬ", "ПРOПУСТИТЬ", "ПPOПУСТИТЬ", "SKIP"
        ]

        region = OCR_REGIONS['skip_button']

        for variant in skip_variants:
            if self.find_and_click_text(variant, region, timeout=1):
                return True

        # Если не найден, кликаем по координатам
        skip_coords = COORDINATES['skip_button']
        self.click_coord(skip_coords[0], skip_coords[1])
        return True

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

        Args:
            start_step: начальный шаг для выполнения

        Returns:
            bool: True если шаги выполнены успешно, False иначе
        """
        self.logger.info(f"Выполнение оставшихся шагов обучения, начиная с шага {start_step}")

        try:
            # Шаг 7: Ищем слово ПРОПУСТИТЬ и кликаем на него (без цветового анализа)
            if start_step <= 7:
                self.logger.info("Шаг 7: Ищем и нажимаем ПРОПУСТИТЬ (только OCR)")
                self.find_skip_button()
                time.sleep(1)

            # Шаг 8: Ищем слово ПРОПУСТИТЬ и кликаем на него (без цветового анализа)
            if start_step <= 8:
                self.logger.info("Шаг 8: Ищем и нажимаем ПРОПУСТИТЬ (только OCR)")
                self.find_skip_button()
                time.sleep(1)

            # Шаг 9: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 9:
                self.logger.info("Шаг 9: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(1)

            # Шаг 10: Активируем бой нажатием на пушку
            if start_step <= 10:
                self.logger.info("Шаг 10: Ищем изображение cannon_is_ready.png и нажимаем на координаты (718, 438)")
                if self.wait_for_image('cannon_is_ready', timeout=20):
                    self.click_coord(718, 438)
                else:
                    self.logger.warning("Изображение cannon_is_ready.png не найдено, выполняем клик по координатам")
                    self.click_coord(718, 438)

            # Шаг 11: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 11:
                self.logger.info("Шаг 11: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 12: Ждем появления hell_henry.png и нажимаем ПРОПУСТИТЬ
            if start_step <= 12:
                self.logger.info('Шаг 12: Ждем появления изображения hell_henry.png и нажимаем ПРОПУСТИТЬ')
                if self.wait_for_image('hell_henry', timeout=15):
                    self.find_skip_button()
                else:
                    self.logger.warning("Изображение hell_henry.png не найдено, продолжаем выполнение")

            # Шаг 13: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 13:
                self.logger.info("Шаг 13: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 14: Жмем на иконку кораблика
            if start_step <= 14:
                self.logger.info("Шаг 14: Ждем 3 секунды и нажимаем (58, 654) - жмем на иконку кораблика")
                time.sleep(3)
                self.click_coord(58, 654)

            # Шаг 15: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 15:
                self.logger.info("Шаг 15: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 16: Отстраиваем нижнюю палубу
            if start_step <= 16:
                self.logger.info("Шаг 16: Ждем 2 секунды и нажимаем (638, 403) - отстраиваем нижнюю палубу")
                time.sleep(2)
                self.click_coord(638, 403)

            # Шаг 17: Отстраиваем паб в нижней палубе
            if start_step <= 17:
                self.logger.info("Шаг 17: Ждем 2.5 секунды и нажимаем (635, 373) - отстраиваем паб в нижней палубе")
                time.sleep(2.5)
                self.click_coord(635, 373)

            # Шаг 18: Латаем дыры в складе на нижней палубе
            if start_step <= 18:
                self.logger.info("Шаг 18: Ждем 2.5 секунды и нажимаем (635, 373) - латаем дыры в складе на нижней палубе")
                time.sleep(2.5)
                self.click_coord(635, 373)

            # Шаг 19: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 19:
                self.logger.info("Шаг 19: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 20: Отстраиваем верхнюю палубу
            if start_step <= 20:
                self.logger.info("Шаг 20: Ждем 2.5 секунды и нажимаем (345, 386) - отстраиваем верхнюю палубу")
                time.sleep(2.5)
                self.click_coord(345, 386)

            # Шаг 21: Выбираем пушку
            if start_step <= 21:
                self.logger.info("Шаг 21: Ждем 1.5 секунды и нажимаем (77, 276) - выбираем пушку")
                time.sleep(1.5)
                self.click_coord(77, 276)

            # Шаг 22: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 22:
                self.logger.info("Шаг 22: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 23: Ищем collect_items.png и нажимаем на координаты
            if start_step <= 23:
                self.logger.info('Шаг 23: Ищем изображение collect_items.png и нажимаем на координаты (741, 145)')
                if self.wait_for_image('collect_items', timeout=15):
                    self.click_coord(741, 145)
                else:
                    self.logger.warning("Изображение collect_items.png не найдено, выполняем клик по координатам")
                    self.click_coord(741, 145)

            # Шаг 24: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 24:
                self.logger.info("Шаг 24: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(0.5)

            # Шаг 25: Нажимаем на квест "Старый соперник"
            if start_step <= 25:
                self.logger.info('Шаг 25: Ждем 1.5 секунды и нажимаем по координатам (93, 285) - квест "Старый соперник"')
                time.sleep(1.5)
                self.click_coord(93, 285)

            # Шаг 26: Ищем слово ПРОПУСТИТЬ и кликаем на него (с задержкой 8 сек)
            if start_step <= 26:
                self.logger.info("Шаг 26: Ждем 8 секунд и нажимаем по координатам (1142, 42)")
                time.sleep(8)
                self.click_coord(1142, 42)

            # Шаг 27: Нажимаем на квест "Старый соперник"
            if start_step <= 27:
                self.logger.info('Шаг 27: Ждем 1.5 секунды и нажимаем на координаты (93, 285) - квест "Старый соперник"')
                time.sleep(1.5)
                self.click_coord(93, 285)

            # Шаг 28: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 28:
                self.logger.info("Шаг 28: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(1)

            # Шаг 29: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 29:
                self.logger.info("Шаг 29: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(3)

            # Шаг 30: Жмем на любую часть экрана чтобы продолжить после победы
            if start_step <= 30:
                self.logger.info("Шаг 30: Клик по координатам (630, 413) - жмем на любую часть экрана")
                self.click_coord(630, 413)

            # Шаг 31: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 31:
                self.logger.info("Шаг 31: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(0.5)

            # Шаг 32: Ждем появления gold_compas.png и жмем на компас
            if start_step <= 32:
                self.logger.info("Шаг 32: Ожидаем изображение gold_compas.png и нажимаем (1074, 88) - жмем на компас")
                if self.wait_for_image('gold_compas', timeout=15):
                    time.sleep(0.5)
                    self.click_coord(1074, 88)
                else:
                    self.logger.warning("Изображение gold_compas.png не найдено, выполняем клик по координатам")
                    time.sleep(0.5)
                    self.click_coord(1074, 88)

            # Шаг 33: Еще раз жмем на компас
            if start_step <= 33:
                self.logger.info("Шаг 33: Клик по координатам (701, 258) - еще раз жмем на компас")
                time.sleep(1.5)
                self.click_coord(701, 258)

            # Шаг 34: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 34:
                self.logger.info("Шаг 34: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 35: Жмем назад чтобы выйти из вкладки компаса
            if start_step <= 35:
                self.logger.info("Шаг 35: Клик по координатам (145, 25) - жмем назад")
                self.click_coord(145, 25)

            # Шаг 36: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 36:
                self.logger.info("Шаг 36: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 37: Ожидаем изображение long_song.png и нажимаем на квест "Далекая песня"
            if start_step <= 37:
                self.logger.info('Шаг 37: Ожидаем изображение long_song.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 38: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 38:
                self.logger.info("Шаг 38: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 39: Ожидаем изображение long_song.png и еще раз нажимаем на квест "Далекая песня"
            if start_step <= 39:
                self.logger.info('Шаг 39: Ожидаем изображение long_song.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 40: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 40:
                self.logger.info("Шаг 40: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 41: Ожидаем изображение confirm_trade.png и нажимаем "Согласиться на обмен"
            if start_step <= 41:
                self.logger.info('Шаг 41: Ожидаем изображение confirm_trade.png и нажимаем на координаты (151, 349)')
                if self.wait_for_image('confirm_trade', timeout=15):
                    self.click_coord(151, 349)
                else:
                    self.logger.warning("Изображение confirm_trade.png не найдено, выполняем клик по координатам")
                    self.click_coord(151, 349)

            # Шаг 42: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 42:
                self.logger.info("Шаг 42: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 43: Ожидаем изображение long_song.png и нажимаем на квест "Исследовать залив Мертвецов"
            if start_step <= 43:
                self.logger.info('Шаг 43: Ожидаем изображение long_song.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 44: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 44:
                self.logger.info("Шаг 44: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 45: Ожидаем изображение prepare_for_battle.png и нажимаем на координаты (85, 634)
            if start_step <= 45:
                self.logger.info('Шаг 45: Ожидаем изображение prepare_for_battle.png и нажимаем на координаты (85, 634)')
                if self.wait_for_image('prepare_for_battle', timeout=15):
                    self.click_coord(85, 634)
                else:
                    self.logger.warning("Изображение prepare_for_battle.png не найдено, выполняем клик по координатам")
                    self.click_coord(85, 634)

            # Шаг 46: Нажимаем начать битву
            if start_step <= 46:
                self.logger.info("Шаг 46: Ждем 0.5 секунды и нажимаем на координаты (1157, 604)")
                time.sleep(0.5)
                self.click_coord(1157, 604)
                time.sleep(3)

            # Шаг 47: Нажимаем каждые 1.5 секунды по координатам, пока не появится start_battle.png
            if start_step <= 47:
                self.logger.info("Шаг 47: Нажимаем каждые 1.5 секунды, пока не найдем start_battle.png")
                found = False
                for _ in range(20):  # Максимум 20 попыток
                    if self.click_image("start_battle", timeout=1):
                        found = True
                        break
                    self.click_coord(642, 334)
                    time.sleep(1.5)

                if not found:
                    self.logger.warning("Не удалось найти изображение start_battle.png за 20 попыток")

            # Шаг 48: Нажимаем каждые 1.5 секунды, пока не найдем изображение ship_waiting_zaliz.png
            if start_step <= 48:
                self.logger.info("Шаг 48: Нажимаем каждые 1.5 секунды, пока не найдем ship_waiting_zaliz.png")
                found = False
                for _ in range(20):  # Максимум 20 попыток
                    if self.wait_for_image('ship_waiting_zaliz', timeout=1):
                        found = True
                        self.click_coord(93, 285)
                        break
                    self.click_coord(642, 334)
                    time.sleep(1.5)

                if not found:
                    self.logger.warning("Не удалось найти изображение ship_waiting_zaliz.png за 20 попыток")
                    self.click_coord(93, 285)

            # Шаг 49: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 49:
                self.logger.info("Шаг 49: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 50: Ожидаем изображение long_song_2.png и нажимаем на квест
            if start_step <= 50:
                self.logger.info('Шаг 50: Ожидаем изображение long_song_2.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song_2', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song_2.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 51: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 51:
                self.logger.info("Шаг 51: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 52: Нажимаем на череп в заливе мертвецов
            if start_step <= 52:
                self.logger.info("Шаг 52: Ждем 2 секунды и нажимаем на координаты (653, 403) - череп")
                time.sleep(2)
                self.click_coord(653, 403)

            # Шаги 53-57: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(53, 58):
                if start_step <= step:
                    self.logger.info(f"Шаг {step}: Ищем и нажимаем ПРОПУСТИТЬ")
                    self.find_skip_button()
                    # Добавляем задержку 0.5 секунд для шагов с 53 по 56 включительно
                    if 53 <= step <= 56:
                        time.sleep(0.5)  # Задержка для избежания двойного нажатия между анимациями

            # Шаг 58: Ожидаем изображение long_song_3.png и нажимаем на квест
            if start_step <= 58:
                self.logger.info('Шаг 58: Ожидаем изображение long_song_3.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song_3', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song_3.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 59: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 59:
                self.logger.info("Шаг 59: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 60: Ожидаем изображение long_song.png и нажимаем на квест
            if start_step <= 60:
                self.logger.info('Шаг 60: Ожидаем изображение long_song.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 61: Ожидаем изображение long_song_4.png и нажимаем на иконку молоточка
            if start_step <= 61:
                self.logger.info('Шаг 61: Ожидаем изображение long_song_4.png и нажимаем на координаты (43, 481)')
                if self.wait_for_image('long_song_4', timeout=15):
                    self.click_coord(43, 481)
                else:
                    self.logger.warning("Изображение long_song_4.png не найдено, выполняем клик по координатам")
                    self.click_coord(43, 481)

            # Шаг 62: Ожидаем изображение long_song_5.png и выбираем корабль для улучшения
            if start_step <= 62:
                self.logger.info('Шаг 62: Ожидаем изображение long_song_5.png и нажимаем на координаты (127, 216)')
                if self.wait_for_image('long_song_5', timeout=15):
                    self.click_coord(127, 216)
                else:
                    self.logger.warning("Изображение long_song_5.png не найдено, выполняем клик по координатам")
                    self.click_coord(127, 216)

            # Шаг 63: Дожидаемся надписи "УЛУЧШИТЬ" и кликаем на нее
            if start_step <= 63:
                self.logger.info('Шаг 63: Дожидаемся надписи "УЛУЧШИТЬ" и кликаем на нее')
                if not self.find_and_click_text("УЛУЧШИТЬ", region=(983, 588, 200, 100), timeout=5):
                    self.click_coord(1083, 638)  # Клик по примерным координатам, если текст не найден

            # Шаг 64: Жмем назад чтобы выйти из вкладки корабля
            if start_step <= 64:
                self.logger.info("Шаг 64: Ждем 2 секунды и нажимаем координаты (145, 25) - назад")
                time.sleep(3)
                self.click_coord(145, 25)

            # Шаг 65: Жмем на кнопку постройки
            if start_step <= 65:
                self.logger.info("Шаг 65: Ждем 1.5 секунды и нажимаем координаты (639, 603) - кнопка постройки")
                time.sleep(1.5)
                self.click_coord(639, 603)

            # Шаг 66: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 66:
                self.logger.info("Шаг 66: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 67: Жмем на иконку компаса
            if start_step <= 67:
                self.logger.info("Шаг 67: Ждем 1.5 секунды и нажимаем координаты (1072, 87) - иконка компаса")
                time.sleep(1.5)
                self.click_coord(1072, 87)
                time.sleep(5)

            # Шаг 68: Ожидаем изображение long_song_6.png и нажимаем на квест
            if start_step <= 68:
                self.logger.info('Шаг 68: Ожидаем изображение long_song_6.png и нажимаем на координаты (89, 280)')
                if self.wait_for_image('long_song_6', timeout=25):
                    time.sleep(1.5)
                    self.click_coord(101, 284)
                else:
                    self.logger.warning("Изображение long_song_6.png не найдено, выполняем клик по координатам")
                    time.sleep(1.5)
                    self.click_coord(101, 284)

            # Шаг 69: УДАЛЕН согласно ТЗ

            # Шаг 70: Нажимаем на иконку молоточка
            if start_step <= 70:
                self.logger.info("Шаг 70: Ждем 2.5 секунды и нажимаем координаты (43, 481) - иконка молоточка")
                time.sleep(2.5)
                self.click_coord(43, 481)

            # Шаг 71: Ожидаем изображение cannon_long.png и выбираем каюту гребцов
            if start_step <= 71:
                self.logger.info('Шаг 71: Ожидаем изображение cannon_long.png и нажимаем на координаты (968, 507)')
                if self.wait_for_image('cannon_long', timeout=15):
                    time.sleep(2.5)
                    self.click_coord(968, 436)
                else:
                    self.logger.warning("Изображение cannon_long.png не найдено, выполняем клик по координатам")
                    time.sleep(2.5)
                    self.click_coord(968, 436)

            # Шаг 72: Ожидаем изображение long_song_6.png и подтверждаем постройку
            if start_step <= 72:
                self.logger.info('Шаг 72: Ожидаем изображение long_song_6.png и нажимаем на координаты (676, 580)')
                if self.wait_for_image('long_song_6', timeout=15):
                    time.sleep(2.5)
                    self.click_coord(676, 580)
                else:
                    self.logger.warning("Изображение long_song_6.png не найдено, выполняем клик по координатам")
                    time.sleep(2.5)
                    self.click_coord(676, 580)

            # Шаг 73: УДАЛЕН согласно ТЗ

            # Шаг 74: Нажимаем на квест "Заполучи кают гребцов: 1"
            if start_step <= 74:
                self.logger.info('Шаг 74: Ждем 5 секунд и нажимаем на координаты (89, 280)')
                time.sleep(5)
                self.click_coord(123, 280)

            # Шаг 75: Нажимаем на квест "Заполучи орудийных палуб: 1"
            if start_step <= 75:
                self.logger.info('Шаг 75: Ждем 2.5 секунды и нажимаем на координаты (89, 280)')
                time.sleep(3.5)
                self.click_coord(123, 280)

            # Шаг 76: Нажимаем на иконку молоточка
            if start_step <= 76:
                self.logger.info('Шаг 76: Ждем 2.5 секунды и нажимаем на координаты (43, 481)')
                time.sleep(2.5)
                self.click_coord(43, 481)

            # Шаг 77: Ожидаем изображение cannon_long.png и выбираем орудийную палубу
            if start_step <= 77:
                self.logger.info('Шаг 77: Ожидаем изображение cannon_long.png и нажимаем на координаты (687, 514)')
                if self.wait_for_image('cannon_long', timeout=15):
                    self.click_coord(687, 514)
                    time.sleep(2.5)
                else:
                    self.logger.warning("Изображение cannon_long.png не найдено, выполняем клик по координатам")
                    self.click_coord(687, 514)
                    time.sleep(2.5)

            # Шаг 78: Ожидаем изображение long_song_6.png и подтверждаем постройку
            if start_step <= 78:
                self.logger.info('Шаг 78: Ожидаем изображение long_song_6.png и нажимаем на координаты (679, 581)')
                if self.wait_for_image('long_song_6', timeout=15):
                    self.click_coord(679, 581)
                    time.sleep(2.5)
                else:
                    self.logger.warning("Изображение long_song_6.png не найдено, выполняем клик по координатам")
                    self.click_coord(679, 581)
                    time.sleep(2.5)

            # Шаг 79: УДАЛЕН согласно ТЗ

            # Шаг 80: Нажимаем на квест "Заполучи орудийных палуб: 1"
            if start_step <= 80:
                self.logger.info('Шаг 80: Ждем 3 секунды и нажимаем на координаты (89, 280)')
                time.sleep(3)
                self.click_coord(134, 280)

            # Шаг 81: УДАЛЕН согласно ТЗ

            # Шаг 82: Жмем на иконку компаса
            if start_step <= 82:
                self.logger.info('Шаг 82: Ждем 1.5 секунды и нажимаем на координаты (1072, 87)')
                time.sleep(1.5)
                self.click_coord(1072, 87)

            # Шаг 83: Жмем на указатель на экране
            if start_step <= 83:
                self.logger.info('Шаг 83: Ждем 2.5 секунды и нажимаем на координаты (698, 273)')
                time.sleep(2.5)
                self.click_coord(698, 273)

            # Шаг 84: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 84:
                self.logger.info("Шаг 84: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 85: УДАЛЕН согласно ТЗ

            # Шаг 86: Ожидаем изображение ship_song.png и нажимаем на иконку компаса над кораблем
            if start_step <= 86:
                self.logger.info('Шаг 86: Ожидаем изображение ship_song.png и нажимаем на координаты (652, 214)')
                if self.wait_for_image('ship_song', timeout=5):
                    self.click_coord(652, 214)
                else:
                    self.logger.warning("Изображение ship_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(652, 214)

            # Шаги 87-89: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(87, 90):
                if start_step <= step:
                    self.logger.info(f"Шаг {step}: Ищем и нажимаем ПРОПУСТИТЬ")
                    self.find_skip_button()
                    # Добавляем задержку 1 секунду для шагов с 87 по 89 включительно
                    if 87 <= step <= 89:
                        time.sleep(1)  # Задержка для избежания двойного нажатия между анимациями

            # Шаг 90: Нажимаем на квест "Богатая добыча"
            if start_step <= 90:
                self.logger.info('Шаг 90: Ждем 2.5 секунды и нажимаем на координаты (89, 280)')
                time.sleep(2.5)
                self.click_coord(89, 280)

            # Шаг 91: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 91:
                self.logger.info("Шаг 91: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 92: Ищем изображение griffin.png и нажимаем на координаты (1142, 42)
            if start_step <= 92:
                self.logger.info("Шаг 92: Ищем изображение griffin.png и нажимаем на координаты (1142, 42)")
                if self.wait_for_image('griffin', timeout=30):
                    self.click_coord(1142, 42)
                else:
                    self.logger.warning("Изображение griffin.png не найдено, выполняем клик по координатам")
                    self.click_coord(1142, 42)

            # Шаг 93: Ждем 7 секунд и ищем изображение molly.png
            if start_step <= 93:
                self.logger.info("Шаг 93: Ждем 7 секунд, ищем изображение molly.png и нажимаем на координаты (1142, 42)")
                time.sleep(7)
                if self.wait_for_image('molly', timeout=40):
                    self.click_coord(1142, 42)
                    time.sleep(3)
                else:
                    self.logger.warning("Изображение molly.png не найдено, выполняем клик по координатам")
                    self.click_coord(1142, 42)
                    time.sleep(3)

            # Шаг 94: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 94:
                self.logger.info("Шаг 94: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 95: Ищем картинку coins.png и нажимаем на координаты
            if start_step <= 95:
                self.logger.info("Шаг 95: Ищем картинку coins.png и нажимаем на координаты (931, 620)")
                if self.click_image("coins", timeout=25):
                    self.logger.info("Картинка coins.png найдена и нажата")
                else:
                    self.logger.warning("Картинка coins.png не найдена, нажимаем по координатам")
                    self.click_coord(931, 620)

            # Шаг 96: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_step <= 96:
                self.logger.info("Шаг 96: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 97: Проверяем наличие баннера ПРОПУСТИТЬ, если нет - нажимаем на квест
            if start_step <= 97:
                self.logger.info('Шаг 97: Ждем 6 секунд, проверяем наличие ПРОПУСТИТЬ, затем нажимаем на квест')
                time.sleep(6)

                # Проверяем наличие кнопки ПРОПУСТИТЬ
                if self.find_skip_button():
                    time.sleep(4)
                    self.logger.info('Найден ПРОПУСТИТЬ, нажимаем на квест "На волосок от смерти"')
                    self.click_coord(89, 280)
                else:
                    self.logger.info('ПРОПУСТИТЬ не найден, нажимаем на квест "На волосок от смерти"')
                    self.click_coord(89, 280)

            self.logger.info("Все шаги обучения успешно выполнены")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка при выполнении оставшихся шагов обучения: {e}", exc_info=True)
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