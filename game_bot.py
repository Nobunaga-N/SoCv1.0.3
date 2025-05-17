"""
Основной модуль бота для прохождения обучения в игре Sea of Conquest.
"""
import time
import random
import logging
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Union

from config import (
    IMAGE_PATHS, COORDINATES, SEASONS, DEFAULT_TIMEOUT, LOADING_TIMEOUT,
    GAME_PACKAGE, GAME_ACTIVITY, TEMPLATE_MATCHING_THRESHOLD
)


class GameBot:
    """Основной класс бота для прохождения обучения в игре Sea of Conquest."""

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

    def _check_ocr_availability(self):
        """Проверка доступности OCR (Tesseract)."""
        try:
            import pytesseract
            self.logger.info("OCR (Tesseract) доступен для использования")
            return True
        except ImportError:
            self.logger.warning("OCR (Tesseract) не доступен, возможны проблемы с распознаванием текста")
            return False

    def start_game(self):
        """Запуск игры."""
        self.logger.info("Запуск игры")
        self.adb.start_app(GAME_PACKAGE, GAME_ACTIVITY)
        time.sleep(LOADING_TIMEOUT)

    def stop_game(self):
        """Остановка игры."""
        self.logger.info("Остановка игры")
        self.adb.stop_app(GAME_PACKAGE)

    def click_coord(self, x, y):
        """
        Клик по координатам.

        Args:
            x: координата x
            y: координата y
        """
        self.logger.debug(f"Клик по координатам ({x}, {y})")
        self.adb.tap(x, y)
        time.sleep(DEFAULT_TIMEOUT)

    def click_random(self, center_x, center_y, radius=50):
        """
        Клик по случайным координатам в заданной области.

        Args:
            center_x: центр области по x
            center_y: центр области по y
            radius: радиус области
        """
        self.logger.debug(f"Случайный клик в области ({center_x}±{radius}, {center_y}±{radius})")
        self.adb.tap_random(center_x, center_y, radius)
        time.sleep(DEFAULT_TIMEOUT)

    def click_image(self, image_key, timeout=30):
        """
        Найти изображение и кликнуть по нему.

        Args:
            image_key: ключ изображения в словаре IMAGE_PATHS
            timeout: максимальное время ожидания в секундах

        Returns:
            bool: True если изображение найдено и клик выполнен, False иначе
        """
        if image_key not in IMAGE_PATHS:
            self.logger.error(f"Изображение с ключом '{image_key}' не найдено в конфигурации")
            return False

        return self.image.tap_on_template(IMAGE_PATHS[image_key], timeout)

    def wait_for_image(self, image_key, timeout=30):
        """
        Ждать появления изображения на экране.

        Args:
            image_key: ключ изображения в словаре IMAGE_PATHS
            timeout: максимальное время ожидания в секундах

        Returns:
            tuple: (x, y, w, h) координаты и размеры найденного изображения или None
        """
        if image_key not in IMAGE_PATHS:
            self.logger.error(f"Изображение с ключом '{image_key}' не найдено в конфигурации")
            return None

        return self.image.wait_for_template(IMAGE_PATHS[image_key], timeout)

    def is_image_on_screen(self, image_key, timeout=1):
        """
        Проверить наличие изображения на экране.

        Args:
            image_key: ключ изображения в словаре IMAGE_PATHS
            timeout: максимальное время ожидания в секундах

        Returns:
            bool: True если изображение найдено, False иначе
        """
        if image_key not in IMAGE_PATHS:
            self.logger.error(f"Изображение с ключом '{image_key}' не найдено в конфигурации")
            return False

        return self.image.is_template_on_screen(IMAGE_PATHS[image_key], timeout)

    def find_text_on_screen(self, text, region=None, timeout=5):
        """
        Поиск текста на экране с использованием OCR.

        Args:
            text: искомый текст
            region: область поиска (x, y, w, h) или None для всего экрана
            timeout: максимальное время ожидания в секундах

        Returns:
            tuple: (x, y, w, h) координаты и размеры найденного текста или None
        """
        if not self.ocr_available:
            self.logger.warning("OCR не доступен, невозможно найти текст на экране")
            return None

        self.logger.debug(f"Поиск текста '{text}' на экране")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Получение скриншота
            screenshot = self.adb.screenshot()

            if screenshot is None:
                time.sleep(0.5)
                continue

            # Преобразование скриншота в оттенки серого
            gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # Если указана область, вырезаем ее
            if region:
                x, y, w, h = region
                roi = gray[y:y+h, x:x+w]
            else:
                roi = gray
                x, y = 0, 0

            # Бинаризация изображения для улучшения OCR
            _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)

            try:
                # Распознавание текста с помощью Tesseract
                import pytesseract
                result = pytesseract.image_to_string(binary, lang='rus+eng')

                # Поиск текста в результате
                if text.lower() in result.lower():
                    self.logger.info(f"Текст '{text}' найден на экране")

                    # Для простоты возвращаем центр области
                    if region:
                        return (region[0] + region[2]//2, region[1] + region[3]//2, region[2], region[3])
                    else:
                        h, w = screenshot.shape[:2]
                        return (w//2, h//2, w, h)
            except Exception as e:
                self.logger.error(f"Ошибка при распознавании текста: {e}")

            time.sleep(0.5)

        self.logger.warning(f"Текст '{text}' не найден на экране за {timeout} сек")
        return None

    def wait_for_text(self, text, region=None, timeout=30):
        """
        Ожидание появления текста на экране.

        Args:
            text: ожидаемый текст
            region: область поиска (x, y, w, h) или None для всего экрана
            timeout: максимальное время ожидания в секундах

        Returns:
            tuple: (x, y, w, h) координаты и размеры найденного текста или None
        """
        self.logger.info(f"Ожидание появления текста '{text}' на экране")
        return self.find_text_on_screen(text, region, timeout)

    def find_and_click_text(self, text, region=None, timeout=10):
        """
        Поиск текста на экране и клик по нему.

        Args:
            text: искомый текст
            region: область поиска (x, y, w, h) или None для всего экрана
            timeout: максимальное время ожидания в секундах

        Returns:
            bool: True если текст найден и клик выполнен, False иначе
        """
        self.logger.info(f"Поиск и клик по тексту '{text}' на экране")
        result = self.find_text_on_screen(text, region, timeout)

        if result:
            x, y, _, _ = result
            self.click_coord(x, y)
            return True

        return False

    def swipe(self, start_x, start_y, end_x, end_y, duration=1000):
        """
        Выполнение свайпа.

        Args:
            start_x: начальная координата x
            start_y: начальная координата y
            end_x: конечная координата x
            end_y: конечная координата y
            duration: продолжительность свайпа в мс
        """
        self.logger.debug(f"Свайп от ({start_x}, {start_y}) к ({end_x}, {end_y})")
        self.adb.swipe(start_x, start_y, end_x, end_y, duration)
        time.sleep(DEFAULT_TIMEOUT)

    def determine_season_for_server(self, server_id):
        """
        Определение сезона, в котором находится сервер.

        Args:
            server_id: номер сервера

        Returns:
            str: идентификатор сезона или None, если сезон не найден
        """
        for season_id, season_data in SEASONS.items():
            if season_data['min_server'] >= server_id >= season_data['max_server']:
                return season_id
        return None

    def select_season(self, season_id):
        """
        Выбор сезона.

        Args:
            season_id: идентификатор сезона (S1, S2, S3, S4, S5, X1, X2, X3, X4)

        Returns:
            bool: True если сезон выбран успешно, False иначе
        """
        self.logger.info(f"Выбор сезона: {season_id}")

        # Проверка существования сезона
        if season_id not in SEASONS:
            self.logger.error(f"Сезон '{season_id}' не найден в конфигурации")
            return False

        # Определение необходимости скроллинга
        if season_id in ['X2', 'X3', 'X4']:
            # Скроллинг для отображения нижних сезонов
            self.swipe(257, 573, 238, 240)
            time.sleep(1)  # Дополнительная задержка после скролла

        # Приблизительные координаты сезонов (эти значения могут потребовать корректировки)
        season_y_coords = {
            'S1': 180,
            'S2': 220,
            'S3': 260,
            'S4': 300,
            'S5': 340,
            'X1': 380,
            'X2': 260,  # После скроллинга
            'X3': 300,  # После скроллинга
            'X4': 340   # После скроллинга
        }

        # Клик по сезону
        if season_id in season_y_coords:
            # Клик по текстовой метке сезона
            self.click_coord(400, season_y_coords[season_id])
            time.sleep(1)  # Дополнительная задержка после выбора сезона
            return True

        return False

    def select_server(self, server_id):
        """
        Выбор сервера.

        Args:
            server_id: номер сервера

        Returns:
            bool: True если сервер выбран успешно, False иначе
        """
        self.logger.info(f"Выбор сервера: {server_id}")

        # Определение сезона для сервера
        season_id = self.determine_season_for_server(server_id)
        if not season_id:
            self.logger.error(f"Не удалось определить сезон для сервера {server_id}")
            return False

        # Выбор сезона
        if not self.select_season(season_id):
            return False

        # Получение диапазона серверов в сезоне
        min_server = SEASONS[season_id]['max_server']
        max_server = SEASONS[season_id]['min_server']

        # Порядковый номер сервера в сезоне (от новых к старым)
        server_index = max_server - server_id

        # Количество серверов на странице и расчет страницы
        servers_per_page = 10

        # Номер страницы (начиная с 0)
        page = server_index // servers_per_page

        # Индекс на странице (начиная с 0)
        page_index = server_index % servers_per_page

        self.logger.debug(f"Сервер {server_id}: сезон {season_id}, индекс {server_index}, страница {page}, позиция {page_index}")

        # Скроллинг до нужной страницы
        for _ in range(page):
            self.swipe(777, 566, 772, 100)
            time.sleep(DEFAULT_TIMEOUT)  # Дополнительная задержка после скролла

        # Клик по серверу (приблизительные координаты)
        server_base_y = 150  # Примерная Y-координата первого сервера
        server_step_y = 40   # Шаг между серверами по Y

        server_y = server_base_y + page_index * server_step_y
        server_x = 500  # Примерная X-координата сервера

        self.click_coord(server_x, server_y)
        time.sleep(DEFAULT_TIMEOUT)  # Дополнительная задержка после выбора сервера

        return True

    def is_server_available(self, server_id):
        """
        Проверка доступности сервера.

        Args:
            server_id: номер сервера

        Returns:
            bool: True если сервер доступен, False иначе
        """
        # В реальном сценарии здесь можно добавить проверку доступности сервера
        # Для примера считаем, что все серверы доступны
        return True

    def find_skip_button(self, max_attempts=5, timeout=2):
        """
        Поиск и клик по кнопке "ПРОПУСТИТЬ".

        Args:
            max_attempts: максимальное количество попыток
            timeout: таймаут между попытками

        Returns:
            bool: True если кнопка найдена и нажата, False иначе
        """
        self.logger.info("Поиск кнопки ПРОПУСТИТЬ")

        # Координаты области, где обычно находится кнопка ПРОПУСТИТЬ
        region = (1040, 12, 200, 60)

        for i in range(max_attempts):
            self.logger.debug(f"Попытка {i+1}/{max_attempts}")

            # Пробуем найти текст ПРОПУСТИТЬ через OCR
            if self.find_and_click_text("ПРОПУСТИТЬ", region=region, timeout=timeout):
                return True

            # Если не нашли, делаем клик по примерным координатам
            self.click_coord(1142, 42)
            time.sleep(timeout)

        self.logger.warning(f"Не удалось найти кнопку ПРОПУСТИТЬ за {max_attempts} попыток")
        return False

    def perform_tutorial(self, server_id):
        """
        Выполнение полного цикла обучения согласно новому ТЗ.

        Args:
            server_id: номер сервера для создания персонажа

        Returns:
            bool: True если обучение успешно завершено, False иначе
        """
        self.logger.info(f"Начало выполнения обучения на сервере {server_id}")

        try:
            # Шаг 1: Открываем профиль
            self.logger.info("Шаг 1: Клик по координатам (52, 50) - открываем профиль")
            self.click_coord(52, 50)

            # Шаг 2: Открываем настройки
            self.logger.info("Шаг 2: Клик по координатам (1076, 31) - открываем настройки")
            self.click_coord(1076, 31)

            # Шаг 3: Открываем вкладку персонажей
            self.logger.info("Шаг 3: Клик по координатам (643, 319) - открываем вкладку персонажей")
            self.click_coord(643, 319)

            # Шаг 4: Создаем персонажа на новом сервере
            self.logger.info("Шаг 4: Клик по координатам (271, 181) - создаем персонажа на новом сервере")
            self.click_coord(271, 181)

            # Шаг 5: Выбор сервера
            self.logger.info(f"Шаг 5: Выбор сервера {server_id}")
            if not self.select_server(server_id):
                self.logger.error(f"Не удалось выбрать сервер {server_id}")
                return False

            # Шаг 6: Подтверждаем создание персонажа
            self.logger.info("Шаг 6: Клик по координатам (787, 499) - подтверждаем создание персонажа")
            self.click_coord(787, 499)
            time.sleep(LOADING_TIMEOUT)  # Ожидание загрузки

            # Продолжение обучения
            if not self.execute_remaining_steps():
                self.logger.error("Ошибка при выполнении оставшихся шагов обучения")
                return False

            self.logger.info(f"Обучение на сервере {server_id} успешно завершено")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка при выполнении обучения: {e}", exc_info=True)
            return False

    def execute_remaining_steps(self):
        """
        Выполнение оставшихся шагов обучения согласно новому ТЗ.

        Returns:
            bool: True если шаги выполнены успешно, False иначе
        """
        self.logger.info("Выполнение оставшихся шагов обучения")

        try:
            # Шаг 7: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 7: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 8: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 8: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 9: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 9: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 10: Активируем бой нажатием на пушку
            self.logger.info("Шаг 10: Клик по координатам (718, 438) - активируем бой")
            self.click_coord(718, 438)

            # Шаг 11: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 11: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 12: Ждем появления "Адский Генри" и нажимаем ПРОПУСТИТЬ
            self.logger.info('Шаг 12: Ждем появления текста "Адский Генри"')
            # Ждем появления текста "Адский Генри" (координаты примерные)
            found = False
            for _ in range(20):  # Пробуем до 20 раз с интервалом в 1 секунду
                if self.find_text_on_screen("Адский Генри", region=(389, 440, 200, 100), timeout=1):
                    found = True
                    break
                time.sleep(1)

            if found:
                self.logger.info('Текст "Адский Генри" найден, нажимаем ПРОПУСТИТЬ')
                self.find_skip_button()
            else:
                self.logger.warning('Текст "Адский Генри" не найден, продолжаем выполнение')

            # Шаг 13: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 13: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 14: Жмем на иконку кораблика
            self.logger.info("Шаг 14: Клик по координатам (58, 654) - жмем на иконку кораблика")
            self.click_coord(58, 654)

            # Шаг 15: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 15: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 16: Отстраиваем нижнюю палубу
            self.logger.info("Шаг 16: Клик по координатам (638, 403) - отстраиваем нижнюю палубу")
            self.click_coord(638, 403)

            # Шаг 17: Отстраиваем паб в нижней палубе
            self.logger.info("Шаг 17: Клик по координатам (635, 373) - отстраиваем паб в нижней палубе")
            self.click_coord(635, 373)

            # Шаг 18: Латаем дыры в складе на нижней палубе
            self.logger.info("Шаг 18: Клик по координатам (635, 373) - латаем дыры в складе на нижней палубе")
            self.click_coord(635, 373)

            # Шаг 19: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 19: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 20: Отстраиваем верхнюю палубу
            self.logger.info("Шаг 20: Клик по координатам (345, 386) - отстраиваем верхнюю палубу")
            self.click_coord(345, 386)

            # Шаг 21: Выбираем пушку
            self.logger.info("Шаг 21: Клик по координатам (77, 276) - выбираем пушку")
            self.click_coord(77, 276)

            # Шаг 22: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 22: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 23: Начинаем плыть на корабле
            self.logger.info("Шаг 23: Клик по координатам (741, 145) - начинаем плыть на корабле")
            self.click_coord(741, 145)

            # Шаг 24: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 24: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 25: Нажимаем на квест "Старый соперник"
            self.logger.info('Шаг 25: Клик по координатам (93, 285) - нажимаем на квест "Старый соперник"')
            self.click_coord(93, 285)

            # Шаг 26: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 26: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 27: Нажимаем на квест "Старый соперник"
            self.logger.info('Шаг 27: Клик по координатам (93, 285) - нажимаем на квест "Старый соперник"')
            self.click_coord(93, 285)

            # Шаг 28: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 28: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 29: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 29: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 30: Жмем на любую часть экрана чтобы продолжить после победы
            self.logger.info("Шаг 30: Клик по координатам (630, 413) - жмем на любую часть экрана")
            self.click_coord(630, 413)

            # Шаг 31: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 31: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 32: Жмем на компас
            self.logger.info("Шаг 32: Клик по координатам (1074, 88) - жмем на компас")
            self.click_coord(1074, 88)

            # Шаг 33: Еще раз жмем на компас
            self.logger.info("Шаг 33: Клик по координатам (701, 258) - еще раз жмем на компас")
            self.click_coord(701, 258)

            # Шаг 34: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 34: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 35: Жмем назад чтобы выйти из вкладки компаса
            self.logger.info("Шаг 35: Клик по координатам (145, 25) - жмем назад")
            self.click_coord(145, 25)

            # Шаг 36: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 36: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 37: Нажимаем на квест "Далекая песня"
            self.logger.info('Шаг 37: Клик по координатам (93, 285) - нажимаем на квест "Далекая песня"')
            self.click_coord(93, 285)

            # Шаг 38: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 38: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 39: Еще раз нажимаем на квест "Далекая песня"
            self.logger.info('Шаг 39: Клик по координатам (93, 285) - еще раз нажимаем на квест "Далекая песня"')
            self.click_coord(93, 285)

            # Шаг 40: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 40: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 41: Жмем на фразу "Согласиться на обмен"
            self.logger.info('Шаг 41: Клик по координатам (151, 349) - жмем на фразу "Согласиться на обмен"')
            self.click_coord(151, 349)

            # Шаг 42: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 42: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 43: Нажимаем на квест "Исследовать залив Мертвецов"
            self.logger.info('Шаг 43: Клик по координатам (93, 285) - нажимаем на квест "Исследовать залив Мертвецов"')
            self.click_coord(93, 285)

            # Шаг 44: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 44: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 45: Выбираем героя для отряда
            self.logger.info("Шаг 45: Клик по координатам (85, 634) - выбираем героя для отряда")
            self.click_coord(85, 634)

            # Шаг 46: Нажимаем начать битву
            self.logger.info("Шаг 46: Клик по координатам (1157, 604) - нажимаем начать битву")
            self.click_coord(1157, 604)

            # Шаг 47: Нажимаем на картинку start_battle.png
            self.logger.info("Шаг 47: Ждем и нажимаем на картинку start_battle.png")
            for _ in range(20):  # 20 попыток с интервалом 3 секунды
                if self.click_image("start_battle", timeout=1):
                    break
                self.click_coord(642, 334)
                time.sleep(3)

            # Шаг 48: Ждем появления надписи "Отправиться в залив Мертвецов" и нажимаем на нее
            self.logger.info('Шаг 48: Ждем появления надписи "Отправиться в залив Мертвецов" и нажимаем на нее')
            for _ in range(20):  # 20 попыток с интервалом 3 секунды
                if self.find_text_on_screen("Отправиться в залив Мертвецов", region=(0, 200, 250, 170), timeout=1):
                    self.click_coord(93, 285)
                    break
                self.click_coord(642, 334)
                time.sleep(3)

            # Шаг 49: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 49: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 50: Нажимаем на квест "Отправиться в залив Мертвецов"
            self.logger.info('Шаг 50: Клик по координатам (93, 285) - нажимаем на квест "Отправиться в залив Мертвецов"')
            self.click_coord(93, 285)

            # Шаг 51: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 51: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 52: Нажимаем на череп в заливе мертвецов
            self.logger.info("Шаг 52: Клик по координатам (653, 403) - нажимаем на череп")
            self.click_coord(653, 403)

            # Шаг 53-57: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(53, 58):
                self.logger.info(f"Шаг {step}: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 58: Нажимаем на квест "Покинуть залив Мертвецов"
            self.logger.info('Шаг 58: Клик по координатам (93, 285) - нажимаем на квест "Покинуть залив Мертвецов"')
            self.click_coord(93, 285)

            # Шаг 59: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 59: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 60: Нажимаем на квест "Улучшение корабля"
            self.logger.info('Шаг 60: Клик по координатам (93, 285) - нажимаем на квест "Улучшение корабля"')
            self.click_coord(93, 285)

            # Шаг 61: Нажимаем на иконку молоточка чтобы что-то построить
            self.logger.info("Шаг 61: Клик по координатам (43, 481) - нажимаем на иконку молоточка")
            self.click_coord(43, 481)

            # Шаг 62: Выбираем корабль для улучшения
            self.logger.info("Шаг 62: Клик по координатам (127, 216) - выбираем корабль для улучшения")
            self.click_coord(127, 216)

            # Шаг 63: Дожидаемся надписи "УЛУЧШИТЬ" и кликаем на эту надпись
            self.logger.info('Шаг 63: Дожидаемся надписи "УЛУЧШИТЬ" и кликаем на нее')
            time.sleep(2)  # Дополнительная задержка для появления надписи
            if not self.find_and_click_text("УЛУЧШИТЬ", region=(983, 588, 200, 100), timeout=5):
                self.click_coord(1083, 638)  # Клик по примерным координатам, если текст не найден

            # Шаг 64: Жмем назад чтобы выйти из вкладки корабля
            self.logger.info("Шаг 64: Клик по координатам (145, 25) - жмем назад")
            self.click_coord(145, 25)

            # Шаг 65: Жмем на кнопку постройки
            self.logger.info("Шаг 65: Клик по координатам (639, 603) - жмем на кнопку постройки")
            self.click_coord(639, 603)

            # Шаг 66: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 66: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 67: Жмем на иконку компаса
            self.logger.info("Шаг 67: Клик по координатам (1072, 87) - жмем на иконку компаса")
            self.click_coord(1072, 87)

            # Шаг 68: Нажимаем на любую часть экрана
            self.logger.info("Шаг 68: Клик по координатам (1072, 87) - нажимаем на любую часть экрана")
            self.click_coord(1072, 87)

            # Шаг 69: Нажимаем на квест "Заполучи кают гребцов: 1"
            self.logger.info('Шаг 69: Клик по координатам (89, 280) - нажимаем на квест "Заполучи кают гребцов: 1"')
            self.click_coord(89, 280)

            # Шаг 70: Нажимаем на иконку молоточка чтобы что-то построить
            self.logger.info("Шаг 70: Клик по координатам (43, 481) - нажимаем на иконку молоточка")
            self.click_coord(43, 481)

            # Шаг 71: Выбираем каюту гребцов
            self.logger.info("Шаг 71: Клик по координатам (968, 507) - выбираем каюту гребцов")
            self.click_coord(968, 507)

            # Шаг 72: Подтверждаем постройку каюты гребцов
            self.logger.info("Шаг 72: Клик по координатам (676, 580) - подтверждаем постройку")
            self.click_coord(676, 580)

            # Шаг 73: Снимаем тент с каюты гребцов
            self.logger.info("Шаг 73: Клик по координатам (642, 580) - снимаем тент")
            self.click_coord(642, 580)

            # Шаг 74: Нажимаем на квест "Заполучи кают гребцов: 1"
            self.logger.info('Шаг 74: Клик по координатам (89, 280) - нажимаем на квест "Заполучи кают гребцов: 1"')
            self.click_coord(89, 280)

            # Шаг 75: Нажимаем на квест "Заполучи орудийных палуб: 1"
            self.logger.info('Шаг 75: Клик по координатам (89, 280) - нажимаем на квест "Заполучи орудийных палуб: 1"')
            self.click_coord(89, 280)

            # Шаг 76: Нажимаем на иконку молоточка чтобы что-то построить
            self.logger.info("Шаг 76: Клик по координатам (43, 481) - нажимаем на иконку молоточка")
            self.click_coord(43, 481)

            # Шаг 77: Выбираем орудийную палубу
            self.logger.info("Шаг 77: Клик по координатам (687, 514) - выбираем орудийную палубу")
            self.click_coord(687, 514)

            # Шаг 78: Подтверждаем постройку орудийной палубы
            self.logger.info("Шаг 78: Клик по координатам (679, 581) - подтверждаем постройку")
            self.click_coord(679, 581)

            # Шаг 79: Снимаем тент с орудийной палубы
            self.logger.info("Шаг 79: Клик по координатам (638, 473) - снимаем тент")
            self.click_coord(638, 473)

            # Шаг 80: Нажимаем на квест "Заполучи орудийных палуб: 1"
            self.logger.info('Шаг 80: Клик по координатам (89, 280) - нажимаем на квест "Заполучи орудийных палуб: 1"')
            self.click_coord(89, 280)

            # Шаг 81: Нажимаем на квест "Путь, что указал компас"
            self.logger.info('Шаг 81: Клик по координатам (89, 280) - нажимаем на квест "Путь, что указал компас"')
            self.click_coord(89, 280)

            # Шаг 82: Жмем на иконку компаса
            self.logger.info("Шаг 82: Клик по координатам (1072, 87) - жмем на иконку компаса")
            self.click_coord(1072, 87)

            # Шаг 83: Жмем на указатель на экране
            self.logger.info("Шаг 83: Клик по координатам (698, 273) - жмем на указатель")
            self.click_coord(698, 273)

            # Шаг 84: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 84: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 85: Нажимаем на квест "Сокровищница"
            self.logger.info('Шаг 85: Клик по координатам (89, 280) - нажимаем на квест "Сокровищница"')
            self.click_coord(89, 280)

            # Шаг 86: Нажимаем на иконку компаса над кораблем
            self.logger.info("Шаг 86: Клик по координатам (652, 214) - нажимаем на иконку компаса над кораблем")
            self.click_coord(652, 214)

            # Шаг 87-89: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(87, 90):
                self.logger.info(f"Шаг {step}: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 90: Нажимаем на квест "Богатая добыча"
            self.logger.info('Шаг 90: Клик по координатам (89, 280) - нажимаем на квест "Богатая добыча"')
            self.click_coord(89, 280)

            # Шаг 91-92: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(91, 93):
                self.logger.info(f"Шаг {step}: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 93: Ждем 7 секунд потом ищем пропустить и жмем
            self.logger.info("Шаг 93: Ждем 7 секунд, затем ищем и нажимаем ПРОПУСТИТЬ")
            time.sleep(7)
            self.find_skip_button()

            # Шаг 94: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 94: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 95: Ищем картинку coins.png и нажимаем на координаты
            self.logger.info("Шаг 95: Ищем картинку coins.png и нажимаем на координаты (931, 620)")
            if self.click_image("coins", timeout=5):
                self.logger.info("Картинка coins.png найдена и нажата")
            else:
                self.logger.warning("Картинка coins.png не найдена, нажимаем по координатам")
                self.click_coord(931, 620)

            # Шаг 96: Ищем слово ПРОПУСТИТЬ и кликаем на него
            self.logger.info("Шаг 96: Ищем и нажимаем ПРОПУСТИТЬ")
            self.find_skip_button()

            # Шаг 97: Нажимаем на квест "На волосок от смерти"
            self.logger.info('Шаг 97: Клик по координатам (89, 280) - нажимаем на квест "На волосок от смерти"')
            self.click_coord(89, 280)

            self.logger.info("Все шаги обучения успешно выполнены")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка при выполнении оставшихся шагов обучения: {e}", exc_info=True)
            return False

    def run_bot(self, cycles=1, start_server=619, end_server=1):
        """
        Запуск бота на выполнение заданного количества циклов обучения.

        Args:
            cycles: количество циклов обучения
            start_server: начальный сервер для прокачки
            end_server: конечный сервер для прокачки
        """
        self.logger.info(f"Запуск бота на {cycles} циклов с серверами от {start_server} до {end_server}")

        # Проверка корректности диапазона серверов
        if start_server < end_server:
            self.logger.error("Начальный сервер должен быть больше или равен конечному")
            return 0

        # Запуск игры в начале работы бота (только один раз)
        self.start_game()

        successful_cycles = 0
        current_server = start_server

        for cycle in range(1, cycles + 1):
            self.logger.info(f"===== Начало цикла {cycle}/{cycles}, сервер {current_server} =====")

            try:
                if self.perform_tutorial(current_server):
                    self.logger.info(f"Цикл {cycle}/{cycles} на сервере {current_server} завершен успешно")
                    successful_cycles += 1
                else:
                    self.logger.error(f"Ошибка при выполнении цикла {cycle}/{cycles} на сервере {current_server}")

                # Определяем следующий сервер для прокачки
                if cycle < cycles:
                    current_server -= 1
                    # Если достигли конечного сервера, заканчиваем работу
                    if current_server < end_server:
                        self.logger.info(f"Достигнут конечный сервер {end_server}. Завершение работы.")
                        break

                # Пауза между циклами
                time.sleep(DEFAULT_TIMEOUT * 2)

            except Exception as e:
                self.logger.error(f"Критическая ошибка в цикле {cycle}: {e}", exc_info=True)
                # Продолжаем выполнение следующих циклов

        self.logger.info(f"Бот завершил работу. Успешно выполнено {successful_cycles}/{min(cycles, start_server - end_server + 1)} циклов.")
        return successful_cycles