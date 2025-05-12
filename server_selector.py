"""
Модуль для автоматизации выбора сервера и сезона в игре.
"""
import cv2
import numpy as np
import re
import logging
import random
from typing import Optional, List, Tuple, Dict

from config import SEASONS


class ServerSelector:
    """
    Класс для автоматизации выбора сервера и сезона.
    Использует распознавание текста и распознавание изображений.
    """

    def __init__(self, adb_controller, image_handler):
        """
        Инициализация селектора серверов.

        Args:
            adb_controller: контроллер ADB
            image_handler: обработчик изображений
        """
        self.logger = logging.getLogger('sea_conquest_bot.server_selector')
        self.adb = adb_controller
        self.image = image_handler

        # Координаты для навигации по сезонам и серверам
        self.season_coords = {
            'S1': (400, 160),
            'S2': (400, 200),
            'S3': (400, 240),
            'S4': (400, 280),
            'S5': (400, 320),
            'X1': (400, 360),
            'X2': (400, 240),  # После скроллинга
            'X3': (400, 280)  # После скроллинга
        }

        # Координаты для скроллинга
        self.season_scroll_start = (257, 353)
        self.season_scroll_end = (254, 187)
        self.server_scroll_start = (778, 567)
        self.server_scroll_end = (778, 130)

        # Область для поиска текста сезонов
        self.season_text_roi = (300, 150, 200, 250)  # (x, y, w, h)

        # Область для поиска текста серверов
        self.server_text_roi = (350, 150, 400, 450)  # (x, y, w, h)

        # Попытка загрузки OCR, если доступен
        try:
            import pytesseract
            self.ocr_available = True
            self.logger.info("OCR (Tesseract) доступен для использования")
        except ImportError:
            self.ocr_available = False
            self.logger.warning("OCR (Tesseract) не доступен, будет использоваться распознавание шаблонов")

    def get_random_server_in_season(self, season_id):
        """
        Получение случайного сервера в указанном сезоне.

        Args:
            season_id: идентификатор сезона (S1, S2, S3, S4, S5, X1, X2, X3)

        Returns:
            int: номер сервера или None, если сезон не найден
        """
        if season_id not in SEASONS:
            self.logger.error(f"Сезон '{season_id}' не найден в конфигурации")
            return None

        min_server = SEASONS[season_id]['max_server']
        max_server = SEASONS[season_id]['min_server']

        server_id = random.randint(min_server, max_server)
        self.logger.info(f"Выбран случайный сервер {server_id} в сезоне {season_id}")

        return server_id

    def recognize_seasons(self, screenshot):
        """
        Распознавание сезонов на экране.

        Args:
            screenshot: скриншот экрана

        Returns:
            list: список распознанных сезонов
        """
        if not self.ocr_available:
            self.logger.warning("OCR не доступен, используется предопределенный список сезонов")
            return list(self.season_coords.keys())

        # Получение области интереса
        x, y, w, h = self.season_text_roi
        roi = screenshot[y:y + h, x:x + w]

        # Предобработка для OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Распознавание текста
        import pytesseract
        text = pytesseract.image_to_string(binary, lang='eng')

        # Поиск сезонов в тексте
        seasons = []
        for line in text.splitlines():
            # Поиск шаблона "Сезон S1", "Сезон X2" и т.д.
            match = re.search(r'Сезон\s+([SX][1-5])', line)
            if match:
                season_id = match.group(1)
                seasons.append(season_id)

        self.logger.info(f"Распознанные сезоны: {seasons}")
        return seasons

    def recognize_servers(self, screenshot):
        """
        Распознавание серверов на экране.

        Args:
            screenshot: скриншот экрана

        Returns:
            list: список распознанных серверов (номеров)
        """
        if not self.ocr_available:
            self.logger.warning("OCR не доступен, используется предопределенный список серверов")
            # Возвращаем примерные номера серверов для текущей страницы
            return [i for i in range(580, 570, -1)]

        # Получение области интереса
        x, y, w, h = self.server_text_roi
        roi = screenshot[y:y + h, x:x + w]

        # Предобработка для OCR
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Распознавание текста
        import pytesseract
        text = pytesseract.image_to_string(binary, lang='eng')

        # Поиск номеров серверов в тексте
        servers = []
        for line in text.splitlines():
            # Поиск шаблона номера сервера
            match = re.search(r'(\d{1,3})', line)
            if match:
                server_id = int(match.group(1))
                servers.append(server_id)

        self.logger.info(f"Распознанные серверы: {servers}")
        return servers

    def select_season(self, season_id):
        """
        Выбор сезона.

        Args:
            season_id: идентификатор сезона (S1, S2, S3, S4, S5, X1, X2, X3)

        Returns:
            bool: True если сезон выбран успешно, False иначе
        """
        self.logger.info(f"Выбор сезона: {season_id}")

        # Проверка существования сезона
        if season_id not in self.season_coords:
            self.logger.error(f"Сезон '{season_id}' не найден в конфигурации")
            return False

        # Определение необходимости скроллинга
        if season_id in ['X2', 'X3']:
            # Скроллинг для отображения нижних сезонов
            self.adb.swipe(
                self.season_scroll_start[0],
                self.season_scroll_start[1],
                self.season_scroll_end[0],
                self.season_scroll_end[1]
            )

        # Клик по сезону
        x, y = self.season_coords[season_id]
        self.adb.tap(x, y)

        # В реальном сценарии здесь можно добавить проверку успешности выбора сезона
        return True

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
        season_id = None
        for s_id, s_data in SEASONS.items():
            if s_data['min_server'] >= server_id >= s_data['max_server']:
                season_id = s_id
                break

        if not season_id:
            self.logger.error(f"Не удалось определить сезон для сервера {server_id}")
            return False

        # Выбор сезона
        if not self.select_season(season_id):
            return False

        # Количество серверов на странице и расчет страницы
        servers_per_page = 10

        # Порядковый номер сервера в сезоне (от новых к старым)
        max_server = SEASONS[season_id]['min_server']
        min_server = SEASONS[season_id]['max_server']
        server_index = max_server - server_id

        # Номер страницы (начиная с 0)
        page = server_index // servers_per_page

        # Индекс на странице (начиная с 0)
        page_index = server_index % servers_per_page

        # Скроллинг до нужной страницы
        for _ in range(page):
            self.adb.swipe(
                self.server_scroll_start[0],
                self.server_scroll_start[1],
                self.server_scroll_end[0],
                self.server_scroll_end[1]
            )

        # Клик по серверу (примерные координаты)
        server_base_y = 150  # Примерная Y-координата первого сервера
        server_step_y = 45  # Шаг между серверами по Y

        server_y = server_base_y + page_index * server_step_y
        server_x = 400  # Примерная X-координата сервера

        self.adb.tap(server_x, server_y)

        # В реальном сценарии здесь можно добавить проверку успешности выбора сервера
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
        # Например, после выбора сервера проверить наличие кнопки подтверждения
        # или отсутствие сообщения о недоступности сервера

        # Для примера считаем, что все серверы доступны
        return True

    def find_available_server(self, season_id, start_server=None):
        """
        Поиск доступного сервера в указанном сезоне.

        Args:
            season_id: идентификатор сезона (S1, S2, S3, S4, S5, X1, X2, X3)
            start_server: начальный сервер для поиска (если None, будет выбран максимальный в сезоне)

        Returns:
            int: номер доступного сервера или None, если не найдено
        """
        if season_id not in SEASONS:
            self.logger.error(f"Сезон '{season_id}' не найден в конфигурации")
            return None

        max_server = SEASONS[season_id]['min_server']
        min_server = SEASONS[season_id]['max_server']

        # Если начальный сервер не указан, берем максимальный
        if start_server is None:
            start_server = max_server

        # Проверка, что сервер находится в пределах сезона
        if start_server > max_server or start_server < min_server:
            self.logger.error(f"Сервер {start_server} не входит в сезон {season_id}")
            return None

        # Перебираем серверы в порядке убывания (от новых к старым)
        for server_id in range(start_server, min_server - 1, -1):
            self.logger.info(f"Проверка доступности сервера {server_id}")

            # Выбор сервера
            if not self.select_server(server_id):
                continue

            # Проверка доступности
            if self.is_server_available(server_id):
                self.logger.info(f"Найден доступный сервер: {server_id}")
                return server_id

        self.logger.warning(f"Не найдено доступных серверов в сезоне {season_id}")
        return None