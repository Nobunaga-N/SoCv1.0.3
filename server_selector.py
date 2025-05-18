"""
Улучшенный модуль для автоматизации выбора сервера и сезона в игре.
Исправлены координаты и добавлены паузы между действиями.
"""
import cv2
import numpy as np
import re
import logging
import random
from typing import Optional, List, Tuple, Dict
import time

from config import SEASONS


class ImprovedServerSelector:
    """
    Улучшенный класс для автоматизации выбора сервера и сезона.
    """

    def __init__(self, adb_controller, image_handler):
        """
        Инициализация селектора серверов.

        Args:
            adb_controller: контроллер ADB
            image_handler: обработчик изображений
        """
        self.logger = logging.getLogger('sea_conquest_bot.improved_server_selector')
        self.adb = adb_controller
        self.image = image_handler

        # ИСПРАВЛЕННЫЕ координаты для сезонов на основе изображения
        self.season_coords = {
            'S1': (250, 180),   # Первый сезон
            'S2': (250, 230),   # Второй сезон
            'S3': (250, 280),   # Третий сезон
            'S4': (250, 380),   # Четвертый сезон (исправлено с 300 на 380)
            'S5': (250, 470),   # Пятый сезон (исправлено с 340 на 470)
            'X1': (250, 520),   # X1 сезон (исправлено с 390 на 520)
            'X2': (250, 230),   # После скроллинга
            'X3': (250, 280),   # После скроллинга
            'X4': (250, 330)    # После скроллинга
        }

        # Координаты для скроллинга (более точные)
        self.season_scroll_start = (257, 550)
        self.season_scroll_end = (257, 200)
        self.server_scroll_start = (640, 550)  # Центр области серверов
        self.server_scroll_end = (640, 200)

        # Область для поиска текста сезонов (левая панель)
        self.season_text_roi = (150, 150, 220, 400)

        # Область для поиска текста серверов (правая панель)
        self.server_text_roi = (400, 150, 500, 450)

        # Попытка загрузки OCR
        try:
            import pytesseract
            self.ocr_available = True
            self.logger.info("OCR (Tesseract) доступен для использования")
        except ImportError:
            self.ocr_available = False
            self.logger.warning("OCR (Tesseract) не доступен, будет использоваться распознавание шаблонов")

        # ИСПРАВЛЕННЫЕ координаты для серверов
        self.server_coords = {
            'left_column': 500,   # X-координата для левого столбца серверов
            'right_column': 900,  # X-координата для правого столбца серверов
            'base_y': 155,        # Y-координата первого сервера
            'step_y': 45          # Шаг между серверами по вертикали
        }

    def select_season(self, season_id):
        """
        Улучшенный метод выбора сезона с паузами.

        Args:
            season_id: идентификатор сезона (S1, S2, S3, S4, S5, X1, X2, X3, X4)

        Returns:
            bool: True если сезон выбран успешно, False иначе
        """
        self.logger.info(f"Выбор сезона: {season_id}")

        # Проверка существования сезона
        if season_id not in self.season_coords:
            self.logger.error(f"Сезон '{season_id}' не найден в конфигурации")
            return False

        # Определение необходимости скроллинга для нижних сезонов
        if season_id in ['X2', 'X3', 'X4']:
            self.logger.info(f"Выполняем скроллинг для отображения сезона {season_id}")

            # Скроллинг вниз для показа нижних сезонов
            self.adb.swipe(
                self.season_scroll_start[0],
                self.season_scroll_start[1],
                self.season_scroll_end[0],
                self.season_scroll_end[1],
                duration=1000  # Уменьшенная продолжительность для плавности
            )

            # Обязательная пауза после скроллинга
            time.sleep(2)
            self.logger.info("Пауза 2 секунды после скроллинга сезонов")

            # Проверка видимости нужного сезона после скроллинга
            if not self.is_season_visible(season_id):
                self.logger.warning(f"Сезон {season_id} не виден после скроллинга, пробуем еще раз")

                # Повторный скроллинг с немного большим смещением
                self.adb.swipe(
                    self.season_scroll_start[0],
                    self.season_scroll_start[1],
                    self.season_scroll_end[0] - 30,  # Больше смещение
                    self.season_scroll_end[1],
                    duration=1000
                )

                time.sleep(2)
                self.logger.info("Дополнительная пауза 2 секунды после повторного скроллинга")

        # Клик по сезону с паузой до и после
        x, y = self.season_coords[season_id]
        self.logger.info(f"Клик по сезону {season_id} по координатам ({x}, {y})")

        # Небольшая пауза перед кликом
        time.sleep(0.5)
        self.adb.tap(x, y)

        # Пауза после клика для загрузки серверов
        time.sleep(1.5)
        self.logger.info("Пауза 1.5 секунды после выбора сезона")

        return True

    def is_season_visible(self, season_id):
        """
        Проверка видимости сезона на экране с использованием OCR.

        Args:
            season_id: идентификатор сезона

        Returns:
            bool: True если сезон виден на экране, False иначе
        """
        if not self.ocr_available:
            # Если OCR недоступен, просто предполагаем, что сезон виден
            return True

        try:
            screenshot = self.adb.screenshot()
            x, y, w, h = self.season_text_roi
            roi = screenshot[y:y + h, x:x + w]

            import pytesseract
            # Предобработка для улучшения распознавания
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Распознавание текста
            text = pytesseract.image_to_string(binary, lang='rus+eng')

            # Проверка наличия сезона в тексте
            season_text = f"Сезон {season_id}"
            if season_text in text or season_id in text:
                self.logger.info(f"Сезон {season_id} виден на экране")
                return True

            self.logger.warning(f"Сезон {season_id} не обнаружен в тексте: {text}")
            return False

        except Exception as e:
            self.logger.error(f"Ошибка при проверке видимости сезона: {e}")
            return True

    def get_visible_servers(self):
        """
        Получение списка видимых на экране серверов с улучшенным распознаванием.

        Returns:
            list: список видимых серверов (номеров)
        """
        visible_servers = []

        if not self.ocr_available:
            self.logger.warning("OCR не доступен, возвращаем примерный список серверов")
            return []

        try:
            screenshot = self.adb.screenshot()
            x, y, w, h = self.server_text_roi
            roi = screenshot[y:y + h, x:x + w]

            import pytesseract
            # Улучшенная предобработка
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Применяем несколько методов для лучшего распознавания
            methods_results = []

            # Метод 1: Стандартная бинаризация
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            text1 = pytesseract.image_to_string(binary, lang='rus+eng')
            methods_results.append(text1)

            # Метод 2: Адаптивная бинаризация
            binary_adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            text2 = pytesseract.image_to_string(binary_adaptive, lang='rus+eng')
            methods_results.append(text2)

            # Объединяем результаты всех методов
            combined_text = ' '.join(methods_results)

            # Поиск номеров серверов в тексте
            servers = []

            # Ищем паттерн "Море #XXX" или просто "#XXX"
            patterns = [r"Море\s*#(\d{1,3})", r"#(\d{1,3})", r"(\d{3})"]

            for pattern in patterns:
                matches = re.findall(pattern, combined_text)
                for match in matches:
                    try:
                        server_id = int(match)
                        # Фильтруем только реалистичные номера серверов
                        if 1 <= server_id <= 619:
                            servers.append(server_id)
                    except ValueError:
                        continue

            # Убираем дубликаты и сортируем
            servers = sorted(list(set(servers)), reverse=True)
            self.logger.info(f"Распознанные сервера: {servers}")
            return servers

        except Exception as e:
            self.logger.error(f"Ошибка при распознавании серверов: {e}")
            return []

    def get_server_coordinates(self, server_id, visible_servers):
        """
        ИСПРАВЛЕННЫЙ метод получения координат для клика по серверу.

        Args:
            server_id: номер сервера
            visible_servers: список видимых серверов (отсортированный по убыванию)

        Returns:
            tuple: (x, y) координаты для клика по серверу или None
        """
        if server_id not in visible_servers:
            self.logger.warning(f"Сервер {server_id} не найден в списке видимых серверов: {visible_servers}")
            return None

        # Определяем индекс сервера в списке видимых (отсортированных по убыванию)
        visible_servers_sorted = sorted(visible_servers, reverse=True)
        try:
            index = visible_servers_sorted.index(server_id)
        except ValueError:
            self.logger.error(f"Сервер {server_id} не найден в отсортированном списке")
            return None

        # Определяем позицию сервера (два столбца)
        row = index // 2  # Номер строки (0, 1, 2, ...)
        column = index % 2  # Столбец (0 - левый, 1 - правый)

        # Рассчитываем координаты
        if column == 0:  # Левый столбец
            x = self.server_coords['left_column']
        else:  # Правый столбец
            x = self.server_coords['right_column']

        y = self.server_coords['base_y'] + row * self.server_coords['step_y']

        self.logger.info(f"Сервер {server_id}: индекс {index}, строка {row}, столбец {column}, координаты ({x}, {y})")
        return (x, y)

    def select_server(self, server_id):
        """
        Улучшенный метод выбора сервера с паузами и более точными координатами.

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

        # Получение списка видимых серверов
        visible_servers = self.get_visible_servers()

        # Если сервер уже виден, кликаем по нему
        if server_id in visible_servers:
            self.logger.info(f"Сервер {server_id} уже виден на экране")
            server_coords = self.get_server_coordinates(server_id, visible_servers)
            if server_coords:
                # Пауза перед кликом
                time.sleep(0.5)
                self.adb.tap(server_coords[0], server_coords[1])
                # Пауза после клика
                time.sleep(1.5)
                self.logger.info("Пауза 1.5 секунды после выбора сервера")
                return True
        else:
            # Нужно проскроллить до сервера
            self.logger.info(f"Сервер {server_id} не виден, выполняем скроллинг")

            # Определяем направление скроллинга
            max_attempts = 5
            for attempt in range(max_attempts):
                self.logger.info(f"Попытка скроллинга {attempt + 1}/{max_attempts}")

                # Скроллинг вниз для поиска сервера
                self.adb.swipe(
                    self.server_scroll_start[0],
                    self.server_scroll_start[1],
                    self.server_scroll_end[0],
                    self.server_scroll_end[1],
                    duration=1000
                )

                # Пауза после скроллинга
                time.sleep(1.5)
                self.logger.info("Пауза 1.5 секунды после скроллинга серверов")

                # Получаем обновленный список видимых серверов
                visible_servers = self.get_visible_servers()

                # Проверяем, найден ли нужный сервер
                if server_id in visible_servers:
                    self.logger.info(f"Сервер {server_id} найден после скроллинга")
                    server_coords = self.get_server_coordinates(server_id, visible_servers)
                    if server_coords:
                        # Пауза перед кликом
                        time.sleep(0.5)
                        self.adb.tap(server_coords[0], server_coords[1])
                        # Пауза после клика
                        time.sleep(1.5)
                        self.logger.info("Пауза 1.5 секунды после выбора сервера")
                        return True

                # Если сервер не найден, но видны сервера с меньшими номерами,
                # значит мы проскролили слишком далеко
                if visible_servers and min(visible_servers) < server_id:
                    self.logger.info("Проскролили слишком далеко, ищем ближайший доступный сервер")
                    break

            # Если не нашли точный сервер, ищем ближайший доступный
            available_server = self.find_next_available_server(server_id, visible_servers)
            if available_server:
                self.logger.info(f"Выбираем ближайший доступный сервер {available_server}")
                server_coords = self.get_server_coordinates(available_server, visible_servers)
                if server_coords:
                    # Пауза перед кликом
                    time.sleep(0.5)
                    self.adb.tap(server_coords[0], server_coords[1])
                    # Пауза после клика
                    time.sleep(1.5)
                    self.logger.info("Пауза 1.5 секунды после выбора сервера")
                    return True

        self.logger.error(f"Не удалось выбрать сервер {server_id}")
        return False

    def find_next_available_server(self, target_server, visible_servers):
        """
        Поиск следующего доступного сервера, если целевой недоступен.

        Args:
            target_server: целевой номер сервера
            visible_servers: список видимых серверов

        Returns:
            int: номер следующего доступного сервера или None
        """
        if not visible_servers:
            return None

        # Сортируем видимые сервера по убыванию
        visible_servers = sorted(visible_servers, reverse=True)

        # Ищем ближайший доступный сервер (с меньшим номером)
        for server in visible_servers:
            if server < target_server:
                self.logger.info(f"Найден ближайший доступный сервер {server} (меньше целевого {target_server})")
                return server

        # Если не нашли сервер с меньшим номером, берем наибольший доступный
        if visible_servers:
            self.logger.info(f"Выбираем наибольший доступный сервер {visible_servers[0]}")
            return visible_servers[0]

        return None

    def select_server_with_fallback(self, server_id):
        """
        Метод выбора сервера с резервными вариантами.

        Args:
            server_id: номер сервера

        Returns:
            tuple: (success, selected_server) - успех операции и выбранный сервер
        """
        self.logger.info(f"Попытка выбора сервера {server_id} с резервными вариантами")

        # Сначала пытаемся выбрать точный сервер
        if self.select_server(server_id):
            return True, server_id

        # Если не получилось, ищем ближайшие сервера в том же сезоне
        season_id = None
        for s_id, s_data in SEASONS.items():
            if s_data['min_server'] >= server_id >= s_data['max_server']:
                season_id = s_id
                break

        if not season_id:
            self.logger.error(f"Не удалось определить сезон для сервера {server_id}")
            return False, None

        # Пытаемся найти альтернативные сервера в том же сезоне
        season_data = SEASONS[season_id]
        alternative_servers = []

        # Добавляем сервера с меньшими номерами (приоритет)
        for alt_server in range(server_id - 1, season_data['max_server'] - 1, -1):
            alternative_servers.append(alt_server)
            if len(alternative_servers) >= 3:  # Ограничиваем количество попыток
                break

        # Пытаемся выбрать альтернативные сервера
        for alt_server in alternative_servers:
            self.logger.info(f"Пытаемся выбрать альтернативный сервер {alt_server}")
            if self.select_server(alt_server):
                self.logger.info(f"Успешно выбран альтернативный сервер {alt_server}")
                return True, alt_server

        self.logger.error(f"Не удалось выбрать сервер {server_id} или его альтернативы")
        return False, None