"""
Модуль для автоматизации выбора сервера и сезона в игре.
"""
import cv2
import numpy as np
import re
import logging
import random
from typing import Optional, List, Tuple, Dict
import time

from config import SEASONS


class ServerSelector:
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
        self.logger = logging.getLogger('sea_conquest_bot.server_selector')
        self.adb = adb_controller
        self.image = image_handler

        # Координаты для навигации по сезонам и серверам
        self.season_coords = {
            'S1': (250, 160),  # Изменена X-координата для лучшего попадания по тексту
            'S2': (250, 220),
            'S3': (250, 260),
            'S4': (250, 300),
            'S5': (250, 340),
            'X1': (250, 390),
            'X2': (250, 260),  # После скроллинга
            'X3': (250, 300),  # После скроллинга
            'X4': (250, 340)  # После скроллинга
        }

        # Координаты для скроллинга (уменьшены на 20% для большей точности)
        self.season_scroll_start = (257, 553)
        self.season_scroll_end = (254, 287)  # Было 187, увеличено на 20%
        self.server_scroll_start = (778, 567)
        self.server_scroll_end = (778, 180)  # Было 130, увеличено на 20%

        # Область для поиска текста сезонов
        self.season_text_roi = (150, 150, 350, 250)  # Увеличена область для лучшего распознавания

        # Область для поиска текста серверов
        self.server_text_roi = (350, 150, 500, 450)  # Увеличена область для лучшего распознавания

        # Попытка загрузки OCR, если доступен
        try:
            import pytesseract
            self.ocr_available = True
            self.logger.info("OCR (Tesseract) доступен для использования")
        except ImportError:
            self.ocr_available = False
            self.logger.warning("OCR (Tesseract) не доступен, будет использоваться распознавание шаблонов")

        # Инициализация координат для серверов
        self.server_base_y = 150  # Y-координата первого сервера
        self.server_step_y = 45  # Шаг между серверами по Y
        self.server_x = 500  # X-координата сервера (примерная середина элемента)

    def select_season(self, season_id):
        """
        Улучшенный метод выбора сезона.

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
                duration=1500  # Увеличенная продолжительность для плавности
            )

            # Задержка после скроллинга для загрузки UI
            time.sleep(2)

            # Проверка видимости нужного сезона после скроллинга
            if not self.is_season_visible(season_id):
                self.logger.warning(f"Сезон {season_id} не виден после скроллинга, пробуем еще раз")

                # Повторный скроллинг с большим смещением
                self.adb.swipe(
                    self.season_scroll_start[0],
                    self.season_scroll_start[1],
                    self.season_scroll_end[0],
                    self.season_scroll_end[1] - 20,
                    duration=1500
                )

                time.sleep(2)

                # Если после повторного скроллинга сезон все еще не виден, это ошибка
                if not self.is_season_visible(season_id):
                    self.logger.error(f"Не удалось отобразить сезон {season_id} после скроллинга")
                    return False

        # Клик по сезону
        x, y = self.season_coords[season_id]
        self.logger.info(f"Клик по сезону {season_id} по координатам ({x}, {y})")
        self.adb.tap(x, y)

        # Задержка после выбора сезона
        time.sleep(2)

        # В реальном сценарии здесь можно добавить проверку успешности выбора сезона
        # Например, через распознавание текста или проверку UI-элементов

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
            if season_text in text:
                self.logger.info(f"Сезон {season_id} виден на экране")
                return True

            self.logger.warning(f"Сезон {season_id} не обнаружен в тексте: {text}")
            return False

        except Exception as e:
            self.logger.error(f"Ошибка при проверке видимости сезона: {e}")
            # В случае ошибки предполагаем, что сезон виден
            return True

    def select_server(self, server_id):
        """
        Улучшенный метод выбора сервера.

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

        # Получение информации о серверах в сезоне
        max_server = SEASONS[season_id]['min_server']
        min_server = SEASONS[season_id]['max_server']

        # Порядковый номер сервера в сезоне (от новых к старым)
        server_index = max_server - server_id

        # Примерное количество серверов на странице
        servers_per_page = 8  # Уменьшено с 10 для более точного расчета

        # Номер страницы (начиная с 0)
        page = server_index // servers_per_page

        # Индекс на странице (начиная с 0)
        page_index = server_index % servers_per_page

        self.logger.info(
            f"Сервер {server_id}: сезон {season_id}, номер {server_index}, страница {page}, позиция {page_index}")

        # Попытка распознать текущие видимые сервера
        visible_servers = self.recognize_servers_with_retry()

        # Проверяем, видим ли уже целевой сервер
        if server_id in visible_servers:
            self.logger.info(f"Сервер {server_id} уже виден на экране")
        else:
            # Скроллинг до нужной страницы
            scroll_count = 0
            max_scrolls = page + 2  # Добавляем запас

            while scroll_count < max_scrolls:
                # Если сервер уже виден, прекращаем скроллинг
                if server_id in visible_servers:
                    self.logger.info(f"Сервер {server_id} найден после {scroll_count} скроллов")
                    break

                # Если видны сервера меньше целевого, значит мы проскролили слишком далеко
                if visible_servers and min(visible_servers) < server_id:
                    self.logger.warning(f"Проскролили слишком далеко, видны сервера {visible_servers}")
                    # Скроллинг в обратную сторону
                    self.adb.swipe(
                        self.server_scroll_end[0],
                        self.server_scroll_end[1],
                        self.server_scroll_start[0],
                        self.server_scroll_start[1],
                        duration=1500
                    )
                    time.sleep(2)
                else:
                    # Скроллинг вниз для отображения следующей страницы серверов
                    self.logger.info(f"Скроллинг для отображения сервера {server_id}, попытка {scroll_count + 1}")
                    self.adb.swipe(
                        self.server_scroll_start[0],
                        self.server_scroll_start[1],
                        self.server_scroll_end[0],
                        self.server_scroll_end[1],
                        duration=1500
                    )
                    time.sleep(2)  # Увеличенная задержка после скроллинга

                # Получение обновленного списка видимых серверов
                prev_servers = visible_servers
                visible_servers = self.recognize_servers_with_retry()

                # Если список серверов не изменился, возможно, достигнут конец списка
                if set(visible_servers) == set(prev_servers) and visible_servers:
                    self.logger.warning(
                        "Список серверов не изменился после скроллинга, возможно достигнут конец списка")
                    break

                scroll_count += 1

        # После завершения скроллинга проверяем, найден ли нужный сервер
        if server_id in visible_servers:
            # Определение позиции сервера на экране
            server_position = visible_servers.index(server_id)

            # Примерные координаты серверов (расчет на основе позиции)
            self.server_base_y = 150  # Y-координата первого сервера
            self.server_step_y = 45  # Шаг между серверами по Y
            self.server_x = 500  # X-координата сервера (примерная середина элемента)

            server_y = self.server_base_y + server_position * self.server_step_y

            self.logger.info(
                f"Сервер {server_id} найден на позиции {server_position}, координаты ({self.server_x}, {server_y})")

            # Клик по серверу
            self.adb.tap(self.server_x, server_y)
            time.sleep(2)  # Задержка после выбора сервера

            return True
        else:
            # Сервер не найден, возможно он переполнен
            self.logger.warning(f"Сервер {server_id} не найден в списке. Возможно, он переполнен.")

            # Поиск ближайшего доступного сервера (предпочтительно с меньшим номером)
            available_server = self.find_available_server(server_id, visible_servers)

            if available_server:
                self.logger.info(f"Выбираем доступный сервер {available_server} вместо {server_id}")

                # Определение позиции доступного сервера
                server_position = visible_servers.index(available_server)

                # Расчет координат
                server_y = self.server_base_y + server_position * self.server_step_y

                # Клик по доступному серверу
                self.adb.tap(self.server_x, server_y)
                time.sleep(2)

                return True

            self.logger.error(f"Не найдено доступных серверов вблизи {server_id}")
            return False

    def recognize_servers_with_retry(self, max_attempts=3):
        """
        Распознавание серверов с повторными попытками.

        Args:
            max_attempts: максимальное количество попыток распознавания

        Returns:
            list: список распознанных серверов (номеров)
        """
        for attempt in range(max_attempts):
            servers = self.recognize_servers()
            if servers:
                return servers

            self.logger.warning(f"Не удалось распознать сервера, попытка {attempt + 1} из {max_attempts}")
            time.sleep(1)

        # Если после всех попыток серверы не распознаны, возвращаем пустой список
        return []

    def recognize_servers(self):
        """
        Улучшенный метод распознавания серверов на экране с использованием OCR.

        Returns:
            list: список распознанных серверов (номеров)
        """
        if not self.ocr_available:
            self.logger.warning("OCR не доступен, возвращаем примерный список серверов")
            # Возвращаем примерные номера серверов для текущей страницы
            return [i for i in range(580, 570, -1)]

        try:
            screenshot = self.adb.screenshot()
            x, y, w, h = self.server_text_roi
            roi = screenshot[y:y + h, x:x + w]

            import pytesseract
            # Предобработка для улучшения распознавания
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            # Дополнительная обработка для улучшения распознавания
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)

            # Распознавание текста
            text = pytesseract.image_to_string(binary, lang='rus+eng')

            # Поиск номеров серверов в тексте с использованием регулярных выражений
            servers = []
            import re

            # Поиск шаблона "Море #XXX"
            pattern = r"Море\s+#(\d{1,3})"
            matches = re.findall(pattern, text)

            for match in matches:
                try:
                    server_id = int(match)
                    servers.append(server_id)
                except ValueError:
                    continue

            # Если серверы не найдены, пробуем более простой шаблон
            if not servers:
                # Поиск любых чисел, похожих на номера серверов (3 цифры)
                digit_pattern = r"#?(\d{3})"
                matches = re.findall(digit_pattern, text)

                for match in matches:
                    try:
                        server_id = int(match)
                        # Фильтруем только реалистичные номера серверов (из диапазона всех возможных)
                        if 1 <= server_id <= 619:  # Диапазон всех возможных серверов
                            servers.append(server_id)
                    except ValueError:
                        continue

            # Сортировка серверов по убыванию (от новых к старым)
            servers.sort(reverse=True)

            self.logger.info(f"Распознанные сервера: {servers}")
            return servers

        except Exception as e:
            self.logger.error(f"Ошибка при распознавании серверов: {e}")
            return []

    def find_available_server(self, target_server, visible_servers):
        """
        Поиск доступного сервера, если целевой недоступен или переполнен.

        Args:
            target_server: целевой номер сервера
            visible_servers: список видимых серверов

        Returns:
            int: номер доступного сервера или None
        """
        if not visible_servers:
            return None

        # Сортировка серверов по убыванию (от больших номеров к меньшим)
        visible_servers.sort(reverse=True)

        # Ищем ближайший сервер с меньшим номером
        for server in visible_servers:
            if server < target_server:
                self.logger.info(f"Найден доступный сервер {server} (ближайший меньший к {target_server})")
                return server

        # Если не найден сервер с меньшим номером, берем наибольший из доступных
        self.logger.info(
            f"Не найдено серверов меньше {target_server}, выбираем наибольший доступный {visible_servers[0]}")
        return visible_servers[0]

    def is_server_available(self, server_id):
        """
        Улучшенный метод проверки доступности сервера.

        Args:
            server_id: номер сервера

        Returns:
            bool: True если сервер доступен, False иначе
        """
        # Получение текущих видимых серверов
        visible_servers = self.recognize_servers()

        # Если сервер виден в списке, считаем его доступным
        if server_id in visible_servers:
            self.logger.info(f"Сервер {server_id} доступен")
            return True

        # Дополнительно можно проверить наличие иконок состояния сервера
        # Например, на переполненных серверах может быть специальная иконка

        self.logger.warning(f"Сервер {server_id} не найден в списке, считаем его недоступным")
        return False

    def find_available_server_in_range(self, start_server=None, end_server=None):
        """
        Улучшенный метод поиска доступного сервера в указанном диапазоне.

        Args:
            start_server: начальный сервер для поиска
            end_server: конечный сервер для поиска

        Returns:
            int: номер доступного сервера или None, если не найдено
        """
        if start_server is None or end_server is None:
            self.logger.error("Не указан диапазон серверов")
            return None

        # Определение сезона для начального сервера
        season_id = None
        for s_id, s_data in SEASONS.items():
            if s_data['min_server'] >= start_server >= s_data['max_server']:
                season_id = s_id
                break

        if not season_id:
            # Если сезон не найден, берем сервер из первого сезона
            season_id = 'S1'
            start_server = SEASONS[season_id]['min_server']
            self.logger.warning(f"Не удалось определить сезон для сервера {start_server}. "
                                f"Используем сервер {start_server} из сезона {season_id}")

        # Выбор сезона
        if not self.select_season(season_id):
            self.logger.error(f"Не удалось выбрать сезон {season_id}")
            return None

        # Перебираем серверы в порядке убывания (от новых к старым)
        current_server = start_server
        while current_server >= end_server:
            # Определение сезона для текущего сервера
            current_season_id = None
            for s_id, s_data in SEASONS.items():
                if s_data['min_server'] >= current_server >= s_data['max_server']:
                    current_season_id = s_id
                    break

            # Если сезон изменился, выбираем новый сезон
            if current_season_id and current_season_id != season_id:
                season_id = current_season_id
                self.logger.info(f"Переход к сезону {season_id}")
                if not self.select_season(season_id):
                    self.logger.error(f"Не удалось выбрать сезон {season_id}")
                    # Продолжаем с поиском в текущем сезоне

            # Получение текущих видимых серверов
            visible_servers = self.recognize_servers()

            # Проверка, видим ли текущий сервер
            if current_server in visible_servers:
                # Сервер найден, проверяем его доступность
                if self.is_server_available(current_server):
                    self.logger.info(f"Найден доступный сервер: {current_server}")
                    return current_server
            else:
                # Сервер не виден, пробуем прокрутить список
                # Определяем направление скроллинга
                should_scroll_down = True

                if visible_servers:
                    # Если видны сервера с большими номерами, нужно скроллить вниз
                    if min(visible_servers) > current_server:
                        should_scroll_down = True
                    # Если видны сервера с меньшими номерами, возможно нужно скроллить вверх
                    elif max(visible_servers) < current_server:
                        should_scroll_down = False

                if should_scroll_down:
                    # Скроллинг вниз
                    self.logger.info(f"Скроллинг вниз для поиска сервера {current_server}")
                    self.adb.swipe(
                        self.server_scroll_start[0],
                        self.server_scroll_start[1],
                        self.server_scroll_end[0],
                        self.server_scroll_end[1],
                        duration=1500
                    )
                else:
                    # Скроллинг вверх
                    self.logger.info(f"Скроллинг вверх для поиска сервера {current_server}")
                    self.adb.swipe(
                        self.server_scroll_end[0],
                        self.server_scroll_end[1],
                        self.server_scroll_start[0],
                        self.server_scroll_start[1],
                        duration=1500
                    )

                time.sleep(2)  # Задержка после скроллинга

                # Получаем обновленный список видимых серверов
                updated_servers = self.recognize_servers()

                # Проверяем, изменился ли список серверов после скроллинга
                if set(updated_servers) == set(visible_servers) and visible_servers:
                    # Список не изменился, возможно достигнут предел скроллинга
                    self.logger.warning("Список серверов не изменился после скроллинга")

                    # Ищем ближайший доступный сервер среди видимых
                    for server in sorted(visible_servers, reverse=True):
                        if server <= current_server and server >= end_server:
                            if self.is_server_available(server):
                                self.logger.info(f"Выбираем доступный сервер {server} вместо {current_server}")
                                return server

            # Переход к следующему серверу в диапазоне
            current_server -= 1

        self.logger.warning(f"Не найдено доступных серверов в диапазоне от {start_server} до {end_server}")
        return None