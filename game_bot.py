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
        # Удаляем задержку time.sleep(DEFAULT_TIMEOUT)

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
        # Удаляем задержку time.sleep(DEFAULT_TIMEOUT)

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

    def find_text_on_screen(self, text, region=None, timeout=None,
                            invert=False, adaptive=False, morph=False):
        """
        Поиск текста на экране с использованием OCR с улучшенным распознаванием.

        Args:
            text: искомый текст
            region: область поиска (x, y, w, h) или None для всего экрана
            timeout: максимальное время ожидания в секундах, None - бесконечное ожидание
            invert: инвертировать изображение перед распознаванием
            adaptive: использовать адаптивную бинаризацию
            morph: применять морфологические операции для улучшения распознавания

        Returns:
            tuple: (x, y, w, h) координаты и размеры найденного текста или None
        """
        if not self.ocr_available:
            self.logger.warning("OCR не доступен, невозможно найти текст на экране")
            return None

        self.logger.debug(f"Поиск текста '{text}' на экране")
        start_time = time.time()
        attempt = 0

        while timeout is None or time.time() - start_time < timeout:
            attempt += 1
            if attempt % 10 == 0:
                elapsed_time = int(time.time() - start_time)
                self.logger.info(f"Ожидание текста '{text}' продолжается {elapsed_time} секунд...")

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
                roi = gray[y:y + h, x:x + w]
            else:
                roi = gray
                x, y = 0, 0

            # Список обработанных изображений для OCR
            processed_images = []

            # Гистограммное выравнивание
            equalized = cv2.equalizeHist(roi)
            processed_images.append(equalized)

            # Стандартная бинаризация
            _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
            processed_images.append(binary)

            # Стандартная бинаризация на выровненном изображении
            _, binary_eq = cv2.threshold(equalized, 150, 255, cv2.THRESH_BINARY_INV)
            processed_images.append(binary_eq)

            # Инвертированная бинаризация
            if invert:
                _, binary_inv = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
                processed_images.append(binary_inv)

                # Инвертированная бинаризация на выровненном изображении
                _, binary_eq_inv = cv2.threshold(equalized, 150, 255, cv2.THRESH_BINARY)
                processed_images.append(binary_eq_inv)

            # Адаптивная бинаризация
            if adaptive:
                binary_adaptive = cv2.adaptiveThreshold(
                    roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                processed_images.append(binary_adaptive)

                # Инвертированная адаптивная бинаризация
                binary_adaptive_inv = cv2.adaptiveThreshold(
                    roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                processed_images.append(binary_adaptive_inv)

            # Морфологические операции
            if morph:
                kernel = np.ones((2, 2), np.uint8)

                # Эрозия (утоньшение) - полезно для разделения склеенных символов
                eroded = cv2.erode(binary, kernel, iterations=1)
                processed_images.append(eroded)

                # Дилатация (утолщение) - полезно для соединения разорванных символов
                dilated = cv2.dilate(binary, kernel, iterations=1)
                processed_images.append(dilated)

                # Открытие (удаление шума)
                opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                processed_images.append(opened)

                # Закрытие (заполнение мелких отверстий)
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                processed_images.append(closed)

            try:
                import pytesseract

                # Пробуем распознать текст на всех обработанных изображениях
                for img_index, img in enumerate(processed_images):
                    result = pytesseract.image_to_string(img, lang='rus+eng')

                    # Дополнительная обработка результата для повышения шансов совпадения
                    result_lower = result.lower()
                    text_lower = text.lower()

                    # Проверка на точное соответствие
                    if text_lower in result_lower:
                        self.logger.info(f"Текст '{text}' найден на экране (метод {img_index}) после {attempt} попыток")

                        # Для простоты возвращаем центр области
                        if region:
                            return (region[0] + region[2] // 2, region[1] + region[3] // 2, region[2], region[3])
                        else:
                            h, w = screenshot.shape[:2]
                            return (w // 2, h // 2, w, h)

                    # Проверка на нечеткое соответствие (например, для буквы "О" вместо "0")
                    # Используем расстояние Левенштейна для определения схожести строк
                    elif text_lower.replace('о', '0') in result_lower or text_lower.replace('0', 'о') in result_lower:
                        self.logger.info(
                            f"Текст '{text}' найден с заменой О/0 (метод {img_index}) после {attempt} попыток")

                        if region:
                            return (region[0] + region[2] // 2, region[1] + region[3] // 2, region[2], region[3])
                        else:
                            h, w = screenshot.shape[:2]
                            return (w // 2, h // 2, w, h)

                    # Проверка на схожесть слов (для пропустить/просутить/пр0пустить и т.д.)
                    elif self._is_text_similar(text_lower, result_lower, threshold=0.7):
                        self.logger.info(
                            f"Текст похожий на '{text}' найден (метод {img_index}) после {attempt} попыток")

                        if region:
                            return (region[0] + region[2] // 2, region[1] + region[3] // 2, region[2], region[3])
                        else:
                            h, w = screenshot.shape[:2]
                            return (w // 2, h // 2, w, h)

            except Exception as e:
                self.logger.error(f"Ошибка при распознавании текста: {e}")

            time.sleep(0.5)

        # Если текст не найден за отведенное время
        self.logger.warning(f"Текст '{text}' не найден на экране за {timeout} сек")
        return None

    def _is_text_similar(self, text1, text2, threshold=0.7):
        """
        Проверка схожести двух текстов с использованием расстояния Левенштейна.

        Args:
            text1: первый текст
            text2: второй текст
            threshold: порог схожести (0-1), где 1 - полное совпадение

        Returns:
            bool: True если тексты достаточно похожи
        """
        try:
            import Levenshtein
            # Ищем text1 в text2
            for word in text2.split():
                distance = Levenshtein.ratio(text1, word)
                if distance >= threshold:
                    return True
            return False
        except ImportError:
            # Если библиотека Levenshtein не доступна, используем простое сравнение
            return text1 in text2

    def wait_for_text(self, text, region=None, timeout=None):
        """
        Ожидание появления текста на экране.

        Args:
            text: ожидаемый текст
            region: область поиска (x, y, w, h) или None для всего экрана
            timeout: максимальное время ожидания в секундах, None - бесконечное ожидание

        Returns:
            tuple: (x, y, w, h) координаты и размеры найденного текста или None
        """
        self.logger.info(f"Ожидание появления текста '{text}' на экране")
        return self.find_text_on_screen(text, region, timeout)

    def find_and_click_text(self, text, region=None, timeout=None,
                            invert=False, adaptive=False, morph=False):
        """
        Поиск текста на экране и клик по нему с улучшенными опциями распознавания.

        Args:
            text: искомый текст
            region: область поиска (x, y, w, h) или None для всего экрана
            timeout: максимальное время ожидания в секундах, None - бесконечное ожидание
            invert: инвертировать изображение перед распознаванием
            adaptive: использовать адаптивную бинаризацию
            morph: применять морфологические операции для улучшения распознавания

        Returns:
            bool: True если текст найден и клик выполнен, False иначе
        """
        self.logger.info(f"Поиск и клик по тексту '{text}' на экране")
        result = self.find_text_on_screen(text, region, timeout, invert, adaptive, morph)

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
        # Удаляем задержку time.sleep(DEFAULT_TIMEOUT)

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

    def find_skip_button(self, max_attempts=None, timeout=None, use_color_analysis=True):
        """
        Поиск и клик по кнопке "ПРОПУСТИТЬ".
        Использует комбинацию методов OCR и обработки изображения для
        надёжного распознавания на разных фонах.

        Args:
            max_attempts: максимальное количество попыток, None - бесконечные попытки
            timeout: таймаут между попытками в секундах
            use_color_analysis: использовать ли анализ по цвету (для первых шагов после загрузки отключаем)

        Returns:
            bool: True если кнопка найдена и нажата, False иначе
        """
        self.logger.info("Поиск кнопки ПРОПУСТИТЬ")

        # Координаты области, где обычно находится кнопка ПРОПУСТИТЬ
        region = (1040, 12, 200, 60)
        skip_position = (1142, 42)  # Позиция, где обычно находится кнопка ПРОПУСТИТЬ

        attempt = 0
        while max_attempts is None or attempt < max_attempts:
            attempt += 1
            if attempt % 10 == 0:  # Логируем каждые 10 попыток
                self.logger.info(f"Попытка {attempt} найти кнопку ПРОПУСТИТЬ")

            # Получение скриншота
            screenshot = self.adb.screenshot()

            if screenshot is None:
                time.sleep(0.5)
                continue

            # Вырезаем область интереса
            if region:
                x, y, w, h = region
                roi = screenshot[y:y + h, x:x + w]
            else:
                roi = screenshot

            # Преобразование в оттенки серого
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Различные методы обработки для улучшения OCR
            try:
                import pytesseract

                # МЕТОД 1: Стандартная бинаризация
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                result1 = pytesseract.image_to_string(binary, lang='rus+eng')

                # МЕТОД 2: Инвертированная бинаризация
                _, binary_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                result2 = pytesseract.image_to_string(binary_inv, lang='rus+eng')

                # МЕТОД 3: Адаптивная бинаризация
                binary_adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                result3 = pytesseract.image_to_string(binary_adaptive, lang='rus+eng')

                # МЕТОД 4: Гистограммное выравнивание
                equalized = cv2.equalizeHist(gray)
                _, binary_eq = cv2.threshold(equalized, 150, 255, cv2.THRESH_BINARY_INV)
                result4 = pytesseract.image_to_string(binary_eq, lang='rus+eng')

                # МЕТОД 5: Морфологические операции для удаления шума
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                result5 = pytesseract.image_to_string(morphed, lang='rus+eng')

                # МЕТОД 6: Увеличение масштаба (для улучшения распознавания мелкого текста)
                resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                _, binary_resized = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY_INV)
                result6 = pytesseract.image_to_string(binary_resized, lang='rus+eng')

                # МЕТОД 7: Повышение контраста
                alpha = 1.5  # Коэффициент контраста
                adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)
                _, binary_adjusted = cv2.threshold(adjusted, 150, 255, cv2.THRESH_BINARY_INV)
                result7 = pytesseract.image_to_string(binary_adjusted, lang='rus+eng')

                # Объединяем результаты всех методов
                results = [result1, result2, result3, result4, result5, result6, result7]

                # Проверяем каждый результат на наличие слова "ПРОПУСТИТЬ"
                # Используем различные варианты написания для учета потенциальных ошибок распознавания
                skip_variants = ["ПРОПУСТИТЬ", "ПРОПУСТИTЬ", "ПРОNYСТИТЬ", "ПPОПУСТИТЬ",
                                 "ПРОПУCTИТЬ", "ПРOПУСТИТЬ", "ПPOПУСТИТЬ", "ПPOПУCTИTЬ",
                                 "SKIP", "ПРОПУСТИТ", "ПPОПУCTИTb", "ПРОПУCТИТЬ"]

                for i, result in enumerate(results):
                    result_upper = result.upper()
                    for variant in skip_variants:
                        if variant in result_upper:
                            self.logger.info(f"Слово '{variant}' найдено методом {i + 1}")
                            self.click_coord(skip_position[0], skip_position[1])
                            return True

                    # Проверка на нечеткое соответствие для еще большей надежности
                    # С порогом менее строгим, чтобы учесть больше вариаций
                    if self._is_text_similar("ПРОПУСТИТЬ", result_upper, threshold=0.6):
                        self.logger.info(f"Текст, похожий на 'ПРОПУСТИТЬ', найден методом {i + 1}")
                        self.click_coord(skip_position[0], skip_position[1])
                        return True

            except Exception as e:
                self.logger.error(f"Ошибка при распознавании текста: {e}")

            # МЕТОД 8: Поиск по цветовым характеристикам (применяем только если разрешено)
            # (Зеленый текст "ПРОПУСТИТЬ" часто имеет определенный диапазон HSV)
            if use_color_analysis:
                try:
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    # Определение диапазонов HSV для зеленого текста (настроить по вашим изображениям)
                    lower_green = np.array([40, 50, 50])
                    upper_green = np.array([80, 255, 255])

                    # Создание маски для зеленого текста
                    green_mask = cv2.inRange(hsv, lower_green, upper_green)

                    # Вычисление общего числа пикселей в маске и процента зеленых пикселей
                    total_pixels = roi.shape[0] * roi.shape[1]
                    green_pixels = np.sum(green_mask > 0)
                    green_percent = green_pixels / total_pixels

                    # Проверка характерного расположения зеленых пикселей для кнопки ПРОПУСТИТЬ
                    # (только если достаточное количество зеленых пикселей и они расположены
                    # в правой части области, где обычно находится кнопка)
                    if green_percent > 0.05 and green_percent < 0.3:
                        # Вычисление центра масс зеленых пикселей
                        M = cv2.moments(green_mask)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            # Проверка, что центр масс в правой части области
                            if cx > w * 0.5:
                                self.logger.info("Кнопка ПРОПУСТИТЬ найдена по цветовому анализу")
                                self.click_coord(skip_position[0], skip_position[1])
                                return True

                except Exception as e:
                    self.logger.error(f"Ошибка при поиске по цвету: {e}")

            time.sleep(0.5 if timeout is None else timeout)

        # Эта часть кода выполнится только если задан max_attempts
        self.logger.warning(f"Не удалось найти кнопку ПРОПУСТИТЬ за {max_attempts} попыток")
        return False

    def wait_for_image_in_region(self, image_key, region=None, timeout=10, threshold=TEMPLATE_MATCHING_THRESHOLD):
        """
        Ожидание появления изображения в указанной области экрана.

        Args:
            image_key: ключ изображения в словаре IMAGE_PATHS
            region: область поиска (x, y, w, h) или None для всего экрана
            timeout: максимальное время ожидания в секундах
            threshold: порог соответствия (0-1)

        Returns:
            tuple: (x, y, w, h) координаты и размеры найденного изображения или None
        """
        self.logger.debug(f"Ожидание изображения '{image_key}' в области {region} с таймаутом {timeout} сек")

        if image_key not in IMAGE_PATHS:
            self.logger.error(f"Изображение с ключом '{image_key}' не найдено в конфигурации")
            return None

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Получение скриншота
            screenshot = self.adb.screenshot()

            if screenshot is None or screenshot.size == 0:
                self.logger.warning("Получен пустой скриншот")
                time.sleep(0.5)
                continue

            # Вырезание области, если указана
            if region:
                x, y, w, h = region
                roi = screenshot[y:y + h, x:x + w]
            else:
                roi = screenshot
                x, y = 0, 0

            # Поиск шаблона в выбранной области
            template_path = IMAGE_PATHS[image_key]
            result = self.image.find_template(roi, template_path, threshold)

            if result:
                center_x, center_y, found_w, found_h = result
                # Корректируем координаты с учетом области поиска
                abs_center_x = x + center_x
                abs_center_y = y + center_y
                self.logger.info(f"Изображение '{image_key}' найдено по координатам: ({abs_center_x}, {abs_center_y})")
                return (abs_center_x, abs_center_y, found_w, found_h)

            time.sleep(0.5)

        self.logger.warning(f"Изображение '{image_key}' не найдено за {timeout} сек")
        return None

    def click_image_in_region(self, image_key, region=None, timeout=10, threshold=TEMPLATE_MATCHING_THRESHOLD):
        """
        Поиск изображения в указанной области и клик по нему.

        Args:
            image_key: ключ изображения в словаре IMAGE_PATHS
            region: область поиска (x, y, w, h) или None для всего экрана
            timeout: максимальное время ожидания в секундах
            threshold: порог соответствия (0-1)

        Returns:
            bool: True если изображение найдено и клик выполнен, False иначе
        """
        self.logger.info(f"Поиск и клик по изображению '{image_key}' в области {region}")

        result = self.wait_for_image_in_region(image_key, region, timeout, threshold)

        if result:
            x, y, _, _ = result
            self.click_coord(x, y)
            return True

        return False

    def execute_step(self, step_number):
        """
        Выполнение отдельного шага обучения.

        Args:
            step_number: номер шага

        Returns:
            bool: True если шаг выполнен успешно, False иначе
        """
        self.logger.info(f"Выполнение шага {step_number}")

        try:
            # Шаг 1: Открываем профиль
            if step_number == 1:
                self.logger.info("Шаг 1: Клик по координатам (52, 50) - открываем профиль")
                self.click_coord(52, 50)
                return True

            # Шаг 2: Открываем настройки
            elif step_number == 2:
                self.logger.info("Шаг 2: Клик по координатам (1076, 31) - открываем настройки")
                self.click_coord(1076, 31)
                return True

            # Шаг 3: Открываем вкладку персонажей
            elif step_number == 3:
                self.logger.info("Шаг 3: Клик по координатам (643, 319) - открываем вкладку персонажей")
                self.click_coord(643, 319)
                return True

            # Шаг 4: Создаем персонажа на новом сервере
            elif step_number == 4:
                self.logger.info("Шаг 4: Клик по координатам (271, 181) - создаем персонажа на новом сервере")
                self.click_coord(271, 181)
                return True

            # Шаг 5: Выбор сервера - этот шаг обрабатывается отдельно в perform_tutorial
            elif step_number == 5:
                self.logger.info("Шаг 5: Выбор сервера - будет выполнен в методе perform_tutorial")
                return True

            # Шаг 6: Подтверждаем создание персонажа
            elif step_number == 6:
                self.logger.info("Шаг 6: Клик по координатам (787, 499) - подтверждаем создание персонажа")
                self.click_coord(787, 499)
                time.sleep(LOADING_TIMEOUT)  # Ожидание загрузки
                return True

            # Остальные шаги выполняются в execute_remaining_steps
            elif step_number >= 7:
                return self.execute_step_range(7, step_number)

            else:
                self.logger.error(f"Неизвестный шаг: {step_number}")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка при выполнении шага {step_number}: {e}", exc_info=True)
            return False

    def execute_step_range(self, start_step, end_step):
        """
        Выполнение диапазона шагов обучения.

        Args:
            start_step: начальный шаг
            end_step: конечный шаг

        Returns:
            bool: True если все шаги выполнены успешно, False иначе
        """
        self.logger.info(f"Выполнение шагов с {start_step} по {end_step}")

        # Для шагов с 7 по конец мы вызываем execute_remaining_steps с кастомной логикой
        if start_step == 7 and end_step >= 7:
            # Для шагов с 7 - мы фактически выполняем оставшиеся шаги обучения,
            # но нужно добавить проверку для начала выполнения с определенного шага
            return self.execute_remaining_steps(start_from_step=start_step)

        # Если диапазон не начинается с 7, выполняем шаги по одному
        for step in range(start_step, end_step + 1):
            if not self.execute_step(step):
                self.logger.error(f"Ошибка при выполнении шага {step}")
                return False

        return True

    def perform_tutorial(self, server_id, start_step=1):
        """
        Выполнение полного цикла обучения согласно новому ТЗ.

        Args:
            server_id: номер сервера для создания персонажа
            start_step: начальный шаг обучения (по умолчанию 1)

        Returns:
            bool: True если обучение успешно завершено, False иначе
        """
        self.logger.info(f"Начало выполнения обучения на сервере {server_id} с шага {start_step}")

        try:
            # Если начинаем с первого шага, выполняем все по порядку
            if start_step == 1:
                # Шаг 1: Открываем профиль (первый шаг без задержки по ТЗ)
                self.logger.info("Шаг 1: Клик по координатам (52, 50) - открываем профиль")
                # Нет задержки для первого шага согласно ТЗ
                self.adb.execute_adb_command('shell', 'input', 'tap', str(52), str(50))

                # Шаг 2: Открываем настройки - добавляем задержку до действия
                self.logger.info("Шаг 2: Ждем 1.5 сек и открываем настройки")
                time.sleep(1.5)  # Обязательная задержка перед шагом 2
                self.adb.execute_adb_command('shell', 'input', 'tap', str(1076), str(31))

                # Шаг 3: Открываем вкладку персонажей - добавляем задержку до действия
                self.logger.info("Шаг 3: Ждем 1.5 сек и открываем вкладку персонажей")
                time.sleep(1.5)  # Обязательная задержка перед шагом 3
                self.adb.execute_adb_command('shell', 'input', 'tap', str(643), str(319))

                # Шаг 4: Создаем персонажа на новом сервере - добавляем задержку до действия
                self.logger.info("Шаг 4: Ждем 1.5 сек и создаем персонажа на новом сервере")
                time.sleep(1.5)  # Обязательная задержка перед шагом 4
                self.adb.execute_adb_command('shell', 'input', 'tap', str(271), str(181))

                # Шаг 5: Выбор сервера
                self.logger.info(f"Шаг 5: Выбор сервера {server_id}")
                if not self.select_server(server_id):
                    self.logger.error(f"Не удалось выбрать сервер {server_id}")
                    return False

                # Шаг 6: Подтверждаем создание персонажа
                self.logger.info("Шаг 6: Ждем 1.5 сек и подтверждаем создание персонажа")
                time.sleep(1.5)  # Обязательная задержка перед шагом 6
                self.adb.execute_adb_command('shell', 'input', 'tap', str(787), str(499))

                # Добавляем дополнительную задержку в 12 секунд после шага 6 для загрузки
                self.logger.info("Ожидание 12 секунд после подтверждения создания персонажа...")
                time.sleep(12)

                # Продолжение обучения
                if not self.execute_remaining_steps():
                    self.logger.error("Ошибка при выполнении оставшихся шагов обучения")
                    return False

            # Если начинаем с шага 2-6, выполняем только шаги из диапазона [start_step, 6],
            # а затем выполняем оставшиеся шаги
            elif 2 <= start_step <= 6:
                # Выполняем шаги с start_step по 6
                for step in range(start_step, 7):
                    if step == 2:
                        # Шаг 2: Открываем настройки
                        self.logger.info("Шаг 2: Ждем 1.5 сек и открываем настройки")
                        time.sleep(1.5)  # Обязательная задержка перед шагом 2
                        self.adb.execute_adb_command('shell', 'input', 'tap', str(1076), str(31))

                    elif step == 3:
                        # Шаг 3: Открываем вкладку персонажей
                        self.logger.info("Шаг 3: Ждем 1.5 сек и открываем вкладку персонажей")
                        time.sleep(1.5)  # Обязательная задержка перед шагом 3
                        self.adb.execute_adb_command('shell', 'input', 'tap', str(643), str(319))

                    elif step == 4:
                        # Шаг 4: Создаем персонажа на новом сервере
                        self.logger.info("Шаг 4: Ждем 1.5 сек и создаем персонажа на новом сервере")
                        time.sleep(1.5)  # Обязательная задержка перед шагом 4
                        self.adb.execute_adb_command('shell', 'input', 'tap', str(271), str(181))

                    elif step == 5:
                        # Шаг 5: Выбор сервера
                        self.logger.info(f"Шаг 5: Выбор сервера {server_id}")
                        if not self.select_server(server_id):
                            self.logger.error(f"Не удалось выбрать сервер {server_id}")
                            return False

                    elif step == 6:
                        # Шаг 6: Подтверждаем создание персонажа
                        self.logger.info("Шаг 6: Ждем 1.5 сек и подтверждаем создание персонажа")
                        time.sleep(1.5)  # Обязательная задержка перед шагом 6
                        self.adb.execute_adb_command('shell', 'input', 'tap', str(787), str(499))

                        # Добавляем дополнительную задержку в 12 секунд после шага 6 для загрузки
                        self.logger.info("Ожидание 17 секунд после подтверждения создания персонажа...")
                        time.sleep(17)

                # Продолжение обучения с шага 7
                if not self.execute_remaining_steps():
                    self.logger.error("Ошибка при выполнении оставшихся шагов обучения")
                    return False

            # Если начинаем с шага 7 или выше, выполняем только оставшиеся шаги
            elif start_step >= 7:
                # Для шага 5 всегда нужно выбрать сервер, чтобы функция знала, на каком сервере работать
                self.logger.info(f"Шаг 5: Выбор сервера {server_id} (виртуально, без клика)")

                # Продолжение обучения с указанного шага
                if not self.execute_remaining_steps(start_from_step=start_step):
                    self.logger.error(f"Ошибка при выполнении шагов, начиная с {start_step}")
                    return False

            else:
                self.logger.error(f"Некорректный начальный шаг: {start_step}")
                return False

            self.logger.info(f"Обучение на сервере {server_id} успешно завершено")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка при выполнении обучения: {e}", exc_info=True)
            return False

    def execute_remaining_steps(self, start_from_step=7):
        """
        Выполнение оставшихся шагов обучения согласно новому ТЗ.

        Args:
            start_from_step: начальный шаг для выполнения (по умолчанию 7)

        Returns:
            bool: True если шаги выполнены успешно, False иначе
        """
        self.logger.info(f"Выполнение оставшихся шагов обучения, начиная с шага {start_from_step}")

        try:
            # Шаг 7: Ищем слово ПРОПУСТИТЬ и кликаем на него (без цветового анализа)
            if start_from_step <= 7:
                self.logger.info("Шаг 7: Ищем и нажимаем ПРОПУСТИТЬ (только OCR)")
                # Отключаем цветовой анализ для шага 7
                self.find_skip_button(use_color_analysis=False)
                time.sleep(1)

            # Шаг 8: Ищем слово ПРОПУСТИТЬ и кликаем на него (без цветового анализа)
            if start_from_step <= 8:
                self.logger.info("Шаг 8: Ищем и нажимаем ПРОПУСТИТЬ (только OCR)")
                # Отключаем цветовой анализ для шага 8
                self.find_skip_button(use_color_analysis=False)
                time.sleep(1)

            # Шаг 9: Ищем слово ПРОПУСТИТЬ и кликаем на него (здесь уже можно использовать цветовой анализ)
            if start_from_step <= 9:
                self.logger.info("Шаг 9: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(1)

            # Шаг 10: Активируем бой нажатием на пушку
            if start_from_step <= 10:
                self.logger.info("Шаг 10: Ищем изображение cannon_is_ready.png и нажимаем на координаты (718, 438)")
                if self.wait_for_image('cannon_is_ready', timeout=20):
                    self.click_coord(718, 438)
                else:
                    self.logger.warning("Изображение cannon_is_ready.png не найдено, выполняем клик по координатам")
                    self.click_coord(718, 438)

            # Шаг 11: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 11:
                self.logger.info("Шаг 11: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 12: Ждем появления hell_henry.png и нажимаем ПРОПУСТИТЬ
            if start_from_step <= 12:
                self.logger.info('Шаг 12: Ждем появления изображения hell_henry.png и нажимаем ПРОПУСТИТЬ')
                if self.wait_for_image('hell_henry', timeout=15):
                    self.find_skip_button()
                else:
                    self.logger.warning("Изображение hell_henry.png не найдено, продолжаем выполнение")

            # Шаг 13: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 13:
                self.logger.info("Шаг 13: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 14: Жмем на иконку кораблика
            if start_from_step <= 14:
                self.logger.info("Шаг 14: Ждем 3 секунды и нажимаем (58, 654) - жмем на иконку кораблика")
                time.sleep(3)
                self.click_coord(58, 654)

            # Шаг 15: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 15:
                self.logger.info("Шаг 15: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 16: Отстраиваем нижнюю палубу
            if start_from_step <= 16:
                self.logger.info("Шаг 16: Ждем 2 секунды и нажимаем (638, 403) - отстраиваем нижнюю палубу")
                time.sleep(2)
                self.click_coord(638, 403)

            # Шаг 17: Отстраиваем паб в нижней палубе
            if start_from_step <= 17:
                self.logger.info("Шаг 17: Ждем 2.5 секунды и нажимаем (635, 373) - отстраиваем паб в нижней палубе")
                time.sleep(2.5)
                self.click_coord(635, 373)

            # Шаг 18: Латаем дыры в складе на нижней палубе
            if start_from_step <= 18:
                self.logger.info(
                    "Шаг 18: Ждем 2.5 секунды и нажимаем (635, 373) - латаем дыры в складе на нижней палубе")
                time.sleep(2.5)
                self.click_coord(635, 373)

            # Шаг 19: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 19:
                self.logger.info("Шаг 19: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 20: Отстраиваем верхнюю палубу
            if start_from_step <= 20:
                self.logger.info("Шаг 20: Ждем 2.5 секунды и нажимаем (345, 386) - отстраиваем верхнюю палубу")
                time.sleep(2.5)
                self.click_coord(345, 386)

            # Шаг 21: Выбираем пушку
            if start_from_step <= 21:
                self.logger.info("Шаг 21: Ждем 1.5 секунды и нажимаем (77, 276) - выбираем пушку")
                time.sleep(1.5)
                self.click_coord(77, 276)

            # Шаг 22: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 22:
                self.logger.info("Шаг 22: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 23: Ищем collect_items.png и нажимаем на координаты (741, 145)
            if start_from_step <= 23:
                self.logger.info('Шаг 23: Ищем изображение collect_items.png и нажимаем на координаты (741, 145)')
                if self.wait_for_image('collect_items', timeout=15):
                    self.click_coord(741, 145)
                else:
                    self.logger.warning("Изображение collect_items.png не найдено, выполняем клик по координатам")
                    self.click_coord(741, 145)

            # Шаг 24: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 24:
                self.logger.info("Шаг 24: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(0.5)

            # Шаг 25: Нажимаем на квест "Старый соперник"
            if start_from_step <= 25:
                self.logger.info(
                    'Шаг 25: Ждем 1.5 секунды и нажимаем по координатам (93, 285) - квест "Старый соперник"')
                time.sleep(1.5)
                self.click_coord(93, 285)

            # Шаг 26: Ищем слово ПРОПУСТИТЬ и кликаем на него (с задержкой 8 сек без поиска текста)
            if start_from_step <= 26:
                self.logger.info("Шаг 26: Ждем 8 секунд и нажимаем по координатам (1142, 42)")
                time.sleep(8)
                self.click_coord(1142, 42)

            # Шаг 27: Нажимаем на квест "Старый соперник"
            if start_from_step <= 27:
                self.logger.info(
                    'Шаг 27: Ждем 1.5 секунды и нажимаем на координаты (93, 285) - квест "Старый соперник"')
                time.sleep(1.5)
                self.click_coord(93, 285)

            # Шаг 28: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 28:
                self.logger.info("Шаг 28: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(1)

            # Шаг 29: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 29:
                self.logger.info("Шаг 29: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(3)

            # Шаг 30: Жмем на любую часть экрана чтобы продолжить после победы
            if start_from_step <= 30:
                self.logger.info("Шаг 30: Клик по координатам (630, 413) - жмем на любую часть экрана")
                self.click_coord(630, 413)

            # Шаг 31: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 31:
                self.logger.info("Шаг 31: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()
                time.sleep(0.5)

            # Шаг 32: Ждем появления gold_compas.png и жмем на компас
            if start_from_step <= 32:
                self.logger.info("Шаг 32: Ожидаем изображение gold_compas.png и нажимаем (1074, 88) - жмем на компас")
                if self.wait_for_image('gold_compas', timeout=15):
                    time.sleep(0.5)
                    self.click_coord(1074, 88)
                else:
                    self.logger.warning("Изображение gold_compas.png не найдено, выполняем клик по координатам")
                    time.sleep(0.5)
                    self.click_coord(1074, 88)

            # Шаг 33: Еще раз жмем на компас
            if start_from_step <= 33:
                self.logger.info("Шаг 33: Клик по координатам (701, 258) - еще раз жмем на компас")
                time.sleep(1.5)
                self.click_coord(701, 258)

            # Шаг 34: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 34:
                self.logger.info("Шаг 34: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 35: Жмем назад чтобы выйти из вкладки компаса
            if start_from_step <= 35:
                self.logger.info("Шаг 35: Клик по координатам (145, 25) - жмем назад")
                self.click_coord(145, 25)

            # Шаг 36: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 36:
                self.logger.info("Шаг 36: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 37: Ожидаем изображение long_song.png и нажимаем на квест "Далекая песня"
            if start_from_step <= 37:
                self.logger.info('Шаг 37: Ожидаем изображение long_song.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 38: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 38:
                self.logger.info("Шаг 38: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 39: Ожидаем изображение long_song.png и еще раз нажимаем на квест "Далекая песня"
            if start_from_step <= 39:
                self.logger.info('Шаг 39: Ожидаем изображение long_song.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 40: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 40:
                self.logger.info("Шаг 40: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 41: Ожидаем изображение confirm_trade.png и нажимаем "Согласиться на обмен"
            if start_from_step <= 41:
                self.logger.info('Шаг 41: Ожидаем изображение confirm_trade.png и нажимаем на координаты (151, 349)')
                if self.wait_for_image('confirm_trade', timeout=15):
                    self.click_coord(151, 349)
                else:
                    self.logger.warning("Изображение confirm_trade.png не найдено, выполняем клик по координатам")
                    self.click_coord(151, 349)

            # Шаг 42: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 42:
                self.logger.info("Шаг 42: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 43: Ожидаем изображение long_song.png и нажимаем на квест "Исследовать залив Мертвецов"
            if start_from_step <= 43:
                self.logger.info('Шаг 43: Ожидаем изображение long_song.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 44: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 44:
                self.logger.info("Шаг 44: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 45: Ожидаем изображение prepare_for_battle.png и нажимаем на координаты (85, 634)
            if start_from_step <= 45:
                self.logger.info(
                    'Шаг 45: Ожидаем изображение prepare_for_battle.png и нажимаем на координаты (85, 634)')
                if self.wait_for_image('prepare_for_battle', timeout=15):
                    self.click_coord(85, 634)
                else:
                    self.logger.warning("Изображение prepare_for_battle.png не найдено, выполняем клик по координатам")
                    self.click_coord(85, 634)

            # Шаг 46: Нажимаем начать битву
            if start_from_step <= 46:
                self.logger.info("Шаг 46: Ждем 0.5 секунды и нажимаем на координаты (1157, 604)")
                time.sleep(0.5)
                self.click_coord(1157, 604)
                time.sleep(3)

            # Шаг 47: Нажимаем каждые 1.5 секунды по координатам, пока не появится start_battle.png
            if start_from_step <= 47:
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
            if start_from_step <= 48:
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
            if start_from_step <= 49:
                self.logger.info("Шаг 49: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 50: Ожидаем изображение long_song_2.png и нажимаем на квест
            if start_from_step <= 50:
                self.logger.info('Шаг 50: Ожидаем изображение long_song_2.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song_2', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song_2.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 51: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 51:
                self.logger.info("Шаг 51: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 52: Нажимаем на череп в заливе мертвецов
            if start_from_step <= 52:
                self.logger.info("Шаг 52: Ждем 2 секунды и нажимаем на координаты (653, 403) - череп")
                time.sleep(2)
                self.click_coord(653, 403)

            # Шаг 53-57: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(53, 58):
                if start_from_step <= step:
                    self.logger.info(f"Шаг {step}: Ищем и нажимаем ПРОПУСТИТЬ")
                    self.find_skip_button()
                    # Добавляем задержку 0.5 секунд для шагов с 53 по 56 включительно
                    if 53 <= step <= 56:
                        time.sleep(0.5)  # Задержка для избежания двойного нажатия между анимациями

            # Шаг 58: Ожидаем изображение long_song_3.png и нажимаем на квест
            if start_from_step <= 58:
                self.logger.info('Шаг 58: Ожидаем изображение long_song_3.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song_3', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song_3.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 59: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 59:
                self.logger.info("Шаг 59: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 60: Ожидаем изображение long_song.png и нажимаем на квест
            if start_from_step <= 60:
                self.logger.info('Шаг 60: Ожидаем изображение long_song.png и нажимаем на координаты (93, 285)')
                if self.wait_for_image('long_song', timeout=15):
                    self.click_coord(93, 285)
                else:
                    self.logger.warning("Изображение long_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(93, 285)

            # Шаг 61: Ожидаем изображение long_song_4.png и нажимаем на иконку молоточка
            if start_from_step <= 61:
                self.logger.info('Шаг 61: Ожидаем изображение long_song_4.png и нажимаем на координаты (43, 481)')
                if self.wait_for_image('long_song_4', timeout=15):
                    self.click_coord(43, 481)
                else:
                    self.logger.warning("Изображение long_song_4.png не найдено, выполняем клик по координатам")
                    self.click_coord(43, 481)

            # Шаг 62: Ожидаем изображение long_song_5.png и выбираем корабль для улучшения
            if start_from_step <= 62:
                self.logger.info('Шаг 62: Ожидаем изображение long_song_5.png и нажимаем на координаты (127, 216)')
                if self.wait_for_image('long_song_5', timeout=15):
                    self.click_coord(127, 216)
                else:
                    self.logger.warning("Изображение long_song_5.png не найдено, выполняем клик по координатам")
                    self.click_coord(127, 216)

            # Шаг 63: Дожидаемся надписи "УЛУЧШИТЬ" и кликаем на нее
            if start_from_step <= 63:
                self.logger.info('Шаг 63: Дожидаемся надписи "УЛУЧШИТЬ" и кликаем на нее')
                if not self.find_and_click_text("УЛУЧШИТЬ", region=(983, 588, 200, 100), timeout=5):
                    self.click_coord(1083, 638)  # Клик по примерным координатам, если текст не найден

            # Шаг 64: Жмем назад чтобы выйти из вкладки корабля
            if start_from_step <= 64:
                self.logger.info("Шаг 64: Ждем 2 секунды и нажимаем координаты (145, 25) - назад")
                time.sleep(3)
                self.click_coord(145, 25)

            # Шаг 65: Жмем на кнопку постройки
            if start_from_step <= 65:
                self.logger.info("Шаг 65: Ждем 1.5 секунды и нажимаем координаты (639, 603) - кнопка постройки")
                time.sleep(1.5)
                self.click_coord(639, 603)

            # Шаг 66: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 66:
                self.logger.info("Шаг 66: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 67: Жмем на иконку компаса
            if start_from_step <= 67:
                self.logger.info("Шаг 67: Ждем 1.5 секунды и нажимаем координаты (1072, 87) - иконка компаса и возвращаемся обратно")
                time.sleep(1.5)
                self.click_coord(1072, 87)


            # Шаг 68: Ожидаем изображение long_song_6.png и нажимаем на квест
            if start_from_step <= 68:
                self.logger.info('Шаг 68: Ожидаем изображение long_song_6.png и нажимаем на координаты (89, 280)')
                if self.wait_for_image('long_song_6', timeout=25):
                    self.click_coord(89, 280)
                else:
                    self.logger.warning("Изображение long_song_6.png не найдено, выполняем клик по координатам")
                    self.click_coord(89, 280)

            # Шаг 69: УДАЛЕН согласно новому ТЗ

            # Шаг 70: Нажимаем на иконку молоточка
            if start_from_step <= 70:
                self.logger.info("Шаг 70: Ждем 0.5 секунды и нажимаем координаты (43, 481) - иконка молоточка")
                time.sleep(1.5)
                self.click_coord(43, 481)

            # Шаг 71: Ожидаем изображение cannon_long.png и выбираем каюту гребцов
            if start_from_step <= 71:
                self.logger.info('Шаг 71: Ожидаем изображение cannon_long.png и нажимаем на координаты (968, 507)')
                if self.wait_for_image('cannon_long', timeout=15):
                    self.click_coord(968, 507)
                else:
                    self.logger.warning("Изображение cannon_long.png не найдено, выполняем клик по координатам")
                    self.click_coord(968, 507)

            # Шаг 72: Ожидаем изображение long_song_6.png и подтверждаем постройку
            if start_from_step <= 72:
                self.logger.info('Шаг 72: Ожидаем изображение long_song_6.png и нажимаем на координаты (676, 580)')
                if self.wait_for_image('long_song_6', timeout=15):
                    self.click_coord(676, 580)
                else:
                    self.logger.warning("Изображение long_song_6.png не найдено, выполняем клик по координатам")
                    self.click_coord(676, 580)

            # Шаг 73: УДАЛЕН согласно новому ТЗ

            # Шаг 74: Нажимаем на квест "Заполучи кают гребцов: 1"
            if start_from_step <= 74:
                self.logger.info('Шаг 74: Ждем 4 секунды и нажимаем на координаты (89, 280)')
                time.sleep(4)
                self.click_coord(89, 280)

            # Шаг 75: Нажимаем на квест "Заполучи орудийных палуб: 1"
            if start_from_step <= 75:
                self.logger.info('Шаг 75: Ждем 2.5 секунды и нажимаем на координаты (89, 280)')
                time.sleep(2.5)
                self.click_coord(89, 280)

            # Шаг 76: Нажимаем на иконку молоточка
            if start_from_step <= 76:
                self.logger.info('Шаг 76: Ждем 2.5 секунды и нажимаем на координаты (43, 481)')
                self.logger.info("Шаг 67: Ждем 1.5 секунды и нажимаем координаты (1072, 87) - иконка компаса")
                self.click_coord(43, 481)

            # Шаг 77: Ожидаем изображение cannon_long.png и выбираем орудийную палубу
            if start_from_step <= 77:
                self.logger.info('Шаг 77: Ожидаем изображение cannon_long.png и нажимаем на координаты (687, 514)')
                if self.wait_for_image('cannon_long', timeout=15):
                    self.click_coord(687, 514)
                    time.sleep(2.5)
                else:
                    self.logger.warning("Изображение cannon_long.png не найдено, выполняем клик по координатам")
                    self.click_coord(687, 514)
                    time.sleep(2.5)

            # Шаг 78: Ожидаем изображение long_song_6.png и подтверждаем постройку
            if start_from_step <= 78:
                self.logger.info('Шаг 78: Ожидаем изображение long_song_6.png и нажимаем на координаты (679, 581)')
                if self.wait_for_image('long_song_6', timeout=15):
                    self.click_coord(679, 581)
                    time.sleep(2.5)
                else:
                    self.logger.warning("Изображение long_song_6.png не найдено, выполняем клик по координатам")
                    self.click_coord(679, 581)
                    time.sleep(2.5)

            # Шаг 79: УДАЛЕН согласно новому ТЗ

            # Шаг 80: Нажимаем на квест "Заполучи орудийных палуб: 1"
            if start_from_step <= 80:
                self.logger.info('Шаг 80: Ждем 3 секунды и нажимаем на координаты (89, 280)')
                time.sleep(3)
                self.click_coord(89, 280)

            # Шаг 81: УДАЛЕН согласно новому ТЗ

            # Шаг 82: Жмем на иконку компаса
            if start_from_step <= 82:
                self.logger.info('Шаг 82: Ждем 1.5 секунды и нажимаем на координаты (1072, 87)')
                time.sleep(1.5)
                self.click_coord(1072, 87)

            # Шаг 83: Жмем на указатель на экране
            if start_from_step <= 83:
                self.logger.info('Шаг 83: Ждем 2.5 секунды и нажимаем на координаты (698, 273)')
                time.sleep(2.5)
                self.click_coord(698, 273)

            # Шаг 84: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 84:
                self.logger.info("Шаг 84: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 85: УДАЛЕН согласно новому ТЗ

            # Шаг 86: Ожидаем изображение ship_song.png и нажимаем на иконку компаса над кораблем
            if start_from_step <= 86:
                self.logger.info('Шаг 86: Ожидаем изображение ship_song.png и нажимаем на координаты (652, 214)')
                if self.wait_for_image('ship_song', timeout=15):
                    self.click_coord(652, 214)
                else:
                    self.logger.warning("Изображение ship_song.png не найдено, выполняем клик по координатам")
                    self.click_coord(652, 214)

            # Шаг 87-89: Последовательно нажимаем ПРОПУСТИТЬ
            for step in range(87, 90):
                if start_from_step <= step:
                    self.logger.info(f"Шаг {step}: Ищем и нажимаем ПРОПУСТИТЬ")
                    self.find_skip_button()
                    # Добавляем задержку 1 секунд для шагов с 87 по 88 включительно
                    if 87 <= step <= 89:
                        time.sleep(1)  # Задержка для избежания двойного нажатия между анимациями

            # Шаг 90: Нажимаем на квест "Богатая добыча"
            if start_from_step <= 90:
                self.logger.info('Шаг 90: Ждем 2 секунды и нажимаем на координаты (89, 280)')
                time.sleep(2)
                self.click_coord(89, 280)

            # Шаг 91: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 91:
                self.logger.info("Шаг 91: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 92: Ищем изображение griffin.png и нажимаем на координаты (1142, 42)
            if start_from_step <= 92:
                self.logger.info("Шаг 92: Ищем изображение griffin.png и нажимаем на координаты (1142, 42)")
                if self.wait_for_image('griffin', timeout=30):
                    self.click_coord(1142, 42)
                else:
                    self.logger.warning("Изображение griffin.png не найдено, выполняем клик по координатам")
                    self.click_coord(1142, 42)

            # Шаг 93: Ждем 7 секунд и ищем изображение molly.png
            if start_from_step <= 93:
                self.logger.info(
                    "Шаг 93: Ждем 7 секунд, ищем изображение molly.png и нажимаем на координаты (1142, 42)")
                time.sleep(7)
                if self.wait_for_image('molly', timeout=30):
                    self.click_coord(1142, 42)
                else:
                    self.logger.warning("Изображение molly.png не найдено, выполняем клик по координатам")
                    self.click_coord(1142, 42)

            # Шаг 94: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 94:
                self.logger.info("Шаг 94: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 95: Ищем картинку coins.png и нажимаем на координаты
            if start_from_step <= 95:
                self.logger.info("Шаг 95: Ищем картинку coins.png и нажимаем на координаты (931, 620)")
                if self.click_image("coins", timeout=15):
                    self.logger.info("Картинка coins.png найдена и нажата")
                else:
                    self.logger.warning("Картинка coins.png не найдена, нажимаем по координатам")
                    self.click_coord(931, 620)

            # Шаг 96: Ищем слово ПРОПУСТИТЬ и кликаем на него
            if start_from_step <= 96:
                self.logger.info("Шаг 96: Ищем и нажимаем ПРОПУСТИТЬ")
                self.find_skip_button()

            # Шаг 97: Проверяем наличие баннера ПРОПУСТИТЬ, если нет - нажимаем на квест
            if start_from_step <= 97:
                self.logger.info('Шаг 97: Ждем 6 секунд, проверяем наличие ПРОПУСТИТЬ, затем нажимаем на квест')
                time.sleep(6)

                # Проверяем наличие кнопки ПРОПУСТИТЬ
                if self.find_skip_button(max_attempts=1):
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

    def run_bot(self, cycles=1, start_server=619, end_server=1, first_server_start_step=1):
        """
        Запуск бота на выполнение заданного количества циклов обучения.

        Args:
            cycles: количество циклов обучения
            start_server: начальный сервер для прокачки
            end_server: конечный сервер для прокачки
            first_server_start_step: начальный шаг для первого сервера (по умолчанию 1)
        """
        self.logger.info(f"Запуск бота на {cycles} циклов с серверами от {start_server} до {end_server}")
        self.logger.info(f"Начальный шаг для первого сервера: {first_server_start_step}")

        # Проверка корректности диапазона серверов
        if start_server < end_server:
            self.logger.error("Начальный сервер должен быть больше или равен конечному")
            return 0

        # Строка self.start_game() удалена - пользователь запускает игру сам

        successful_cycles = 0
        current_server = start_server

        for cycle in range(1, cycles + 1):
            self.logger.info(f"===== Начало цикла {cycle}/{cycles}, сервер {current_server} =====")

            try:
                # Для первого цикла используем указанный пользователем начальный шаг
                current_step = first_server_start_step if cycle == 1 else 1

                self.logger.info(f"Начальный шаг для цикла {cycle}: {current_step}")

                if self.perform_tutorial(current_server, start_step=current_step):
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