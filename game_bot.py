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

    def find_skip_button(self, max_attempts=None, timeout=None):
        """
        Поиск и клик по кнопке "ПРОПУСТИТЬ".
        Использует комбинацию методов OCR и обработки изображения для
        надёжного распознавания на разных фонах.

        Args:
            max_attempts: максимальное количество попыток, None - бесконечные попытки
            timeout: таймаут между попытками в секундах

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

            # МЕТОД 8: Поиск по цветовым характеристикам
            # (Зеленый текст "ПРОПУСТИТЬ" часто имеет определенный диапазон HSV)
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

            # Шаг 23: Ждем надпись "Сбор припасов" и начинаем плыть на корабле (ОБНОВЛЕНО В НОВОМ ТЗ)
            self.logger.info('Шаг 23: Ждем надписи "Сбор припасов" и нажимаем на координаты (741, 145)')
            found = False
            for _ in range(20):  # 20 попыток с интервалом 1 секунду
                if self.find_text_on_screen("Сбор припасов", region=(0, 200, 250, 170), timeout=1):
                    found = True
                    break
                time.sleep(1)

            if found:
                self.click_coord(741, 145)
            else:
                self.logger.warning('Надпись "Сбор припасов" не найдена, продолжаем выполнение')
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
            self.logger.info(
                'Шаг 50: Клик по координатам (93, 285) - нажимаем на квест "Отправиться в залив Мертвецов"')
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

            # Шаг 68: Ждем надписи "Заполучи кают гребцов: 1" (ОБНОВЛЕНО В НОВОМ ТЗ)
            self.logger.info('Шаг 68: Ждем надписи "Заполучи кают гребцов: 1" и нажимаем на нее')
            found = False
            for _ in range(20):  # 20 попыток с интервалом 1 секунду
                if self.find_text_on_screen("Заполучи кают гребцов: 1", region=(0, 200, 250, 170), timeout=1):
                    found = True
                    break
                time.sleep(1)

            if found:
                self.click_coord(89, 280)
            else:
                self.logger.warning('Надпись "Заполучи кают гребцов: 1" не найдена, продолжаем выполнение')
                self.click_coord(89, 280)

            # Шаг 69: УДАЛЕН В НОВОМ ТЗ

            # Шаг 70: Нажимаем на иконку молоточка чтобы что-то построить
            self.logger.info("Шаг 70: Клик по координатам (43, 481) - нажимаем на иконку молоточка")
            self.click_coord(43, 481)

            # Шаг 71: Выбираем каюту гребцов
            self.logger.info("Шаг 71: Клик по координатам (968, 507) - выбираем каюту гребцов")
            self.click_coord(968, 507)

            # Шаг 72: Подтверждаем постройку каюты гребцов
            self.logger.info("Шаг 72: Клик по координатам (676, 580) - подтверждаем постройку")
            self.click_coord(676, 580)

            # Шаг 73: УДАЛЕН В НОВОМ ТЗ

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

            # Шаг 79: УДАЛЕН В НОВОМ ТЗ

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

        # Строка self.start_game() удалена - пользователь запускает игру сам

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