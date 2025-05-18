"""
Модуль с конфигурационными параметрами для бота Sea of Conquest.
Обновленная версия с ИСПРАВЛЕННЫМИ координатами серверов и областью OCR.
"""
import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent.absolute()
IMAGES_DIR = BASE_DIR / 'images'

# Таймауты
DEFAULT_TIMEOUT = 3  # Минимальный таймаут между действиями
LOADING_TIMEOUT = 14   # Таймаут для ожидания загрузки
SCREENSHOT_TIMEOUT = 0.5  # Таймаут между скриншотами при поиске изображений

# Настройки игры
GAME_PACKAGE = "com.seaofconquest.global"
GAME_ACTIVITY = "com.kingsgroup.mo.KGUnityPlayerActivity"

# Пути к изображениям (согласно ТЗ, используются только две картинки плюс дополнительные)
IMAGE_PATHS = {
    'start_battle': str(IMAGES_DIR / 'start_battle.png'),
    'coins': str(IMAGES_DIR / 'coins.png'),
    'cannon_is_ready': str(IMAGES_DIR / 'cannon_is_ready.png'),
    'hell_henry': str(IMAGES_DIR / 'hell_henry.png'),
    'collect_items': str(IMAGES_DIR / 'collect_items.png'),
    'gold_compas': str(IMAGES_DIR / 'gold_compas.png'),
    'long_song': str(IMAGES_DIR / 'long_song.png'),
    'confirm_trade': str(IMAGES_DIR / 'confirm_trade.png'),
    'prepare_for_battle': str(IMAGES_DIR / 'prepare_for_battle.png'),
    'ship_waiting_zaliz': str(IMAGES_DIR / 'ship_waiting_zaliz.png'),
    'long_song_2': str(IMAGES_DIR / 'long_song_2.png'),
    'long_song_3': str(IMAGES_DIR / 'long_song_3.png'),
    'long_song_4': str(IMAGES_DIR / 'long_song_4.png'),
    'long_song_5': str(IMAGES_DIR / 'long_song_5.png'),
    'long_song_6': str(IMAGES_DIR / 'long_song_6.png'),
    'cannon_long': str(IMAGES_DIR / 'cannon_long.png'),
    'ship_song': str(IMAGES_DIR / 'ship_song.png'),
    'griffin': str(IMAGES_DIR / 'griffin.png'),
    'molly': str(IMAGES_DIR / 'molly.png')
}

# Координаты для кликов (ОБНОВЛЕНЫ на основе изображения)
COORDINATES = {
    'profile_icon': (52, 50),
    'settings_icon': (1076, 31),
    'characters_icon': (643, 319),
    'add_character_icon': (271, 181),
    'skip_button': (1142, 42),

    # Координаты для скроллинга
    'season_scroll_start': (257, 550),
    'season_scroll_end': (257, 200),
    'server_scroll_start': (640, 550),
    'server_scroll_end': (640, 200),

    # НОВЫЕ координаты для мелкого скроллинга серверов
    'server_small_scroll_start': (640, 380),
    'server_small_scroll_end': (640, 320),

    # Координаты сезонов
    'seasons': {
        'S1': (250, 180),
        'S2': (250, 230),
        'S3': (250, 280),
        'S4': (250, 380),
        'S5': (250, 470),
        'X1': (250, 520),
        'X2': (250, 230),
        'X3': (250, 280),
        'X4': (250, 330)
    },

    # ИСПРАВЛЕННЫЕ координаты для серверов (на основе скриншота)
    'servers': {
        'left_column_x': 466,    # X-координата левого столбца серверов (было 500)
        'right_column_x': 816,   # X-координата правого столбца серверов (было 900)
        'base_y': 136,          # Y-координата первого сервера (было 155)
        'step_y': 90            # Шаг между серверами по вертикали (было 45)
    }
}

# Информация о сезонах и серверах (обновлено согласно новому ТЗ)
SEASONS = {
    'S1': {'min_server': 619, 'max_server': 598},
    'S2': {'min_server': 597, 'max_server': 571},
    'S3': {'min_server': 564, 'max_server': 541},
    'S4': {'min_server': 570, 'max_server': 505},
    'S5': {'min_server': 504, 'max_server': 457},
    'X1': {'min_server': 456, 'max_server': 433},
    'X2': {'min_server': 432, 'max_server': 363},
    'X3': {'min_server': 360, 'max_server': 97},
    'X4': {'min_server': 93, 'max_server': 1},
}

# ИСПРАВЛЕННЫЕ области для OCR распознавания
OCR_REGIONS = {
    'seasons': (150, 150, 220, 400),
    'servers': (400, 130, 570, 470),
    # Увеличиваем область поиска и покрываем весь правый верхний угол
    'skip_button': (900, 0, 380, 100),  # Было: (950, 5, 320, 80)
    # Добавляем дополнительную расширенную область как fallback
    'skip_button_extended': (800, 0, 480, 120)
}

# РАСШИРЕННЫЙ список вариантов для лучшего распознавания
SKIP_BUTTON_VARIANTS = [
    # Основные варианты
    "ПРОПУСТИТЬ", "ПРОПУСТИТЬ >>", "ПРОПУСТИТЬ>", "ПРОПУСТИТЬ >",
    "ПРОПУСТИTЬ", "ПРОПУСТИTЬ >>", "ПРОПУСТИTЬ>", "ПРОПУСТИTЬ >",

    # Варианты с возможными ошибками OCR
    "ПРОNYСТИТЬ", "ПРОПYСТИТЬ", "ПPОПУСТИТЬ", "ПРОПУCTИТЬ",
    "ПРOПУСТИТЬ", "ПPOПУСТИТЬ", "ПРОПУСТИTь",
    "ПРОПУСТИТЬ>>", "ПРОNYСТИТЬ >>", "ПРОПYСТИТЬ >>",

    # Английские варианты
    "SKIP", "SKIP >>", "SKIP>", "SKIP >",

    # Только стрелки (для случаев когда текст не распознается)
    ">>", "> >", "»", "» »",

    # Частичные совпадения
    "РОПУСТИТЬ", "ПРОПУСТИ", "РОПУСТИ", "ПУСТИ", "ПУСТ",

    # С учетом возможных пробелов внутри
    "П Р О П У С Т И Т Ь", "П РОПУСТИТЬ", "ПРО ПУСТИТЬ",
]

# Настройки для пауз между действиями (в секундах)
PAUSE_SETTINGS = {
    'before_season_click': 0.5,     # Пауза перед кликом по сезону
    'after_season_click': 1.5,      # Пауза после клика по сезону
    'after_season_scroll': 2.0,     # Пауза после скроллинга сезонов
    'before_server_click': 0.5,     # Пауза перед кликом по серверу
    'after_server_click': 1.5,      # Пауза после клика по серверу
    'after_server_scroll': 1.5,     # Пауза после скроллинга серверов
    'between_tutorial_steps': 1.0,  # Пауза между шагами обучения
}

# Настройки для распознавания изображений
TEMPLATE_MATCHING_THRESHOLD = 0.7  # Порог уверенности при поиске шаблона

# Последовательность действий для прохождения обучения
TUTORIAL_STEPS = [
    {'type': 'click_image', 'params': {'image_key': 'open_profile'}, 'desc': 'Клик по иконке профиля'},
    {'type': 'click_coord', 'params': {'x': 1073, 'y': 35}, 'desc': 'Клик по иконке настроек'},
    {'type': 'click_coord', 'params': {'x': 738, 'y': 319}, 'desc': 'Клик по иконке персонажей'},
    {'type': 'click_coord', 'params': {'x': 270, 'y': 184}, 'desc': 'Клик по иконке добавления персонажей'},
    {'type': 'select_server', 'params': {}, 'desc': 'Выбор сезона и сервера'},
    {'type': 'click_image', 'params': {'image_key': 'confirm_new_acc'}, 'desc': 'Клик по кнопке подтвердить'},
    {'type': 'delay', 'params': {'seconds': 10}, 'desc': 'Ожидание загрузки'},
    # Остальные шаги из ТЗ будут добавлены в полной реализации
]

# Шаги поиска skip кнопки
FIND_SKIP_STEPS = [
    {'type': 'click_random', 'params': {'center_x': 640, 'center_y': 360, 'radius': 50}, 'desc': 'Клик в центр экрана'},
    {'type': 'find_and_click', 'params': {'image_key': 'skip', 'timeout': 4}, 'desc': 'Поиск и клик по кнопке skip'},
]

# Настройки для улучшенного распознавания серверов
SERVER_RECOGNITION_SETTINGS = {
    'max_scroll_attempts': 10,       # Увеличено до 10 для лучшего поиска
    'scroll_duration': 1000,         # Продолжительность обычного скроллинга в мс
    'small_scroll_duration': 300,    # Уменьшено с 500 до 300 для более точного скроллинга
    'recognition_attempts': 1,       # Уменьшено с 3 до 1 для ускорения
    'servers_per_screen': 10,        # Примерное количество серверов на экране (5 пар)
    'small_scroll_distance': 50,     # Уменьшено с 100 до 50 пикселей
    'max_server_difference': 3,      # Уменьшено с 10 до 3 для большей точности
    'overshoot_threshold': 20,       # Уменьшено с 30 до 20 для более чувствительного определения
}

# УЛУЧШЕННЫЕ настройки для OCR с лучшей фильтрацией
OCR_SETTINGS = {
    'language': 'rus+eng',
    'config': '--psm 6 -c tessedit_char_whitelist=0123456789#№Море ',  # Ограничиваем символы
    'threshold_binary': 150,
    'threshold_adaptive_block_size': 11,
    'threshold_adaptive_c': 2,
    'resize_factor': 2,              # Фактор увеличения изображения для лучшего распознавания
    'min_server_number': 100,        # Минимальный номер сервера для фильтрации
    'max_server_number': 619,        # Максимальный номер сервера для фильтрации
    'max_difference_from_target': 50,  # Уменьшено с 100 до 50
    'min_confidence': 40,            # Минимальная уверенность OCR
}


# Дополнительные координаты для различных элементов интерфейса
ADDITIONAL_COORDINATES = {
    # Координаты для шагов обучения
    'cannon_activation': (718, 438),
    'ship_icon': (58, 654),
    'lower_deck_build': (638, 403),
    'pub_build': (635, 373),
    'storage_repair': (635, 373),
    'upper_deck_build': (345, 386),
    'cannon_select': (77, 276),
    'travel_start': (741, 145),
    'quest_area': (93, 285),
    'screen_continue': (630, 413),
    'compass_icon': (1074, 88),
    'compass_pointer': (701, 258),
    'back_button': (145, 25),
    'trade_confirm': (151, 349),
    'hero_select': (85, 634),
    'battle_start': (1157, 604),
    'battle_area': (642, 334),
    'skull_dead_bay': (653, 403),
    'hammer_icon': (43, 481),
    'ship_upgrade': (127, 216),
    'upgrade_button': (1083, 638),
    'build_button': (639, 603),
    'rowers_cabin': (968, 507),
    'cannon_deck': (687, 514),
    'ship_compass': (652, 214),
    'coins_collect': (931, 620),
}

# Валидация конфигурации
def validate_config():
    """
    Проверка корректности конфигурации.

    Returns:
        bool: True если конфигурация корректна, False иначе
    """
    # Проверка существования директории изображений
    if not IMAGES_DIR.exists():
        print(f"Предупреждение: Директория изображений не существует: {IMAGES_DIR}")
        return False

    # Проверка корректности данных сезонов
    for season_id, season_data in SEASONS.items():
        if season_data['min_server'] < season_data['max_server']:
            print(f"Ошибка: В сезоне {season_id} min_server меньше max_server")
            return False

    # Проверка наличия обязательных изображений
    required_images = ['start_battle', 'coins']
    missing_images = []
    for img_key in required_images:
        if img_key in IMAGE_PATHS:
            img_path = Path(IMAGE_PATHS[img_key])
            if not img_path.exists():
                missing_images.append(img_key)

    if missing_images:
        print(f"Предупреждение: Отсутствуют обязательные изображения: {missing_images}")

    return True

# Запуск валидации при импорте
if __name__ == "__main__":
    if validate_config():
        print("Конфигурация прошла валидацию успешно")
    else:
        print("Обнаружены ошибки в конфигурации")