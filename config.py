"""
Модуль с конфигурационными параметрами для бота Sea of Conquest.
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

# Пути к изображениям (согласно новому ТЗ, используются только две картинки)
IMAGE_PATHS = {
    'start_battle': str(IMAGES_DIR / 'start_battle.png'),
    'coins': str(IMAGES_DIR / 'coins.png'),
}

# Координаты для кликов
COORDINATES = {
    'profile_icon': (52, 50),
    'settings_icon': (1076, 31),
    'characters_icon': (643, 319),
    'add_character_icon': (271, 181),
    'skip_button': (1142, 42),
    'season_scroll_start': (257, 573),
    'season_scroll_end': (238, 240),
    'server_scroll_start': (777, 566),
    'server_scroll_end': (772, 100),
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