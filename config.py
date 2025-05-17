"""
Модуль с конфигурационными параметрами для бота Sea of Conquest.
"""
import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent.absolute()
IMAGES_DIR = BASE_DIR / 'images'

# Таймауты
DEFAULT_TIMEOUT = 1.5  # Минимальный таймаут между действиями
LOADING_TIMEOUT = 10   # Таймаут для ожидания загрузки
SCREENSHOT_TIMEOUT = 0.5  # Таймаут между скриншотами при поиске изображений

# Настройки игры
GAME_PACKAGE = "com.seaofconquest.global"
GAME_ACTIVITY = "com.kingsgroup.mo.KGUnityPlayerActivity"

# Пути к изображениям
IMAGE_PATHS = {
    'open_profile': str(IMAGES_DIR / 'open_profile.png'),
    'settings': str(IMAGES_DIR / 'settings.png'),
    'characters': str(IMAGES_DIR / 'characters.png'),
    'add_character': str(IMAGES_DIR / 'add_character.png'),
    'confirm_new_acc': str(IMAGES_DIR / 'confirm_new_acc.png'),
    'skip': str(IMAGES_DIR / 'skip.png'),
    'shoot': str(IMAGES_DIR / 'shoot.png'),
    'hell_genry': str(IMAGES_DIR / 'Hell_Genry.png'),
    'lite_apks': str(IMAGES_DIR / 'lite_apks.png'),
    'close_menu': str(IMAGES_DIR / 'close_menu.png'),
    'ship': str(IMAGES_DIR / 'ship.png'),
    'navigator': str(IMAGES_DIR / 'navigator.png'),
    'hero_face': str(IMAGES_DIR / 'hero_face.png'),
    'start_battle': str(IMAGES_DIR / 'start_battle.png'),
    'upgrade_ship': str(IMAGES_DIR / 'upgrade_ship.png'),
    'open_new_local_building': str(IMAGES_DIR / 'open_new_local_building.png'),
}

# Координаты для кликов
COORDINATES = {
    'profile_icon': (54, 55),
    'settings_icon': (1073, 35),
    'characters_icon': (732, 319),
    'add_character_icon': (270, 184),
    'season_scroll_start': (257, 353),
    'season_scroll_end': (254, 187),
    'server_scroll_start': (778, 567),
    'server_scroll_end': (778, 130),
    'swipe_points': [(154, 351), (288, 355), (507, 353), (627, 351)],
}

# Информация о сезонах и серверах
SEASONS = {
    'S1': {'min_server': 598, 'max_server': 577},
    'S2': {'min_server': 576, 'max_server': 541},
    'S3': {'min_server': 540, 'max_server': 505},
    'S4': {'min_server': 504, 'max_server': 481},
    'S5': {'min_server': 480, 'max_server': 433},
    'X1': {'min_server': 432, 'max_server': 409},
    'X2': {'min_server': 407, 'max_server': 266},
    'X3': {'min_server': 264, 'max_server': 1},
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