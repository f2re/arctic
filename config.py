# Конфигурационные параметры
"""
Конфигурационные параметры для обнаружения арктических циклонов.
"""

# Параметры для обнаружения синоптического масштаба циклонов (500-1500 км)
SYNOPTIC_CYCLONE_PARAMS = {
    'laplacian_threshold': -0.15,        # Порог лапласиана давления (гПа/м²)
    'min_pressure_threshold': 1015.0,    # Верхний порог давления (гПа)
    'min_size': 3,                       # Минимальный размер области (ячеек сетки)
    'min_depth': 2.5,                    # Минимальная глубина циклона (гПа)
    'smooth_sigma': 1.0,                 # Параметр сглаживания Гаусса
    'pressure_gradient_threshold': 0.7,  # Минимальный градиент давления (гПа/градус)
    'closed_contour_radius': 600,        # Радиус проверки замкнутости контура (км)
    'neighbor_search_radius': 6,         # Радиус поиска соседних точек (ячеек)
    'use_topology_check': True,          # Использовать топологический анализ
    'min_cyclone_radius': 200,           # Минимальный радиус циклона (км)
    'max_cyclone_radius': 1500,          # Максимальный радиус циклона (км)
    'max_cyclones_per_map': 10,          # Максимальное количество циклонов на одной карте
    'min_wind_speed': 15.0,              # Минимальная скорость ветра (м/с)
    'min_vorticity': 0.5e-5,             # Минимальная относительная завихренность (с⁻¹)
    'min_duration': 0                    # Минимальная продолжительность (часов)
}

# Параметры для обнаружения мезомасштабных циклонов (200-600 км)
MESOSCALE_CYCLONE_PARAMS = {
    'laplacian_threshold': -0.12,        # Порог лапласиана для мезомасштабных структур
    'min_pressure_threshold': 1018.0,    # Верхний порог давления для мезомасштабных структур
    'min_size': 1,                       # Меньший минимальный размер для мезомасштабных структур
    'min_depth': 1.0,                    # Меньшая минимальная глубина для мезомасштабных структур
    'smooth_sigma': 0.8,                 # Меньшее сглаживание для сохранения деталей
    'pressure_gradient_threshold': 0.5,  # Меньший порог градиента давления
    'closed_contour_radius': 400,        # Меньший радиус проверки замкнутости контура
    'neighbor_search_radius': 5,         # Меньший радиус поиска соседних точек
    'use_topology_check': True,          # Использовать топологический анализ
    'min_cyclone_radius': 100,           # Меньший минимальный радиус циклона
    'max_cyclone_radius': 600,           # Меньший максимальный радиус циклона
    'max_cyclones_per_map': 15,          # Больше циклонов на карте
    'min_wind_speed': 15.0,              # Минимальная скорость ветра (м/с)
    'min_vorticity': 1.0e-5,             # Повышенное требование к завихренности для мезовихрей
    'min_duration': 24                   # Минимальная продолжительность (часов)
}

# Параметры для обнаружения полярных мезоциклонов (полярных низких)
POLAR_LOW_PARAMS = {
    'laplacian_threshold': -0.10,        # Более мягкий порог лапласиана для полярных низких
    'min_pressure_threshold': 1020.0,    # Более мягкий порог давления для полярных низких
    'min_size': 1,                       # Минимальный размер для мелких структур
    'min_depth': 0.8,                    # Минимальная глубина для мелких структур
    'smooth_sigma': 0.6,                 # Минимальное сглаживание для сохранения деталей
    'pressure_gradient_threshold': 0.4,  # Меньший порог градиента давления
    'closed_contour_radius': 300,        # Меньший радиус проверки замкнутости контура
    'neighbor_search_radius': 4,         # Меньший радиус поиска соседних точек
    'use_topology_check': True,          # Использовать топологический анализ
    'min_cyclone_radius': 50,            # Меньший минимальный радиус циклона
    'max_cyclone_radius': 400,           # Меньший максимальный радиус циклона
    'max_cyclones_per_map': 20,          # Больше циклонов на карте
    'min_wind_speed': 15.0,              # Минимальная скорость ветра (м/с)
    'min_vorticity': 1.5e-5,             # Высокое требование к завихренности для полярных низких
    'min_duration': 12                   # Минимальная продолжительность (часов)
}

# Пути для сохранения файлов
# DEFAULT_DATA_DIR = '/content/drive/MyDrive/arctic/data'
# DEFAULT_IMAGE_DIR = '/content/drive/MyDrive/arctic/images'
# DEFAULT_CHECKPOINT_DIR = '/content/drive/MyDrive/arctic/checkpoints'

DEFAULT_DATA_DIR = '~/arctic_git/data'
DEFAULT_IMAGE_DIR = '~/arctic_git/images'
DEFAULT_CHECKPOINT_DIR = '~/arctic_git/checkpoints'
DEFAULT_CRITERIA_DIR = '~/arctic_git/criteria_images'