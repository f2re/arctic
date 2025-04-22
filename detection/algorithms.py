# Алгоритмы обнаружения
"""
Алгоритмы обнаружения циклонов.
"""
import sys, os
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# Добавляем корневой каталог в путь для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.preprocessing import analyze_grid_scale, adapt_detection_params
from detection.parameters import get_cyclone_params

def calculate_laplacian_improved(pressure_field, dx, method='adaptive', smooth_sigma=0.7):
    """
    Улучшенное вычисление лапласиана поля давления с адаптивными методами расчета.

    Параметры:
    ----------
    pressure_field : numpy.ndarray
        Двумерный массив значений давления
    dx : float
        Пространственное разрешение сетки в метрах
    method : str
        Метод расчета лапласиана: 'standard', 'filtered', 'highorder' или 'adaptive'
    smooth_sigma : float
        Параметр сглаживания для методов 'filtered' и 'adaptive'

    Возвращает:
    -----------
    numpy.ndarray
        Поле лапласиана давления
    tuple
        Дополнительные поля (градиент давления) для диагностики
    """
    # Реализация без изменений из оригинального файла
    # Создаем копию поля давления для предотвращения изменения оригинала
    pressure = pressure_field.copy()

    # Вычисляем градиенты давления для диагностики с использованием центральных разностей
    dy = dx  # Предполагаем равномерную сетку

    # Создаем массивы для градиентов
    grad_x = np.zeros_like(pressure)
    grad_y = np.zeros_like(pressure)

    # Вычисляем градиенты только для внутренних точек (оптимизация)
    grad_x[1:-1, 1:-1] = (pressure[1:-1, 2:] - pressure[1:-1, :-2]) / (2 * dx)
    grad_y[1:-1, 1:-1] = (pressure[2:, 1:-1] - pressure[:-2, 1:-1]) / (2 * dy)

    # Величина градиента
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Выбор метода расчета лапласиана
    if method == 'adaptive':
        # Предварительная фильтрация для подавления шума
        pressure_filtered = gaussian_filter(pressure, sigma=smooth_sigma)

        # Стандартный 5-точечный шаблон
        laplacian_standard = np.zeros_like(pressure)
        laplacian_standard[1:-1, 1:-1] = (
            pressure_filtered[:-2, 1:-1] +
            pressure_filtered[2:, 1:-1] +
            pressure_filtered[1:-1, :-2] +
            pressure_filtered[1:-1, 2:] -
            4 * pressure_filtered[1:-1, 1:-1]
        ) / (dx**2)

        # 8-точечный шаблон высокого порядка
        laplacian_highorder = np.zeros_like(pressure)
        laplacian_highorder[1:-1, 1:-1] = (
            pressure_filtered[:-2, 1:-1] +
            pressure_filtered[2:, 1:-1] +
            pressure_filtered[1:-1, :-2] +
            pressure_filtered[1:-1, 2:] +
            0.5 * pressure_filtered[:-2, :-2] +
            0.5 * pressure_filtered[:-2, 2:] +
            0.5 * pressure_filtered[2:, :-2] +
            0.5 * pressure_filtered[2:, 2:] -
            6 * pressure_filtered[1:-1, 1:-1]
        ) / (dx**2)

        # Нормализуем градиент к диапазону [0, 1]
        max_gradient = np.max(gradient_magnitude)
        if max_gradient > 0:
            norm_gradient = gradient_magnitude / max_gradient
        else:
            norm_gradient = np.zeros_like(gradient_magnitude)

        # Весовые коэффициенты
        weight_standard = 0.7 * norm_gradient + 0.3
        weight_highorder = 1.0 - weight_standard

        # Комбинируем результаты
        laplacian = weight_standard * laplacian_standard + weight_highorder * laplacian_highorder

    elif method == 'filtered':
        # Метод с предварительной фильтрацией
        pressure = gaussian_filter(pressure, sigma=smooth_sigma)

        laplacian = np.zeros_like(pressure)
        laplacian[1:-1, 1:-1] = (
            pressure[:-2, 1:-1] +
            pressure[2:, 1:-1] +
            pressure[1:-1, :-2] +
            pressure[1:-1, 2:] -
            4 * pressure[1:-1, 1:-1]
        ) / (dx**2)

    elif method == 'highorder':
        # 8-точечный шаблон
        laplacian = np.zeros_like(pressure)
        laplacian[1:-1, 1:-1] = (
            pressure[:-2, 1:-1] +
            pressure[2:, 1:-1] +
            pressure[1:-1, :-2] +
            pressure[1:-1, 2:] +
            0.5 * pressure[:-2, :-2] +
            0.5 * pressure[:-2, 2:] +
            0.5 * pressure[2:, :-2] +
            0.5 * pressure[2:, 2:] -
            6 * pressure[1:-1, 1:-1]
        ) / (dx**2)

    else:  # 'standard'
        # Стандартный 5-точечный шаблон
        laplacian = np.zeros_like(pressure)
        laplacian[1:-1, 1:-1] = (
            pressure[:-2, 1:-1] +
            pressure[2:, 1:-1] +
            pressure[1:-1, :-2] +
            pressure[1:-1, 2:] -
            4 * pressure[1:-1, 1:-1]
        ) / (dx**2)

    # Обрабатываем границы массива
    laplacian[0, :] = laplacian[1, :]
    laplacian[-1, :] = laplacian[-2, :]
    laplacian[:, 0] = laplacian[:, 1]
    laplacian[:, -1] = laplacian[:, -2]

    return laplacian, (gradient_magnitude, grad_x, grad_y)

def find_pressure_minima(pressure_field, lat_values, lon_values, threshold=1018.0, window_size=3):
    """
    Оптимизированная функция для поиска локальных минимумов давления.

    Параметры:
    ----------
    pressure_field : numpy.ndarray
        Двумерное поле давления
    lat_values : numpy.ndarray
        Массив значений широты
    lon_values : numpy.ndarray
        Массив значений долготы
    threshold : float
        Пороговое значение давления
    window_size : int
        Размер окна для поиска

    Возвращает:
    -----------
    numpy.ndarray
        Булева маска с потенциальными центрами циклонов
    """
    # Реализация без изменений из оригинального файла
    print(f"Поиск локальных минимумов давления на основе топологии...")

    # Определяем половину размера окна
    half_window = window_size // 2

    # Создаем маску для точек с давлением ниже порога
    pressure_threshold_mask = pressure_field <= threshold

    # Инициализируем маску локальных минимумов
    minima_mask = np.zeros_like(pressure_field, dtype=bool)

    # Ограничиваем область поиска точками с давлением ниже порога
    # и точками не на границе (для корректной работы с окнами)
    valid_mask = pressure_threshold_mask.copy()
    valid_mask[:half_window, :] = False
    valid_mask[-half_window:, :] = False
    valid_mask[:, :half_window] = False
    valid_mask[:, -half_window:] = False

    # Находим индексы точек, где давление ниже порога
    valid_points = np.where(valid_mask)

    # Проверяем каждую точку с давлением ниже порога
    for i, j in zip(*valid_points):
        # Выделяем окрестность
        neighborhood = pressure_field[i-half_window:i+half_window+1, j-half_window:j+half_window+1]

        # Проверяем, является ли точка локальным минимумом
        if pressure_field[i, j] == np.min(neighborhood):
            minima_mask[i, j] = True

    # Фильтруем точки по широте - только выше 70°N
    for i in range(len(lat_values)):
        if lat_values[i] < 70.0:
            minima_mask[i, :] = False

    print(f"Найдено {np.sum(minima_mask)} локальных минимумов давления")
    return minima_mask

def estimate_cyclone_radius_optimized(pressure_field, lat_idx, lon_idx, lat_values, lon_values):
    """
    Оптимизированная функция для быстрой оценки радиуса циклона.
    Заменяет более сложные функции calculate_cyclone_radius и calculate_cyclone_size.

    Параметры:
    ----------
    pressure_field : numpy.ndarray
        Двумерное поле давления
    lat_idx, lon_idx : int
        Индексы сетки центра циклона
    lat_values, lon_values : numpy.ndarray
        Массивы значений широты и долготы сетки

    Возвращает:
    -----------
    float
        Оценка радиуса циклона в километрах
    dict
        Дополнительная информация о циклоне
    """
    # Получаем центральное давление и координаты
    center_pressure = pressure_field[lat_idx, lon_idx]
    center_lat = lat_values[lat_idx]
    center_lon = lon_values[lon_idx]
    
    # Определяем пороговое давление: центральное + смещение
    # Эмпирически оптимальное значение для большинства циклонов
    pressure_offset = 2.0  # гПа
    threshold_pressure = center_pressure + pressure_offset
    
    # Создаем маску для области циклона
    cyclone_mask = pressure_field < threshold_pressure
    
    # Выделяем связную область, содержащую центр циклона
    labeled_regions, num_regions = ndimage.label(cyclone_mask)
    center_region = labeled_regions[lat_idx, lon_idx]
    
    if center_region == 0:
        return 0.0, {'area_km2': 0, 'last_closed_isobar': center_pressure}
    
    region_mask = labeled_regions == center_region
    
    # Находим все точки региона
    region_points = np.where(region_mask)
    
    if len(region_points[0]) == 0:
        return 0.0, {'area_km2': 0, 'last_closed_isobar': center_pressure}
    
    # Вычисляем максимальное расстояние от центра до точек региона
    max_distance_km = 0.0
    
    # Оптимизация: используем векторизацию вместо циклов
    lat_points = lat_values[region_points[0]]
    lon_points = lon_values[region_points[1]]
    
    # Применяем формулу гаверсинуса в векторизованном виде
    lat1_rad = np.radians(center_lat)
    lat2_rad = np.radians(lat_points)
    dlon = np.radians(lon_points - center_lon)
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances_km = 6371 * c  # 6371 км - средний радиус Земли
    
    if len(distances_km) > 0:
        max_distance_km = np.max(distances_km)
    
    # Оценка площади
    # Приближенная площадь ячейки на широте центра
    dlat = np.abs(lat_values[1] - lat_values[0]) if len(lat_values) > 1 else 1.0
    dlon = np.abs(lon_values[1] - lon_values[0]) if len(lon_values) > 1 else 1.0
    
    # Площадь одной ячейки сетки в км²
    cell_area = (dlat * 111.32) * (dlon * 111.32 * np.cos(np.radians(center_lat)))
    
    # Общая площадь
    area_km2 = len(region_points[0]) * cell_area
    
    return max_distance_km, {
        'area_km2': area_km2,
        'last_closed_isobar': threshold_pressure
    }

def check_closed_contour_optimized(pressure, lat_idx, lon_idx, lat_values, lon_values, radius_km=300):
    """
    Оптимизированная проверка наличия замкнутых изобар вокруг потенциального центра циклона.

    Параметры:
    ----------
    pressure : numpy.ndarray
        Двумерное поле давления
    lat_idx, lon_idx : int
        Индексы сетки потенциального центра циклона
    lat_values, lon_values : numpy.ndarray
        Массивы значений широты и долготы сетки
    radius_km : float
        Радиус проверки в километрах

    Возвращает:
    -----------
    bool
        True, если обнаружен замкнутый контур
    float
        Глубина циклона (разница между средним давлением по периметру и в центре)
    """
    # Реализация без изменений из оригинального файла
    # Получаем центральные координаты и давление
    center_lat = lat_values[lat_idx]
    center_lon = lon_values[lon_idx]
    center_pressure = pressure[lat_idx, lon_idx]

    # Оптимизация: вместо попиксельного расчета расстояний, создаем приближенную
    # маску для региона с заданным радиусом
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing='ij')

    # Упрощенный расчет расстояний (в км)
    lat_distances = np.abs(lat_grid - center_lat) * 111.32
    lon_distances = np.abs(lon_grid - center_lon) * 111.32 * np.cos(np.radians(center_lat))
    distances = np.sqrt(lat_distances**2 + lon_distances**2)

    # Маска точек внутри заданного радиуса
    inner_mask = distances <= radius_km

    # Создаем маску периметра - точки на границе радиуса
    # Используем дилатацию с последующим вычитанием исходной маски
    kernel = np.ones((3, 3), dtype=bool)
    expanded_mask = ndimage.binary_dilation(inner_mask, structure=kernel)
    perimeter_mask = expanded_mask & ~inner_mask

    # Если периметр пустой, нет замкнутого контура
    if not np.any(perimeter_mask):
        return False, 0.0

    # Давление на периметре
    perimeter_pressures = pressure[perimeter_mask]
    min_perimeter_pressure = np.min(perimeter_pressures)

    # Глубина циклона - разница между средним давлением на периметре и в центре
    perimeter_mean_pressure = np.mean(perimeter_pressures)
    depth = perimeter_mean_pressure - center_pressure

    # Условие замкнутого контура: давление в центре ниже минимального на периметре
    return center_pressure < min_perimeter_pressure, depth

def has_significant_pressure_gradient(gradient_magnitude, lat_idx, lon_idx, neighbor_radius=4, threshold=0.8):
    """
    Оптимизированная проверка наличия значительного градиента давления в окрестности.

    Параметры:
    ----------
    gradient_magnitude : numpy.ndarray
        Двумерное поле величины градиента давления
    lat_idx, lon_idx : int
        Индексы сетки потенциального центра циклона
    neighbor_radius : int
        Радиус проверки в ячейках сетки
    threshold : float
        Пороговое значение градиента давления

    Возвращает:
    -----------
    bool
        True, если обнаружен значительный градиент
    float
        Максимальный градиент в окрестности
    """
    # Реализация без изменений из оригинального файла
    # Определяем границы окрестности
    lat_min = max(0, lat_idx - neighbor_radius)
    lat_max = min(gradient_magnitude.shape[0], lat_idx + neighbor_radius + 1)
    lon_min = max(0, lon_idx - neighbor_radius)
    lon_max = min(gradient_magnitude.shape[1], lon_idx + neighbor_radius + 1)

    # Выделяем окрестность
    neighborhood = gradient_magnitude[lat_min:lat_max, lon_min:lon_max]

    # Находим максимальный градиент в окрестности
    max_gradient = np.max(neighborhood)

    # Проверяем превышает ли максимальный градиент пороговое значение
    return max_gradient >= threshold, max_gradient

def detect_cyclones_improved(ds, time_idx, time_dim, pressure_var, lat_var, lon_var, 
                           cyclone_params=None, cyclone_type='mesoscale', detection_methods=None):
    """
    Оптимизированный алгоритм обнаружения арктических циклонов.
    
    Функция выполняет многоуровневую фильтрацию потенциальных циклонических систем 
    на основе лапласиана давления, градиента, замкнутых изобар и других критериев.
    При наличии данных также проверяются скорость ветра и относительная завихренность.
    
    Параметры:
    ----------
    ds : xarray.Dataset
        Набор данных с полем приземного давления и другими переменными
    time_idx : int
        Индекс временного шага
    time_dim : str
        Имя измерения времени
    pressure_var : str
        Имя переменной давления
    lat_var : str
        Имя переменной широты
    lon_var : str
        Имя переменной долготы
    cyclone_params : dict, optional
        Словарь с параметрами алгоритма. Если None, используются выбранные параметры.
    cyclone_type : str, default='mesoscale'
        Тип циклонов для обнаружения: 'synoptic', 'mesoscale' или 'polar_low'
    detection_methods : list, optional
        Список методов обнаружения циклонов для использования. 
        Возможные значения: 'laplacian', 'pressure_minima', 'closed_contour', 'gradient', 'vorticity', 'wind_speed'.
        По умолчанию используются все методы, соответствующие типу циклона.
    
    Возвращает:
    -----------
    tuple
        (давление, лапласиан, координаты центров циклонов, найдены ли циклоны, маска, диагностические данные)
    """
    # Установка методов обнаружения по умолчанию, если не указаны
    if detection_methods is None:
        # Методы по умолчанию для каждого типа циклонов
        if cyclone_type == 'synoptic':
            detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 'gradient']
        elif cyclone_type == 'mesoscale':
            detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 'gradient', 'vorticity', 'wind_speed']
        else:  # polar_low
            detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 'gradient', 'vorticity', 'wind_speed']
    
    # Проверяем и устанавливаем параметры алгоритма
    if cyclone_params is None:
        cyclone_params = get_cyclone_params(cyclone_type)
    
    # Выводим информацию о типе обнаруживаемых циклонов и используемых методах
    print(f"Обнаружение циклонов типа: {cyclone_type}")
    print(f"Используемые методы обнаружения: {', '.join(detection_methods)}")
    print(f"Диапазон размеров: {cyclone_params['min_cyclone_radius']}-{cyclone_params['max_cyclone_radius']} км")

    # Извлекаем параметры алгоритма
    laplacian_threshold = cyclone_params['laplacian_threshold']
    min_pressure_threshold = cyclone_params['min_pressure_threshold']
    min_size = cyclone_params['min_size']
    min_depth = cyclone_params['min_depth']
    smooth_sigma = cyclone_params['smooth_sigma']
    pressure_gradient_threshold = cyclone_params['pressure_gradient_threshold']
    closed_contour_radius = cyclone_params['closed_contour_radius']
    neighbor_search_radius = cyclone_params['neighbor_search_radius']
    use_topology_check = cyclone_params['use_topology_check']
    min_cyclone_radius = cyclone_params['min_cyclone_radius']
    max_cyclone_radius = cyclone_params['max_cyclone_radius']
    max_cyclones_per_map = cyclone_params['max_cyclones_per_map']
    min_vorticity = cyclone_params['min_vorticity']
    min_wind_speed = cyclone_params['min_wind_speed']

    # Получаем поле давления для указанного времени
    try:
        if time_dim in ds[pressure_var].dims:
            pressure = ds[pressure_var].isel({time_dim: time_idx}).values
        else:
            pressure = ds[pressure_var].values

        # Проверяем размерность массива
        if len(pressure.shape) > 2:
            # Если давление - многомерный массив, извлекаем первое измерение
            pressure = pressure[0] if pressure.shape[0] <= 10 else pressure
            print(f"Предупреждение: Поле давления имеет размерность {pressure.shape}. Извлечен первый слой.")
    except Exception as e:
        print(f"Ошибка при получении поля давления: {e}")
        # Возвращаем пустые результаты в случае ошибки
        return None, None, [], False, None, {"error": str(e)}

    # Получаем координаты
    try:
        lat_values = ds[lat_var].values
        lon_values = ds[lon_var].values
    except Exception as e:
        print(f"Ошибка при получении координат: {e}")
        return None, None, [], False, None, {"error": str(e)}

    # Анализируем масштаб сетки
    grid_info = analyze_grid_scale(lat_values, lon_values)
    print(f"\nАнализ масштаба сетки:")
    print(f"Шаг по широте: {grid_info['lat_step_deg']:.4f}° ({grid_info['lat_step_km']:.2f} км)")
    print(f"Шаг по долготе: {grid_info['lon_step_deg']:.4f}° ({grid_info['lon_step_km']:.2f} км)")
    print(f"Разрешение сетки: {grid_info['grid_resolution']}")

    # Адаптируем параметры в зависимости от разрешения сетки
    try:
        adapted_params = adapt_detection_params(cyclone_params, grid_info)
        
        # Используем адаптированные параметры
        laplacian_threshold = adapted_params.get('laplacian_threshold', laplacian_threshold)
        min_pressure_threshold = adapted_params.get('min_pressure_threshold', min_pressure_threshold)
        min_size = adapted_params.get('min_size', min_size)
        min_depth = adapted_params.get('min_depth', min_depth)

        print(f"Адаптированные параметры обнаружения для данной сетки:")
        print(f"Порог лапласиана: {laplacian_threshold}")
        print(f"Порог давления: {min_pressure_threshold}")
        print(f"Минимальный размер: {min_size}")
        print(f"Минимальная глубина: {min_depth}")
    except Exception as e:
        print(f"Предупреждение: Не удалось адаптировать параметры: {e}")
        # Продолжаем с исходными параметрами

    # Вычисляем пространственное разрешение в метрах
    try:
        if len(lon_values) > 1:
            if lon_values[1] - lon_values[0] < 0:
                dx = np.deg2rad(abs(lon_values[1] - lon_values[0])) * 6371000 * np.cos(np.deg2rad(75))
            else:
                dx = np.deg2rad(lon_values[1] - lon_values[0]) * 6371000 * np.cos(np.deg2rad(75))
        else:
            dx = 10000  # Значение по умолчанию, если не удается определить
            print("Предупреждение: Невозможно определить шаг сетки, используется значение по умолчанию.")
    except Exception as e:
        print(f"Ошибка при расчете разрешения сетки: {e}")
        dx = 10000  # Значение по умолчанию

    # Преобразуем давление в гПа для расчетов
    pressure_hpa = pressure / 100.0 if np.max(pressure) > 10000 else pressure

    # Вычисляем лапласиан давления с использованием указанного метода
    try:
        laplacian, diagnostic_fields = calculate_laplacian_improved(
            pressure_hpa, dx, method='adaptive', smooth_sigma=smooth_sigma)
    except Exception as e:
        print(f"Ошибка при расчете лапласиана: {e}")
        return pressure_hpa, None, [], False, None, {"error": str(e)}

    # Распаковываем диагностические поля
    gradient_magnitude, grad_x, grad_y = diagnostic_fields

    # Диагностическая информация
    print(f"Диапазон лапласиана: {np.min(laplacian):.3f} - {np.max(laplacian):.3f}")
    print(f"Диапазон градиента давления: {np.min(gradient_magnitude):.3f} - {np.max(gradient_magnitude):.3f} гПа/градус")
    print(f"Пороговое значение лапласиана: {laplacian_threshold}")

    # Применяем пороговое значение для выделения потенциальных циклонов
    cyclone_mask = laplacian < laplacian_threshold

    # Если используется топологическая проверка, добавляем потенциальные центры на основе топологии
    topology_mask = np.zeros_like(cyclone_mask, dtype=bool)
    if use_topology_check:
        print("Применяется дополнительная топологическая проверка...")
        try:
            topology_mask = find_pressure_minima(pressure_hpa, lat_values, lon_values, threshold=min_pressure_threshold)
            # Объединяем маски
            cyclone_mask = cyclone_mask | topology_mask
            print(f"Добавлено {np.sum(topology_mask & ~(laplacian < laplacian_threshold))} потенциальных центров на основе топологии")
        except Exception as e:
            print(f"Ошибка при выполнении топологической проверки: {e}")
            # Продолжаем без топологической проверки

    # Фильтруем по широте - ограничиваем регион выше 70°N
    min_latitude = 70.0  # Можно параметризовать в будущем
    for i in range(len(lat_values)):
        if lat_values[i] < min_latitude:
            cyclone_mask[i, :] = False

    # Маркируем отдельные циклонические системы
    try:
        from scipy import ndimage
        labeled_cyclones, num_cyclones = ndimage.label(cyclone_mask)
    except Exception as e:
        print(f"Ошибка при маркировке циклонических систем: {e}")
        return pressure_hpa, laplacian, [], False, None, {"error": str(e)}

    print(f"Обнаружено {num_cyclones} потенциальных циклонических систем")

    # Находим центры циклонов с дополнительными критериями
    cyclone_centers = []
    valid_cyclones = []
    all_potential_centers = []  # Для диагностики
    rejected_centers = []       # Для диагностики: причины отклонения

    # Проверка наличия данных о ветре
    has_wind_data = False
    wind_field = None
    # print(ds.variables)
    if '10m_wind_speed' in ds:
        has_wind_data = True
        try:
            if time_dim in ds['10m_wind_speed'].dims:
                wind_field = ds['10m_wind_speed'].isel({time_dim: time_idx}).values
            else:
                wind_field = ds['10m_wind_speed'].values
                
            if len(wind_field.shape) > 2:
                wind_field = wind_field[0] if wind_field.shape[0] <= 10 else wind_field
                
            print(f"Данные о скорости ветра доступны. Максимальная скорость: {np.max(wind_field):.2f} м/с")
        except Exception as e:
            print(f"Ошибка при чтении данных о скорости ветра: {e}")
            has_wind_data = False
    else:
        print("Данные о скорости ветра отсутствуют. Критерий скорости ветра не будет применяться.")
    
    # Проверка наличия данных о завихренности
    has_vorticity_data = False
    vorticity_field = None
    vorticity_var = None
    
    # Поиск переменной завихренности в наборе данных
    for var in ds.variables:
        if 'vo' in var.lower():
            vorticity_var = var
            has_vorticity_data = True
            break
    
    if has_vorticity_data:
        try:
            if time_dim in ds[vorticity_var].dims:
                vorticity_field = ds[vorticity_var].isel({time_dim: time_idx}).values
            else:
                vorticity_field = ds[vorticity_var].values
                
            if len(vorticity_field.shape) > 2:
                vorticity_field = vorticity_field[0] if vorticity_field.shape[0] <= 10 else vorticity_field
                
            print(f"Данные о завихренности доступны (переменная: {vorticity_var}). "
                  f"Диапазон значений: {np.min(vorticity_field):.2e} - {np.max(vorticity_field):.2e} с^-1")
        except Exception as e:
            print(f"Ошибка при чтении данных о завихренности: {e}")
            has_vorticity_data = False
    else:
        print("Данные о завихренности отсутствуют. Критерий завихренности не будет применяться.")

    # Оптимизация: обрабатываем только ограниченное количество потенциальных циклонов
    # для предотвращения чрезмерных вычислений
    max_potential_cyclones = min(num_cyclones, 100)  # Ограничиваем количество обрабатываемых систем

    for i in range(1, min(num_cyclones + 1, max_potential_cyclones + 1)):
        # Находим все точки, принадлежащие i-му циклону
        cyclone_points = np.where(labeled_cyclones == i)

        # Проверяем размер области
        if len(cyclone_points[0]) < min_size:
            rejected_info = {
                "reason": "small_size", 
                "size": len(cyclone_points[0]), 
                "threshold": min_size,
                "details": f"Размер области ({len(cyclone_points[0])}) меньше минимального ({min_size})"
            }
            rejected_centers.append(rejected_info)
            continue

        # Находим давление в каждой точке циклона
        cyclone_pressures = pressure_hpa[cyclone_points]

        # Находим индекс минимального давления
        min_pressure_idx = np.argmin(cyclone_pressures)
        min_pressure = cyclone_pressures[min_pressure_idx]

        # Координаты точки с минимальным давлением
        min_lat_idx = cyclone_points[0][min_pressure_idx]
        min_lon_idx = cyclone_points[1][min_pressure_idx]

        # Проверяем широту - циклон должен быть севернее min_latitude
        if lat_values[min_lat_idx] < min_latitude:
            rejected_info = {
                "reason": "low_latitude",
                "latitude": lat_values[min_lat_idx],
                "threshold": min_latitude,
                "details": f"Широта ({lat_values[min_lat_idx]:.2f}°N) ниже минимальной ({min_latitude}°N)"
            }
            rejected_centers.append(rejected_info)
            continue

        # Быстрая оценка радиуса циклона
        try:
            cyclone_radius, cyclone_info = estimate_cyclone_radius_optimized(
                pressure_hpa, min_lat_idx, min_lon_idx, lat_values, lon_values)
        except Exception as e:
            print(f"Ошибка при оценке радиуса циклона: {e}")
            cyclone_radius = 0
            cyclone_info = {'area_km2': 0, 'last_closed_isobar': 0}

        # Проверяем размеры циклона - должен быть в диапазоне min_cyclone_radius-max_cyclone_radius км
        if cyclone_radius < min_cyclone_radius or cyclone_radius > max_cyclone_radius:
            rejected_info = {
                "reason": "invalid_size",
                "radius": cyclone_radius,
                "min_threshold": min_cyclone_radius,
                "max_threshold": max_cyclone_radius,
                "details": f"Радиус циклона ({cyclone_radius:.1f} км) вне допустимого диапазона "
                          f"({min_cyclone_radius}-{max_cyclone_radius} км)"
            }
            rejected_centers.append(rejected_info)
            continue

        # Записываем информацию о потенциальном центре для диагностики
        potential_center = {
            "lat_idx": min_lat_idx,
            "lon_idx": min_lon_idx,
            "latitude": lat_values[min_lat_idx],
            "longitude": lon_values[min_lon_idx],
            "pressure": min_pressure,
            "laplacian": laplacian[min_lat_idx, min_lon_idx],
            "gradient": gradient_magnitude[min_lat_idx, min_lon_idx],
            "region_size": len(cyclone_points[0]),
            "radius_km": cyclone_radius
        }
        
        # Добавляем данные о скорости ветра если доступны
        if has_wind_data and wind_field is not None:
            # Находим максимальную скорость ветра в окрестности циклона
            neighborhood_radius = 5  # ячеек сетки
            lat_min = max(0, min_lat_idx - neighborhood_radius)
            lat_max = min(wind_field.shape[0], min_lat_idx + neighborhood_radius + 1)
            lon_min = max(0, min_lon_idx - neighborhood_radius)
            lon_max = min(wind_field.shape[1], min_lon_idx + neighborhood_radius + 1)
            
            try:
                local_wind = wind_field[lat_min:lat_max, lon_min:lon_max]
                max_wind = np.max(local_wind) if local_wind.size > 0 else 0
                potential_center["max_wind"] = max_wind
            except Exception as e:
                print(f"Ошибка при анализе скорости ветра: {e}")
                potential_center["max_wind"] = 0
        
        # Добавляем данные о завихренности если доступны
        if has_vorticity_data and vorticity_field is not None:
            # Находим максимальную завихренность в окрестности циклона
            neighborhood_radius = 5  # ячеек сетки
            lat_min = max(0, min_lat_idx - neighborhood_radius)
            lat_max = min(vorticity_field.shape[0], min_lat_idx + neighborhood_radius + 1)
            lon_min = max(0, min_lon_idx - neighborhood_radius)
            lon_max = min(vorticity_field.shape[1], min_lon_idx + neighborhood_radius + 1)
            
            try:
                local_vorticity = vorticity_field[lat_min:lat_max, lon_min:lon_max]
                max_vorticity = np.max(local_vorticity) if local_vorticity.size > 0 else 0
                potential_center["max_vorticity"] = max_vorticity
            except Exception as e:
                print(f"Ошибка при анализе завихренности: {e}")
                potential_center["max_vorticity"] = 0
        
        all_potential_centers.append(potential_center)

        # Проверяем наличие замкнутых изобар (оптимизированная версия)
        try:
            has_closed_contour, depth = check_closed_contour_optimized(
                pressure_hpa, min_lat_idx, min_lon_idx, lat_values, lon_values,
                radius_km=closed_contour_radius)
        except Exception as e:
            print(f"Ошибка при проверке замкнутых изобар: {e}")
            has_closed_contour, depth = False, 0

        # Проверяем наличие значительного градиента давления в окрестности
        try:
            has_gradient, max_gradient = has_significant_pressure_gradient(
                gradient_magnitude, min_lat_idx, min_lon_idx,
                neighbor_radius=neighbor_search_radius,
                threshold=pressure_gradient_threshold)
        except Exception as e:
            print(f"Ошибка при анализе градиента давления: {e}")
            has_gradient, max_gradient = False, 0

        # Дополняем информацию о потенциальном центре
        potential_center["depth"] = depth
        potential_center["has_closed_contour"] = has_closed_contour
        potential_center["has_gradient"] = has_gradient
        potential_center["max_gradient"] = max_gradient

        # Применяем критерии для фильтрации циклонов
        # Смягчаем критерии для топологических центров
        is_topology_center = use_topology_check and topology_mask[min_lat_idx, min_lon_idx]

        #  Применяем только выбранные методы детекции
        # Проверка давления
        if 'pressure_minima' in detection_methods:
            if min_pressure >= min_pressure_threshold and not is_topology_center:
                rejected_info = {
                    "reason": "high_pressure",
                    "pressure": min_pressure,
                    "threshold": min_pressure_threshold,
                    "lat": lat_values[min_lat_idx],
                    "lon": lon_values[min_lon_idx],
                    "details": f"Давление в центре ({min_pressure:.1f} гПа) выше порога ({min_pressure_threshold} гПа)"
                }
                rejected_centers.append(rejected_info)
                continue

        # Проверка глубины
        if 'pressure_minima' in detection_methods:
            if depth < min_depth and not is_topology_center:
                rejected_info = {
                    "reason": "shallow_depth",
                    "depth": depth,
                    "threshold": min_depth,
                    "lat": lat_values[min_lat_idx],
                    "lon": lon_values[min_lon_idx],
                    "details": f"Глубина циклона ({depth:.1f} гПа) меньше минимальной ({min_depth} гПа)"
                }
                rejected_centers.append(rejected_info)
                continue

        # Проверка замкнутых изобар
        if 'closed_contour' in detection_methods:
            if not has_closed_contour:
                rejected_info = {
                    "reason": "no_closed_contour",
                    "lat": lat_values[min_lat_idx],
                    "lon": lon_values[min_lon_idx],
                    "depth": depth,
                    "details": "Отсутствуют замкнутые изобары вокруг минимума давления"
                }
                rejected_centers.append(rejected_info)
                continue

        # Проверка градиента давления
        if 'gradient' in detection_methods:
            if not has_gradient and not is_topology_center:
                rejected_info = {
                    "reason": "weak_gradient",
                    "gradient": max_gradient,
                    "threshold": pressure_gradient_threshold,
                    "lat": lat_values[min_lat_idx],
                    "lon": lon_values[min_lon_idx],
                    "details": f"Градиент давления ({max_gradient:.2f} гПа/градус) ниже порога "
                              f"({pressure_gradient_threshold} гПа/градус)"
                }
                rejected_centers.append(rejected_info)
                continue

        # Проверка скорости ветра (если данные доступны и метод выбран)
        if 'wind_speed' in detection_methods and has_wind_data and "max_wind" in potential_center:
            max_wind = potential_center["max_wind"]
            if max_wind < min_wind_speed:
                rejected_info = {
                    "reason": "weak_wind",
                    "wind_speed": max_wind,
                    "threshold": min_wind_speed,
                    "lat": lat_values[min_lat_idx],
                    "lon": lon_values[min_lon_idx],
                    "details": f"Максимальная скорость ветра ({max_wind:.1f} м/с) ниже порога "
                              f"({min_wind_speed} м/с)"
                }
                rejected_centers.append(rejected_info)
                continue

        # Проверка завихренности (если данные доступны и метод выбран)
        if 'vorticity' in detection_methods and has_vorticity_data and "max_vorticity" in potential_center:
            max_vorticity = potential_center["max_vorticity"]
            if max_vorticity < min_vorticity:
                rejected_info = {
                    "reason": "weak_vorticity",
                    "vorticity": max_vorticity,
                    "threshold": min_vorticity,
                    "lat": lat_values[min_lat_idx],
                    "lon": lon_values[min_lon_idx],
                    "details": f"Максимальная завихренность ({max_vorticity:.2e} с^-1) ниже порога "
                              f"({min_vorticity} с^-1)"
                }
                rejected_centers.append(rejected_info)
                continue

        # Если прошли все критерии, добавляем циклон в список валидных
        valid_cyclones.append(i)

        # Создаем кортеж информации о циклоне
        cyclone_info = [
            lat_values[min_lat_idx],  # широта
            lon_values[min_lon_idx],  # долгота
            min_pressure,            # давление в центре (гПа)
            depth,                   # глубина циклона (гПа)
            max_gradient,            # максимальный градиент давления
            cyclone_radius           # приблизительный радиус в км
        ]
        
        # Добавляем информацию о скорости ветра и завихренности если доступны
        if has_wind_data and "max_wind" in potential_center:
            cyclone_info.append(potential_center["max_wind"])
        
        if has_vorticity_data and "max_vorticity" in potential_center:
            cyclone_info.append(potential_center["max_vorticity"])
        
        # Добавляем в список центров циклонов
        cyclone_centers.append(tuple(cyclone_info))

    # Сортируем центры циклонов по давлению (по возрастанию, т.е. сначала самые глубокие)
    cyclone_centers.sort(key=lambda x: x[2])

    # Ограничиваем количество циклонов на одной карте
    if len(cyclone_centers) > max_cyclones_per_map:
        print(f"Ограничиваем количество отображаемых циклонов до {max_cyclones_per_map}")
        cyclone_centers = cyclone_centers[:max_cyclones_per_map]

    # Выводим информацию о причинах отклонения
    if rejected_centers:
        print("\nПричины отклонения потенциальных циклонов:")
        reasons = {}
        for center in rejected_centers:
            reason = center["reason"]
            if reason in reasons:
                reasons[reason] += 1
            else:
                reasons[reason] = 1

        for reason, count in reasons.items():
            if reason == "small_size":
                print(f"- Малый размер области: {count} случаев")
            elif reason == "high_pressure":
                print(f"- Высокое давление: {count} случаев")
            elif reason == "shallow_depth":
                print(f"- Недостаточная глубина: {count} случаев")
            elif reason == "no_closed_contour":
                print(f"- Отсутствие замкнутых изобар: {count} случаев")
            elif reason == "weak_gradient":
                print(f"- Слабый градиент давления: {count} случаев")
            elif reason == "low_latitude":
                print(f"- Широта ниже {min_latitude}°N: {count} случаев")
            elif reason == "invalid_size":
                print(f"- Размер вне допустимого диапазона ({min_cyclone_radius}-{max_cyclone_radius} км): {count} случаев")
            elif reason == "weak_wind":
                print(f"- Слабый ветер: {count} случаев")
            elif reason == "weak_vorticity":
                print(f"- Слабая завихренность: {count} случаев")

    print(f"После применения критериев идентифицировано {len(cyclone_centers)} циклонов типа '{cyclone_type}'")
    cyclones_found = len(cyclone_centers) > 0

    # Создаем новую маску, содержащую только валидные циклоны
    valid_mask = np.isin(labeled_cyclones, valid_cyclones)
    
    # Собираем диагностические данные
    diagnostic_data = {
        "all_potential_centers": all_potential_centers,
        "rejected_centers": rejected_centers,
        "gradient_magnitude": gradient_magnitude,
        "valid_mask": valid_mask,
        "labeled_cyclones": labeled_cyclones,
        "grid_info": grid_info,
        "wind_field": wind_field if has_wind_data else None,
        "vorticity_field": vorticity_field if has_vorticity_data else None,
        "has_wind_data": has_wind_data,
        "has_vorticity_data": has_vorticity_data,
        "min_wind_speed": min_wind_speed,
        "min_vorticity": min_vorticity,
        "cyclone_type": cyclone_type,
        "cyclone_params": cyclone_params,
        "detection_methods":detection_methods
    }

    return pressure, laplacian, cyclone_centers, cyclones_found, valid_mask, diagnostic_data