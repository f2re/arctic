# Классификация термической структуры
"""
Функции для классификации циклонов по термической структуре.
"""

import numpy as np

def determine_thermal_structure(ds, lat_idx, lon_idx, lat_values, lon_values, time_idx, time_dim):
    """
    Определяет термическую структуру циклона на основе распределения температуры на уровне 700 гПа.

    Параметры:
    ----------
    ds : xarray.Dataset
        Набор данных с полями давления и температуры
    lat_idx, lon_idx : int
        Индексы сетки центра циклона
    lat_values, lon_values : numpy.ndarray
        Массивы значений широты и долготы
    time_idx : int
        Индекс временного шага
    time_dim : str
        Имя измерения времени

    Возвращает:
    -----------
    str
        Тип термической структуры: 'cold', 'warm' или 'mixed'
    dict
        Дополнительная информация о термической структуре
    """
    # Реализация без изменений из оригинального файла
    # Проверяем наличие данных температуры на 700 гПа
    temp_var = None
    for var_name in ds.variables:
        if var_name == 'temperature' and '700' in str(ds[var_name].attrs.get('level', '')):
            temp_var = var_name
            break
            
    if temp_var is None:
        # Поиск по любому имени, содержащему 't' и '700'
        for var_name in ds.variables:
            if ('t' in var_name.lower() and 
               ('700' in var_name or '70' in var_name or 
                '700' in str(ds[var_name].attrs.get('level', '')))):
                temp_var = var_name
                break

    # Если нет данных температуры, используем косвенные признаки
    if temp_var is None:
        # Оцениваем термическую структуру на основе параметров циклона
        # Глубокие циклоны с сильным градиентом и небольшим размером чаще теплоядерные
        cyclone_radius, cyclone_info = estimate_cyclone_radius_optimized(
            ds['msl'].isel({time_dim: time_idx}).values / 100.0,
            lat_idx, lon_idx, lat_values, lon_values)

        try:
            # Вычисляем градиент давления
            pressure_field = ds['msl'].isel({time_dim: time_idx}).values / 100.0
            dx = np.deg2rad(lon_values[1] - lon_values[0]) * 6371000 * np.cos(np.deg2rad(75))
            _, (gradient_magnitude, _, _) = calculate_laplacian_improved(
                pressure_field, dx, method='standard', smooth_sigma=0.7)

            # Получаем максимальный градиент в окрестности циклона
            radius = 5  # радиус в ячейках сетки
            lat_min = max(0, lat_idx - radius)
            lat_max = min(pressure_field.shape[0], lat_idx + radius + 1)
            lon_min = max(0, lon_idx - radius)
            lon_max = min(pressure_field.shape[1], lon_idx + radius + 1)
            neighborhood = gradient_magnitude[lat_min:lat_max, lon_min:lon_max]
            max_gradient = np.max(neighborhood)

            # Получаем минимальное давление
            min_pressure = pressure_field[lat_idx, lon_idx]

            # Классифицируем на основе эмпирических критериев
            if min_pressure < 980 and max_gradient > 1.2 and cyclone_radius < 400:
                thermal_type = 'warm'  # Интенсивный компактный циклон с сильным градиентом
            elif min_pressure < 985 and max_gradient > 0.8 and cyclone_radius < 600:
                thermal_type = 'mixed'  # Промежуточный случай
            else:
                thermal_type = 'cold'  # По умолчанию считаем холодным ядром

            structure_info = {
                'min_pressure': min_pressure,
                'max_gradient': max_gradient,
                'radius': cyclone_radius,
                'temp_data_available': False
            }

            return thermal_type, structure_info

        except Exception as e:
            print(f"Ошибка при оценке термической структуры: {e}")
            return 'cold', {'error': str(e), 'temp_data_available': False}

    # Если данные температуры доступны, используем их
    try:
        # Получаем поле температуры на 700 гПа
        if time_dim in ds[temp_var].dims:
            temp_field = ds[temp_var].isel({time_dim: time_idx}).values
        else:
            temp_field = ds[temp_var].values

        # Проверяем размерность массива для многоуровневых данных
        if len(temp_field.shape) > 2:
            # Определяем количество уровней
            num_levels = temp_field.shape[0]

            # Если это вертикальные уровни, выбираем уровень для 700 гПа
            temp_field = temp_field[num_levels // 2] if num_levels <= 10 else temp_field

        # Получаем центральные координаты
        center_lat = lat_values[lat_idx]
        center_lon = lon_values[lon_idx]

        # Создаем маску для точек на периферии (300-500 км от центра)
        inner_radius = 300  # км
        outer_radius = 500  # км

        # Используем оптимизированный подход для расчета расстояний
        lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing='ij')

        # Упрощенный расчет расстояний (в км)
        lat_distances = np.abs(lat_grid - center_lat) * 111.32
        lon_distances = np.abs(lon_grid - center_lon) * 111.32 * np.cos(np.radians(center_lat))
        distances = np.sqrt(lat_distances**2 + lon_distances**2)

        # Маска точек на периферии
        periphery_mask = (distances >= inner_radius) & (distances <= outer_radius)

        # Получаем температуры
        center_temp = temp_field[lat_idx, lon_idx]
        periphery_temps = temp_field[periphery_mask]

        if len(periphery_temps) > 0:
            mean_periphery_temp = np.mean(periphery_temps)

            # Определяем порог для классификации (обычно 1-2K)
            threshold = 1.5  # K

            # Классифицируем на основе разницы температур
            temp_diff = center_temp - mean_periphery_temp

            if temp_diff > threshold:
                thermal_type = 'warm'  # Теплоядерный
            elif temp_diff < -threshold:
                thermal_type = 'cold'  # Холодноядерный
            else:
                thermal_type = 'mixed'  # Смешанный тип

            structure_info = {
                'center_temp': center_temp,
                'mean_periphery_temp': mean_periphery_temp,
                'temp_diff': temp_diff,
                'threshold': threshold,
                'temp_data_available': True
            }
        else:
            # Если не удалось найти точки на периферии
            thermal_type = 'cold'  # По умолчанию
            structure_info = {
                'center_temp': center_temp,
                'error': 'No periphery points found',
                'temp_data_available': True
            }
    except Exception as e:
        print(f"Ошибка при анализе температурного поля: {e}")
        thermal_type = 'cold'  # По умолчанию
        structure_info = {
            'error': str(e),
            'temp_data_available': False
        }

    return thermal_type, structure_info

def classify_cyclones(cyclone_centers, ds, time_idx, time_dim):
    """
    Классифицирует обнаруженные циклоны по термической структуре.

    Параметры:
    ----------
    cyclone_centers : list
        Список центров циклонов
    ds : xarray.Dataset
        Набор данных с полями давления и температуры
    time_idx : int
        Индекс временного шага
    time_dim : str
        Имя измерения времени

    Возвращает:
    -----------
    list
        Список циклонов с добавленной информацией о термической структуре
    dict
        Статистика по типам циклонов
    """
    # Реализация без изменений из оригинального файла
    classified_cyclones = []

    # Счетчики для статистики
    stats = {'cold': 0, 'warm': 0, 'mixed': 0}

    # Получаем координаты
    lat_var = 'latitude' if 'latitude' in ds else 'lat'
    lon_var = 'longitude' if 'longitude' in ds else 'lon'
    lat_values = ds[lat_var].values
    lon_values = ds[lon_var].values

    # Для каждого циклона определяем его термическую структуру
    for center in cyclone_centers:
        # Распаковываем информацию о циклоне - обрабатываем 6 значений
        lat, lon, pressure_value, depth, gradient, radius = center

        # Находим ближайшие индексы в сетке
        lat_idx = np.abs(lat_values - lat).argmin()
        lon_idx = np.abs(lon_values - lon).argmin()

        # Определяем термическую структуру
        core_type, structure_info = determine_thermal_structure(
            ds, lat_idx, lon_idx, lat_values, lon_values, time_idx, time_dim)

        # Обновляем статистику
        stats[core_type] += 1

        # Создаем словарь с информацией о циклоне
        cyclone_info = {
            'latitude': lat,
            'longitude': lon,
            'pressure': pressure_value,
            'depth': depth,
            'gradient': gradient,
            'radius': radius,
            'core_type': core_type,
            'thermal_info': structure_info
        }

        classified_cyclones.append(cyclone_info)

    return classified_cyclones, stats