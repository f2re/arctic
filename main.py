# Главный скрипт запуска
"""
Главный скрипт для запуска обнаружения арктических циклонов.
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import warnings
import json
import pickle

# Подавление предупреждений
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Импорт модулей

from config import *
from data.download import setup_cdsapirc, download_era5_data_extended, inspect_netcdf
from data.preprocessing import analyze_grid_scale, adapt_detection_params
from detection.algorithms import detect_cyclones_improved
from detection.thermal import classify_cyclones
from visualization.plots import visualize_cyclones_with_diagnostics, create_cyclone_statistics, visualize_detection_methods_effect
from visualization.diagnostics import visualize_detection_criteria
from analysis.tracking import track_cyclones
from analysis.metrics import haversine_distance
from detection.parameters import get_cyclone_params


def is_running_in_colab():
    """
    Проверяет, запущен ли код в Google Colab.
    
    Возвращает:
    -----------
    bool
        True, если код выполняется в Google Colab, False - если на локальном ПК
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def mount_drive_and_setup():
    """
    Монтирование Google Drive (если в Colab) или настройка локальных каталогов и создание
    необходимой структуры директорий для хранения данных.
    """
    import os
    from pathlib import Path
    
    is_colab = is_running_in_colab()
    print(f"Определено окружение: {'Google Colab' if is_colab else 'Локальный компьютер'}")
    
    if is_colab:
        # Настройка для Google Colab
        from google.colab import drive
        print("Монтирование Google Drive...")
        drive.mount('/content/drive')
        
        # Создаем базовый каталог для хранения данных
        arctic_dir = '/content/drive/MyDrive/arctic'
    else:
        # Настройка для локального ПК
        # Используем директорию в домашнем каталоге пользователя
        arctic_dir = os.path.join(os.path.expanduser('~'), 'arctic_git')
        print(f"Использование локальной директории: {arctic_dir}")
    
    # Общая настройка структуры каталогов для обеих сред
    data_dir = os.path.join(arctic_dir, 'data', 'nc')
    image_dir = os.path.join(arctic_dir, 'images')
    checkpoint_dir = os.path.join(arctic_dir, 'checkpoints')
    model_dir = os.path.join(arctic_dir, 'models')

    # Создаем структуру каталогов, если они не существуют
    for directory in [arctic_dir, data_dir, image_dir, checkpoint_dir, model_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Каталог создан или уже существует: {directory}")

    return arctic_dir, data_dir, image_dir, checkpoint_dir, model_dir

def process_era5_data(file_path, output_dir, checkpoint_dir, model_dir, resume=True, 
                  save_diagnostic=False, use_daily_step=True, cyclone_type='mesoscale',
                  detection_methods=None):
    """
    Оптимизированная обработка данных ERA5 и обнаружение циклонов с ежедневным шагом по времени.
    Включает отслеживание циклонов во времени и проверку дополнительных критериев для мезомасштабных циклонов.

    Параметры:
    ----------
    file_path : str
        Путь к файлу с данными ERA5
    output_dir : str
        Директория для сохранения изображений
    checkpoint_dir : str
        Директория для сохранения контрольных точек
    model_dir : str
        Директория для моделей и параметров
    resume : bool
        Возобновить обработку с последней контрольной точки
    save_diagnostic : bool
        Сохранять ли диагностические изображения
    use_daily_step : bool
        Использовать ежедневный шаг (вместо обработки каждого временного шага)
    cyclone_type : str
        Тип циклонов для обнаружения: 'synoptic', 'mesoscale' или 'polar_low'
    detection_methods : list, optional
        Список методов обнаружения циклонов для использования. 
        Возможные значения: 'laplacian', 'pressure_minima', 'closed_contour', 'gradient', 'vorticity', 'wind_speed'.
        По умолчанию используются все доступные методы для выбранного типа циклонов.

    Возвращает:
    -----------
    list
        Список обнаруженных циклонов
    """
    print(f"Обработка данных из файла: {file_path}")

    # Установка методов обнаружения по умолчанию, если не указаны
    if detection_methods is None:
        # Методы по умолчанию для каждого типа циклонов
        if cyclone_type == 'synoptic':
            detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 'gradient']
        elif cyclone_type == 'mesoscale':
            detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 'gradient', 'vorticity', 'wind_speed']
        else:  # polar_low
            detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 'gradient', 'vorticity', 'wind_speed']
    
    print(f"Используемые методы обнаружения: {', '.join(detection_methods)}")
    
    # Автоматическое определение необходимых параметров реанализа ERA5
    required_variables = determine_required_era5_variables(detection_methods)
    print(f"Необходимые переменные реанализа ERA5: {', '.join(required_variables)}")

    # Загружаем параметры алгоритма - используем MESOSCALE_CYCLONE_PARAMS для мезовихрей
    cyclone_params = MESOSCALE_CYCLONE_PARAMS if 'MESOSCALE_CYCLONE_PARAMS' in globals() else OPTIMIZED_CYCLONE_PARAMS

    # Проверяем наличие контрольной точки
    checkpoint_file = os.path.join(checkpoint_dir, f"cyclone_checkpoint_{os.path.basename(file_path)}.pkl")
    start_idx = 0
    all_cyclones = []
    tracked_cyclones = {}  # Словарь для отслеживания циклонов во времени
    
    if resume and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                start_idx = checkpoint_data.get('last_processed_idx', 0) + 1
                all_cyclones = checkpoint_data.get('all_cyclones', [])
                tracked_cyclones = checkpoint_data.get('tracked_cyclones', {})
                print(f"Загружена контрольная точка. Продолжаем с шага {start_idx}")
        except Exception as e:
            print(f"Ошибка при загрузке контрольной точки: {e}. Начинаем с начала.")
            start_idx = 0
            all_cyclones = []
            tracked_cyclones = {}

    # Исследуем структуру файла перед загрузкой
    file_info = inspect_netcdf(file_path)

    if not file_info:
        print("Не удалось проанализировать структуру файла.")
        return []

    # Загружаем данные
    ds = xr.open_dataset(file_path)

    # Определяем ключевые переменные на основе анализа файла
    if 'time_dim' in file_info:
        time_dim = file_info['time_dim']
    else:
        # Пытаемся определить временное измерение
        for dim in ds.dims:
            if dim in ds and hasattr(ds[dim], 'units'):
                if 'since' in ds[dim].units:
                    print(f"Использую {dim} в качестве временного измерения.")
                    time_dim = dim
                    break
        else:
            print("ОШИБКА: Невозможно обработать файл без временного измерения.")
            ds.close()
            return []

    # Определяем переменную давления
    if 'pressure_vars' in file_info and file_info['pressure_vars']:
        pressure_var = file_info['pressure_vars'][0]
    else:
        # Ищем переменные приземного давления
        possible_vars = [var for var in ds.variables if 'pressure' in var.lower() or 'msl' in var.lower()]
        if possible_vars:
            pressure_var = possible_vars[0]
            print(f"Используется переменная давления: {pressure_var}")
        else:
            print("ОШИБКА: В файле не найдены переменные приземного давления.")
            ds.close()
            return []

    # Определяем переменные координат
    if 'lat_var' in file_info:
        lat_var = file_info['lat_var']
    elif 'latitude' in ds:
        lat_var = 'latitude'
    elif 'lat' in ds:
        lat_var = 'lat'
    else:
        print("ОШИБКА: Не удалось определить переменную широты.")
        ds.close()
        return []

    if 'lon_var' in file_info:
        lon_var = file_info['lon_var']
    elif 'longitude' in ds:
        lon_var = 'longitude'
    elif 'lon' in ds:
        lon_var = 'lon'
    else:
        print("ОШИБКА: Не удалось определить переменную долготы.")
        ds.close()
        return []

    # Определяем количество временных шагов
    if time_dim in ds.dims:
        time_steps = ds.dims[time_dim]
    else:
        # Если нет временного измерения, обрабатываем как один шаг
        time_steps = 1

    print(f"Количество временных шагов для обработки: {time_steps}")

    # Получаем временные метки и определяем временной шаг для обработки
    time_step = 1  # по умолчанию обрабатываем каждый шаг
    time_values = None
    hours_per_step = 0

    if time_steps > 1 and time_dim in ds:
        time_values = ds[time_dim].values
        
        # Проверяем, что это массив datetime64
        if np.issubdtype(time_values.dtype, np.datetime64):
            # Определяем шаг по времени в часах
            if len(time_values) > 1:
                time_delta = np.diff(time_values)[0]
                hours_delta = time_delta.astype('timedelta64[h]').astype(int)
                hours_per_step = hours_delta
                
                # Если данные с шагом менее суток и нужен ежедневный шаг
                if use_daily_step and hours_delta < 24:
                    time_step = 24 // hours_delta  # шаг для получения ежедневных данных
                    print(f"Временной шаг данных: {hours_delta} часов")
                    print(f"Устанавливаем шаг обработки: каждый {time_step}-й шаг (ежедневно)")

    

    # Получаем координаты для вычисления расстояний
    lat_values = ds[lat_var].values
    lon_values = ds[lon_var].values

    # Определяем минимальную продолжительность циклона
    min_duration_hours = cyclone_params.get('min_duration', 24)
    print(f"Минимальная продолжительность циклонов: {min_duration_hours} часов")

    # Обрабатываем временные шаги с заданным шагом
    for time_idx in range(start_idx, time_steps, time_step):
        if time_dim in ds.dims:
            current_time = ds[time_dim].values[time_idx]
            time_str = str(current_time)
        else:
            time_str = f"step_{time_idx}"

        print(f"\nОбработка временного шага {time_idx+1}/{time_steps}: {time_str}")
        
        
        if 'u10' in ds.variables and 'v10' in ds.variables:
            print(f"Найдены компоненты ветра: u10, v10. Вычисляю скорость ветра для всего датасета...")
            
            # Вычисляем скорость ветра для всего датасета
            wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
            
            # Добавляем вычисленную переменную как новую переменную датасета
            ds['10m_wind_speed'] = wind_speed
            has_wind_data = True
            print(f"Скорость ветра успешно вычислена для всего датасета")
        else:
            has_wind_data = False
            print("ВНИМАНИЕ: Компоненты ветра не найдены. Критерий скорости ветра не будет применяться.")


        # Аналогично для завихренности
        if 'vo' in ds.variables:
            print(f"Найдена переменная завихренности: vo. Подготовка данных...")
            
            # Определяем, зависит ли завихренность от времени и уровня давления
            if time_dim in ds['vo'].dims:
                if 'pressure_level' in ds['vo'].dims:
                    # Выбираем уровень 850 гПа или ближайший доступный
                    pressure_levels = ds['pressure_level'].values
                    target_level = 850
                    idx_level = np.abs(pressure_levels - target_level).argmin()
                    pressure_level = pressure_levels[idx_level]
                    
                    vorticity_field = ds['vo'].isel({time_dim: time_idx, 'pressure_level': idx_level}).values
                    print(f"Используется завихренность на уровне {pressure_level} гПа")
                else:
                    vorticity_field = ds['vo'].isel({time_dim: time_idx}).values
            else:
                vorticity_field = ds['vo'].values
            
            # Проверяем размерность массива
            if len(vorticity_field.shape) > 2:
                vorticity_field = vorticity_field[0] if vorticity_field.shape[0] <= 10 else vorticity_field
            
            # Добавляем вычисленную переменную в набор данных
            ds['vorticity'] = (('latitude', 'longitude'), vorticity_field)
            has_vorticity_data = True
            print(f"Завихренность успешно подготовлена. Диапазон значений: {np.min(vorticity_field):.2e} - {np.max(vorticity_field):.2e} с⁻¹")
        else:
            has_vorticity_data = False
            print("ВНИМАНИЕ: Переменная завихренности не найдена. Критерий завихренности не будет применяться.")

        # Модификация блока обработки данных о ветре
        has_wind_data = False
        wind_field = None

        if '10m_wind_speed' in ds.variables:
            has_wind_data = True
            print(f"Найдены данные о скорости ветра: переменная '10m_wind_speed'")
            wind_field = ds['10m_wind_speed']
        elif 'u10' in ds.variables and 'v10' in ds.variables:
            has_wind_data = True
            print(f"Найдены компоненты ветра: u10, v10. Вычисляю скорость ветра для всего датасета...")
            
            # Вычисляем скорость ветра для всего датасета
            wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
            
            # Добавляем вычисленную переменную как новую переменную датасета
            ds['10m_wind_speed'] = wind_speed
            wind_field = ds['10m_wind_speed']
            print(f"Скорость ветра успешно вычислена для всего датасета")
        elif '10m_u_component_of_wind' in ds.variables and '10m_v_component_of_wind' in ds.variables:
            has_wind_data = True
            print(f"Найдены компоненты ветра: 10m_u_component_of_wind, 10m_v_component_of_wind. Вычисляю скорость ветра...")
            
            # Вычисляем скорость ветра для всего датасета
            wind_speed = np.sqrt(ds['10m_u_component_of_wind']**2 + ds['10m_v_component_of_wind']**2)
            
            # Добавляем вычисленную переменную как новую переменную датасета
            ds['10m_wind_speed'] = wind_speed
            wind_field = ds['10m_wind_speed']
            print(f"Скорость ветра успешно вычислена для всего датасета")
        else:
            print("ВНИМАНИЕ: Данные о скорости ветра не найдены. Критерий скорости ветра не будет применяться.")

        vorticity_var = next((var for var in ds.variables if 'vo' in var.lower()), None)
        has_vorticity_data = vorticity_var is not None
        
        if has_wind_data:
            print(f"Найдена переменная скорости ветра: 10m_wind_speed")
        else:
            print("ВНИМАНИЕ: Переменная скорости ветра не найдена. Критерий скорости ветра не будет применяться.")
            
        if has_vorticity_data:
            print(f"Найдена переменная завихренности: {vorticity_var}")
        else:
            print("ВНИМАНИЕ: Переменная завихренности не найдена. Критерий завихренности не будет применяться.")

        # Первый шаг: обнаружение циклонов
        # При вызове функции обнаружения циклонов передаем выбранные методы
        pressure, laplacian, cyclone_centers, cyclones_found, cyclone_mask, diagnostic_data = detect_cyclones_improved(
            ds, time_idx, time_dim, pressure_var, lat_var, lon_var, 
            cyclone_params=cyclone_params,
            cyclone_type=cyclone_type,
            detection_methods=detection_methods
        )
        
        # Второй шаг: дополнительные критерии для мезомасштабных циклонов
        if cyclones_found and len(cyclone_centers) > 0:
            # Проверка скорости ветра
            if has_wind_data:
                try:
                    # Получаем поле скорости ветра
                    if time_dim in ds['10m_wind_speed'].dims:
                        wind_speed = ds['10m_wind_speed'].isel({time_dim: time_idx}).values
                    else:
                        wind_speed = ds['10m_wind_speed'].values
                    
                    # Проверяем размерность массива
                    if len(wind_speed.shape) > 2:
                        wind_speed = wind_speed[0] if wind_speed.shape[0] <= 10 else wind_speed
                    
                    # Минимальная скорость ветра для арктического мезовихря
                    min_wind_speed = cyclone_params.get('min_wind_speed', 15.0)
                    
                    # Проверка превышения скорости ветра для каждого циклона
                    filtered_centers = []
                    for cyclone_center in cyclone_centers:
                        lat, lon = cyclone_center[:2]
                        lat_idx = np.abs(lat_values - lat).argmin()
                        lon_idx = np.abs(lon_values - lon).argmin()
                        
                        # Находим максимальную скорость ветра в окрестности циклона
                        neighborhood_radius = 5  # ячеек сетки
                        lat_min = max(0, lat_idx - neighborhood_radius)
                        lat_max = min(wind_speed.shape[0], lat_idx + neighborhood_radius + 1)
                        lon_min = max(0, lon_idx - neighborhood_radius)
                        lon_max = min(wind_speed.shape[1], lon_idx + neighborhood_radius + 1)
                        
                        local_wind = wind_speed[lat_min:lat_max, lon_min:lon_max]
                        max_wind = np.max(local_wind) if local_wind.size > 0 else 0
                        
                        # Если скорость ветра превышает порог, сохраняем циклон
                        if max_wind >= min_wind_speed:
                            # Добавляем информацию о скорости ветра
                            cyclone_center += (max_wind,)
                            filtered_centers.append(cyclone_center)
                        else:
                            print(f"Циклон на {lat:.2f}°N, {lon:.2f}°E отклонен: скорость ветра {max_wind:.1f} м/с < {min_wind_speed} м/с")
                    
                    # Обновляем список центров циклонов
                    cyclone_centers = filtered_centers
                    cyclones_found = len(cyclone_centers) > 0
                    
                except Exception as e:
                    print(f"Ошибка при проверке скорости ветра: {e}")
            
            # Проверка завихренности
            if has_vorticity_data and cyclones_found:
                try:
                    # Получаем поле завихренности
                    if time_dim in ds[vorticity_var].dims:
                        vorticity = ds[vorticity_var].isel({time_dim: time_idx}).values
                    else:
                        vorticity = ds[vorticity_var].values
                    
                    # Проверяем размерность массива
                    if len(vorticity.shape) > 2:
                        vorticity = vorticity[0] if vorticity.shape[0] <= 10 else vorticity
                    
                    # Минимальная завихренность для арктического мезовихря
                    min_vorticity = cyclone_params.get('min_vorticity', 0.5e-5)
                    
                    # Проверка завихренности для каждого циклона
                    filtered_centers = []
                    for cyclone_center in cyclone_centers:
                        lat, lon = cyclone_center[:2]
                        lat_idx = np.abs(lat_values - lat).argmin()
                        lon_idx = np.abs(lon_values - lon).argmin()
                        
                        # Находим максимальную завихренность в окрестности циклона
                        neighborhood_radius = 5  # ячеек сетки
                        lat_min = max(0, lat_idx - neighborhood_radius)
                        lat_max = min(vorticity.shape[0], lat_idx + neighborhood_radius + 1)
                        lon_min = max(0, lon_idx - neighborhood_radius)
                        lon_max = min(vorticity.shape[1], lon_idx + neighborhood_radius + 1)
                        
                        local_vorticity = vorticity[lat_min:lat_max, lon_min:lon_max]
                        max_vorticity = np.max(local_vorticity) if local_vorticity.size > 0 else 0
                        
                        # Если завихренность превышает порог, сохраняем циклон
                        if max_vorticity >= min_vorticity:
                            # Добавляем информацию о завихренности
                            cyclone_center += (max_vorticity,)
                            filtered_centers.append(cyclone_center)
                        else:
                            print(f"Циклон на {lat:.2f}°N, {lon:.2f}°E отклонен: завихренность {max_vorticity:.2e} с^-1 < {min_vorticity} с^-1")
                    
                    # Обновляем список центров циклонов
                    cyclone_centers = filtered_centers
                    cyclones_found = len(cyclone_centers) > 0
                    
                except Exception as e:
                    print(f"Ошибка при проверке завихренности: {e}")
        
        # Третий шаг: Визуализация результатов
        # При вызове функции визуализации также передаем выбранные методы
        if cyclones_found:
            visualize_cyclones_with_diagnostics(
                ds, time_idx, time_dim, pressure_var, lat_var, lon_var,
                output_dir, cyclone_params=cyclone_params,
                save_diagnostic=save_diagnostic, file_prefix="meso_",
                detection_methods=detection_methods
            )
        
        # Четвертый шаг: Отслеживание циклонов во времени
        current_tracked = {}  # Словарь для текущего шага
        
        # Обновляем информацию о циклонах и добавляем в список
        for cyclone_center in cyclone_centers:
            lat, lon = cyclone_center[:2]
            pressure = cyclone_center[2]
            depth = cyclone_center[3]
            gradient = cyclone_center[4]
            radius = cyclone_center[5]
            
            # Дополнительные поля, если они есть
            max_wind = cyclone_center[6] if len(cyclone_center) > 6 else None
            max_vorticity = cyclone_center[7] if len(cyclone_center) > 7 else None
            
            # Создаем идентификатор циклона на основе координат
            cyclone_id = f"{lat:.2f}_{lon:.2f}"
            
            # Проверяем, есть ли уже такой циклон в списке отслеживания
            is_tracked = False
            for track_id, track_info in tracked_cyclones.items():
                last_lat = track_info['last_lat']
                last_lon = track_info['last_lon']
                
                # Вычисляем расстояние между последним положением и текущим
                # Используем haversine_distance
                distance = haversine_distance(lat, lon, last_lat, last_lon)
                
                # Если расстояние меньше допустимого порога, считаем циклоны одним и тем же
                if distance < 300:  # 300 км - типичный порог для отслеживания
                    # Обновляем информацию о циклоне
                    tracked_cyclones[track_id]['last_lat'] = lat
                    tracked_cyclones[track_id]['last_lon'] = lon
                    tracked_cyclones[track_id]['last_time'] = time_str
                    tracked_cyclones[track_id]['positions'].append((lat, lon))
                    tracked_cyclones[track_id]['times'].append(time_str)
                    tracked_cyclones[track_id]['duration'] += hours_per_step
                    
                    # Обновляем минимальное давление если текущее меньше
                    if pressure < tracked_cyclones[track_id]['min_pressure']:
                        tracked_cyclones[track_id]['min_pressure'] = pressure
                        tracked_cyclones[track_id]['max_depth'] = depth
                    
                    # Используем трек_id вместо cyclone_id
                    cyclone_id = track_id
                    is_tracked = True
                    break
            
            # Если это новый циклон, добавляем его в словарь отслеживания
            if not is_tracked:
                tracked_cyclones[cyclone_id] = {
                    'start_time': time_str,
                    'last_time': time_str,
                    'first_lat': lat,
                    'first_lon': lon,
                    'last_lat': lat,
                    'last_lon': lon,
                    'min_pressure': pressure,
                    'max_depth': depth,
                    'positions': [(lat, lon)],
                    'times': [time_str],
                    'duration': hours_per_step
                }
            
            # Добавляем циклон в текущий список отслеживания
            current_tracked[cyclone_id] = True
            
            # Создаем запись о циклоне с дополнительной информацией
            cyclone_info = {
                'time': time_str,
                'latitude': lat,
                'longitude': lon,
                'pressure': pressure,  # гПа
                'depth': depth,        # гПа
                'gradient': gradient,  # гПа/градус
                'radius': radius,      # км
                'track_id': cyclone_id
            }
            
            # Добавляем дополнительные поля, если они есть
            if max_wind is not None:
                cyclone_info['max_wind'] = max_wind  # м/с
            
            if max_vorticity is not None:
                cyclone_info['max_vorticity'] = max_vorticity  # с^-1
            
            # Добавляем информацию о продолжительности (если это отслеживаемый циклон)
            if is_tracked:
                cyclone_info['duration'] = tracked_cyclones[cyclone_id]['duration']
            
            # Добавляем в общий список циклонов
            all_cyclones.append(cyclone_info)
        
        # Сохраняем контрольную точку после каждого шага
        if time_idx % 5 == 0 or time_idx == time_steps - 1:
            try:
                checkpoint_data = {
                    'last_processed_idx': time_idx,
                    'all_cyclones': all_cyclones,
                    'tracked_cyclones': tracked_cyclones
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"Сохранена контрольная точка после шага {time_idx}")
            except Exception as e:
                print(f"Ошибка при сохранении контрольной точки: {e}")

    # Закрываем датасет
    ds.close()

    # Фильтрация циклонов по минимальной продолжительности
    if min_duration_hours > 0:
        print(f"\nПроверка минимальной продолжительности циклонов ({min_duration_hours} часов)...")
        original_count = len(all_cyclones)
        
        # Фильтруем по трек_id и продолжительности
        filtered_cyclones = []
        track_durations = {}
        
        # Сначала определяем продолжительность для каждого трек_id
        for track_id, track_info in tracked_cyclones.items():
            if track_info['duration'] >= min_duration_hours:
                track_durations[track_id] = True
        
        # Затем фильтруем циклоны по наличию трек_id в списке допустимых
        for cyclone in all_cyclones:
            track_id = cyclone.get('track_id')
            if track_id in track_durations:
                filtered_cyclones.append(cyclone)
        
        all_cyclones = filtered_cyclones
        print(f"Отфильтровано {original_count - len(all_cyclones)} циклонов с продолжительностью менее {min_duration_hours} часов")
        print(f"Осталось {len(all_cyclones)} циклонов")

    # Сохраняем информацию о всех обнаруженных циклонах
    if all_cyclones:
        cyclones_df = pd.DataFrame(all_cyclones)
        csv_file = os.path.join(output_dir, f'detected_mesoscale_cyclones.csv')
        cyclones_df.to_csv(csv_file, index=False)
        print(f"\nОбнаружено всего {len(all_cyclones)} мезоциклонических систем")
        print(f"Данные о циклонах сохранены в файл: {csv_file}")
        
        # Сохраняем информацию о траекториях циклонов
        tracks_data = []
        for track_id, track_info in tracked_cyclones.items():
            if track_info['duration'] >= min_duration_hours:
                tracks_data.append({
                    'track_id': track_id,
                    'start_time': track_info['start_time'],
                    'end_time': track_info['last_time'],
                    'duration_hours': track_info['duration'],
                    'min_pressure': track_info['min_pressure'],
                    'max_depth': track_info['max_depth'],
                    'positions': track_info['positions'],
                    'times': track_info['times']
                })
        
        # Сохраняем траектории в отдельный файл
        tracks_file = os.path.join(output_dir, f'cyclone_tracks.json')
        with open(tracks_file, 'w') as f:
            # Применяем функцию преобразования типов данных перед сериализацией
            tracks_data_converted = convert_numpy_types(tracks_data)
            json.dump(tracks_data_converted, f, indent=2)
        print(f"Информация о {len(tracks_data)} траекториях циклонов сохранена в файл: {tracks_file}")
    else:
        print("\nНе обнаружено циклонических систем")

    return all_cyclones

def determine_required_era5_variables(detection_methods, analysis_level='basic'):
    """
    Определяет необходимые переменные реанализа ERA5 на основе выбранных методов обнаружения
    и уровня детализации анализа.
    
    Параметры:
    ----------
    detection_methods : list
        Список методов обнаружения циклонов
    analysis_level : str, optional
        Уровень детализации анализа: 'basic', 'extended', 'comprehensive'
        
    Возвращает:
    -----------
    dict
        Информация о необходимых данных ERA5: 
        {
            'variables': список всех переменных,
            'single_level_vars': список однослойных переменных,
            'pressure_level_vars': список многослойных переменных,
            'pressure_levels': список уровней давления,
            'variables_by_level': словарь с переменными для каждого уровня
        }
    """
    # Инициализируем структуры данных
    single_level_vars = ['msl']  # Базовые переменные (приземное давление всегда нужно)
    pressure_level_vars = {}     # Словарь {уровень: [переменные]}
    
    # Добавляем переменные в зависимости от выбранных методов
    if 'wind_speed' in detection_methods:
        # Для расчета скорости ветра нужны компоненты ветра на 10м
        single_level_vars.extend(['u10', 'v10', '10m_u_component_of_wind', '10m_v_component_of_wind'])
    
    if 'vorticity' in detection_methods:
        # Для анализа завихренности нужна переменная vorticity (vo)
        level = '850'  # Базовый уровень для завихренности
        if level not in pressure_level_vars:
            pressure_level_vars[level] = []
        pressure_level_vars[level].append('vo')
        
        # Добавляем дополнительные уровни в зависимости от уровня детализации
        if analysis_level in ['extended', 'comprehensive']:
            additional_levels = ['925', '700']
            for level in additional_levels:
                if level not in pressure_level_vars:
                    pressure_level_vars[level] = []
                pressure_level_vars[level].append('vo')
                
        if analysis_level == 'comprehensive':
            additional_levels = ['500', '300']
            for level in additional_levels:
                if level not in pressure_level_vars:
                    pressure_level_vars[level] = []
                pressure_level_vars[level].append('vo')
    
    if 'thermal' in detection_methods:
        # Для определения термической структуры нужна температура на 700 гПа
        level = '700'
        if level not in pressure_level_vars:
            pressure_level_vars[level] = []
        pressure_level_vars[level].append('t')
        
        # Добавляем дополнительные уровни и переменные в зависимости от уровня детализации
        if analysis_level in ['extended', 'comprehensive']:
            additional_levels = ['850', '925']
            for level in additional_levels:
                if level not in pressure_level_vars:
                    pressure_level_vars[level] = []
                pressure_level_vars[level].append('t')
            
            # Добавляем геопотенциальную высоту для анализа термической структуры
            thermal_levels = ['925', '850', '700', '500']
            for level in thermal_levels:
                if level in pressure_level_vars:
                    pressure_level_vars[level].append('z')
    
    # Добавляем дополнительные переменные для расширенного анализа
    if analysis_level in ['extended', 'comprehensive']:
        if 'wind_speed' in detection_methods:
            # Добавляем компоненты ветра на разных уровнях
            for level in pressure_level_vars.keys():
                pressure_level_vars[level].extend(['u', 'v'])
        
        if 'sst_gradient' in detection_methods or analysis_level == 'comprehensive':
            single_level_vars.append('sst')  # Температура поверхности моря
    
    # Добавляем переменные для комплексного анализа
    if analysis_level == 'comprehensive':
        # Влагосодержание и осадки для полного анализа
        single_level_vars.extend(['tcwv', 'tp'])
        
        # Добавляем удельную влажность на разных уровнях
        for level in ['925', '850', '700']:
            if level in pressure_level_vars:
                pressure_level_vars[level].append('q')
    
    # Формируем общий список многослойных переменных
    all_pressure_level_variables = []
    for level_vars in pressure_level_vars.values():
        for var in level_vars:
            if var not in all_pressure_level_variables:
                all_pressure_level_variables.append(var)
    
    # Удаляем дубликаты и сортируем
    single_level_vars = sorted(list(set(single_level_vars)))
    all_pressure_level_variables = sorted(list(set(all_pressure_level_variables)))
    pressure_levels = sorted(list(pressure_level_vars.keys()), key=lambda x: int(x))
    
    # Оптимизируем словарь переменных по уровням (удаляем дубликаты)
    for level in pressure_level_vars:
        pressure_level_vars[level] = sorted(list(set(pressure_level_vars[level])))
    
    # Формируем результат
    result = {
        'variables': single_level_vars + all_pressure_level_variables,
        'single_level_vars': single_level_vars,
        'pressure_level_vars': all_pressure_level_variables,
        'pressure_levels': pressure_levels,
        'variables_by_level': pressure_level_vars
    }
    
    return result

# Добавьте эту функцию после импортов
def convert_numpy_types(obj):
    """
    Рекурсивно преобразует numpy типы данных в стандартные типы Python
    для корректной сериализации в JSON.
    
    Параметры:
    ----------
    obj : любой объект
        Исходный объект для преобразования
        
    Возвращает:
    -----------
    объект
        Преобразованный объект с стандартными типами Python
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

def main():
    """
    Главная функция для запуска процесса обнаружения арктических циклонов.
    """
    # Настройка окружения с автоматическим определением Google Colab или локального ПК
    arctic_dir, data_dir, image_dir, checkpoint_dir, model_dir = mount_drive_and_setup()
    
    # Настройка доступа к API CDS для загрузки данных ERA5
    setup_cdsapirc(arctic_dir)
    
    # Настраиваемые параметры: период анализа
    start_date = "2020-01-01"
    end_date = "2020-01-31"  # Тестовый период - один месяц
    region = [90, -180, 65, 180]  # [север, запад, юг, восток]
    save_diagnostic = True  # Сохранять диагностические изображения
    cyclone_type = 'mesoscale'  # Тип обнаруживаемых циклонов
    
    # Выбор уровня детализации анализа
    analysis_level = 'basic'  # Можно выбрать: 'basic', 'extended', 'comprehensive'
    
    # Выбор методов обнаружения циклонов
    # Можно выбрать любую комбинацию из: 'laplacian', 'pressure_minima', 'closed_contour', 
    # 'gradient', 'vorticity', 'wind_speed', 'thermal', 'sst_gradient'
    detection_methods = ['laplacian', 'wind_speed','thermal', 'sst_gradient', 'gradient', 'vorticity']
    
    # Создаем визуализации выбранных критериев обнаружения
    visualize_detection_criteria(detection_methods)
    
    # Определяем необходимые данные ERA5 на основе выбранных методов и уровня анализа
    era5_data_info = determine_required_era5_variables(detection_methods, analysis_level)
    
    print(f"Необходимые данные реанализа ERA5:")
    print(f"  Однослойные переменные: {', '.join(era5_data_info['single_level_vars'])}")
    if era5_data_info['pressure_level_vars']:
        print(f"  Многослойные переменные: {', '.join(era5_data_info['pressure_level_vars'])}")
        print(f"  Уровни давления: {', '.join(era5_data_info['pressure_levels'])}")
    
    # Загрузка данных ERA5 с указанием необходимых параметров
    output_file = f"era5_arctic_{cyclone_type}_{start_date}_{end_date}.nc"
    file_path = download_era5_data_extended(
        start_date=start_date, 
        end_date=end_date, 
        data_dir=data_dir, 
        output_file=output_file,
        region=region,
        era5_data_info=era5_data_info
    )
    
    if file_path is None or not os.path.exists(file_path):
        print("Ошибка: не удалось загрузить или найти файл данных ERA5.")
        return
    
    # Анализ структуры загруженных данных
    file_info = inspect_netcdf(file_path)
    
    # Обработка данных и обнаружение циклонов с выбранными методами
    cyclones = process_era5_data(
        file_path, 
        image_dir, 
        checkpoint_dir, 
        model_dir, 
        resume=True, 
        save_diagnostic=save_diagnostic, 
        use_daily_step=True,
        cyclone_type=cyclone_type,
        detection_methods=detection_methods
    )
    
    print(f"Обработка завершена. Обнаружено циклонов: {len(cyclones) if cyclones else 0}")
    
    # Создание статистических визуализаций
    if cyclones:
        # Создаем базовую статистику
        create_cyclone_statistics(cyclones, image_dir, file_prefix=f"{cyclone_type}_")
        
        # Визуализация эффекта выбранных методов обнаружения
        visualize_detection_methods_effect(
            cyclones, 
            detection_methods, 
            image_dir, 
            file_prefix=f"{cyclone_type}_"
        )


if __name__ == "__main__":
    main()