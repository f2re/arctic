# Предобработка данных
"""
Функции для предобработки данных ERA5.
"""

import numpy as np
import xarray as xr
from scipy import ndimage

def analyze_grid_scale(lat_values, lon_values):
    """
    Анализирует масштаб сетки и выдает рекомендации.

    Параметры:
    ----------
    lat_values : numpy.ndarray
        Массив значений широты
    lon_values : numpy.ndarray
        Массив значений долготы

    Возвращает:
    -----------
    dict
        Словарь с информацией о масштабе сетки и рекомендациями
    """
    # Реализация без изменений из оригинального файла
    # Вычисляем шаг сетки
    if len(lat_values) > 1:
        lat_step = abs(lat_values[1] - lat_values[0])
    else:
        lat_step = 0

    if len(lon_values) > 1:
        lon_step = abs(lon_values[1] - lon_values[0])
    else:
        lon_step = 0

    # Расчет размера сетки в метрах на широте 75°N
    lat_km = lat_step * 111.32  # Приблизительно 111.32 км на градус широты
    lon_km = lon_step * 111.32 * np.cos(np.deg2rad(75))  # Учитываем сжатие на высокой широте

    # Анализ размера сетки
    grid_info = {
        'lat_step_deg': lat_step,
        'lon_step_deg': lon_step,
        'lat_step_km': lat_km,
        'lon_step_km': lon_km,
        'grid_resolution': 'unknown',
        'recommendation': ''
    }

    # Определяем разрешение сетки
    if lat_km > 100 or lon_km > 100:
        grid_info['grid_resolution'] = 'very_coarse'
        grid_info['recommendation'] = ('Сетка очень грубая. Рекомендуется использовать данные с более '
                                      'высоким разрешением для лучшего обнаружения циклонов.')
    elif lat_km > 50 or lon_km > 50:
        grid_info['grid_resolution'] = 'coarse'
        grid_info['recommendation'] = ('Сетка грубая. Лапласиан может быть недостаточно точным. '
                                      'Рекомендуется снизить пороговые значения обнаружения.')
    elif lat_km > 25 or lon_km > 25:
        grid_info['grid_resolution'] = 'medium'
        grid_info['recommendation'] = ('Сетка среднего разрешения. Для лучшего обнаружения мелких '
                                      'циклонов рекомендуется адаптивный подход к расчету лапласиана.')
    elif lat_km > 10 or lon_km > 10:
        grid_info['grid_resolution'] = 'fine'
        grid_info['recommendation'] = ('Сетка хорошего разрешения. Можно использовать стандартные '
                                      'методы обнаружения с небольшими корректировками.')
    else:
        grid_info['grid_resolution'] = 'very_fine'
        grid_info['recommendation'] = ('Сетка очень высокого разрешения. Возможно обнаружение '
                                      'микромасштабных особенностей, рекомендуется фильтрация шума.')

    return grid_info

def adapt_detection_params(params, grid_info):
    """
    Адаптирует параметры обнаружения циклонов в зависимости от масштаба сетки.

    Параметры:
    ----------
    params : dict
        Исходные параметры алгоритма
    grid_info : dict
        Информация о масштабе сетки

    Возвращает:
    -----------
    dict
        Адаптированные параметры
    """
    # Реализация без изменений из оригинального файла
    # Создаем копию параметров для модификации
    adapted_params = params.copy()

    # Адаптируем параметры в зависимости от разрешения сетки
    if grid_info['grid_resolution'] == 'very_coarse':
        # Для очень грубой сетки смягчаем параметры
        adapted_params['laplacian_threshold'] *= 1.3  # Значительно увеличиваем порог
        adapted_params['min_size'] = max(1, params['min_size'] - 1)  # Уменьшаем минимальный размер
        adapted_params['min_depth'] *= 0.8  # Уменьшаем требуемую глубину
        adapted_params['pressure_gradient_threshold'] *= 0.8  # Уменьшаем порог градиента

    elif grid_info['grid_resolution'] == 'coarse':
        # Для грубой сетки смягчаем параметры
        adapted_params['laplacian_threshold'] *= 1.2  # Увеличиваем порог
        adapted_params['min_depth'] *= 0.9  # Уменьшаем требуемую глубину
        adapted_params['pressure_gradient_threshold'] *= 0.9  # Уменьшаем порог градиента

    elif grid_info['grid_resolution'] == 'medium':
        # Для средней сетки небольшие корректировки
        adapted_params['laplacian_threshold'] *= 1.1  # Немного увеличиваем порог

    elif grid_info['grid_resolution'] == 'fine':
        # Для мелкой сетки можно оставить как есть или немного ужесточить
        pass

    elif grid_info['grid_resolution'] == 'very_fine':
        # Для очень мелкой сетки увеличиваем сглаживание для фильтрации шума
        adapted_params['smooth_sigma'] *= 1.2  # Увеличиваем сглаживание
        adapted_params['min_size'] += 1  # Увеличиваем минимальный размер

    return adapted_params