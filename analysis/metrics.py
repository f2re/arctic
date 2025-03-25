# Расчет метрик циклонов
"""
Функции для расчета метрик и характеристик циклонов.
"""

import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Вычисляет расстояние между точками на поверхности Земли по формуле гаверсинуса.
    Поддерживает как скалярные, так и векторные вычисления.
    
    Параметры:
    ----------
    lat1, lon1 : float
        Координаты первой точки (в градусах)
    lat2, lon2 : float или numpy.ndarray
        Координаты второй точки или массивы координат (в градусах)
        
    Возвращает:
    -----------
    float или numpy.ndarray
        Расстояние в километрах или массив расстояний
    """
    # Реализация без изменений из оригинального файла
    # Радиус Земли в километрах
    R = 6371.0
    
    # Проверяем, являются ли входные данные массивами
    vectorized_input = isinstance(lat2, np.ndarray) or isinstance(lon2, np.ndarray)
    
    # Преобразуем градусы в радианы
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    
    if vectorized_input:
        # Векторизованная версия
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
    else:
        # Скалярная версия
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
    
    # Разница в координатах
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Формула гаверсинуса
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Расстояние в километрах
    distance = R * c
    
    return distance

def calculate_okubo_weiss(u, v, dx, dy):
    """
    Рассчитывает параметр Okubo-Weiss на основе полей скорости.
    
    Параметры:
    ----------
    u, v : numpy.ndarray
        Компоненты скорости
    dx, dy : float
        Шаг сетки по x и y в метрах
        
    Возвращает:
    -----------
    numpy.ndarray
        Поле параметра Okubo-Weiss
    tuple
        Компоненты параметра (нормальная деформация, сдвиговая деформация, завихренность)
    """
    # Вычисляем производные скорости
    du_dx = np.gradient(u, axis=1) / dx
    du_dy = np.gradient(u, axis=0) / dy
    dv_dx = np.gradient(v, axis=1) / dx
    dv_dy = np.gradient(v, axis=0) / dy
    
    # Нормальная деформация
    normal_strain = du_dx - dv_dy
    
    # Сдвиговая деформация
    shear_strain = dv_dx + du_dy
    
    # Относительная завихренность
    relative_vorticity = dv_dx - du_dy
    
    # Параметр Okubo-Weiss
    okubo_weiss = normal_strain**2 + shear_strain**2 - relative_vorticity**2
    
    return okubo_weiss, (normal_strain, shear_strain, relative_vorticity)

def calculate_sst_gradients(sst, dx, dy):
    """
    Вычисляет градиенты температуры поверхности моря.
    
    Параметры:
    ----------
    sst : numpy.ndarray
        Поле температуры поверхности моря
    dx, dy : float
        Шаг сетки по x и y в метрах
        
    Возвращает:
    -----------
    tuple
        Градиенты SST по x и y, величина градиента
    """
    # Вычисляем градиенты
    sst_dx = np.gradient(sst, axis=1) / dx
    sst_dy = np.gradient(sst, axis=0) / dy
    
    # Величина градиента
    sst_gradient_magnitude = np.sqrt(sst_dx**2 + sst_dy**2)
    
    return sst_dx, sst_dy, sst_gradient_magnitude