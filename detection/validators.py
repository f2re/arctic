"""
Модуль валидации обнаружения циклонов для системы ArcticCyclone.

Предоставляет классы и функции для проверки корректности обнаруженных
циклонов и фильтрации ложных срабатываний.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

from models.cyclone import Cyclone

# Инициализация логгера
logger = logging.getLogger(__name__)

class DetectionValidator:
    """
    Проверяет корректность обнаруженных циклонов.
    
    Предоставляет методы для проверки обнаруженных циклонов на соответствие
    физическим критериям и фильтрации ложных срабатываний.
    """
    
    def __init__(self, min_vorticity: float = 1e-5,
                min_pressure_gradient: float = 0.5,
                min_wind_speed: float = 8.0):
        """
        Инициализирует валидатор обнаружения.
        
        Аргументы:
            min_vorticity: Минимальное значение завихренности (1/с).
            min_pressure_gradient: Минимальный градиент давления (гПа/100км).
            min_wind_speed: Минимальная скорость ветра (м/с).
        """
        self.min_vorticity = min_vorticity
        self.min_pressure_gradient = min_pressure_gradient
        self.min_wind_speed = min_wind_speed
        
        logger.info(f"Инициализирован валидатор обнаружения циклонов с параметрами: "
                   f"min_vorticity={min_vorticity}, "
                   f"min_pressure_gradient={min_pressure_gradient}, "
                   f"min_wind_speed={min_wind_speed}")
    
    def validate_cyclone(self, cyclone: Cyclone, dataset: xr.Dataset) -> bool:
        """
        Проверяет корректность обнаруженного циклона.
        
        Аргументы:
            cyclone: Объект циклона для проверки.
            dataset: Набор метеорологических данных.
            
        Возвращает:
            True, если циклон прошел проверку, иначе False.
        """
        # Выполняем базовые проверки
        if not self._check_latitude(cyclone):
            logger.debug(f"Циклон не прошел проверку широты: {cyclone.latitude}")
            return False
        
        # Проверяем завихренность (если есть)
        if hasattr(cyclone.parameters, 'vorticity_850hPa'):
            if not self._check_vorticity(cyclone):
                logger.debug(f"Циклон не прошел проверку завихренности: {cyclone.parameters.vorticity_850hPa}")
                return False
        
        # Проверяем градиент давления
        if not self._check_pressure_gradient(cyclone, dataset):
            logger.debug(f"Циклон не прошел проверку градиента давления")
            return False
        
        # Проверяем скорость ветра
        if hasattr(cyclone.parameters, 'max_wind_speed'):
            if not self._check_wind_speed(cyclone):
                logger.debug(f"Циклон не прошел проверку скорости ветра: {cyclone.parameters.max_wind_speed}")
                return False
        
        # Все проверки пройдены
        return True
    
    def _check_latitude(self, cyclone: Cyclone) -> bool:
        """
        Проверяет широту циклона (должна быть в Арктическом регионе).
        
        Аргументы:
            cyclone: Объект циклона для проверки.
            
        Возвращает:
            True, если широта корректна, иначе False.
        """
        return cyclone.latitude >= 65.0
    
    def _check_vorticity(self, cyclone: Cyclone) -> bool:
        """
        Проверяет значение завихренности циклона.
        
        Аргументы:
            cyclone: Объект циклона для проверки.
            
        Возвращает:
            True, если завихренность превышает минимальный порог, иначе False.
        """
        return cyclone.parameters.vorticity_850hPa > self.min_vorticity
    
    def _check_pressure_gradient(self, cyclone: Cyclone, dataset: xr.Dataset) -> bool:
        """
        Проверяет градиент давления вокруг центра циклона.
        
        Аргументы:
            cyclone: Объект циклона для проверки.
            dataset: Набор метеорологических данных.
            
        Возвращает:
            True, если градиент давления превышает минимальный порог, иначе False.
        """
        # Определяем переменную давления
        pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
        pressure_var = None
        
        for var in pressure_vars:
            if var in dataset:
                pressure_var = var
                break
        
        if pressure_var is None:
            logger.warning("Не удается определить переменную давления для проверки градиента")
            return True  # Пропускаем проверку, если нет данных
        
        # Извлекаем регион вокруг циклона
        try:
            # Радиус области в градусах (примерно 300 км на средних широтах)
            radius_deg = 3.0
            
            region = dataset.sel(
                latitude=slice(cyclone.latitude - radius_deg, cyclone.latitude + radius_deg),
                longitude=slice(cyclone.longitude - radius_deg, cyclone.longitude + radius_deg),
                time=cyclone.time
            )
            
            # Вычисляем градиент давления
            pressure_field = region[pressure_var]
            dy, dx = np.gradient(pressure_field.values)
            
            # Рассчитываем величину градиента
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            
            # Преобразуем в гПа/100км (приблизительно)
            # Предполагаем среднее расстояние между точками сетки
            grid_spacing_deg = np.mean([
                np.mean(np.diff(region.latitude.values)),
                np.mean(np.diff(region.longitude.values))
            ])
            grid_spacing_km = grid_spacing_deg * 111.0  # приблизительно 111 км на градус
            
            # Преобразуем градиент
            gradient_hPa_per_100km = np.max(gradient_magnitude) * (100.0 / grid_spacing_km)
            
            return gradient_hPa_per_100km >= self.min_pressure_gradient
            
        except Exception as e:
            logger.warning(f"Ошибка при проверке градиента давления: {str(e)}")
            return True  # Пропускаем проверку в случае ошибки
    
    def _check_wind_speed(self, cyclone: Cyclone) -> bool:
        """
        Проверяет скорость ветра вокруг циклона.
        
        Аргументы:
            cyclone: Объект циклона для проверки.
            
        Возвращает:
            True, если скорость ветра превышает минимальный порог, иначе False.
        """
        return cyclone.parameters.max_wind_speed >= self.min_wind_speed
    
    def validate_tracks(self, tracks: List[List[Cyclone]], 
                      min_duration: float = 6.0,
                      min_points: int = 3,
                      min_displacement: float = 100.0) -> List[List[Cyclone]]:
        """
        Проверяет корректность треков циклонов.
        
        Аргументы:
            tracks: Список треков циклонов.
            min_duration: Минимальная продолжительность трека в часах.
            min_points: Минимальное количество точек в треке.
            min_displacement: Минимальное смещение циклона (км).
            
        Возвращает:
            Список треков, прошедших проверку.
        """
        valid_tracks = []
        
        for track in tracks:
            # Проверяем количество точек
            if len(track) < min_points:
                continue
            
            # Проверяем продолжительность
            start_time = track[0].time
            end_time = track[-1].time
            duration = (end_time - start_time).total_seconds() / 3600
            
            if duration < min_duration:
                continue
            
            # Проверяем смещение
            start_lat, start_lon = track[0].latitude, track[0].longitude
            end_lat, end_lon = track[-1].latitude, track[-1].longitude
            
            displacement = self._calculate_distance(
                start_lat, start_lon, end_lat, end_lon)
            
            if displacement < min_displacement:
                continue
            
            # Все проверки пройдены
            valid_tracks.append(track)
        
        logger.info(f"Валидировано {len(valid_tracks)} треков из {len(tracks)}")
        return valid_tracks
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Вычисляет расстояние между двумя точками на сфере (формула гаверсинуса).
        
        Аргументы:
            lat1: Широта первой точки (градусы).
            lon1: Долгота первой точки (градусы).
            lat2: Широта второй точки (градусы).
            lon2: Долгота второй точки (градусы).
            
        Возвращает:
            Расстояние в километрах.
        """
        from math import radians, sin, cos, sqrt, atan2
        
        # Конвертируем координаты из градусов в радианы
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Формула гаверсинуса
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Радиус Земли в километрах
        
        return r * c