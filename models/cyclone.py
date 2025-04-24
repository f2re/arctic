"""
Модуль представления циклонов для системы ArcticCyclone.

Предоставляет основной класс Cyclone для хранения и обработки
данных о циклонах, их треках и свойствах.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
from datetime import datetime
import logging
import uuid
from pathlib import Path
import os
import warnings

from .parameters import CycloneParameters
from .classifications import CycloneType, CycloneIntensity, CycloneLifeStage

# Инициализация логгера
logger = logging.getLogger(__name__)

class Cyclone:
    """
    Комплексное представление арктического мезоциклона.
    
    Содержит всю информацию о циклоне, включая его положение,
    динамические характеристики, параметры и жизненный цикл.
    """
    
    def __init__(self, latitude: float, longitude: float, time: Union[str, datetime, pd.Timestamp],
                central_pressure: float, dataset: Optional[xr.Dataset] = None):
        """
        Инициализирует циклон с базовыми свойствами.
        
        Аргументы:
            latitude: Широта центра циклона (градусы).
            longitude: Долгота центра циклона (градусы).
            time: Время наблюдения.
            central_pressure: Центральное давление на уровне моря (гПа).
            dataset: Исходный набор метеорологических данных.
        """
        # Положение и время
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        
        # Нормализуем время
        if isinstance(time, str):
            self.time = pd.to_datetime(time)
        elif isinstance(time, datetime):
            self.time = time
        elif isinstance(time, pd.Timestamp):
            self.time = time.to_pydatetime()
        else:
            raise ValueError(f"Неподдерживаемый тип времени: {type(time)}")
        
        # Базовые свойства
        self.central_pressure = float(central_pressure)
        self.track_id = None
        
        # Рассчитываем параметры, если доступен набор данных
        if dataset is not None:
            self.parameters = self._calculate_parameters(dataset)
        else:
            # Создаем пустой объект параметров
            self.parameters = CycloneParameters(
                central_pressure=central_pressure,
                vorticity_850hPa=None,
                max_wind_speed=None,
                radius=None
            )
        
        # Отслеживание жизненного цикла
        self.age = 0  # Часы с момента формирования
        self.intensity_history = [(self.time, central_pressure)]
        self.track = [(latitude, longitude, self.time)]
        
        # Стадия жизненного цикла
        self.life_stage = CycloneLifeStage.UNKNOWN
        
        # Дополнительные свойства
        self.isobars = None
        self.pressure_anomaly = None
        
        # Метаданные обнаружения
        self.detection_method = None
        self.detection_confidence = None
        
        logger.debug(f"Создан циклон: lat={latitude}, lon={longitude}, "
                    f"time={self.time}, pressure={central_pressure} гПа")
    
    def _calculate_parameters(self, dataset: xr.Dataset) -> CycloneParameters:
        """
        Рассчитывает комплексный набор параметров циклона.
        
        Аргументы:
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Объект с параметрами циклона.
        """
        try:
            # Извлекаем регион вокруг циклона
            region = self._extract_cyclone_region(dataset)
            
            # Рассчитываем завихренность на 850 гПа
            vorticity = self._calculate_max_vorticity(region)
            
            # Рассчитываем скорость ветра
            wind_speed = self._calculate_max_wind_speed(region)
            
            # Рассчитываем радиус по замкнутым изобарам
            radius = self._calculate_radius(region)
            
            # Определяем термическую структуру
            thermal_type, t_anomaly = self._determine_thermal_structure(region)
            
            # Рассчитываем градиент давления
            pressure_gradient = self._calculate_pressure_gradient(region)
            
            # Создаем объект параметров
            parameters = CycloneParameters(
                central_pressure=self.central_pressure,
                vorticity_850hPa=vorticity,
                max_wind_speed=wind_speed,
                radius=radius,
                thermal_type=thermal_type,
                temperature_anomaly=t_anomaly,
                pressure_gradient=pressure_gradient
            )
            
            logger.debug(f"Рассчитаны параметры циклона: vorticity={vorticity}, "
                       f"wind_speed={wind_speed}, radius={radius}, "
                       f"thermal_type={thermal_type.value}")
            
            return parameters
            
        except Exception as e:
            logger.warning(f"Ошибка при расчете параметров циклона: {str(e)}")
            
            # Возвращаем параметры с минимальной информацией
            return CycloneParameters(
                central_pressure=self.central_pressure,
                vorticity_850hPa=None,
                max_wind_speed=None,
                radius=None
            )
    
    def _extract_cyclone_region(self, dataset: xr.Dataset, 
                              radius_degrees: float = 5.0) -> xr.Dataset:
        """
        Извлекает регион вокруг центра циклона.
        
        Аргументы:
            dataset: Набор метеорологических данных.
            radius_degrees: Радиус региона в градусах.
            
        Возвращает:
            Поднабор данных для региона вокруг циклона.
        """
        # Определяем границы региона
        lat_min = max(self.latitude - radius_degrees, -90)
        lat_max = min(self.latitude + radius_degrees, 90)
        lon_min = self.longitude - radius_degrees
        lon_max = self.longitude + radius_degrees
        
        # Нормализуем долготу
        if lon_min < -180:
            lon_min += 360
        if lon_max > 180:
            lon_max -= 360
        
        # Извлекаем регион
        try:
            # Проверяем измерение времени
            if 'time' in dataset.dims:
                # Находим ближайший временной шаг
                time = dataset.sel(time=self.time, method='nearest').time.values
                
                # Извлекаем данные для региона и времени
                region = dataset.sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max),
                    time=time
                )
            else:
                # Извлекаем данные только для региона
                region = dataset.sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max)
                )
            
            return region
            
        except Exception as e:
            logger.warning(f"Ошибка при извлечении региона циклона: {str(e)}")
            
            # Возвращаем оригинальный набор данных, если не удалось извлечь регион
            return dataset
    
    def _calculate_max_vorticity(self, region: xr.Dataset) -> Optional[float]:
        """
        Рассчитывает максимальную относительную завихренность на уровне 850 гПа.
        
        Аргументы:
            region: Набор метеорологических данных для региона циклона.
            
        Возвращает:
            Максимальное значение завихренности или None, если расчет невозможен.
        """
        # Проверяем наличие переменной завихренности
        vorticity_vars = ['vorticity', 'vo', 'relative_vorticity']
        vorticity_var = None
        
        # Add diagnostics about available variables in the region dataset
        logger.debug(f"Available variables in region dataset: {list(region.variables)}")
        logger.debug(f"Region dataset dimensions: {region.dims}")
        
        for var in vorticity_vars:
            if var in region:
                vorticity_var = var
                logger.debug(f"Found vorticity variable: {var}")
                break
        
        if vorticity_var is None:
            logger.warning("No vorticity variable found in region dataset")
            return None
        
        # Если переменная завихренности есть, используем ее
        if 'level' in region.dims and vorticity_var in region:
            # Ищем уровень 850 гПа или ближайший
            levels = region.level.values
            level_850 = min(levels, key=lambda x: abs(x - 850))
            
            vorticity = region[vorticity_var].sel(level=level_850)
        else:
            vorticity = region[vorticity_var]
        
        # Находим максимальное значение
        try:
            # Check if the array is empty before trying to find the max
            if vorticity.size == 0:
                logger.warning("Empty vorticity array, cannot calculate maximum")
                return None
            
            # Check if we got NaN values
            if np.isnan(vorticity).all():
                logger.warning("All vorticity values are NaN, cannot calculate maximum")
                return None
                
            # Use np.nanmax to ignore NaN values
            max_vorticity = float(np.nanmax(vorticity.values))
            return max_vorticity
        except Exception as e:
            logger.warning(f"Error calculating max vorticity: {str(e)}")
            return None
    
    def _calculate_max_wind_speed(self, region: xr.Dataset) -> Optional[float]:
        """
        Рассчитывает максимальную скорость ветра вблизи циклона.
        
        Аргументы:
            region: Набор метеорологических данных для региона циклона.
            
        Возвращает:
            Максимальное значение скорости ветра или None, если расчет невозможен.
        """
        # Проверяем наличие компонентов ветра
        if ('u' in region or 'u_component_of_wind' in region) and \
           ('v' in region or 'v_component_of_wind' in region):
            
            # Получаем компоненты ветра
            u_var = 'u' if 'u' in region else 'u_component_of_wind'
            v_var = 'v' if 'v' in region else 'v_component_of_wind'
            
            # Проверяем наличие уровней давления
            if 'level' in region.dims:
                # Ищем уровень 850 гПа или ближайший
                levels = region.level.values
                level_850 = min(levels, key=lambda x: abs(x - 850))
                
                u = region[u_var].sel(level=level_850)
                v = region[v_var].sel(level=level_850)
            else:
                u = region[u_var]
                v = region[v_var]
            
            # Рассчитываем скорость ветра
            wind_speed = np.sqrt(u**2 + v**2)
            
            # Находим максимальное значение
            max_wind = float(wind_speed.max().values)
            
            return max_wind
        
        # Проверяем наличие скорости ветра
        wind_vars = ['wind_speed', 'wspd']
        for var in wind_vars:
            if var in region:
                wind_speed = region[var]
                
                # Проверяем наличие уровней давления
                if 'level' in region.dims and 'level' in wind_speed.dims:
                    # Ищем уровень 850 гПа или ближайший
                    levels = region.level.values
                    level_850 = min(levels, key=lambda x: abs(x - 850))
                    
                    wind_speed = wind_speed.sel(level=level_850)
                
                # Находим максимальное значение
                max_wind = float(wind_speed.max().values)
                
                return max_wind
        
        # Если не нашли данные о ветре
        return None
    
    def _calculate_radius(self, region: xr.Dataset) -> Optional[float]:
        """
        Рассчитывает радиус циклона на основе замкнутых изобар.
        
        Аргументы:
            region: Набор метеорологических данных для региона циклона.
            
        Возвращает:
            Радиус циклона в километрах или None, если расчет невозможен.
        """
        # Проверяем наличие переменной давления
        pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
        pressure_var = None
        
        for var in pressure_vars:
            if var in region:
                pressure_var = var
                break
        
        if pressure_var is None:
            return None
        
        # Получаем поле давления
        pressure_field = region[pressure_var]
        
        # Находим внешнюю замкнутую изобару
        pressure_increment = 2.0  # гПа
        max_radius = 0.0
        
        for p_threshold in np.arange(self.central_pressure, 
                                    self.central_pressure + 20, 
                                    pressure_increment):
            # Создаем контур на этом уровне давления
            mask = pressure_field <= p_threshold
            
            # Проверяем, замкнут ли контур
            if not self._is_contour_closed(mask):
                break
            
            # Рассчитываем эквивалентный радиус
            # Приблизительная площадь в км²
            lat_mean = np.mean(region.latitude)
            lat_km = 111.0  # 1 градус широты ≈ 111 км
            lon_km = 111.0 * np.cos(np.radians(lat_mean))  # 1 градус долготы зависит от широты
            
            # Площадь в км²
            area = float(mask.sum().values) * lat_km * lon_km
            
            # Радиус в км
            radius = np.sqrt(area / np.pi)
            
            max_radius = radius
        
        return max_radius
    
    def _is_contour_closed(self, mask: xr.DataArray) -> bool:
        """
        Проверяет, является ли контур замкнутым.
        
        Аргументы:
            mask: Маска с True для точек внутри контура.
            
        Возвращает:
            True, если контур замкнут (не касается границы), иначе False.
        """
        # Проверяем, касается ли маска какой-либо границы
        touches_boundary = (
            mask.isel(latitude=0).any() or
            mask.isel(latitude=-1).any() or
            mask.isel(longitude=0).any() or
            mask.isel(longitude=-1).any()
        )
        
        return not touches_boundary
    
    def _determine_thermal_structure(self, region: xr.Dataset) -> Tuple[CycloneType, Optional[float]]:
        """
        Определяет термическую структуру циклона.
        
        Аргументы:
            region: Набор метеорологических данных для региона циклона.
            
        Возвращает:
            Тип термической структуры и аномалия температуры.
        """
        # Проверяем наличие данных о температуре
        temp_vars = ['temperature', 't']
        temp_var = None
        
        for var in temp_vars:
            if var in region:
                temp_var = var
                break
        
        if temp_var is None:
            return CycloneType.UNCLASSIFIED, None
        
        # Проверяем наличие уровней давления
        if 'level' not in region.dims:
            return CycloneType.UNCLASSIFIED, None
        
        # Проверяем наличие необходимых уровней
        required_levels = [500, 850]
        available_levels = region.level.values
        
        for level in required_levels:
            if level not in available_levels and not any(abs(l - level) <= 25 for l in available_levels):
                return CycloneType.UNCLASSIFIED, None
        
        # Находим ближайшие уровни к требуемым
        level_500 = min(available_levels, key=lambda x: abs(x - 500))
        level_850 = min(available_levels, key=lambda x: abs(x - 850))
        
        # Получаем температуру на уровнях
        t_500 = region[temp_var].sel(level=level_500)
        t_850 = region[temp_var].sel(level=level_850)
        
        # Рассчитываем зональные средние
        t_500_zonal_mean = t_500.mean(dim='longitude')
        t_850_zonal_mean = t_850.mean(dim='longitude')
        
        # Рассчитываем аномалии в центре циклона
        center_lat = self.latitude
        center_lon = self.longitude
        
        # Находим ближайшие точки сетки к центру
        t_500_center = float(t_500.sel(latitude=center_lat, longitude=center_lon, method='nearest').values)
        t_850_center = float(t_850.sel(latitude=center_lat, longitude=center_lon, method='nearest').values)
        
        # Находим зональные средние для широты центра
        center_lat_idx = abs(region.latitude - center_lat).argmin()
        t_500_zonal_at_center = float(t_500_zonal_mean.isel(latitude=center_lat_idx).values)
        t_850_zonal_at_center = float(t_850_zonal_mean.isel(latitude=center_lat_idx).values)
        
        # Рассчитываем аномалии
        t_500_anomaly = t_500_center - t_500_zonal_at_center
        t_850_anomaly = t_850_center - t_850_zonal_at_center
        
        # Определяем тип термической структуры
        if t_500_anomaly > 0 and t_850_anomaly > 0:
            # Теплый центр во всей толще
            return CycloneType.WARM_CORE, t_850_anomaly
        elif t_500_anomaly < 0 and t_850_anomaly < 0:
            # Холодный центр во всей толще
            return CycloneType.COLD_CORE, t_850_anomaly
        elif t_500_anomaly < 0 and t_850_anomaly > 0:
            # Холодный верх, теплый низ (типично для мезоциклонов)
            return CycloneType.HYBRID, t_850_anomaly
        else:
            # Теплый верх, холодный низ (нетипично)
            return CycloneType.UNCLASSIFIED, t_850_anomaly
    
    def _calculate_pressure_gradient(self, region: xr.Dataset) -> Optional[float]:
        """
        Рассчитывает максимальный градиент давления вблизи центра циклона.
        
        Аргументы:
            region: Набор метеорологических данных для региона циклона.
            
        Возвращает:
            Градиент давления в гПа/100км или None, если расчет невозможен.
        """
        # Проверяем наличие переменной давления
        pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
        pressure_var = None
        
        for var in pressure_vars:
            if var in region:
                pressure_var = var
                break
        
        if pressure_var is None:
            return None
        
        # Получаем поле давления
        pressure_field = region[pressure_var]
        
        # Рассчитываем градиенты
        grad_y, grad_x = np.gradient(pressure_field.values)
        
        # Рассчитываем величину градиента
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Находим максимальное значение
        max_grad = float(np.max(grad_magnitude))
        
        # Преобразуем в гПа/100км
        lat_mean = np.mean(region.latitude)
        lat_spacing = float(np.mean(np.diff(region.latitude.values)))
        lon_spacing = float(np.mean(np.diff(region.longitude.values)))
        
        lat_km = 111.0  # 1 градус широты ≈ 111 км
        lon_km = 111.0 * np.cos(np.radians(lat_mean))  # 1 градус долготы зависит от широты
        
        dx_km = lon_spacing * lon_km
        dy_km = lat_spacing * lat_km
        
        # Среднее расстояние между точками в км
        grid_spacing_km = np.mean([dx_km, dy_km])
        
        # Преобразуем градиент в гПа/100км
        gradient_hPa_per_100km = max_grad * (100.0 / grid_spacing_km)
        
        return gradient_hPa_per_100km
    
    def update(self, new_latitude: float, new_longitude: float, 
              new_time: Union[str, datetime, pd.Timestamp], new_pressure: float,
              dataset: Optional[xr.Dataset] = None) -> None:
        """
        Обновляет циклон новым наблюдением.
        
        Аргументы:
            new_latitude: Новая широта центра циклона.
            new_longitude: Новая долгота центра циклона.
            new_time: Новое время наблюдения.
            new_pressure: Новое центральное давление.
            dataset: Обновленный набор метеорологических данных.
        """
        # Нормализуем время
        if isinstance(new_time, str):
            new_time_obj = pd.to_datetime(new_time)
        elif isinstance(new_time, datetime):
            new_time_obj = new_time
        elif isinstance(new_time, pd.Timestamp):
            new_time_obj = new_time.to_pydatetime()
        else:
            raise ValueError(f"Неподдерживаемый тип времени: {type(new_time)}")
        
        # Рассчитываем время с момента предыдущего наблюдения
        time_diff = (new_time_obj - self.time).total_seconds() / 3600  # часы
        
        # Обновляем положение и давление
        self.latitude = float(new_latitude)
        self.longitude = float(new_longitude)
        self.central_pressure = float(new_pressure)
        
        # Обновляем информацию об отслеживании
        self.age += time_diff
        self.time = new_time_obj
        
        # Обновляем историю
        self.intensity_history.append((new_time_obj, new_pressure))
        self.track.append((new_latitude, new_longitude, new_time_obj))
        
        # Обновляем параметры, если доступен набор данных
        if dataset is not None:
            self.parameters = self._calculate_parameters(dataset)
        
        # Обновляем стадию жизненного цикла
        self._update_life_stage()
        
        logger.debug(f"Обновлен циклон: lat={new_latitude}, lon={new_longitude}, "
                    f"time={new_time_obj}, pressure={new_pressure} гПа, age={self.age} ч")
    
    def _update_life_stage(self) -> None:
        """
        Обновляет стадию жизненного цикла циклона.
        """
        if len(self.intensity_history) < 2:
            self.life_stage = CycloneLifeStage.GENESIS
            return
        
        # Находим минимальное давление в истории
        min_pressure = min(p for _, p in self.intensity_history)
        current_pressure = self.intensity_history[-1][1]
        
        # Определяем стадию
        if len(self.intensity_history) == 2:
            # Только начало трека
            self.life_stage = CycloneLifeStage.GENESIS
        elif current_pressure == min_pressure:
            # Достигнуто минимальное давление (максимальная интенсивность)
            self.life_stage = CycloneLifeStage.MATURE
        elif current_pressure > self.intensity_history[-2][1]:
            # Давление растет (циклон заполняется)
            self.life_stage = CycloneLifeStage.DISSIPATION
        elif current_pressure < self.intensity_history[-2][1]:
            # Давление падает (циклон углубляется)
            self.life_stage = CycloneLifeStage.INTENSIFICATION
        else:
            # Давление не меняется
            self.life_stage = CycloneLifeStage.MATURE
    
    def calculate_lifecycle_metrics(self) -> Dict[str, float]:
        """
        Рассчитывает метрики жизненного цикла циклона.
        
        Возвращает:
            Словарь с метриками жизненного цикла:
                - lifespan_hours: Продолжительность жизни (часы)
                - deepening_rate: Скорость углубления (гПа/ч)
                - displacement: Смещение (км)
                - mean_speed: Средняя скорость (км/ч)
        """
        if len(self.intensity_history) < 2:
            return {
                'lifespan_hours': 0,
                'deepening_rate': 0,
                'displacement': 0,
                'mean_speed': 0
            }
        
        # Рассчитываем продолжительность жизни
        start_time = self.intensity_history[0][0]
        end_time = self.intensity_history[-1][0]
        lifespan = (end_time - start_time).total_seconds() / 3600
        
        # Рассчитываем скорость углубления (гПа/час)
        min_pressure = min(p for _, p in self.intensity_history)
        initial_pressure = self.intensity_history[0][1]
        
        # Находим время достижения минимального давления
        time_to_min = None
        for t, p in self.intensity_history:
            if p == min_pressure:
                time_to_min = (t - start_time).total_seconds() / 3600
                break
        
        if time_to_min and time_to_min > 0:
            deepening_rate = (initial_pressure - min_pressure) / time_to_min
        else:
            deepening_rate = 0
        
        # Рассчитываем смещение и скорость
        start_pos = self.track[0][:2]
        end_pos = self.track[-1][:2]
        
        displacement = self._calculate_distance(
            start_pos[0], start_pos[1], end_pos[0], end_pos[1]
        )
        
        mean_speed = displacement / max(1, lifespan)
        
        return {
            'lifespan_hours': lifespan,
            'deepening_rate': deepening_rate,
            'displacement': displacement,
            'mean_speed': mean_speed
        }
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Рассчитывает расстояние между двумя точками на сфере (формула гаверсинуса).
        
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
    
    def calculate_travel_speed(self) -> Optional[float]:
        """
        Рассчитывает текущую скорость перемещения циклона.
        
        Возвращает:
            Скорость перемещения в км/ч или None, если недостаточно данных.
        """
        if len(self.track) < 2:
            return None
        
        # Получаем текущее и предыдущее положение и время
        current_lat, current_lon, current_time = self.track[-1]
        prev_lat, prev_lon, prev_time = self.track[-2]
        
        # Рассчитываем расстояние
        distance = self._calculate_distance(prev_lat, prev_lon, current_lat, current_lon)
        
        # Рассчитываем время в часах
        time_diff = (current_time - prev_time).total_seconds() / 3600
        
        if time_diff > 0:
            speed = distance / time_diff
            return speed
        else:
            return 0.0
    
    def calculate_intensity(self) -> CycloneIntensity:
        """
        Определяет интенсивность циклона на основе параметров.
        
        Возвращает:
            Категория интенсивности циклона.
        """
        # Определяем интенсивность по центральному давлению
        pressure = self.central_pressure
        
        if pressure < 960:
            intensity = CycloneIntensity.VERY_STRONG
        elif pressure < 980:
            intensity = CycloneIntensity.STRONG
        elif pressure < 995:
            intensity = CycloneIntensity.MODERATE
        else:
            intensity = CycloneIntensity.WEAK
        
        return intensity
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Возвращает текущие параметры циклона в виде словаря.
        
        Возвращает:
            Словарь с текущими параметрами циклона.
        """
        intensity = self.calculate_intensity()
        travel_speed = self.calculate_travel_speed()
        lifecycle_metrics = self.calculate_lifecycle_metrics()
        
        parameters = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'time': self.time,
            'central_pressure': self.central_pressure,
            'age_hours': self.age,
            'intensity': intensity.value,
            'life_stage': self.life_stage.value,
            'travel_speed_kmh': travel_speed,
            'track_id': self.track_id,
        }
        
        # Добавляем параметры из объекта parameters
        if hasattr(self, 'parameters'):
            parameters.update({
                'vorticity_850hPa': getattr(self.parameters, 'vorticity_850hPa', None),
                'max_wind_speed': getattr(self.parameters, 'max_wind_speed', None),
                'radius_km': getattr(self.parameters, 'radius', None),
                'thermal_type': getattr(self.parameters, 'thermal_type', CycloneType.UNCLASSIFIED).value,
                'temperature_anomaly': getattr(self.parameters, 'temperature_anomaly', None),
                'pressure_gradient': getattr(self.parameters, 'pressure_gradient', None),
            })
        
        # Добавляем метрики жизненного цикла
        parameters.update(lifecycle_metrics)
        
        return parameters
    
    def __str__(self) -> str:
        """
        Возвращает строковое представление циклона.
        
        Возвращает:
            Строка с основной информацией о циклоне.
        """
        return (f"Cyclone(lat={self.latitude:.2f}, lon={self.longitude:.2f}, "
               f"time={self.time}, pressure={self.central_pressure:.1f} hPa, "
               f"age={self.age:.1f} h, track_id={self.track_id})")
    
    def __repr__(self) -> str:
        """
        Возвращает строковое представление циклона для отладки.
        
        Возвращает:
            Строка с подробной информацией о циклоне.
        """
        return (f"Cyclone(latitude={self.latitude}, longitude={self.longitude}, "
               f"time='{self.time}', central_pressure={self.central_pressure}, "
               f"age={self.age}, track_id='{self.track_id}', "
               f"n_history={len(self.intensity_history)}, n_track={len(self.track)})")