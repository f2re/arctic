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
                central_pressure: float, dataset: Optional[xr.Dataset] = None, detector: Optional['CycloneDetector'] = None) -> None:
        """
        Инициализирует циклон с базовыми свойствами.
        
        Аргументы:
            latitude: Широта центра циклона (градусы).
            longitude: Долгота центра циклона (градусы).
            time: Время наблюдения.
            central_pressure: Центральное давление на уровне моря (гПа).
            dataset: Исходный набор метеорологических данных.
            detector: Детектор циклонов (необязательный).
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
        
        # Сохраняем детектор и инициализируем параметры
        self.detector = detector
        if dataset is not None and detector is not None:
            self.parameters = self._calculate_parameters(dataset, detector)
        else:
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
    
    def _calculate_parameters(self, dataset: xr.Dataset, detector: 'CycloneDetector') -> Dict[str, Any]:
        """
        Рассчитывает параметры циклона с использованием переданного экземпляра CycloneDetector.
        Аргументы:
            dataset: xarray.Dataset с метеорологическими данными
            detector: Экземпляр CycloneDetector, созданный в основном workflow (обязателен)
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Объект с параметрами циклона.
        """
        try:
            # Извлекаем регион вокруг циклона
            region = self._extract_cyclone_region(dataset)
            
            # Словарь для хранения рассчитанных параметров
            calculated_params = {
                'central_pressure': self.central_pressure
            }
            
            # Получаем активные критерии и их параметры
            active_criteria = detector.criteria_manager.get_active_criteria()
            criteria_params = getattr(detector, 'criteria_params', {})
            
            # Применяем только зарегистрированные методы на основе активных критериев
            if 'vorticity' in active_criteria and region is not None:
                vorticity = self._calculate_max_vorticity(region)
                calculated_params['vorticity_850hPa'] = vorticity
            
            if 'wind_threshold' in active_criteria and region is not None:
                wind_speed = self._calculate_max_wind_speed(region)
                calculated_params['max_wind_speed'] = wind_speed
            
            if 'closed_contour' in active_criteria and region is not None:
                radius = self._calculate_radius(region)
                calculated_params['radius'] = radius
            
            if 'pressure_laplacian' in active_criteria and region is not None:
                pressure_gradient = self._calculate_pressure_gradient(region)
                calculated_params['pressure_gradient'] = pressure_gradient
            
            # Определяем термическую структуру, если доступны необходимые данные
            if region is not None and any(var in region for var in ['temperature', 't']):
                thermal_type, t_anomaly = self._determine_thermal_structure(region)
                calculated_params['thermal_type'] = thermal_type
                calculated_params['temperature_anomaly'] = t_anomaly
            
            # Создаем объект параметров с рассчитанными значениями
            parameters = CycloneParameters(**calculated_params)
            
            # Логируем рассчитанные параметры
            log_params = {k: v for k, v in calculated_params.items() 
                         if k != 'central_pressure' and k != 'thermal_type' and v is not None}
            if 'thermal_type' in calculated_params and calculated_params['thermal_type'] is not None:
                log_params['thermal_type'] = calculated_params['thermal_type'].value
                
            logger.debug(f"Рассчитаны параметры циклона: {log_params}")
            
            return parameters
            
        except Exception as e:
            logger.warning(f"Ошибка при расчете параметров циклона: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
            
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
        try:
            # Проверяем наличие переменной завихренности
            vorticity_vars = ['vorticity', 'vo', 'relative_vorticity']
            vorticity_var = None
            
            # Добавляем диагностику о доступных переменных в наборе данных
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
            
            # Проверяем, что массив не пустой
            if vorticity.size == 0:
                logger.warning("Empty vorticity array, cannot calculate maximum")
                return None
            
            # Проверяем наличие NaN значений
            if np.isnan(vorticity).all():
                logger.warning("All vorticity values are NaN, cannot calculate maximum")
                return None
                
            # Используем np.nanmax для игнорирования NaN значений
            max_vorticity = float(np.nanmax(vorticity.values))
            return max_vorticity
            
        except Exception as e:
            logger.warning(f"Error calculating max vorticity: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
            return None
    
    def _calculate_max_wind_speed(self, region: xr.Dataset) -> Optional[float]:
        """
        Рассчитывает максимальную скорость ветра вблизи циклона.
        
        Аргументы:
            region: Набор метеорологических данных для региона циклона.
            
        Возвращает:
            Максимальное значение скорости ветра или None, если расчет невозможен.
        """
        try:
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
                
                # Проверяем, что массивы не пустые
                if u.size == 0 or v.size == 0:
                    logger.warning("Empty wind component arrays, cannot calculate wind speed")
                    return None
                
                # Проверяем наличие NaN значений
                if np.isnan(u).all() or np.isnan(v).all():
                    logger.warning("All wind component values are NaN, cannot calculate wind speed")
                    return None
                
                # Рассчитываем скорость ветра
                wind_speed = np.sqrt(u**2 + v**2)
                
                # Проверяем, что результат не пустой
                if wind_speed.size == 0:
                    logger.warning("Resulting wind speed array is empty")
                    return None
                
                # Находим максимальное значение, игнорируя NaN
                if np.isnan(wind_speed).all():
                    logger.warning("All wind speed values are NaN")
                    return None
                
                max_wind = float(np.nanmax(wind_speed.values))
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
                    
                    # Проверяем, что массив не пустой
                    if wind_speed.size == 0:
                        logger.warning(f"Empty {var} array, cannot calculate maximum")
                        return None
                    
                    # Проверяем наличие NaN значений
                    if np.isnan(wind_speed).all():
                        logger.warning(f"All {var} values are NaN, cannot calculate maximum")
                        return None
                    
                    # Находим максимальное значение, игнорируя NaN
                    max_wind = float(np.nanmax(wind_speed.values))
                    return max_wind
            
            # Если не нашли данные о ветре
            return None
            
        except Exception as e:
            logger.warning(f"Error calculating max wind speed: {str(e)}")
            return None
    
    def _calculate_radius(self, region: xr.Dataset) -> Optional[float]:
        """
        Рассчитывает радиус циклона на основе замкнутых изобар.
        
        Аргументы:
            region: Набор метеорологических данных для региона циклона.
            
        Возвращает:
            Радиус циклона в километрах или None, если расчет невозможен.
        """
        try:
            # Проверяем наличие переменной давления
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            
            for var in pressure_vars:
                if var in region:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                logger.warning("No pressure variable found in region dataset")
                return None
            
            # Получаем поле давления
            pressure_field = region[pressure_var]
            
            # Проверяем, что массив не пустой
            if pressure_field.size == 0:
                logger.warning("Empty pressure field array, cannot calculate radius")
                return None
            
            # Проверяем наличие NaN значений
            if np.isnan(pressure_field).all():
                logger.warning("All pressure field values are NaN, cannot calculate radius")
                return None
            
            # Находим внешнюю замкнутую изобару
            pressure_increment = 2.0  # гПа
            max_radius = 0.0
            
            for p_threshold in np.arange(self.central_pressure, 
                                        self.central_pressure + 20, 
                                        pressure_increment):
                # Создаем контур на этом уровне давления
                mask = pressure_field <= p_threshold
                
                # Проверяем, что маска не пустая
                if mask.size == 0 or not mask.any():
                    logger.debug(f"Empty or all-False mask at pressure threshold {p_threshold} hPa")
                    break
                
                # Проверяем, замкнут ли контур
                if not self._is_contour_closed(mask):
                    break
                
                # Рассчитываем эквивалентный радиус
                # Приблизительная площадь в км²
                try:
                    lat_mean = np.nanmean(region.latitude)
                    lat_km = 111.0  # 1 градус широты ≈ 111 км
                    lon_km = 111.0 * np.cos(np.radians(lat_mean))  # 1 градус долготы зависит от широты
                    
                    # Площадь в км²
                    mask_sum = mask.sum().values
                    if mask_sum > 0:
                        area = float(mask_sum) * lat_km * lon_km
                        
                        # Радиус в км
                        radius = np.sqrt(area / np.pi)
                        max_radius = radius
                    else:
                        logger.debug(f"Mask sum is zero at pressure threshold {p_threshold} hPa")
                except Exception as e:
                    logger.warning(f"Error calculating radius at threshold {p_threshold}: {str(e)}")
                    continue
            
            return max_radius if max_radius > 0 else None
            
        except Exception as e:
            logger.warning(f"Error calculating cyclone radius: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
            return None
    
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
        try:
            # Проверяем наличие переменной давления
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            
            for var in pressure_vars:
                if var in region:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                logger.warning("No pressure variable found in region dataset")
                return None
            
            # Получаем поле давления
            pressure_field = region[pressure_var]
            
            # Проверяем, что массив не пустой
            if pressure_field.size == 0:
                logger.warning("Empty pressure field array, cannot calculate pressure gradient")
                return None
            
            # Проверяем наличие NaN значений
            if np.isnan(pressure_field).all():
                logger.warning("All pressure field values are NaN, cannot calculate pressure gradient")
                return None
                
            # Проверяем размер массива - нужно минимум 2x2 точки для расчета градиента
            if pressure_field.shape[0] < 2 or pressure_field.shape[1] < 2:
                logger.warning(f"Pressure field array too small for gradient calculation: {pressure_field.shape}")
                return None
            
            # Заменяем NaN значения на среднее для расчета градиента
            pressure_values = pressure_field.values.copy()
            if np.isnan(pressure_values).any():
                # Заменяем NaN на среднее значение
                mean_val = np.nanmean(pressure_values)
                pressure_values = np.nan_to_num(pressure_values, nan=mean_val)
            
            # Рассчитываем градиенты
            grad_y, grad_x = np.gradient(pressure_values)
            
            # Проверяем на наличие данных в градиентах
            if grad_x.size == 0 or grad_y.size == 0:
                logger.warning("Empty gradient arrays")
                return None
            
            # Рассчитываем величину градиента
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Проверяем на наличие данных в величине градиента
            if grad_magnitude.size == 0 or np.isnan(grad_magnitude).all():
                logger.warning("Empty or all-NaN gradient magnitude array")
                return None
            
            # Находим максимальное значение, игнорируя NaN
            max_grad = float(np.nanmax(grad_magnitude))
            
            # Проверяем наличие широты и долготы
            if region.latitude.size == 0 or region.longitude.size == 0:
                logger.warning("Empty latitude or longitude arrays")
                return None
            
            # Преобразуем в гПа/100км
            try:
                lat_mean = np.nanmean(region.latitude)
                
                # Проверяем, что есть минимум 2 точки для расчета расстояния
                if region.latitude.size < 2 or region.longitude.size < 2:
                    # Используем стандартное значение для сетки
                    grid_spacing_km = 25.0  # Стандартное значение для сетки 0.25 градуса
                else:
                    lat_spacing = float(np.nanmean(np.diff(region.latitude.values)))
                    lon_spacing = float(np.nanmean(np.diff(region.longitude.values)))
                    
                    lat_km = 111.0  # 1 градус широты ≈ 111 км
                    lon_km = 111.0 * np.cos(np.radians(lat_mean))  # 1 градус долготы зависит от широты
                    
                    dx_km = lon_spacing * lon_km
                    dy_km = lat_spacing * lat_km
                    
                    # Среднее расстояние между точками в км
                    grid_spacing_km = np.mean([dx_km, dy_km])
                
                # Преобразуем градиент в гПа/100км
                gradient_hPa_per_100km = max_grad * (100.0 / grid_spacing_km)
                
                return gradient_hPa_per_100km
            except Exception as e:
                logger.warning(f"Error converting gradient to hPa/100km: {str(e)}")
                return None
                
        except Exception as e:
            logger.warning(f"Error calculating pressure gradient: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
            return None
    
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
        
        # Обновляем параметры, если доступен набор данных и детектор
        if dataset is not None and getattr(self, 'detector', None) is not None:
            self.parameters = self._calculate_parameters(dataset, self.detector)
        
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