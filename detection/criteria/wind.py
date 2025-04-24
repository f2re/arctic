"""
Модуль критерия скорости ветра для обнаружения циклонов.

Предоставляет критерий обнаружения циклонов на основе
превышения пороговых значений скорости ветра.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import scipy.ndimage as ndimage

from . import BaseCriterion
from core.exceptions import DetectionError

# Инициализация логгера
logger = logging.getLogger(__name__)

class WindThresholdCriterion(BaseCriterion):
    """
    Критерий обнаружения циклонов на основе скорости ветра.
    
    Определяет области с высокой скоростью ветра, характерные для циклонов,
    на основе компонентов ветра (u, v) и порогового значения.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                min_speed: float = 15.0,
                pressure_level: int = 1000,
                window_size: int = 3,
                smooth_sigma: float = 1.0):
        """
        Инициализирует критерий скорости ветра.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            min_speed: Минимальная скорость ветра для обнаружения (м/с).
            pressure_level: Уровень давления для анализа (гПа).
            window_size: Размер окна для поиска локальных максимумов.
            smooth_sigma: Параметр сглаживания поля скорости ветра.
        """
        self.min_latitude = min_latitude
        self.min_speed = min_speed
        self.pressure_level = pressure_level
        self.window_size = window_size
        self.smooth_sigma = smooth_sigma
        
        logger.debug(f"Инициализирован критерий скорости ветра: "
                    f"min_latitude={min_latitude}, "
                    f"min_speed={min_speed}, "
                    f"pressure_level={pressure_level}, "
                    f"window_size={window_size}, "
                    f"smooth_sigma={smooth_sigma}")
    
    def apply(self, dataset: xr.Dataset, time_step: Any) -> List[Dict]:
        """
        Применяет критерий к набору данных.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список кандидатов в циклоны (словари с координатами и свойствами).
            
        Вызывает:
            DetectionError: При ошибке обнаружения циклонов.
        """
        try:
            # Определяем переменные компонентов ветра
            u_vars = ['u', 'u_component_of_wind', 'uwnd']
            v_vars = ['v', 'v_component_of_wind', 'vwnd']
            
            u_var, v_var = None, None
            
            # Поиск переменных u и v в наборе данных
            for u in u_vars:
                if u in dataset:
                    u_var = u
                    break
                    
            for v in v_vars:
                if v in dataset:
                    v_var = v
                    break
            
            if u_var is None or v_var is None:
                available_vars = list(dataset.variables)
                logger.error(f"Не удается определить компоненты ветра в наборе данных. Доступные переменные: {available_vars}")
                raise ValueError(f"Не удается определить компоненты ветра в наборе данных. Доступные переменные: {available_vars}")
            
            # Выбираем временной шаг и применяем маску региона
            time_data = dataset.sel(time=time_step)
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Выбираем нужный уровень давления, если есть измерение уровня
            pressure_level_names = ['level', 'pressure_level', 'lev', 'plev']
            has_levels = False
            
            for level_name in pressure_level_names:
                if level_name in arctic_data.dims:
                    has_levels = True
                    # Выбираем ближайший доступный уровень к указанному давлению
                    available_levels = arctic_data[level_name].values
                    closest_level = available_levels[np.abs(available_levels - self.pressure_level).argmin()]
                    
                    logger.debug(f"Используем уровень давления {closest_level} гПа")
                    u_data = arctic_data[u_var].sel({level_name: closest_level})
                    v_data = arctic_data[v_var].sel({level_name: closest_level})
                    break
            
            if not has_levels:
                # Если нет измерения уровня, используем данные как есть
                logger.debug("Уровни давления не найдены, используем данные без выбора уровня")
                u_data = arctic_data[u_var]
                v_data = arctic_data[v_var]
            
            # Рассчитываем скорость ветра
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            
            # Сглаживаем поле для уменьшения шума
            if self.smooth_sigma > 0:
                try:
                    wind_speed_values = wind_speed.values
                    smoothed_field = ndimage.gaussian_filter(wind_speed_values, sigma=self.smooth_sigma)
                except Exception as e:
                    logger.warning(f"Ошибка при сглаживании поля скорости ветра: {str(e)}")
                    # Fallback to non-smoothed field
                    smoothed_field = wind_speed.values
            else:
                smoothed_field = wind_speed.values
                
            # Находим локальные максимумы выше порога
            try:
                # Ensure arrays are properly shaped 2D arrays for the filter
                if smoothed_field.ndim != 2:
                    logger.warning(f"Expected 2D array for wind speed, got shape {smoothed_field.shape}")
                    if smoothed_field.size > 0:
                        # Try to reshape if possible
                        if hasattr(arctic_data, 'latitude') and hasattr(arctic_data, 'longitude'):
                            new_shape = (len(arctic_data.latitude), len(arctic_data.longitude))
                            try:
                                if np.prod(new_shape) == smoothed_field.size:
                                    smoothed_field = smoothed_field.reshape(new_shape)
                                    logger.info(f"Reshaped wind speed field to {new_shape}")
                                else:
                                    raise ValueError(f"Cannot reshape array of size {smoothed_field.size} to {new_shape}")
                            except Exception as reshape_err:
                                logger.error(f"Failed to reshape wind field: {str(reshape_err)}")
                                return []
                        else:
                            logger.error("Cannot determine proper dimensions for reshaping wind field")
                            return []
                
                # Apply maximum filter for finding local maxima
                max_filter = ndimage.maximum_filter(smoothed_field, size=self.window_size)
                high_wind_areas = (smoothed_field == max_filter) & (smoothed_field > self.min_speed)
                
                # Получаем координаты максимумов
                maxima_indices = np.where(high_wind_areas)
            except Exception as e:
                logger.error(f"Ошибка при поиске локальных максимумов скорости ветра: {str(e)}")
                return []
            
            # Формируем список кандидатов
            candidates = []
            
            for i in range(len(maxima_indices[0])):
                lat_idx = maxima_indices[0][i]
                lon_idx = maxima_indices[1][i]
                
                latitude = float(arctic_data.latitude.values[lat_idx])
                longitude = float(arctic_data.longitude.values[lon_idx])
                speed = float(smoothed_field[lat_idx, lon_idx])
                
                # Создаем кандидата
                candidate = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'wind_speed': speed,
                    'criterion': 'wind_threshold'
                }
                
                candidates.append(candidate)
            
            logger.debug(f"Критерий скорости ветра нашел {len(candidates)} кандидатов")
            return candidates
            
        except Exception as e:
            error_msg = f"Ошибка при применении критерия скорости ветра: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
