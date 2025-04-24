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
            # Select the specific time step first to simplify processing
            time_data = dataset.sel(time=time_step)
            
            # Apply mask for Arctic region
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Определяем переменные компонентов ветра
            u_vars = ['u', 'u_component_of_wind', 'uwnd']
            v_vars = ['v', 'v_component_of_wind', 'vwnd']
            
            u_var, v_var = None, None
            
            # Поиск переменных u и v в наборе данных
            for u in u_vars:
                if u in arctic_data:
                    u_var = u
                    break
                    
            for v in v_vars:
                if v in arctic_data:
                    v_var = v
                    break
            
            if u_var is None or v_var is None:
                available_vars = list(arctic_data.variables)
                logger.error(f"Не удается определить компоненты ветра в наборе данных. Доступные переменные: {available_vars}")
                raise ValueError(f"Не удается определить компоненты ветра в наборе данных. Доступные переменные: {available_vars}")
            
            # Handle pressure levels if present
            u_data = None
            v_data = None
            
            # Check if we have pressure levels
            pressure_level_names = ['level', 'pressure_level', 'lev', 'plev']
            
            for level_name in pressure_level_names:
                if level_name in arctic_data.dims:
                    # Find the closest available level to the specified pressure level
                    available_levels = arctic_data[level_name].values
                    closest_level = available_levels[np.abs(available_levels - self.pressure_level).argmin()]
                    logger.debug(f"Using pressure level {closest_level} hPa (closest to target {self.pressure_level} hPa)")
                    
                    # Select the appropriate level
                    u_data = arctic_data[u_var].sel({level_name: closest_level})
                    v_data = arctic_data[v_var].sel({level_name: closest_level})
                    break
            
            # If no level dimension found, use the data as is
            if u_data is None or v_data is None:
                u_data = arctic_data[u_var]
                v_data = arctic_data[v_var]
            
            # Рассчитываем скорость ветра
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            
            # Log wind data dimensions for debugging
            logger.debug(f"Wind data dimensions - u: {u_data.shape}, v: {v_data.shape}, wind_speed: {wind_speed.shape}")
            
            # Сглаживаем поле для уменьшения шума
            try:
                wind_values = wind_speed.values
                
                # Handle multi-dimensional arrays
                if wind_values.ndim > 2:
                    logger.warning(f"Wind data has shape {wind_values.shape}, reducing to 2D")
                    
                    # If we have more than 2 dimensions, flatten all but lat/lon
                    if hasattr(wind_speed, 'latitude') and hasattr(wind_speed, 'longitude'):
                        # If lat and lon are the last two dimensions
                        lat_dim = len(arctic_data.latitude)
                        lon_dim = len(arctic_data.longitude)
                        
                        if wind_values.shape[-2:] == (lat_dim, lon_dim):
                            # Use the last 2D slice if it matches lat/lon dimensions
                            wind_values = wind_values.reshape(-1, lat_dim, lon_dim)[-1]
                            logger.info(f"Using last 2D slice of wind data with shape {wind_values.shape}")
                        else:
                            # Try to average across extra dimensions
                            wind_values = np.mean(wind_values, axis=tuple(range(wind_values.ndim - 2)))
                            logger.info(f"Averaged wind data to shape {wind_values.shape}")
                
                # Apply Gaussian smoothing
                if self.smooth_sigma > 0 and wind_values.ndim == 2:
                    smoothed_field = ndimage.gaussian_filter(wind_values, sigma=self.smooth_sigma)
                else:
                    smoothed_field = wind_values
                    
            except Exception as e:
                logger.error(f"Error processing wind data: {str(e)}")
                return []
            
            # Находим локальные максимумы выше порога
            try:
                # Make sure smoothed_field is 2D before finding local maxima
                if smoothed_field.ndim != 2:
                    logger.warning(f"Smoothed wind field has {smoothed_field.ndim} dimensions, attempting to reduce to 2D")
                    if smoothed_field.ndim > 2:
                        # Use the mean across extra dimensions or the first slice
                        if smoothed_field.size > 0:
                            if smoothed_field.shape[0] == 1:
                                smoothed_field = smoothed_field[0]
                            else:
                                # Try to average across the first dimension
                                smoothed_field = np.mean(smoothed_field, axis=0)
                    else:
                        # If it's 1D, can't use it for maxima detection
                        logger.error("Cannot use 1D wind speed data for maxima detection")
                        return []
                
                max_filter = ndimage.maximum_filter(smoothed_field, size=self.window_size)
                high_wind_areas = (smoothed_field == max_filter) & (smoothed_field > self.min_speed)
                
                # Получаем координаты максимумов
                maxima_indices = np.where(high_wind_areas)
                
                if len(maxima_indices) < 2 or len(maxima_indices[0]) == 0:
                    logger.warning("No wind speed maxima found above threshold")
                    return []
                
                # Формируем список кандидатов
                candidates = []
                
                for i in range(len(maxima_indices[0])):
                    lat_idx = maxima_indices[0][i]
                    lon_idx = maxima_indices[1][i]
                    
                    if lat_idx < len(arctic_data.latitude) and lon_idx < len(arctic_data.longitude):
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
                    else:
                        logger.warning(f"Invalid wind maxima indices: lat_idx={lat_idx}, lon_idx={lon_idx}")
                
                logger.debug(f"Критерий скорости ветра нашел {len(candidates)} кандидатов")
                return candidates
                
            except Exception as e:
                logger.error(f"Ошибка при поиске локальных максимумов скорости ветра: {str(e)}")
                return []
            
        except Exception as e:
            error_msg = f"Ошибка при применении критерия скорости ветра: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
