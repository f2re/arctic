"""
Модуль критерия завихренности для обнаружения циклонов.

Предоставляет критерий обнаружения циклонов на основе локальных
максимумов завихренности на уровне 850 гПа.
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

class VorticityCriterion(BaseCriterion):
    """
    Критерий обнаружения циклонов на основе завихренности.
    
    Ищет локальные максимумы в поле относительной завихренности
    на уровне 850 гПа как индикаторы циклонических образований.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                vorticity_threshold: float = 1e-5,
                pressure_level: int = 850,
                window_size: int = 3,
                smooth_sigma: float = 1.0):
        """
        Инициализирует критерий завихренности.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            vorticity_threshold: Пороговое значение завихренности (1/с).
            pressure_level: Уровень давления для анализа (гПа).
            window_size: Размер окна для поиска локальных максимумов.
            smooth_sigma: Параметр сглаживания поля завихренности.
        """
        self.min_latitude = min_latitude
        self.vorticity_threshold = vorticity_threshold
        self.pressure_level = pressure_level
        self.window_size = window_size
        self.smooth_sigma = smooth_sigma
        
        logger.debug(f"Инициализирован критерий завихренности: "
                    f"min_latitude={min_latitude}, "
                    f"vorticity_threshold={vorticity_threshold}, "
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
            # Определяем переменную завихренности в наборе данных
            vorticity_vars = ['vorticity', 'vo', 'relative_vorticity']
            vorticity_var = None
            
            # Select the specific time step first to simplify processing
            time_data = dataset.sel(time=time_step)
            
            # Log available variables for debugging
            logger.debug(f"Available variables: {list(time_data.variables)}")
            logger.debug(f"Available dimensions: {time_data.dims}")
            
            for var in vorticity_vars:
                if var in time_data:
                    vorticity_var = var
                    break
            
            # Если переменная завихренности не найдена, пытаемся рассчитать ее
            if vorticity_var is None:
                logger.info("Переменная завихренности не найдена, пытаемся рассчитать")
                
                # Проверяем наличие компонентов ветра
                wind_vars = [
                    ('u_component_of_wind', 'v_component_of_wind'),
                    ('u', 'v'),
                    ('uwnd', 'vwnd')
                ]
                
                u_var, v_var = None, None
                
                for u, v in wind_vars:
                    if u in time_data and v in time_data:
                        u_var, v_var = u, v
                        break
                
                if u_var is None or v_var is None:
                    raise ValueError("Не удается найти компоненты ветра для расчета завихренности")
                
                # Рассчитываем завихренность
                vorticity_var = self._calculate_vorticity(time_data, time_step, u_var, v_var)
            
            # Apply mask for Arctic region
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Check if vorticity is available in the dataset
            if vorticity_var not in arctic_data:
                logger.error(f"Переменная завихренности не найдена в отфильтрованном наборе данных")
                return []
            
            # Extract vorticity at the specified pressure level
            vorticity_data = None
            
            # Check if we have pressure levels
            pressure_level_names = ['level', 'pressure_level', 'lev', 'plev']
            
            for level_name in pressure_level_names:
                if level_name in arctic_data.dims:
                    # Find the closest available level to the specified pressure level
                    available_levels = arctic_data[level_name].values
                    closest_level = available_levels[np.abs(available_levels - self.pressure_level).argmin()]
                    logger.debug(f"Using pressure level {closest_level} hPa (closest to target {self.pressure_level} hPa)")
                    
                    # Select the appropriate level
                    vorticity_data = arctic_data[vorticity_var].sel({level_name: closest_level})
                    break
            
            # If no level dimension found, use the data as is
            if vorticity_data is None:
                vorticity_data = arctic_data[vorticity_var]
            
            # Apply a threshold to identify areas with high vorticity
            if self.smooth_sigma > 0:
                try:
                    # Make sure we have a 2D array for smoothing
                    vort_values = vorticity_data.values
                    if vort_values.ndim > 2:
                        logger.warning(f"Vorticity data has shape {vort_values.shape}, flattening extra dimensions")
                        # If we have more than 2 dimensions, flatten all but lat/lon
                        if hasattr(vorticity_data, 'latitude') and hasattr(vorticity_data, 'longitude'):
                            # Reshape to (lat, lon) if those are the last two dimensions
                            if vort_values.shape[-2:] == (len(arctic_data.latitude), len(arctic_data.longitude)):
                                vort_values = vort_values.reshape(-1, *vort_values.shape[-2:])[-1]
                            else:
                                # Try other reshaping approaches
                                vort_values = vort_values.mean(axis=tuple(range(vort_values.ndim - 2)))
                    
                    # Apply Gaussian smoothing
                    smoothed_vorticity = ndimage.gaussian_filter(vort_values, sigma=self.smooth_sigma)
                except Exception as e:
                    logger.warning(f"Error during vorticity smoothing: {str(e)}")
                    smoothed_vorticity = vorticity_data.values
            else:
                smoothed_vorticity = vorticity_data.values
            
            # Find local maxima above threshold
            try:
                # Make sure smoothed_vorticity is 2D before finding local maxima
                if smoothed_vorticity.ndim != 2:
                    logger.warning(f"Smoothed vorticity has {smoothed_vorticity.ndim} dimensions, attempting to reduce to 2D")
                    if smoothed_vorticity.ndim > 2:
                        # Use the mean across extra dimensions or the first slice
                        if smoothed_vorticity.size > 0:
                            if smoothed_vorticity.shape[0] == 1:
                                smoothed_vorticity = smoothed_vorticity[0]
                            else:
                                # Try to average across the first dimension
                                smoothed_vorticity = np.mean(smoothed_vorticity, axis=0)
                    else:
                        # If it's 1D, can't use it for maxima detection
                        logger.error("Cannot use 1D vorticity data for maxima detection")
                        return []
                
                max_filter = ndimage.maximum_filter(smoothed_vorticity, size=self.window_size)
                vorticity_maxima = (smoothed_vorticity == max_filter) & (smoothed_vorticity > self.vorticity_threshold)
                
                # Get coordinates of maxima
                maxima_indices = np.where(vorticity_maxima)
                
                candidates = []
                
                for i in range(len(maxima_indices[0])):
                    lat_idx = maxima_indices[0][i]
                    lon_idx = maxima_indices[1][i]
                    
                    if lat_idx < len(arctic_data.latitude) and lon_idx < len(arctic_data.longitude):
                        latitude = float(arctic_data.latitude.values[lat_idx])
                        longitude = float(arctic_data.longitude.values[lon_idx])
                        vorticity_value = float(smoothed_vorticity[lat_idx, lon_idx])
                        
                        candidate = {
                            'latitude': latitude,
                            'longitude': longitude,
                            'vorticity': vorticity_value,
                            'criterion': 'vorticity'
                        }
                        
                        candidates.append(candidate)
                    else:
                        logger.warning(f"Invalid vorticity maxima indices: lat_idx={lat_idx}, lon_idx={lon_idx}")
                
                logger.debug(f"Критерий завихренности нашел {len(candidates)} кандидатов")
                return candidates
            except Exception as e:
                logger.error(f"Error finding vorticity maxima: {str(e)}")
                return []
                
        except Exception as e:
            error_msg = f"Ошибка при применении критерия завихренности: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
    
    def _calculate_vorticity(self, dataset: xr.Dataset, time_step: Any,
                           u_var: str, v_var: str) -> str:
        """
        Рассчитывает относительную завихренность на основе компонентов ветра.
        
        Аргументы:
            dataset: Набор метеорологических данных.
            time_step: Временной шаг для анализа.
            u_var: Имя переменной зональной компоненты ветра.
            v_var: Имя переменной меридиональной компоненты ветра.
            
        Возвращает:
            Имя созданной переменной завихренности.
        """
        # Выбираем временной шаг
        time_data = dataset.sel(time=time_step)
        
        # Выбираем нужный уровень давления, если есть измерение уровня
        pressure_level_names = ['level', 'pressure_level', 'lev', 'plev']
        has_levels = False
        
        for level_name in pressure_level_names:
            if level_name in time_data.dims:
                has_levels = True
                # Выбираем ближайший доступный уровень к 850 гПа
                available_levels = time_data[level_name].values
                closest_level = available_levels[np.abs(available_levels - self.pressure_level).argmin()]
                
                level_data = time_data.sel({level_name: closest_level})
                u = level_data[u_var]
                v = level_data[v_var]
                break
        
        if not has_levels:
            # Если нет измерения уровня, используем данные как есть
            u = time_data[u_var]
            v = time_data[v_var]
        
        # Рассчитываем градиенты для завихренности
        # ζ = ∂v/∂x - ∂u/∂y
        
        try:
            # Преобразуем координаты в радианы
            lat_rad = np.radians(time_data.latitude)
            lon_rad = np.radians(time_data.longitude)
            
            # Рассчитываем шаг сетки
            dlat = np.gradient(lat_rad)
            dlon = np.gradient(lon_rad)
            
            # Рассчитываем расстояния в метрах
            dy = 6371000 * dlat
            dx = 6371000 * np.cos(lat_rad) * dlon
            
            # Рассчитываем градиенты компонентов ветра
            try:
                # Convert to numpy arrays and ensure they're 2D
                u_values = np.asarray(u.values)
                v_values = np.asarray(v.values)
                
                # Check array dimensions
                if u_values.ndim != 2 or v_values.ndim != 2:
                    logger.warning(f"Expected 2D arrays for wind components, got shapes u:{u_values.shape}, v:{v_values.shape}")
                    
                    # Try to reshape if possible
                    if hasattr(time_data, 'latitude') and hasattr(time_data, 'longitude'):
                        new_shape = (len(time_data.latitude), len(time_data.longitude))
                        
                        if u_values.size > 0 and v_values.size > 0:
                            try:
                                if np.prod(new_shape) == u_values.size:
                                    u_values = u_values.reshape(new_shape)
                                if np.prod(new_shape) == v_values.size:
                                    v_values = v_values.reshape(new_shape)
                                logger.info(f"Reshaped wind components to {new_shape}")
                            except Exception as reshape_err:
                                logger.error(f"Failed to reshape wind components: {str(reshape_err)}")
                                raise ValueError(f"Cannot reshape wind components: {str(reshape_err)}")
                    else:
                        raise ValueError("Cannot determine proper dimensions for reshaping wind components")
                
                # Reshape dx and dy for proper broadcasting
                dx_2d = dx[:, np.newaxis]
                dy_2d = dy[:, np.newaxis]
                
                # Calculate gradients with error handling
                dudx = np.gradient(u_values, axis=1) / dx_2d
                dudy = np.gradient(u_values, axis=0) / dy_2d
                dvdx = np.gradient(v_values, axis=1) / dx_2d
                dvdy = np.gradient(v_values, axis=0) / dy_2d
                
                # Make sure all arrays have the same shape
                if not (dudx.shape == dudy.shape == dvdx.shape == dvdy.shape):
                    logger.warning(f"Gradient shapes don't match: dudx={dudx.shape}, dudy={dudy.shape}, dvdx={dvdx.shape}, dvdy={dvdy.shape}")
                    # Ensure all have the same shape by truncating to the smallest shape
                    min_shape = np.min([arr.shape for arr in [dudx, dudy, dvdx, dvdy]], axis=0)
                    dudx = dudx[:min_shape[0], :min_shape[1]]
                    dudy = dudy[:min_shape[0], :min_shape[1]]
                    dvdx = dvdx[:min_shape[0], :min_shape[1]]
                    dvdy = dvdy[:min_shape[0], :min_shape[1]]
            except Exception as e:
                logger.error(f"Error calculating gradients: {str(e)}")
                raise ValueError(f"Failed to calculate wind gradients: {str(e)}")
            
            # Рассчитываем завихренность
            vorticity = dvdx - dudy
        except Exception as e:
            logger.error(f"Error during vorticity calculation: {str(e)}")
            # Create a minimal valid vorticity field as fallback
            if hasattr(time_data, 'latitude') and hasattr(time_data, 'longitude'):
                lat_coords = time_data.latitude
                lon_coords = time_data.longitude
                dummy_vorticity = np.zeros((len(lat_coords), len(lon_coords)))
                vorticity = dummy_vorticity
            else:
                raise ValueError(f"Cannot create fallback vorticity field: {str(e)}")
        
        # Добавляем переменную в набор данных
        # Ensure vorticity is correctly added to the dataset with proper dimensions
        try:
            # Get the dimensions for the latitude and longitude coordinates
            lat_dim = time_data.latitude.dims[0]
            lon_dim = time_data.longitude.dims[0]
            
            # Check if the vorticity array has the right shape
            if vorticity.shape != (len(time_data.latitude), len(time_data.longitude)):
                logger.warning(f"Vorticity shape {vorticity.shape} doesn't match expected shape {(len(time_data.latitude), len(time_data.longitude))}")
                # Resize or pad the array if needed
                new_vorticity = np.zeros((len(time_data.latitude), len(time_data.longitude)))
                min_lat = min(vorticity.shape[0], new_vorticity.shape[0])
                min_lon = min(vorticity.shape[1], new_vorticity.shape[1])
                new_vorticity[:min_lat, :min_lon] = vorticity[:min_lat, :min_lon]
                vorticity = new_vorticity
            
            # Directly use the dataset's dimensions
            dataset['vorticity'] = ((lat_dim, lon_dim), vorticity)
            dataset.vorticity.attrs['long_name'] = 'Relative vorticity'
            dataset.vorticity.attrs['units'] = 's^-1'
            logger.info(f"Successfully calculated vorticity field with shape {vorticity.shape}")
        except Exception as e:
            logger.error(f"Error adding vorticity to dataset: {str(e)}")
            # Create a minimal valid vorticity field as fallback
            lat_coords = dataset.latitude
            lon_coords = dataset.longitude
            dummy_vorticity = np.ones((len(lat_coords), len(lon_coords))) * self.vorticity_threshold
            dataset['vorticity'] = (('latitude', 'longitude'), dummy_vorticity)
            dataset.vorticity.attrs['long_name'] = 'Dummy relative vorticity (fallback)'
            dataset.vorticity.attrs['units'] = 's^-1'
            logger.warning(f"Created fallback vorticity field due to error: {str(e)}")
        
        return 'vorticity'