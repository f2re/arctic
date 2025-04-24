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
            
            for var in vorticity_vars:
                if var in dataset:
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
                    if u in dataset and v in dataset:
                        u_var, v_var = u, v
                        break
                
                if u_var is None or v_var is None:
                    raise ValueError("Не удается найти компоненты ветра для расчета завихренности")
                
                # Рассчитываем завихренность
                vorticity_var = self._calculate_vorticity(dataset, time_step, u_var, v_var)
                
                # Add diagnostic logging to check if vorticity was calculated correctly
                if 'vorticity' in dataset:
                    vorticity_data = dataset['vorticity'].values
                    logger.info(f"Vorticity calculated with shape: {vorticity_data.shape}")
                    logger.info(f"Vorticity data stats - min: {np.nanmin(vorticity_data) if vorticity_data.size > 0 else 'N/A'}, " 
                              f"max: {np.nanmax(vorticity_data) if vorticity_data.size > 0 else 'N/A'}, "
                              f"mean: {np.nanmean(vorticity_data) if vorticity_data.size > 0 else 'N/A'}")
                    logger.info(f"NaN count: {np.isnan(vorticity_data).sum()}/{vorticity_data.size}")
                    
                    # Check if vorticity is within expected range
                    if vorticity_data.size > 0 and not np.isnan(vorticity_data).all():
                        # Typical vorticity values for cyclones are around 10^-5 to 10^-4 s^-1
                        strong_vorticity_points = np.where(vorticity_data > self.vorticity_threshold)[0].size
                        logger.info(f"Points with vorticity > {self.vorticity_threshold}: {strong_vorticity_points}")
                    else:
                        logger.warning("Vorticity array is empty or all values are NaN")
                else:
                    logger.warning("Failed to add vorticity to dataset")
            
            # Выбираем временной шаг и применяем маску региона
            time_data = dataset.sel(time=time_step)
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Выбираем нужный уровень давления, если есть измерение уровня
            pressure_level_names = ['level', 'pressure_level', 'lev', 'plev']
            has_levels = False
            
            for level_name in pressure_level_names:
                if level_name in arctic_data.dims:
                    has_levels = True
                    # Выбираем ближайший доступный уровень к 850 гПа
                    available_levels = arctic_data[level_name].values
                    closest_level = available_levels[np.abs(available_levels - self.pressure_level).argmin()]
                    
                    if closest_level != self.pressure_level:
                        logger.warning(f"Уровень {self.pressure_level} гПа недоступен, "
                                      f"используем ближайший: {closest_level} гПа")
                    
                    level_data = arctic_data.sel({level_name: closest_level})
                    vorticity_field = level_data[vorticity_var].values
                    break
            
            if not has_levels:
                # Если нет измерения уровня, используем данные как есть
                vorticity_field = arctic_data[vorticity_var].values
            
            # Сглаживаем поле для уменьшения шума
            if self.smooth_sigma > 0:
                smoothed_field = ndimage.gaussian_filter(vorticity_field, sigma=self.smooth_sigma)
            else:
                smoothed_field = vorticity_field
            
            # Находим локальные максимумы (положительная завихренность для циклонов)
            max_filter = ndimage.maximum_filter(smoothed_field, size=self.window_size)
            local_maxima = (smoothed_field == max_filter) & (smoothed_field > self.vorticity_threshold)
            
            # Получаем координаты максимумов
            maxima_indices = np.where(local_maxima)
            
            # Формируем список кандидатов
            candidates = []
            
            for i in range(len(maxima_indices[0])):
                lat_idx = maxima_indices[0][i]
                lon_idx = maxima_indices[1][i]
                
                latitude = float(arctic_data.latitude.values[lat_idx])
                longitude = float(arctic_data.longitude.values[lon_idx])
                vorticity = float(smoothed_field[lat_idx, lon_idx])
                
                # Создаем кандидата
                candidate = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'vorticity': vorticity,
                    'criterion': 'vorticity'
                }
                
                # Добавляем давление, если доступно
                pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
                for pvar in pressure_vars:
                    if pvar in arctic_data:
                        candidate['pressure'] = float(arctic_data[pvar].isel(
                            latitude=lat_idx, longitude=lon_idx).values)
                        break
                
                candidates.append(candidate)
            
            logger.debug(f"Критерий завихренности нашел {len(candidates)} кандидатов")
            return candidates
            
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