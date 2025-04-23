"""
Модуль критерия градиента давления для обнаружения циклонов.

Предоставляет критерий обнаружения циклонов на основе сильных
градиентов давления, характерных для циклонических образований.
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

class PressureGradientCriterion(BaseCriterion):
    """
    Критерий обнаружения циклонов на основе градиента давления.
    
    Ищет области с сильным градиентом давления на уровне моря,
    характерные для циклонических образований.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                gradient_threshold: float = 0.5,  # гПа/100км
                window_size: int = 5,
                smooth_sigma: float = 1.5):
        """
        Инициализирует критерий градиента давления.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            gradient_threshold: Пороговое значение градиента давления (гПа/100км).
            window_size: Размер окна для анализа градиента.
            smooth_sigma: Параметр сглаживания поля давления.
        """
        self.min_latitude = min_latitude
        self.gradient_threshold = gradient_threshold
        self.window_size = window_size
        self.smooth_sigma = smooth_sigma
        
        logger.debug(f"Инициализирован критерий градиента давления: "
                    f"min_latitude={min_latitude}, "
                    f"gradient_threshold={gradient_threshold}, "
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
            # Определяем переменную давления в наборе данных
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            
            for var in pressure_vars:
                if var in dataset:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                raise ValueError("Не удается определить переменную давления в наборе данных")
                
            # Выбираем временной шаг и применяем маску региона
            time_data = dataset.sel(time=time_step)
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Получаем данные о давлении
            pressure_field = arctic_data[pressure_var].values
            
            # Сглаживаем поле для уменьшения шума
            if self.smooth_sigma > 0:
                smoothed_field = ndimage.gaussian_filter(pressure_field, sigma=self.smooth_sigma)
            else:
                smoothed_field = pressure_field
            
            # Рассчитываем градиент давления
            grad_y, grad_x = np.gradient(smoothed_field)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Преобразуем градиент в гПа/100км
            # Приближенный расчет с учетом широты
            lat_values = arctic_data.latitude.values
            lon_values = arctic_data.longitude.values
            
            # Среднее расстояние между точками сетки в градусах
            lat_spacing = np.mean(np.diff(lat_values)) if len(lat_values) > 1 else 1.0
            lon_spacing = np.mean(np.diff(lon_values)) if len(lon_values) > 1 else 1.0
            
            # Преобразуем в километры (приблизительно)
            km_per_degree_lat = 111.0  # 1 градус широты ≈ 111 км
            # 1 градус долготы зависит от широты: cos(lat) * 111 км
            avg_lat_rad = np.radians(np.mean(lat_values))
            km_per_degree_lon = np.cos(avg_lat_rad) * 111.0
            
            # Среднее расстояние между точками в км
            grid_spacing_km = np.mean([lat_spacing * km_per_degree_lat, 
                                      lon_spacing * km_per_degree_lon])
            
            # Преобразуем градиент в гПа/100км
            gradient_hPa_per_100km = gradient_magnitude * (100.0 / grid_spacing_km)
            
            # Находим локальные максимумы градиента
            max_filter = ndimage.maximum_filter(gradient_hPa_per_100km, size=self.window_size)
            local_maxima = (gradient_hPa_per_100km == max_filter) & (gradient_hPa_per_100km > self.gradient_threshold)
            
            # Получаем координаты максимумов
            maxima_indices = np.where(local_maxima)
            
            # Формируем список кандидатов
            candidates = []
            
            for i in range(len(maxima_indices[0])):
                lat_idx = maxima_indices[0][i]
                lon_idx = maxima_indices[1][i]
                
                latitude = float(arctic_data.latitude.values[lat_idx])
                longitude = float(arctic_data.longitude.values[lon_idx])
                gradient = float(gradient_hPa_per_100km[lat_idx, lon_idx])
                pressure = float(smoothed_field[lat_idx, lon_idx])
                
                # Создаем кандидата
                candidate = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'pressure': pressure,
                    'pressure_gradient': gradient,
                    'criterion': 'pressure_gradient'
                }
                
                candidates.append(candidate)
            
            logger.debug(f"Критерий градиента давления нашел {len(candidates)} кандидатов")
            return candidates
            
        except Exception as e:
            error_msg = f"Ошибка при применении критерия градиента давления: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)