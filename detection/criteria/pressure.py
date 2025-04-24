"""
Модуль критерия минимума давления для обнаружения циклонов.

Предоставляет критерий обнаружения циклонов на основе локальных
минимумов давления на уровне моря.
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

class PressureMinimumCriterion(BaseCriterion):
    """
    Критерий обнаружения циклонов на основе локальных минимумов давления.
    
    Ищет локальные минимумы в поле давления на уровне моря
    с использованием фильтрации и проверки порогов.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                pressure_threshold: float = 1010.0,
                window_size: int = 3,
                smooth_sigma: float = 1.0):
        """
        Инициализирует критерий минимума давления.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            pressure_threshold: Пороговое значение давления (гПа).
            window_size: Размер окна для поиска локальных минимумов.
            smooth_sigma: Параметр сглаживания поля давления.
        """
        self.min_latitude = min_latitude
        self.pressure_threshold = pressure_threshold
        self.window_size = window_size
        self.smooth_sigma = smooth_sigma
        
        logger.debug(f"Инициализирован критерий минимума давления: "
                    f"min_latitude={min_latitude}, "
                    f"pressure_threshold={pressure_threshold}, "
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
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp','pressure_level']
            pressure_var = None
            
            for var in pressure_vars:
                if var in dataset:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                available_vars = list(dataset.variables)
                logger.error(f"Не удается определить переменную давления в наборе данных. Доступные переменные: {available_vars}")
                raise ValueError(f"Не удается определить переменную давления в наборе данных. Доступные переменные: {available_vars}")
                
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
            
            # Находим локальные минимумы
            min_filter = ndimage.minimum_filter(smoothed_field, size=self.window_size)
            local_minima = (smoothed_field == min_filter) & (smoothed_field < self.pressure_threshold)
            
            # Получаем координаты минимумов
            minima_indices = np.where(local_minima)
            
            # Формируем список кандидатов
            candidates = []
            
            for i in range(len(minima_indices[0])):
                lat_idx = minima_indices[0][i]
                lon_idx = minima_indices[1][i]
                
                latitude = float(arctic_data.latitude.values[lat_idx])
                longitude = float(arctic_data.longitude.values[lon_idx])
                pressure = float(smoothed_field[lat_idx, lon_idx])
                
                # Создаем кандидата
                candidate = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'pressure': pressure,
                    'criterion': 'pressure_minimum'
                }
                
                candidates.append(candidate)
            
            logger.debug(f"Критерий минимума давления нашел {len(candidates)} кандидатов")
            return candidates
            
        except Exception as e:
            error_msg = f"Ошибка при применении критерия минимума давления: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)