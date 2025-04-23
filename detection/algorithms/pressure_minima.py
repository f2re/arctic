"""
Модуль алгоритма обнаружения циклонов на основе минимумов давления.

Реализует традиционный алгоритм обнаружения циклонов
по локальным минимумам давления на уровне моря.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import scipy.ndimage as ndimage

from .base_algorithm import BaseDetectionAlgorithm
from core.exceptions import DetectionError

# Инициализация логгера
logger = logging.getLogger(__name__)

class PressureMinimaAlgorithm(BaseDetectionAlgorithm):
    """
    Алгоритм обнаружения циклонов на основе минимумов давления.
    
    Ищет локальные минимумы в поле давления на уровне моря
    и применяет дополнительные фильтры для уточнения результатов.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                pressure_threshold: float = 1010.0,
                min_gradient: float = 0.5,  # гПа/100км
                window_size: int = 3,
                smooth_data: bool = True):
        """
        Инициализирует алгоритм обнаружения по минимумам давления.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            pressure_threshold: Пороговое значение давления (гПа).
            min_gradient: Минимальный градиент давления (гПа/100км).
            window_size: Размер окна для поиска локальных минимумов.
            smooth_data: Применять ли сглаживание данных перед обнаружением.
        """
        super().__init__(
            min_latitude=min_latitude,
            smooth_data=smooth_data,
            name="Алгоритм минимумов давления",
            description="Обнаружение циклонов на основе локальных минимумов давления на уровне моря"
        )
        
        self.pressure_threshold = pressure_threshold
        self.min_gradient = min_gradient
        self.window_size = window_size
        
        logger.debug(f"Инициализирован алгоритм минимумов давления с параметрами: "
                    f"pressure_threshold={pressure_threshold}, "
                    f"min_gradient={min_gradient}, "
                    f"window_size={window_size}")
    
    def detect(self, dataset: xr.Dataset, time_step: Any) -> List[Dict]:
        """
        Обнаруживает циклоны в наборе данных.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список обнаруженных циклонов (словари с координатами и свойствами).
            
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
            
            # Получаем данные о давлении
            pressure_field = dataset[pressure_var].values
            
            # Находим локальные минимумы
            min_filter = ndimage.minimum_filter(pressure_field, size=self.window_size)
            local_minima = (pressure_field == min_filter) & (pressure_field < self.pressure_threshold)
            
            # Рассчитываем градиент давления
            grad_y, grad_x = np.gradient(pressure_field)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Преобразуем градиент в гПа/100км
            # Приближенный расчет с учетом широты
            lat_values = dataset.latitude.values
            lon_values = dataset.longitude.values
            
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
            
            # Получаем координаты минимумов с учетом градиента
            minima_indices = np.where(local_minima & (gradient_hPa_per_100km > self.min_gradient))
            
            # Формируем список кандидатов
            candidates = []
            
            for i in range(len(minima_indices[0])):
                lat_idx = minima_indices[0][i]
                lon_idx = minima_indices[1][i]
                
                latitude = float(dataset.latitude.values[lat_idx])
                longitude = float(dataset.longitude.values[lon_idx])
                pressure = float(pressure_field[lat_idx, lon_idx])
                gradient = float(gradient_hPa_per_100km[lat_idx, lon_idx])
                
                # Создаем кандидата
                candidate = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'pressure': pressure,
                    'pressure_gradient': gradient,
                    'detection_algorithm': self.name
                }
                
                # Добавляем информацию о завихренности, если доступна
                vorticity_vars = ['vorticity', 'vo', 'relative_vorticity']
                for var in vorticity_vars:
                    if var in dataset:
                        candidate['vorticity'] = float(dataset[var].isel(
                            latitude=lat_idx, longitude=lon_idx).values)
                        break
                
                candidates.append(candidate)
            
            logger.info(f"Алгоритм минимумов давления нашел {len(candidates)} кандидатов")
            return candidates
            
        except Exception as e:
            error_msg = f"Ошибка при обнаружении циклонов по минимумам давления: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
    
    def verify_results(self, candidates: List[Dict], dataset: xr.Dataset) -> List[Dict]:
        """
        Проверяет результаты обнаружения и фильтрует ложные срабатывания.
        
        Аргументы:
            candidates: Список кандидатов в циклоны.
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Отфильтрованный список кандидатов.
        """
        verified_candidates = []
        
        for candidate in candidates:
            # Проверяем дополнительные условия для подтверждения циклона
            
            # 1. Проверка замкнутости изобар (если кандидат находится слишком близко к границе, пропускаем проверку)
            lat = candidate['latitude']
            lon = candidate['longitude']
            
            # Проверяем расстояние до границы области данных
            lat_min, lat_max = float(dataset.latitude.min()), float(dataset.latitude.max())
            lon_min, lon_max = float(dataset.longitude.min()), float(dataset.longitude.max())
            
            # Если кандидат слишком близко к границе, пропускаем проверку замкнутости
            lat_margin = (lat_max - lat_min) * 0.1
            lon_margin = (lon_max - lon_min) * 0.1
            
            near_boundary = (
                lat < lat_min + lat_margin or
                lat > lat_max - lat_margin or
                lon < lon_min + lon_margin or
                lon > lon_max - lon_margin
            )
            
            if not near_boundary:
                # Проверяем замкнутость изобар (упрощенно)
                pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
                pressure_var = None
                
                for var in pressure_vars:
                    if var in dataset:
                        pressure_var = var
                        break
                
                if pressure_var is not None:
                    # Находим ближайшую точку к кандидату
                    point_data = dataset[pressure_var].sel(
                        latitude=lat, longitude=lon, method='nearest')
                    
                    # Проверяем, что это действительно минимум
                    if float(point_data) <= candidate['pressure']:
                        verified_candidates.append(candidate)
                else:
                    # Если переменная давления не найдена, добавляем кандидата без проверки
                    verified_candidates.append(candidate)
            else:
                # Для кандидатов у границы пропускаем проверку замкнутости
                verified_candidates.append(candidate)
        
        logger.debug(f"Верифицировано {len(verified_candidates)} из {len(candidates)} кандидатов")
        return verified_candidates