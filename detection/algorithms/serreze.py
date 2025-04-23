"""
Модуль алгоритма Серреза для обнаружения циклонов.

Реализует алгоритм Серреза (Serreze) для обнаружения
арктических циклонов, широко используемый в научной литературе.
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

class SerrezeAlgorithm(BaseDetectionAlgorithm):
    """
    Алгоритм Серреза для обнаружения арктических циклонов.
    
    Реализует классический алгоритм обнаружения циклонов, описанный
    в работах Серреза и др. (Serreze et al., 1993, 1997).
    """
    
    def __init__(self, min_latitude: float = 70.0,
                min_pressure_diff: float = 1.0,  # гПа
                search_radius: int = 8,  # количество точек сетки
                smooth_data: bool = True):
        """
        Инициализирует алгоритм Серреза.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            min_pressure_diff: Минимальная разница давления (гПа).
            search_radius: Радиус поиска в точках сетки.
            smooth_data: Применять ли сглаживание данных перед обнаружением.
        """
        super().__init__(
            min_latitude=min_latitude,
            smooth_data=smooth_data,
            name="Алгоритм Серреза",
            description="Алгоритм Серреза для обнаружения арктических циклонов"
        )
        
        self.min_pressure_diff = min_pressure_diff
        self.search_radius = search_radius
        
        logger.debug(f"Инициализирован алгоритм Серреза с параметрами: "
                    f"min_pressure_diff={min_pressure_diff} гПа, "
                    f"search_radius={search_radius}")
    
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
            
            # Размер поля данных
            ny, nx = pressure_field.shape
            
            # Находим все точки как потенциальные минимумы
            candidates = []
            
            for i in range(ny):
                for j in range(nx):
                    # Проверяем, что точка внутри области поиска (не слишком близко к краю)
                    if (i >= self.search_radius and i < ny - self.search_radius and
                       j >= self.search_radius and j < nx - self.search_radius):
                        
                        center_pressure = pressure_field[i, j]
                        
                        # Проверяем, является ли точка минимумом в своем радиусе поиска
                        is_minimum = True
                        
                        # Проверяем 8 направлений (по алгоритму Серреза)
                        directions = [
                            (-1, 0),   # север
                            (-1, 1),   # северо-восток
                            (0, 1),    # восток
                            (1, 1),    # юго-восток
                            (1, 0),    # юг
                            (1, -1),   # юго-запад
                            (0, -1),   # запад
                            (-1, -1)   # северо-запад
                        ]
                        
                        for di, dj in directions:
                            # Проверяем наличие точек с более низким давлением вдоль направления
                            max_step = self.search_radius
                            
                            for step in range(1, max_step + 1):
                                ni, nj = i + di * step, j + dj * step
                                
                                # Проверяем, что точка в пределах поля
                                if 0 <= ni < ny and 0 <= nj < nx:
                                    neighbor_pressure = pressure_field[ni, nj]
                                    
                                    # Если нашли точку с давлением ниже или равным, то не минимум
                                    if neighbor_pressure <= center_pressure:
                                        is_minimum = False
                                        break
                            
                            if not is_minimum:
                                break
                        
                        # Если все направления проверены и точка является минимумом
                        if is_minimum:
                            # Проверяем минимальную разницу давления
                            # Находим максимальное давление по периметру радиуса поиска
                            max_pressure = center_pressure
                            
                            for di in range(-self.search_radius, self.search_radius + 1):
                                for dj in range(-self.search_radius, self.search_radius + 1):
                                    # Только точки на периметре
                                    if abs(di) == self.search_radius or abs(dj) == self.search_radius:
                                        ni, nj = i + di, j + dj
                                        
                                        # Проверяем, что точка в пределах поля
                                        if 0 <= ni < ny and 0 <= nj < nx:
                                            max_pressure = max(max_pressure, pressure_field[ni, nj])
                            
                            pressure_diff = max_pressure - center_pressure
                            
                            # Если разница достаточно большая, добавляем кандидата
                            if pressure_diff >= self.min_pressure_diff:
                                latitude = float(dataset.latitude.values[i])
                                longitude = float(dataset.longitude.values[j])
                                
                                candidate = {
                                    'latitude': latitude,
                                    'longitude': longitude,
                                    'pressure': float(center_pressure),
                                    'pressure_diff': float(pressure_diff),
                                    'detection_algorithm': self.name
                                }
                                
                                candidates.append(candidate)
            
            logger.info(f"Алгоритм Серреза нашел {len(candidates)} кандидатов")
            return candidates
            
        except Exception as e:
            error_msg = f"Ошибка при обнаружении циклонов алгоритмом Серреза: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)