"""
Модуль базового алгоритма обнаружения циклонов для системы ArcticCyclone.

Определяет абстрактный базовый класс для всех алгоритмов обнаружения
циклонов, устанавливая общий интерфейс и базовую функциональность.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging

from models.cyclone import Cyclone
from core.exceptions import DetectionError

# Инициализация логгера
logger = logging.getLogger(__name__)

class BaseDetectionAlgorithm(ABC):
    """
    Базовый абстрактный класс для всех алгоритмов обнаружения циклонов.
    
    Определяет интерфейс и общую функциональность для алгоритмов обнаружения.
    """
    
    def __init__(self, min_latitude: float = 65.0,
                smooth_data: bool = True,
                name: Optional[str] = None,
                description: Optional[str] = None):
        """
        Инициализирует базовый алгоритм обнаружения.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            smooth_data: Применять ли сглаживание данных перед обнаружением.
            name: Имя алгоритма.
            description: Описание алгоритма.
        """
        self.min_latitude = min_latitude
        self.smooth_data = smooth_data
        self._name = name or self.__class__.__name__
        self._description = description or self.__doc__ or "Алгоритм обнаружения циклонов"
        
        logger.debug(f"Инициализирован алгоритм обнаружения {self._name} с параметрами: "
                    f"min_latitude={min_latitude}")
    
    @property
    def name(self) -> str:
        """
        Возвращает имя алгоритма.
        
        Возвращает:
            Строка с именем алгоритма.
        """
        return self._name
    
    @property
    def description(self) -> str:
        """
        Возвращает описание алгоритма.
        
        Возвращает:
            Строка с описанием алгоритма.
        """
        return self._description
    
    @abstractmethod
    def detect(self, dataset: xr.Dataset, time_step: Any) -> List[Dict]:
        """
        Обнаруживает циклоны в наборе данных для указанного временного шага.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список обнаруженных циклонов (словари с координатами и свойствами).
            
        Вызывает:
            DetectionError: При ошибке обнаружения циклонов.
        """
        pass
    
    def preprocess_data(self, dataset: xr.Dataset, time_step: Any) -> xr.Dataset:
        """
        Предварительная обработка данных перед обнаружением.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Предобработанный набор данных.
        """
        try:
            # Выбираем временной шаг
            if time_step is not None:
                time_data = dataset.sel(time=time_step)
            else:
                # Если временной шаг не указан, используем первый доступный
                time_data = dataset.isel(time=0)
                logger.warning(f"Временной шаг не указан, используем первый доступный: {time_data.time.values}")
            
            # Применяем фильтрацию по широте для Арктического региона
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Применяем сглаживание, если требуется
            if self.smooth_data:
                # Список переменных для сглаживания
                smooth_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp',
                              'vorticity', 'vo', 'relative_vorticity']
                
                for var in smooth_vars:
                    if var in arctic_data:
                        # Применяем фильтр Гаусса для сглаживания
                        import scipy.ndimage as ndimage
                        smoothed = arctic_data[var].copy()
                        
                        # Для данных с уровнями давления
                        if 'level' in arctic_data.dims and var not in ['mean_sea_level_pressure', 'msl', 'psl', 'slp']:
                            # Сглаживаем отдельно для каждого уровня
                            for lev in arctic_data.level.values:
                                level_data = arctic_data.sel(level=lev)[var].values
                                smoothed_data = ndimage.gaussian_filter(level_data, sigma=1.0)
                                smoothed.loc[{'level': lev}] = smoothed_data
                        else:
                            # Сглаживаем 2D-поле
                            smoothed_data = ndimage.gaussian_filter(arctic_data[var].values, sigma=1.0)
                            smoothed.values = smoothed_data
                        
                        # Заменяем исходные данные сглаженными
                        arctic_data[var] = smoothed
                        logger.debug(f"Применено сглаживание для переменной {var}")
            
            return arctic_data
            
        except Exception as e:
            error_msg = f"Ошибка при предварительной обработке данных: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
    
    def create_cyclone_objects(self, candidates: List[Dict], dataset: xr.Dataset, 
                             time_step: Any) -> List[Cyclone]:
        """
        Создает объекты Cyclone из списка кандидатов.
        
        Аргументы:
            candidates: Список кандидатов в циклоны (словари с характеристиками).
            dataset: Исходный набор метеорологических данных.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список объектов Cyclone.
        """
        cyclones = []
        
        for candidate in candidates:
            try:
                # Получаем основные характеристики циклона
                latitude = candidate['latitude']
                longitude = candidate['longitude']
                
                # Находим центральное давление
                if 'pressure' in candidate:
                    central_pressure = candidate['pressure']
                else:
                    # Определяем переменную давления в наборе данных
                    pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
                    pressure_var = None
                    
                    for var in pressure_vars:
                        if var in dataset:
                            pressure_var = var
                            break
                    
                    if pressure_var is None:
                        raise ValueError("Не удается определить переменную давления в наборе данных")
                    
                    # Выбираем временной шаг и находим ближайшую точку сетки
                    time_data = dataset.sel(time=time_step)
                    central_pressure = float(time_data[pressure_var].sel(
                        latitude=latitude, longitude=longitude, method='nearest').values)
                
                # Создаем объект циклона
                cyclone = Cyclone(
                    latitude=latitude,
                    longitude=longitude,
                    time=time_step,
                    central_pressure=central_pressure,
                    dataset=dataset.sel(time=time_step)
                )
                
                # Добавляем дополнительные свойства из кандидата
                for key, value in candidate.items():
                    if key not in ['latitude', 'longitude', 'pressure']:
                        setattr(cyclone, key, value)
                
                cyclones.append(cyclone)
                
            except Exception as e:
                logger.warning(f"Ошибка при создании объекта циклона: {str(e)}")
        
        return cyclones
    
    def detect_cyclones(self, dataset: xr.Dataset, time_step: Any = None) -> List[Cyclone]:
        """
        Полный процесс обнаружения циклонов с предобработкой и созданием объектов.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список объектов Cyclone.
            
        Вызывает:
            DetectionError: При ошибке обнаружения циклонов.
        """
        try:
            # Предобработка данных
            preprocessed_data = self.preprocess_data(dataset, time_step)
            
            # Обнаружение циклонов
            candidates = self.detect(preprocessed_data, time_step)
            
            # Создание объектов циклонов
            cyclones = self.create_cyclone_objects(candidates, dataset, time_step)
            
            logger.info(f"Алгоритм {self.name} обнаружил {len(cyclones)} циклонов "
                      f"для временного шага {time_step}")
            
            return cyclones
            
        except Exception as e:
            error_msg = f"Ошибка при обнаружении циклонов алгоритмом {self.name}: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)