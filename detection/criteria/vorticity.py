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
        
        # Рассчитываем сетку координат в метрах
        R = 6371000  # Радиус Земли в метрах
        
        # Преобразуем координаты в радианы
        lat_rad = np.radians(time_data.latitude)
        lon_rad = np.radians(time_data.longitude)
        
        # Рассчитываем шаг сетки
        dlat = np.gradient(lat_rad)
        dlon = np.gradient(lon_rad)
        
        # Рассчитываем расстояния в метрах
        dy = R * dlat
        dx = R * np.cos(lat_rad) * dlon
        
        # Рассчитываем градиенты компонентов ветра
        dudx = np.gradient(u.values, axis=1) / dx[:, np.newaxis]
        dudy = np.gradient(u.values, axis=0) / dy[:, np.newaxis]
        dvdx = np.gradient(v.values, axis=1) / dx[:, np.newaxis]
        dvdy = np.gradient(v.values, axis=0) / dy[:, np.newaxis]
        
        # Рассчитываем завихренность
        vorticity = dvdx - dudy
        
        # Добавляем переменную в набор данных
        dataset['vorticity'] = (('latitude', 'longitude'), vorticity)
        dataset.vorticity.attrs['long_name'] = 'Relative vorticity'
        dataset.vorticity.attrs['units'] = 's^-1'
        
        return 'vorticity'