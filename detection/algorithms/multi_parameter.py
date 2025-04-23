"""
Модуль комплексного алгоритма обнаружения циклонов.

Реализует алгоритм обнаружения циклонов на основе
комбинации нескольких метеорологических параметров.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import scipy.ndimage as ndimage
import pandas as pd
from scipy.spatial.distance import cdist

from .base_algorithm import BaseDetectionAlgorithm
from core.exceptions import DetectionError

# Инициализация логгера
logger = logging.getLogger(__name__)

class MultiParameterAlgorithm(BaseDetectionAlgorithm):
    """
    Комплексный алгоритм обнаружения циклонов.
    
    Использует комбинацию нескольких метеорологических параметров
    для надежного обнаружения циклонов: давление, завихренность,
    градиент давления и ветер.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                pressure_threshold: float = 1010.0,
                vorticity_threshold: float = 1e-5,
                gradient_threshold: float = 0.5,  # гПа/100км
                wind_threshold: float = 8.0,  # м/с
                matching_radius: float = 1.0,  # градусы
                smooth_data: bool = True):
        """
        Инициализирует комплексный алгоритм обнаружения.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            pressure_threshold: Пороговое значение давления (гПа).
            vorticity_threshold: Пороговое значение завихренности (1/с).
            gradient_threshold: Минимальный градиент давления (гПа/100км).
            wind_threshold: Минимальная скорость ветра (м/с).
            matching_radius: Радиус сопоставления сигнатур (градусы).
            smooth_data: Применять ли сглаживание данных перед обнаружением.
        """
        super().__init__(
            min_latitude=min_latitude,
            smooth_data=smooth_data,
            name="Комплексный алгоритм обнаружения",
            description="Обнаружение циклонов на основе комбинации нескольких метеорологических параметров"
        )
        
        self.pressure_threshold = pressure_threshold
        self.vorticity_threshold = vorticity_threshold
        self.gradient_threshold = gradient_threshold
        self.wind_threshold = wind_threshold
        self.matching_radius = matching_radius
        
        logger.debug(f"Инициализирован комплексный алгоритм обнаружения с параметрами: "
                    f"pressure_threshold={pressure_threshold}, "
                    f"vorticity_threshold={vorticity_threshold}, "
                    f"gradient_threshold={gradient_threshold}, "
                    f"wind_threshold={wind_threshold}, "
                    f"matching_radius={matching_radius}")
    
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
            # Шаг 1: Обнаружение кандидатов по давлению
            pressure_candidates = self._detect_pressure_minima(dataset)
            
            # Шаг 2: Обнаружение кандидатов по завихренности
            vorticity_candidates = self._detect_vorticity_maxima(dataset)
            
            # Шаг 3: Объединение кандидатов
            combined_candidates = self._merge_candidates(
                pressure_candidates, vorticity_candidates)
            
            # Шаг 4: Дополнительные проверки и уточнение результатов
            refined_candidates = self._refine_candidates(combined_candidates, dataset)
            
            logger.info(f"Комплексный алгоритм обнаружил {len(refined_candidates)} кандидатов")
            return refined_candidates
            
        except Exception as e:
            error_msg = f"Ошибка при комплексном обнаружении циклонов: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
    
    def _detect_pressure_minima(self, dataset: xr.Dataset) -> List[Dict]:
        """
        Обнаруживает кандидатов в циклоны по минимумам давления.
        
        Аргументы:
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Список кандидатов в циклоны.
        """
        # Определяем переменную давления
        pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
        pressure_var = None
        
        for var in pressure_vars:
            if var in dataset:
                pressure_var = var
                break
        
        if pressure_var is None:
            logger.warning("Не удается определить переменную давления в наборе данных")
            return []
        
        # Получаем данные о давлении
        pressure_field = dataset[pressure_var].values
        
        # Находим локальные минимумы
        min_filter = ndimage.minimum_filter(pressure_field, size=3)
        local_minima = (pressure_field == min_filter) & (pressure_field < self.pressure_threshold)
        
        # Получаем координаты минимумов
        minima_indices = np.where(local_minima)
        
        # Формируем список кандидатов
        candidates = []
        
        for i in range(len(minima_indices[0])):
            lat_idx = minima_indices[0][i]
            lon_idx = minima_indices[1][i]
            
            latitude = float(dataset.latitude.values[lat_idx])
            longitude = float(dataset.longitude.values[lon_idx])
            pressure = float(pressure_field[lat_idx, lon_idx])
            
            # Создаем кандидата
            candidate = {
                'latitude': latitude,
                'longitude': longitude,
                'pressure': pressure,
                'source': 'pressure_minimum'
            }
            
            candidates.append(candidate)
        
        logger.debug(f"Обнаружено {len(candidates)} кандидатов по минимумам давления")
        return candidates
    
    def _detect_vorticity_maxima(self, dataset: xr.Dataset) -> List[Dict]:
        """
        Обнаруживает кандидатов в циклоны по максимумам завихренности.
        
        Аргументы:
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Список кандидатов в циклоны.
        """
        # Определяем переменную завихренности
        vorticity_vars = ['vorticity', 'vo', 'relative_vorticity']
        vorticity_var = None
        
        for var in vorticity_vars:
            if var in dataset:
                vorticity_var = var
                break
        
        if vorticity_var is None:
            logger.warning("Не удается определить переменную завихренности в наборе данных")
            return []
        
        # Получаем данные о завихренности
        # Проверяем, есть ли уровни давления
        if 'level' in dataset.dims and vorticity_var in dataset:
            # Ищем уровень 850 гПа или ближайший
            available_levels = dataset.level.values
            target_level = 850
            closest_level = available_levels[np.abs(available_levels - target_level).argmin()]
            
            vorticity_field = dataset.sel(level=closest_level)[vorticity_var].values
            logger.debug(f"Используем завихренность на уровне {closest_level} гПа")
        else:
            # Используем завихренность без уровня
            vorticity_field = dataset[vorticity_var].values
        
        # Находим локальные максимумы
        max_filter = ndimage.maximum_filter(vorticity_field, size=3)
        local_maxima = (vorticity_field == max_filter) & (vorticity_field > self.vorticity_threshold)
        
        # Получаем координаты максимумов
        maxima_indices = np.where(local_maxima)
        
        # Формируем список кандидатов
        candidates = []
        
        for i in range(len(maxima_indices[0])):
            lat_idx = maxima_indices[0][i]
            lon_idx = maxima_indices[1][i]
            
            latitude = float(dataset.latitude.values[lat_idx])
            longitude = float(dataset.longitude.values[lon_idx])
            vorticity = float(vorticity_field[lat_idx, lon_idx])
            
            # Создаем кандидата
            candidate = {
                'latitude': latitude,
                'longitude': longitude,
                'vorticity': vorticity,
                'source': 'vorticity_maximum'
            }
            
            # Добавляем давление, если доступно
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            for pvar in pressure_vars:
                if pvar in dataset:
                    candidate['pressure'] = float(dataset[pvar].isel(
                        latitude=lat_idx, longitude=lon_idx).values)
                    break
            
            candidates.append(candidate)
        
        logger.debug(f"Обнаружено {len(candidates)} кандидатов по максимумам завихренности")
        return candidates
    
    def _merge_candidates(self, pressure_candidates: List[Dict], 
                        vorticity_candidates: List[Dict]) -> List[Dict]:
        """
        Объединяет кандидатов, обнаруженных разными методами.
        
        Аргументы:
            pressure_candidates: Кандидаты, обнаруженные по минимумам давления.
            vorticity_candidates: Кандидаты, обнаруженные по максимумам завихренности.
            
        Возвращает:
            Объединенный список кандидатов.
        """
        if not pressure_candidates and not vorticity_candidates:
            return []
            
        if not pressure_candidates:
            return vorticity_candidates
            
        if not vorticity_candidates:
            return pressure_candidates
        
        # Создаем массивы координат
        p_coords = np.array([[c['latitude'], c['longitude']] for c in pressure_candidates])
        v_coords = np.array([[c['latitude'], c['longitude']] for c in vorticity_candidates])
        
        # Рассчитываем матрицу расстояний
        distances = cdist(p_coords, v_coords)
        
        # Находим пары кандидатов в пределах указанного радиуса
        merged_candidates = []
        used_pressure = set()
        used_vorticity = set()
        
        # Сначала объединяем близкие кандидаты
        for i in range(len(pressure_candidates)):
            for j in range(len(vorticity_candidates)):
                if distances[i, j] <= self.matching_radius:
                    # Объединяем кандидатов
                    p_candidate = pressure_candidates[i]
                    v_candidate = vorticity_candidates[j]
                    
                    merged = p_candidate.copy()
                    for key, value in v_candidate.items():
                        if key not in merged:
                            merged[key] = value
                    
                    merged['source'] = 'pressure_vorticity_match'
                    merged_candidates.append(merged)
                    
                    used_pressure.add(i)
                    used_vorticity.add(j)
        
        # Добавляем оставшихся кандидатов из обоих списков
        for i in range(len(pressure_candidates)):
            if i not in used_pressure:
                merged_candidates.append(pressure_candidates[i])
        
        for j in range(len(vorticity_candidates)):
            if j not in used_vorticity:
                merged_candidates.append(vorticity_candidates[j])
        
        logger.debug(f"После объединения получено {len(merged_candidates)} кандидатов")
        return merged_candidates
    
    def _refine_candidates(self, candidates: List[Dict], dataset: xr.Dataset) -> List[Dict]:
        """
        Уточняет результаты обнаружения и фильтрует ложные срабатывания.
        
        Аргументы:
            candidates: Список кандидатов в циклоны.
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Уточненный список кандидатов.
        """
        if not candidates:
            return []
        
        refined_candidates = []
        
        for candidate in candidates:
            # Проверяем градиент давления
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            
            for var in pressure_vars:
                if var in dataset:
                    pressure_var = var
                    break
            
            if pressure_var is not None:
                # Извлекаем регион вокруг кандидата
                lat, lon = candidate['latitude'], candidate['longitude']
                
                # Радиус области в градусах (примерно 300 км на средних широтах)
                radius_deg = 3.0
                
                # Проверяем, выходит ли область за границы данных
                lat_min, lat_max = float(dataset.latitude.min()), float(dataset.latitude.max())
                lon_min, lon_max = float(dataset.longitude.min()), float(dataset.longitude.max())
                
                region_lat_min = max(lat - radius_deg, lat_min)
                region_lat_max = min(lat + radius_deg, lat_max)
                region_lon_min = max(lon - radius_deg, lon_min)
                region_lon_max = min(lon + radius_deg, lon_max)
                
                region = dataset.sel(
                    latitude=slice(region_lat_min, region_lat_max),
                    longitude=slice(region_lon_min, region_lon_max)
                )
                
                # Вычисляем градиент давления
                pressure_field = region[pressure_var]
                dy, dx = np.gradient(pressure_field.values)
                
                # Рассчитываем величину градиента
                gradient_magnitude = np.sqrt(dx**2 + dy**2)
                
                # Преобразуем в гПа/100км (приблизительно)
                # Предполагаем среднее расстояние между точками сетки
                grid_spacing_deg = np.mean([
                    np.mean(np.diff(region.latitude.values)) if len(region.latitude) > 1 else 1.0,
                    np.mean(np.diff(region.longitude.values)) if len(region.longitude) > 1 else 1.0
                ])
                grid_spacing_km = grid_spacing_deg * 111.0  # приблизительно 111 км на градус
                
                # Преобразуем градиент
                gradient_hPa_per_100km = np.max(gradient_magnitude) * (100.0 / grid_spacing_km)
                
                # Добавляем информацию о градиенте
                candidate['pressure_gradient'] = float(gradient_hPa_per_100km)
                
                # Проверяем условие градиента
                if gradient_hPa_per_100km < self.gradient_threshold:
                    continue  # Пропускаем кандидата с недостаточным градиентом
            
            # Проверяем скорость ветра
            wind_var_pairs = [
                ('u_component_of_wind', 'v_component_of_wind'),
                ('u', 'v'),
                ('uwnd', 'vwnd'),
                ('10u', '10v'),
                ('10m_u_component_of_wind', '10m_v_component_of_wind')
            ]
            
            wind_found = False
            
            for u_var, v_var in wind_var_pairs:
                if u_var in dataset and v_var in dataset:
                    # Проверяем, есть ли уровни давления
                    if 'level' in dataset.dims and u_var in dataset.dims and u_var not in ['10u', '10v', '10m_u_component_of_wind', '10m_v_component_of_wind']:
                        # Ищем уровень 850 гПа или ближайший
                        available_levels = dataset.level.values
                        target_level = 850
                        closest_level = available_levels[np.abs(available_levels - target_level).argmin()]
                        
                        u = dataset.sel(level=closest_level)[u_var]
                        v = dataset.sel(level=closest_level)[v_var]
                    else:
                        # Используем данные без уровня
                        u = dataset[u_var]
                        v = dataset[v_var]
                    
                    # Находим ближайшую точку к кандидату
                    lat, lon = candidate['latitude'], candidate['longitude']
                    u_point = float(u.sel(latitude=lat, longitude=lon, method='nearest').values)
                    v_point = float(v.sel(latitude=lat, longitude=lon, method='nearest').values)
                    
                    # Рассчитываем скорость ветра
                    wind_speed = np.sqrt(u_point**2 + v_point**2)
                    
                    # Добавляем информацию о ветре
                    candidate['wind_speed'] = float(wind_speed)
                    
                    # Проверяем условие скорости ветра
                    if wind_speed < self.wind_threshold:
                        continue  # Пропускаем кандидата с недостаточной скоростью ветра
                    
                    wind_found = True
                    break
            
            # Если данные о ветре не найдены, пропускаем проверку
            if not wind_found:
                logger.debug("Данные о ветре не найдены, пропускаем проверку скорости ветра")
            
            # Добавляем дополнительную информацию
            candidate['detection_algorithm'] = self.name
            
            # Добавляем кандидата в уточненный список
            refined_candidates.append(candidate)
        
        logger.debug(f"После уточнения осталось {len(refined_candidates)} кандидатов")
        return refined_candidates