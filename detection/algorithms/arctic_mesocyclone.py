"""
Модуль алгоритма обнаружения арктических мезоциклонов.

Реализует специализированный алгоритм для обнаружения именно
мезоциклонов в арктическом регионе с учетом их особенностей.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import scipy.ndimage as ndimage
from skimage import measure

from .base_algorithm import BaseDetectionAlgorithm
from core.exceptions import DetectionError

# Инициализация логгера
logger = logging.getLogger(__name__)

class ArcticMesocycloneAlgorithm(BaseDetectionAlgorithm):
    """
    Специализированный алгоритм обнаружения арктических мезоциклонов.
    
    Учитывает особенности арктических мезоциклонов: меньший размер,
    большую интенсивность завихренности, быстрое развитие и др.
    """
    
    def __init__(self, min_latitude: float = 65.0,
                max_diameter: float = 1000.0,  # км
                min_vorticity: float = 2e-5,  # 1/с
                min_pressure_anomaly: float = 2.0,  # гПа
                check_thermal_structure: bool = True,
                smooth_data: bool = True):
        """
        Инициализирует алгоритм обнаружения арктических мезоциклонов.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            max_diameter: Максимальный диаметр мезоциклона (км).
            min_vorticity: Минимальное значение завихренности (1/с).
            min_pressure_anomaly: Минимальная аномалия давления (гПа).
            check_thermal_structure: Проверять ли термическую структуру.
            smooth_data: Применять ли сглаживание данных перед обнаружением.
        """
        super().__init__(
            min_latitude=min_latitude,
            smooth_data=smooth_data,
            name="Алгоритм обнаружения арктических мезоциклонов",
            description="Специализированный алгоритм для обнаружения мезоциклонов в арктическом регионе"
        )
        
        self.max_diameter = max_diameter
        self.min_vorticity = min_vorticity
        self.min_pressure_anomaly = min_pressure_anomaly
        self.check_thermal_structure = check_thermal_structure
        
        logger.debug(f"Инициализирован алгоритм обнаружения арктических мезоциклонов с параметрами: "
                    f"max_diameter={max_diameter} км, "
                    f"min_vorticity={min_vorticity}, "
                    f"min_pressure_anomaly={min_pressure_anomaly} гПа, "
                    f"check_thermal_structure={check_thermal_structure}")
    
    def detect(self, dataset: xr.Dataset, time_step: Any) -> List[Dict]:
        """
        Обнаруживает арктические мезоциклоны в наборе данных.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список обнаруженных мезоциклонов (словари с координатами и свойствами).
            
        Вызывает:
            DetectionError: При ошибке обнаружения мезоциклонов.
        """
        try:
            # Шаг 1: Обнаружение кандидатов по завихренности
            vorticity_candidates = self._detect_vorticity_maxima(dataset)
            
            # Шаг 2: Проверка аномалии давления
            pressure_filtered_candidates = self._check_pressure_anomaly(vorticity_candidates, dataset)
            
            # Шаг 3: Определение размера циклона и фильтрация по размеру
            size_filtered_candidates = self._filter_by_size(pressure_filtered_candidates, dataset)
            
            # Шаг 4: Проверка термической структуры (если требуется)
            if self.check_thermal_structure:
                final_candidates = self._check_thermal_structure(size_filtered_candidates, dataset)
            else:
                final_candidates = size_filtered_candidates
            
            # Добавляем информацию об алгоритме
            for candidate in final_candidates:
                candidate['detection_algorithm'] = self.name
                candidate['cyclone_type'] = 'arctic_mesocyclone'
            
            logger.info(f"Алгоритм обнаружения арктических мезоциклонов нашел {len(final_candidates)} кандидатов")
            return final_candidates
            
        except Exception as e:
            error_msg = f"Ошибка при обнаружении арктических мезоциклонов: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
    
    def _detect_vorticity_maxima(self, dataset: xr.Dataset) -> List[Dict]:
        """
        Обнаруживает кандидатов в мезоциклоны по максимумам завихренности.
        
        Аргументы:
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Список кандидатов в мезоциклоны.
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
        
        # Находим локальные максимумы с более высоким порогом для мезоциклонов
        max_filter = ndimage.maximum_filter(vorticity_field, size=3)
        local_maxima = (vorticity_field == max_filter) & (vorticity_field > self.min_vorticity)
        
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
    
    def _check_pressure_anomaly(self, candidates: List[Dict], dataset: xr.Dataset) -> List[Dict]:
        """
        Проверяет аномалию давления и фильтрует кандидатов.
        
        Аргументы:
            candidates: Список кандидатов в мезоциклоны.
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Отфильтрованный список кандидатов.
        """
        if not candidates:
            return []
        
        # Определяем переменную давления
        pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
        pressure_var = None
        
        for var in pressure_vars:
            if var in dataset:
                pressure_var = var
                break
        
        if pressure_var is None:
            logger.warning("Не удается определить переменную давления в наборе данных")
            return candidates  # Возвращаем исходных кандидатов без фильтрации
        
        # Получаем среднее давление в регионе
        mean_pressure = float(dataset[pressure_var].mean().values)
        
        # Фильтруем кандидатов по аномалии давления
        filtered_candidates = []
        
        for candidate in candidates:
            if 'pressure' in candidate:
                pressure = candidate['pressure']
                anomaly = mean_pressure - pressure
                
                candidate['pressure_anomaly'] = float(anomaly)
                
                if anomaly >= self.min_pressure_anomaly:
                    filtered_candidates.append(candidate)
            else:
                # Если у кандидата нет давления, находим его
                lat, lon = candidate['latitude'], candidate['longitude']
                pressure = float(dataset[pressure_var].sel(
                    latitude=lat, longitude=lon, method='nearest').values)
                
                anomaly = mean_pressure - pressure
                
                candidate['pressure'] = pressure
                candidate['pressure_anomaly'] = float(anomaly)
                
                if anomaly >= self.min_pressure_anomaly:
                    filtered_candidates.append(candidate)
        
        logger.debug(f"После проверки аномалии давления осталось {len(filtered_candidates)} кандидатов")
        return filtered_candidates
    
    def _filter_by_size(self, candidates: List[Dict], dataset: xr.Dataset) -> List[Dict]:
        """
        Фильтрует кандидатов по размеру (должны быть мезомасштабными).
        
        Аргументы:
            candidates: Список кандидатов в мезоциклоны.
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Отфильтрованный список кандидатов.
        """
        if not candidates:
            return []
        
        # Определяем переменную давления
        pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
        pressure_var = None
        
        for var in pressure_vars:
            if var in dataset:
                pressure_var = var
                break
        
        if pressure_var is None:
            logger.warning("Не удается определить переменную давления для оценки размера")
            return candidates  # Возвращаем исходных кандидатов без фильтрации
        
        filtered_candidates = []
        
        for candidate in candidates:
            lat, lon = candidate['latitude'], candidate['longitude']
            
            # Извлекаем регион вокруг кандидата
            # Радиус области в градусах (примерно 600 км на средних широтах)
            radius_deg = 6.0
            
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
            
            # Получаем давление в центре
            central_pressure = candidate['pressure']
            
            # Определяем контуры по давлению
            pressure_field = region[pressure_var].values
            
            # Пороговое значение для контура (2 гПа выше центрального давления)
            threshold = central_pressure + 2.0
            
            # Создаем маску для областей ниже порогового давления
            mask = pressure_field < threshold
            
            # Находим связные компоненты в маске
            labeled_mask, num_features = ndimage.label(mask)
            
            # Находим компоненту, содержащую центр циклона
            center_lat_idx = np.abs(region.latitude - lat).argmin()
            center_lon_idx = np.abs(region.longitude - lon).argmin()
            
            if center_lat_idx >= labeled_mask.shape[0] or center_lon_idx >= labeled_mask.shape[1]:
                continue  # Пропускаем, если центр находится вне маски
            
            center_label = labeled_mask[center_lat_idx, center_lon_idx]
            
            if center_label > 0:
                # Измеряем свойства компоненты
                props = measure.regionprops(labeled_mask == center_label)
                
                if props:
                    # Оцениваем диаметр циклона
                    area_pixels = props[0].area
                    
                    # Преобразуем площадь в пикселях в площадь в км²
                    # Примерный расчет в зависимости от широты
                    avg_lat = (region_lat_min + region_lat_max) / 2
                    lat_spacing = np.mean(np.diff(region.latitude.values)) if len(region.latitude) > 1 else 1.0
                    lon_spacing = np.mean(np.diff(region.longitude.values)) if len(region.longitude) > 1 else 1.0
                    
                    # 1 градус широты ≈ 111 км
                    km_per_degree_lat = 111.0
                    # 1 градус долготы зависит от широты: cos(lat) * 111 км
                    km_per_degree_lon = np.cos(np.radians(avg_lat)) * 111.0
                    
                    # Средняя площадь одного пикселя в км²
                    pixel_area_km2 = lat_spacing * lon_spacing * km_per_degree_lat * km_per_degree_lon
                    
                    # Площадь циклона в км²
                    area_km2 = area_pixels * pixel_area_km2
                    
                    # Диаметр, предполагая круговую форму
                    diameter_km = 2 * np.sqrt(area_km2 / np.pi)
                    
                    # Добавляем информацию о размере
                    candidate['diameter_km'] = float(diameter_km)
                    candidate['area_km2'] = float(area_km2)
                    
                    # Фильтруем по размеру (мезоциклоны имеют диаметр до ~1000 км)
                    if diameter_km <= self.max_diameter:
                        filtered_candidates.append(candidate)
            
        logger.debug(f"После фильтрации по размеру осталось {len(filtered_candidates)} кандидатов")
        return filtered_candidates
    
    def _check_thermal_structure(self, candidates: List[Dict], dataset: xr.Dataset) -> List[Dict]:
        """
        Проверяет термическую структуру циклонов.
        
        Аргументы:
            candidates: Список кандидатов в мезоциклоны.
            dataset: Набор метеорологических данных.
            
        Возвращает:
            Список кандидатов с информацией о термической структуре.
        """
        if not candidates:
            return []
        
        # Проверяем наличие необходимых переменных
        if 'temperature' not in dataset and 't' not in dataset:
            logger.warning("Отсутствуют данные о температуре для проверки термической структуры")
            return candidates  # Возвращаем исходных кандидатов без анализа термической структуры
        
        temp_var = 'temperature' if 'temperature' in dataset else 't'
        
        # Проверяем наличие уровней давления
        if 'level' not in dataset.dims:
            logger.warning("Отсутствуют данные о уровнях давления для проверки термической структуры")
            return candidates  # Возвращаем исходных кандидатов без анализа термической структуры
        
        # Проверяем наличие необходимых уровней
        required_levels = [500, 850]
        available_levels = dataset.level.values
        
        if not all(level in available_levels for level in required_levels):
            missing_levels = [level for level in required_levels if level not in available_levels]
            logger.warning(f"Отсутствуют необходимые уровни давления: {missing_levels}")
            return candidates  # Возвращаем исходных кандидатов без анализа термической структуры
        
        for candidate in candidates:
            lat, lon = candidate['latitude'], candidate['longitude']
            
            # Получаем температуру на уровнях 500 и 850 гПа
            t500 = float(dataset[temp_var].sel(level=500, latitude=lat, longitude=lon, method='nearest').values)
            t850 = float(dataset[temp_var].sel(level=850, latitude=lat, longitude=lon, method='nearest').values)
            
            # Вычисляем толщину слоя 500-850 гПа
            # Если есть геопотенциальная высота
            if 'geopotential_height' in dataset:
                z500 = float(dataset.geopotential_height.sel(level=500, latitude=lat, longitude=lon, method='nearest').values)
                z850 = float(dataset.geopotential_height.sel(level=850, latitude=lat, longitude=lon, method='nearest').values)
                thickness = z500 - z850
            elif 'geopotential' in dataset:
                # Преобразуем геопотенциал в высоту
                z500 = float(dataset.geopotential.sel(level=500, latitude=lat, longitude=lon, method='nearest').values) / 9.80665
                z850 = float(dataset.geopotential.sel(level=850, latitude=lat, longitude=lon, method='nearest').values) / 9.80665
                thickness = z500 - z850
            else:
                # Если нет данных о геопотенциале, оцениваем толщину по формуле гипсометрии
                # толщина ≈ R * ln(p1/p2) * T_avg / g, где R - газовая постоянная, T_avg - средняя температура
                R = 287.0  # Дж/(кг·К)
                g = 9.80665  # м/с²
                T_avg = (t500 + t850) / 2
                thickness = R * np.log(850 / 500) * T_avg / g
            
            # Добавляем информацию о термической структуре
            candidate['temperature_500hPa'] = t500
            candidate['temperature_850hPa'] = t850
            candidate['thickness_500_850'] = float(thickness)
            
            # Определяем тип циклона по термической структуре
            # В арктических мезоциклонах обычно холодный центр на верхних уровнях
            
            # Получаем среднюю температуру в регионе для расчета аномалии
            avg_t500 = float(dataset[temp_var].sel(level=500).mean().values)
            avg_t850 = float(dataset[temp_var].sel(level=850).mean().values)
            
            t500_anomaly = t500 - avg_t500
            t850_anomaly = t850 - avg_t850
            
            candidate['t500_anomaly'] = float(t500_anomaly)
            candidate['t850_anomaly'] = float(t850_anomaly)
            
            # Определяем тип структуры
            if t500_anomaly < -0.5 and t850_anomaly > 0.5:
                thermal_type = 'cold_core'
            elif t500_anomaly > 0.5 and t850_anomaly > 0.5:
                thermal_type = 'warm_core'
            else:
                thermal_type = 'hybrid'
            
            candidate['thermal_type'] = thermal_type
        
        return candidates