"""
Модуль статистического анализа для системы ArcticCyclone.

Предоставляет классы и функции для статистического анализа
арктических циклонов, их свойств и характеристик.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import logging
from pathlib import Path
from datetime import datetime, timedelta
import calendar
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde

from models.cyclone import Cyclone
from models.classifications import CycloneType, CycloneIntensity, CycloneLifeStage
from core.exceptions import ArcticCycloneError

# Инициализация логгера
logger = logging.getLogger(__name__)


class CycloneStatistics:
    """
    Класс для расчета и анализа статистики циклонов.
    
    Предоставляет методы для расчета различных статистических
    характеристик циклонов и их свойств.
    """
    
    def __init__(self, min_latitude: float = 70.0):
        """
        Инициализирует объект статистики циклонов.
        
        Аргументы:
            min_latitude: Минимальная широта для анализа (градусы с.ш.).
        """
        self.min_latitude = min_latitude
        
        logger.debug(f"Инициализирован CycloneStatistics с минимальной широтой {min_latitude}°N")
    
    def calculate_basic_stats(self, cyclones: List[Cyclone]) -> Dict[str, Any]:
        """
        Рассчитывает базовую статистику для набора циклонов.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            
        Возвращает:
            Словарь с базовой статистикой.
        """
        try:
            # Фильтруем циклоны по широте
            filtered_cyclones = [c for c in cyclones if c.latitude >= self.min_latitude]
            
            if not filtered_cyclones:
                logger.warning(f"Нет циклонов севернее {self.min_latitude}°N для анализа")
                return {
                    'count': 0,
                    'min_latitude': self.min_latitude,
                    'error': f"Нет циклонов севернее {self.min_latitude}°N"
                }
            
            # Собираем данные по давлению
            pressures = [c.central_pressure for c in filtered_cyclones]
            
            # Собираем данные по положению
            latitudes = [c.latitude for c in filtered_cyclones]
            longitudes = [c.longitude for c in filtered_cyclones]
            
            # Собираем данные по времени
            timestamps = [c.time for c in filtered_cyclones]
            date_start = min(timestamps)
            date_end = max(timestamps)
            
            # Собираем данные по завихренности, если доступны
            vorticities = [
                c.parameters.vorticity_850hPa for c in filtered_cyclones
                if hasattr(c.parameters, 'vorticity_850hPa') and c.parameters.vorticity_850hPa is not None
            ]
            
            # Собираем данные по скорости ветра, если доступны
            wind_speeds = [
                c.parameters.max_wind_speed for c in filtered_cyclones
                if hasattr(c.parameters, 'max_wind_speed') and c.parameters.max_wind_speed is not None
            ]
            
            # Собираем данные по радиусу, если доступны
            radii = [
                c.parameters.radius for c in filtered_cyclones
                if hasattr(c.parameters, 'radius') and c.parameters.radius is not None
            ]
            
            # Собираем данные по типу, если доступны
            types = [
                c.parameters.thermal_type for c in filtered_cyclones
                if hasattr(c.parameters, 'thermal_type') and c.parameters.thermal_type != CycloneType.UNCLASSIFIED
            ]
            
            # Собираем данные по стадии жизненного цикла, если доступны
            life_stages = [
                c.life_stage for c in filtered_cyclones
                if hasattr(c, 'life_stage') and c.life_stage != CycloneLifeStage.UNKNOWN
            ]
            
            # Рассчитываем статистику по давлению
            pressure_stats = {
                'mean': np.mean(pressures),
                'median': np.median(pressures),
                'min': np.min(pressures),
                'max': np.max(pressures),
                'std': np.std(pressures),
                'count': len(pressures)
            }
            
            # Рассчитываем статистику по положению
            position_stats = {
                'mean_latitude': np.mean(latitudes),
                'median_latitude': np.median(latitudes),
                'min_latitude': np.min(latitudes),
                'max_latitude': np.max(latitudes),
                'std_latitude': np.std(latitudes),
                
                'mean_longitude': self._mean_longitude(longitudes),
                'median_longitude': self._median_longitude(longitudes),
                'min_longitude': np.min(longitudes),
                'max_longitude': np.max(longitudes),
                'std_longitude': self._std_longitude(longitudes)
            }
            
            # Формируем результат
            result = {
                'count': len(filtered_cyclones),
                'date_range': {
                    'start': date_start,
                    'end': date_end,
                    'days': (date_end - date_start).total_seconds() / (24 * 3600)
                },
                'pressure': pressure_stats,
                'position': position_stats,
                'min_latitude': self.min_latitude
            }
            
            # Добавляем статистику по завихренности, если доступна
            if vorticities:
                result['vorticity'] = {
                    'mean': np.mean(vorticities),
                    'median': np.median(vorticities),
                    'min': np.min(vorticities),
                    'max': np.max(vorticities),
                    'std': np.std(vorticities),
                    'count': len(vorticities)
                }
            
            # Добавляем статистику по скорости ветра, если доступна
            if wind_speeds:
                result['wind_speed'] = {
                    'mean': np.mean(wind_speeds),
                    'median': np.median(wind_speeds),
                    'min': np.min(wind_speeds),
                    'max': np.max(wind_speeds),
                    'std': np.std(wind_speeds),
                    'count': len(wind_speeds)
                }
            
            # Добавляем статистику по радиусу, если доступна
            if radii:
                result['radius'] = {
                    'mean': np.mean(radii),
                    'median': np.median(radii),
                    'min': np.min(radii),
                    'max': np.max(radii),
                    'std': np.std(radii),
                    'count': len(radii)
                }
            
            # Добавляем статистику по типам, если доступна
            if types:
                type_counts = {}
                for t in types:
                    type_name = t.value
                    if type_name not in type_counts:
                        type_counts[type_name] = 0
                    type_counts[type_name] += 1
                
                result['thermal_types'] = {
                    'counts': type_counts,
                    'total': len(types)
                }
            
            # Добавляем статистику по стадиям жизненного цикла, если доступна
            if life_stages:
                stage_counts = {}
                for s in life_stages:
                    stage_name = s.value
                    if stage_name not in stage_counts:
                        stage_counts[stage_name] = 0
                    stage_counts[stage_name] += 1
                
                result['life_stages'] = {
                    'counts': stage_counts,
                    'total': len(life_stages)
                }
            
            # Добавляем информацию о треках, если доступна
            if all(hasattr(c, 'track_id') for c in filtered_cyclones):
                track_ids = set(c.track_id for c in filtered_cyclones if c.track_id)
                result['track_count'] = len(track_ids)
            
            logger.info(f"Рассчитана базовая статистика для {len(filtered_cyclones)} циклонов")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка при расчете базовой статистики: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def _mean_longitude(self, longitudes: List[float]) -> float:
        """
        Рассчитывает среднюю долготу с учетом цикличности.
        
        Аргументы:
            longitudes: Список значений долготы.
            
        Возвращает:
            Средняя долгота.
        """
        # Преобразуем в радианы
        radians = np.radians(longitudes)
        
        # Рассчитываем средние значения синуса и косинуса
        mean_sin = np.mean(np.sin(radians))
        mean_cos = np.mean(np.cos(radians))
        
        # Преобразуем обратно в градусы
        mean_lon = np.degrees(np.arctan2(mean_sin, mean_cos))
        
        return mean_lon
    
    def _median_longitude(self, longitudes: List[float]) -> float:
        """
        Рассчитывает медианную долготу с учетом цикличности.
        
        Аргументы:
            longitudes: Список значений долготы.
            
        Возвращает:
            Медианная долгота.
        """
        # Преобразуем в радианы
        radians = np.radians(longitudes)
        
        # Находим индекс медианы на основе минимальной суммы расстояний
        distances = []
        for i, rad in enumerate(radians):
            # Рассчитываем угловое расстояние до каждой точки
            angular_distances = [min(abs(rad - r), 2 * np.pi - abs(rad - r)) for r in radians]
            # Суммируем расстояния
            distances.append(sum(angular_distances))
        
        # Индекс медианы - точка с минимальной суммой расстояний
        median_idx = np.argmin(distances)
        
        return longitudes[median_idx]
    
    def _std_longitude(self, longitudes: List[float]) -> float:
        """
        Рассчитывает стандартное отклонение долготы с учетом цикличности.
        
        Аргументы:
            longitudes: Список значений долготы.
            
        Возвращает:
            Стандартное отклонение долготы.
        """
        # Преобразуем в радианы
        radians = np.radians(longitudes)
        
        # Рассчитываем средние значения синуса и косинуса
        mean_sin = np.mean(np.sin(radians))
        mean_cos = np.mean(np.cos(radians))
        
        # Рассчитываем дисперсию
        R = np.sqrt(mean_sin**2 + mean_cos**2)
        variance = -2 * np.log(R)
        
        # Стандартное отклонение в радианах
        std_rad = np.sqrt(variance)
        
        # Преобразуем в градусы
        std_lon = np.degrees(std_rad)
        
        return std_lon
    
    def calculate_spatial_stats(self, cyclones: List[Cyclone], 
                              grid_resolution: float = 2.0,
                              region: Optional[Dict[str, float]] = None) -> xr.Dataset:
        """
        Рассчитывает пространственную статистику циклонов.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            grid_resolution: Разрешение сетки в градусах.
            region: Область для анализа в формате {'north': north, 'south': south, 'east': east, 'west': west}.
                  Если None, используется вся Арктика (севернее min_latitude).
            
        Возвращает:
            Набор данных xarray с пространственной статистикой.
        """
        try:
            # Определяем область анализа
            if region is None:
                region = {'north': 90.0, 'south': self.min_latitude, 'east': 180.0, 'west': -180.0}
            
            # Фильтруем циклоны по региону
            filtered_cyclones = [
                c for c in cyclones
                if (region['south'] <= c.latitude <= region['north'] and
                    (region['west'] <= c.longitude <= region['east'] or
                     (region['west'] > region['east'] and  # Случай пересечения 180° долготы
                      (c.longitude >= region['west'] or c.longitude <= region['east']))))
            ]
            
            if not filtered_cyclones:
                logger.warning(f"Нет циклонов в указанном регионе для анализа")
                raise ValueError(f"Нет циклонов в указанном регионе: {region}")
            
            # Создаем пространственную сетку
            lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
            lat_bins = np.arange(region['south'], region['north'] + grid_resolution, grid_resolution)
            
            # Инициализируем массивы для подсчета
            count_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
            pressure_sum = np.zeros_like(count_grid)
            pressure_min = np.full_like(count_grid, np.nan)
            vorticity_sum = np.zeros_like(count_grid)
            wind_sum = np.zeros_like(count_grid)
            radius_sum = np.zeros_like(count_grid)
            
            # Счетчики для параметров с возможными пропусками
            vorticity_count = np.zeros_like(count_grid)
            wind_count = np.zeros_like(count_grid)
            radius_count = np.zeros_like(count_grid)
            
            # Заполняем гриды данными о циклонах
            for cyclone in filtered_cyclones:
                # Определяем пространственные индексы
                lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
                lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
                
                # Проверяем, что индексы в пределах сетки
                if (0 <= lat_idx < len(lat_bins)-1 and 
                    0 <= lon_idx < len(lon_bins)-1):
                    # Увеличиваем счетчик
                    count_grid[lat_idx, lon_idx] += 1
                    
                    # Суммируем давление для расчета среднего
                    pressure_sum[lat_idx, lon_idx] += cyclone.central_pressure
                    
                    # Обновляем минимальное давление
                    if np.isnan(pressure_min[lat_idx, lon_idx]) or cyclone.central_pressure < pressure_min[lat_idx, lon_idx]:
                        pressure_min[lat_idx, lon_idx] = cyclone.central_pressure
                    
                    # Обрабатываем дополнительные параметры, если доступны
                    if hasattr(cyclone.parameters, 'vorticity_850hPa') and cyclone.parameters.vorticity_850hPa is not None:
                        vorticity_sum[lat_idx, lon_idx] += cyclone.parameters.vorticity_850hPa
                        vorticity_count[lat_idx, lon_idx] += 1
                    
                    if hasattr(cyclone.parameters, 'max_wind_speed') and cyclone.parameters.max_wind_speed is not None:
                        wind_sum[lat_idx, lon_idx] += cyclone.parameters.max_wind_speed
                        wind_count[lat_idx, lon_idx] += 1
                    
                    if hasattr(cyclone.parameters, 'radius') and cyclone.parameters.radius is not None:
                        radius_sum[lat_idx, lon_idx] += cyclone.parameters.radius
                        radius_count[lat_idx, lon_idx] += 1
            
            # Рассчитываем средние значения
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_pressure = np.where(count_grid > 0, pressure_sum / count_grid, np.nan)
                mean_vorticity = np.where(vorticity_count > 0, vorticity_sum / vorticity_count, np.nan)
                mean_wind = np.where(wind_count > 0, wind_sum / wind_count, np.nan)
                mean_radius = np.where(radius_count > 0, radius_sum / radius_count, np.nan)
            
            # Рассчитываем плотность циклонов (событий на единицу площади)
            # Площадь ячейки сетки зависит от широты
            area_grid = np.zeros_like(count_grid)
            
            for i in range(len(lat_bins)-1):
                lat = (lat_bins[i] + lat_bins[i+1]) / 2
                cell_area = self._calculate_grid_cell_area(
                    lat_bins[i], lat_bins[i+1], 
                    grid_resolution, grid_resolution
                )
                
                for j in range(len(lon_bins)-1):
                    area_grid[i, j] = cell_area
            
            # Рассчитываем плотность (циклонов на 1000 км²)
            with np.errstate(divide='ignore', invalid='ignore'):
                density_grid = np.where(area_grid > 0, count_grid / (area_grid / 1000), np.nan)
            
            # Средние координаты ячеек сетки
            lon_centers = lon_bins[:-1] + grid_resolution / 2
            lat_centers = lat_bins[:-1] + grid_resolution / 2
            
            # Создаем набор данных
            ds = xr.Dataset(
                data_vars={
                    'cyclone_count': (
                        ['latitude', 'longitude'], 
                        count_grid
                    ),
                    'density': (
                        ['latitude', 'longitude'], 
                        density_grid
                    ),
                    'mean_pressure': (
                        ['latitude', 'longitude'], 
                        mean_pressure
                    ),
                    'min_pressure': (
                        ['latitude', 'longitude'], 
                        pressure_min
                    ),
                    'mean_vorticity': (
                        ['latitude', 'longitude'], 
                        mean_vorticity
                    ),
                    'mean_wind_speed': (
                        ['latitude', 'longitude'], 
                        mean_wind
                    ),
                    'mean_radius': (
                        ['latitude', 'longitude'], 
                        mean_radius
                    ),
                    'cell_area': (
                        ['latitude', 'longitude'], 
                        area_grid
                    )
                },
                coords={
                    'latitude': lat_centers,
                    'longitude': lon_centers
                },
                attrs={
                    'description': 'Spatial statistics of cyclones',
                    'region': str(region),
                    'grid_resolution': f'{grid_resolution} degrees',
                    'creation_date': datetime.now().isoformat(),
                    'total_cyclones': len(filtered_cyclones),
                    'min_latitude': self.min_latitude
                }
            )
            
            logger.info(f"Рассчитана пространственная статистика для {len(filtered_cyclones)} циклонов")
            
            return ds
            
        except Exception as e:
            error_msg = f"Ошибка при расчете пространственной статистики: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def _calculate_grid_cell_area(self, lat1: float, lat2: float, 
                                lon1: float, lon2: float) -> float:
        """
        Рассчитывает площадь ячейки сетки на сфере.
        
        Аргументы:
            lat1: Южная граница ячейки (градусы).
            lat2: Северная граница ячейки (градусы).
            lon1: Западная граница ячейки (градусы).
            lon2: Восточная граница ячейки (градусы).
            
        Возвращает:
            Площадь в квадратных километрах.
        """
        # Радиус Земли в километрах
        R = 6371.0
        
        # Конвертируем в радианы
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)
        
        # Рассчитываем площадь
        area = (R**2) * abs(lon2_rad - lon1_rad) * abs(np.sin(lat2_rad) - np.sin(lat1_rad))
        
        return area
    
    def calculate_temporal_stats(self, cyclones: List[Cyclone],
                               temporal_resolution: str = 'monthly',
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Рассчитывает временную статистику циклонов.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            temporal_resolution: Временное разрешение анализа ('daily', 'monthly', 'seasonal', 'annual').
            start_date: Начальная дата анализа. Если None, используется минимальная дата в данных.
            end_date: Конечная дата анализа. Если None, используется максимальная дата в данных.
            
        Возвращает:
            DataFrame с временной статистикой.
        """
        try:
            # Фильтруем циклоны по широте
            filtered_cyclones = [c for c in cyclones if c.latitude >= self.min_latitude]
            
            if not filtered_cyclones:
                logger.warning(f"Нет циклонов севернее {self.min_latitude}°N для анализа")
                return pd.DataFrame()
            
            # Определяем временной диапазон
            if start_date is None or end_date is None:
                dates = [c.time for c in filtered_cyclones]
                min_date = min(dates)
                max_date = max(dates)
                
                start_date = start_date or min_date
                end_date = end_date or max_date
            
            # Создаем временную сетку в зависимости от разрешения
            if temporal_resolution == 'daily':
                time_bins = pd.date_range(start=start_date, end=end_date, freq='D')
                time_labels = [d.strftime('%Y-%m-%d') for d in time_bins]
                bin_edges = time_bins.tolist() + [time_bins[-1] + pd.Timedelta(days=1)]
            elif temporal_resolution == 'monthly':
                time_bins = pd.date_range(start=start_date, end=end_date, freq='MS')
                time_labels = [d.strftime('%Y-%m') for d in time_bins]
                bin_edges = time_bins.tolist() + [time_bins[-1] + pd.DateOffset(months=1)]
            elif temporal_resolution == 'seasonal':
                # Определяем начало сезона, ближайшее к start_date
                season_months = {
                    'DJF': [12, 1, 2],
                    'MAM': [3, 4, 5],
                    'JJA': [6, 7, 8],
                    'SON': [9, 10, 11]
                }
                
                # Находим первый день ближайшего сезона
                month = start_date.month
                year = start_date.year
                
                if month in season_months['DJF']:
                    if month == 12:
                        season_start = datetime(year, 12, 1)
                    else:
                        season_start = datetime(year, 12, 1) - pd.DateOffset(years=1)
                elif month in season_months['MAM']:
                    season_start = datetime(year, 3, 1)
                elif month in season_months['JJA']:
                    season_start = datetime(year, 6, 1)
                else:  # 'SON'
                    season_start = datetime(year, 9, 1)
                
                # Создаем последовательность сезонов
                time_bins = []
                season_names = []
                current_date = season_start
                
                while current_date <= end_date:
                    time_bins.append(current_date)
                    
                    month = current_date.month
                    year = current_date.year
                    
                    if month == 12:
                        season_names.append(f"DJF {year+1}")
                        next_date = datetime(year+1, 3, 1)
                    elif month == 3:
                        season_names.append(f"MAM {year}")
                        next_date = datetime(year, 6, 1)
                    elif month == 6:
                        season_names.append(f"JJA {year}")
                        next_date = datetime(year, 9, 1)
                    else:  # month == 9
                        season_names.append(f"SON {year}")
                        next_date = datetime(year, 12, 1)
                    
                    current_date = next_date
                
                bin_edges = time_bins + [current_date]
                time_labels = season_names
            elif temporal_resolution == 'annual':
                # Определяем начало года для начальной даты
                year_start = datetime(start_date.year, 1, 1)
                
                # Создаем последовательность лет
                time_bins = pd.date_range(start=year_start, end=end_date, freq='AS')
                time_labels = [d.strftime('%Y') for d in time_bins]
                bin_edges = time_bins.tolist() + [time_bins[-1] + pd.DateOffset(years=1)]
            else:
                raise ValueError(f"Неподдерживаемое временное разрешение: {temporal_resolution}")
            
            # Инициализируем DataFrame с результатами
            result = pd.DataFrame(index=time_labels)
            result['start_date'] = time_bins
            result['end_date'] = bin_edges[1:]
            result['count'] = 0
            
            # Счетчики для параметров с возможными пропусками
            pressure_values = {label: [] for label in time_labels}
            vorticity_values = {label: [] for label in time_labels}
            wind_values = {label: [] for label in time_labels}
            radius_values = {label: [] for label in time_labels}
            
            # Распределяем циклоны по временным интервалам
            for cyclone in filtered_cyclones:
                # Находим соответствующий временной интервал
                bin_idx = None
                
                for i in range(len(bin_edges) - 1):
                    if bin_edges[i] <= cyclone.time < bin_edges[i+1]:
                        bin_idx = i
                        break
                
                if bin_idx is not None:
                    # Увеличиваем счетчик циклонов
                    label = time_labels[bin_idx]
                    result.loc[label, 'count'] += 1
                    
                    # Добавляем параметры для расчета статистики
                    pressure_values[label].append(cyclone.central_pressure)
                    
                    if hasattr(cyclone.parameters, 'vorticity_850hPa') and cyclone.parameters.vorticity_850hPa is not None:
                        vorticity_values[label].append(cyclone.parameters.vorticity_850hPa)
                    
                    if hasattr(cyclone.parameters, 'max_wind_speed') and cyclone.parameters.max_wind_speed is not None:
                        wind_values[label].append(cyclone.parameters.max_wind_speed)
                    
                    if hasattr(cyclone.parameters, 'radius') and cyclone.parameters.radius is not None:
                        radius_values[label].append(cyclone.parameters.radius)
            
            # Рассчитываем статистики для каждого параметра
            for label in time_labels:
                # Давление
                pressures = pressure_values[label]
                if pressures:
                    result.loc[label, 'mean_pressure'] = np.mean(pressures)
                    result.loc[label, 'min_pressure'] = np.min(pressures)
                    result.loc[label, 'max_pressure'] = np.max(pressures)
                    result.loc[label, 'std_pressure'] = np.std(pressures)
                
                # Завихренность
                vorticities = vorticity_values[label]
                if vorticities:
                    result.loc[label, 'mean_vorticity'] = np.mean(vorticities)
                    result.loc[label, 'max_vorticity'] = np.max(vorticities)
                    result.loc[label, 'std_vorticity'] = np.std(vorticities)
                
                # Скорость ветра
                winds = wind_values[label]
                if winds:
                    result.loc[label, 'mean_wind_speed'] = np.mean(winds)
                    result.loc[label, 'max_wind_speed'] = np.max(winds)
                    result.loc[label, 'std_wind_speed'] = np.std(winds)
                
                # Радиус
                radii = radius_values[label]
                if radii:
                    result.loc[label, 'mean_radius'] = np.mean(radii)
                    result.loc[label, 'max_radius'] = np.max(radii)
                    result.loc[label, 'std_radius'] = np.std(radii)
            
            # Добавляем колонку с плотностью циклонов
            # Рассчитываем длительность каждого интервала в днях
            if temporal_resolution == 'daily':
                interval_days = 1
            elif temporal_resolution == 'monthly':
                # Количество дней в месяце
                result['interval_days'] = [(end - start).days for start, end in zip(result['start_date'], result['end_date'])]
                interval_days = result['interval_days']
            elif temporal_resolution == 'seasonal':
                # Длительность сезона - примерно 3 месяца или ~90-92 дня
                result['interval_days'] = [(end - start).days for start, end in zip(result['start_date'], result['end_date'])]
                interval_days = result['interval_days']
            elif temporal_resolution == 'annual':
                # Количество дней в году (учитываем високосные)
                result['interval_days'] = [(end - start).days for start, end in zip(result['start_date'], result['end_date'])]
                interval_days = result['interval_days']
            
            # Циклонов в день
            result['cyclones_per_day'] = result['count'] / interval_days
            
            # Маркируем временное разрешение
            result['resolution'] = temporal_resolution
            
            # Добавляем метаданные
            result.attrs = {
                'description': f'Temporal statistics of cyclones with {temporal_resolution} resolution',
                'min_latitude': self.min_latitude,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_cyclones': len(filtered_cyclones),
                'creation_date': datetime.now().isoformat()
            }
            
            logger.info(f"Рассчитана временная статистика с разрешением {temporal_resolution} для {len(filtered_cyclones)} циклонов")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка при расчете временной статистики: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def calculate_ensemble_stats(self, cyclones_list: List[List[Cyclone]],
                               max_distance: float = 500.0,
                               max_time_diff: float = 12.0) -> Dict[str, Any]:
        """
        Рассчитывает статистику ансамбля циклонов.
        
        Аргументы:
            cyclones_list: Список наборов циклонов (ансамбль прогнозов или наблюдений).
            max_distance: Максимальное расстояние (км) для сопоставления циклонов.
            max_time_diff: Максимальная разница во времени (часы) для сопоставления.
            
        Возвращает:
            Словарь со статистикой ансамбля.
        """
        try:
            if not cyclones_list:
                raise ValueError("Пустой список наборов циклонов")
            
            # Фильтруем циклоны по широте
            filtered_ensembles = [
                [c for c in cyclones if c.latitude >= self.min_latitude]
                for cyclones in cyclones_list
            ]
            
            # Проверяем, есть ли данные после фильтрации
            if not any(filtered_ensembles):
                logger.warning(f"Нет циклонов севернее {self.min_latitude}°N для анализа")
                return {
                    'error': f"Нет циклонов севернее {self.min_latitude}°N",
                    'min_latitude': self.min_latitude
                }
            
            # Количество ансамблей
            ensemble_count = len(filtered_ensembles)
            
            # Рассчитываем среднее количество циклонов в ансамбле
            cyclone_counts = [len(ensemble) for ensemble in filtered_ensembles]
            mean_cyclone_count = np.mean(cyclone_counts)
            std_cyclone_count = np.std(cyclone_counts)
            
            # Создаем плоский список всех циклонов для группировки
            all_cyclones = []
            for i, ensemble in enumerate(filtered_ensembles):
                for cyclone in ensemble:
                    all_cyclones.append((i, cyclone))
            
            # Группируем циклоны по близости в пространстве и времени
            cyclone_groups = []
            used_indices = set()
            
            for i, (ensemble_idx, cyclone) in enumerate(all_cyclones):
                if i in used_indices:
                    continue
                
                # Создаем новую группу
                group = [(ensemble_idx, cyclone)]
                used_indices.add(i)
                
                # Ищем соответствующие циклоны в других ансамблях
                for j, (other_ensemble_idx, other_cyclone) in enumerate(all_cyclones):
                    if j in used_indices or ensemble_idx == other_ensemble_idx:
                        continue
                    
                    # Рассчитываем расстояние
                    distance = self._calculate_distance(
                        cyclone.latitude, cyclone.longitude,
                        other_cyclone.latitude, other_cyclone.longitude
                    )
                    
                    # Рассчитываем временную разницу
                    time_diff = abs((cyclone.time - other_cyclone.time).total_seconds()) / 3600
                    
                    # Проверяем соответствие
                    if distance <= max_distance and time_diff <= max_time_diff:
                        group.append((other_ensemble_idx, other_cyclone))
                        used_indices.add(j)
                
                cyclone_groups.append(group)
            
            # Счетчик групп по размеру (количеству ансамблей)
            group_size_counts = {}
            for group in cyclone_groups:
                ensembles_in_group = set(idx for idx, _ in group)
                size = len(ensembles_in_group)
                
                if size not in group_size_counts:
                    group_size_counts[size] = 0
                group_size_counts[size] += 1
            
            # Рассчитываем количество групп с циклонами из всех ансамблей
            full_agreement_groups = sum(1 for group in cyclone_groups 
                                       if len(set(idx for idx, _ in group)) == ensemble_count)
            
            # Рассчитываем среднюю позиционную неопределенность
            position_uncertainties = []
            
            for group in cyclone_groups:
                if len(group) < 2:
                    continue
                
                # Координаты циклонов в группе
                lats = [c.latitude for _, c in group]
                lons = [c.longitude for _, c in group]
                
                # Рассчитываем среднее положение
                mean_lat = np.mean(lats)
                mean_lon = self._mean_longitude(lons)
                
                # Рассчитываем среднее расстояние от циклонов до центра
                distances = [
                    self._calculate_distance(mean_lat, mean_lon, c.latitude, c.longitude)
                    for _, c in group
                ]
                
                position_uncertainties.append(np.mean(distances))
            
            mean_position_uncertainty = np.mean(position_uncertainties) if position_uncertainties else np.nan
            
            # Рассчитываем статистику по давлению
            pressure_uncertainties = []
            
            for group in cyclone_groups:
                if len(group) < 2:
                    continue
                
                # Давления циклонов в группе
                pressures = [c.central_pressure for _, c in group]
                
                # Стандартное отклонение давления в группе
                pressure_uncertainties.append(np.std(pressures))
            
            mean_pressure_uncertainty = np.mean(pressure_uncertainties) if pressure_uncertainties else np.nan
            
            # Формируем результат
            result = {
                'ensemble_count': ensemble_count,
                'mean_cyclone_count': mean_cyclone_count,
                'std_cyclone_count': std_cyclone_count,
                'cyclone_group_count': len(cyclone_groups),
                'group_size_distribution': group_size_counts,
                'full_agreement_groups': full_agreement_groups,
                'agreement_ratio': full_agreement_groups / len(cyclone_groups) if cyclone_groups else 0,
                'mean_position_uncertainty_km': mean_position_uncertainty,
                'mean_pressure_uncertainty_hPa': mean_pressure_uncertainty,
                'min_latitude': self.min_latitude,
                'max_distance_km': max_distance,
                'max_time_diff_hours': max_time_diff
            }
            
            logger.info(f"Рассчитана статистика ансамбля для {ensemble_count} наборов с общим количеством {len(cyclone_groups)} групп циклонов")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка при расчете статистики ансамбля: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Рассчитывает расстояние между двумя точками на сфере (формула гаверсинуса).
        
        Аргументы:
            lat1: Широта первой точки (градусы).
            lon1: Долгота первой точки (градусы).
            lat2: Широта второй точки (градусы).
            lon2: Долгота второй точки (градусы).
            
        Возвращает:
            Расстояние в километрах.
        """
        from math import radians, sin, cos, sqrt, atan2
        
        # Конвертируем координаты из градусов в радианы
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Формула гаверсинуса
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Радиус Земли в километрах
        
        return r * c
    
    def calculate_cyclone_flux(self, cyclone_tracks: List[List[Cyclone]],
                             grid_resolution: float = 2.0) -> xr.Dataset:
        """
        Рассчитывает поток циклонов через границы ячеек сетки.
        
        Аргументы:
            cyclone_tracks: Список треков циклонов (каждый трек - список циклонов).
            grid_resolution: Разрешение сетки в градусах.
            
        Возвращает:
            Набор данных xarray с потоками циклонов.
        """
        try:
            # Проверяем наличие треков
            if not cyclone_tracks:
                raise ValueError("Пустой список треков циклонов")
            
            # Фильтруем треки по широте
            filtered_tracks = []
            for track in cyclone_tracks:
                if track and any(c.latitude >= self.min_latitude for c in track):
                    # Обрезаем трек до арктического региона
                    arctic_portion = [c for c in track if c.latitude >= self.min_latitude]
                    if len(arctic_portion) >= 2:  # Минимум две точки для определения направления
                        filtered_tracks.append(arctic_portion)
            
            if not filtered_tracks:
                logger.warning(f"Нет треков севернее {self.min_latitude}°N для анализа")
                raise ValueError(f"Нет треков севернее {self.min_latitude}°N")
            
            # Создаем пространственную сетку
            lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
            lat_bins = np.arange(self.min_latitude, 90 + grid_resolution, grid_resolution)
            
            # Инициализируем массивы для потоков
            # u_flux - поток в зональном направлении (восток-запад)
            # v_flux - поток в меридиональном направлении (север-юг)
            u_flux = np.zeros((len(lat_bins)-1, len(lon_bins)))
            v_flux = np.zeros((len(lat_bins), len(lon_bins)-1))
            
            # Инициализируем массивы для подсчета пересечений
            crossings_u = np.zeros_like(u_flux)
            crossings_v = np.zeros_like(v_flux)
            
            # Для каждого трека
            for track in filtered_tracks:
                # Сортируем по времени
                sorted_track = sorted(track, key=lambda c: c.time)
                
                # Для каждой пары последовательных точек в треке
                for i in range(len(sorted_track) - 1):
                    c1 = sorted_track[i]
                    c2 = sorted_track[i+1]
                    
                    # Координаты текущей и следующей точки
                    lat1, lon1 = c1.latitude, c1.longitude
                    lat2, lon2 = c2.latitude, c2.longitude
                    
                    # Проверяем пересечение меридианов (долготных линий)
                    for j in range(len(lon_bins)):
                        lon_boundary = lon_bins[j]
                        
                        # Проверяем, пересекает ли сегмент трека меридиан
                        if ((lon1 < lon_boundary < lon2) or
                            (lon2 < lon_boundary < lon1) or
                            (lon1 == lon_boundary and lon2 > lon_boundary) or
                            (lon2 == lon_boundary and lon1 > lon_boundary)):
                            
                            # Находим параметр t, где происходит пересечение
                            if lon2 != lon1:  # Избегаем деления на ноль
                                t = (lon_boundary - lon1) / (lon2 - lon1)
                                # Интерполируем широту в точке пересечения
                                lat_crossing = lat1 + t * (lat2 - lat1)
                                
                                # Находим индекс ячейки по широте
                                lat_idx = np.searchsorted(lat_bins, lat_crossing) - 1
                                
                                if 0 <= lat_idx < len(lat_bins) - 1:
                                    # Определяем направление пересечения
                                    direction = 1 if lon2 > lon1 else -1
                                    
                                    # Увеличиваем счетчик пересечений
                                    crossings_u[lat_idx, j] += 1
                                    
                                    # Обновляем поток
                                    u_flux[lat_idx, j] += direction
                    
                    # Проверяем пересечение параллелей (широтных линий)
                    for j in range(len(lat_bins)):
                        lat_boundary = lat_bins[j]
                        
                        # Проверяем, пересекает ли сегмент трека параллель
                        if ((lat1 < lat_boundary < lat2) or
                            (lat2 < lat_boundary < lat1) or
                            (lat1 == lat_boundary and lat2 > lat_boundary) or
                            (lat2 == lat_boundary and lat1 > lat_boundary)):
                            
                            # Находим параметр t, где происходит пересечение
                            if lat2 != lat1:  # Избегаем деления на ноль
                                t = (lat_boundary - lat1) / (lat2 - lat1)
                                # Интерполируем долготу в точке пересечения
                                lon_crossing = lon1 + t * (lon2 - lon1)
                                
                                # Обрабатываем пересечение линии смены дат
                                if abs(lon2 - lon1) > 180:
                                    # Нормализуем, учитывая пересечение 180° долготы
                                    if lon1 > 0:
                                        lon1 -= 360
                                    else:
                                        lon2 -= 360
                                    
                                    # Пересчитываем точку пересечения
                                    t = (lat_boundary - lat1) / (lat2 - lat1)
                                    lon_crossing = lon1 + t * (lon2 - lon1)
                                    
                                    # Приводим обратно к диапазону -180..180
                                    if lon_crossing < -180:
                                        lon_crossing += 360
                                
                                # Находим индекс ячейки по долготе
                                lon_idx = np.searchsorted(lon_bins, lon_crossing) - 1
                                
                                if 0 <= lon_idx < len(lon_bins) - 1:
                                    # Определяем направление пересечения
                                    direction = 1 if lat2 > lat1 else -1
                                    
                                    # Увеличиваем счетчик пересечений
                                    crossings_v[j, lon_idx] += 1
                                    
                                    # Обновляем поток
                                    v_flux[j, lon_idx] += direction
            
            # Нормализуем потоки количеством пересечений
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_u_flux = np.where(crossings_u > 0, u_flux / crossings_u, 0)
                normalized_v_flux = np.where(crossings_v > 0, v_flux / crossings_v, 0)
            
            # Координаты центров ячеек сетки и границ
            lon_centers = lon_bins[:-1] + grid_resolution / 2
            lat_centers = lat_bins[:-1] + grid_resolution / 2
            
            # Создаем набор данных
            ds = xr.Dataset(
                data_vars={
                    'u_flux': (
                        ['latitude', 'longitude_bounds'], 
                        normalized_u_flux
                    ),
                    'v_flux': (
                        ['latitude_bounds', 'longitude'], 
                        normalized_v_flux
                    ),
                    'crossings_u': (
                        ['latitude', 'longitude_bounds'], 
                        crossings_u
                    ),
                    'crossings_v': (
                        ['latitude_bounds', 'longitude'], 
                        crossings_v
                    )
                },
                coords={
                    'latitude': lat_centers,
                    'longitude': lon_centers,
                    'latitude_bounds': lat_bins,
                    'longitude_bounds': lon_bins
                },
                attrs={
                    'description': 'Cyclone flux analysis',
                    'grid_resolution': f'{grid_resolution} degrees',
                    'creation_date': datetime.now().isoformat(),
                    'total_tracks': len(filtered_tracks),
                    'min_latitude': self.min_latitude
                }
            )
            
            logger.info(f"Рассчитаны потоки циклонов для {len(filtered_tracks)} треков")
            
            return ds
            
        except Exception as e:
            error_msg = f"Ошибка при расчете потоков циклонов: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def identify_genesis_regions(self, cyclone_tracks: List[List[Cyclone]],
                              grid_resolution: float = 2.0,
                              min_cluster_size: int = 5) -> xr.Dataset:
        """
        Идентифицирует регионы зарождения циклонов.
        
        Аргументы:
            cyclone_tracks: Список треков циклонов (каждый трек - список циклонов).
            grid_resolution: Разрешение сетки в градусах.
            min_cluster_size: Минимальный размер кластера для выделения региона.
            
        Возвращает:
            Набор данных xarray с регионами зарождения.
        """
        try:
            # Проверяем наличие треков
            if not cyclone_tracks:
                raise ValueError("Пустой список треков циклонов")
            
            # Извлекаем точки зарождения (первые точки треков)
            genesis_points = []
            for track in cyclone_tracks:
                if track:
                    # Сортируем по времени
                    sorted_track = sorted(track, key=lambda c: c.time)
                    # Берем первую точку трека в Арктике
                    for cyclone in sorted_track:
                        if cyclone.latitude >= self.min_latitude:
                            genesis_points.append(cyclone)
                            break
            
            if not genesis_points:
                logger.warning(f"Нет точек зарождения севернее {self.min_latitude}°N для анализа")
                raise ValueError(f"Нет точек зарождения севернее {self.min_latitude}°N")
            
            # Создаем пространственную сетку
            lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
            lat_bins = np.arange(self.min_latitude, 90 + grid_resolution, grid_resolution)
            
            # Инициализируем массив плотности точек зарождения
            genesis_density = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
            
            # Заполняем массив плотности
            for cyclone in genesis_points:
                lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
                lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
                
                if 0 <= lat_idx < len(lat_bins)-1 and 0 <= lon_idx < len(lon_bins)-1:
                    genesis_density[lat_idx, lon_idx] += 1
            
            # Координаты центров ячеек сетки
            lon_centers = lon_bins[:-1] + grid_resolution / 2
            lat_centers = lat_bins[:-1] + grid_resolution / 2
            
            # Применяем сглаживание
            from scipy.ndimage import gaussian_filter
            smoothed_density = gaussian_filter(genesis_density, sigma=1.0)
            
            # Находим кластеры точек зарождения с использованием DBSCAN
            # Подготавливаем данные для кластеризации
            coords = []
            for cyclone in genesis_points:
                # Нормализуем координаты для кластеризации
                # Преобразуем широту в радиусы от центра (90° соответствует 0, 0° соответствует 1)
                r = (90 - cyclone.latitude) / 90
                
                # Преобразуем долготу в радианы для перехода в декартовы координаты
                phi = np.radians(cyclone.longitude)
                
                # Преобразуем в декартовы координаты для корректной кластеризации на сфере
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                
                coords.append([x, y])
            
            # Применяем DBSCAN
            from sklearn.cluster import DBSCAN
            
            # Определяем эпсилон (максимальное расстояние между точками в кластере)
            # Примерно соответствует 300 км на поверхности Земли
            epsilon = 300 / (6371 * np.pi)  # в нормализованных координатах
            
            db = DBSCAN(eps=epsilon, min_samples=min_cluster_size)
            clusters = db.fit_predict(coords)
            
            # Создаем DataFrame с информацией о точках зарождения и кластерах
            genesis_data = pd.DataFrame({
                'latitude': [c.latitude for c in genesis_points],
                'longitude': [c.longitude for c in genesis_points],
                'time': [c.time for c in genesis_points],
                'pressure': [c.central_pressure for c in genesis_points],
                'cluster': clusters
            })
            
            # Подсчитываем количество точек в каждом кластере
            cluster_stats = genesis_data[genesis_data['cluster'] >= 0].groupby('cluster').agg({
                'latitude': ['mean', 'min', 'max', 'count'],
                'longitude': ['mean', 'min', 'max'],
                'pressure': ['mean', 'min', 'max']
            })
            
            # Преобразуем статистику кластеров в массив для xarray
            cluster_count = len(cluster_stats)
            cluster_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
            
            # Инициализируем массивы для характеристик кластеров
            cluster_info = []
            
            for cluster_id, stats in enumerate(cluster_stats.itertuples()):
                # Среднее положение кластера
                mean_lat = stats.latitude[0]  # mean
                mean_lon = stats.longitude[0]  # mean
                
                # Находим ячейку сетки для среднего положения кластера
                lon_idx = np.searchsorted(lon_bins, mean_lon) - 1
                lat_idx = np.searchsorted(lat_bins, mean_lat) - 1
                
                if 0 <= lat_idx < len(lat_bins)-1 and 0 <= lon_idx < len(lon_bins)-1:
                    # Маркируем ячейку номером кластера + 1 (0 - отсутствие кластера)
                    cluster_grid[lat_idx, lon_idx] = cluster_id + 1
                
                # Собираем информацию о кластере
                points_in_cluster = stats.latitude[3]  # count
                
                cluster_info.append({
                    'cluster_id': cluster_id,
                    'mean_latitude': mean_lat,
                    'mean_longitude': mean_lon,
                    'min_latitude': stats.latitude[1],  # min
                    'max_latitude': stats.latitude[2],  # max
                    'min_longitude': stats.longitude[1],  # min
                    'max_longitude': stats.longitude[2],  # max
                    'mean_pressure': stats.pressure[0],  # mean
                    'min_pressure': stats.pressure[1],  # min
                    'max_pressure': stats.pressure[2],  # max
                    'point_count': points_in_cluster,
                    'percentage': points_in_cluster / len(genesis_points) * 100
                })
            
            # Создаем набор данных
            ds = xr.Dataset(
                data_vars={
                    'genesis_density': (
                        ['latitude', 'longitude'], 
                        genesis_density
                    ),
                    'smoothed_density': (
                        ['latitude', 'longitude'], 
                        smoothed_density
                    ),
                    'cluster_grid': (
                        ['latitude', 'longitude'], 
                        cluster_grid
                    )
                },
                coords={
                    'latitude': lat_centers,
                    'longitude': lon_centers
                },
                attrs={
                    'description': 'Cyclone genesis regions analysis',
                    'grid_resolution': f'{grid_resolution} degrees',
                    'creation_date': datetime.now().isoformat(),
                    'total_genesis_points': len(genesis_points),
                    'total_clusters': cluster_count,
                    'min_latitude': self.min_latitude,
                    'cluster_info': str(cluster_info)
                }
            )
            
            logger.info(f"Идентифицировано {cluster_count} регионов зарождения циклонов из {len(genesis_points)} точек")
            
            return ds
            
        except Exception as e:
            error_msg = f"Ошибка при идентификации регионов зарождения циклонов: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def estimate_uncertainties(self, cyclones: List[Cyclone], 
                            bootstrap_samples: int = 1000,
                            confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Оценивает неопределенности в статистических характеристиках циклонов.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            bootstrap_samples: Количество бутстрап-выборок.
            confidence_level: Уровень доверия для интервалов.
            
        Возвращает:
            Словарь с оценками неопределенностей.
        """
        try:
            # Фильтруем циклоны по широте
            filtered_cyclones = [c for c in cyclones if c.latitude >= self.min_latitude]
            
            if not filtered_cyclones:
                logger.warning(f"Нет циклонов севернее {self.min_latitude}°N для анализа")
                return {
                    'error': f"Нет циклонов севернее {self.min_latitude}°N",
                    'min_latitude': self.min_latitude
                }
            
            # Определяем параметры для анализа
            parameters = {
                'latitude': [c.latitude for c in filtered_cyclones],
                'longitude': [c.longitude for c in filtered_cyclones],
                'pressure': [c.central_pressure for c in filtered_cyclones]
            }
            
            # Добавляем параметры с возможными пропусками
            vorticity_values = [
                c.parameters.vorticity_850hPa for c in filtered_cyclones
                if hasattr(c.parameters, 'vorticity_850hPa') and c.parameters.vorticity_850hPa is not None
            ]
            if vorticity_values:
                parameters['vorticity'] = vorticity_values
            
            wind_values = [
                c.parameters.max_wind_speed for c in filtered_cyclones
                if hasattr(c.parameters, 'max_wind_speed') and c.parameters.max_wind_speed is not None
            ]
            if wind_values:
                parameters['wind_speed'] = wind_values
            
            radius_values = [
                c.parameters.radius for c in filtered_cyclones
                if hasattr(c.parameters, 'radius') and c.parameters.radius is not None
            ]
            if radius_values:
                parameters['radius'] = radius_values
            
            # Выполняем бутстрап-анализ для каждого параметра
            results = {}
            
            for param_name, param_values in parameters.items():
                # Выполняем бутстрап для среднего
                bootstrap_means = []
                bootstrap_medians = []
                
                np.random.seed(42)  # Для воспроизводимости
                
                for _ in range(bootstrap_samples):
                    # Генерируем бутстрап-выборку
                    sample_indices = np.random.choice(len(param_values), len(param_values), replace=True)
                    bootstrap_sample = [param_values[i] for i in sample_indices]
                    
                    # Рассчитываем статистики
                    if param_name == 'longitude':
                        bootstrap_means.append(self._mean_longitude(bootstrap_sample))
                        bootstrap_medians.append(self._median_longitude(bootstrap_sample))
                    else:
                        bootstrap_means.append(np.mean(bootstrap_sample))
                        bootstrap_medians.append(np.median(bootstrap_sample))
                
                # Рассчитываем доверительные интервалы
                alpha = (1 - confidence_level) / 2
                ci_lower_mean = np.percentile(bootstrap_means.compressed() if hasattr(bootstrap_means, 'compressed') else bootstrap_means, alpha * 100)
                ci_upper_mean = np.percentile(bootstrap_means.compressed() if hasattr(bootstrap_means, 'compressed') else bootstrap_means, (1 - alpha) * 100)
                
                ci_lower_median = np.percentile(bootstrap_medians.compressed() if hasattr(bootstrap_medians, 'compressed') else bootstrap_medians, alpha * 100)
                ci_upper_median = np.percentile(bootstrap_medians.compressed() if hasattr(bootstrap_medians, 'compressed') else bootstrap_medians, (1 - alpha) * 100)
                
                # Рассчитываем стандартное отклонение бутстрап-выборок
                bootstrap_std = np.std(bootstrap_means)
                
                # Собираем результаты
                results[param_name] = {
                    'mean': np.mean(param_values) if param_name != 'longitude' else self._mean_longitude(param_values),
                    'median': np.median(param_values) if param_name != 'longitude' else self._median_longitude(param_values),
                    'std': np.std(param_values) if param_name != 'longitude' else self._std_longitude(param_values),
                    'bootstrap_std': bootstrap_std,
                    'ci_mean_lower': ci_lower_mean,
                    'ci_mean_upper': ci_upper_mean,
                    'ci_median_lower': ci_lower_median,
                    'ci_median_upper': ci_upper_median,
                    'ci_width_mean': ci_upper_mean - ci_lower_mean,
                    'ci_width_median': ci_upper_median - ci_lower_median,
                    'sample_size': len(param_values)
                }
            
            # Добавляем метаданные
            results['metadata'] = {
                'bootstrap_samples': bootstrap_samples,
                'confidence_level': confidence_level,
                'total_cyclones': len(filtered_cyclones),
                'min_latitude': self.min_latitude,
                'creation_date': datetime.now().isoformat()
            }
            
            logger.info(f"Оценены неопределенности для {len(filtered_cyclones)} циклонов с {bootstrap_samples} бутстрап-выборками")
            
            return results
            
        except Exception as e:
            error_msg = f"Ошибка при оценке неопределенностей: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def perform_trend_analysis(self, cyclone_stats: pd.DataFrame,
                            parameter: str = 'count',
                            period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """
        Выполняет анализ трендов в характеристиках циклонов.
        
        Аргументы:
            cyclone_stats: DataFrame с временной статистикой циклонов.
            parameter: Параметр для анализа тренда.
            period: Период для анализа в формате (начало, конец).
                   Если None, используется весь доступный период.
            
        Возвращает:
            Словарь с результатами анализа тренда.
        """
        try:
            # Проверяем наличие данных
            if cyclone_stats.empty:
                raise ValueError("Пустой DataFrame с данными")
            
            # Проверяем наличие параметра
            if parameter not in cyclone_stats.columns:
                raise ValueError(f"Параметр '{parameter}' отсутствует в данных")
            
            # Фильтруем данные по периоду, если указан
            if period is not None:
                start_date, end_date = period
                
                # Проверяем, есть ли колонки с датами
                date_columns = [col for col in cyclone_stats.columns if 'date' in col.lower()]
                
                if 'start_date' in cyclone_stats.columns:
                    filtered_stats = cyclone_stats[
                        (cyclone_stats['start_date'] >= start_date) & 
                        (cyclone_stats['start_date'] <= end_date)
                    ]
                elif date_columns:
                    # Используем первую найденную колонку с датой
                    date_col = date_columns[0]
                    filtered_stats = cyclone_stats[
                        (cyclone_stats[date_col] >= start_date) & 
                        (cyclone_stats[date_col] <= end_date)
                    ]
                else:
                    # Если нет колонок с датами, используем индекс
                    filtered_stats = cyclone_stats
                    logger.warning("Не найдены колонки с датами, используется весь DataFrame")
            else:
                filtered_stats = cyclone_stats
            
            if filtered_stats.empty:
                raise ValueError("После фильтрации по периоду не осталось данных")
            
            # Определяем ось X для регрессии
            if 'start_date' in filtered_stats.columns:
                # Преобразуем даты в числовой формат (дни от начала периода)
                x = (filtered_stats['start_date'] - filtered_stats['start_date'].min()).dt.total_seconds() / (24 * 3600)
                x_type = 'date'
                x_label = 'Дата'
            else:
                # Используем индекс
                x = np.arange(len(filtered_stats))
                x_type = 'index'
                x_label = 'Индекс'
            
            # Получаем значения параметра
            y = filtered_stats[parameter]
            
            # Проверяем на наличие пропусков
            valid_indices = ~np.isnan(y)
            if np.sum(valid_indices) < 2:
                raise ValueError(f"Недостаточно данных для анализа тренда параметра '{parameter}'")
            
            x_valid = x[valid_indices]
            y_valid = y[valid_indices]
            
            # Выполняем линейную регрессию
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
            
            # Рассчитываем значения тренда
            trend_values = intercept + slope * x_valid
            
            # Вычисляем абсолютное и относительное изменение
            absolute_change = slope * (x_valid.max() - x_valid.min())
            relative_change = absolute_change / np.mean(y_valid) * 100
            
            # Формируем результат
            result = {
                'parameter': parameter,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err,
                'absolute_change': absolute_change,
                'relative_change': relative_change,
                'trend_significance': p_value < 0.05,
                'x_type': x_type,
                'x_label': x_label,
                'sample_size': len(y_valid),
                'period': f"{filtered_stats['start_date'].min()} - {filtered_stats['start_date'].max()}" 
                          if 'start_date' in filtered_stats.columns else "Unknown"
            }
            
            logger.info(f"Выполнен анализ тренда для параметра '{parameter}': "
                      f"slope={slope:.6f}, p_value={p_value:.4f}, "
                      f"абсолютное изменение={absolute_change:.2f}, "
                      f"относительное изменение={relative_change:.2f}%")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка при анализе тренда: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)


def calculate_basic_statistics(cyclones: List[Cyclone], min_latitude: float = 70.0) -> Dict[str, Any]:
    """
    Рассчитывает базовую статистику для набора циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        min_latitude: Минимальная широта для анализа (градусы с.ш.).
        
    Возвращает:
        Словарь с базовой статистикой.
    """
    analyzer = CycloneStatistics(min_latitude=min_latitude)
    return analyzer.calculate_basic_stats(cyclones)


def calculate_spatial_statistics(cyclones: List[Cyclone], 
                              grid_resolution: float = 2.0,
                              min_latitude: float = 70.0,
                              region: Optional[Dict[str, float]] = None) -> xr.Dataset:
    """
    Рассчитывает пространственную статистику циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        grid_resolution: Разрешение сетки в градусах.
        min_latitude: Минимальная широта для анализа (градусы с.ш.).
        region: Область для анализа в формате {'north': north, 'south': south, 'east': east, 'west': west}.
              Если None, используется вся Арктика (севернее min_latitude).
        
    Возвращает:
        Набор данных xarray с пространственной статистикой.
    """
    analyzer = CycloneStatistics(min_latitude=min_latitude)
    return analyzer.calculate_spatial_stats(cyclones, grid_resolution, region)


def calculate_temporal_statistics(cyclones: List[Cyclone],
                               temporal_resolution: str = 'monthly',
                               min_latitude: float = 70.0,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Рассчитывает временную статистику циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        temporal_resolution: Временное разрешение анализа ('daily', 'monthly', 'seasonal', 'annual').
        min_latitude: Минимальная широта для анализа (градусы с.ш.).
        start_date: Начальная дата анализа. Если None, используется минимальная дата в данных.
        end_date: Конечная дата анализа. Если None, используется максимальная дата в данных.
        
    Возвращает:
        DataFrame с временной статистикой.
    """
    analyzer = CycloneStatistics(min_latitude=min_latitude)
    return analyzer.calculate_temporal_stats(cyclones, temporal_resolution, start_date, end_date)


def calculate_ensemble_statistics(cyclones_list: List[List[Cyclone]],
                              max_distance: float = 500.0,
                              max_time_diff: float = 12.0,
                              min_latitude: float = 70.0) -> Dict[str, Any]:
    """
    Рассчитывает статистику ансамбля циклонов.
    
    Аргументы:
        cyclones_list: Список наборов циклонов (ансамбль прогнозов или наблюдений).
        max_distance: Максимальное расстояние (км) для сопоставления циклонов.
        max_time_diff: Максимальная разница во времени (часы) для сопоставления.
        min_latitude: Минимальная широта для анализа (градусы с.ш.).
        
    Возвращает:
        Словарь со статистикой ансамбля.
    """
    analyzer = CycloneStatistics(min_latitude=min_latitude)
    return analyzer.calculate_ensemble_stats(cyclones_list, max_distance, max_time_diff)


def estimate_uncertainties(cyclones: List[Cyclone], 
                        bootstrap_samples: int = 1000,
                        confidence_level: float = 0.95,
                        min_latitude: float = 70.0) -> Dict[str, Any]:
    """
    Оценивает неопределенности в статистических характеристиках циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        bootstrap_samples: Количество бутстрап-выборок.
        confidence_level: Уровень доверия для интервалов.
        min_latitude: Минимальная широта для анализа (градусы с.ш.).
        
    Возвращает:
        Словарь с оценками неопределенностей.
    """
    analyzer = CycloneStatistics(min_latitude=min_latitude)
    return analyzer.estimate_uncertainties(cyclones, bootstrap_samples, confidence_level)


def calculate_cyclone_flux(cyclone_tracks: List[List[Cyclone]],
                        grid_resolution: float = 2.0,
                        min_latitude: float = 70.0) -> xr.Dataset:
    """
    Рассчитывает поток циклонов через границы ячеек сетки.
    
    Аргументы:
        cyclone_tracks: Список треков циклонов (каждый трек - список циклонов).
        grid_resolution: Разрешение сетки в градусах.
        min_latitude: Минимальная широта для анализа (градусы с.ш.).
        
    Возвращает:
        Набор данных xarray с потоками циклонов.
    """
    analyzer = CycloneStatistics(min_latitude=min_latitude)
    return analyzer.calculate_cyclone_flux(cyclone_tracks, grid_resolution)


def perform_trend_analysis(cyclone_stats: pd.DataFrame,
                        parameter: str = 'count',
                        period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
    """
    Выполняет анализ трендов в характеристиках циклонов.
    
    Аргументы:
        cyclone_stats: DataFrame с временной статистикой циклонов.
        parameter: Параметр для анализа тренда.
        period: Период для анализа в формате (начало, конец).
               Если None, используется весь доступный период.
        
    Возвращает:
        Словарь с результатами анализа тренда.
    """
    analyzer = CycloneStatistics()
    return analyzer.perform_trend_analysis(cyclone_stats, parameter, period)