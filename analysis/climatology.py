"""
Модуль климатологического анализа для системы ArcticCyclone.

Предоставляет функции и классы для анализа долговременных паттернов
и климатологических характеристик арктических циклонов.
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
import matplotlib.pyplot as plt

from models.cyclone import Cyclone
from models.classifications import CycloneType, CycloneIntensity
from core.exceptions import ArcticCycloneError

# Инициализация логгера
logger = logging.getLogger(__name__)


class ClimateAnalyzer:
    """
    Класс для климатологического анализа арктических циклонов.
    
    Предоставляет методы для исследования долговременных характеристик
    и климатологических паттернов циклонической активности в Арктике.
    """
    
    def __init__(self, reference_period: Optional[Tuple[int, int]] = None, 
                grid_resolution: float = 2.0):
        """
        Инициализирует анализатор климатологических данных.
        
        Аргументы:
            reference_period: Базовый период для климатологии в формате (начальный_год, конечный_год).
                            Если None, используются все доступные данные.
            grid_resolution: Пространственное разрешение сетки в градусах.
        """
        self.reference_period = reference_period
        self.grid_resolution = grid_resolution
        self.climatology = None
        
        logger.debug(f"Инициализирован ClimateAnalyzer с разрешением сетки {grid_resolution}° "
                    f"и опорным периодом {reference_period}")
    
    def create_climatology(self, cyclones: List[Cyclone], 
                         temporal_resolution: str = 'monthly') -> xr.Dataset:
        """
        Создает климатологию циклонов на основе списка циклонов.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            temporal_resolution: Временное разрешение климатологии ('daily', 'monthly', 'seasonal', 'annual').
            
        Возвращает:
            Набор данных xarray с климатологической информацией.
        """
        try:
            logger.info(f"Создание климатологии с временным разрешением '{temporal_resolution}'")
            
            # Фильтруем циклоны по базовому периоду, если он указан
            if self.reference_period is not None:
                start_year, end_year = self.reference_period
                cyclones = [c for c in cyclones if start_year <= c.time.year <= end_year]
                
                if not cyclones:
                    raise ValueError(f"Нет циклонов в указанном базовом периоде {self.reference_period}")
            
            # Создаем пространственную сетку
            lon_bins = np.arange(-180, 180 + self.grid_resolution, self.grid_resolution)
            lat_bins = np.arange(60, 90 + self.grid_resolution, self.grid_resolution)
            
            # Создаем временную сетку в зависимости от разрешения
            if temporal_resolution == 'daily':
                # 366 дней (включая 29 февраля)
                time_bins = np.arange(1, 367)
                time_dim = 'day_of_year'
            elif temporal_resolution == 'monthly':
                # 12 месяцев
                time_bins = np.arange(1, 13)
                time_dim = 'month'
            elif temporal_resolution == 'seasonal':
                # 4 сезона (ДЯФ, МАМ, ИИА, СОН)
                time_bins = np.arange(1, 5)
                time_dim = 'season'
            elif temporal_resolution == 'annual':
                # Годовое среднее
                time_bins = np.array([1])
                time_dim = 'annual'
            else:
                raise ValueError(f"Неподдерживаемое временное разрешение: {temporal_resolution}")
            
            # Инициализируем массивы для подсчета
            count_grid = np.zeros((len(time_bins), len(lat_bins)-1, len(lon_bins)-1))
            intensity_grid = np.zeros_like(count_grid)
            
            # Заполняем гриды данными о циклонах
            for cyclone in cyclones:
                # Определяем индекс временного бина
                if temporal_resolution == 'daily':
                    # День года (1-366)
                    time_idx = cyclone.time.timetuple().tm_yday - 1
                elif temporal_resolution == 'monthly':
                    # Месяц (1-12)
                    time_idx = cyclone.time.month - 1
                elif temporal_resolution == 'seasonal':
                    # Сезон (1-4)
                    month = cyclone.time.month
                    if month in [12, 1, 2]:
                        time_idx = 0  # Зима (ДЯФ)
                    elif month in [3, 4, 5]:
                        time_idx = 1  # Весна (МАМ)
                    elif month in [6, 7, 8]:
                        time_idx = 2  # Лето (ИИА)
                    else:
                        time_idx = 3  # Осень (СОН)
                else:
                    # Годовое среднее
                    time_idx = 0
                
                # Определяем пространственные индексы
                lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
                lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
                
                # Проверяем, что индексы в пределах сетки
                if (0 <= time_idx < len(time_bins) and 
                    0 <= lat_idx < len(lat_bins)-1 and 
                    0 <= lon_idx < len(lon_bins)-1):
                    # Увеличиваем счетчик
                    count_grid[time_idx, lat_idx, lon_idx] += 1
                    
                    # Добавляем информацию об интенсивности (инверсия давления)
                    intensity_grid[time_idx, lat_idx, lon_idx] += (1020 - cyclone.central_pressure)
            
            # Считаем средние значения
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_intensity = np.where(count_grid > 0, intensity_grid / count_grid, np.nan)
            
            # Создаем набор данных xarray
            if temporal_resolution == 'daily':
                time_coord = pd.date_range(start='2000-01-01', periods=366, freq='D')
                time_values = np.arange(1, 367)
            elif temporal_resolution == 'monthly':
                time_coord = [f"{calendar.month_name[i]}" for i in range(1, 13)]
                time_values = np.arange(1, 13)
            elif temporal_resolution == 'seasonal':
                time_coord = ['DJF', 'MAM', 'JJA', 'SON']
                time_values = np.arange(1, 5)
            else:
                time_coord = ['Annual']
                time_values = np.array([1])
            
            # Средние координаты ячеек сетки
            lon_centers = lon_bins[:-1] + self.grid_resolution / 2
            lat_centers = lat_bins[:-1] + self.grid_resolution / 2
            
            # Создаем набор данных
            ds = xr.Dataset(
                data_vars={
                    'cyclone_count': (
                        [time_dim, 'latitude', 'longitude'], 
                        count_grid
                    ),
                    'mean_intensity': (
                        [time_dim, 'latitude', 'longitude'], 
                        mean_intensity
                    )
                },
                coords={
                    time_dim: time_values,
                    'time_labels': ([time_dim], time_coord),
                    'latitude': lat_centers,
                    'longitude': lon_centers
                },
                attrs={
                    'description': f'Cyclone climatology with {temporal_resolution} resolution',
                    'reference_period': str(self.reference_period) if self.reference_period else 'all data',
                    'grid_resolution': f'{self.grid_resolution} degrees',
                    'creation_date': datetime.now().isoformat(),
                    'total_cyclones': len(cyclones)
                }
            )
            
            # Сохраняем климатологию
            self.climatology = ds
            
            logger.info(f"Создана климатология с разрешением {temporal_resolution}")
            
            return ds
            
        except Exception as e:
            error_msg = f"Ошибка при создании климатологии: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def calculate_anomalies(self, cyclones: List[Cyclone], 
                          period: Tuple[datetime, datetime],
                          temporal_resolution: str = 'monthly') -> xr.Dataset:
        """
        Рассчитывает аномалии циклонической активности относительно климатологии.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            period: Период для расчета аномалий (начало, конец).
            temporal_resolution: Временное разрешение ('monthly', 'seasonal', 'annual').
            
        Возвращает:
            Набор данных xarray с аномалиями.
        """
        try:
            # Проверяем наличие климатологии
            if self.climatology is None:
                logger.info("Климатология не найдена, создание новой климатологии")
                self.create_climatology(cyclones, temporal_resolution)
            
            # Фильтруем циклоны по указанному периоду
            start_date, end_date = period
            period_cyclones = [c for c in cyclones if start_date <= c.time <= end_date]
            
            if not period_cyclones:
                raise ValueError(f"Нет циклонов в указанном периоде {period}")
            
            # Создаем климатологию для указанного периода
            period_climatology = ClimateAnalyzer(
                grid_resolution=self.grid_resolution
            ).create_climatology(period_cyclones, temporal_resolution)
            
            # Рассчитываем аномалии
            anomalies = period_climatology['cyclone_count'] - self.climatology['cyclone_count']
            relative_anomalies = (
                (period_climatology['cyclone_count'] - self.climatology['cyclone_count']) / 
                self.climatology['cyclone_count']
            ) * 100  # в процентах
            
            # Создаем набор данных с аномалиями
            ds = xr.Dataset(
                data_vars={
                    'anomaly': anomalies,
                    'relative_anomaly': relative_anomalies
                },
                coords=period_climatology.coords,
                attrs={
                    'description': f'Cyclone anomalies with {temporal_resolution} resolution',
                    'reference_period': str(self.reference_period) if self.reference_period else 'all data',
                    'analysis_period': f'{start_date.isoformat()} to {end_date.isoformat()}',
                    'grid_resolution': f'{self.grid_resolution} degrees',
                    'creation_date': datetime.now().isoformat(),
                    'total_cyclones_in_period': len(period_cyclones)
                }
            )
            
            logger.info(f"Рассчитаны аномалии для периода {start_date} - {end_date}")
            
            return ds
            
        except Exception as e:
            error_msg = f"Ошибка при расчете аномалий: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def analyze_trends(self, cyclones: List[Cyclone], 
                     start_year: int, end_year: int,
                     parameter: str = 'frequency') -> Dict[str, Any]:
        """
        Анализирует тренды в параметрах циклонов за указанный период.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            start_year: Начальный год анализа.
            end_year: Конечный год анализа.
            parameter: Параметр для анализа ('frequency', 'intensity', 'size', 'duration').
            
        Возвращает:
            Словарь с результатами анализа трендов.
        """
        try:
            # Фильтруем циклоны по годам
            period_cyclones = [c for c in cyclones if start_year <= c.time.year <= end_year]
            
            if not period_cyclones:
                raise ValueError(f"Нет циклонов в указанном периоде {start_year}-{end_year}")
            
            # Инициализируем результаты
            results = {
                'parameter': parameter,
                'period': f"{start_year}-{end_year}",
                'years': list(range(start_year, end_year + 1)),
                'annual_values': [],
                'trend': None,
                'trend_significance': None,
                'correlation': None,
                'p_value': None
            }
            
            # Анализируем данные для каждого года
            for year in range(start_year, end_year + 1):
                year_cyclones = [c for c in period_cyclones if c.time.year == year]
                
                if parameter == 'frequency':
                    # Количество циклонов в год
                    value = len(year_cyclones)
                elif parameter == 'intensity':
                    # Средняя интенсивность (инверсия давления)
                    if year_cyclones:
                        pressures = [c.central_pressure for c in year_cyclones]
                        value = np.mean(1020 - np.array(pressures))
                    else:
                        value = np.nan
                elif parameter == 'size':
                    # Средний радиус циклонов
                    if year_cyclones:
                        radii = [c.parameters.radius for c in year_cyclones 
                               if hasattr(c.parameters, 'radius') and c.parameters.radius is not None]
                        value = np.mean(radii) if radii else np.nan
                    else:
                        value = np.nan
                elif parameter == 'duration':
                    # Средняя продолжительность треков
                    # Группируем циклоны по трекам
                    tracks = {}
                    for c in year_cyclones:
                        if hasattr(c, 'track_id') and c.track_id:
                            if c.track_id not in tracks:
                                tracks[c.track_id] = []
                            tracks[c.track_id].append(c)
                    
                    # Считаем среднюю продолжительность
                    durations = []
                    for track_id, track_cyclones in tracks.items():
                        if len(track_cyclones) >= 2:
                            sorted_track = sorted(track_cyclones, key=lambda c: c.time)
                            duration = (sorted_track[-1].time - sorted_track[0].time).total_seconds() / 3600
                            durations.append(duration)
                    
                    value = np.mean(durations) if durations else np.nan
                else:
                    raise ValueError(f"Неподдерживаемый параметр для анализа трендов: {parameter}")
                
                results['annual_values'].append(value)
            
            # Рассчитываем тренд и статистическую значимость
            years = np.array(results['years'])
            values = np.array(results['annual_values'])
            
            # Исключаем NaN значения
            valid_indices = ~np.isnan(values)
            if np.sum(valid_indices) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    years[valid_indices], values[valid_indices]
                )
                
                results['trend'] = slope
                results['intercept'] = intercept
                results['correlation'] = r_value
                results['p_value'] = p_value
                results['std_error'] = std_err
                results['trend_significance'] = p_value < 0.05
                
                # Добавляем рассчитанные значения тренда
                results['trend_values'] = intercept + slope * years
            
            logger.info(f"Проанализированы тренды для параметра {parameter} "
                      f"за период {start_year}-{end_year}")
            
            return results
            
        except Exception as e:
            error_msg = f"Ошибка при анализе трендов: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def analyze_seasonal_patterns(self, cyclones: List[Cyclone],
                                reference_period: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Анализирует сезонные паттерны циклонической активности.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            reference_period: Базовый период для анализа. Если None, используется период из конструктора.
            
        Возвращает:
            Словарь с результатами анализа сезонных паттернов.
        """
        try:
            # Определяем базовый период
            if reference_period is None:
                reference_period = self.reference_period
            
            # Фильтруем циклоны по базовому периоду, если он указан
            if reference_period is not None:
                start_year, end_year = reference_period
                period_cyclones = [c for c in cyclones if start_year <= c.time.year <= end_year]
            else:
                period_cyclones = cyclones
            
            if not period_cyclones:
                raise ValueError("Нет циклонов для анализа сезонных паттернов")
            
            # Инициализируем результаты
            results = {
                'monthly_frequency': np.zeros(12),
                'seasonal_frequency': np.zeros(4),
                'monthly_intensity': np.zeros(12),
                'seasonal_intensity': np.zeros(4),
                'season_names': ['DJF', 'MAM', 'JJA', 'SON'],
                'month_names': [calendar.month_name[i] for i in range(1, 13)]
            }
            
            # Счетчики для расчета средних значений
            monthly_counts = np.zeros(12)
            seasonal_counts = np.zeros(4)
            
            # Анализируем данные по месяцам и сезонам
            for cyclone in period_cyclones:
                month = cyclone.time.month - 1  # 0-11
                
                # Определяем сезон
                if month in [11, 0, 1]:
                    season = 0  # Зима (ДЯФ)
                elif month in [2, 3, 4]:
                    season = 1  # Весна (МАМ)
                elif month in [5, 6, 7]:
                    season = 2  # Лето (ИИА)
                else:
                    season = 3  # Осень (СОН)
                
                # Увеличиваем счетчики
                monthly_counts[month] += 1
                seasonal_counts[season] += 1
                
                # Добавляем интенсивность (инверсия давления)
                results['monthly_intensity'][month] += (1020 - cyclone.central_pressure)
                results['seasonal_intensity'][season] += (1020 - cyclone.central_pressure)
            
            # Заполняем результаты частоты
            results['monthly_frequency'] = monthly_counts
            results['seasonal_frequency'] = seasonal_counts
            
            # Рассчитываем средние значения интенсивности
            with np.errstate(divide='ignore', invalid='ignore'):
                results['monthly_intensity'] = np.where(
                    monthly_counts > 0, 
                    results['monthly_intensity'] / monthly_counts, 
                    np.nan
                )
                
                results['seasonal_intensity'] = np.where(
                    seasonal_counts > 0,
                    results['seasonal_intensity'] / seasonal_counts,
                    np.nan
                )
            
            # Определяем сезон максимальной активности
            max_season_idx = np.argmax(results['seasonal_frequency'])
            max_month_idx = np.argmax(results['monthly_frequency'])
            
            results['max_activity_season'] = results['season_names'][max_season_idx]
            results['max_activity_month'] = results['month_names'][max_month_idx]
            
            # Определяем сезон максимальной интенсивности
            valid_seasonal_intensity = np.nan_to_num(results['seasonal_intensity'], nan=0)
            valid_monthly_intensity = np.nan_to_num(results['monthly_intensity'], nan=0)
            
            max_intensity_season_idx = np.argmax(valid_seasonal_intensity)
            max_intensity_month_idx = np.argmax(valid_monthly_intensity)
            
            results['max_intensity_season'] = results['season_names'][max_intensity_season_idx]
            results['max_intensity_month'] = results['month_names'][max_intensity_month_idx]
            
            # Добавляем информацию о количестве циклонов
            results['total_cyclones'] = len(period_cyclones)
            results['period'] = str(reference_period) if reference_period else 'all data'
            
            logger.info(f"Проанализированы сезонные паттерны для {len(period_cyclones)} циклонов")
            
            return results
            
        except Exception as e:
            error_msg = f"Ошибка при анализе сезонных паттернов: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def analyze_spatial_distribution(self, cyclones: List[Cyclone],
                                   region: Optional[Dict[str, float]] = None) -> xr.Dataset:
        """
        Анализирует пространственное распределение циклонической активности.
        
        Аргументы:
            cyclones: Список циклонов для анализа.
            region: Область для анализа в формате {'north': north, 'south': south, 'east': east, 'west': west}.
                  Если None, используется вся Арктика (севернее 60°N).
            
        Возвращает:
            Набор данных xarray с пространственным распределением.
        """
        try:
            # Определяем область анализа
            if region is None:
                region = {'north': 90.0, 'south': 60.0, 'east': 180.0, 'west': -180.0}
            
            # Фильтруем циклоны по региону
            region_cyclones = [
                c for c in cyclones
                if (region['south'] <= c.latitude <= region['north'] and
                    (region['west'] <= c.longitude <= region['east'] or
                     (region['west'] > region['east'] and  # Случай пересечения 180° долготы
                      (c.longitude >= region['west'] or c.longitude <= region['east']))))
            ]
            
            if not region_cyclones:
                raise ValueError(f"Нет циклонов в указанном регионе: {region}")
            
            # Создаем пространственную сетку
            lon_bins = np.arange(-180, 180 + self.grid_resolution, self.grid_resolution)
            lat_bins = np.arange(region['south'], region['north'] + self.grid_resolution, self.grid_resolution)
            
            # Инициализируем массивы для подсчета
            count_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
            genesis_grid = np.zeros_like(count_grid)
            lysis_grid = np.zeros_like(count_grid)
            pressure_sum = np.zeros_like(count_grid)
            vorticity_sum = np.zeros_like(count_grid)
            
            # Группируем циклоны по трекам для определения мест зарождения и распада
            tracks = {}
            for cyclone in region_cyclones:
                if hasattr(cyclone, 'track_id') and cyclone.track_id:
                    if cyclone.track_id not in tracks:
                        tracks[cyclone.track_id] = []
                    tracks[cyclone.track_id].append(cyclone)
            
            # Заполняем гриды данными о циклонах
            for cyclone in region_cyclones:
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
                    
                    # Суммируем завихренность для расчета среднего
                    if hasattr(cyclone.parameters, 'vorticity_850hPa'):
                        vorticity_sum[lat_idx, lon_idx] += cyclone.parameters.vorticity_850hPa
            
            # Определяем места зарождения и распада циклонов
            for track_id, track_cyclones in tracks.items():
                if len(track_cyclones) >= 2:
                    # Сортируем по времени
                    sorted_track = sorted(track_cyclones, key=lambda c: c.time)
                    
                    # Место зарождения (первая точка трека)
                    genesis = sorted_track[0]
                    lon_idx = np.searchsorted(lon_bins, genesis.longitude) - 1
                    lat_idx = np.searchsorted(lat_bins, genesis.latitude) - 1
                    
                    if (0 <= lat_idx < len(lat_bins)-1 and 
                        0 <= lon_idx < len(lon_bins)-1):
                        genesis_grid[lat_idx, lon_idx] += 1
                    
                    # Место распада (последняя точка трека)
                    lysis = sorted_track[-1]
                    lon_idx = np.searchsorted(lon_bins, lysis.longitude) - 1
                    lat_idx = np.searchsorted(lat_bins, lysis.latitude) - 1
                    
                    if (0 <= lat_idx < len(lat_bins)-1 and 
                        0 <= lon_idx < len(lon_bins)-1):
                        lysis_grid[lat_idx, lon_idx] += 1
            
            # Рассчитываем средние значения
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_pressure = np.where(count_grid > 0, pressure_sum / count_grid, np.nan)
                mean_vorticity = np.where(count_grid > 0, vorticity_sum / count_grid, np.nan)
            
            # Средние координаты ячеек сетки
            lon_centers = lon_bins[:-1] + self.grid_resolution / 2
            lat_centers = lat_bins[:-1] + self.grid_resolution / 2
            
            # Создаем набор данных
            ds = xr.Dataset(
                data_vars={
                    'cyclone_count': (
                        ['latitude', 'longitude'], 
                        count_grid
                    ),
                    'genesis_count': (
                        ['latitude', 'longitude'], 
                        genesis_grid
                    ),
                    'lysis_count': (
                        ['latitude', 'longitude'], 
                        lysis_grid
                    ),
                    'mean_pressure': (
                        ['latitude', 'longitude'], 
                        mean_pressure
                    ),
                    'mean_vorticity': (
                        ['latitude', 'longitude'], 
                        mean_vorticity
                    )
                },
                coords={
                    'latitude': lat_centers,
                    'longitude': lon_centers
                },
                attrs={
                    'description': 'Spatial distribution of cyclone activity',
                    'region': str(region),
                    'grid_resolution': f'{self.grid_resolution} degrees',
                    'creation_date': datetime.now().isoformat(),
                    'total_cyclones': len(region_cyclones),
                    'total_tracks': len(tracks)
                }
            )
            
            logger.info(f"Проанализировано пространственное распределение для {len(region_cyclones)} циклонов")
            
            return ds
            
        except Exception as e:
            error_msg = f"Ошибка при анализе пространственного распределения: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)


def calculate_monthly_frequencies(cyclones: List[Cyclone], 
                               start_year: Optional[int] = None, 
                               end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Рассчитывает ежемесячные частоты появления циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        start_year: Начальный год анализа. Если None, используется минимальный год в данных.
        end_year: Конечный год анализа. Если None, используется максимальный год в данных.
        
    Возвращает:
        DataFrame с ежемесячными частотами циклонов.
    """
    try:
        # Определяем временной диапазон
        if start_year is None or end_year is None:
            years = [c.time.year for c in cyclones]
            start_year = start_year or min(years)
            end_year = end_year or max(years)
        
        # Создаем диапазон месяцев
        months = pd.date_range(
            start=f"{start_year}-01-01",
            end=f"{end_year}-12-31",
            freq='MS'  # Месячный старт
        )
        
        # Инициализируем DataFrame
        df = pd.DataFrame(index=months, columns=['count'])
        df['count'] = 0
        
        # Считаем циклоны по месяцам
        for cyclone in cyclones:
            # Округляем дату до начала месяца
            month_start = cyclone.time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            if month_start in df.index:
                df.loc[month_start, 'count'] += 1
        
        # Добавляем информацию о месяце и годе
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['month_name'] = df.index.strftime('%B')
        
        # Добавляем сезон
        df['season'] = 'DJF'  # По умолчанию - зима
        df.loc[df['month'].isin([3, 4, 5]), 'season'] = 'MAM'  # Весна
        df.loc[df['month'].isin([6, 7, 8]), 'season'] = 'JJA'  # Лето
        df.loc[df['month'].isin([9, 10, 11]), 'season'] = 'SON'  # Осень
        
        logger.info(f"Рассчитаны ежемесячные частоты для периода {start_year}-{end_year}")
        
        return df
        
    except Exception as e:
        error_msg = f"Ошибка при расчете ежемесячных частот: {str(e)}"
        logger.error(error_msg)
        raise ArcticCycloneError(error_msg)


def calculate_annual_cycle(cyclones: List[Cyclone], 
                         parameter: str = 'frequency',
                         normalize: bool = False) -> Dict[str, np.ndarray]:
    """
    Рассчитывает годовой цикл параметра циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        parameter: Параметр для анализа ('frequency', 'intensity', 'size', 'duration').
        normalize: Нормализовать ли значения (0-1).
        
    Возвращает:
        Словарь с месячными значениями параметра.
    """
    try:
        # Инициализируем массивы для подсчета
        monthly_values = np.zeros(12)
        monthly_counts = np.zeros(12)
        
        # Для расчета продолжительности трека
        tracks = {}
        
        # Считаем параметры по месяцам
        for cyclone in cyclones:
            month = cyclone.time.month - 1  # 0-11
            
            if parameter == 'frequency':
                # Простой подсчет
                monthly_values[month] += 1
            elif parameter == 'intensity':
                # Интенсивность (инверсия давления)
                monthly_values[month] += (1020 - cyclone.central_pressure)
                monthly_counts[month] += 1
            elif parameter == 'size':
                # Размер циклона
                if hasattr(cyclone.parameters, 'radius') and cyclone.parameters.radius is not None:
                    monthly_values[month] += cyclone.parameters.radius
                    monthly_counts[month] += 1
            elif parameter == 'duration':
                # Для продолжительности треков
                if hasattr(cyclone, 'track_id') and cyclone.track_id:
                    if cyclone.track_id not in tracks:
                        tracks[cyclone.track_id] = []
                    tracks[cyclone.track_id].append(cyclone)
            else:
                raise ValueError(f"Неподдерживаемый параметр: {parameter}")
        
        # Расчет продолжительности для треков
        if parameter == 'duration':
            track_durations = {}
            
            for track_id, track_cyclones in tracks.items():
                if len(track_cyclones) >= 2:
                    # Сортируем по времени
                    sorted_track = sorted(track_cyclones, key=lambda c: c.time)
                    
                    # Рассчитываем продолжительность
                    duration = (sorted_track[-1].time - sorted_track[0].time).total_seconds() / 3600
                    
                    # Определяем месяц начала трека
                    start_month = sorted_track[0].time.month - 1  # 0-11
                    
                    # Добавляем продолжительность к соответствующему месяцу
                    monthly_values[start_month] += duration
                    monthly_counts[start_month] += 1
        
        # Рассчитываем средние значения для параметров, где это необходимо
        if parameter in ['intensity', 'size', 'duration']:
            with np.errstate(divide='ignore', invalid='ignore'):
                monthly_values = np.where(
                    monthly_counts > 0, 
                    monthly_values / monthly_counts, 
                    np.nan
                )
        
        # Нормализуем значения, если требуется
        if normalize and not np.all(np.isnan(monthly_values)):
            valid_values = monthly_values[~np.isnan(monthly_values)]
            if len(valid_values) > 0:
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                if max_val > min_val:
                    monthly_values = (monthly_values - min_val) / (max_val - min_val)
        
        # Создаем результат
        result = {
            'parameter': parameter,
            'monthly_values': monthly_values,
            'month_names': [calendar.month_name[i] for i in range(1, 13)],
            'data_points': int(np.sum(monthly_counts)) if parameter in ['intensity', 'size', 'duration'] else len(cyclones)
        }
        
        logger.info(f"Рассчитан годовой цикл для параметра {parameter}")
        
        return result
        
    except Exception as e:
        error_msg = f"Ошибка при расчете годового цикла: {str(e)}"
        logger.error(error_msg)
        raise ArcticCycloneError(error_msg)


def calculate_interannual_variability(cyclones: List[Cyclone], 
                                    parameter: str = 'frequency',
                                    start_year: Optional[int] = None, 
                                    end_year: Optional[int] = None) -> pd.DataFrame:
    """
    Рассчитывает межгодовую изменчивость параметра циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        parameter: Параметр для анализа ('frequency', 'intensity', 'size', 'duration').
        start_year: Начальный год анализа. Если None, используется минимальный год в данных.
        end_year: Конечный год анализа. Если None, используется максимальный год в данных.
        
    Возвращает:
        DataFrame с ежегодными значениями параметра.
    """
    try:
        # Определяем временной диапазон
        years = [c.time.year for c in cyclones]
        start_year = start_year or min(years)
        end_year = end_year or max(years)
        
        # Создаем диапазон лет
        year_range = range(start_year, end_year + 1)
        
        # Инициализируем DataFrame
        df = pd.DataFrame(index=year_range, columns=['value'])
        df['value'] = 0
        
        # Добавляем счетчик для расчета средних значений
        df['count'] = 0
        
        # Для расчета продолжительности трека
        if parameter == 'duration':
            tracks = {}
            
            for cyclone in cyclones:
                if cyclone.time.year in year_range:
                    if hasattr(cyclone, 'track_id') and cyclone.track_id:
                        # Группируем циклоны по треку и году
                        track_key = (cyclone.track_id, cyclone.time.year)
                        if track_key not in tracks:
                            tracks[track_key] = []
                        tracks[track_key].append(cyclone)
            
            # Рассчитываем продолжительность для каждого трека
            for (track_id, year), track_cyclones in tracks.items():
                if len(track_cyclones) >= 2:
                    # Сортируем по времени
                    sorted_track = sorted(track_cyclones, key=lambda c: c.time)
                    
                    # Рассчитываем продолжительность
                    duration = (sorted_track[-1].time - sorted_track[0].time).total_seconds() / 3600
                    
                    # Добавляем продолжительность к соответствующему году
                    df.loc[year, 'value'] += duration
                    df.loc[year, 'count'] += 1
        else:
            # Для других параметров
            for cyclone in cyclones:
                year = cyclone.time.year
                
                if year in year_range:
                    if parameter == 'frequency':
                        # Простой подсчет
                        df.loc[year, 'value'] += 1
                    elif parameter == 'intensity':
                        # Интенсивность (инверсия давления)
                        df.loc[year, 'value'] += (1020 - cyclone.central_pressure)
                        df.loc[year, 'count'] += 1
                    elif parameter == 'size':
                        # Размер циклона
                        if hasattr(cyclone.parameters, 'radius') and cyclone.parameters.radius is not None:
                            df.loc[year, 'value'] += cyclone.parameters.radius
                            df.loc[year, 'count'] += 1
        
        # Рассчитываем средние значения для параметров, где это необходимо
        if parameter in ['intensity', 'size', 'duration']:
            df['mean_value'] = df['value'] / df['count']
            df.loc[df['count'] == 0, 'mean_value'] = np.nan
            df = df.drop(columns=['value', 'count'])
            df = df.rename(columns={'mean_value': 'value'})
        
        # Добавляем параметр
        df['parameter'] = parameter
        
        # Рассчитываем статистики
        df['anomaly'] = df['value'] - df['value'].mean()
        df['std_anomaly'] = df['anomaly'] / df['value'].std()
        
        logger.info(f"Рассчитана межгодовая изменчивость для параметра {parameter}")
        
        return df
        
    except Exception as e:
        error_msg = f"Ошибка при расчете межгодовой изменчивости: {str(e)}"
        logger.error(error_msg)
        raise ArcticCycloneError(error_msg)


def create_climatology(cyclones: List[Cyclone], 
                     grid_resolution: float = 2.0,
                     temporal_resolution: str = 'monthly',
                     reference_period: Optional[Tuple[int, int]] = None) -> xr.Dataset:
    """
    Создает климатологию циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        grid_resolution: Разрешение сетки в градусах.
        temporal_resolution: Временное разрешение климатологии ('daily', 'monthly', 'seasonal', 'annual').
        reference_period: Базовый период для климатологии в формате (начальный_год, конечный_год).
            
    Возвращает:
        Набор данных xarray с климатологической информацией.
    """
    analyzer = ClimateAnalyzer(reference_period=reference_period, grid_resolution=grid_resolution)
    return analyzer.create_climatology(cyclones, temporal_resolution)


def analyze_seasonal_patterns(cyclones: List[Cyclone], 
                           reference_period: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """
    Анализирует сезонные паттерны циклонической активности.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        reference_period: Базовый период для анализа в формате (начальный_год, конечный_год).
            
    Возвращает:
        Словарь с результатами анализа сезонных паттернов.
    """
    analyzer = ClimateAnalyzer(reference_period=reference_period)
    return analyzer.analyze_seasonal_patterns(cyclones, reference_period)


def analyze_climate_trends(cyclones: List[Cyclone], 
                         start_year: int, end_year: int, 
                         parameters: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Анализирует климатические тренды в параметрах циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        start_year: Начальный год анализа.
        end_year: Конечный год анализа.
        parameters: Список параметров для анализа. Если None, анализируются все основные параметры.
            
    Возвращает:
        Словарь с результатами анализа трендов для каждого параметра.
    """
    if parameters is None:
        parameters = ['frequency', 'intensity', 'size', 'duration']
    
    analyzer = ClimateAnalyzer()
    results = {}
    
    for param in parameters:
        results[param] = analyzer.analyze_trends(cyclones, start_year, end_year, param)
    
    return results