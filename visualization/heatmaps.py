"""
Модуль визуализации пространственного распределения для системы ArcticCyclone.

Предоставляет функции для создания тепловых карт и пространственных
распределений арктических циклонов и их параметров.
"""

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import xarray as xr
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import logging
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from datetime import datetime, timedelta

from models.cyclone import Cyclone
from models.classifications import CycloneType, CycloneIntensity
from core.exceptions import VisualizationError
from .mappers import create_arctic_map, save_figure

# Инициализация логгера
logger = logging.getLogger(__name__)


def create_cyclone_frequency_map(cyclones: List[Cyclone], 
                              grid_resolution: float = 1.0,
                              smoothing_sigma: float = 1.0,
                              min_latitude: float = 60.0,
                              cmap: str = 'YlOrRd',
                              figsize: Tuple[float, float] = (10, 8),
                              projection: str = 'NorthPolarStereo') -> Tuple[Figure, Axes]:
    """
    Создает карту частоты появления циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        grid_resolution: Разрешение сетки в градусах.
        smoothing_sigma: Параметр сглаживания для гауссового фильтра.
        min_latitude: Минимальная широта для визуализации.
        cmap: Цветовая карта для отображения частоты.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип проекции карты.
        
    Возвращает:
        Кортеж (figure, axis) с созданной картой.
    """
    try:
        # Создаем базовую карту
        fig, ax = create_arctic_map(
            min_latitude=min_latitude, 
            figsize=figsize,
            projection=projection
        )
        
        # Создаем сетку для подсчета циклонов
        lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
        lat_bins = np.arange(min_latitude, 90 + grid_resolution, grid_resolution)
        
        # Инициализируем массив для подсчета циклонов
        grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        
        # Считаем количество циклонов в каждой ячейке сетки
        for cyclone in cyclones:
            lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
            lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
            
            # Проверяем, что индексы в пределах сетки
            if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
                grid[lat_idx, lon_idx] += 1
        
        # Применяем сглаживание
        if smoothing_sigma > 0:
            grid = gaussian_filter(grid, sigma=smoothing_sigma)
        
        # Создаем сетку координат для отображения
        lon_grid, lat_grid = np.meshgrid(
            lon_bins[:-1] + grid_resolution/2,
            lat_bins[:-1] + grid_resolution/2
        )
        
        # Отображаем тепловую карту
        contour = ax.contourf(
            lon_grid, lat_grid, grid, 
            transform=ccrs.PlateCarree(),
            cmap=cmap, 
            levels=20,
            extend='max'
        )
        
        # Добавляем цветовую шкалу
        cbar = fig.colorbar(contour, ax=ax, pad=0.05)
        cbar.set_label('Количество циклонов')
        
        # Устанавливаем заголовок
        ax.set_title('Частота появления арктических циклонов')
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при создании карты частоты циклонов: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def create_genesis_density_map(cyclones: List[Cyclone], 
                            grid_resolution: float = 1.0,
                            smoothing_sigma: float = 1.0,
                            min_latitude: float = 60.0,
                            cmap: str = 'viridis',
                            figsize: Tuple[float, float] = (10, 8),
                            projection: str = 'NorthPolarStereo') -> Tuple[Figure, Axes]:
    """
    Создает карту плотности зарождения циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        grid_resolution: Разрешение сетки в градусах.
        smoothing_sigma: Параметр сглаживания для гауссового фильтра.
        min_latitude: Минимальная широта для визуализации.
        cmap: Цветовая карта для отображения плотности.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип проекции карты.
        
    Возвращает:
        Кортеж (figure, axis) с созданной картой.
    """
    try:
        # Создаем базовую карту
        fig, ax = create_arctic_map(
            min_latitude=min_latitude, 
            figsize=figsize,
            projection=projection
        )
        
        # Создаем сетку для подсчета циклонов
        lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
        lat_bins = np.arange(min_latitude, 90 + grid_resolution, grid_resolution)
        
        # Инициализируем массив для подсчета мест зарождения
        grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        
        # Группируем циклоны по трекам
        tracks = {}
        for cyclone in cyclones:
            if hasattr(cyclone, 'track_id') and cyclone.track_id:
                if cyclone.track_id not in tracks:
                    tracks[cyclone.track_id] = []
                tracks[cyclone.track_id].append(cyclone)
        
        # Находим места зарождения (первая точка каждого трека)
        for track_id, track_cyclones in tracks.items():
            # Сортируем по времени
            track_cyclones.sort(key=lambda c: c.time)
            
            # Берем первую точку трека
            if track_cyclones:
                genesis = track_cyclones[0]
                lon_idx = np.searchsorted(lon_bins, genesis.longitude) - 1
                lat_idx = np.searchsorted(lat_bins, genesis.latitude) - 1
                
                # Проверяем, что индексы в пределах сетки
                if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
                    grid[lat_idx, lon_idx] += 1
        
        # Применяем сглаживание
        if smoothing_sigma > 0:
            grid = gaussian_filter(grid, sigma=smoothing_sigma)
        
        # Создаем сетку координат для отображения
        lon_grid, lat_grid = np.meshgrid(
            lon_bins[:-1] + grid_resolution/2,
            lat_bins[:-1] + grid_resolution/2
        )
        
        # Отображаем тепловую карту
        contour = ax.contourf(
            lon_grid, lat_grid, grid, 
            transform=ccrs.PlateCarree(),
            cmap=cmap, 
            levels=20,
            extend='max'
        )
        
        # Добавляем цветовую шкалу
        cbar = fig.colorbar(contour, ax=ax, pad=0.05)
        cbar.set_label('Количество зарождений')
        
        # Устанавливаем заголовок
        ax.set_title('Плотность зарождения арктических циклонов')
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при создании карты плотности зарождения: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def create_track_density_map(cyclones: List[Cyclone], 
                          grid_resolution: float = 1.0,
                          smoothing_sigma: float = 1.0,
                          min_latitude: float = 60.0,
                          cmap: str = 'Blues',
                          figsize: Tuple[float, float] = (10, 8),
                          projection: str = 'NorthPolarStereo') -> Tuple[Figure, Axes]:
    """
    Создает карту плотности треков циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        grid_resolution: Разрешение сетки в градусах.
        smoothing_sigma: Параметр сглаживания для гауссового фильтра.
        min_latitude: Минимальная широта для визуализации.
        cmap: Цветовая карта для отображения плотности.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип проекции карты.
        
    Возвращает:
        Кортеж (figure, axis) с созданной картой.
    """
    try:
        # Создаем базовую карту
        fig, ax = create_arctic_map(
            min_latitude=min_latitude, 
            figsize=figsize,
            projection=projection
        )
        
        # Создаем сетку для подсчета циклонов
        lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
        lat_bins = np.arange(min_latitude, 90 + grid_resolution, grid_resolution)
        
        # Инициализируем массив для подсчета треков
        grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        
        # Группируем циклоны по трекам
        tracks = {}
        for cyclone in cyclones:
            if hasattr(cyclone, 'track_id') and cyclone.track_id:
                if cyclone.track_id not in tracks:
                    tracks[cyclone.track_id] = []
                tracks[cyclone.track_id].append(cyclone)
        
        # Проходим по каждому треку и увеличиваем значение в ячейках сетки
        for track_id, track_cyclones in tracks.items():
            # Сортируем по времени
            track_cyclones.sort(key=lambda c: c.time)
            
            # Создаем множество уникальных ячеек для этого трека
            track_cells = set()
            
            for cyclone in track_cyclones:
                lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
                lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
                
                # Проверяем, что индексы в пределах сетки
                if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
                    track_cells.add((lat_idx, lon_idx))
            
            # Увеличиваем счетчик для каждой уникальной ячейки
            for lat_idx, lon_idx in track_cells:
                grid[lat_idx, lon_idx] += 1
        
        # Применяем сглаживание
        if smoothing_sigma > 0:
            grid = gaussian_filter(grid, sigma=smoothing_sigma)
        
        # Создаем сетку координат для отображения
        lon_grid, lat_grid = np.meshgrid(
            lon_bins[:-1] + grid_resolution/2,
            lat_bins[:-1] + grid_resolution/2
        )
        
        # Отображаем тепловую карту
        contour = ax.contourf(
            lon_grid, lat_grid, grid, 
            transform=ccrs.PlateCarree(),
            cmap=cmap, 
            levels=20,
            extend='max'
        )
        
        # Добавляем цветовую шкалу
        cbar = fig.colorbar(contour, ax=ax, pad=0.05)
        cbar.set_label('Количество треков')
        
        # Устанавливаем заголовок
        ax.set_title('Плотность треков арктических циклонов')
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при создании карты плотности треков: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def create_pressure_intensity_map(cyclones: List[Cyclone], 
                               grid_resolution: float = 1.0,
                               smoothing_sigma: float = 1.0,
                               min_latitude: float = 60.0,
                               cmap: str = 'coolwarm_r',
                               figsize: Tuple[float, float] = (10, 8),
                               projection: str = 'NorthPolarStereo') -> Tuple[Figure, Axes]:
    """
    Создает карту интенсивности циклонов по давлению.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        grid_resolution: Разрешение сетки в градусах.
        smoothing_sigma: Параметр сглаживания для гауссового фильтра.
        min_latitude: Минимальная широта для визуализации.
        cmap: Цветовая карта для отображения интенсивности.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип проекции карты.
        
    Возвращает:
        Кортеж (figure, axis) с созданной картой.
    """
    try:
        # Создаем базовую карту
        fig, ax = create_arctic_map(
            min_latitude=min_latitude, 
            figsize=figsize,
            projection=projection
        )
        
        # Создаем сетку для суммирования давления и счетчик
        lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
        lat_bins = np.arange(min_latitude, 90 + grid_resolution, grid_resolution)
        
        # Инициализируем массивы для суммирования давления и счетчика циклонов
        pressure_sum = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        count = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        
        # Суммируем давление по ячейкам сетки
        for cyclone in cyclones:
            lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
            lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
            
            # Проверяем, что индексы в пределах сетки
            if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
                pressure_sum[lat_idx, lon_idx] += cyclone.central_pressure
                count[lat_idx, lon_idx] += 1
        
        # Рассчитываем среднее давление (избегаем деления на ноль)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_pressure = np.where(count > 0, pressure_sum / count, np.nan)
        
        # Применяем сглаживание (только для непустых ячеек)
        if smoothing_sigma > 0:
            # Создаем маску для непустых ячеек
            valid_mask = ~np.isnan(mean_pressure)
            
            # Сглаживаем только непустые ячейки
            smoothed_data = np.copy(mean_pressure)
            smoothed_data[valid_mask] = gaussian_filter(
                mean_pressure[valid_mask], 
                sigma=smoothing_sigma
            )
            mean_pressure = smoothed_data
        
        # Создаем сетку координат для отображения
        lon_grid, lat_grid = np.meshgrid(
            lon_bins[:-1] + grid_resolution/2,
            lat_bins[:-1] + grid_resolution/2
        )
        
        # Отображаем тепловую карту
        contour = ax.contourf(
            lon_grid, lat_grid, mean_pressure, 
            transform=ccrs.PlateCarree(),
            cmap=cmap, 
            levels=np.linspace(950, 1020, 21),
            extend='both'
        )
        
        # Добавляем цветовую шкалу
        cbar = fig.colorbar(contour, ax=ax, pad=0.05)
        cbar.set_label('Среднее давление (гПа)')
        
        # Устанавливаем заголовок
        ax.set_title('Интенсивность арктических циклонов по давлению')
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при создании карты интенсивности циклонов: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def create_parameter_distribution(cyclones: List[Cyclone],
                               parameter: str,
                               grid_resolution: float = 1.0,
                               smoothing_sigma: float = 1.0,
                               min_latitude: float = 60.0,
                               cmap: str = 'viridis',
                               figsize: Tuple[float, float] = (10, 8),
                               projection: str = 'NorthPolarStereo') -> Tuple[Figure, Axes]:
    """
    Создает карту пространственного распределения выбранного параметра циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        parameter: Название параметра для анализа ('vorticity_850hPa', 'max_wind_speed', 'radius' и т.д.).
        grid_resolution: Разрешение сетки в градусах.
        smoothing_sigma: Параметр сглаживания для гауссового фильтра.
        min_latitude: Минимальная широта для визуализации.
        cmap: Цветовая карта для отображения параметра.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип проекции карты.
        
    Возвращает:
        Кортеж (figure, axis) с созданной картой.
    """
    try:
        # Создаем базовую карту
        fig, ax = create_arctic_map(
            min_latitude=min_latitude, 
            figsize=figsize,
            projection=projection
        )
        
        # Создаем сетку для суммирования параметра и счетчик
        lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
        lat_bins = np.arange(min_latitude, 90 + grid_resolution, grid_resolution)
        
        # Инициализируем массивы для суммирования параметра и счетчика циклонов
        param_sum = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        count = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        
        # Определяем функцию для получения значения параметра
        def get_parameter_value(cyclone: Cyclone) -> Optional[float]:
            # Проверяем, есть ли параметр в атрибутах циклона
            if hasattr(cyclone, parameter):
                return getattr(cyclone, parameter)
            
            # Проверяем, есть ли параметр в объекте parameters
            if hasattr(cyclone, 'parameters') and hasattr(cyclone.parameters, parameter):
                return getattr(cyclone.parameters, parameter)
            
            # Если параметр - центральное давление
            if parameter == 'central_pressure':
                return cyclone.central_pressure
            
            # Если параметр - интенсивность
            if parameter == 'intensity':
                return cyclone.calculate_intensity_index() if hasattr(cyclone, 'calculate_intensity_index') else None
            
            # Для других параметров
            return None
        
        # Суммируем параметр по ячейкам сетки
        for cyclone in cyclones:
            param_value = get_parameter_value(cyclone)
            
            # Пропускаем циклоны без указанного параметра
            if param_value is None:
                continue
                
            lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
            lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
            
            # Проверяем, что индексы в пределах сетки
            if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
                param_sum[lat_idx, lon_idx] += param_value
                count[lat_idx, lon_idx] += 1
        
        # Рассчитываем среднее значение параметра (избегаем деления на ноль)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_param = np.where(count > 0, param_sum / count, np.nan)
        
        # Применяем сглаживание (только для непустых ячеек)
        if smoothing_sigma > 0:
            # Создаем маску для непустых ячеек
            valid_mask = ~np.isnan(mean_param)
            
            # Сглаживаем только непустые ячейки
            smoothed_data = np.copy(mean_param)
            smoothed_data[valid_mask] = gaussian_filter(
                mean_param[valid_mask], 
                sigma=smoothing_sigma
            )
            mean_param = smoothed_data
        
        # Создаем сетку координат для отображения
        lon_grid, lat_grid = np.meshgrid(
            lon_bins[:-1] + grid_resolution/2,
            lat_bins[:-1] + grid_resolution/2
        )
        
        # Определяем единицы измерения и настройки для разных параметров
        param_units = {
            'central_pressure': 'гПа',
            'vorticity_850hPa': '10⁻⁵ с⁻¹',
            'max_wind_speed': 'м/с',
            'radius': 'км',
            'intensity': 'ед.'
        }
        
        # Нормализация для разных параметров
        if parameter == 'vorticity_850hPa':
            # Преобразуем в 10⁻⁵ с⁻¹ для читаемости
            mean_param *= 1e5
        
        # Отображаем тепловую карту
        contour = ax.contourf(
            lon_grid, lat_grid, mean_param, 
            transform=ccrs.PlateCarree(),
            cmap=cmap, 
            levels=20,
            extend='both'
        )
        
        # Добавляем цветовую шкалу
        cbar = fig.colorbar(contour, ax=ax, pad=0.05)
        cbar.set_label(f"Среднее значение {parameter} ({param_units.get(parameter, '')})")
        
        # Устанавливаем заголовок
        ax.set_title(f'Пространственное распределение {parameter}')
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при создании карты распределения параметра: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def create_seasonal_analysis(cyclones: List[Cyclone],
                          parameter: str = 'frequency',
                          seasons: Dict[str, Tuple[int, int]] = None,
                          grid_resolution: float = 1.0,
                          smoothing_sigma: float = 1.0,
                          min_latitude: float = 60.0,
                          cmap: str = 'YlOrRd',
                          figsize: Tuple[float, float] = (16, 12)) -> Figure:
    """
    Создает сезонный анализ указанного параметра циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        parameter: Параметр для анализа ('frequency', 'pressure', 'vorticity_850hPa', 'max_wind_speed', и т.д.).
        seasons: Словарь сезонов в формате {'name': (start_month, end_month)}.
        grid_resolution: Разрешение сетки в градусах.
        smoothing_sigma: Параметр сглаживания для гауссового фильтра.
        min_latitude: Минимальная широта для визуализации.
        cmap: Цветовая карта для отображения параметра.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        
    Возвращает:
        Фигура с картами для каждого сезона.
    """
    try:
        # Определяем сезоны, если не указаны
        if seasons is None:
            seasons = {
                'Winter': (12, 2),
                'Spring': (3, 5),
                'Summer': (6, 8),
                'Autumn': (9, 11)
            }
        
        # Создаем фигуру с подграфиками
        fig = plt.figure(figsize=figsize)
        n_seasons = len(seasons)
        
        # Определяем сетку для размещения подграфиков
        n_cols = 2
        n_rows = (n_seasons + 1) // 2  # +1 для общей карты
        
        # Группируем циклоны по сезонам
        seasonal_cyclones = {season: [] for season in seasons}
        
        for cyclone in cyclones:
            month = cyclone.time.month
            for season, (start, end) in seasons.items():
                if start <= end:
                    # Обычный сезон (например, весна: 3-5)
                    if start <= month <= end:
                        seasonal_cyclones[season].append(cyclone)
                else:
                    # Сезон, переходящий через год (например, зима: 12-2)
                    if month >= start or month <= end:
                        seasonal_cyclones[season].append(cyclone)
        
        # Создаем карту для каждого сезона
        for i, (season, season_cyclones) in enumerate(seasonal_cyclones.items()):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, 
                               projection=ccrs.NorthPolarStereo())
            
            # Настраиваем карту
            ax.set_extent([-180, 180, min_latitude, 90], ccrs.PlateCarree())
            ax.coastlines(resolution='50m')
            ax.gridlines()
            
            # Создаем сетку для визуализации
            lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
            lat_bins = np.arange(min_latitude, 90 + grid_resolution, grid_resolution)
            
            # Инициализируем массивы в зависимости от параметра
            if parameter == 'frequency':
                # Для частоты - просто считаем циклоны
                grid = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
                
                for cyclone in season_cyclones:
                    lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
                    lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
                    
                    if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
                        grid[lat_idx, lon_idx] += 1
                
                # Применяем сглаживание
                if smoothing_sigma > 0:
                    grid = gaussian_filter(grid, sigma=smoothing_sigma)
                
                # Создаем сетку координат
                lon_grid, lat_grid = np.meshgrid(
                    lon_bins[:-1] + grid_resolution/2,
                    lat_bins[:-1] + grid_resolution/2
                )
                
                # Отображаем тепловую карту
                contour = ax.contourf(
                    lon_grid, lat_grid, grid, 
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, 
                    levels=20,
                    extend='max'
                )
                
                # Добавляем цветовую шкалу
                cbar = fig.colorbar(contour, ax=ax, pad=0.05, fraction=0.046)
                cbar.set_label('Количество циклонов')
                
            else:
                # Для других параметров - считаем среднее значение
                param_sum = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
                count = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
                
                # Определяем функцию для получения значения параметра
                def get_parameter_value(cyclone: Cyclone) -> Optional[float]:
                    if parameter == 'central_pressure':
                        return cyclone.central_pressure
                    elif hasattr(cyclone.parameters, parameter):
                        return getattr(cyclone.parameters, parameter)
                    return None
                
                # Суммируем параметр
                for cyclone in season_cyclones:
                    param_value = get_parameter_value(cyclone)
                    
                    if param_value is None:
                        continue
                        
                    lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
                    lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
                    
                    if 0 <= lon_idx < len(lon_bins) - 1 and 0 <= lat_idx < len(lat_bins) - 1:
                        param_sum[lat_idx, lon_idx] += param_value
                        count[lat_idx, lon_idx] += 1
                
                # Рассчитываем среднее значение
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_param = np.where(count > 0, param_sum / count, np.nan)
                
                # Применяем сглаживание
                if smoothing_sigma > 0:
                    valid_mask = ~np.isnan(mean_param)
                    smoothed_data = np.copy(mean_param)
                    smoothed_data[valid_mask] = gaussian_filter(
                        mean_param[valid_mask], 
                        sigma=smoothing_sigma
                    )
                    mean_param = smoothed_data
                
                # Создаем сетку координат
                lon_grid, lat_grid = np.meshgrid(
                    lon_bins[:-1] + grid_resolution/2,
                    lat_bins[:-1] + grid_resolution/2
                )
                
                # Нормализация для разных параметров
                if parameter == 'vorticity_850hPa':
                    mean_param *= 1e5
                
                # Отображаем тепловую карту
                contour = ax.contourf(
                    lon_grid, lat_grid, mean_param, 
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, 
                    levels=20,
                    extend='both'
                )
                
                # Добавляем цветовую шкалу
                cbar = fig.colorbar(contour, ax=ax, pad=0.05, fraction=0.046)
                param_units = {
                    'central_pressure': 'гПа',
                    'vorticity_850hPa': '10⁻⁵ с⁻¹',
                    'max_wind_speed': 'м/с',
                    'radius': 'км'
                }
                cbar.set_label(f"Среднее значение {parameter} ({param_units.get(parameter, '')})")
            
            # Устанавливаем заголовок для сезона
            ax.set_title(f"{season} ({len(season_cyclones)} циклонов)")
        
        # Добавляем общий заголовок
        if parameter == 'frequency':
            title = "Сезонное распределение частоты арктических циклонов"
        else:
            title = f"Сезонное распределение параметра {parameter} арктических циклонов"
            
        fig.suptitle(title, fontsize=16, y=0.98)
        
        # Регулируем расстояние между подграфиками
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
        
    except Exception as e:
        error_msg = f"Ошибка при создании сезонного анализа: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)