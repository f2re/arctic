"""
Модуль критерия лапласиана давления для обнаружения арктических мезоциклонов.

Предоставляет критерий обнаружения циклонов на основе расчета лапласиана поля давления.
Лапласиан давления (∇²p) пропорционален геострофической относительной завихренности,
что делает его эффективным для идентификации циклонических образований.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import scipy.ndimage as ndimage
from scipy import signal

from . import BaseCriterion
from core.exceptions import DetectionError
from visualization.criteria import plot_laplacian_field

# Инициализация логгера
logger = logging.getLogger(__name__)

class PressureLaplacianCriterion(BaseCriterion):
    """
    Критерий обнаружения циклонов на основе лапласиана давления.
    
    Вычисляет лапласиан (∇²p) поля давления на уровне моря для выявления
    областей с сильными циклоническими свойствами. Положительные значения
    лапласиана соответствуют циклоническим системам в Северном полушарии.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                laplacian_threshold: float = 0.15,  # Pa/km²
                window_size: int = 5,
                smooth_sigma: float = 1.5):
        """
        Инициализирует критерий лапласиана давления.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            laplacian_threshold: Пороговое значение лапласиана давления (Па/км²).
            window_size: Размер окна для поиска локальных максимумов.
            smooth_sigma: Параметр сглаживания поля давления перед расчетом лапласиана.
        """
        self.min_latitude = min_latitude
        self.laplacian_threshold = laplacian_threshold
        self.window_size = window_size
        self.smooth_sigma = smooth_sigma
        
        logger.debug(f"Инициализирован критерий лапласиана давления: "
                    f"min_latitude={min_latitude}, "
                    f"laplacian_threshold={laplacian_threshold}, "
                    f"window_size={window_size}, "
                    f"smooth_sigma={smooth_sigma}")
    
    def apply(self, dataset: xr.Dataset, time_step: Any, debug_plot: bool = False, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Применяет критерий к набору данных.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            debug_plot: Если True, включает построение графиков полей критериев для отладки.
            output_dir: Каталог для сохранения графиков, если debug_plot=True.
            
        Возвращает:
            Список кандидатов в циклоны (словари с координатами и свойствами).
            
        Вызывает:
            DetectionError: При ошибке обнаружения циклонов.
        """
        try:
            # Выбираем временной шаг
            time_data = dataset.sel(time=time_step)
            
            # Применяем маску для арктического региона
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Определяем переменную давления в наборе данных
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            
            # Log available variables for debugging
            logger.debug(f"Available variables: {list(arctic_data.variables)}")
            
            for var in pressure_vars:
                if var in arctic_data:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                available_vars = list(arctic_data.variables)
                logger.error(f"Не удается определить переменную давления в наборе данных. Доступные переменные: {available_vars}")
                raise ValueError(f"Не удается определить переменную давления в наборе данных. Доступные переменные: {available_vars}")
            
            # Получаем данные о давлении
            pressure_data = arctic_data[pressure_var]
            
            # Преобразуем координаты в радианы
            lat_rad = np.radians(arctic_data.latitude)
            lon_rad = np.radians(arctic_data.longitude)
            
            # Рассчитываем расстояния в километрах
            R = 6371.0  # Радиус Земли в километрах
            
            # Получаем размеры сетки
            logger.debug(f"Размеры сетки: lat={len(lat_rad)}, lon={len(lon_rad)}")
            
            # Создаем сетку расстояний с учетом размерности
            # Используем скалярные значения для упрощения расчетов
            dlat = np.abs(np.mean(np.gradient(lat_rad)))  # Средний шаг по широте в радианах
            dlon = np.abs(np.mean(np.gradient(lon_rad)))  # Средний шаг по долготе в радианах
            
            # Переводим в километры
            dlat_km = R * dlat  # Шаг по широте в км
            dlon_km = R * np.mean(np.cos(lat_rad)) * dlon  # Шаг по долготе в км
            
            logger.debug(f"Шаг сетки: dlat_km={dlat_km}, dlon_km={dlon_km}")
            
            # Получаем значения давления в Паскалях (1 гПа = 100 Па)
            try:
                pressure_values = pressure_data.values
                
                # Сглаживаем поле для уменьшения шума
                if self.smooth_sigma > 0:
                    pressure_values = ndimage.gaussian_filter(pressure_values, sigma=self.smooth_sigma)
                
                # Рассчитываем лапласиан давления
                # Используем сетку расстояний для корректного расчета производных
                # ∇²p = ∂²p/∂x² + ∂²p/∂y²
                
                # Рассчитываем вторые производные
                # Рассчитываем вторые производные с учетом шага сетки
                # Используем скалярные значения шагов, которые уже рассчитаны выше
                # Убедимся, что dlat_km и dlon_km - скалярные значения
                dlat_km_scalar = float(dlat_km)
                dlon_km_scalar = float(dlon_km)
                
                # Вычисляем вторые производные
                d2p_dy2 = np.gradient(np.gradient(pressure_values, axis=0, edge_order=2), axis=0, edge_order=2) / (dlat_km_scalar**2)
                d2p_dx2 = np.gradient(np.gradient(pressure_values, axis=1, edge_order=2), axis=1, edge_order=2) / (dlon_km_scalar**2)
                
                # Суммируем для получения лапласиана и убедимся, что результат - numpy массив
                laplacian = np.array(d2p_dy2 + d2p_dx2)
                
                # Лапласиан в Па/км²
                # Положительные значения соответствуют циклонам (минимумам давления)

                if debug_plot and output_dir:
                    try:
                        # Ensure lats and lons are 2D for plotting if they are 1D
                        plot_lons, plot_lats = np.meshgrid(arctic_data.longitude.values, arctic_data.latitude.values)
                        plot_laplacian_field(
                            laplacian=laplacian, 
                            lats=plot_lats, 
                            lons=plot_lons,
                            threshold=self.laplacian_threshold,
                            time_step=time_step,
                            output_dir=output_dir
                        )
                        logger.debug(f"Saved pressure_laplacian plot for {time_step} to {output_dir}")
                    except Exception as plot_e:
                        logger.error(f"Error plotting pressure_laplacian field for {time_step}: {plot_e}")
                
                # Находим локальные максимумы лапласиана
                max_filter = ndimage.maximum_filter(laplacian, size=self.window_size)
                laplacian_maxima = (laplacian == max_filter) & (laplacian > self.laplacian_threshold)
                
                # Получаем координаты максимумов
                maxima_indices = np.where(laplacian_maxima)
                
                # Формируем список кандидатов
                candidates = []
                
                for i in range(len(maxima_indices[0])):
                    lat_idx = maxima_indices[0][i]
                    lon_idx = maxima_indices[1][i]
                    
                    if lat_idx < len(arctic_data.latitude) and lon_idx < len(arctic_data.longitude):
                        latitude = float(arctic_data.latitude.values[lat_idx])
                        longitude = float(arctic_data.longitude.values[lon_idx])
                        laplacian_value = float(laplacian[lat_idx, lon_idx])
                        pressure_value = float(pressure_values[lat_idx, lon_idx]) / 100.0  # Конвертируем в гПа
                        
                        # Создаем кандидата
                        candidate = {
                            'latitude': latitude,
                            'longitude': longitude,
                            'laplacian': laplacian_value,
                            'pressure': pressure_value,
                            'criterion': 'pressure_laplacian'
                        }
                        
                        candidates.append(candidate)
                    else:
                        logger.warning(f"Invalid laplacian maxima indices: lat_idx={lat_idx}, lon_idx={lon_idx}")
                
                logger.debug(f"Критерий лапласиана давления нашел {len(candidates)} кандидатов")
                return candidates
                
            except Exception as e:
                logger.error(f"Ошибка при расчете лапласиана давления: {str(e)}")
                return []
                
        except Exception as e:
            error_msg = f"Ошибка при применении критерия лапласиана давления: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
    
    def _calculate_laplacian_kernel(self, dx: float, dy: float) -> np.ndarray:
        """
        Создает ядро для дискретного расчета лапласиана на неравномерной сетке.
        
        Аргументы:
            dx: Шаг сетки по оси x (долгота) в километрах.
            dy: Шаг сетки по оси y (широта) в километрах.
            
        Возвращает:
            Ядро для дискретного лапласиана.
        """
        # Создаем базовое ядро лапласиана для неравномерной сетки
        # Используем 5-точечную схему второго порядка
        kernel = np.zeros((3, 3))
        
        # Центральная точка
        kernel[1, 1] = -2.0 / (dx**2) - 2.0 / (dy**2)
        
        # Точки по осям
        kernel[0, 1] = 1.0 / (dy**2)  # Верх
        kernel[2, 1] = 1.0 / (dy**2)  # Низ
        kernel[1, 0] = 1.0 / (dx**2)  # Лево
        kernel[1, 2] = 1.0 / (dx**2)  # Право
        
        return kernel
