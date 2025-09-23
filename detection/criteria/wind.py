"""
Модуль критерия скорости ветра для обнаружения циклонов.

Предоставляет улучшенный критерий обнаружения циклонов на основе
превышения пороговых значений скорости ветра с поддержкой фильтрации
и многоуровневого анализа.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import scipy.ndimage as ndimage

from . import BaseCriterion
from core.exceptions import DetectionError
from visualization.criteria import plot_wind_field

# Инициализация логгера
logger = logging.getLogger(__name__)

class WindCriterion(BaseCriterion):
    """
    Критерий обнаружения циклонов на основе скорости ветра.
    
    Улучшенная реализация критерия обнаружения циклонов на основе скорости ветра
    с поддержкой фильтрации полярных артефактов и многоуровневого анализа данных ветра.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                 min_speed: float = 15.0,
                 thresholds: Optional[Dict[str, float]] = None,
                 polar_filtering: Optional[Dict[str, Any]] = None,
                 pressure_level: int = 1000,
                 window_size: int = 3,
                 smooth_sigma: float = 1.5):
        """
        Инициализирует критерий скорости ветра.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            min_speed: Минимальная скорость ветра для обнаружения (м/с).
            thresholds: Словарь порогов для разных типов данных ветра.
            polar_filtering: Конфигурация фильтрации полярных артефактов.
            pressure_level: Уровень давления для анализа (гПа).
            window_size: Размер окна для поиска локальных максимумов.
            smooth_sigma: Параметр сглаживания поля скорости ветра.
        """
        self.min_latitude = min_latitude
        self.min_speed = min_speed
        self.pressure_level = pressure_level
        self.window_size = window_size
        self.smooth_sigma = smooth_sigma
        
        # Расширенная конфигурация порогов
        self.thresholds = thresholds or {
            'surface': 12.0,    # Порог для приповерхностного ветра (10м)
            'upper': 15.0,      # Порог для верхнего уровня (850гПа)
            'default': 15.0     # Порог по умолчанию
        }
        
        # Конфигурация фильтрации полярных артефактов
        self.polar_filtering = polar_filtering or {
            'enabled': True,
            'exclusion_radius': 2.0  # Радиус исключения вокруг Северного полюса (градусы)
        }
        
        # Поля для визуализации
        self.u_data = None
        self.v_data = None
        self.wind_speed = None
        
        logger.debug(f"Инициализирован критерий скорости ветра: "
                    f"min_latitude={min_latitude}, "
                    f"min_speed={min_speed}, "
                    f"thresholds={self.thresholds}, "
                    f"polar_filtering={self.polar_filtering}, "
                    f"pressure_level={pressure_level}, "
                    f"window_size={window_size}, "
                    f"smooth_sigma={smooth_sigma}")

    def _filter_polar_false_positives(self, candidates: List[Dict]) -> List[Dict]:
        """
        Фильтрует кандидаты, которые являются ложными срабатываниями у полюсов.
        
        Аргументы:
            candidates: Список словарей кандидатов
            
        Возвращает:
            Отфильтрованный список кандидатов
        """
        if not self.polar_filtering.get('enabled', True):
            return candidates
            
        exclusion_radius = self.polar_filtering.get('exclusion_radius', 2.0)
        filtered_candidates = []
        
        for candidate in candidates:
            lat = candidate['latitude']
            
            # Проверяем, не слишком ли близко к Северному полюсу
            if lat > (90 - exclusion_radius):
                logger.debug(f"Фильтрация кандидата на {lat}°N из-за близости к Северному полюсу")
                continue
            
            # Проверяем крайний случай: кандидаты точно на 90° широты
            if abs(lat - 90.0) < 0.01:
                logger.debug(f"Фильтрация кандидата точно на {lat}°N")
                continue
                
            filtered_candidates.append(candidate)
        
        logger.debug(f"Отфильтровано {len(candidates) - len(filtered_candidates)} ложных срабатываний у полюса")
        return filtered_candidates

    def _get_threshold_for_level(self, level_type: str) -> float:
        """
        Получает подходящий порог для типа данных ветра.
        
        Аргументы:
            level_type: Тип уровня данных ветра
            
        Возвращает:
            Пороговое значение скорости ветра
        """
        return self.thresholds.get(level_type, self.thresholds.get('default', self.min_speed))

    def _identify_wind_level(self, u_var: str, v_var: str, level_info: Optional[Dict] = None) -> str:
        """
        Определяет тип уровня ветровых данных по именам переменных и информации об уровне.
        
        Аргументы:
            u_var: Имя переменной зонального ветра
            v_var: Имя переменной меридионального ветра
            level_info: Информация об уровне давления
            
        Возвращает:
            Тип уровня данных ветра
        """
        # Проверяем приповерхностные переменные ветра
        surface_vars = ['u10', '10u', 'u_component_of_wind_10m', '10m_u_component_of_wind', 'u10n', 'v10n']
        if u_var in surface_vars or v_var in surface_vars:
            return 'surface'
            
        # Проверяем информацию об уровне давления
        if level_info and any(level_key in level_info for level_key in ['level', 'pressure_level', 'lev', 'plev']):
            level_value = list(level_info.values())[0] if level_info else None
            if level_value and level_value in [850, 925]:
                return 'upper'
            elif level_value:
                return f'level_{level_value}hPa'
        
        return 'default'

    def _calculate_wind_speed(self, dataset: xr.Dataset, time_step: Any, 
                             lat: float, lon: float) -> Optional[Dict]:
        """
        Рассчитывает скорость ветра в определенной точке с выбором подходящего уровня.
        
        Аргументы:
            dataset: Метеорологический набор данных
            time_step: Временной шаг для анализа
            lat: Широта
            lon: Долгота
            
        Возвращает:
            Словарь с информацией о ветре или None, если данные недоступны
        """
        try:
            # Выбираем конкретный временной шаг
            time_data = dataset.sel(time=time_step)
            
            # Сначала ищем приповерхностные данные ветра
            surface_wind_var_pairs = [
                ('u10', 'v10'),
                ('10u', '10v'),
                ('u_component_of_wind_10m', 'v_component_of_wind_10m'),
                ('10m_u_component_of_wind', '10m_v_component_of_wind')
            ]
            
            u_wind_var = None
            v_wind_var = None
            level_to_use = None
            
            # Ищем приповерхностные данные ветра
            for u_var, v_var in surface_wind_var_pairs:
                if u_var in time_data and v_var in time_data:
                    u_wind_var, v_wind_var = u_var, v_var
                    break
            
            # Если приповерхностные данные не найдены, ищем данные на уровнях давления
            if u_wind_var is None or v_wind_var is None:
                wind_var_pairs = [
                    ('u', 'v'),
                    ('u_component_of_wind', 'v_component_of_wind')
                ]
                
                for u_var, v_var in wind_var_pairs:
                    if u_var in time_data and v_var in time_data:
                        # Проверяем наличие уровней давления
                        pressure_level_names = ['level', 'pressure_level', 'lev', 'plev']
                        
                        for level_name in pressure_level_names:
                            if level_name in time_data.dims:
                                # Находим уровень 850 гПа
                                available_levels = time_data[level_name].values
                                target_level = 850
                                closest_level = available_levels[
                                    np.abs(available_levels - target_level).argmin()]
                                
                                u_wind_var = u_var
                                v_wind_var = v_var
                                level_to_use = {level_name: closest_level}
                                logger.debug(f"Используется уровень давления {closest_level} гПа для данных ветра")
                                break
                        
                        if level_to_use is not None:
                            break
                        else:
                            u_wind_var, v_wind_var = u_var, v_var
                            break
            
            if u_wind_var is None or v_wind_var is None:
                logger.warning("Не удалось найти компоненты ветра в наборе данных")
                return None
            
            # Извлекаем компоненты ветра в точке
            if level_to_use is not None:
                u_point = float(time_data[u_wind_var].sel(
                    **level_to_use, latitude=lat, longitude=lon, 
                    method='nearest').values)
                v_point = float(time_data[v_wind_var].sel(
                    **level_to_use, latitude=lat, longitude=lon, 
                    method='nearest').values)
            else:
                u_point = float(time_data[u_wind_var].sel(
                    latitude=lat, longitude=lon, method='nearest').values)
                v_point = float(time_data[v_wind_var].sel(
                    latitude=lat, longitude=lon, method='nearest').values)
            
            # Рассчитываем скорость ветра
            wind_speed = np.sqrt(u_point**2 + v_point**2)
            
            # Определяем тип уровня ветра для порога
            level_type = self._identify_wind_level(u_wind_var, v_wind_var, level_to_use)
            min_threshold = self._get_threshold_for_level(level_type)
            
            return {
                'u_wind': u_point,
                'v_wind': v_point,
                'wind_speed': wind_speed,
                'wind_level': level_type,
                'min_threshold': min_threshold
            }
            
        except Exception as e:
            logger.error(f"Ошибка при расчете скорости ветра: {str(e)}")
            return None

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
            # Выбираем конкретный временной шаг для упрощения обработки
            time_data = dataset.sel(time=time_step)
            
            # Применяем маску арктического региона
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Определяем переменные компонентов ветра
            u_vars = ['u10', '10u', 'u_component_of_wind_10m', '10m_u_component_of_wind', 'u', 'u_component_of_wind']
            v_vars = ['v10', '10v', 'v_component_of_wind_10m', '10m_v_component_of_wind', 'v', 'v_component_of_wind']
            
            u_var, v_var = None, None
            
            # Поиск переменных u и v в наборе данных
            for u in u_vars:
                if u in arctic_data:
                    u_var = u
                    break
                    
            for v in v_vars:
                if v in arctic_data:
                    v_var = v
                    break
            
            if u_var is None or v_var is None:
                available_vars = list(arctic_data.variables)
                logger.error(f"Не удается определить компоненты ветра в наборе данных. Доступные переменные: {available_vars}")
                raise ValueError(f"Не удается определить компоненты ветра в наборе данных. Доступные переменные: {available_vars}")
            
            # Обработка уровней давления, если они есть
            u_data = None
            v_data = None
            
            # Проверяем наличие уровней давления
            pressure_level_names = ['level', 'pressure_level', 'lev', 'plev']
            
            for level_name in pressure_level_names:
                if level_name in arctic_data.dims:
                    # Находим ближайший доступный уровень к указанному уровню давления
                    available_levels = arctic_data[level_name].values
                    closest_level = available_levels[np.abs(available_levels - self.pressure_level).argmin()]
                    logger.debug(f"Используется уровень давления {closest_level} гПа (ближайший к целевому {self.pressure_level} гПа)")
                    
                    # Выбираем соответствующий уровень
                    u_data = arctic_data[u_var].sel({level_name: closest_level})
                    v_data = arctic_data[v_var].sel({level_name: closest_level})
                    break
            
            # Если измерение уровня не найдено, используем данные как есть
            if u_data is None or v_data is None:
                u_data = arctic_data[u_var]
                v_data = arctic_data[v_var]
            
            # Рассчитываем скорость ветра
            wind_speed = np.sqrt(u_data**2 + v_data**2)
            
            # Логируем размерности данных ветра для отладки
            logger.debug(f"Размерности данных ветра - u: {u_data.shape}, v: {v_data.shape}, скорость ветра: {wind_speed.shape}")
            
            # Сглаживаем поле для уменьшения шума
            try:
                wind_values = wind_speed.values
                
                # Обработка многомерных массивов
                if wind_values.ndim > 2:
                    logger.warning(f"Данные ветра имеют форму {wind_values.shape}, уменьшаем до 2D")
                    
                    # Если у нас больше 2 измерений, выравниваем все кроме широты/долготы
                    if hasattr(wind_speed, 'latitude') and hasattr(wind_speed, 'longitude'):
                        # Если широта и долгота являются последними двумя измерениями
                        lat_dim = len(arctic_data.latitude)
                        lon_dim = len(arctic_data.longitude)
                        
                        if wind_values.shape[-2:] == (lat_dim, lon_dim):
                            # Используем последний 2D срез, если он соответствует размерам широты/долготы
                            wind_values = wind_values.reshape(-1, lat_dim, lon_dim)[-1]
                            logger.info(f"Используется последний 2D срез данных ветра с формой {wind_values.shape}")
                        else:
                            # Пытаемся усреднить по дополнительным измерениям
                            wind_values = np.mean(wind_values, axis=tuple(range(wind_values.ndim - 2)))
                            logger.info(f"Усредненные данные ветра до формы {wind_values.shape}")
                
                # Применяем гауссово сглаживание
                if self.smooth_sigma > 0 and wind_values.ndim == 2:
                    smoothed_field = ndimage.gaussian_filter(wind_values, sigma=self.smooth_sigma)
                else:
                    smoothed_field = wind_values
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке данных ветра: {str(e)}")
                return []
            
            # Находим локальные максимумы выше порога
            try:
                # Убедимся, что smoothed_field двумерный перед поиском локальных максимумов
                if smoothed_field.ndim != 2:
                    logger.warning(f"Сглаженное поле ветра имеет {smoothed_field.ndim} измерений, пытаемся уменьшить до 2D")
                    if smoothed_field.ndim > 2:
                        # Используем среднее по дополнительным измерениям или первый срез
                        if smoothed_field.size > 0:
                            if smoothed_field.shape[0] == 1:
                                smoothed_field = smoothed_field[0]
                            else:
                                # Пытаемся усреднить по первому измерению
                                smoothed_field = np.mean(smoothed_field, axis=0)
                    else:
                        # Если это 1D, нельзя использовать для обнаружения максимумов
                        logger.error("Невозможно использовать 1D данные скорости ветра для обнаружения максимумов")
                        return []
                
                # Определяем подходящий порог на основе конфигурации
                # По умолчанию используем порог для приповерхностного ветра
                threshold = self._get_threshold_for_level('surface')
                
                max_filter = ndimage.maximum_filter(smoothed_field, size=self.window_size)
                wind_speed_maxima = (smoothed_field == max_filter) & (smoothed_field >= threshold)
                
                maxima_indices = np.where(wind_speed_maxima)

                # Сохраняем данные поля ветра для визуализации
                # Убеждаемся, что широты и долготы двумерные для построения графиков
                plot_lons, plot_lats = np.meshgrid(arctic_data.longitude.values, arctic_data.latitude.values)
                
                # Убеждаемся, что u_data и v_data соответствуют форме plot_lats и plot_lons
                # Если они одномерные массивы, преобразуем их в двумерные с помощью meshgrid
                if u_data.ndim == 1 and v_data.ndim == 1:
                    u_grid, v_grid = np.meshgrid(u_data, v_data)
                # Если они уже двумерные, но не соответствуют форме meshgrid
                elif u_data.shape != plot_lats.shape or v_data.shape != plot_lats.shape:
                    # Интерполируем для соответствия сетке, если формы разные
                    # Пока будем изменять форму, если возможно
                    try:
                        u_grid = np.reshape(u_data, plot_lats.shape)
                        v_grid = np.reshape(v_data, plot_lats.shape)
                    except ValueError:
                        # Если изменение формы не удается, пытаемся транслировать, если размерности совместимы
                        if u_data.shape[0] == plot_lats.shape[0] and v_data.shape[0] == plot_lats.shape[0]:
                            # Транслируем одномерные массивы в двумерные, если первое измерение совпадает
                            u_grid = np.broadcast_to(u_data[:, np.newaxis], plot_lats.shape)
                            v_grid = np.broadcast_to(v_data[:, np.newaxis], plot_lats.shape)
                        else:
                            # Логируем ошибку и пропускаем построение графиков, если формы не могут быть согласованы
                            logger.error(f"Невозможно изменить форму данных ветра с {u_data.shape} на {plot_lats.shape} для построения графиков")
                            raise ValueError(f"Форма данных ветра {u_data.shape} несовместима с формой сетки {plot_lats.shape}")
                else:
                    # Уже правильная форма
                    u_grid = u_data
                    v_grid = v_data
                    
                # Сохраняем данные поля ветра для визуализации
                self.u_data = u_grid
                self.v_data = v_grid
                self.wind_speed = smoothed_field
                
                # Создаем визуализацию, если debug_plot включен
                if debug_plot and output_dir:
                    try:
                        # Логируем формы для отладки
                        logger.debug(f"Построение поля ветра с формами - широты: {plot_lats.shape}, долготы: {plot_lons.shape}, u: {u_grid.shape}, v: {v_grid.shape}")
                        
                        plot_wind_field(
                            u_wind=u_grid, 
                            v_wind=v_grid,
                            lats=plot_lats, 
                            lons=plot_lons,
                            threshold=threshold,
                            time_step=time_step,
                            output_dir=output_dir
                        )
                        logger.debug(f"Сохранен график критерия wind для {time_step} в {output_dir}")
                    except Exception as plot_e:
                        logger.error(f"Ошибка при построении поля ветра для {time_step}: {plot_e}")

                if len(maxima_indices) < 2 or len(maxima_indices[0]) == 0:
                    logger.warning("Не найдено максимумов скорости ветра выше порога")
                    return []
                
                # Формируем список кандидатов
                candidates = []
                
                for i in range(len(maxima_indices[0])):
                    lat_idx = maxima_indices[0][i]
                    lon_idx = maxima_indices[1][i]
                    
                    if lat_idx < len(arctic_data.latitude) and lon_idx < len(arctic_data.longitude):
                        latitude = float(arctic_data.latitude.values[lat_idx])
                        longitude = float(arctic_data.longitude.values[lon_idx])
                        speed = float(smoothed_field[lat_idx, lon_idx])
                        
                        # Проверяем скорость ветра в этой точке с правильным выбором уровня
                        wind_info = self._calculate_wind_speed(dataset, time_step, latitude, longitude)
                        
                        if wind_info is None:
                            continue  # Пропускаем, если данные ветра недоступны
                        
                        # Проверяем, соответствует ли скорость ветра порогу
                        if wind_info['wind_speed'] < wind_info['min_threshold']:
                            continue  # Пропускаем, если ниже порога
                        
                        # Создаем кандидата
                        candidate = {
                            'latitude': latitude,
                            'longitude': longitude,
                            'wind_speed': wind_info['wind_speed'],
                            'u_wind': wind_info['u_wind'],
                            'v_wind': wind_info['v_wind'],
                            'wind_level': wind_info['wind_level'],
                            'criterion': 'wind'
                        }
                        
                        candidates.append(candidate)
                    else:
                        logger.warning(f"Недопустимые индексы максимумов ветра: lat_idx={lat_idx}, lon_idx={lon_idx}")
                
                # Фильтруем ложные срабатывания у полюса
                candidates = self._filter_polar_false_positives(candidates)
                
                logger.debug(f"Критерий скорости ветра нашел {len(candidates)} кандидатов")
                return candidates
                
            except Exception as e:
                logger.error(f"Ошибка при поиске локальных максимумов скорости ветра: {str(e)}")
                return []
                
        except Exception as e:
            error_msg = f"Ошибка при применении критерия скорости ветра: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)