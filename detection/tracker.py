"""
Модуль обнаружения и отслеживания циклонов для системы ArcticCyclone.

Предоставляет алгоритмы для обнаружения и отслеживания полного жизненного цикла
арктических мезоциклонов в наборах метеорологических данных.
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Set, Callable
from pathlib import Path
from datetime import datetime, timedelta
import scipy.ndimage as ndimage
import logging
from scipy.spatial.distance import cdist
import warnings
import uuid

from core.exceptions import DetectionError, TrackingError
from models.cyclone import Cyclone, CycloneType, CycloneParameters
from .criteria import CriteriaManager, BaseCriterion
from .validators import DetectionValidator

# Инициализация логгера
logger = logging.getLogger(__name__)

class CycloneDetector:
    """
    Обнаруживает арктические мезоциклоны на основе настраиваемых критериев.
    
    Предоставляет методы для поиска циклонов в наборах метеорологических данных
    с использованием гибкой системы критериев обнаружения.
    """
    
    def __init__(self, min_latitude: float = 70.0, config=None):
        """
        Инициализирует детектор циклонов с указанной минимальной широтой.
        
        Аргументы:
            min_latitude: Минимальная широта для Арктического региона (по умолчанию 70°N).
            config: Конфигурация обнаружения циклонов из config.yaml
        """
        self.min_latitude = min_latitude
        self.criteria_manager = CriteriaManager()
        self.validator = DetectionValidator()
        self.config = config
        
        
        # Если конфигурация предоставлена, устанавливаем активные критерии из конфигурации
        if self.config:
            self._configure_criteria_from_config()
        else:
            # Регистрируем стандартные критерии обнаружения
            self._register_default_criteria()
        
        logger.info(f"Инициализирован детектор циклонов с минимальной широтой {min_latitude}°N")
    
    def _register_default_criteria(self):
        """
        Регистрирует стандартные критерии обнаружения в менеджере критериев.
        """
        from detection.criteria import (
            PressureMinimumCriterion,
            VorticityCriterion,
            ClosedContourCriterion,
            WindThresholdCriterion,
            PressureLaplacianCriterion
        )

        self.criteria_manager.register_criterion('pressure_minimum', PressureMinimumCriterion)
        self.criteria_manager.register_criterion('vorticity', VorticityCriterion)
        self.criteria_manager.register_criterion('closed_contour', ClosedContourCriterion)
        self.criteria_manager.register_criterion('wind_threshold', WindThresholdCriterion)
        self.criteria_manager.register_criterion('pressure_laplacian', PressureLaplacianCriterion)
        
        # Настраиваем стандартную комбинацию критериев
        self.criteria_manager.set_active_criteria(["pressure_minimum", "vorticity"])
        
        logger.info("Зарегистрированы стандартные критерии обнаружения циклонов")
    
    def _configure_criteria_from_config(self) -> None:
        """
        Устанавливает активные критерии обнаружения циклонов из конфигурации.
        """
        if not self.config or 'detection' not in self.config or 'criteria' not in self.config['detection']:
            logger.warning("Конфигурация критериев не найдена, используем стандартные критерии")
            return
            
        criteria_config = self.config['detection']['criteria']
        active_criteria = []
        
        # Сохраняем настройки критериев для использования при создании экземпляров
        self.criteria_params = {}
        
        # Проходим по всем критериям в конфигурации и добавляем активные
        for criterion_name, settings in criteria_config.items():
            if settings.get('enabled', False):
                # Проверяем, зарегистрирован ли критерий
                if criterion_name in self.criteria_manager.criteria:
                    active_criteria.append(criterion_name)
                    
                    # Сохраняем параметры критерия для использования при создании экземпляра
                    self.criteria_params[criterion_name] = {
                        k: v for k, v in settings.items() if k != 'enabled'
                    }
                    logger.debug(f"Сохранены параметры для критерия {criterion_name}: {self.criteria_params[criterion_name]}")
                else:
                    logger.warning(f"Критерий {criterion_name} указан в конфигурации, но не зарегистрирован")
        
        # Устанавливаем активные критерии
        if active_criteria:
            self.criteria_manager.set_active_criteria(active_criteria)
            logger.info(f"Установлены активные критерии из конфигурации: {', '.join(active_criteria)}")
        else:
            logger.warning("В конфигурации не найдены активные критерии, используем стандартные")
    
    def detect(self, dataset: xr.Dataset, time_step: Optional[str] = None) -> List[Cyclone]:
        """
        Обнаруживает циклоны в наборе данных для указанного временного шага.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа. Если None, используется первый доступный.
            
        Возвращает:
            Список обнаруженных объектов Cyclone.
            
        Вызывает:
            DetectionError: При ошибке обнаружения циклонов.
        """
        try:
            if 'valid_time' in dataset.dims and 'time' not in dataset.dims:
                dataset = dataset.rename({'valid_time': 'time'})
            # Проверяем набор данных
            if 'time' not in dataset.dims:
                raise ValueError("Набор данных должен содержать измерение 'time'")
                
            # Выбираем временной шаг
            if time_step is None:
                time_step = dataset.time.values[0]
                logger.info(f"Используется первый доступный временной шаг: {time_step}")
            else:
                if time_step not in dataset.time.values:
                    raise ValueError(f"Временной шаг {time_step} отсутствует в наборе данных")
            
            # Применяем маску арктического региона
            arctic_mask = dataset.latitude >= self.min_latitude
            arctic_data = dataset.where(arctic_mask, drop=True)
            
            # Применяем критерии обнаружения
            cyclone_candidates = self._apply_detection_criteria(arctic_data, time_step)
            
            # Создаем объекты циклонов
            cyclones = []
            for candidate in cyclone_candidates:
                try:
                    # Создаем объект циклона
                    cyclone = self._create_cyclone(candidate, arctic_data, time_step)
                    
                    # Проверяем валидность
                    if self.validator.validate_cyclone(cyclone, arctic_data):
                        cyclones.append(cyclone)
                    else:
                        logger.debug(f"Отклонен кандидат в циклоны в точке ({candidate['latitude']}, {candidate['longitude']})")
                except Exception as e:
                    logger.warning(f"Ошибка при создании циклона: {str(e)}")
            
            logger.info(f"Обнаружено {len(cyclones)} циклонов для временного шага {time_step}")
            return cyclones
            
        except Exception as e:
            error_msg = f"Ошибка при обнаружении циклонов: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
    
    def _apply_detection_criteria(self, dataset: xr.Dataset, time_step: Any) -> List[Dict]:
        """
        Применяет набор критериев обнаружения циклонов.
        
        Аргументы:
            dataset: Набор метеорологических данных.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список кандидатов в циклоны (словари с координатами и свойствами).
        """
        # Получаем активные критерии
        active_criteria_names = self.criteria_manager.get_active_criterion_names()
        active_criteria_classes = self.criteria_manager.get_active_criteria()
        
        if not active_criteria_classes:
            logger.warning("Не заданы активные критерии обнаружения, используется критерий минимума давления")
            from .criteria.pressure import PressureMinimumCriterion
            active_criteria_classes = [PressureMinimumCriterion]
            active_criteria_names = ['pressure_minimum']
        
        # Применяем каждый критерий
        candidates_by_criterion = []
        
        for i, criterion_class in enumerate(active_criteria_classes):
            criterion_name = criterion_class.__name__
            criterion_key = active_criteria_names[i] if i < len(active_criteria_names) else None
            logger.debug(f"Применение критерия: {criterion_name}")
            
            try:
                # Prepare params for criterion instantiation
                params = {'min_latitude': self.min_latitude}
                
                # Add parameters from config if available
                if hasattr(self, 'criteria_params') and criterion_key in self.criteria_params:
                    params.update(self.criteria_params[criterion_key])
                    logger.debug(f"Применение параметров для {criterion_key}: {params}")
                
                # Instantiate the criterion with parameters
                criterion = criterion_class(**params)
                
                # Apply the criterion with time_step parameter
                candidates = criterion.apply(dataset, time_step)
                candidates_by_criterion.append(candidates)
                logger.debug(f"Критерий {criterion_name} нашел {len(candidates)} кандидатов")
            except Exception as e:
                logger.warning(f"Ошибка при применении критерия {criterion_class.__name__}: {str(e)}")
        
        # Фильтруем кандидатов, подходящих под все критерии
        if len(candidates_by_criterion) == 0:
            return []
            
        if len(candidates_by_criterion) == 1:
            return candidates_by_criterion[0]
            
        # Объединяем результаты всех критериев
        # Это можно сделать по-разному в зависимости от требований:
        # 1. Пересечение (циклон должен удовлетворять всем критериям)
        # 2. Объединение (циклон должен удовлетворять хотя бы одному критерию)
        # 3. Взвешенное решение (баллы за каждый критерий)
        
        # Реализуем пересечение по координатам (с некоторой допустимой погрешностью)
        tolerance = 1.0  # допустимое отклонение в градусах
        
        # Начинаем с кандидатов первого критерия
        filtered_candidates = candidates_by_criterion[0].copy()
        
        # Проверяем соответствие кандидатов остальным критериям
        for candidates in candidates_by_criterion[1:]:
            # Создаем массивы координат
            fc_coords = np.array([[c['latitude'], c['longitude']] for c in filtered_candidates])
            c_coords = np.array([[c['latitude'], c['longitude']] for c in candidates])
            
            if len(fc_coords) == 0 or len(c_coords) == 0:
                filtered_candidates = []
                break
            
            # Рассчитываем матрицу расстояний
            distances = cdist(fc_coords, c_coords)
            
            # Находим минимальные расстояния и соответствующие индексы
            min_distances = np.min(distances, axis=1)
            min_indices = np.argmin(distances, axis=1)
            
            # Отбираем кандидатов, удовлетворяющих порогу
            valid_indices = np.where(min_distances <= tolerance)[0]
            
            # Обновляем отфильтрованный список
            new_filtered = []
            for i in valid_indices:
                # Объединяем свойства кандидатов из разных критериев
                merged_candidate = filtered_candidates[i].copy()
                j = min_indices[i]
                
                for key, value in candidates[j].items():
                    if key not in merged_candidate:
                        merged_candidate[key] = value
                
                new_filtered.append(merged_candidate)
            
            filtered_candidates = new_filtered
        
        return filtered_candidates
    
    def _create_cyclone(self, candidate: Dict, dataset: xr.Dataset, time_step: Any) -> Cyclone:
        """
        Создает объект циклона из кандидата.
        
        Аргументы:
            candidate: Словарь с характеристиками кандидата.
            dataset: Набор метеорологических данных.
            time_step: Временной шаг.
            
        Возвращает:
            Объект Cyclone.
        """
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
                
            # Находим ближайшую точку сетки к координатам кандидата
            time_data = dataset.sel(time=time_step)
            lat_idx = np.abs(time_data.latitude.values - latitude).argmin()
            lon_idx = np.abs(time_data.longitude.values - longitude).argmin()
            
            # Получаем значение давления
            central_pressure = float(time_data[pressure_var].isel(latitude=lat_idx, longitude=lon_idx).values)
        
        # Преобразуем временной шаг в datetime, если необходимо
        if isinstance(time_step, str) or isinstance(time_step, np.datetime64):
            time_obj = pd.to_datetime(time_step)
        else:
            time_obj = time_step
        
        # Создаем объект циклона
        cyclone = Cyclone(
            latitude=latitude,
            longitude=longitude,
            time=time_obj,
            central_pressure=central_pressure,
            dataset=dataset.sel(time=time_step)
        )
        
        # Добавляем дополнительные свойства из кандидата
        for key, value in candidate.items():
            if key not in ['latitude', 'longitude', 'pressure']:
                setattr(cyclone, key, value)
        
        return cyclone
    
    def detect_all_timesteps(self, dataset: xr.Dataset) -> Dict[Any, List[Cyclone]]:
        """
        Обнаруживает циклоны для всех временных шагов в наборе данных.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            
        Возвращает:
            Словарь с временными шагами в качестве ключей и списками циклонов в качестве значений.
        """
        time_steps = dataset.time.values
        result = {}
        
        for time_step in time_steps:
            try:
                cyclones = self.detect(dataset, time_step)
                result[time_step] = cyclones
            except Exception as e:
                logger.error(f"Ошибка при обнаружении циклонов для временного шага {time_step}: {str(e)}")
                result[time_step] = []
        
        return result
    
    def set_criteria(self, criterion_names: List[str]) -> None:
        """
        Устанавливает активные критерии обнаружения циклонов.
        
        Аргументы:
            criterion_names: Список имен активных критериев.
        """
        self.criteria_manager.set_active_criteria(criterion_names)
        logger.info(f"Установлены активные критерии обнаружения: {criterion_names}")


class CycloneTracker:
    """
    Отслеживает циклоны на протяжении их жизненного цикла.
    
    Предоставляет методы для связывания обнаруженных циклонов в разные
    моменты времени в непрерывные треки, представляющие жизненный цикл циклона.
    """
    
    def __init__(self, max_distance: float = 500.0, 
                max_pressure_change: float = 15.0,
                max_time_gap: int = 12):
        """
        Инициализирует трекер циклонов.
        
        Аргументы:
            max_distance: Максимальное расстояние (км) для ассоциации циклонов между шагами.
            max_pressure_change: Максимальное изменение давления (гПа) для ассоциации.
            max_time_gap: Максимальный разрыв во времени (часы) для отслеживания.
        """
        self.max_distance = max_distance
        self.max_pressure_change = max_pressure_change
        self.max_time_gap = max_time_gap
        
        logger.info(f"Инициализирован трекер циклонов с параметрами: "
                   f"max_distance={max_distance} км, "
                   f"max_pressure_change={max_pressure_change} гПа, "
                   f"max_time_gap={max_time_gap} ч")
    
    def track(self, cyclone_sequences: Dict[Any, List[Cyclone]]) -> List[List[Cyclone]]:
        """
        Отслеживает циклоны между временными шагами.
        
        Аргументы:
            cyclone_sequences: Словарь с временными шагами и обнаруженными циклонами.
            
        Возвращает:
            Список треков циклонов (каждый трек - список объектов Cyclone).
        """
        try:
            # Проверяем входные данные
            if not cyclone_sequences:
                logger.warning("Пустой словарь циклонов для отслеживания")
                return []
            
            # Сортируем временные шаги
            time_steps = sorted(cyclone_sequences.keys())
            
            if len(time_steps) < 2:
                logger.warning("Недостаточно временных шагов для отслеживания")
                return [[c] for c in cyclone_sequences.get(time_steps[0], [])]
            
            # Словарь для хранения неназначенных циклонов для каждого шага
            unassigned = {ts: list(cyclones) for ts, cyclones in cyclone_sequences.items()}
            
            # Список для хранения треков
            tracks = []
            
            # Словарь для отслеживания активных треков
            active_tracks = {}  # track_id -> последний циклон в треке
            
            # Обрабатываем каждую пару последовательных временных шагов
            for i in range(len(time_steps) - 1):
                current_ts = time_steps[i]
                next_ts = time_steps[i + 1]
                
                # Вычисляем разницу во времени между шагами (в часах)
                time_diff = self._calculate_time_difference(current_ts, next_ts)
                
                # Пропускаем, если разрыв слишком большой
                if time_diff > self.max_time_gap:
                    logger.warning(f"Слишком большой временной разрыв между {current_ts} и {next_ts}: {time_diff} ч")
                    continue
                
                current_cyclones = unassigned[current_ts]
                next_cyclones = unassigned[next_ts]
                
                if not current_cyclones or not next_cyclones:
                    continue
                
                # Создаем и решаем задачу назначения
                assignments = self._solve_assignment_problem(
                    current_cyclones, next_cyclones, time_diff)
                
                # Применяем назначения
                for current_idx, next_idx in assignments:
                    current = current_cyclones[current_idx]
                    next_cyclone = next_cyclones[next_idx]
                    
                    # Получаем или создаем идентификатор трека
                    if hasattr(current, 'track_id') and current.track_id:
                        track_id = current.track_id
                    else:
                        track_id = f"track_{len(tracks)}"
                        current.track_id = track_id
                    
                    # Устанавливаем идентификатор трека для следующего циклона
                    next_cyclone.track_id = track_id
                    
                    # Обновляем данные следующего циклона
                    next_cyclone.age = current.age + time_diff
                    
                    # Обновляем или создаем трек
                    if track_id in active_tracks:
                        # Находим существующий трек
                        for track in tracks:
                            if track[-1].track_id == track_id:
                                track.append(next_cyclone)
                                break
                    else:
                        # Создаем новый трек
                        tracks.append([current, next_cyclone])
                        
                    # Обновляем активные треки
                    active_tracks[track_id] = next_cyclone
                    
                    # Удаляем назначенные циклоны из списков неназначенных
                    unassigned[current_ts].remove(current)
                    unassigned[next_ts].remove(next_cyclone)
            
            # Добавляем оставшиеся неназначенные циклоны как одноточечные треки
            for ts in time_steps:
                for cyclone in unassigned[ts]:
                    if not hasattr(cyclone, 'track_id') or not cyclone.track_id:
                        track_id = f"track_{len(tracks)}"
                        cyclone.track_id = track_id
                        tracks.append([cyclone])
            
            # Сортируем треки по продолжительности (от наиболее длинных к коротким)
            tracks.sort(key=len, reverse=True)
            
            logger.info(f"Сформировано {len(tracks)} треков циклонов")
            
            return tracks
            
        except Exception as e:
            error_msg = f"Ошибка при отслеживании циклонов: {str(e)}"
            logger.error(error_msg)
            raise TrackingError(error_msg)
    
    def _calculate_time_difference(self, time1: Any, time2: Any) -> float:
        """
        Вычисляет разницу между двумя временными точками в часах.
        
        Аргументы:
            time1: Первая временная точка.
            time2: Вторая временная точка.
            
        Возвращает:
            Разница во времени в часах.
        """
        # Преобразуем входные значения в pandas Timestamp
        t1 = pd.to_datetime(time1)
        t2 = pd.to_datetime(time2)
        
        # Вычисляем разницу в часах
        return (t2 - t1).total_seconds() / 3600
    
    def _solve_assignment_problem(self, current_cyclones: List[Cyclone], 
                                next_cyclones: List[Cyclone],
                                time_diff: float) -> List[Tuple[int, int]]:
        """
        Решает задачу назначения для связывания циклонов между временными шагами.
        
        Аргументы:
            current_cyclones: Список циклонов на текущем временном шаге.
            next_cyclones: Список циклонов на следующем временном шаге.
            time_diff: Разница во времени между шагами в часах.
            
        Возвращает:
            Список пар индексов (текущий, следующий) для связывания циклонов.
        """
        # Адаптируем максимальное расстояние в зависимости от временного интервала
        adjusted_max_distance = self.max_distance * (time_diff / 6.0) if time_diff > 6.0 else self.max_distance
        
        # Создаем матрицу стоимости
        n_current = len(current_cyclones)
        n_next = len(next_cyclones)
        
        cost_matrix = np.full((n_current, n_next), np.inf)
        
        for i, current in enumerate(current_cyclones):
            for j, next_cyclone in enumerate(next_cyclones):
                # Рассчитываем расстояние
                distance = self._calculate_distance(
                    current.latitude, current.longitude,
                    next_cyclone.latitude, next_cyclone.longitude
                )
                
                # Проверяем условия назначения
                if distance <= adjusted_max_distance:
                    # Рассчитываем давление и его изменение
                    pressure_diff = abs(current.central_pressure - next_cyclone.central_pressure)
                    
                    if pressure_diff <= self.max_pressure_change:
                        # Вычисляем стоимость назначения
                        # Более близкие циклоны и с меньшим изменением давления имеют меньшую стоимость
                        cost = distance / adjusted_max_distance + pressure_diff / self.max_pressure_change
                        cost_matrix[i, j] = cost
        
        # Находим оптимальные назначения с помощью венгерского алгоритма
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Фильтруем назначения с бесконечной стоимостью
        valid_assignments = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < np.inf:
                valid_assignments.append((i, j))
        
        return valid_assignments
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """
        Вычисляет расстояние между двумя точками на сфере (формула гаверсинуса).
        
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
    
    def filter_tracks(self, tracks: List[List[Cyclone]], 
                     min_duration: float = 6.0,
                     min_points: int = 3) -> List[List[Cyclone]]:
        """
        Фильтрует треки циклонов по минимальной продолжительности и числу точек.
        
        Аргументы:
            tracks: Список треков циклонов.
            min_duration: Минимальная продолжительность трека в часах.
            min_points: Минимальное количество точек в треке.
            
        Возвращает:
            Отфильтрованный список треков.
        """
        filtered_tracks = []
        
        for track in tracks:
            if len(track) < min_points:
                continue
                
            # Вычисляем продолжительность трека
            start_time = pd.to_datetime(track[0].time)
            end_time = pd.to_datetime(track[-1].time)
            duration = (end_time - start_time).total_seconds() / 3600
            
            if duration >= min_duration:
                filtered_tracks.append(track)
        
        logger.info(f"Отфильтровано {len(filtered_tracks)} треков из {len(tracks)} "
                   f"(мин. продолжительность: {min_duration} ч, мин. точек: {min_points})")
        
        return filtered_tracks
    
    def analyze_lifecycle(self, track: List[Cyclone]) -> Dict[str, Any]:
        """
        Анализирует жизненный цикл циклона.
        
        Аргументы:
            track: Трек циклона (список объектов Cyclone).
            
        Возвращает:
            Словарь с характеристиками жизненного цикла.
        """
        if not track:
            return {}
        
        # Извлекаем основные характеристики
        times = [pd.to_datetime(c.time) for c in track]
        pressures = [c.central_pressure for c in track]
        latitudes = [c.latitude for c in track]
        longitudes = [c.longitude for c in track]
        
        # Продолжительность жизненного цикла
        duration = (times[-1] - times[0]).total_seconds() / 3600
        
        # Минимальное давление и момент его достижения
        min_pressure = min(pressures)
        min_pressure_idx = pressures.index(min_pressure)
        min_pressure_time = times[min_pressure_idx]
        
        # Время от генезиса до достижения минимального давления
        time_to_min_pressure = (min_pressure_time - times[0]).total_seconds() / 3600
        
        # Скорость углубления (гПа/час)
        if min_pressure_idx > 0:
            deepening_rate = (pressures[0] - min_pressure) / max(1, time_to_min_pressure)
        else:
            deepening_rate = 0.0
        
        # Скорость движения
        speeds = []
        for i in range(1, len(track)):
            distance = self._calculate_distance(
                latitudes[i-1], longitudes[i-1],
                latitudes[i], longitudes[i]
            )
            time_diff = (times[i] - times[i-1]).total_seconds() / 3600
            if time_diff > 0:
                speed = distance / time_diff
                speeds.append(speed)
        
        mean_speed = np.mean(speeds) if speeds else 0.0
        max_speed = np.max(speeds) if speeds else 0.0
        
        # Общее пройденное расстояние
        total_distance = sum([
            self._calculate_distance(
                latitudes[i-1], longitudes[i-1],
                latitudes[i], longitudes[i]
            )
            for i in range(1, len(track))
        ])
        
        # Результат анализа
        result = {
            'track_id': track[0].track_id,
            'genesis_time': times[0],
            'lysis_time': times[-1],
            'duration_hours': duration,
            'min_pressure': min_pressure,
            'min_pressure_time': min_pressure_time,
            'time_to_min_pressure': time_to_min_pressure,
            'deepening_rate': deepening_rate,
            'mean_speed': mean_speed,
            'max_speed': max_speed,
            'total_distance': total_distance,
            'num_points': len(track)
        }
        
        return result