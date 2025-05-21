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
import os

from core.exceptions import DetectionError, TrackingError
from models.cyclone import Cyclone, CycloneType, CycloneParameters
from .criteria import CriteriaManager, BaseCriterion
from .validators import DetectionValidator
from visualization.criteria import plot_laplacian_field, plot_vorticity_field, plot_pressure_field, plot_wind_field, plot_closed_contour_field, plot_combined_criteria

# Инициализация логгера
logger = logging.getLogger(__name__)

class CycloneDetector:
    """
    Обнаруживает арктические мезоциклоны на основе настраиваемых критериев.
    
    Предоставляет методы для поиска циклонов в наборах метеорологических данных
    с использованием гибкой системы критериев обнаружения.
    """
    
    def __init__(self, min_latitude: float = 65.0, config: Optional[Dict] = None, debug_plot: bool = False):
        import logging
        self._instance_id = id(self)
        logging.getLogger(__name__).debug(f"[CycloneDetector __init__] Instance id: {self._instance_id}, min_latitude={min_latitude}, debug_plot={debug_plot}")
        """
        Инициализирует детектор циклонов с указанной минимальной широтой.
        
        Аргументы:
            min_latitude: Минимальная широта для Арктического региона (по умолчанию 65°N).
            config: Конфигурация обнаружения циклонов из config.yaml
            debug_plot: Флаг для визуализации полей критериев обнаружения (по умолчанию False)
        """
        self.min_latitude = min_latitude
        self.criteria_manager = CriteriaManager()
        self.validator = DetectionValidator()
        self.config = config
        self.debug_plot = debug_plot
        
        # Если конфигурация предоставлена, устанавливаем активные критерии из конфигурации
        if self.config:
            self._register_default_criteria()
            self._configure_criteria_from_config()
        else:
            # Регистрируем стандартные критерии обнаружения
            self._register_default_criteria()
        
        logger.info(f"Инициализирован детектор циклонов с минимальной широтой {min_latitude}°N, debug_plot={debug_plot}")
    
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
        else:
            logger.warning("В конфигурации не найдены активные критерии, используем стандартные")
    
    def detect(self, dataset: xr.Dataset, time_step: Any) -> List[Cyclone]:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[CycloneDetector.detect] Instance id: {getattr(self, '_instance_id', id(self))}, time_step={time_step}")
        if dataset is None or len(dataset.variables) == 0:
            logger.warning(f"[CycloneDetector.detect] Called with empty or None dataset at time_step={time_step} (instance id: {getattr(self, '_instance_id', id(self))})")
            return []
        # Check for critical variables
        required_vars = ['vorticity', 'mean_sea_level_pressure', 'u_component_of_wind', 'v_component_of_wind']
        missing = [var for var in required_vars if var not in dataset]
        if missing:
            logger.warning(f"[CycloneDetector.detect] Dataset missing required variables {missing} at time_step={time_step} (instance id: {getattr(self, '_instance_id', id(self))})")        
        # try:
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
        candidates = self._apply_detection_criteria(arctic_data, time_step)
        
        # --- DEBUG: Print all candidate values before filtering/creation ---
        logger.debug(f"[CycloneDetector.detect] Number of candidates: {len(candidates)}")
        for i, candidate in enumerate(candidates):
            logger.debug(f"Candidate {i}: {candidate}")
            for key, value in candidate.items():
                logger.debug(f"  {key}: {value} (type={type(value)})")
            # Check for None in critical fields
            critical_fields = ['latitude', 'longitude', 'pressure', 'vorticity', 'wind_speed']
            for field in critical_fields:
                if field in candidate and candidate[field] is None:
                    logger.error(f"Candidate {i} has None for {field}, skipping this candidate!")
                    candidate['skip_due_to_none'] = True
        # --- END DEBUG ---
        
        cyclones = []
        for i, candidate in enumerate(candidates):
            if candidate.get('skip_due_to_none'):
                continue
            try:
                cyclone = self._create_cyclone(candidate, arctic_data, time_step)
                cyclones.append(cyclone)
            except Exception as e:
                logger.error(f"Error creating cyclone from candidate {i}: {e}\nCandidate: {candidate}")

        logger.info(f"Обнаружено {len(cyclones)} циклонов для временного шага {time_step}")
        return cyclones
            
        # except Exception as e:
        #     error_msg = f"Ошибка при обнаружении циклонов: {str(e)}"
        #     logger.error(error_msg)
        #     raise DetectionError(error_msg)
    
    def _apply_detection_criteria(self, dataset: xr.Dataset, time_step: Any) -> List[Dict]:
        """
        Применяет набор критериев обнаружения циклонов.
        
        Аргументы:
            dataset: Набор метеорологических данных.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список кандидатов в циклоны (словари с координатами и свойствами).
        """
        logger.info(f"Применение критериев обнаружения для временного шага: {time_step}")
        
        # Получаем список активных критериев
        active_criteria = self.criteria_manager.get_active_criteria()
        
        if not active_criteria:
            logger.warning("Нет активных критериев обнаружения циклонов")
            return []
        
        # Создаем директорию для отладочных графиков, если включен режим отладки
        output_dir = None
        timestamp = format_timestep(time_step)
        if self.debug_plot:
            output_dir = Path(f"output/plots/criteria/{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Создана директория для отладочных графиков: {output_dir}")
        
        # Применяем каждый критерий и собираем кандидатов
        all_candidates = []
        
        # Dictionary to collect visualization data for combined plot
        criteria_viz_data = {}
        
        for criterion_name, criterion in active_criteria.items():
            logger.debug(f"Применяем критерий: {criterion_name}")
            
            try:
                # Если включен режим отладки, но мы хотим комбинированную визуализацию,
                # то отключаем отдельную визуализацию для каждого критерия
                individual_debug_plot = self.debug_plot
                
                # Применяем критерий с включенной отладочной визуализацией, если требуется
                candidates = criterion.apply(
                    dataset=dataset, 
                    time_step=time_step,
                    debug_plot=individual_debug_plot,
                    output_dir=output_dir
                )
                
                # Добавляем имя критерия к каждому кандидату
                for candidate in candidates:
                    if 'criterion' not in candidate:
                        candidate['criterion'] = criterion_name
                
                logger.debug(f"Критерий {criterion_name} нашел {len(candidates)} кандидатов")
                all_candidates.extend(candidates)
                
                # Collect data for combined visualization if debug_plot is enabled
                if self.debug_plot:
                    # Extract data for visualization based on criterion type
                    if criterion_name == 'pressure_laplacian' and hasattr(criterion, 'laplacian_field'):
                        # Get the actual dimensions used in the criterion
                        lats = dataset.sel(time=time_step).latitude.values
                        lons = dataset.sel(time=time_step).longitude.values
                        
                        # Filter to Arctic region if needed to match the data dimensions
                        if hasattr(criterion, 'min_latitude'):
                            arctic_lats = lats[lats >= criterion.min_latitude]
                            if len(arctic_lats) != criterion.laplacian_field.shape[0]:
                                logger.warning(f"Latitude dimension mismatch: {len(arctic_lats)} vs {criterion.laplacian_field.shape[0]}")
                        
                        # Store the visualization data ensuring dimensions match
                        criteria_viz_data['pressure_laplacian'] = {
                            'laplacian': criterion.laplacian_field,
                            'lats': arctic_lats if 'arctic_lats' in locals() else lats,
                            'lons': lons,
                            'threshold': criterion.laplacian_threshold,
                            'time_step': time_step,
                            'output_dir': output_dir
                        }
                    elif criterion_name == 'vorticity' and hasattr(criterion, 'vorticity_field'):
                        # Get the actual dimensions used in the criterion
                        lats = dataset.sel(time=time_step).latitude.values
                        lons = dataset.sel(time=time_step).longitude.values
                        
                        # Filter to Arctic region if needed to match the data dimensions
                        if hasattr(criterion, 'min_latitude'):
                            arctic_lats = lats[lats >= criterion.min_latitude]
                            if len(arctic_lats) != criterion.vorticity_field.shape[0]:
                                logger.warning(f"Vorticity latitude dimension mismatch: {len(arctic_lats)} vs {criterion.vorticity_field.shape[0]}")
                        
                        # Store the visualization data ensuring dimensions match
                        criteria_viz_data['vorticity'] = {
                            'vorticity': criterion.vorticity_field,
                            'lats': arctic_lats if 'arctic_lats' in locals() else lats,
                            'lons': lons,
                            'threshold': criterion.vorticity_threshold,
                            'time_step': time_step,
                            'output_dir': output_dir
                        }
                    elif criterion_name == 'wind_threshold' and hasattr(criterion, 'u_data') and hasattr(criterion, 'v_data'):
                        # Get the actual dimensions used in the criterion
                        lats = dataset.sel(time=time_step).latitude.values
                        lons = dataset.sel(time=time_step).longitude.values
                        
                        # Filter to Arctic region if needed to match the data dimensions
                        if hasattr(criterion, 'min_latitude'):
                            arctic_lats = lats[lats >= criterion.min_latitude]
                            if len(arctic_lats) != criterion.u_data.shape[0]:
                                logger.warning(f"Wind latitude dimension mismatch: {len(arctic_lats)} vs {criterion.u_data.shape[0]}")
                        
                        # Store the visualization data ensuring dimensions match
                        criteria_viz_data['wind_threshold'] = {
                            'u_wind': criterion.u_data,
                            'v_wind': criterion.v_data,
                            'lats': arctic_lats if 'arctic_lats' in locals() else lats,
                            'lons': lons,
                            'threshold': criterion.min_speed,
                            'time_step': time_step,
                            'output_dir': output_dir
                        }
                    elif criterion_name == 'closed_contour' and hasattr(criterion, 'pressure_field') and hasattr(criterion, 'contour_mask'):
                        # Get the actual dimensions used in the criterion
                        lats = dataset.sel(time=time_step).latitude.values
                        lons = dataset.sel(time=time_step).longitude.values
                        
                        # Filter to Arctic region if needed to match the data dimensions
                        if hasattr(criterion, 'min_latitude'):
                            arctic_lats = lats[lats >= criterion.min_latitude]
                            if len(arctic_lats) != criterion.pressure_field.shape[0]:
                                logger.warning(f"Closed contour latitude dimension mismatch: {len(arctic_lats)} vs {criterion.pressure_field.shape[0]}")
                        
                        # Store the visualization data ensuring dimensions match
                        criteria_viz_data['closed_contour'] = {
                            'pressure': criterion.pressure_field,
                            'contour_mask': criterion.contour_mask,
                            'lats': arctic_lats if 'arctic_lats' in locals() else lats,
                            'lons': lons,
                            'time_step': time_step,
                            'output_dir': output_dir
                        }
                
            except Exception as e:
                logger.error(f"Ошибка при применении критерия {criterion_name}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Create combined visualization if debug_plot is enabled and we have data to visualize
        if self.debug_plot:
            # Check if any criteria were not added to visualization data
            active_criteria_names = set(active_criteria.keys())
            visualized_criteria = set(criteria_viz_data.keys())
            missing_criteria = active_criteria_names - visualized_criteria
            
            if missing_criteria:
                logger.warning(f"Some active criteria were not visualized: {missing_criteria}")
                
            # Log which criteria will be visualized
            logger.info(f"Visualizing criteria: {list(criteria_viz_data.keys())}")
            
            # Create the combined visualization
            try:
                if criteria_viz_data:  # Only proceed if we have data to visualize
                    logger.info(f"Creating combined visualization for {len(criteria_viz_data)} criteria")
                    combined_output = plot_combined_criteria(
                        criteria_data=criteria_viz_data,
                        time_step=time_step,
                        output_dir=output_dir
                    )
                    logger.info(f"Saved combined criteria visualization to {combined_output}")
                else:
                    logger.warning("No criteria data available for visualization")
            except Exception as e:
                logger.error(f"Error creating combined visualization: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Проверяем результаты
        if not all_candidates:
            logger.info(f"Не найдено кандидатов в циклоны для временного шага {time_step}")
            return []
        
        logger.info(f"Найдено {len(all_candidates)} кандидатов в циклоны для временного шага {time_step}")
        return all_candidates
    
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
        # DEBUG: Log all candidate attributes before creation
        logger = logging.getLogger(__name__)
        logger.debug(f"[CycloneDetector._create_cyclone] Creating cyclone from candidate: {candidate}")
        # Validate all required attributes are not None
        required_fields = ['latitude', 'longitude']
        for field in required_fields:
            if field not in candidate or candidate[field] is None:
                logger.error(f"Cannot create cyclone: candidate missing or has None for required field '{field}'. Candidate: {candidate}")
                raise ValueError(f"Cannot create cyclone: missing or None '{field}'")
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
        
        # Ensure vorticity data is available in the dataset
        time_data = dataset.sel(time=time_step)
        
        # Check if vorticity data is missing and try to add it
        vorticity_vars = ['vorticity', 'vo', 'relative_vorticity']
        has_vorticity = any(var in time_data for var in vorticity_vars)
        
        if not has_vorticity:
            try:
                # Try to get vorticity criterion to calculate vorticity
                vorticity_criterion = self.criteria_manager.get_criterion('vorticity')
                if vorticity_criterion:
                    logger.info("Adding vorticity data to dataset for cyclone creation")
                    # Apply the criterion to calculate vorticity but don't use it for detection
                    vorticity_criterion._calculate_vorticity(
                        dataset=time_data, 
                        time_step=time_step,
                        u_var='u', 
                        v_var='v'
                    )
            except Exception as e:
                logger.warning(f"Could not add vorticity data: {str(e)}")
        
        # Создаем объект циклона
        cyclone = Cyclone(
            latitude=latitude,
            longitude=longitude,
            time=time_obj,
            central_pressure=central_pressure,
            dataset=time_data
        )
        
        # Set detector instance for parameter calculation
        cyclone.detector = self
        
        # Добавляем дополнительные свойства из кандидата
        for key, value in candidate.items():
            if key not in ['latitude', 'longitude', 'pressure']:
                setattr(cyclone, key, value)
        
        # If candidate has vorticity_field, store it in cyclone parameters
        if 'vorticity_field' in candidate:
            if not hasattr(cyclone, 'parameters'):
                from models.parameters import CycloneParameters
                cyclone.parameters = CycloneParameters(
                    central_pressure=central_pressure,
                    vorticity_850hPa=None,
                    max_wind_speed=None,
                    radius=None
                )
            cyclone.parameters.vorticity_850hPa = np.max(candidate['vorticity_field'])
        
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
    
    _active_criteria_set = False  # class-level guard
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
                    # Validate indices to prevent index out of range errors
                    if current_idx >= len(current_cyclones) or next_idx >= len(next_cyclones):
                        logger.warning(f"Invalid index pair: current_idx={current_idx}, next_idx={next_idx}, "  
                                      f"list lengths: current={len(current_cyclones)}, next={len(next_cyclones)}")
                        continue
                        
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
    


def format_timestep(time_step):
    # Проверяем тип данных
    if isinstance(time_step, np.datetime64):
        # Метод 1: Преобразование через строку (простой и не требует доп. библиотек)
        dt_str = str(time_step)
        if 'T' in dt_str:
            date_part = dt_str.split('T')[0]  # Получаем '2010-01-01'
            time_part = dt_str.split('T')[1]  # Получаем '00:00:00.000000000'
            hour_part = time_part.split(':')[0]  # Получаем '00'
            return f"{date_part}_{hour_part}"
        
        # Метод 2: Через datetime (если первый метод не сработал)
        try:
            dt_obj = time_step.astype('datetime64[s]').item()  # Преобразуем в Python datetime
            return dt_obj.strftime('%Y-%m-%d_%H')
        except (AttributeError, ValueError):
            pass
        
    # Обработка других типов временных меток
    elif hasattr(time_step, 'strftime'):  # Для объектов datetime
        return time_step.strftime('%Y-%m-%d_%H')
    
    elif isinstance(time_step, str):  # Для строковых представлений
        if 'T' in time_step:  # ISO формат
            date_part = time_step.split('T')[0]
            time_part = time_step.split('T')[1]
            hour_part = time_part[:2] if ':' in time_part else time_part
            return f"{date_part}_{hour_part}"
    
    # Fallback for other cases
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return timestamp