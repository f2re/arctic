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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import logging
from functools import partial
import inspect

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
            # Apply configuration parameters to criteria constructors
            for name in active_criteria:
                cls = self.criteria_manager.criteria.get(name)
                if cls:
                    sig = inspect.signature(cls.__init__)
                    settings = criteria_config.get(name, {})
                    # Remove 'enabled' flag
                    conf = {k: v for k, v in settings.items() if k != 'enabled'}
                    # Map config keys to __init__ params
                    params = {}
                    for param in list(sig.parameters)[1:]:  # skip 'self'
                        if param == 'min_latitude':
                            params['min_latitude'] = self.min_latitude
                        elif param in conf:
                            params[param] = conf[param]
                        elif param.endswith('_threshold') and 'threshold' in conf:
                            params[param] = conf['threshold']
                        elif param == 'pressure_level' and 'level' in conf:
                            params[param] = conf['level']
                    # Replace class with partial constructor
                    self.criteria_manager.criteria[name] = partial(cls, **params)
                    logger.debug(f"Configured criterion {name} with params {params}")
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
                    # logger.info(f"Collecting data for combined visualization for {criterion_name}")
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
                    elif criterion_name == 'pressure_minimum' and hasattr(criterion, 'pressure') or hasattr(criterion, 'pressure_field'):
                        # Collect data for pressure minimum visualization
                        lats = dataset.sel(time=time_step).latitude.values
                        lons = dataset.sel(time=time_step).longitude.values
                        # Filter to Arctic region for consistency
                        if hasattr(criterion, 'min_latitude'):
                            arctic_lats = lats[lats >= criterion.min_latitude]
                            if hasattr(criterion, 'pressure_field') and len(arctic_lats) != criterion.pressure_field.shape[0]:
                                logger.warning(f"Pressure minimum latitude dimension mismatch: {len(arctic_lats)} vs {criterion.pressure_field.shape[0]}")
                        
                        # Store the visualization data
                        criteria_viz_data['pressure_minimum'] = {
                            'pressure': getattr(criterion, 'pressure_field', getattr(criterion, 'pressure', None)),
                            'lats': arctic_lats if 'arctic_lats' in locals() else lats,
                            'lons': lons,
                            'threshold': getattr(criterion, 'pressure_threshold', None),
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

import numpy as np
import pandas as pd
import csv
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from pathlib import Path
from models.cyclone import Cyclone
from core.exceptions import TrackingError

logger = logging.getLogger(__name__)

class CycloneTracker:
    """
    Улучшенная система отслеживания треков арктических циклонов с кластеризацией.
    """

    def __init__(self,
                 max_distance: float = 500.0,  # км - уменьшено для более строгого трекинга
                 max_time_gap: float = 12.0,   # часы - уменьшено для непрерывности
                 max_pressure_change: float = 20.0,  # гПа - более реалистично
                 min_track_duration: float = 9.0,    # часы - минимум 3 временных шага по 3ч
                 min_track_points: int = 3,           # минимум 3 точки для трека
                 cluster_distance: float = 100.0,    # км - расстояние для кластеризации
                 cluster_pressure_diff: float = 5.0, # гПа - разность давления для кластеризации
                 max_cyclone_speed: float = 120.0,   # км/ч - максимальная физическая скорость
                 debug_save_csv: bool = False,
                 debug_dir: str = 'debug'):
        """
        Инициализация улучшенного трекера с кластеризацией.
        """
        self.max_distance = max_distance
        self.max_time_gap = max_time_gap
        self.max_pressure_change = max_pressure_change
        self.min_track_duration = min_track_duration
        self.min_track_points = min_track_points
        self.cluster_distance = cluster_distance
        self.cluster_pressure_diff = cluster_pressure_diff
        self.max_cyclone_speed = max_cyclone_speed
        
        # Debug опции
        self.debug_save_csv = debug_save_csv
        self.debug_dir = Path(debug_dir)
        if self.debug_save_csv:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
        self.track_counter = 0
        
        logger.info(f"Улучшенный трекер: max_dist={max_distance}км, cluster_dist={cluster_distance}км, "
                   f"min_points={min_track_points}, max_speed={max_cyclone_speed}км/ч")

    def cluster_cyclones_in_timestep(self, cyclones: List[Cyclone]) -> List[Cyclone]:
        """
        Кластеризует близкие циклоны в одном временном шаге для устранения дублей.
        
        Args:
            cyclones: Список циклонов в одном временном шаге
            
        Returns:
            Список представительных циклонов после кластеризации
        """
        if len(cyclones) <= 1:
            return cyclones
            
        # Фильтруем подозрительные точки на полюсе (90°N)
        realistic_cyclones = []
        polar_cyclones = []
        
        for cyclone in cyclones:
            if cyclone.latitude >= 89.5:  # Очень близко к полюсу
                polar_cyclones.append(cyclone)
            else:
                realistic_cyclones.append(cyclone)
        
        if polar_cyclones:
            logger.warning(f"Найдено {len(polar_cyclones)} циклонов на полюсе - возможная ошибка обнаружения")
            
        # Работаем только с реалистичными циклонами
        if len(realistic_cyclones) <= 1:
            return realistic_cyclones
            
        # Подготавливаем данные для кластеризации
        coordinates = np.array([[c.latitude, c.longitude] for c in realistic_cyclones])
        pressures = np.array([c.central_pressure for c in realistic_cyclones])
        
        # Вычисляем расстояния между всеми парами
        clusters = []
        used_indices = set()
        
        for i, cyclone in enumerate(realistic_cyclones):
            if i in used_indices:
                continue
                
            # Начинаем новый кластер
            cluster_indices = [i]
            used_indices.add(i)
            
            # Ищем близкие циклоны
            for j in range(i + 1, len(realistic_cyclones)):
                if j in used_indices:
                    continue
                    
                # Проверяем критерии объединения
                distance = self.calculate_distance(realistic_cyclones[i], realistic_cyclones[j])
                pressure_diff = abs(pressures[i] - pressures[j])
                
                if distance <= self.cluster_distance and pressure_diff <= self.cluster_pressure_diff:
                    cluster_indices.append(j)
                    used_indices.add(j)
            
            # Создаем представительный циклон
            if len(cluster_indices) == 1:
                clusters.append(realistic_cyclones[cluster_indices[0]])
            else:
                # Объединяем циклоны - выбираем самый интенсивный (низкое давление)
                cluster_cyclones = [realistic_cyclones[idx] for idx in cluster_indices]
                representative = min(cluster_cyclones, key=lambda c: c.central_pressure)
                
                # Добавляем информацию о кластеризации
                representative.clustered_count = len(cluster_cyclones)
                clusters.append(representative)
                
                logger.debug(f"Объединено {len(cluster_cyclones)} циклонов в один")
        
        logger.debug(f"Кластеризация: {len(realistic_cyclones)} -> {len(clusters)} циклонов")
        return clusters

    def calculate_distance(self, cyclone1: Cyclone, cyclone2: Cyclone) -> float:
        """
        Вычисляет расстояние между двумя циклонами в километрах.
        """
        lat1, lon1 = np.radians(cyclone1.latitude), np.radians(cyclone1.longitude)
        lat2, lon2 = np.radians(cyclone2.latitude), np.radians(cyclone2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371.0 * c  # Радиус Земли в км

    def calculate_time_difference(self, cyclone1: Cyclone, cyclone2: Cyclone) -> float:
        """
        Вычисляет разность времени между циклонами в часах.
        """
        def parse_time(time_obj):
            if isinstance(time_obj, str):
                return pd.to_datetime(time_obj)
            elif hasattr(time_obj, 'to_pydatetime'):
                return time_obj.to_pydatetime()
            else:
                return time_obj

        time1 = parse_time(cyclone1.time)
        time2 = parse_time(cyclone2.time)
        
        return abs((time2 - time1).total_seconds() / 3600.0)

    def calculate_cyclone_compatibility(self, cyclone1: Cyclone, cyclone2: Cyclone) -> Tuple[bool, float]:
        """
        Улучшенная проверка совместимости циклонов с физическими ограничениями.
        
        Returns:
            Tuple (is_compatible, cost) - совместимость и стоимость связи
        """
        # Временная разность
        time_diff = self.calculate_time_difference(cyclone1, cyclone2)
        if time_diff > self.max_time_gap:
            return False, np.inf
            
        # Пространственное расстояние
        distance = self.calculate_distance(cyclone1, cyclone2)
        if distance > self.max_distance:
            return False, np.inf
            
        # Физическая скорость перемещения
        if time_diff > 0:
            speed_kmh = distance / time_diff
            if speed_kmh > self.max_cyclone_speed:
                return False, np.inf
        
        # Изменение давления
        pressure_change = abs(cyclone1.central_pressure - cyclone2.central_pressure)
        if pressure_change > self.max_pressure_change:
            return False, np.inf
            
        # Вычисляем нормализованную стоимость
        distance_cost = distance / self.max_distance
        time_cost = time_diff / self.max_time_gap
        pressure_cost = pressure_change / self.max_pressure_change
        
        # Комбинированная стоимость с весами
        total_cost = 0.5 * distance_cost + 0.3 * time_cost + 0.2 * pressure_cost
        
        return True, total_cost

    def find_best_matches_hungarian(self, current_cyclones: List[Cyclone],
                                  previous_cyclones: List[Cyclone]) -> Dict[int, int]:
        """
        Использует венгерский алгоритм для оптимального назначения циклонов.
        """
        if not current_cyclones or not previous_cyclones:
            return {}

        n_current = len(current_cyclones)
        n_previous = len(previous_cyclones)

        # Создаем матрицу стоимостей
        cost_matrix = np.full((n_current, n_previous), np.inf)

        for i, current in enumerate(current_cyclones):
            for j, previous in enumerate(previous_cyclones):
                compatible, cost = self.calculate_cyclone_compatibility(previous, current)
                if compatible:
                    cost_matrix[i, j] = cost

        # Применяем венгерский алгоритм
        # Заменяем inf на большое число для решения
        finite_cost_matrix = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)
        
        try:
            row_indices, col_indices = linear_sum_assignment(finite_cost_matrix)
            
            # Фильтруем назначения с бесконечной стоимостью
            matches = {}
            for i, j in zip(row_indices, col_indices):
                if cost_matrix[i, j] < np.inf:
                    matches[i] = j
            
            return matches
            
        except Exception as e:
            logger.warning(f"Ошибка в венгерском алгоритме: {e}, используем жадный подход")
            return self.find_best_matches(current_cyclones, previous_cyclones)

    def find_best_matches(self, current_cyclones: List[Cyclone],
                         previous_cyclones: List[Cyclone]) -> Dict[int, int]:
        """
        Жадный алгоритм назначения как запасной вариант.
        """
        if not current_cyclones or not previous_cyclones:
            return {}

        # Создаем список всех возможных соединений
        connections = []
        for i, current in enumerate(current_cyclones):
            for j, previous in enumerate(previous_cyclones):
                compatible, cost = self.calculate_cyclone_compatibility(previous, current)
                if compatible:
                    connections.append((cost, i, j))

        # Сортируем по стоимости
        connections.sort()

        # Выбираем лучшие непересекающиеся соединения
        matches = {}
        used_previous = set()
        
        for cost, i, j in connections:
            if i not in matches and j not in used_previous:
                matches[i] = j
                used_previous.add(j)

        return matches

    def track(self, all_cyclones: Dict[Any, List[Cyclone]]) -> List[List[Cyclone]]:
        """
        Основной метод трекинга с кластеризацией и венгерским алгоритмом.
        """
        logger.info(f"Начинаем улучшенный трекинг для {len(all_cyclones)} временных шагов")
        
        # Этап 1: Кластеризация в каждом временном шаге
        clustered_cyclones = {}
        total_before = 0
        total_after = 0
        
        sorted_times = sorted(all_cyclones.keys())
        
        for time_step in sorted_times:
            original_cyclones = all_cyclones[time_step]
            total_before += len(original_cyclones)
            
            # Применяем кластеризацию
            clustered = self.cluster_cyclones_in_timestep(original_cyclones)
            clustered_cyclones[time_step] = clustered
            total_after += len(clustered)
            
            logger.debug(f"Время {time_step}: {len(original_cyclones)} -> {len(clustered)} циклонов")
        
        logger.info(f"Кластеризация завершена: {total_before} -> {total_after} циклонов")
        
        # Сохраняем результат кластеризации
        if self.debug_save_csv:
            clustered_points = []
            for cyclones in clustered_cyclones.values():
                clustered_points.extend(cyclones)
            self.save_points_to_csv(
                clustered_points,
                '01_clustered_points.csv',
                f"Точки после кластеризации ({len(clustered_points)} шт.)"
            )
        
        # Этап 2: Трекинг с венгерским алгоритмом
        active_tracks = []
        completed_tracks = []
        previous_cyclones = []

        for step_num, time_step in enumerate(sorted_times):
            current_cyclones = clustered_cyclones[time_step]
            
            if not current_cyclones:
                previous_cyclones = []
                continue

            logger.debug(f"Шаг {step_num + 1}/{len(sorted_times)}: {time_step} - {len(current_cyclones)} циклонов")

            if not previous_cyclones:
                # Первый шаг - создаем новые треки
                for cyclone in current_cyclones:
                    track_id = self._generate_track_id()
                    cyclone.track_id = track_id
                    active_tracks.append([cyclone])
                logger.info(f"Создано {len(current_cyclones)} новых треков")
            else:
                # Применяем венгерский алгоритм
                matches = self.find_best_matches_hungarian(current_cyclones, previous_cyclones)
                
                # Обновляем треки
                new_active_tracks = []
                matched_current = set()
                
                for prev_idx, curr_idx in matches.items():
                    # Находим трек с предыдущим циклоном
                    for track in active_tracks:
                        if track[-1] == previous_cyclones[curr_idx]: 
                            # Продолжаем трек
                            current_cyclones[prev_idx].track_id = track[-1].track_id 
                            track.append(current_cyclones[prev_idx]) 
                            new_active_tracks.append(track)
                            matched_current.add(prev_idx) 
                            break

                # Завершаем треки без продолжения
                for track in active_tracks:
                    if track not in new_active_tracks:
                        completed_tracks.append(track)

                # Создаем новые треки для несопоставленных циклонов
                new_tracks_count = 0
                for i, cyclone in enumerate(current_cyclones):
                    if i not in matched_current:
                        track_id = self._generate_track_id()
                        cyclone.track_id = track_id
                        new_active_tracks.append([cyclone])
                        new_tracks_count += 1

                active_tracks = new_active_tracks
                logger.debug(f"Назначений: {len(matches)}, новых треков: {new_tracks_count}")

            previous_cyclones = current_cyclones

        # Завершаем оставшиеся треки
        completed_tracks.extend(active_tracks)
        
        logger.info(f"Трекинг завершен: создано {len(completed_tracks)} треков")
        
        # Debug сохранение
        if self.debug_save_csv:
            all_tracked_points = []
            for track in completed_tracks:
                all_tracked_points.extend(track)
            self.save_points_to_csv(
                all_tracked_points,
                '02_tracks_after_hungarian.csv',
                f"Все точки после трекинга ({len(all_tracked_points)} шт.) в {len(completed_tracks)} треках"
            )
        
        return completed_tracks

    def filter_tracks(self, tracks: List[List[Cyclone]],
                     min_duration: float = None,
                     min_points: int = None) -> List[List[Cyclone]]:
        """
        Улучшенная фильтрация треков с физическими критериями.
        """
        if min_duration is None:
            min_duration = self.min_track_duration
        if min_points is None:
            min_points = self.min_track_points

        filtered_tracks = []
        
        filter_stats = {
            'original': len(tracks),
            'too_few_points': 0,
            'too_short_duration': 0,
            'unrealistic_movement': 0,
            'pressure_unrealistic': 0,
            'passed': 0
        }

        for track in tracks:
            if len(track) < min_points:
                filter_stats['too_few_points'] += 1
                continue

            # Сортируем по времени
            track_sorted = sorted(track, key=lambda c: c.time)

            # Проверяем длительность
            duration_hours = self._calculate_track_duration(track_sorted)
            if duration_hours < min_duration:
                filter_stats['too_short_duration'] += 1
                continue

            # Проверяем реалистичность движения
            if not self._is_movement_realistic(track_sorted):
                filter_stats['unrealistic_movement'] += 1
                continue

            # Проверяем реалистичность давления
            if not self._is_pressure_realistic(track_sorted):
                filter_stats['pressure_unrealistic'] += 1
                continue

            filtered_tracks.append(track_sorted)
            filter_stats['passed'] += 1

        logger.info(f"Фильтрация: {filter_stats}")
        
        # Debug сохранение
        if self.debug_save_csv:
            final_points = []
            for track in filtered_tracks:
                final_points.extend(track)
            self.save_points_to_csv(
                final_points,
                '03_final_filtered_tracks.csv',
                f"Финальные треки после фильтрации ({len(final_points)} шт.) в {len(filtered_tracks)} треках"
            )

        return filtered_tracks

    def _is_movement_realistic(self, track: List[Cyclone]) -> bool:
        """Проверяет реалистичность движения циклона."""
        for i in range(len(track) - 1):
            curr = track[i]
            next_cyclone = track[i + 1]
            
            distance = self.calculate_distance(curr, next_cyclone)
            time_diff = self.calculate_time_difference(curr, next_cyclone)
            
            if time_diff > 0:
                speed = distance / time_diff
                if speed > self.max_cyclone_speed:
                    return False
        return True

    def _is_pressure_realistic(self, track: List[Cyclone]) -> bool:
        """Проверяет реалистичность изменений давления."""
        pressures = [c.central_pressure for c in track]
        
        # Проверяем диапазон
        min_pressure = min(pressures)
        max_pressure = max(pressures)
        
        if min_pressure < 900 or max_pressure > 1050:  # Нереалистичные значения
            return False
        
        # Проверяем скорость изменения давления
        max_change_rate = 10.0  # гПа/час
        
        for i in range(len(track) - 1):
            curr = track[i]
            next_cyclone = track[i + 1]
            
            pressure_change = abs(curr.central_pressure - next_cyclone.central_pressure)
            time_diff = self.calculate_time_difference(curr, next_cyclone)
            
            if time_diff > 0:
                change_rate = pressure_change / time_diff
                if change_rate > max_change_rate:
                    return False
        
        return True

    def _calculate_track_duration(self, track: List[Cyclone]) -> float:
        """Вычисляет длительность трека в часах."""
        if len(track) < 2:
            return 0.0
        
        start_time = track[0].time
        end_time = track[-1].time
        
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        return (end_time - start_time).total_seconds() / 3600.0

    def _generate_track_id(self) -> str:
        """Генерирует уникальный ID для трека."""
        self.track_counter += 1
        return f"ARCTIC_TRACK_{self.track_counter:06d}"

    def save_points_to_csv(self, points: List[Cyclone], filename: str, description: str = ""):
        """
        Сохраняет точки циклонов в CSV файл для отладки.
        """
        if not self.debug_save_csv:
            return

        filepath = self.debug_dir / filename

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Заголовок с описанием
                writer.writerow([f"# {description}"])
                writer.writerow([f"# Количество точек: {len(points)}"])
                writer.writerow([f"# Время создания: {datetime.now()}"])
                writer.writerow([]) # Пустая строка
                
                # Заголовки колонок
                writer.writerow([
                    'track_id', 'latitude', 'longitude', 'time',
                    'central_pressure', 'clustered_count', 'step_number'
                ])
                
                # Данные точек
                for i, cyclone in enumerate(points):
                    track_id = getattr(cyclone, 'track_id', '')
                    clustered_count = getattr(cyclone, 'clustered_count', 1)
                    
                    writer.writerow([
                        track_id,
                        cyclone.latitude,
                        cyclone.longitude,
                        cyclone.time,
                        cyclone.central_pressure,
                        clustered_count,
                        i + 1
                    ])
                    
            logger.info(f"Сохранено {len(points)} точек в {filepath} - {description}")
        except Exception as e:
            logger.error(f"Ошибка сохранения debug CSV {filepath}: {str(e)}")


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