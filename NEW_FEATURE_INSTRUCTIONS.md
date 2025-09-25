# Инструкции по добавлению новых функций в систему ArcticCyclone

Этот документ описывает шаги, необходимые для добавления новых параметров и функций в систему обнаружения и анализа арктических мезоциклонов.

## Общая архитектура системы

Система ArcticCyclone использует модульную архитектуру, где каждая функция реализована в отдельных модулях:

- `models/parameters.py` - определение параметров циклона
- `models/cyclone.py` - основной класс циклона
- `core/config.py` - обработка конфигурации
- `config.yaml` - файл конфигурации
- `detection/` - алгоритмы обнаружения
- `visualization/` - визуализация параметров
- `detection/criteria/` - критерии обнаружения

## Шаги для добавления нового параметра

### 1. Добавление параметра в модель

**Файл**: `models/parameters.py`

1. Добавьте новый параметр в класс `CycloneParameters`:
   ```python
   class CycloneParameters:
       # ... существующие параметры ...
       new_parameter: Optional[float] = None  # описание единиц измерения
   ```

2. В методе `__post_init__` добавьте валидацию, если это необходимо:
   ```python
   def __post_init__(self):
       # ... существующая валидация ...
       if self.new_parameter is not None:
           # добавьте проверки диапазона, если нужно
           assert -9999 < self.new_parameter < 9999, "new_parameter вне допустимого диапазона"
   ```

3. В методе `to_dict()` добавьте преобразование:
   ```python
   def to_dict(self) -> Dict[str, Any]:
       result = {
           'central_pressure': self.central_pressure
       }
       # ... существующие параметры ...
       if self.new_parameter is not None:
           result['new_parameter'] = self.new_parameter
       return result
   ```

4. В методе `from_dict()` добавьте восстановление:
   ```python
   @classmethod
   def from_dict(cls, data: Dict[str, Any]) -> 'CycloneParameters':
       # ... существующий код ...
       if 'new_parameter' in data:
           params.new_parameter = data['new_parameter']
       return params
   ```

### 2. Обновление модели циклона

**Файл**: `models/cyclone.py`

1. В методе `_calculate_parameters()` добавьте расчет нового параметра:
   ```python
   def _calculate_parameters(self, dataset: xr.Dataset, detector: 'CycloneDetector') -> Dict[str, Any]:
       # ... существующий код ...
       active_criteria = detector.criteria_manager.get_active_criteria()
       
       # Добавьте логику для нового критерия/параметра
       if 'new_criterion' in active_criteria or 'new_parameter_logic' in some_condition:
           new_param_value = self._calculate_new_parameter(region)
           calculated_params['new_parameter'] = new_param_value
   ```

2. Создайте метод для расчета нового параметра:
   ```python
   def _calculate_new_parameter(self, region: xr.Dataset) -> Optional[float]:
       """
       Рассчитывает новый параметр на основе региона данных.
       
       Args:
           region: Набор метеорологических данных для региона циклона.
           
       Returns:
           Значение нового параметра или None, если расчет невозможен.
       """
       try:
           # Логика расчета нового параметра
           # Проверьте наличие необходимых переменных в регионе
           if 'necessary_variable' not in region:
               logger.warning("Необходимая переменная отсутствует в регионе")
               return None
           
           # Выполните вычисления
           result = some_calculation(region['necessary_variable'])
           return float(result)
           
       except Exception as e:
           logger.warning(f"Ошибка при расчете нового параметра: {str(e)}")
           return None
   ```

### 3. Обновление конфигурации

**Файл**: `config.yaml`

1. Добавьте новый параметр в соответствующую секцию:
   ```yaml
   detection:
     criteria:
       new_criterion:
         enabled: true
         weight: 0.25
         threshold: значение_порога
         additional_param: значение
   ```

**Файл**: `core/config.py`

2. Конфигурация автоматически читается из YAML файла, но при необходимости можно добавить валидацию:
   ```python
   class ConfigManager:
       # ... существующий код ...
       
       def get_new_criterion_config(self) -> Dict[str, Any]:
           """Получает конфигурацию для нового критерия."""
           try:
               return self.config['detection']['criteria']['new_criterion']
           except KeyError:
               # Возвращаем параметры по умолчанию
               return {
                   'enabled': False,
                   'threshold': 0.0,
                   'additional_param': 'default_value'
               }
   ```

### 4. Обновление системы обнаружения

**Файл**: `detection/criteria/criterion_name.py` или создайте новый файл

1. Если нужен новый критерий обнаружения, создайте файл в `detection/criteria/`:
   ```python
   # detection/criteria/new_criterion.py
   import xarray as xr
   import numpy as np
   from typing import Dict, List, Any, Optional
   import logging
   import scipy.ndimage as ndimage

   from . import BaseCriterion
   from core.exceptions import DetectionError

   logger = logging.getLogger(__name__)

   class NewCriterion(BaseCriterion):
       def __init__(self, min_latitude: float = 70.0,
                   new_criterion_threshold: float = 1.0,
                   additional_param: float = 0.5):
           self.min_latitude = min_latitude
           self.new_criterion_threshold = new_criterion_threshold
           self.additional_param = additional_param

       def apply(self, dataset: xr.Dataset, time_step: Any, debug_plot: bool = False, output_dir: Optional[str] = None) -> List[Dict]:
           try:
               # Логика критерия
               candidates = []
               # ... реализация ...
               return candidates
           except Exception as e:
               error_msg = f"Ошибка при применении нового критерия: {str(e)}"
               logger.error(error_msg)
               raise DetectionError(error_msg)
   ```

2. Зарегистрируйте новый критерий в `detection/tracker.py`:
   ```python
   def _register_default_criteria(self):
       # ... существующие критерии ...
       from detection.criteria import NewCriterion
       self.criteria_manager.register_criterion('new_criterion', NewCriterion)
   ```

3. Обновите метод `_configure_criteria_from_config()` для обработки нового критерия:
   ```python
   def _configure_criteria_from_config(self) -> None:
       # ... существующий код ...
       for criterion_name, settings in criteria_config.items():
           # ... существующий код ...
           elif criterion_name == 'new_criterion':
               # Обработка специфичных для нового критерия параметров
               params = {}
               if 'new_criterion_threshold' in settings:
                   params['new_criterion_threshold'] = settings['new_criterion_threshold']
               if 'additional_param' in settings:
                   params['additional_param'] = settings['additional_param']
               self.criteria_manager.criteria[new_criterion] = partial(cls, **params)
   ```

### 5. Обновление визуализации

**Файл**: `visualization/parameters.py`

1. В функции `plot_cyclone_parameters()` добавьте поддержку нового параметра:
   ```python
   def plot_cyclone_parameters(cyclone: Cyclone,
                            parameters: List[str] = None,
                            figsize: Tuple[float, float] = (10, 6)) -> Tuple[Figure, List[Axes]]:
       # ... существующий код ...
       param_labels = {
           # ... существующие параметры ...
           'new_parameter': 'Название нового параметра',
       }
       param_units = {
           # ... существующие параметры ...
           'new_parameter': 'единица_измерения',
       }
       # ... остальная логика ...
   ```

2. Добавьте новый параметр в функцию `plot_parameter_correlation()`:
   ```python
   def plot_parameter_correlation(cyclones: List[Cyclone],
                               x_param: str,
                               y_param: str,
                               color_by: Optional[str] = None,
                               figsize: Tuple[float, float] = (10, 8)) -> Tuple[Figure, Axes]:
       # ... существующий код ...
       param_labels = {
           # ... существующие параметры ...
           'new_parameter': 'Название нового параметра (единица_измерения)',
       }
       # ... остальная логика ...
   ```

3. Добавьте новый параметр в функцию `plot_parameter_histogram()`:
   ```python
   def plot_parameter_histogram(cyclones: List[Cyclone],
                             parameter: str,
                             bins: int = 20,
                             kde: bool = True,
                             color: str = 'blue',
                             figsize: Tuple[float, float] = (10, 6)) -> Tuple[Figure, Axes]:
       # ... существующий код ...
       param_labels = {
           # ... существующие параметры ...
           'new_parameter': 'Название нового параметра (единица_измерения)',
       }
       # ... остальная логика ...
   ```

4. Добавьте новый параметр в функцию `plot_parameter_evolution()`:
   ```python
   def plot_parameter_evolution(cyclone_track: List[Cyclone],
                             parameters: List[str] = None,
                             figsize: Tuple[float, float] = (12, 8)) -> Tuple[Figure, List[Axes]]:
       # ... существующий код ...
       param_labels = {
           # ... существующие параметры ...
           'new_parameter': 'Название нового параметра',
       }
       param_units = {
           # ... существующие параметры ...
           'new_parameter': 'единица_измерения',
       }
       # ... остальная логика ...
   ```

### 6. Обновление многопараметрической фильтрации

**Файл**: `detection/multi_criteria_validator.py` (или соответствующий файл)

1. Добавьте логику для оценки нового параметра в комбинации с другими:
   ```python
   def validate_candidate(self, candidate, dataset) -> Tuple[bool, float, Dict[str, float]]:
       # ... существующий код ...
       individual_scores = {}
       
       # ... существующие оценки ...
       
       # Оценка нового параметра
       if 'new_parameter' in candidate and hasattr(self, 'new_criterion_threshold'):
           new_score = self._calculate_new_parameter_score(candidate['new_parameter'])
           individual_scores['new_parameter'] = new_score
       
       # ... остальная логика ...
       return is_valid, final_score, individual_scores
   ```

2. Добавьте метод для оценки нового параметра:
   ```python
   def _calculate_new_parameter_score(self, new_param_value) -> float:
       """
       Рассчитывает оценку для нового параметра.
       
       Args:
           new_param_value: Значение нового параметра.
           
       Returns:
           Оценка от 0 до 1, где 1 - идеальное значение.
       """
       if new_param_value is None:
           return 0.0
       
       # Пример: оценка на основе порога
       threshold = getattr(self, 'new_criterion_threshold', 0.0)
       score = min(1.0, max(0.0, abs(new_param_value - threshold) / threshold))
       return score
   ```

### 7. Обновление отслеживания треков

**Файл**: `detection/tracker.py`

1. В функции `calculate_cyclone_compatibility()` добавьте логику для нового параметра:
   ```python
   def calculate_cyclone_compatibility(self, cyclone1: Cyclone, cyclone2: Cyclone) -> Tuple[bool, float]:
       # ... существующая проверка ...
       
       # Проверка совместимости по новому параметру (если применимо)
       if hasattr(cyclone1.parameters, 'new_parameter') and hasattr(cyclone2.parameters, 'new_parameter'):
           param_diff = abs(cyclone1.parameters.new_parameter - cyclone2.parameters.new_parameter)
           if param_diff > max_allowed_difference:
               return False, np.inf
       
       # ... остальная логика ...
       
       return True, total_cost
   ```

### 8. Обновление документации

1. Добавьте описание нового параметра в документацию:
   - `docs/scientific_methods.md` - научное обоснование
   - `docs/api_reference.md` - техническая документация
   - `README.md` - краткое описание при необходимости

### 9. Тестирование

1. Создайте тесты для новой функциональности:
   - `tests/test_detection.py` - тесты обнаружения
   - `tests/test_parameters.py` - тесты параметров
   - `tests/test_visualization.py` - тесты визуализации

2. Пример теста для нового параметра:
   ```python
   # tests/test_parameters.py
   import pytest
   import numpy as np
   from models.parameters import CycloneParameters
   
   def test_new_parameter():
       # Создание параметров с новым значением
       params = CycloneParameters(
           central_pressure=1000.0,
           new_parameter=5.0
       )
       
       # Проверка что параметр сохраняется правильно
       assert params.new_parameter == 5.0
       
       # Проверка сериализации
       data_dict = params.to_dict()
       assert 'new_parameter' in data_dict
       assert data_dict['new_parameter'] == 5.0
       
       # Проверка десериализации
       params_restored = CycloneParameters.from_dict(data_dict)
       assert params_restored.new_parameter == 5.0
   ```

## Особые случаи

### Добавление параметра без критерия обнаружения

Если параметр вычисляется не в процессе обнаружения, а после (например, в анализе), то:

1. Добавьте его только в модель (`models/parameters.py` и `models/cyclone.py`)
2. Реализуйте вычисление в методе `get_current_parameters()` или в модуле анализа
3. Пропустите шаги, связанные с критериями обнаружения

### Добавление параметра с кастомной визуализацией

Если требуются специальные графики для нового параметра:

1. Создайте новую функцию визуализации в `visualization/`
2. Используйте существующие шаблоны для графиков (карт, диаграмм рассеяния и т.д.)

## Заключение

После выполнения всех этих шагов новый параметр будет полностью интегрирован в систему ArcticCyclone и будет доступен для:

- Обнаружения и отслеживания циклонов
- Визуализации и анализа параметров
- Фильтрации и валидации по нескольким критериям
- Настройки через конфигурационный файл
- Сохранения и загрузки в сериализованных данных