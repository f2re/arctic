# ArcticCyclone: Фреймворк для обнаружения и анализа арктических мезоциклонов

ArcticCyclone — это Python-фреймворк для обнаружения, отслеживания, анализа и визуализации арктических мезоциклонов на основе метеорологических данных реанализа.

## Содержание

- [Обзор проекта](#обзор-проекта)
- [Установка](#установка)
- [Архитектура системы](#архитектура-системы)
- [Руководство пользователя](#руководство-пользователя)
  - [Базовый рабочий процесс](#базовый-рабочий-процесс)
  - [Настройка источников данных](#настройка-источников-данных)
  - [Добавление нового критерия обнаружения](#добавление-нового-критерия-обнаружения)
  - [Настройка параметров обнаружения](#настройка-параметров-обнаружения)
  - [Создание новой визуализации](#создание-новой-визуализации)
- [API документация](#api-документация)

## Обзор проекта

ArcticCyclone предназначен для исследования мезоциклонов в Арктике — интенсивных атмосферных вихрей диаметром 100-1000 км. Система обеспечивает:

- Получение данных из источников реанализа (ERA5 по умолчанию)
- Обнаружение циклонов с помощью различных алгоритмов и критериев
- Отслеживание циклонов на протяжении их жизненного цикла
- Анализ характеристик циклонов и их классификацию
- Визуализацию треков циклонов и метеорологических полей
- Экспорт результатов в различные форматы

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/f2re/arctic.git
cd arctic

# Установка зависимостей
pip install -r requirements.txt
```

## Архитектура системы

См. подробную и актуальную структуру в файле [`structure.md`](structure.md) — он является каноническим источником для организации модулей и их ответственности.

Основные директории:
- **analysis/**: Инструменты анализа циклонов (климатология, сравнения, статистика)
- **core/**: Базовые компоненты (конфигурация, логирование, обработка исключений)
- **data/**: Получение, адаптеры (ERA5 и др.), кэширование, обработка, каталог, менеджер учётных данных
- **detection/**: Алгоритмы и критерии обнаружения и отслеживания циклонов (pressure minima, vorticity, и др.)
- **export/**: Экспорт данных (CSV, GeoJSON, NetCDF)
- **models/**: Модели данных циклонов и параметров
- **visualization/**: Визуализация (карты, треки, тепловые карты, критерии)
- **main.py**: Точка входа
- **config.yaml**: Конфигурация
- **requirements.txt**: Зависимости
- **structure.md**: Описание структуры
- **README.md**: Документация

> **Важно:** Все новые функции и изменения должны соответствовать структуре [`structure.md`](structure.md) и стандартам кодирования проекта.

Подробная структура проекта описана в файле `structure.md`.

## Руководство пользователя

### Быстрый старт и базовый рабочий процесс

---

### Аргументы командной строки для main.py

Скрипт `main.py` принимает несколько аргументов командной строки для управления рабочим процессом обнаружения и анализа арктических мезоциклонов. Ниже приведено описание каждого аргумента и примеры их использования.

#### Аргументы

- **--config**  
  *Тип*: `str`, по умолчанию: `config.yaml`  
  *Описание*: Путь к файлу конфигурации. В этом файле задаются параметры источников данных, критерии обнаружения и настройки рабочего процесса.  
  *Пример*:  
  ```bash
  python main.py --config custom_config.yaml ...
  ```

- **--start-date**  
  *Тип*: `str`, обязательно  
  *Описание*: Дата начала периода анализа в формате `YYYY-MM-DD`. Определяет начало временного окна для обнаружения и анализа циклонов.  
  *Пример*:  
  ```bash
  python main.py --start-date 2021-06-01 ...
  ```

- **--end-date**  
  *Тип*: `str`, обязательно  
  *Описание*: Дата окончания периода анализа в формате `YYYY-MM-DD`. Определяет конец временного окна для обнаружения и анализа циклонов.  
  *Пример*:  
  ```bash
  python main.py --end-date 2021-08-31 ...
  ```

- **--output-dir**  
  *Тип*: `str`, по умолчанию: `output`  
  *Описание*: Директория, в которую будут сохраняться все выходные файлы (результаты, логи, визуализации).  
  *Пример*:  
  ```bash
  python main.py --output-dir results/ ...
  ```

- **--log-level**  
  *Тип*: `str`, по умолчанию: `INFO`, варианты: `DEBUG`, `INFO`, `WARNING`, `ERROR`  
  *Описание*: Уровень детализации логирования. Используйте `DEBUG` для подробной отладки, `INFO` для стандартной работы, `WARNING` или `ERROR` для сообщений о проблемах.  
  *Пример*:  
  ```bash
  python main.py --log-level DEBUG ...
  ```

- **--debug-tracking**  
  *Тип*: `store_true` (флаг)  
  *Описание*: Включает режим отладки для трекинга циклонов. При активации будут сохраняться промежуточные CSV-файлы с деталями трекинга для диагностики и разработки.  
  *Пример*:  
  ```bash
  python main.py --debug-tracking ...
  ```

- **--debug-plot**  
  *Тип*: `store_true` (флаг)  
  *Описание*: Включает режим отладки для визуализации. При активации будут сохраняться дополнительные изображения для диагностики.  
  *Пример*:  
  ```bash
  python main.py --debug-plot ...
  ```

---

#### Пример запуска

Обычный запуск рабочего процесса:

```bash
python main.py --config config.yaml --start-date 2021-06-01 --end-date 2021-08-31 --output-dir output --log-level INFO
```

С включёнными режимами отладки:

```bash
python main.py --start-date 2021-06-01 --end-date 2021-08-31 --debug-tracking --debug-plot
```

---

Каждый аргумент настраивает ключевой аспект пайплайна обнаружения и анализа циклонов, позволяя пользователям выбирать временной диапазон, конфигурацию, директорию вывода, уровень логирования и опции отладки в соответствии с задачами исследования. Такой интерфейс обеспечивает воспроизводимость и прозрачность рабочих процессов, как описано в README и документации проекта.


### FAQ и устранение неполадок

- **Ошибка: Переменная давления не найдена в наборе данных**
  - Проверьте корректность разделения переменных на surface/pressure_levels в конфиге.
  - Убедитесь, что данные загружаются с нужным 'dataset_type'.
- **Ошибка авторизации ERA5**
  - Проверьте корректность API-ключа и файла `.cdsapirc`.
- **Файл превышает 500 строк**
  - Разделите код на модули по назначению, обновите `structure.md`.

---

### Базовый рабочий процесс

```python
from datetime import datetime
from pathlib import Path

from core.config import ConfigManager
from core.logging_setup import setup_logging
from data.acquisition import DataSourceManager
from data.credentials import CredentialManager
from detection.tracker import CycloneDetector, CycloneTracker
from export.formats.csv_exporter import CycloneCSVExporter
from visualization.tracks import plot_cyclone_tracks

# Настройка логирования
setup_logging('INFO', 'arctic_cyclone.log')

# Загрузка конфигурации
config = ConfigManager('config.yaml')

# Настройка учетных данных для ERA5
credentials = CredentialManager()
credentials.set('ERA5', api_key='ваш_ключ_API_ERA5')

# Инициализация менеджера данных
data_manager = DataSourceManager(config_path='config.yaml', credentials=credentials)

# Определение параметров запроса
region = {
    'north': 90.0,  # Северная граница (Северный полюс)
    'south': 70.0,  # Южная граница (Арктический круг)
    'east': 180.0,  # Восточная граница
    'west': -180.0  # Западная граница (полный диапазон долготы)
}

timeframe = {
    'years': ['2020'],
    'months': ['01', '02', '03'],
    'days': [str(d).zfill(2) for d in range(1, 32)],
    'hours': ['00:00', '06:00', '12:00', '18:00']
}

parameters = {
    'variables': ['z', 'u', 'v', 't', 'vo'],  # Геопотенциал, компоненты ветра, температура, завихренность
    'levels': [1000, 925, 850, 700, 500]       # Стандартные уровни давления в гПа
}

# Загрузка данных из ERA5
dataset = data_manager.get_data(
    source="ERA5",
    parameters=parameters,
    region=region,
    timeframe=timeframe,
    use_cache=True
)

# Инициализация детектора циклонов
detector = CycloneDetector(min_latitude=65.0)

# Установка критериев обнаружения
detector.set_criteria(["pressure_minimum", "vorticity"])

# Обнаружение циклонов для каждого временного шага
all_cyclones = {}
for time_step in dataset.time.values:
    cyclones = detector.detect(dataset, time_step)
    all_cyclones[time_step] = cyclones
    print(f"Обнаружено {len(cyclones)} циклонов для {time_step}")

# Инициализация трекера циклонов
tracker = CycloneTracker()

# Отслеживание циклонов
cyclone_tracks = tracker.track(all_cyclones)

# Фильтрация треков (минимальная продолжительность - 12 часов, 3 точки)
filtered_tracks = tracker.filter_tracks(cyclone_tracks, min_duration=12.0, min_points=3)
print(f"Найдено {len(filtered_tracks)} треков циклонов продолжительностью >= 12 часов")

# Экспорт треков в CSV
exporter = CycloneCSVExporter()
exporter.export_cyclone_tracks(filtered_tracks, 'cyclone_tracks.csv')

# Визуализация треков
plot_cyclone_tracks(filtered_tracks, region, 'cyclone_tracks.png')
```

### Настройка источников данных

**Обработка ERA5:**
- Переменные поверхности (например, 'msl' — давление на уровне моря) и переменные на уровнях давления (например, 'z', 'u', 'v', 't', 'vo') запрашиваются из разных подмножеств ERA5 и требуют разных параметров 'dataset_type'.
- В конфиге переменные делятся на две группы: surface и pressure_levels.
- Запрос данных делается отдельно для каждой группы, затем данные объединяются (merge).
- Пример в `data/adapters/era5.py` и соответствующих обработчиках.
- Подробнее см. комментарии в коде и [`structure.md`](structure.md).


Пример конфигурации для источника данных ERA5:

```python
from pathlib import Path
from data.acquisition import DataSourceManager
from data.credentials import CredentialManager

# Создание менеджера учетных данных и установка API-ключа для ERA5
credentials = CredentialManager()
credentials.set('ERA5', api_key='ваш_ключ_API_ERA5')

# Инициализация менеджера источников данных
data_manager = DataSourceManager(credentials=credentials)

# Получение данных из ERA5
era5_data = data_manager.get_data(
    source="ERA5",
    parameters={
        'variables': ['z', 'u', 'v', 't', 'vo'],
        'levels': [1000, 925, 850, 700, 500]
    },
    region={
        'north': 90.0,
        'south': 70.0,
        'east': 180.0,
        'west': -180.0
    },
    timeframe={
        'years': ['2020'],
        'months': ['01'],
        'days': ['01', '02', '03'],
        'hours': ['00:00', '06:00', '12:00', '18:00']
    }
)

# Сохранение данных в кэш
print(f"Загружены данные размерностью: {era5_data.dims}")
```

### Добавление нового критерия обнаружения и визуализации

#### 1. Создание нового критерия

1. Создайте новый файл, например, `my_criterion.py` в директории [`detection/criteria`](detection/criteria/).
2. Класс критерия должен наследоваться от `BaseCriterion` и реализовывать метод `apply`.

```python
from detection.criteria import BaseCriterion
from core.exceptions import DetectionError
import xarray as xr
import numpy as np

class MyCustomCriterion(BaseCriterion):
    def __init__(self, threshold: float = 1000.0):
        self.threshold = threshold
    def apply(self, dataset: xr.Dataset, time_step):
        # Логика обнаружения
        ...
```

#### 2. Регистрация критерия в системе

1. Зарегистрируйте новый критерий через менеджер критериев:
```python
from detection.criteria import CriteriaManager
from detection.tracker import CycloneDetector
from detection.criteria.my_criterion import MyCustomCriterion

criteria_manager = CriteriaManager()
criteria_manager.register_criterion("my_custom", MyCustomCriterion(threshold=995.0))

detector = CycloneDetector(min_latitude=70.0)
detector.criteria_manager = criteria_manager
# Установите активные критерии
 detector.set_criteria(["my_custom"])
```

#### 3. Добавление критерия в конфигурацию

1. В `config.yaml` добавьте новый критерий в секцию `detection > criteria`:
```yaml
detection:
  criteria:
    my_custom:
      enabled: true
      threshold: 995.0
```
2. Убедитесь, что ваш критерий корректно читает параметры из конфигурации через `ConfigManager`.

#### 4. Добавление функции визуализации выбранного критерия

1. Создайте новый файл, например, `my_criterion_plot.py` в [`visualization/`](visualization/) или добавьте функцию в существующий модуль визуализации.
2. Функция должна принимать результаты работы критерия (например, список кандидатов или циклонов) и строить график.

```python
import matplotlib.pyplot as plt
from typing import List, Dict

def plot_my_custom_criterion(candidates: List[Dict], region: Dict, output_file=None):
    lats = [c['latitude'] for c in candidates]
    lons = [c['longitude'] for c in candidates]
    plt.figure(figsize=(8, 6))
    plt.scatter(lons, lats, c='red', label='Кандидаты')
    plt.title('Результаты критерия my_custom')
    plt.xlabel('Долгота')
    plt.ylabel('Широта')
    plt.legend()
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()
```

3. Используйте функцию визуализации после обнаружения кандидатов:
```python
from visualization.my_criterion_plot import plot_my_custom_criterion
plot_my_custom_criterion(candidates, region, output_file='output/my_custom_criterion.png')
```

#### 5. Рекомендации по структуре
- Все новые критерии размещайте в [`detection/criteria/`](detection/criteria/).
- Все функции визуализации — в [`visualization/`](visualization/), группируя по назначению.
- Описание и параметры новых критериев добавляйте в `config.yaml` и обновляйте [`structure.md`](structure.md).
- Следуйте стандартам кодирования и модульности (см. выше).

---


Создайте новый файл `my_criterion.py` в директории `detection/criteria`:

```python
from typing import Dict, List, Any
import xarray as xr
import numpy as np
import logging

from detection.criteria import BaseCriterion
from core.exceptions import DetectionError

# Инициализация логгера
logger = logging.getLogger(__name__)

class MyCustomCriterion(BaseCriterion):
    """
    Пользовательский критерий обнаружения циклонов.
    
    Описание алгоритма и логики работы критерия.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                threshold: float = 1.0,
                window_size: int = 3):
        """
        Инициализирует пользовательский критерий.
        
        Аргументы:
            min_latitude: Минимальная широта для обнаружения (градусы с.ш.).
            threshold: Пороговое значение для критерия.
            window_size: Размер окна для анализа.
        """
        self.min_latitude = min_latitude
        self.threshold = threshold
        self.window_size = window_size
        
        logger.debug(f"Инициализирован пользовательский критерий с параметрами: "
                    f"threshold={threshold}, window_size={window_size}")
    
    def apply(self, dataset: xr.Dataset, time_step: Any) -> List[Dict]:
        """
        Применяет критерий к набору данных.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            
        Возвращает:
            Список кандидатов в циклоны (словари с координатами и свойствами).
        """
        try:
            # Ваш алгоритм обнаружения
            # Пример: поиск точек, где значение параметра превышает порог
            
            # Определяем переменную для анализа (например, давление)
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            
            for var in pressure_vars:
                if var in dataset:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                raise ValueError("Не удается определить переменную давления в наборе данных")
                
            # Выбираем временной шаг и применяем маску региона
            time_data = dataset.sel(time=time_step)
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Получаем поле для анализа
            field = arctic_data[pressure_var].values
            
            # Примените вашу логику обнаружения
            # Пример: найдите точки, удовлетворяющие условию
            import scipy.ndimage as ndimage
            
            # Пример: находим локальные минимумы
            min_filter = ndimage.minimum_filter(field, size=self.window_size)
            local_minima = (field == min_filter) & (field < self.threshold)
            
            # Получаем координаты найденных точек
            points_indices = np.where(local_minima)
            
            # Формируем список кандидатов
            candidates = []
            
            for i in range(len(points_indices[0])):
                lat_idx = points_indices[0][i]
                lon_idx = points_indices[1][i]
                
                latitude = float(arctic_data.latitude.values[lat_idx])
                longitude = float(arctic_data.longitude.values[lon_idx])
                value = float(field[lat_idx, lon_idx])
                
                # Создаем кандидата
                candidate = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'pressure': value,
                    'criterion': 'my_custom_criterion'
                }
                
                candidates.append(candidate)
            
            logger.debug(f"Пользовательский критерий нашел {len(candidates)} кандидатов")
            return candidates
            
        except Exception as e:
            error_msg = f"Ошибка при применении пользовательского критерия: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
```

Затем зарегистрируйте и используйте ваш критерий:

```python
from detection.tracker import CycloneDetector
from detection.criteria import CriteriaManager
from my_criterion import MyCustomCriterion

# Создаем менеджер критериев
criteria_manager = CriteriaManager()

# Регистрируем наш пользовательский критерий
criteria_manager.register_criterion("my_custom", MyCustomCriterion(threshold=1000.0))

# Создаем детектор циклонов
detector = CycloneDetector(min_latitude=70.0)

# Устанавливаем новый критерий как активный
detector.criteria_manager.set_active_criteria(["my_custom"])

# Используем детектор с новым критерием
cyclones = detector.detect(dataset, time_step)
```

### Настройка параметров обнаружения

Создайте или модифицируйте файл конфигурации `config.yaml`:

```yaml
data:
  default_source: 'ERA5'
  cache_dir: 'data/cache'
  sources:
    ERA5:
      type: 'reanalysis'
      variables:
        - 'z'   # геопотенциал
        - 'u'   # зональный ветер
        - 'v'   # меридиональный ветер
        - 't'   # температура
        - 'vo'  # завихренность
      levels:
        - 1000
        - 925
        - 850
        - 700
        - 500

detection:
  min_latitude: 70.0
  criteria:
    pressure_minimum:
      enabled: true
      pressure_threshold: 1005.0  # гПа
      min_gradient: 0.7          # гПа/100км
      window_size: 5
    vorticity:
      enabled: true
      vorticity_threshold: 2e-5  # 1/с
      pressure_level: 850        # гПа
    pressure_gradient:
      enabled: false
    closed_contour:
      enabled: true
      contour_interval: 2.0      # гПа
      min_contours: 2
  
  tracking:
    max_distance: 400.0          # км
    max_pressure_change: 10.0    # гПа
    max_time_gap: 12             # часов

visualization:
  default_projection: 'NorthPolarStereo'
  map_resolution: 'intermediate'
  output_dir: 'output/figures'

export:
  output_dir: 'output/data'
  formats:
    - 'csv'
    - 'netcdf'
```

Код для загрузки и применения конфигурации:

```python
from core.config import ConfigManager
from detection.tracker import CycloneDetector
from detection.algorithms import PressureMinimaAlgorithm

# Загружаем конфигурацию
config = ConfigManager('config.yaml')

# Получаем настройки обнаружения
detection_config = config.get('detection')

# Инициализируем детектор с параметрами из конфигурации
detector = CycloneDetector(min_latitude=detection_config['min_latitude'])

# Настройка алгоритма обнаружения на основе минимумов давления
pressure_criteria_config = detection_config['criteria']['pressure_minimum']
if pressure_criteria_config['enabled']:
    from detection.criteria.pressure import PressureMinimumCriterion
    
    # Создаем критерий с параметрами из конфигурации
    pressure_criterion = PressureMinimumCriterion(
        pressure_threshold=pressure_criteria_config['pressure_threshold'],
        min_gradient=pressure_criteria_config['min_gradient'],
        window_size=pressure_criteria_config['window_size']
    )
    
    # Регистрируем критерий
    detector.criteria_manager.register_criterion('pressure_custom', pressure_criterion)

# Активируем необходимые критерии
active_criteria = [name for name, config in detection_config['criteria'].items() 
                  if config['enabled']]
detector.set_criteria(active_criteria)

# Используем детектор
cyclones = detector.detect(dataset, time_step)
```

### Создание новой визуализации

Создайте файл `my_visualization.py` в директории `visualization`:

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from models.cyclone import Cyclone

def plot_cyclone_intensity_map(cyclones: List[Cyclone], 
                              region: Dict[str, float],
                              output_file: Optional[Path] = None,
                              resolution: str = 'intermediate',
                              grid_size: float = 1.0) -> None:
    """
    Создает карту интенсивности циклонов для арктического региона.
    
    Аргументы:
        cyclones: Список объектов циклонов для визуализации.
        region: Географический регион (север, юг, запад, восток).
        output_file: Путь для сохранения изображения. Если None, показывает изображение.
        resolution: Разрешение карты ('low', 'intermediate', 'high').
        grid_size: Размер ячейки сетки в градусах.
        
    Примечание:
        Цвет на карте соответствует интенсивности циклона (центральное давление).
    """
    # Создаем фигуру с проекцией
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    
    # Устанавливаем границы карты
    ax.set_extent([region['west'], region['east'], 
                  region['south'], region['north']], 
                 crs=ccrs.PlateCarree())
    
    # Добавляем фоновые элементы
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.3, linestyle=':')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    # Создаем сетку для карты интенсивности
    lon_grid = np.arange(region['west'], region['east'] + grid_size, grid_size)
    lat_grid = np.arange(region['south'], region['north'] + grid_size, grid_size)
    intensity_grid = np.zeros((len(lat_grid) - 1, len(lon_grid) - 1))
    count_grid = np.zeros_like(intensity_grid)
    
    # Заполняем сетку данными о циклонах
    for cyclone in cyclones:
        # Пропускаем, если вне региона
        if (cyclone.latitude < region['south'] or cyclone.latitude > region['north'] or
            cyclone.longitude < region['west'] or cyclone.longitude > region['east']):
            continue
            
        # Определяем индексы ячейки для циклона
        lat_idx = int((cyclone.latitude - region['south']) / grid_size)
        lon_idx = int((cyclone.longitude - region['west']) / grid_size)
        
        # Проверяем границы
        if (lat_idx >= 0 and lat_idx < intensity_grid.shape[0] and
            lon_idx >= 0 and lon_idx < intensity_grid.shape[1]):
            
            # Добавляем интенсивность (инвертируем давление, чтобы высокие значения соответствовали низкому давлению)
            intensity_value = 1020 - cyclone.central_pressure  # интенсивность как отклонение от нормы
            intensity_grid[lat_idx, lon_idx] += intensity_value
            count_grid[lat_idx, lon_idx] += 1
    
    # Нормализуем интенсивность по количеству циклонов в ячейке
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_intensity = np.divide(intensity_grid, count_grid)
    avg_intensity = np.nan_to_num(avg_intensity)
    
    # Создаем координатные сетки для отображения
    lon_centers = lon_grid[:-1] + grid_size/2
    lat_centers = lat_grid[:-1] + grid_size/2
    
    # Создаем сетку координат для pcolormesh
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Отображаем интенсивность
    cmap = plt.get_cmap('hot_r')  # обратная цветовая схема 'hot'
    mesh = ax.pcolormesh(lon_mesh, lat_mesh, avg_intensity, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, alpha=0.7)
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(mesh, ax=ax, pad=0.1)
    cbar.set_label('Интенсивность циклонов (инвертированное давление)')
    
    # Добавляем заголовок и метки
    plt.title('Карта интенсивности арктических циклонов', fontsize=16)
    
    # Сохраняем или показываем
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
```

Использование новой визуализации:

```python
from my_visualization import plot_cyclone_intensity_map
from pathlib import Path

# Регион для визуализации
region = {
    'north': 90.0,   # Северный полюс
    'south': 70.0,   # Арктический круг
    'east': 180.0,   # Восточная граница
    'west': -180.0   # Западная граница
}

# Создаем карту интенсивности циклонов
plot_cyclone_intensity_map(
    # Здесь должен быть список циклонов
    [cyclone for track in filtered_tracks for cyclone in track],
    region=region,
    output_file=Path('output/cyclone_intensity_map.png'),
    grid_size=2.0  # Размер ячейки 2 градуса
)
```

## API документация

Документация по основным классам и функциям приведена ниже. Для подробностей по структуре и расширению системы см. [`structure.md`](structure.md) и комментарии в коде.


### Класс Cyclone

Основной класс для представления циклона.

```python
from models.cyclone import Cyclone

# Создание объекта циклона
cyclone = Cyclone(
    latitude=75.5,              # Широта центра (°с.ш.)
    longitude=120.3,            # Долгота центра (°в.д.)
    time='2020-01-15T06:00',    # Время наблюдения
    central_pressure=990.5,     # Центральное давление (гПа)
    dataset=None                # Метеорологические данные (опционально)
)

# Доступ к свойствам циклона
print(f"Координаты: {cyclone.latitude}°с.ш., {cyclone.longitude}°в.д.")
print(f"Давление: {cyclone.central_pressure} гПа")
print(f"Время: {cyclone.time}")

# Анализ жизненного цикла циклона
if len(cyclone.track) > 1:
    metrics = cyclone.calculate_lifecycle_metrics()
    print(f"Продолжительность: {metrics['lifespan_hours']} ч")
    print(f"Скорость углубления: {metrics['deepening_rate']} гПа/ч")
    print(f"Смещение: {metrics['displacement']} км")
    print(f"Средняя скорость: {metrics['mean_speed']} км/ч")

# Получение интенсивности циклона
intensity = cyclone.calculate_intensity()
print(f"Интенсивность: {intensity.value}")
```

### Класс DataSourceManager

Управление источниками данных.

```python
from data.acquisition import DataSourceManager
from data.credentials import CredentialManager

# Создание менеджера учетных данных
credentials = CredentialManager()
credentials.set('ERA5', api_key='ваш_ключ_API')

# Создание менеджера источников данных
data_manager = DataSourceManager(credentials=credentials)

# Регистрация пользовательского источника данных
from data.acquisition import BaseDataAdapter

class MyCustomAdapter(BaseDataAdapter):
    # Реализация адаптера...
    pass

data_manager.register_custom_source('MySource', MyCustomAdapter)

# Получение данных
dataset = data_manager.get_data(
    source='ERA5',  # или 'MySource'
    parameters={},
    region={},
    timeframe={}
)
```

### Класс CycloneDetector и CycloneTracker

Обнаружение и отслеживание циклонов.

```python
from detection.tracker import CycloneDetector, CycloneTracker

# Создание детектора циклонов
detector = CycloneDetector(min_latitude=70.0)

# Настройка критериев обнаружения
detector.set_criteria(['pressure_minimum', 'vorticity'])

# Обнаружение циклонов
cyclones = detector.detect(dataset, time_step)

# Создание трекера циклонов
tracker = CycloneTracker(
    max_distance=500.0,           # Максимальное расстояние (км)
    max_pressure_change=15.0,     # Максимальное изменение давления (гПа)
    max_time_gap=12               # Максимальный разрыв во времени (часы)
)

# Отслеживание циклонов
cyclone_tracks = tracker.track(all_cyclones)

# Фильтрация треков
filtered_tracks = tracker.filter_tracks(
    cyclone_tracks,
    min_duration=12.0,  # Минимальная продолжительность (часы)
    min_points=3        # Минимальное количество точек
)
```