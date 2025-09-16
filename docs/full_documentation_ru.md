# Полная документация ArcticCyclone

## Содержание

1. [Обзор проекта](#обзор-проекта)
2. [Архитектура системы](#архитектура-системы)
3. [Установка и настройка](#установка-и-настройка)
4. [Быстрый старт](#быстрый-старт)
5. [Настройка конфигурации](#настройка-конфигурации)
6. [Получение данных](#получение-данных)
7. [Алгоритмы обнаружения](#алгоритмы-обнаружения)
8. [Отслеживание циклонов](#отслеживание-циклонов)
9. [Визуализация](#визуализация)
10. [Экспорт данных](#экспорт-данных)
11. [Анализ результатов](#анализ-результатов)
12. [API документация](#api-документация)
13. [Примеры использования](#примеры-использования)
14. [Устранение неполадок](#устранение-неполадок)

## Обзор проекта

ArcticCyclone — это Python-фреймворк для обнаружения, отслеживания, анализа и визуализации арктических мезоциклонов на основе метеорологических данных реанализа. Система предназначена для исследования мезоциклонов в Арктике — интенсивных атмосферных вихрей диаметром 100-1000 км.

Основные возможности системы:

- Получение данных из источников реанализа (ERA5 по умолчанию)
- Обнаружение циклонов с помощью различных алгоритмов и критериев
- Отслеживание циклонов на протяжении их жизненного цикла
- Анализ характеристик циклонов и их классификацию
- Визуализацию треков циклонов и метеорологических полей
- Экспорт результатов в различные форматы

## Архитектура системы

Проект организован в модульную структуру с четким разделением ответственности:

```
arctic-cyclone/
├── analysis/                   # Модули анализа циклонов
│   ├── climatology.py          # Долгосрочные климатологические анализы
│   ├── comparisons.py          # Сравнение циклонов и наборов данных
│   └── statistics.py           # Статистический анализ данных
├── core/                       # Ядро системы
│   ├── config.py               # Управление конфигурацией
│   ├── exceptions.py           # Пользовательские исключения
│   └── logging_setup.py        # Настройка логирования
├── data/                       # Получение и обработка данных
│   ├── acquisition.py          # Получение данных из источников
│   ├── credentials.py          # Управление учетными данными
│   ├── catalog.py              # Каталог наборов данных
│   ├── adapters/               # Адаптеры для разных источников данных
│   │   └── era5.py             # Адаптер ERA5
│   └── processors/             # Обработка данных
│       └── era5_processor.py   # Обработчик данных ERA5
├── detection/                  # Обнаружение и отслеживание циклонов
│   ├── tracker.py              # Отслеживание циклонов
│   ├── validators.py            # Валидация кандидатов
│   ├── algorithms/             # Алгоритмы обнаружения
│   │   ├── pressure_minima.py  # Алгоритм минимумов давления
│   │   ├── arctic_mesocyclone.py # Специализированный алгоритм
│   │   └── multi_parameter.py  # Многопараметрический алгоритм
│   └── criteria/               # Критерии обнаружения
│       ├── pressure.py         # Критерий давления
│       ├── vorticity.py        # Критерий завихренности
│       ├── gradient.py         # Критерий градиента давления
│       ├── laplacian.py        # Критерий лапласиана давления
│       ├── closed_contour.py   # Критерий замкнутых контуров
│       └── wind.py             # Критерий ветра
├── export/                     # Экспорт данных
│   ├── publishers.py            # Публикация данных
│   └── formats/                # Форматы экспорта
│       ├── csv_exporter.py     # Экспорт в CSV
│       ├── geojson_exporter.py # Экспорт в GeoJSON
│       └── netcdf_exporter.py  # Экспорт в NetCDF
├── models/                     # Модели данных
│   ├── cyclone.py              # Модель циклона
│   ├── parameters.py            # Параметры циклона
│   └── classifications.py      # Классификации циклонов
├── visualization/              # Визуализация
│   ├── tracks.py               # Визуализация треков
│   ├── heatmaps.py             # Тепловые карты
│   ├── parameters.py            # Визуализация параметров
│   └── criteria.py             # Визуализация критериев
├── main.py                     # Точка входа
├── config.yaml                 # Конфигурационный файл
├── requirements.txt            # Зависимости
└── setup.py                    # Скрипт установки
```

## Установка и настройка

### Системные требования

- Python 3.8 или выше
- Операционная система: Linux, macOS или Windows
- Доступ к интернету для получения данных

### Установка зависимостей

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/arctic-research/arctic_cyclone.git
   cd arctic_cyclone
   ```

2. Создайте виртуальное окружение:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # или
   venv\Scripts\activate     # Windows
   ```

3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Установите пакет в режиме разработки:
   ```bash
   pip install -e .
   ```

### Настройка учетных данных ERA5

Для получения данных ERA5 необходимо зарегистрироваться в Climate Data Store (CDS) и получить API-ключ:

1. Зарегистрируйтесь на https://cds.climate.copernicus.eu/
2. Получите API-ключ на странице профиля
3. Создайте файл `~/.cdsapirc` с содержимым:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_API_KEY_HERE
   ```

## Быстрый старт

После установки можно запустить базовый рабочий процесс:

```bash
python main.py --start-date 2020-01-01 --end-date 2020-01-03 --output-dir results
```

Эта команда выполнит полный цикл анализа для указанного периода и сохранит результаты в директорию `results`.

## Настройка конфигурации

Конфигурация системы определяется в файле `config.yaml`. Основные параметры:

### Параметры данных

```yaml
data:
  default_source: 'ERA5'
  cache_dir: 'data/cache'
  sources:
    ERA5:
      type: 'reanalysis'
      dataset_type: 'surface'
      variables: ['z', 'u', 'v', 't', 'q', 'vo', 'msl']
      levels: [1000, 925, 850, 700, 500]
      grid_resolution: 0.25
```

### Параметры обнаружения

```yaml
detection:
  min_latitude: 65.0
  criteria:
    pressure_minimum:
      enabled: true
      gradient_threshold: 0.5
    vorticity:
      enabled: true
      threshold: 5.0e-5
      level: 850
    closed_contour:
      enabled: false
      contour_interval: 5.0
    wind_threshold:
      enabled: true
      min_speed: 15.0
    pressure_laplacian:
      enabled: true
      laplacian_threshold: 0.00015
      smooth_sigma: 3
      window_size: 3
  tracking:
    max_distance: 300.0
    max_pressure_change: 10.0
    max_time_gap: 12
    min_duration: 6.0
    min_points: 3
```

### Параметры визуализации

```yaml
visualization:
  default_projection: 'NorthPolarStereo'
  map_resolution: 'intermediate'
  output_dir: 'output/figures'
  default_dpi: 300
  cyclone_marker_size: 30
  track_line_width: 1.5
  color_map: 'viridis'
```

### Параметры экспорта

```yaml
export:
  output_dir: 'output/data'
  formats: ['csv', 'netcdf']
  default_format: 'csv'
```

## Получение данных

Система поддерживает получение данных из различных источников реанализа. По умолчанию используется ERA5 от Copernicus Climate Data Store.

### Поддерживаемые источники данных

- ERA5 (основной источник)
- ERA5-Land (для наземных параметров)
- Другие источники могут быть добавлены через пользовательские адаптеры

### Форматы данных

Система работает с данными в формате NetCDF через библиотеку xarray. Поддерживаются различные типы переменных:

- Геопотенциал (z)
- Компоненты ветра (u, v)
- Температура (t)
- Удельная влажность (q)
- Завихренность (vo)
- Давление на уровне моря (msl)

## Алгоритмы обнаружения

Система использует многокритериальный подход для обнаружения циклонов. Доступны следующие критерии:

### Минимум давления

Ищет локальные минимумы в поле давления на уровне моря. Это основной критерий для обнаружения циклонов.

### Завихренность

Ищет локальные максимумы в поле относительной завихренности на уровне 850 гПа. Этот критерий эффективен для обнаружения динамически активных систем.

### Градиент давления

Определяет области с сильным градиентом давления, что указывает на интенсивные циклоны.

### Лапласиан давления

Использует лапласиан поля давления для более надежного обнаружения центров циклонов.

### Замкнутые контуры

Ищет замкнутые изобары, характерные для хорошо развитых циклонов.

### Порог скорости ветра

Отбирает циклоны с минимальной скоростью ветра, что позволяет исключить слабые системы.

## Отслеживание циклонов

После обнаружения циклонов система отслеживает их во времени с помощью алгоритма ближайших соседей.

### Параметры трекинга

- `max_distance`: Максимальное расстояние между точками (км)
- `max_time_gap`: Максимальный временной разрыв (часы)
- `max_pressure_change`: Максимальное изменение давления (гПа)
- `min_track_duration`: Минимальная длительность трека (часы)
- `min_track_points`: Минимальное количество точек в треке

### Фильтрация треков

После трекинга применяются фильтры для удаления недостоверных треков:

1. Минимальная продолжительность
2. Минимальное количество точек
3. Реалистичность движения (ограничение на скорость)
4. Реалистичность изменений давления

## Визуализация

Система предоставляет широкие возможности для визуализации результатов анализа.

### Типы визуализаций

1. **Треки циклонов**: Отображение путей циклонов на карте
2. **Тепловые карты**: Плотность циклонов в различных регионах
3. **Параметры циклонов**: Графики изменения параметров во времени
4. **Поля критериев**: Визуализация полей, используемых для обнаружения

### Настройка визуализации

Визуализация настраивается через параметры в конфигурационном файле и поддерживает различные проекции карт.

## Экспорт данных

Результаты анализа могут быть экспортированы в различные форматы:

### Поддерживаемые форматы

- **CSV**: Табличные данные для дальнейшего анализа
- **NetCDF**: Научный формат для обмена метеорологическими данными
- **GeoJSON**: Геопространственные данные для использования в GIS

### Структура экспортируемых данных

Экспортируемые данные включают:

- Координаты циклонов
- Время наблюдения
- Центральное давление
- Дополнительные параметры (завихренность, скорость ветра и т.д.)
- Идентификаторы треков

## Анализ результатов

Система предоставляет инструменты для статистического анализа обнаруженных циклонов:

### Статистика циклонов

- Распределение продолжительности жизни
- Распределение минимального давления
- Скорости перемещения
- Скорости углубления

### Климатологический анализ

- Сезонное распределение циклонов
- Пространственное распределение
- Долгосрочные тренды

## API документация

### Основные классы

#### CycloneDetector

Класс для обнаружения циклонов на основе настраиваемых критериев.

##### Методы

- `__init__(min_latitude, config, debug_plot)`: Инициализация детектора
- `detect(dataset, time_step)`: Обнаружение циклонов для временного шага
- `set_criteria(criterion_names)`: Установка активных критериев

#### CycloneTracker

Класс для отслеживания циклонов во времени.

##### Методы

- `__init__(max_distance, max_time_gap, max_pressure_change, ...)`: Инициализация трекера
- `track(all_cyclones)`: Отслеживание циклонов
- `filter_tracks(tracks, min_duration, min_points)`: Фильтрация треков

#### Cyclone

Основной класс для представления циклона.

##### Атрибуты

- `latitude`: Широта центра циклона
- `longitude`: Долгота центра циклона
- `time`: Время наблюдения
- `central_pressure`: Центральное давление
- `track_id`: Идентификатор трека

##### Методы

- `calculate_lifecycle_metrics()`: Вычисление метрик жизненного цикла
- `calculate_intensity()`: Определение интенсивности циклона
- `get_current_parameters()`: Получение текущих параметров

#### DataSourceManager

Управление источниками данных.

##### Методы

- `__init__(config_path, cache_dir, credentials)`: Инициализация менеджера
- `get_data(source, parameters, region, timeframe, use_cache)`: Получение данных
- `register_custom_source(name, adapter_class)`: Регистрация пользовательского источника

## Примеры использования

### Базовый анализ

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

### Настройка пользовательского критерия

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
        pass

# Регистрация критерия
from detection.criteria import CriteriaManager
from detection.tracker import CycloneDetector

criteria_manager = CriteriaManager()
criteria_manager.register_criterion("my_custom", MyCustomCriterion(threshold=995.0))

detector = CycloneDetector(min_latitude=70.0)
detector.criteria_manager = criteria_manager
detector.set_criteria(["my_custom"])
```

## Устранение неполадок

### Распространенные проблемы

#### Ошибка: Переменная давления не найдена в наборе данных

- Проверьте корректность разделения переменных на surface/pressure_levels в конфиге
- Убедитесь, что данные загружаются с нужным 'dataset_type'

#### Ошибка авторизации ERA5

- Проверьте корректность API-ключа и файла `.cdsapirc`
- Убедитесь, что файл `.cdsapirc` находится в домашней директории

#### Медленная работа системы

- Используйте кэширование данных
- Ограничьте временной диапазон анализа
- Уменьшите пространственный регион

#### Нет обнаруженных циклонов

- Проверьте параметры критериев обнаружения
- Убедитесь, что временной период содержит циклоны
- Попробуйте изменить пороговые значения

### Логирование

Система использует стандартное логирование Python. Уровни логирования:

- `DEBUG`: Подробная отладочная информация
- `INFO`: Общая информация о ходе выполнения
- `WARNING`: Предупреждения о возможных проблемах
- `ERROR`: Ошибки, препятствующие нормальному выполнению

Для включения отладочного режима используйте флаг `--log-level DEBUG` при запуске.