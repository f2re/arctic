# Arctic Cyclones Detection and Analysis Toolkit (ACDAT)

Комплексный инструментарий для автоматизированной детекции, классификации и анализа мезомасштабных циклонических структур в Арктическом регионе на основе данных реанализа ERA5.

## Содержание

- [Обзор проекта](#обзор-проекта)
- [Установка и зависимости](#установка-и-зависимости)
- [Структура проекта](#структура-проекта)
- [Научная методология](#научная-методология)
- [Примеры использования](#примеры-использования)
- [Документация функций](#документация-функций)
  - [Загрузка данных](#загрузка-данных)
  - [Алгоритмы детекции](#алгоритмы-детекции)
  - [Классификация циклонов](#классификация-циклонов)
  - [Отслеживание циклонов](#отслеживание-циклонов)
  - [Визуализация](#визуализация)
- [Типичные задачи](#типичные-задачи)
- [Цитирование и ссылки](#цитирование-и-ссылки)

## Обзор проекта

ACDAT представляет собой совокупность алгоритмов и инструментов для комплексного исследования циклонической активности в Арктическом регионе. Проект поддерживает широкий спектр задач, включая:

- Автоматическую детекцию синоптических и мезомасштабных циклонов по данным реанализа ERA5
- Классификацию циклонов по термической структуре (холодное/теплое ядро)
- Отслеживание эволюции циклонов во времени
- Расчет статистических характеристик и пространственного распределения
- Визуализацию результатов в различных форматах

Инструментарий использует современные методы обнаружения циклонов, включая анализ лапласиана давления, параметр Okubo-Weiss, градиенты температуры поверхности моря и другие комплексные подходы для повышения точности идентификации арктических мезомасштабных вихрей.

## Установка и зависимости

Проект разработан для работы в среде Python 3.7+ и Google Colab. Для локального использования требуются следующие библиотеки:

```bash
pip install numpy pandas xarray matplotlib cartopy scipy netCDF4 h5py cfgrib
```

Для доступа к данным ERA5 необходим зарегистрированный API-ключ Copernicus Climate Data Store. 

## Структура проекта

```
arctic_cyclones/
│
├── config.py                # Конфигурационные параметры
├── main.py                  # Главный скрипт запуска
│
├── data/
│   ├── __init__.py
│   ├── download.py          # Функции загрузки данных ERA5
│   └── preprocessing.py     # Предобработка данных
│
├── detection/
│   ├── __init__.py
│   ├── parameters.py        # Параметры для разных типов циклонов
│   ├── algorithms.py        # Алгоритмы обнаружения
│   └── thermal.py           # Классификация термической структуры
│
├── analysis/
│   ├── __init__.py
│   ├── metrics.py           # Расчет метрик циклонов
│   └── tracking.py          # Отслеживание циклонов во времени
│
└── visualization/
    ├── __init__.py
    ├── plots.py             # Функции визуализации
    └── diagnostics.py       # Диагностические визуализации
```

## Научная методология

Проект реализует алгоритм отслеживания арктических циклонов от синоптического до мезо-α-масштаба с разделением на циклоны с холодным и теплым ядром. Методология базируется на применении алгоритма к данным реанализа ERA5 севернее 60°N с возможностью анализа длительных периодов (например, 1950-2019 гг.).

Спектр арктических циклонов характеризуется механизмами циклогенеза от бароклинной до конвективной природы. Синоптические штормы и полярные мезоциклоны в качестве бароклинных неустойчивостей хорошо описаны в многочисленных наблюдательных исследованиях. Мезомасштабные полярные мезоциклоны как конвективные системы концептуализированы как развивающиеся из термических неустойчивостей.

### Типы циклонов и параметры детекции

Инструментарий поддерживает детекцию трех основных типов циклонов с различными пороговыми значениями:

1. **Синоптические циклоны** (500-1500 км)
   - Порог лапласиана: -0.15 гПа/градус²
   - Минимальная глубина: 2.5 гПа
   - Диапазон радиусов: 200-1500 км

2. **Мезомасштабные циклоны** (200-600 км)
   - Порог лапласиана: -0.12 гПа/градус²
   - Минимальная глубина: 1.0 гПа
   - Диапазон радиусов: 100-600 км

3. **Полярные мезоциклоны** (полярные низкие, 50-400 км)
   - Порог лапласиана: -0.10 гПа/градус²
   - Минимальная глубина: 0.8 гПа
   - Диапазон радиусов: 50-400 км

Алгоритм обнаружения использует следующие критерии: (1) определение минимума давления; (2) проверка замкнутых изобар; (3) анализ лапласиана давления; (4) расчет и проверка значительного градиента давления; (5) расчет завихренности; (6) оценка скорости ветра; (7) определение термической структуры.

### Методы детекции

Современные методы обнаружения арктических мезомасштабных вихрей основываются на нескольких ключевых параметрах:

1. **Параметр Okubo-Weiss (W)** - квантифицирует баланс между деформацией и завихренностью. Отрицательные значения W указывают на области, где доминирует вращение (потенциальные вихри).
2. **Относительная завихренность** - описывает вращение жидких элементов. Обычно определяют пороговое значение для идентификации вихрей (типично |ω| > 10⁻⁵ с⁻¹).
3. **Градиенты температуры поверхности моря (SST)** - помогают идентифицировать термические границы, связанные с вихрями.

Наиболее эффективные методы используют комбинацию параметров:

1. **Гибридные методы детекции (HD)** интегрируют критерии метода Okubo-Weiss и метода на основе высоты морской поверхности, что повышает точность обнаружения до ~96,6% и снижает уровень избыточного обнаружения до ~14,2%.
2. **Комбинированный подход основан на уровне доверия**, который зависит от согласия между различными методами.

## Примеры использования

### Базовый пример запуска

```python
# Импорт необходимых модулей
from config import *
from data.download import setup_cdsapirc, download_era5_data_extended
from detection.algorithms import detect_cyclones_improved
from visualization.plots import visualize_cyclones_with_diagnostics

# Настройка окружения и API доступа
setup_cdsapirc('/path/to/config_dir')

# Загрузка данных ERA5 за указанный период
data_file = download_era5_data_extended(
    start_date="2020-01-01", 
    end_date="2020-01-31", 
    data_dir="/path/to/data"
)

# Обработка данных и обнаружение циклонов
import xarray as xr
ds = xr.open_dataset(data_file)
time_idx = 0  # Первый временной шаг
pressure, laplacian, cyclone_centers, cyclones_found, cyclone_mask, diagnostic_data = detect_cyclones_improved(
    ds, time_idx, 'time', 'msl', 'latitude', 'longitude', 
    cyclone_type='mesoscale'
)

# Визуализация результатов
if cyclones_found:
    visualize_cyclones_with_diagnostics(
        ds, time_idx, 'time', 'msl', 'latitude', 'longitude',
        '/path/to/output', cyclone_type='mesoscale'
    )
```

### Анализ временной последовательности

```python
from analysis.tracking import track_cyclones
from main import process_era5_data

# Обработка данных ERA5 с отслеживанием циклонов
cyclones = process_era5_data(
    file_path='path/to/era5_data.nc',
    output_dir='path/to/output',
    checkpoint_dir='path/to/checkpoints',
    model_dir='path/to/models',
    resume=True,
    save_diagnostic=True,
    use_daily_step=True,
    cyclone_type='mesoscale'
)

# Анализ траекторий циклонов
import pandas as pd
cyclones_df = pd.DataFrame(cyclones)
print(f"Всего обнаружено {len(cyclones)} циклонических систем")
print(f"Средняя продолжительность циклонов: {cyclones_df['duration'].mean():.1f} часов")
print(f"Среднее минимальное давление: {cyclones_df['pressure'].mean():.1f} гПа")
```

### Классификация по термической структуре

```python
from detection.thermal import classify_cyclones

# Классификация обнаруженных циклонов
classified_cyclones, stats = classify_cyclones(cyclone_centers, ds, time_idx, 'time')

# Анализ распределения типов
print(f"Холодноядерные циклоны: {stats['cold']}")
print(f"Теплоядерные циклоны: {stats['warm']}")
print(f"Смешанные структуры: {stats['mixed']}")
```

## Документация функций

### Загрузка данных

#### `download_era5_data_extended(start_date, end_date, data_dir, output_file='era5_arctic_data.nc')`

Загрузка расширенного набора данных ERA5 для арктического региона.

**Параметры:**
- `start_date`, `end_date`: Начальная и конечная даты в формате 'YYYY-MM-DD'
- `data_dir`: Директория для сохранения данных
- `output_file`: Имя выходного файла

**Возвращает:**
- Путь к загруженному файлу

**Пример:**
```python
from data.download import download_era5_data_extended

file_path = download_era5_data_extended(
    start_date="2020-01-01", 
    end_date="2020-01-31", 
    data_dir="/path/to/data",
    output_file="era5_arctic_jan2020.nc"
)
```

#### `inspect_netcdf(file_path)`

Подробная проверка структуры NetCDF файла.

**Параметры:**
- `file_path`: Путь к файлу NetCDF

**Возвращает:**
- Словарь с информацией о структуре файла

**Пример:**
```python
from data.download import inspect_netcdf

file_info = inspect_netcdf("/path/to/era5_data.nc")
print(f"Доступные переменные: {file_info['variables']}")
print(f"Временной диапазон: {file_info['time_min']} - {file_info['time_max']}")
```

### Алгоритмы детекции

#### `detect_cyclones_improved(ds, time_idx, time_dim, pressure_var, lat_var, lon_var, cyclone_params=None, cyclone_type='mesoscale')`

Оптимизированный алгоритм обнаружения арктических циклонов.

**Параметры:**
- `ds`: Набор данных xarray с полем приземного давления
- `time_idx`: Индекс временного шага
- `time_dim`: Имя измерения времени
- `pressure_var`: Имя переменной давления
- `lat_var`: Имя переменной широты
- `lon_var`: Имя переменной долготы
- `cyclone_params`: Словарь с параметрами алгоритма
- `cyclone_type`: Тип циклонов для обнаружения

**Возвращает:**
- `pressure`: Поле давления
- `laplacian`: Лапласиан давления
- `cyclone_centers`: Координаты центров циклонов
- `cyclones_found`: Найдены ли циклоны
- `cyclone_mask`: Маска обнаруженных циклонов
- `diagnostic_data`: Диагностические данные

**Пример:**
```python
from detection.algorithms import detect_cyclones_improved
import xarray as xr

ds = xr.open_dataset("era5_data.nc")
pressure, laplacian, cyclone_centers, cyclones_found, cyclone_mask, diagnostic_data = detect_cyclones_improved(
    ds, 0, 'time', 'msl', 'latitude', 'longitude', 
    cyclone_type='polar_low'
)

if cyclones_found:
    print(f"Обнаружено {len(cyclone_centers)} полярных мезоциклонов")
    for center in cyclone_centers:
        lat, lon, pressure, depth = center[:4]
        print(f"Координаты: {lat:.2f}°N, {lon:.2f}°E, Давление: {pressure:.1f} гПа, Глубина: {depth:.1f} гПа")
```

#### `calculate_laplacian_improved(pressure_field, dx, method='adaptive', smooth_sigma=0.7)`

Улучшенное вычисление лапласиана поля давления с адаптивными методами расчета.

**Параметры:**
- `pressure_field`: Двумерный массив значений давления
- `dx`: Пространственное разрешение сетки в метрах
- `method`: Метод расчета лапласиана ('standard', 'filtered', 'highorder', 'adaptive')
- `smooth_sigma`: Параметр сглаживания для методов 'filtered' и 'adaptive'

**Возвращает:**
- Поле лапласиана давления
- Дополнительные поля (градиент давления) для диагностики

### Классификация циклонов

#### `determine_thermal_structure(ds, lat_idx, lon_idx, lat_values, lon_values, time_idx, time_dim)`

Определяет термическую структуру циклона на основе распределения температуры на уровне 700 гПа.

**Параметры:**
- `ds`: Набор данных с полями давления и температуры
- `lat_idx`, `lon_idx`: Индексы сетки центра циклона
- `lat_values`, `lon_values`: Массивы значений широты и долготы
- `time_idx`: Индекс временного шага
- `time_dim`: Имя измерения времени

**Возвращает:**
- Тип термической структуры: 'cold', 'warm' или 'mixed'
- Дополнительная информация о термической структуре

**Пример:**
```python
from detection.thermal import determine_thermal_structure

lat_idx, lon_idx = 100, 150  # Пример индексов центра циклона
thermal_type, structure_info = determine_thermal_structure(
    ds, lat_idx, lon_idx, ds['latitude'].values, ds['longitude'].values, 0, 'time'
)

print(f"Термическая структура циклона: {thermal_type}")
if 'temp_diff' in structure_info:
    print(f"Разница температур: {structure_info['temp_diff']:.1f} K")
```

#### `classify_cyclones(cyclone_centers, ds, time_idx, time_dim)`

Классифицирует обнаруженные циклоны по термической структуре.

**Параметры:**
- `cyclone_centers`: Список центров циклонов
- `ds`: Набор данных с полями давления и температуры
- `time_idx`: Индекс временного шага
- `time_dim`: Имя измерения времени

**Возвращает:**
- Список циклонов с добавленной информацией о термической структуре
- Статистика по типам циклонов

### Отслеживание циклонов

#### `track_cyclones(cyclone_centers, previous_tracks, max_distance=300, hours_per_step=1)`

Отслеживает циклоны во времени, связывая текущие центры с предыдущими треками.

**Параметры:**
- `cyclone_centers`: Список центров циклонов на текущем временном шаге
- `previous_tracks`: Словарь с информацией о предыдущих треках
- `max_distance`: Максимальное расстояние для связывания центров (км)
- `hours_per_step`: Количество часов между временными шагами

**Возвращает:**
- Обновленный словарь треков
- Словарь связей между текущими центрами и треками

**Пример:**
```python
from analysis.tracking import track_cyclones

# Пример центров циклонов для текущего шага
current_centers = [
    (75.5, 45.2, 985.3, 4.2),  # lat, lon, pressure, depth
    (80.1, -30.5, 990.1, 3.5)
]

# Предыдущие треки
previous_tracks = {
    "75.3_45.0": {
        'start_time': '2020-01-01 00:00',
        'last_time': '2020-01-01 12:00',
        'first_lat': 75.3,
        'first_lon': 45.0,
        'last_lat': 75.4,
        'last_lon': 45.1,
        'positions': [(75.3, 45.0), (75.4, 45.1)],
        'times': ['2020-01-01 00:00', '2020-01-01 12:00'],
        'duration': 12
    }
}

updated_tracks, current_tracked = track_cyclones(
    current_centers, previous_tracks, max_distance=300, hours_per_step=6
)

print(f"Обновлено треков: {len(updated_tracks)}")
print(f"Связано центров с существующими треками: {len(current_tracked)}")
```

#### `haversine_distance(lat1, lon1, lat2, lon2)`

Вычисляет расстояние между точками на поверхности Земли по формуле гаверсинуса.

**Параметры:**
- `lat1`, `lon1`: Координаты первой точки (в градусах)
- `lat2`, `lon2`: Координаты второй точки или массивы координат (в градусах)

**Возвращает:**
- Расстояние в километрах или массив расстояний

**Пример:**
```python
from analysis.metrics import haversine_distance

# Расчет расстояния между двумя точками
dist = haversine_distance(75.5, 30.0, 76.0, 31.0)
print(f"Расстояние между точками: {dist:.2f} км")

# Векторизованный расчет расстояний
import numpy as np
lat2_array = np.array([75.0, 76.0, 77.0])
lon2_array = np.array([30.0, 31.0, 32.0])
distances = haversine_distance(75.5, 30.0, lat2_array, lon2_array)
print(f"Расстояния до массива точек: {distances}")
```

### Визуализация

#### `visualize_cyclones_with_diagnostics(ds, time_idx, time_dim, pressure_var, lat_var, lon_var, output_dir, cyclone_params=None, save_diagnostic=True, file_prefix='')`

Оптимизированная функция визуализации обнаруженных циклонов с расширенной диагностикой.

**Параметры:**
- `ds`: Набор данных xarray
- `time_idx`: Индекс временного шага
- `time_dim`: Имя измерения времени
- `pressure_var`: Имя переменной давления
- `lat_var`: Имя переменной широты
- `lon_var`: Имя переменной долготы
- `output_dir`: Директория для сохранения изображений
- `cyclone_params`: Словарь с параметрами алгоритма
- `save_diagnostic`: Флаг для сохранения диагностических изображений
- `file_prefix`: Префикс для имен файлов

**Возвращает:**
- Список обнаруженных центров циклонов
- Найдены ли циклоны
- Диагностические данные

**Пример:**
```python
from visualization.plots import visualize_cyclones_with_diagnostics

# Визуализация результатов с сохранением диагностики
cyclone_centers, cyclones_found, diag_data = visualize_cyclones_with_diagnostics(
    ds, 0, 'time', 'msl', 'latitude', 'longitude',
    '/path/to/output', save_diagnostic=True, file_prefix='polar_low_'
)
```

#### `create_cyclone_statistics(cyclones, output_dir, file_prefix='')`

Создает статистические визуализации для обнаруженных циклонов.

**Параметры:**
- `cyclones`: Список циклонов с их атрибутами
- `output_dir`: Директория для сохранения результатов
- `file_prefix`: Префикс для имен файлов

**Пример:**
```python
from visualization.plots import create_cyclone_statistics

# Создание статистических визуализаций
create_cyclone_statistics(cyclones, '/path/to/output', file_prefix='mesoscale_')
```

#### `visualize_detection_criteria()`

Создает наглядные изображения критериев обнаружения арктических циклонов на основе стандартов детекции.

**Пример:**
```python
from visualization.diagnostics import visualize_detection_criteria

# Создание диагностических визуализаций
visualize_detection_criteria()
```

## Типичные задачи

### Климатология арктических циклонов

Анализ многолетних данных для построения климатологии циклонов:

```python
import pandas as pd
import matplotlib.pyplot as plt
from main import process_era5_data

# Обработка данных за длительный период
all_cyclones = []
for year in range(2010, 2021):
    for month in range(1, 13):
        # Форматирование дат
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year+1}-01-01"
        else:
            end_date = f"{year}-{month+1:02d}-01"
            
        # Загрузка и обработка данных
        file_path = download_era5_data_extended(
            start_date=start_date,
            end_date=end_date,
            data_dir="/path/to/data",
            output_file=f"era5_{year}_{month:02d}.nc"
        )
        
        cyclones = process_era5_data(
            file_path=file_path,
            output_dir="/path/to/output",
            checkpoint_dir="/path/to/checkpoints",
            model_dir="/path/to/models",
            cyclone_type='mesoscale'
        )
        
        # Добавление информации о годе и месяце
        for cyclone in cyclones:
            cyclone['year'] = year
            cyclone['month'] = month
        
        all_cyclones.extend(cyclones)

# Создание DataFrame и анализ
df = pd.DataFrame(all_cyclones)
monthly_counts = df.groupby(['year', 'month']).size().unstack()

# Визуализация сезонного цикла
monthly_avg = monthly_counts.mean(axis=0)
plt.figure(figsize=(10, 6))
monthly_avg.plot(kind='bar')
plt.title('Сезонный цикл арктических мезоциклонов (2010-2020)')
plt.xlabel('Месяц')
plt.ylabel('Среднее количество циклонов')
plt.savefig('/path/to/output/seasonal_cycle.png', dpi=300)
```

### Анализ интенсивных циклонических событий

Поиск и анализ интенсивных циклонических событий:

```python
import pandas as pd
import xarray as xr
from visualization.plots import visualize_cyclones_with_diagnostics

# Загрузка данных о циклонах
cyclones_df = pd.read_csv('/path/to/output/detected_mesoscale_cyclones.csv')

# Фильтрация интенсивных циклонов
intense_cyclones = cyclones_df[
    (cyclones_df['pressure'] < 980) &  # Глубокие циклоны
    (cyclones_df['depth'] > 5.0) &      # Значительная глубина
    (cyclones_df['max_wind'] > 20.0)    # Сильный ветер
]

print(f"Найдено {len(intense_cyclones)} интенсивных циклонических событий")

# Детальный анализ наиболее интенсивного циклона
most_intense = intense_cyclones.sort_values('pressure').iloc[0]
print(f"Наиболее интенсивный циклон:")
print(f"Дата: {most_intense['time']}")
print(f"Координаты: {most_intense['latitude']:.2f}°N, {most_intense['longitude']:.2f}°E")
print(f"Давление: {most_intense['pressure']:.1f} гПа")
print(f"Максимальная скорость ветра: {most_intense['max_wind']:.1f} м/с")

# Визуализация этого события
time_str = most_intense['time']
data_file = f"/path/to/data/era5_{time_str[:10].replace('-', '')}.nc"
ds = xr.open_dataset(data_file)

# Находим индекс времени
time_values = ds['time'].values
time_idx = int(np.where(time_values == np.datetime64(time_str))[0][0])

# Создаем детальную визуализацию
visualize_cyclones_with_diagnostics(
    ds, time_idx, 'time', 'msl', 'latitude', 'longitude',
    '/path/to/output/intense_events', save_diagnostic=True,
    file_prefix='intense_'
)
```

### Статистический анализ термической структуры

Анализ распределения типов термической структуры циклонов:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detection.thermal import classify_cyclones
import xarray as xr

# Загрузка данных о циклонах
cyclones_df = pd.read_csv('/path/to/output/detected_mesoscale_cyclones.csv')

# Группировка данных по месяцам
cyclones_df['month'] = pd.to_datetime(cyclones_df['time']).dt.month

# Анализ термической структуры по месяцам
thermal_structure = []

# Загрузка данных
ds = xr.open_dataset('/path/to/data/era5_data.nc')

for idx, row in cyclones_df.iterrows():
    # Находим индексы в сетке
    lat_idx = np.abs(ds['latitude'].values - row['latitude']).argmin()
    lon_idx = np.abs(ds['longitude'].values - row['longitude']).argmin()
    time_idx = np.abs(pd.to_datetime(ds['time'].values) - pd.to_datetime(row['time'])).argmin()
    
    # Определяем термическую структуру
    core_type, _ = determine_thermal_structure(
        ds, lat_idx, lon_idx, ds['latitude'].values, ds['longitude'].values, time_idx, 'time'
    )
    
    thermal_structure.append({
        'time': row['time'],
        'month': row['month'],
        'latitude': row['latitude'],
        'longitude': row['longitude'],
        'pressure': row['pressure'],
        'core_type': core_type
    })

# Создание DataFrame
thermal_df = pd.DataFrame(thermal_structure)

# Анализ распределения типов по месяцам
monthly_types = thermal_df.groupby(['month', 'core_type']).size().unstack()
monthly_percentage = monthly_types.div(monthly_types.sum(axis=1), axis=0) * 100

# Визуализация
fig, ax = plt.subplots(figsize=(12, 8))
monthly_percentage.plot(kind='bar', stacked=True, ax=ax)
plt.title('Сезонное распределение типов термической структуры циклонов')
plt.xlabel('Месяц')
plt.ylabel('Доля циклонов (%)')
plt.legend(title='Тип структуры')
plt.savefig('/path/to/output/thermal_structure_seasonal.png', dpi=300)
```

