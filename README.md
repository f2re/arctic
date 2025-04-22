# Arctic Cyclones Detection and Analysis Toolkit (ACDAT)

**Комплексный инструментарий для автоматизированной детекции, классификации и анализа циклонических структур в Арктическом регионе на основе данных реанализа ERA5**

## Содержание

- [Введение](#введение)
- [Установка и зависимости](#установка-и-зависимости)
- [Архитектура проекта](#архитектура-проекта)
  - [Структура каталогов](#структура-каталогов)
  - [Процесс обработки данных](#процесс-обработки-данных)
  - [Взаимодействие модулей](#взаимодействие-модулей)
- [Научная методология](#научная-методология)
  - [Типы циклонов и параметры детекции](#типы-циклонов-и-параметры-детекции)
  - [Алгоритмы обнаружения](#алгоритмы-обнаружения)
  - [Термическая классификация](#термическая-классификация)
- [Конфигурационные параметры](#конфигурационные-параметры)
- [Руководство пользователя](#руководство-пользователя)
  - [Быстрый старт](#быстрый-старт)
  - [Базовые операции](#базовые-операции)
  - [Анализ результатов](#анализ-результатов)
- [Руководство по расширению функциональности](#руководство-по-расширению-функциональности)
  - [Реализация пользовательских алгоритмов детекции](#реализация-пользовательских-алгоритмов-детекции)
  - [Добавление новых метрик](#добавление-новых-метрик)
  - [Создание дополнительных типов визуализации](#создание-дополнительных-типов-визуализации)
  - [Интеграция альтернативных источников данных](#интеграция-альтернативных-источников-данных)
  - [Настройка параметров детекции](#настройка-параметров-детекции)
- [Документация API](#документация-api)
  - [Модуль загрузки данных](#модуль-загрузки-данных)
  - [Модуль обнаружения циклонов](#модуль-обнаружения-циклонов)
  - [Модуль анализа](#модуль-анализа)
  - [Модуль визуализации](#модуль-визуализации)
- [Типичные задачи](#типичные-задачи)


## Введение

ACDAT представляет собой набор инструментов для комплексного исследования циклонической активности в Арктическом регионе. Используя современные алгоритмы обработки данных реанализа ERA5, ACDAT обеспечивает автоматизированное обнаружение, классификацию и отслеживание циклонических образований различного масштаба.

Ключевые возможности инструментария:

- **Многокритериальная идентификация циклонов** с использованием параметра Лапласиана давления, проверки на замкнутость изобар, анализа градиента давления, оценки завихренности и скорости ветра
- **Масштабная дифференциация циклонических образований** - от синоптического масштаба до мезомасштабных полярных циклонов
- **Термическая классификация циклонов** (с холодным/теплым ядром) по вертикальной структуре
- **Отслеживание эволюции циклонов** во времени с построением траекторий и анализом жизненного цикла
- **Развитая система визуализации** для репрезентации пространственного распределения, структуры и динамики циклонических образований
- **Гибкая модульная архитектура**, позволяющая расширять функциональность и интегрировать новые методы анализа

ACDAT разработан для метеорологов, климатологов и исследователей полярных регионов, обеспечивая комплексный подход к задаче идентификации и анализа циклонов на основе современных научных методов.

## Установка и зависимости

Проект разработан для работы в среде Python 3.7+ и включает поддержку Google Colab для облачных вычислений. Для локальной установки требуются следующие библиотеки:

```bash
pip install -r requirements.txt
```

Файл `requirements.txt` содержит:

```
numpy>=1.21.0
pandas>=1.3.0
xarray>=0.20.0
matplotlib>=3.4.0
netCDF4>=1.5.7
scipy>=1.7.0
cdsapi>=0.5.1
scikit-image>=0.18.0
scikit-learn>=0.24.0
cartopy>=0.20.0
pyproj>=3.1.0
shapely>=1.7.0
tqdm>=4.62.0
```

### Настройка доступа к данным ERA5

Для загрузки данных реанализа ERA5 требуется зарегистрированный API-ключ Copernicus Climate Data Store:

1. Создайте учетную запись на [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
2. После регистрации перейдите в раздел "API Keys" в вашем профиле
3. Скопируйте ключ API и создайте файл `.cdsapirc` в домашней директории со следующим содержимым:

```
url: https://cds.climate.copernicus.eu/api
key: ваш-ключ-API
```

В ACDAT имеется функция `setup_cdsapirc()`, которая помогает создать этот файл:

```python
from data.download import setup_cdsapirc
setup_cdsapirc('/path/to/config_dir')
```

## Архитектура проекта

### Структура каталогов

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

### Процесс обработки данных

Процесс обработки данных в ACDAT включает следующие этапы:

1. **Загрузка данных**:
   - Определение необходимых переменных на основе выбранных методов обнаружения
   - Загрузка данных реанализа ERA5 с использованием API CDS
   - Проверка и анализ структуры загруженных данных

2. **Предобработка данных**:
   - Анализ масштаба сетки и её пространственного разрешения
   - Адаптация параметров обнаружения циклонов в зависимости от разрешения
   - Вычисление дополнительных полей (градиенты, лапласиан давления)

3. **Обнаружение циклонов**:
   - Применение многокритериального алгоритма обнаружения
   - Фильтрация потенциальных циклонических образований
   - Определение центров и характеристик обнаруженных циклонов

4. **Классификация и анализ**:
   - Классификация циклонов по термической структуре
   - Отслеживание циклонов во времени
   - Расчет статистических характеристик и метрик

5. **Визуализация результатов**:
   - Создание карт с обнаруженными циклонами
   - Визуализация диагностических полей
   - Генерация статистических диаграмм и графиков

### Взаимодействие модулей

```
                                  ┌─────────────┐
                                  │   main.py   │
                                  └──────┬──────┘
                                         │
              ┌──────────────────┬──────┴──────┬───────────────────┐
              │                  │             │                   │
     ┌────────▼──────┐   ┌──────▼───────┐    ┌▼──────────┐   ┌────▼───────┐
     │  data/         │   │  detection/  │    │ analysis/ │   │visualization/│
     │  download.py   │   │  algorithms.py│    │ tracking.py│   │  plots.py  │
     │  preprocessing.py│   │  thermal.py  │    │ metrics.py │   │diagnostics.py│
     └────────┬──────┘   └──────┬───────┘    └┬──────────┘   └────┬───────┘
              │                  │             │                   │
              └──────────────────┴─────────────┴───────────────────┘
                                         │
                                ┌────────▼─────────┐
                                │ Результаты анализа │
                                └──────────────────┘
```

Основные информационные потоки:

1. `main.py` оркестрирует общий процесс анализа
2. `data/download.py` загружает данные ERA5, которые затем обрабатываются модулями обнаружения
3. `detection/algorithms.py` реализует алгоритмы обнаружения циклонов
4. `detection/thermal.py` классифицирует циклоны по термической структуре
5. `analysis/tracking.py` отслеживает циклоны во времени
6. `analysis/metrics.py` рассчитывает метрики циклонов
7. `visualization/plots.py` и `visualization/diagnostics.py` создают визуализации результатов

## Научная методология

### Типы циклонов и параметры детекции

ACDAT поддерживает детекцию трех основных типов циклонов с различными пороговыми значениями и характеристиками:

#### 1. Синоптические циклоны (500-1500 км)

Крупномасштабные циклонические образования синоптического масштаба с характерным размером 500-1500 км.

```python
SYNOPTIC_CYCLONE_PARAMS = {
    'laplacian_threshold': -0.15,        # Порог лапласиана давления (гПа/м²)
    'min_pressure_threshold': 1015.0,    # Верхний порог давления (гПа)
    'min_size': 3,                       # Минимальный размер области (ячеек сетки)
    'min_depth': 2.5,                    # Минимальная глубина циклона (гПа)
    'smooth_sigma': 1.0,                 # Параметр сглаживания Гаусса
    'pressure_gradient_threshold': 0.7,  # Минимальный градиент давления (гПа/градус)
    'closed_contour_radius': 600,        # Радиус проверки замкнутости контура (км)
    'min_cyclone_radius': 200,           # Минимальный радиус циклона (км)
    'max_cyclone_radius': 1500,          # Максимальный радиус циклона (км)
    'min_wind_speed': 15.0,              # Минимальная скорость ветра (м/с)
    'min_vorticity': 0.5e-5,             # Минимальная относительная завихренность (с⁻¹)
    'min_duration': 0                    # Минимальная продолжительность (часов)
}
```

#### 2. Мезомасштабные циклоны (200-600 км)

Циклоны среднего масштаба с характерным размером 200-600 км.

```python
MESOSCALE_CYCLONE_PARAMS = {
    'laplacian_threshold': -0.12,        # Порог лапласиана для мезомасштабных структур
    'min_pressure_threshold': 1018.0,    # Верхний порог давления для мезомасштабных структур
    'min_size': 1,                       # Меньший минимальный размер для мезомасштабных структур
    'min_depth': 1.0,                    # Меньшая минимальная глубина для мезомасштабных структур
    'smooth_sigma': 0.8,                 # Меньшее сглаживание для сохранения деталей
    'pressure_gradient_threshold': 0.5,  # Меньший порог градиента давления
    'closed_contour_radius': 400,        # Меньший радиус проверки замкнутости контура
    'min_cyclone_radius': 100,           # Меньший минимальный радиус циклона
    'max_cyclone_radius': 600,           # Меньший максимальный радиус циклона
    'min_wind_speed': 15.0,              # Минимальная скорость ветра (м/с)
    'min_vorticity': 1.0e-5,             # Повышенное требование к завихренности для мезовихрей
    'min_duration': 24                   # Минимальная продолжительность (часов)
}
```

#### 3. Полярные мезоциклоны (полярные низкие, 50-400 км)

Малые циклонические образования, характерные для полярных регионов, с размером 50-400 км.

```python
POLAR_LOW_PARAMS = {
    'laplacian_threshold': -0.10,        # Более мягкий порог лапласиана для полярных низких
    'min_pressure_threshold': 1020.0,    # Более мягкий порог давления для полярных низких
    'min_size': 1,                       # Минимальный размер для мелких структур
    'min_depth': 0.8,                    # Минимальная глубина для мелких структур
    'smooth_sigma': 0.6,                 # Минимальное сглаживание для сохранения деталей
    'pressure_gradient_threshold': 0.4,  # Меньший порог градиента давления
    'closed_contour_radius': 300,        # Меньший радиус проверки замкнутости контура
    'min_cyclone_radius': 50,            # Меньший минимальный радиус циклона
    'max_cyclone_radius': 400,           # Меньший максимальный радиус циклона
    'min_wind_speed': 15.0,              # Минимальная скорость ветра (м/с)
    'min_vorticity': 1.5e-5,             # Высокое требование к завихренности для полярных низких
    'min_duration': 12                   # Минимальная продолжительность (часов)
}
```

### Алгоритмы обнаружения

ACDAT использует многокритериальный подход к обнаружению циклонов, который включает следующие методы:

#### 1. Анализ лапласиана давления

Лапласиан приземного давления $\nabla^2 p$ является ключевым параметром для идентификации циклонических структур. Отрицательные значения лапласиана указывают на наличие минимума давления, потенциально связанного с циклоном.

```python
# Вычисление лапласиана с адаптивным методом
def calculate_laplacian_improved(pressure_field, dx, method='adaptive', smooth_sigma=0.7):
    # Предварительная фильтрация для подавления шума
    pressure_filtered = gaussian_filter(pressure, sigma=smooth_sigma)

    # Стандартный 5-точечный шаблон
    laplacian_standard = np.zeros_like(pressure)
    laplacian_standard[1:-1, 1:-1] = (
        pressure_filtered[:-2, 1:-1] +
        pressure_filtered[2:, 1:-1] +
        pressure_filtered[1:-1, :-2] +
        pressure_filtered[1:-1, 2:] -
        4 * pressure_filtered[1:-1, 1:-1]
    ) / (dx**2)

    # 8-точечный шаблон высокого порядка
    laplacian_highorder = np.zeros_like(pressure)
    # ... (код реализации)
    
    # Адаптивное комбинирование результатов
    laplacian = weight_standard * laplacian_standard + weight_highorder * laplacian_highorder
    
    return laplacian, (gradient_magnitude, grad_x, grad_y)
```

#### 2. Проверка замкнутых изобар

Циклоны характеризуются системой замкнутых изобар вокруг центра низкого давления.

```python
def check_closed_contour_optimized(pressure, lat_idx, lon_idx, lat_values, lon_values, radius_km=300):
    # Получаем центральные координаты и давление
    center_lat = lat_values[lat_idx]
    center_lon = lon_values[lon_idx]
    center_pressure = pressure[lat_idx, lon_idx]
    
    # Создаем маску для региона с заданным радиусом
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing='ij')
    
    # Упрощенный расчет расстояний (в км)
    lat_distances = np.abs(lat_grid - center_lat) * 111.32
    lon_distances = np.abs(lon_grid - center_lon) * 111.32 * np.cos(np.radians(center_lat))
    distances = np.sqrt(lat_distances**2 + lon_distances**2)
    
    # Маска точек внутри заданного радиуса
    inner_mask = distances <= radius_km
    
    # Создаем маску периметра
    kernel = np.ones((3, 3), dtype=bool)
    expanded_mask = ndimage.binary_dilation(inner_mask, structure=kernel)
    perimeter_mask = expanded_mask & ~inner_mask
    
    # Давление на периметре
    perimeter_pressures = pressure[perimeter_mask]
    min_perimeter_pressure = np.min(perimeter_pressures)
    
    # Глубина циклона - разница между средним давлением на периметре и в центре
    perimeter_mean_pressure = np.mean(perimeter_pressures)
    depth = perimeter_mean_pressure - center_pressure
    
    # Условие замкнутого контура: давление в центре ниже минимального на периметре
    return center_pressure < min_perimeter_pressure, depth
```

#### 3. Параметр Okubo-Weiss

Параметр Okubo-Weiss (W) квантифицирует баланс между деформацией и завихренностью. Отрицательные значения W указывают на области, где доминирует вращение (потенциальные вихри).

```python
def calculate_okubo_weiss(u, v, dx, dy):
    # Вычисляем производные скорости
    du_dx = np.gradient(u, axis=1) / dx
    du_dy = np.gradient(u, axis=0) / dy
    dv_dx = np.gradient(v, axis=1) / dx
    dv_dy = np.gradient(v, axis=0) / dy
    
    # Нормальная деформация
    normal_strain = du_dx - dv_dy
    
    # Сдвиговая деформация
    shear_strain = dv_dx + du_dy
    
    # Относительная завихренность
    relative_vorticity = dv_dx - du_dy
    
    # Параметр Okubo-Weiss
    okubo_weiss = normal_strain**2 + shear_strain**2 - relative_vorticity**2
    
    return okubo_weiss, (normal_strain, shear_strain, relative_vorticity)
```

#### 4. Анализ градиента давления

Значительный градиент давления является важной характеристикой циклонов.

```python
def has_significant_pressure_gradient(gradient_magnitude, lat_idx, lon_idx, neighbor_radius=4, threshold=0.8):
    # Определяем границы окрестности
    lat_min = max(0, lat_idx - neighbor_radius)
    lat_max = min(gradient_magnitude.shape[0], lat_idx + neighbor_radius + 1)
    lon_min = max(0, lon_idx - neighbor_radius)
    lon_max = min(gradient_magnitude.shape[1], lon_idx + neighbor_radius + 1)
    
    # Выделяем окрестность
    neighborhood = gradient_magnitude[lat_min:lat_max, lon_min:lon_max]
    
    # Находим максимальный градиент в окрестности
    max_gradient = np.max(neighborhood)
    
    # Проверяем превышает ли максимальный градиент пороговое значение
    return max_gradient >= threshold, max_gradient
```

#### 5. Критерии скорости ветра и завихренности

Для более точной идентификации циклонов, особенно мезомасштабных, применяются дополнительные критерии скорости ветра и относительной завихренности:

```python
# Проверка скорости ветра
if has_wind_data and "max_wind" in potential_center:
    max_wind = potential_center["max_wind"]
    if max_wind < min_wind_speed:
        rejected_info = {
            "reason": "weak_wind",
            "wind_speed": max_wind,
            "threshold": min_wind_speed,
            # ...
        }
        rejected_centers.append(rejected_info)
        continue

# Проверка завихренности
if has_vorticity_data and "max_vorticity" in potential_center:
    max_vorticity = potential_center["max_vorticity"]
    if max_vorticity < min_vorticity:
        rejected_info = {
            "reason": "weak_vorticity",
            "vorticity": max_vorticity,
            "threshold": min_vorticity,
            # ...
        }
        rejected_centers.append(rejected_info)
        continue
```

### Термическая классификация

ACDAT классифицирует циклоны по их термической структуре на три типа:

1. **Холодноядерные циклоны** - характеризуются более низкой температурой в центре по сравнению с периферией
2. **Теплоядерные циклоны** - имеют более высокую температуру в центре
3. **Циклоны со смешанной структурой** - не имеют выраженного температурного контраста

```python
def determine_thermal_structure(ds, lat_idx, lon_idx, lat_values, lon_values, time_idx, time_dim):
    # Проверяем наличие данных температуры на 700 гПа
    # ...
    
    # Получаем центральную температуру и температуру на периферии
    center_temp = temp_field[lat_idx, lon_idx]
    periphery_temps = temp_field[periphery_mask]
    mean_periphery_temp = np.mean(periphery_temps)
    
    # Определяем порог для классификации
    threshold = 1.5  # K
    
    # Классифицируем на основе разницы температур
    temp_diff = center_temp - mean_periphery_temp
    
    if temp_diff > threshold:
        thermal_type = 'warm'  # Теплоядерный
    elif temp_diff < -threshold:
        thermal_type = 'cold'  # Холодноядерный
    else:
        thermal_type = 'mixed'  # Смешанный тип
        
    return thermal_type, structure_info
```

## Конфигурационные параметры

ACDAT поддерживает гибкую настройку параметров обнаружения циклонов через файл конфигурации `config.py`. Для различных типов циклонов определены отдельные наборы параметров, которые можно модифицировать в соответствии с конкретными исследовательскими задачами.

Базовые пути для сохранения данных и результатов также указываются в конфигурационном файле:

```python
DEFAULT_DATA_DIR = '~/arctic_git/data'
DEFAULT_IMAGE_DIR = '~/arctic_git/images'
DEFAULT_CHECKPOINT_DIR = '~/arctic_git/checkpoints'
DEFAULT_CRITERIA_DIR = '~/arctic_git/criteria_images'
```

Для выбора набора параметров используется функция `get_cyclone_params` из модуля `detection.parameters`:

```python
def get_cyclone_params(cyclone_type):
    """
    Возвращает параметры обнаружения для указанного типа циклонов.
    
    Параметры:
    ----------
    cyclone_type : str
        Тип циклонов: 'synoptic', 'mesoscale' или 'polar_low'
        
    Возвращает:
    -----------
    dict
        Словарь с параметрами обнаружения
    """
    if cyclone_type.lower() == 'synoptic':
        return SYNOPTIC_CYCLONE_PARAMS
    elif cyclone_type.lower() == 'mesoscale':
        return MESOSCALE_CYCLONE_PARAMS
    elif cyclone_type.lower() in ['polar_low', 'polar']:
        return POLAR_LOW_PARAMS
    else:
        # По умолчанию используем параметры для мезомасштабных циклонов
        print(f"Предупреждение: неизвестный тип циклонов '{cyclone_type}'. "
              f"Используются параметры для мезомасштабных циклонов.")
        return MESOSCALE_CYCLONE_PARAMS
```

## Руководство пользователя

### Быстрый старт

Для быстрого запуска анализа циклонов используйте функцию `main()` из скрипта `main.py`:

```python
# Импортируем необходимые модули
from main import main

# Запускаем процесс обнаружения циклонов
main()
```

По умолчанию будут использованы параметры из конфигурационного файла. Настройка параметров анализа выполняется в функции `main()`:

```python
# Настраиваемые параметры: период анализа
start_date = "2020-01-01"
end_date = "2020-01-31"  # Тестовый период - один месяц
region = [90, -180, 65, 180]  # [север, запад, юг, восток]
save_diagnostic = True  # Сохранять диагностические изображения
cyclone_type = 'mesoscale'  # Тип обнаруживаемых циклонов

# Выбор уровня детализации анализа
analysis_level = 'basic'  # Можно выбрать: 'basic', 'extended', 'comprehensive'

# Выбор методов обнаружения циклонов
detection_methods = ['laplacian', 'wind_speed','thermal', 'sst_gradient', 'gradient', 'vorticity']
```

### Базовые операции

#### 1. Загрузка данных ERA5

```python
from data.download import setup_cdsapirc, download_era5_data_extended

# Настройка доступа к API CDS
setup_cdsapirc('/path/to/config_dir')

# Загрузка данных ERA5 за указанный период
data_file = download_era5_data_extended(
    start_date="2020-01-01", 
    end_date="2020-01-31", 
    data_dir="/path/to/data",
    region=[90, -180, 65, 180]  # [север, запад, юг, восток]
)
```

#### 2. Обнаружение циклонов

```python
import xarray as xr
from detection.algorithms import detect_cyclones_improved

# Загрузка данных
ds = xr.open_dataset(data_file)

# Обнаружение циклонов для одного временного шага
time_idx = 0
pressure, laplacian, cyclone_centers, cyclones_found, cyclone_mask, diagnostic_data = detect_cyclones_improved(
    ds, time_idx, 'time', 'msl', 'latitude', 'longitude', 
    cyclone_type='mesoscale'
)

# Вывод информации о найденных циклонах
if cyclones_found:
    print(f"Обнаружено {len(cyclone_centers)} циклонов:")
    for i, center in enumerate(cyclone_centers):
        lat, lon, pressure_value, depth = center[:4]
        print(f"Циклон {i+1}: lat={lat:.2f}°N, lon={lon:.2f}°E, p={pressure_value:.1f} гПа, глубина={depth:.1f} гПа")
```

#### 3. Визуализация результатов

```python
from visualization.plots import visualize_cyclones_with_diagnostics

# Визуализация результатов
if cyclones_found:
    visualize_cyclones_with_diagnostics(
        ds, time_idx, 'time', 'msl', 'latitude', 'longitude',
        '/path/to/output', 
        save_diagnostic=True, 
        cyclone_type='mesoscale'
    )
```

### Анализ результатов

#### 1. Классификация циклонов по термической структуре

```python
from detection.thermal import classify_cyclones

# Классификация обнаруженных циклонов
classified_cyclones, stats = classify_cyclones(cyclone_centers, ds, time_idx, 'time')

# Анализ распределения типов
print(f"Холодноядерные циклоны: {stats['cold']}")
print(f"Теплоядерные циклоны: {stats['warm']}")
print(f"Смешанные структуры: {stats['mixed']}")
```

#### 2. Статистический анализ

```python
from visualization.plots import create_cyclone_statistics
import pandas as pd

# Создание DataFrame из списка циклонов
cyclones_df = pd.DataFrame(classified_cyclones)

# Создание статистических визуализаций
create_cyclone_statistics(cyclones_df.to_dict('records'), '/path/to/output', file_prefix='mesoscale_')
```

#### 3. Анализ траекторий циклонов

```python
import json

# Загрузка данных о траекториях циклонов
with open('/path/to/output/cyclone_tracks.json', 'r') as f:
    tracks_data = json.load(f)

# Анализ продолжительности циклонов
durations = [track['duration_hours'] for track in tracks_data]
print(f"Средняя продолжительность циклонов: {sum(durations)/len(durations):.1f} часов")
print(f"Максимальная продолжительность: {max(durations):.1f} часов")
```

## Руководство по расширению функциональности

### Реализация пользовательских алгоритмов детекции

Вы можете добавлять собственные алгоритмы обнаружения циклонов, реализуя их в модуле `detection/algorithms.py`. Общий подход включает следующие шаги:

1. Реализуйте функцию для вашего алгоритма обнаружения:

```python
# detection/algorithms.py

def detect_cyclones_with_my_algorithm(pressure_field, lat_values, lon_values, params):
    """
    Пользовательский алгоритм обнаружения циклонов.
    
    Параметры:
    ----------
    pressure_field : numpy.ndarray
        Двумерное поле давления
    lat_values, lon_values : numpy.ndarray
        Массивы значений широты и долготы
    params : dict
        Параметры алгоритма
        
    Возвращает:
    -----------
    list
        Список обнаруженных центров циклонов [(lat, lon, pressure, depth, ...), ...]
    """
    # Реализация алгоритма обнаружения
    cyclone_centers = []
    
    # Пример: анализ локальных минимумов давления
    from scipy import ndimage
    
    # Применение фильтра для выделения локальных минимумов
    min_filter = ndimage.minimum_filter(pressure_field, size=3)
    local_min = (pressure_field == min_filter)
    
    # Дополнительные критерии
    min_pressure_mask = pressure_field < params.get('min_pressure_threshold', 1015.0)
    cyclone_mask = local_min & min_pressure_mask
    
    # Находим координаты центров
    center_indices = np.where(cyclone_mask)
    
    for i in range(len(center_indices[0])):
        lat_idx, lon_idx = center_indices[0][i], center_indices[1][i]
        lat, lon = lat_values[lat_idx], lon_values[lon_idx]
        pressure = pressure_field[lat_idx, lon_idx]
        
        # Дополнительные характеристики
        depth = calculate_cyclone_depth(pressure_field, lat_idx, lon_idx)
        
        cyclone_centers.append((lat, lon, pressure, depth))
    
    return cyclone_centers
```

2. Интегрируйте ваш алгоритм в основную функцию `detect_cyclones_improved`:

```python
def detect_cyclones_improved(ds, time_idx, time_dim, pressure_var, lat_var, lon_var, 
                           cyclone_params=None, cyclone_type='mesoscale', detection_methods=None):
    # ... существующий код ...
    
    # Добавляем новый метод в список поддерживаемых методов
    if detection_methods is None:
        if cyclone_type == 'mesoscale':
            detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 
                                'gradient', 'vorticity', 'wind_speed', 'my_algorithm']
    
    # ... существующий код ...
    
    # Применяем новый метод обнаружения
    if 'my_algorithm' in detection_methods:
        custom_centers = detect_cyclones_with_my_algorithm(
            pressure_hpa, lat_values, lon_values, cyclone_params)
        
        # Объединяем результаты с основным списком
        for center in custom_centers:
            if center not in cyclone_centers:
                cyclone_centers.append(center)
    
    # ... существующий код ...
```

3. Обновите функцию `main.py` для включения вашего метода:

```python
# main.py

def main():
    # ... существующий код ...
    
    # Включаем новый метод в список методов обнаружения
    detection_methods = ['laplacian', 'pressure_minima', 'closed_contour', 
                         'gradient', 'vorticity', 'wind_speed', 'my_algorithm']
    
    # ... существующий код ...
```

### Добавление новых метрик

Для расширения аналитических возможностей ACDAT вы можете добавлять новые метрики в модуль `analysis/metrics.py`:

1. Реализуйте функцию для расчета новой метрики:

```python
# analysis/metrics.py

def calculate_cyclone_intensity_index(pressure, depth, radius, max_wind=None, max_vorticity=None):
    """
    Рассчитывает индекс интенсивности циклона на основе его параметров.
    
    Параметры:
    ----------
    pressure : float
        Давление в центре циклона (гПа)
    depth : float
        Глубина циклона (гПа)
    radius : float
        Радиус циклона (км)
    max_wind : float, optional
        Максимальная скорость ветра (м/с)
    max_vorticity : float, optional
        Максимальная завихренность (с⁻¹)
        
    Возвращает:
    -----------
    float
        Индекс интенсивности циклона
    """
    # Базовый индекс на основе глубины и размера
    base_index = depth * (1000 / pressure) * (100 / radius)
    
    # Учет скорости ветра, если доступна
    if max_wind is not None:
        base_index *= (max_wind / 10)
    
    # Учет завихренности, если доступна
    if max_vorticity is not None:
        base_index *= (max_vorticity / 1e-5) 
    
    return base_index
```

2. Интегрируйте метрику в процесс анализа, например, в функцию `process_era5_data` в `main.py`:

```python
# main.py
from analysis.metrics import calculate_cyclone_intensity_index

def process_era5_data(file_path, output_dir, checkpoint_dir, model_dir, resume=True, 
                  save_diagnostic=False, use_daily_step=True, cyclone_type='mesoscale',
                  detection_methods=None):
    # ... существующий код ...
    
    # В цикле обработки циклонов добавляем расчет новой метрики
    for i, cyclone_center in enumerate(cyclone_centers):
        lat, lon, pressure, depth, gradient, radius = cyclone_center[:6]
        
        # Дополнительные поля, если они есть
        max_wind = cyclone_center[6] if len(cyclone_center) > 6 else None
        max_vorticity = cyclone_center[7] if len(cyclone_center) > 7 else None
        
        # Расчет индекса интенсивности
        intensity_index = calculate_cyclone_intensity_index(
            pressure, depth, radius, max_wind, max_vorticity)
        
        # Добавляем индекс в информацию о циклоне
        cyclone_info = {
            'time': time_str,
            'latitude': lat,
            'longitude': lon,
            'pressure': pressure,
            'depth': depth,
            'gradient': gradient,
            'radius': radius,
            'intensity_index': intensity_index,  # Новая метрика
            'track_id': cyclone_id
        }
        
        # ... существующий код ...
```

3. Добавьте визуализацию новой метрики в модуль `visualization/plots.py`:

```python
# visualization/plots.py

def visualize_cyclone_intensity(cyclones, output_dir, file_prefix=''):
    """
    Создает визуализацию интенсивности циклонов.
    
    Параметры:
    ----------
    cyclones : list
        Список циклонов с рассчитанным индексом интенсивности
    output_dir : str
        Директория для сохранения результатов
    file_prefix : str
        Префикс для имен файлов
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Проверяем наличие данных
    if not cyclones or 'intensity_index' not in cyclones[0]:
        print("Нет данных об индексе интенсивности для визуализации")
        return
    
    # Извлекаем данные
    intensities = [c['intensity_index'] for c in cyclones]
    latitudes = [c['latitude'] for c in cyclones]
    
    # Создаем график зависимости интенсивности от широты
    plt.figure(figsize=(12, 8))
    plt.scatter(latitudes, intensities, c=intensities, cmap='viridis', 
               alpha=0.7, s=50)
    plt.colorbar(label='Индекс интенсивности')
    plt.xlabel('Широта (°N)')
    plt.ylabel('Индекс интенсивности')
    plt.title('Зависимость индекса интенсивности циклонов от широты')
    plt.grid(alpha=0.3)
    
    # Сохраняем график
    output_file = os.path.join(output_dir, f"{file_prefix}intensity_latitude.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Визуализация интенсивности циклонов сохранена: {output_file}")
```

### Создание дополнительных типов визуализации

ACDAT поддерживает различные виды визуализации. Для добавления нового типа выполните следующие шаги:

1. Реализуйте функцию визуализации в модуле `visualization/plots.py`:

```python
# visualization/plots.py

def visualize_cyclone_3d_structure(ds, cyclone_center, time_idx, time_dim, output_dir, file_prefix=''):
    """
    Создает 3D визуализацию вертикальной структуры циклона.
    
    Параметры:
    ----------
    ds : xarray.Dataset
        Набор данных с полями давления и температуры
    cyclone_center : tuple
        Координаты и параметры центра циклона (lat, lon, ...)
    time_idx : int
        Индекс временного шага
    time_dim : str
        Имя измерения времени
    output_dir : str
        Директория для сохранения результатов
    file_prefix : str
        Префикс для имен файлов
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import os
    
    lat, lon = cyclone_center[:2]
    
    # Получаем время
    if time_dim in ds.dims:
        current_time = ds[time_dim].values[time_idx]
        if isinstance(current_time, np.datetime64):
            time_str = np.datetime_as_string(current_time, unit='h').replace(':', '-')
            time_display = str(current_time)[:19]
        else:
            time_str = f"step_{time_idx}"
            time_display = f"Шаг {time_idx}"
    else:
        time_str = f"step_{time_idx}"
        time_display = f"Шаг {time_idx}"
    
    # Проверяем наличие необходимых переменных
    required_vars = ['t', 'z']
    missing_vars = [var for var in required_vars if var not in ds.variables]
    
    if missing_vars:
        print(f"Отсутствуют необходимые переменные для 3D визуализации: {', '.join(missing_vars)}")
        return
    
    # Получаем координаты
    lat_var = 'latitude' if 'latitude' in ds else 'lat'
    lon_var = 'longitude' if 'longitude' in ds else 'lon'
    lat_values = ds[lat_var].values
    lon_values = ds[lon_var].values
    
    # Находим ближайшие индексы в сетке
    lat_idx = np.abs(lat_values - lat).argmin()
    lon_idx = np.abs(lon_values - lon).argmin()
    
    # Получаем уровни давления
    if 'level' in ds.dims:
        levels = ds['level'].values
    elif 'pressure_level' in ds.dims:
        levels = ds['pressure_level'].values
    else:
        print("Не найдены уровни давления для 3D визуализации")
        return
    
    # Создаем 3D график
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Определяем размер окрестности циклона
    radius = 5  # ячеек сетки
    lat_min = max(0, lat_idx - radius)
    lat_max = min(lat_values.size, lat_idx + radius + 1)
    lon_min = max(0, lon_idx - radius)
    lon_max = min(lon_values.size, lon_idx + radius + 1)
    
    # Создаем сетку координат
    lats = lat_values[lat_min:lat_max]
    lons = lon_values[lon_min:lon_max]
    X, Y = np.meshgrid(lons, lats)
    
    # Определяем цветовую карту для температуры
    cmap = plt.cm.jet
    
    # Для каждого уровня давления рисуем поверхность
    for i, level in enumerate(levels):
        if i % 2 == 0:  # Пропускаем каждый второй уровень для наглядности
            continue
        
        # Получаем температуру на этом уровне
        if time_dim in ds.dims and 'level' in ds.dims:
            temp = ds['t'].isel({time_dim: time_idx, 'level': i})
        elif time_dim in ds.dims and 'pressure_level' in ds.dims:
            temp = ds['t'].isel({time_dim: time_idx, 'pressure_level': i})
        elif 'level' in ds.dims:
            temp = ds['t'].isel({'level': i})
        else:
            temp = ds['t'].isel({'pressure_level': i})
        
        # Выбираем окрестность циклона
        temp_slice = temp[lat_min:lat_max, lon_min:lon_max].values
        
        # Приводим уровень к километрам для лучшей визуализации
        Z = np.ones_like(X) * (1000 - level) / 100  # приближенная высота в км
        
        # Нормализация температуры для цветовой шкалы
        norm_temp = (temp_slice - np.min(temp_slice)) / (np.max(temp_slice) - np.min(temp_slice))
        
        # Отрисовка поверхности
        ax.plot_surface(X, Y, Z, facecolors=cmap(norm_temp), alpha=0.7, shade=True)
    
    # Отмечаем центр циклона
    for level in levels[::2]:
        z_val = (1000 - level) / 100
        ax.scatter([lon], [lat], [z_val], color='red', s=50, marker='o')
    
    # Соединяем центры на разных уровнях линией
    center_z = [(1000 - level) / 100 for level in levels[::2]]
    ax.plot([lon] * len(center_z), [lat] * len(center_z), center_z, color='red', linestyle='-', linewidth=2)
    
    # Настройка графика
    ax.set_xlabel('Долгота')
    ax.set_ylabel('Широта')
    ax.set_zlabel('Высота (км)')
    ax.set_title(f'3D структура циклона на {time_display}\nКоординаты: {lat:.2f}°N, {lon:.2f}°E')
    
    # Добавляем цветовую шкалу
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Относительная температура')
    
    # Сохраняем результат
    output_file = os.path.join(output_dir, f"{file_prefix}3d_structure_{time_str}_lat{lat:.2f}_lon{lon:.2f}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"3D визуализация структуры циклона сохранена: {output_file}")
```

2. Используйте новую функцию визуализации в основном процессе анализа:

```python
# main.py
from visualization.plots import visualize_cyclone_3d_structure

def process_era5_data(file_path, output_dir, checkpoint_dir, model_dir, resume=True, 
                     save_diagnostic=False, use_daily_step=True, cyclone_type='mesoscale',
                     detection_methods=None):
    # ... существующий код ...
    
    # После обнаружения циклонов
    if cyclones_found:
        # Стандартная визуализация
        visualize_cyclones_with_diagnostics(
            ds, time_idx, time_dim, pressure_var, lat_var, lon_var,
            output_dir, cyclone_params=cyclone_params,
            save_diagnostic=save_diagnostic, file_prefix="meso_",
            detection_methods=detection_methods
        )
        
        # Для каждого циклона создаем 3D визуализацию
        for cyclone_center in cyclone_centers:
            visualize_cyclone_3d_structure(
                ds, cyclone_center, time_idx, time_dim, 
                output_dir, file_prefix=f"{cyclone_type}_3d_"
            )
```

### Интеграция альтернативных источников данных

ACDAT можно расширить для работы с альтернативными источниками данных помимо ERA5. Для этого необходимо реализовать новые функции загрузки данных в модуле `data/download.py`:

1. Реализуйте функцию загрузки новых данных:

```python
# data/download.py

def download_merra2_data(start_date, end_date, data_dir, output_file='merra2_arctic_data.nc', region=None):
    """
    Загрузка данных реанализа MERRA-2 для арктического региона.
    
    Параметры:
    ----------
    start_date : str
        Начальная дата в формате 'YYYY-MM-DD'
    end_date : str
        Конечная дата в формате 'YYYY-MM-DD'
    data_dir : str
        Директория для сохранения данных
    output_file : str, optional
        Имя выходного файла
    region : list, optional
        Регион в формате [север, запад, юг, восток]
        
    Возвращает:
    -----------
    str
        Путь к загруженному файлу
    """
    import os
    import requests
    import xarray as xr
    from datetime import datetime, timedelta
    
    # Установка региона по умолчанию, если не указан
    if region is None:
        region = [90, -180, 65, 180]  # [север, запад, юг, восток]
    
    print(f"Загрузка данных MERRA-2 для периода {start_date} - {end_date}")
    print(f"Регион: {region}")
    
    # Создаем директорию, если она не существует
    os.makedirs(data_dir, exist_ok=True)
    
    # Полный путь к файлу
    file_path = os.path.join(data_dir, output_file)
    
    # Проверяем, существует ли уже файл
    if os.path.exists(file_path):
        print(f"Файл {file_path} уже существует, используем его")
        return file_path
    
    # Преобразование дат в объекты datetime
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Формирование списка дат для загрузки
    dates = []
    current_dt = start_dt
    while current_dt <= end_dt:
        dates.append(current_dt.strftime('%Y%m%d'))
        current_dt += timedelta(days=1)
    
    # Временные файлы для каждой даты
    temp_files = []
    
    # Загрузка данных для каждой даты
    for date in dates:
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        
        # URL для загрузки данных MERRA-2 (пример)
        url = f"https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4/{year}/{month}/MERRA2_400.inst3_3d_asm_Nv.{year}{month}{day}.nc4"
        
        # Имя временного файла
        temp_file = os.path.join(data_dir, f"temp_merra2_{date}.nc")
        temp_files.append(temp_file)
        
        try:
            # Здесь должна быть логика загрузки с учетом аутентификации
            # Для MERRA-2 требуется аутентификация через NASA Earthdata
            # Пример кода может выглядеть так:
            """
            import requests
            
            # Настройка сессии с авторизацией
            session = requests.Session()
            session.auth = ('username', 'password')
            
            # Загрузка файла
            response = session.get(url, stream=True)
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            """
            
            print(f"Данные для {date} загружены")
        except Exception as e:
            print(f"Ошибка при загрузке данных для {date}: {e}")
    
    # Объединение временных файлов
    if temp_files:
        try:
            # Открываем все файлы
            datasets = [xr.open_dataset(file) for file in temp_files]
            
            # Объединяем по измерению времени
            combined_ds = xr.concat(datasets, dim='time')
            
            # Сохраняем объединенный датасет
            combined_ds.to_netcdf(file_path)
            
            # Закрываем датасеты
            for ds in datasets:
                ds.close()
            
            # Удаляем временные файлы
            for file in temp_files:
                os.remove(file)
            
            print(f"Данные успешно объединены и сохранены в {file_path}")
        except Exception as e:
            print(f"Ошибка при объединении файлов: {e}")
            return None
    else:
        print("Нет данных для объединения")
        return None
    
    return file_path
```

2. Обновите функцию `main()` для использования нового источника данных:

```python
# main.py
from data.download import download_merra2_data

def main():
    # ... существующий код ...
    
    # Выбор источника данных
    data_source = 'merra2'  # Можно выбрать: 'era5', 'merra2'
    
    # ... существующий код ...
    
    # Загрузка данных из выбранного источника
    if data_source == 'era5':
        file_path = download_era5_data_extended(
            start_date=start_date, 
            end_date=end_date, 
            data_dir=data_dir, 
            output_file=output_file,
            region=region,
            era5_data_info=era5_data_info
        )
    elif data_source == 'merra2':
        file_path = download_merra2_data(
            start_date=start_date, 
            end_date=end_date, 
            data_dir=data_dir, 
            output_file=output_file.replace('era5', 'merra2'),
            region=region
        )
    
    # ... существующий код ...
```

3. При необходимости адаптируйте функции предобработки для нового источника данных в `data/preprocessing.py`.

### Настройка параметров детекции

ACDAT поддерживает гибкую настройку параметров обнаружения циклонов. Вы можете адаптировать параметры под конкретные задачи, модифицируя конфигурационный файл `config.py` или создавая новые наборы параметров:

1. Создайте новый набор параметров в `config.py`:

```python
# config.py

# Параметры для обнаружения арктических штормов
ARCTIC_STORM_PARAMS = {
    'laplacian_threshold': -0.18,        # Более строгий порог лапласиана
    'min_pressure_threshold': 990.0,     # Более низкий порог давления
    'min_size': 5,                       # Больший минимальный размер области
    'min_depth': 5.0,                    # Большая минимальная глубина
    'smooth_sigma': 1.2,                 # Более сильное сглаживание
    'pressure_gradient_threshold': 1.0,  # Более высокий градиент давления
    'closed_contour_radius': 800,        # Больший радиус проверки
    'neighbor_search_radius': 8,         # Больший радиус поиска
    'use_topology_check': True,          # Использовать топологический анализ
    'min_cyclone_radius': 300,           # Больший минимальный радиус
    'max_cyclone_radius': 2000,          # Больший максимальный радиус
    'max_cyclones_per_map': 5,           # Меньше циклонов на карте
    'min_wind_speed': 25.0,              # Более высокая скорость ветра
    'min_vorticity': 1.0e-5,             # Большая завихренность
    'min_duration': 48                   # Больший период существования
}
```

2. Обновите функцию `get_cyclone_params()` в `detection/parameters.py` для поддержки нового типа:

```python
# detection/parameters.py

def get_cyclone_params(cyclone_type):
    """
    Возвращает параметры обнаружения для указанного типа циклонов.
    
    Параметры:
    ----------
    cyclone_type : str
        Тип циклонов: 'synoptic', 'mesoscale', 'polar_low' или 'arctic_storm'
        
    Возвращает:
    -----------
    dict
        Словарь с параметрами обнаружения
    """
    if cyclone_type.lower() == 'synoptic':
        return SYNOPTIC_CYCLONE_PARAMS
    elif cyclone_type.lower() == 'mesoscale':
        return MESOSCALE_CYCLONE_PARAMS
    elif cyclone_type.lower() in ['polar_low', 'polar']:
        return POLAR_LOW_PARAMS
    elif cyclone_type.lower() == 'arctic_storm':
        return ARCTIC_STORM_PARAMS
    else:
        # По умолчанию используем параметры для мезомасштабных циклонов
        print(f"Предупреждение: неизвестный тип циклонов '{cyclone_type}'. "
              f"Используются параметры для мезомасштабных циклонов.")
        return MESOSCALE_CYCLONE_PARAMS
```

3. Используйте новый набор параметров в функции `main()`:

```python
# main.py

def main():
    # ... существующий код ...
    
    # Изменяем тип обнаруживаемых циклонов
    cyclone_type = 'arctic_storm'
    
    # ... существующий код ...
```

## Документация API

### Модуль загрузки данных

#### `setup_cdsapirc(arctic_dir)`

Настраивает файл конфигурации для доступа к API Climate Data Store.

**Параметры:**
- `arctic_dir` (str): Директория для сохранения файла конфигурации

**Пример:**
```python
from data.download import setup_cdsapirc

setup_cdsapirc('/path/to/config')
```

#### `download_era5_data_extended(start_date, end_date, data_dir, output_file='era5_arctic_data.nc', region=None, era5_data_info=None, detection_methods=None, analysis_level='basic')`

Загружает данные реанализа ERA5 для заданного периода и региона.

**Параметры:**
- `start_date`, `end_date` (str): Начальная и конечная даты в формате 'YYYY-MM-DD'
- `data_dir` (str): Директория для сохранения данных
- `output_file` (str): Имя выходного файла (по умолчанию 'era5_arctic_data.nc')
- `region` (list): Регион в формате [север, запад, юг, восток]
- `era5_data_info` (dict): Информация о необходимых данных ERA5
- `detection_methods` (list): Список методов обнаружения циклонов
- `analysis_level` (str): Уровень детализации анализа ('basic', 'extended', 'comprehensive')

**Возвращает:**
- (str): Путь к загруженному файлу

**Пример:**
```python
from data.download import download_era5_data_extended

file_path = download_era5_data_extended(
    start_date="2020-01-01", 
    end_date="2020-01-31", 
    data_dir="/path/to/data",
    region=[90, -180, 65, 180],
    detection_methods=['laplacian', 'pressure_minima', 'gradient', 'vorticity']
)
```

#### `inspect_netcdf(file_path)`

Анализирует структуру файла NetCDF и выводит информацию о его содержимом.

**Параметры:**
- `file_path` (str): Путь к файлу NetCDF

**Возвращает:**
- (dict): Словарь с информацией о структуре файла

**Пример:**
```python
from data.download import inspect_netcdf

file_info = inspect_netcdf("/path/to/data.nc")
print(f"Временное измерение: {file_info.get('time_dim')}")
print(f"Переменные давления: {file_info.get('pressure_vars', [])}")
```

### Модуль обнаружения циклонов

#### `detect_cyclones_improved(ds, time_idx, time_dim, pressure_var, lat_var, lon_var, cyclone_params=None, cyclone_type='mesoscale', detection_methods=None)`

Обнаруживает циклоны на основе многокритериального анализа.

**Параметры:**
- `ds` (xarray.Dataset): Набор данных с полем приземного давления
- `time_idx` (int): Индекс временного шага
- `time_dim` (str): Имя измерения времени
- `pressure_var` (str): Имя переменной давления
- `lat_var` (str): Имя переменной широты
- `lon_var` (str): Имя переменной долготы
- `cyclone_params` (dict): Словарь с параметрами алгоритма
- `cyclone_type` (str): Тип циклонов для обнаружения
- `detection_methods` (list): Список методов обнаружения циклонов

**Возвращает:**
- (tuple): (давление, лапласиан, координаты центров циклонов, найдены ли циклоны, маска, диагностические данные)

**Пример:**
```python
import xarray as xr
from detection.algorithms import detect_cyclones_improved

ds = xr.open_dataset("era5_data.nc")
pressure, laplacian, cyclone_centers, cyclones_found, cyclone_mask, diagnostic_data = detect_cyclones_improved(
    ds, 0, 'time', 'msl', 'latitude', 'longitude', 
    cyclone_type='mesoscale',
    detection_methods=['laplacian', 'pressure_minima', 'closed_contour', 'gradient']
)
```

#### `determine_thermal_structure(ds, lat_idx, lon_idx, lat_values, lon_values, time_idx, time_dim)`

Определяет термическую структуру циклона.

**Параметры:**
- `ds` (xarray.Dataset): Набор данных с полями давления и температуры
- `lat_idx`, `lon_idx` (int): Индексы сетки центра циклона
- `lat_values`, `lon_values` (numpy.ndarray): Массивы значений широты и долготы
- `time_idx` (int): Индекс временного шага
- `time_dim` (str): Имя измерения времени

**Возвращает:**
- (str): Тип термической структуры: 'cold', 'warm' или 'mixed'
- (dict): Дополнительная информация о термической структуре

**Пример:**
```python
from detection.thermal import determine_thermal_structure

thermal_type, structure_info = determine_thermal_structure(
    ds, lat_idx, lon_idx, ds['latitude'].values, ds['longitude'].values, 0, 'time'
)
print(f"Термическая структура циклона: {thermal_type}")
```

### Модуль анализа

#### `track_cyclones(cyclone_centers, previous_tracks, max_distance=300, hours_per_step=1)`

Отслеживает циклоны во времени, связывая текущие центры с предыдущими треками.

**Параметры:**
- `cyclone_centers` (list): Список центров циклонов на текущем шаге
- `previous_tracks` (dict): Словарь с информацией о предыдущих треках
- `max_distance` (float): Максимальное расстояние для связывания (км)
- `hours_per_step` (int): Количество часов между шагами

**Возвращает:**
- (dict): Обновленный словарь треков
- (dict): Словарь связей между текущими центрами и треками

**Пример:**
```python
from analysis.tracking import track_cyclones

# Текущие центры циклонов
current_centers = [(75.5, 45.2, 985.3, 4.2), (80.1, -30.5, 990.1, 3.5)]

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
```

#### `haversine_distance(lat1, lon1, lat2, lon2)`

Вычисляет расстояние между точками на поверхности Земли.

**Параметры:**
- `lat1`, `lon1` (float): Координаты первой точки (в градусах)
- `lat2`, `lon2` (float или numpy.ndarray): Координаты второй точки (в градусах)

**Возвращает:**
- (float или numpy.ndarray): Расстояние в километрах

**Пример:**
```python
from analysis.metrics import haversine_distance
import numpy as np

# Расчет расстояния между двумя точками
dist = haversine_distance(75.5, 30.0, 76.0, 31.0)
print(f"Расстояние: {dist:.2f} км")

# Векторизованный расчет
lat2_array = np.array([75.0, 76.0, 77.0])
lon2_array = np.array([30.0, 31.0, 32.0])
distances = haversine_distance(75.5, 30.0, lat2_array, lon2_array)
```

### Модуль визуализации

#### `visualize_cyclones_with_diagnostics(ds, time_idx, time_dim, pressure_var, lat_var, lon_var, output_dir, cyclone_params=None, save_diagnostic=True, file_prefix='', detection_methods=None)`

Создает визуализацию обнаруженных циклонов с диагностическими полями.

**Параметры:**
- `ds` (xarray.Dataset): Набор данных
- `time_idx` (int): Индекс временного шага
- `time_dim` (str): Имя измерения времени
- `pressure_var` (str): Имя переменной давления
- `lat_var` (str): Имя переменной широты
- `lon_var` (str): Имя переменной долготы
- `output_dir` (str): Директория для сохранения изображений
- `cyclone_params` (dict): Словарь с параметрами алгоритма
- `save_diagnostic` (bool): Флаг для сохранения диагностики
- `file_prefix` (str): Префикс для имен файлов
- `detection_methods` (list): Список методов обнаружения

**Возвращает:**
- (tuple): (список центров циклонов, найдены ли циклоны, диагностические данные)

**Пример:**
```python
from visualization.plots import visualize_cyclones_with_diagnostics

cyclone_centers, cyclones_found, diag_data = visualize_cyclones_with_diagnostics(
    ds, 0, 'time', 'msl', 'latitude', 'longitude',
    '/path/to/output', save_diagnostic=True, file_prefix='meso_',
    detection_methods=['laplacian', 'pressure_minima', 'gradient']
)
```

#### `create_cyclone_statistics(cyclones, output_dir, file_prefix='')`

Создает статистические визуализации для обнаруженных циклонов.

**Параметры:**
- `cyclones` (list): Список циклонов с их атрибутами
- `output_dir` (str): Директория для сохранения результатов
- `file_prefix` (str): Префикс для имен файлов

**Пример:**
```python
from visualization.plots import create_cyclone_statistics

create_cyclone_statistics(
    cyclones, '/path/to/output', file_prefix='polar_low_'
)
```

#### `visualize_detection_criteria(detection_methods=None)`

Создает наглядные изображения критериев обнаружения циклонов.

**Параметры:**
- `detection_methods` (list): Список методов обнаружения для визуализации

**Пример:**
```python
from visualization.diagnostics import visualize_detection_criteria

# Визуализация всех критериев
visualize_detection_criteria()

# Визуализация выбранных критериев
visualize_detection_criteria(['laplacian', 'pressure_minima', 'gradient'])
```

## Типичные задачи

### Климатология арктических циклонов

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from main import process_era5_data
from data.download import download_era5_data_extended

# Годы для анализа
start_year = 2010
end_year = 2020

# Загрузка и обработка данных для каждого месяца
all_cyclones = []

for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        # Форматирование дат
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year+1}-01-01"
        else:
            end_date = f"{year}-{month+1:02d}-01"
            
        # Загрузка данных
        file_path = download_era5_data_extended(
            start_date=start_date,
            end_date=end_date,
            data_dir="/path/to/data",
            output_file=f"era5_{year}_{month:02d}.nc"
        )
        
        # Обработка данных и обнаружение циклонов
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

# Создание DataFrame для анализа
df = pd.DataFrame(all_cyclones)

# Анализ сезонного цикла
monthly_counts = df.groupby(['month']).size()
monthly_avg_by_year = df.groupby(['year', 'month']).size().unstack()
monthly_avg = monthly_avg_by_year.mean(axis=0)

# Визуализация сезонного цикла
plt.figure(figsize=(12, 8))
monthly_avg.plot(kind='bar', yerr=monthly_avg_by_year.std(axis=0))
plt.title('Сезонный цикл арктических мезоциклонов (2010-2020)')
plt.xlabel('Месяц')
plt.ylabel('Среднее количество циклонов')
plt.xticks(range(12), ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек'])
plt.grid(alpha=0.3)
plt.savefig('/path/to/output/seasonal_cycle.png', dpi=300)
plt.close()

# Пространственное распределение циклонов
plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines()

# Плотность циклонов
h, xedges, yedges = np.histogram2d(
    df['longitude'], df['latitude'], 
    bins=[36, 18], 
    range=[[-180, 180], [65, 90]]
)

lon_grid, lat_grid = np.meshgrid(
    (xedges[:-1] + xedges[1:]) / 2, 
    (yedges[:-1] + yedges[1:]) / 2
)

cs = ax.contourf(lon_grid, lat_grid, h.T, 
                transform=ccrs.PlateCarree(), 
                cmap='YlOrRd', alpha=0.8,
                levels=10)

plt.colorbar(cs, ax=ax, label='Количество циклонов')
plt.title('Пространственное распределение арктических мезоциклонов (2010-2020)')
plt.savefig('/path/to/output/spatial_distribution.png', dpi=300)
plt.close()
```

### Анализ интенсивных циклонических событий

```python
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from visualization.plots import visualize_cyclones_with_diagnostics

# Загрузка данных о циклонах
cyclones_df = pd.read_csv('/path/to/output/detected_mesoscale_cyclones.csv')

# Фильтрация интенсивных циклонов
intense_cyclones = cyclones_df[
    (cyclones_df['pressure'] < 980) &  # Глубокие циклоны
    (cyclones_df['depth'] > 5.0) &     # Значительная глубина
    (cyclones_df['max_wind'] > 20.0)   # Сильный ветер
]

print(f"Найдено {len(intense_cyclones)} интенсивных циклонических событий")

# Детальный анализ наиболее интенсивного циклона
most_intense = intense_cyclones.sort_values('pressure').iloc[0]
print(f"Наиболее интенсивный циклон:")
print(f"Дата: {most_intense['time']}")
print(f"Координаты: {most_intense['latitude']:.2f}°N, {most_intense['longitude']:.2f}°E")
print(f"Давление: {most_intense['pressure']:.1f} гПа")
print(f"Глубина: {most_intense['depth']:.1f} гПа")
print(f"Максимальная скорость ветра: {most_intense['max_wind']:.1f} м/с")

# Визуализация траектории циклона
track_id = most_intense['track_id']
track_points = cyclones_df[cyclones_df['track_id'] == track_id]

# Сортировка по времени
track_points = track_points.sort_values('time')

# Создание карты траектории
plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.NorthPolarStereo())
ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines()

# Отрисовка траектории
lats = track_points['latitude'].values
lons = track_points['longitude'].values
pressures = track_points['pressure'].values

# Отрисовка линии траектории
ax.plot(lons, lats, 'r-', transform=ccrs.PlateCarree(), linewidth=2)

# Отрисовка точек с цветовой кодировкой давления
sc = ax.scatter(lons, lats, c=pressures, cmap='viridis_r', 
              transform=ccrs.PlateCarree(), s=100, edgecolor='k')

# Отмечаем начальную и конечную точки
ax.plot(lons[0], lats[0], 'go', transform=ccrs.PlateCarree(), 
      markersize=12, label='Начало')
ax.plot(lons[-1], lats[-1], 'bo', transform=ccrs.PlateCarree(), 
      markersize=12, label='Конец')

# Добавляем цветовую шкалу и легенду
plt.colorbar(sc, ax=ax, label='Давление (гПа)')
plt.legend()

plt.title(f'Траектория интенсивного циклона\n{track_points["time"].iloc[0]} - {track_points["time"].iloc[-1]}')
plt.savefig('/path/to/output/intense_cyclone_track.png', dpi=300)
plt.close()

# Анализ изменения параметров циклона со временем
plt.figure(figsize=(15, 10))

# Подготовка временной оси
times = pd.to_datetime(track_points['time'])
time_idx = range(len(times))

# График изменения давления
plt.subplot(3, 1, 1)
plt.plot(time_idx, track_points['pressure'], 'b-', linewidth=2)
plt.ylabel('Давление (гПа)')
plt.title('Изменение параметров циклона со временем')
plt.grid(alpha=0.3)

# График изменения скорости ветра
plt.subplot(3, 1, 2)
plt.plot(time_idx, track_points['max_wind'], 'r-', linewidth=2)
plt.ylabel('Макс. скорость ветра (м/с)')
plt.grid(alpha=0.3)

# График изменения радиуса
plt.subplot(3, 1, 3)
plt.plot(time_idx, track_points['radius'], 'g-', linewidth=2)
plt.ylabel('Радиус (км)')
plt.xlabel('Временные шаги')
plt.xticks(time_idx, [t.strftime('%d-%H:%M') for t in times], rotation=45)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/path/to/output/intense_cyclone_evolution.png', dpi=300)
plt.close()
```
