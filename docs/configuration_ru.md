# Подробная документация по конфигурации ArcticCyclone

## Содержание

1. [Обзор конфигурации](#обзор-конфигурации)
2. [Параметры данных](#параметры-данных)
3. [Параметры обнаружения](#параметры-обнаружения)
4. [Параметры трекинга](#параметры-трекинга)
5. [Параметры визуализации](#параметры-визуализации)
6. [Параметры экспорта](#параметры-экспорта)

## Обзор конфигурации

Файл конфигурации `config.yaml` определяет все аспекты работы системы ArcticCyclone. Он разделен на несколько секций, каждая из которых отвечает за определенную функциональность системы.

Пример полного файла конфигурации:

```yaml
# ArcticCyclone Configuration File

# Параметры данных
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

# Параметры обнаружения
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

# Параметры визуализации
visualization:
  default_projection: 'NorthPolarStereo'
  map_resolution: 'intermediate'
  output_dir: 'output/figures'
  default_dpi: 300
  cyclone_marker_size: 30
  track_line_width: 1.5
  color_map: 'viridis'
  map_features:
    coastlines: true
    countries: true
    grid_lines: true
    grid_labels: true

# Параметры экспорта
export:
  output_dir: 'output/data'
  formats: ['csv', 'netcdf']
  default_format: 'csv'
  csv:
    delimiter: ','
    include_header: true
  netcdf:
    compression_level: 4
    include_metadata: true

# Параметры трекинга
tracking:
  max_distance: 600.0
  max_time_gap: 18.0
  max_pressure_change: 25.0
  min_track_duration: 12.0
  min_track_points: 4
  debug_save_csv: false
  debug_dir: "debug"
  algorithm: "nearest_neighbor"
  cost_weights:
    distance: 1.0
    time: 1.0
    pressure: 0.5
```

## Параметры данных

### data.default_source

**Тип**: `string`  
**Значение по умолчанию**: `'ERA5'`  
**Описание**: Имя источника данных, используемого по умолчанию. Должно соответствовать ключу в секции `sources`.

### data.cache_dir

**Тип**: `string`  
**Значение по умолчанию**: `'data/cache'`  
**Описание**: Директория для кэширования загруженных данных. Использование кэша позволяет избежать повторной загрузки одних и тех же данных.

### data.sources

**Тип**: `object`  
**Описание**: Словарь с настройками для различных источников данных.

#### data.sources.{source_name}.type

**Тип**: `string`  
**Описание**: Тип источника данных. Поддерживаемые значения:
- `'reanalysis'`: Данные реанализа (например, ERA5)
- `'satellite'`: Спутниковые данные (заглушка для будущих реализаций)
- `'station'`: Наземные наблюдения (заглушка для будущих реализаций)

#### data.sources.{source_name}.dataset_type

**Тип**: `string`  
**Описание**: Тип набора данных ERA5. Поддерживаемые значения:
- `'pressure_levels'`: Данные на уровнях давления
- `'surface'`: Поверхностные данные
- `'land'`: Наземные данные ERA5-Land
- `'monthly'`: Месячные средние значения

#### data.sources.{source_name}.variables

**Тип**: `array of strings`  
**Описание**: Список переменных для загрузки. Доступные переменные зависят от типа набора данных.

Для поверхностных данных:
- `'msl'`: Давление на уровне моря
- `'sp'`: Давление на поверхности
- `'tp'`: Общее количество осадков
- `'2t'`: Температура на высоте 2 м
- `'10u'`: U-компонента ветра на высоте 10 м
- `'10v'`: V-компонента ветра на высоте 10 м
- `'skt'`: Температура поверхности
- `'tcc'`: Общая облачность
- `'blh'`: Высота пограничного слоя

Для данных на уровнях давления:
- `'z'`: Геопотенциал
- `'u'`: U-компонента ветра
- `'v'`: V-компонента ветра
- `'t'`: Температура
- `'q'`: Удельная влажность
- `'vo'`: Относительная завихренность
- `'d'`: Дивергенция
- `'r'`: Относительная влажность

#### data.sources.{source_name}.levels

**Тип**: `array of integers`  
**Описание**: Список уровней давления в гПа (только для данных на уровнях давления).

#### data.sources.{source_name}.grid_resolution

**Тип**: `float`  
**Описание**: Разрешение сетки в градусах. Значение 0.25 соответствует разрешению ~28 км.

## Параметры обнаружения

### detection.min_latitude

**Тип**: `float`  
**Значение по умолчанию**: `65.0`  
**Описание**: Минимальная широта для анализа арктических циклонов (градусы северной широты).

### detection.criteria

**Тип**: `object`  
**Описание**: Настройки критериев обнаружения циклонов.

#### detection.criteria.pressure_minimum

Критерий обнаружения на основе минимумов давления.

##### detection.criteria.pressure_minimum.enabled

**Тип**: `boolean`  
**Описание**: Включить/выключить критерий.

##### detection.criteria.pressure_minimum.gradient_threshold

**Тип**: `float`  
**Описание**: Минимальный градиент давления (гПа/100км).

#### detection.criteria.vorticity

Критерий обнаружения на основе завихренности.

##### detection.criteria.vorticity.enabled

**Тип**: `boolean`  
**Описание**: Включить/выключить критерий.

##### detection.criteria.vorticity.threshold

**Тип**: `float`  
**Описание**: Минимальная завихренность (1/с).

##### detection.criteria.vorticity.level

**Тип**: `integer`  
**Описание**: Уровень давления для анализа завихренности (гПа).

#### detection.criteria.closed_contour

Критерий обнаружения на основе замкнутых изобар.

##### detection.criteria.closed_contour.enabled

**Тип**: `boolean`  
**Описание**: Включить/выключить критерий.

##### detection.criteria.closed_contour.contour_interval

**Тип**: `float`  
**Описание**: Интервал изобар (гПа).

#### detection.criteria.wind_threshold

Критерий обнаружения на основе порога скорости ветра.

##### detection.criteria.wind_threshold.enabled

**Тип**: `boolean`  
**Описание**: Включить/выключить критерий.

##### detection.criteria.wind_threshold.min_speed

**Тип**: `float`  
**Описание**: Минимальная скорость ветра (м/с).

#### detection.criteria.pressure_laplacian

Критерий обнаружения на основе лапласиана давления.

##### detection.criteria.pressure_laplacian.enabled

**Тип**: `boolean`  
**Описание**: Включить/выключить критерий.

##### detection.criteria.pressure_laplacian.laplacian_threshold

**Тип**: `float`  
**Описание**: Порог для лапласиана давления (Па/км²).

##### detection.criteria.pressure_laplacian.smooth_sigma

**Тип**: `float`  
**Описание**: Параметр сглаживания поля давления.

##### detection.criteria.pressure_laplacian.window_size

**Тип**: `integer`  
**Описание**: Размер окна для поиска локальных экстремумов.

## Параметры трекинга

### tracking.max_distance

**Тип**: `float`  
**Описание**: Максимальное расстояние между точками циклона на последовательных временных шагах (км).

### tracking.max_time_gap

**Тип**: `float`  
**Описание**: Максимальный временной разрыв между наблюдениями циклона (часы).

### tracking.max_pressure_change

**Тип**: `float`  
**Описание**: Максимальное изменение давления между наблюдениями (гПа).

### tracking.min_track_duration

**Тип**: `float`  
**Описание**: Минимальная продолжительность трека циклона (часы).

### tracking.min_track_points

**Тип**: `integer`  
**Описание**: Минимальное количество точек в треке.

### tracking.debug_save_csv

**Тип**: `boolean`  
**Описание**: Сохранять промежуточные результаты трекинга в CSV файлы для отладки.

### tracking.debug_dir

**Тип**: `string`  
**Описание**: Директория для сохранения отладочных файлов.

### tracking.algorithm

**Тип**: `string`  
**Описание**: Алгоритм трекинга. Поддерживаемые значения:
- `'nearest_neighbor'`: Алгоритм ближайших соседей

### tracking.cost_weights

**Тип**: `object`  
**Описание**: Веса для расчета стоимости связывания точек в трекинге.

#### tracking.cost_weights.distance

**Тип**: `float`  
**Описание**: Вес расстояния в функции стоимости.

#### tracking.cost_weights.time

**Тип**: `float`  
**Описание**: Вес временного интервала в функции стоимости.

#### tracking.cost_weights.pressure

**Тип**: `float`  
**Описание**: Вес изменения давления в функции стоимости.

## Параметры визуализации

### visualization.default_projection

**Тип**: `string`  
**Описание**: Проекция карты по умолчанию. Поддерживаемые значения:
- `'NorthPolarStereo'`: Стереографическая проекция Северного полюса

### visualization.map_resolution

**Тип**: `string`  
**Описание**: Разрешение картографических элементов. Поддерживаемые значения:
- `'low'`: Низкое разрешение
- `'intermediate'`: Среднее разрешение
- `'high'`: Высокое разрешение

### visualization.output_dir

**Тип**: `string`  
**Описание**: Директория для сохранения изображений.

### visualization.default_dpi

**Тип**: `integer`  
**Описание**: Разрешение изображений по умолчанию (точек на дюйм).

### visualization.cyclone_marker_size

**Тип**: `integer`  
**Описание**: Размер маркера циклона на карте.

### visualization.track_line_width

**Тип**: `float`  
**Описание**: Толщина линии трека циклона.

### visualization.color_map

**Тип**: `string`  
**Описание**: Цветовая схема по умолчанию для тепловых карт.

### visualization.map_features

**Тип**: `object`  
**Описание**: Настройки отображения картографических элементов.

#### visualization.map_features.coastlines

**Тип**: `boolean`  
**Описание**: Отображать береговые линии.

#### visualization.map_features.countries

**Тип**: `boolean`  
**Описание**: Отображать границы стран.

#### visualization.map_features.grid_lines

**Тип**: `boolean`  
**Описание**: Отображать линии координатной сетки.

#### visualization.map_features.grid_labels

**Тип**: `boolean`  
**Описание**: Отображать подписи координатной сетки.

## Параметры экспорта

### export.output_dir

**Тип**: `string`  
**Описание**: Директория для сохранения экспортируемых данных.

### export.formats

**Тип**: `array of strings`  
**Описание**: Список форматов для экспорта. Поддерживаемые значения:
- `'csv'`: Табличный формат
- `'netcdf'`: Научный формат NetCDF

### export.default_format

**Тип**: `string`  
**Описание**: Формат экспорта по умолчанию.

### export.csv

**Тип**: `object`  
**Описание**: Настройки экспорта в формат CSV.

#### export.csv.delimiter

**Тип**: `string`  
**Описание**: Разделитель полей в CSV файле.

#### export.csv.include_header

**Тип**: `boolean`  
**Описание**: Включать заголовок с именами полей.

### export.netcdf

**Тип**: `object`  
**Описание**: Настройки экспорта в формат NetCDF.

#### export.netcdf.compression_level

**Тип**: `integer`  
**Описание**: Уровень сжатия NetCDF файлов (0-9).

#### export.netcdf.include_metadata

**Тип**: `boolean`  
**Описание**: Включать метаданные в NetCDF файлы.