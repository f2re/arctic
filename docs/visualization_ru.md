# Визуализация в ArcticCyclone

## Содержание

1. [Обзор системы визуализации](#обзор-системы-визуализации)
2. [Архитектура визуализации](#архитектура-визуализации)
3. [Типы визуализаций](#типы-визуализаций)
   - [Треки циклонов](#треки-циклонов)
   - [Тепловые карты](#тепловые-карты)
   - [Параметры циклонов](#параметры-циклонов)
   - [Поля критериев](#поля-критериев)
4. [Настройка визуализации](#настройка-визуализации)
5. [Пользовательские визуализации](#пользовательские-визуализации)
6. [Сохранение и экспорт графиков](#сохранение-и-экспорт-графиков)

## Обзор системы визуализации

Система визуализации ArcticCyclone предоставляет широкие возможности для отображения результатов анализа циклонов. Она использует библиотеки matplotlib и cartopy для создания высококачественных научных графиков и карт.

Основные возможности системы визуализации:

- Отображение треков циклонов на географических картах
- Создание тепловых карт плотности циклонов
- Построение графиков изменения параметров циклонов во времени
- Визуализация полей метеорологических параметров
- Настройка внешнего вида графиков и карт
- Экспорт результатов в различные форматы

## Архитектура визуализации

Система визуализации организована в модульную структуру:

### Основные модули

1. **tracks.py**: Визуализация треков циклонов
2. **heatmaps.py**: Создание тепловых карт
3. **parameters.py**: Графики параметров циклонов
4. **criteria.py**: Визуализация полей критериев
5. **mappers.py**: Базовые функции картографии

### Основные классы и функции

- `plot_cyclone_tracks()`: Основная функция для отображения треков
- `create_cyclone_frequency_map()`: Создание тепловых карт
- `plot_cyclone_parameters()`: Графики параметров циклонов
- `plot_pressure_field()`: Визуализация поля давления
- `plot_vorticity_field()`: Визуализация поля завихренности

## Типы визуализаций

### Треки циклонов

Отображение путей циклонов на карте Арктики.

#### Основные функции

```python
from visualization.tracks import plot_cyclone_tracks

# Базовое отображение треков
fig = plot_cyclone_tracks(
    tracks=cyclone_tracks,
    region={'north': 90.0, 'south': 60.0, 'east': 180.0, 'west': -180.0},
    output_file='cyclone_tracks.png'
)
```

#### Параметры настройки

- `tracks`: Список треков циклонов
- `region`: Географический регион для отображения
- `output_file`: Путь для сохранения изображения
- `figsize`: Размер фигуры (ширина, высота) в дюймах
- `show_intensity`: Цветовая кодировка по интенсивности циклонов
- `title`: Заголовок графика

#### Варианты отображения

1. **Цветовая кодировка по давлению**: Циклоны с более низким давлением отображаются более насыщенными цветами
2. **Цветовая кодировка по времени**: Разные цвета для разных этапов жизненного цикла
3. **Цветовая кодировка по интенсивности**: Различные категории интенсивности
4. **Цветовая кодировка по типу**: Различные типы циклонов (холодные, теплые, гибридные)

#### Примеры использования

```python
# Отображение треков с цветовой кодировкой по давлению
fig1 = plot_cyclone_tracks(
    tracks=cyclone_tracks,
    show_intensity=True,
    title="Треки арктических циклонов (цвет по давлению)"
)

# Отображение треков с уникальными цветами
fig2 = plot_cyclone_tracks(
    tracks=cyclone_tracks,
    show_intensity=False,
    title="Треки арктических циклонов (уникальные цвета)"
)
```

### Тепловые карты

Создание карт плотности циклонов в различных регионах Арктики.

#### Основные функции

```python
from visualization.heatmaps import create_cyclone_frequency_map

# Создание тепловой карты
fig, ax = create_cyclone_frequency_map(
    cyclones=all_cyclones,
    min_latitude=65.0,
    grid_resolution=1.0,
    smoothing_sigma=1.5
)
```

#### Параметры настройки

- `cyclones`: Список всех циклонов
- `min_latitude`: Минимальная широта для анализа
- `grid_resolution`: Разрешение сетки (градусы)
- `smoothing_sigma`: Параметр сглаживания
- `output_file`: Путь для сохранения изображения

#### Методы создания тепловых карт

1. **Простое суммирование**: Подсчет количества циклонов в каждой ячейке сетки
2. **Взвешенное суммирование**: Учет продолжительности пребывания циклона в ячейке
3. **Гауссово сглаживание**: Создание гладких распределений плотности

#### Примеры использования

```python
# Создание тепловой карты с высоким разрешением
fig1, ax1 = create_cyclone_frequency_map(
    cyclones=all_cyclones,
    min_latitude=70.0,
    grid_resolution=0.5,  # Высокое разрешение
    smoothing_sigma=1.0,
    title="Плотность циклонов (высокое разрешение)"
)

# Создание тепловой карты с сильным сглаживанием
fig2, ax2 = create_cyclone_frequency_map(
    cyclones=all_cyclones,
    min_latitude=65.0,
    grid_resolution=2.0,  # Низкое разрешение
    smoothing_sigma=3.0,  # Сильное сглаживание
    title="Плотность циклонов (сильное сглаживание)"
)
```

### Параметры циклонов

Графики изменения метеорологических параметров циклонов во времени.

#### Основные функции

```python
from visualization.parameters import plot_cyclone_parameters

# График параметров для одного трека
fig = plot_cyclone_parameters(
    cyclone_track=track,
    variables=['central_pressure', 'vorticity_850hPa', 'max_wind_speed']
)
```

#### Параметры настройки

- `cyclone_track`: Трек циклона для анализа
- `variables`: Список переменных для отображения
- `map_view`: Включать ли карту с треком
- `min_latitude`: Минимальная широта для карты
- `figsize`: Размер фигуры

#### Поддерживаемые параметры

1. **Центральное давление**: `central_pressure`
2. **Завихренность на 850 гПа**: `vorticity_850hPa`
3. **Максимальная скорость ветра**: `max_wind_speed`
4. **Радиус циклона**: `radius`
5. **Градиент давления**: `pressure_gradient`
6. **Температурная аномалия**: `temperature_anomaly`

#### Примеры использования

```python
# Комплексный график параметров
fig1 = plot_cyclone_parameters(
    cyclone_track=track,
    variables=['central_pressure', 'vorticity_850hPa', 'max_wind_speed'],
    map_view=True,
    figsize=(15, 10)
)

# Только графики параметров без карты
fig2 = plot_cyclone_parameters(
    cyclone_track=track,
    variables=['central_pressure', 'radius', 'pressure_gradient'],
    map_view=False,
    figsize=(12, 8)
)
```

### Поля критериев

Визуализация метеорологических полей, используемых для обнаружения циклонов.

#### Основные функции

```python
from visualization.criteria import plot_pressure_field, plot_vorticity_field

# Визуализация поля давления
plot_pressure_field(
    pressure=pressure_data,
    lats=latitude_data,
    lons=longitude_data,
    time_step=time_step,
    output_dir='output/plots'
)

# Визуализация поля завихренности
plot_vorticity_field(
    vorticity=vorticity_data,
    lats=latitude_data,
    lons=longitude_data,
    threshold=vorticity_threshold,
    time_step=time_step,
    output_dir='output/plots'
)
```

#### Поддерживаемые поля

1. **Поле давления**: `plot_pressure_field()`
2. **Поле завихренности**: `plot_vorticity_field()`
3. **Поле ветра**: `plot_wind_field()`
4. **Поле лапласиана давления**: `plot_laplacian_field()`
5. **Поле замкнутых контуров**: `plot_closed_contour_field()`

#### Параметры настройки

- `data`: Данные для визуализации
- `lats`: Массив широт
- `lons`: Массив долгот
- `threshold`: Пороговое значение (если применимо)
- `time_step`: Временной шаг
- `output_dir`: Директория для сохранения изображений

## Настройка визуализации

Визуализация настраивается через конфигурационный файл `config.yaml` и программно.

### Через конфигурационный файл

```yaml
visualization:
  default_projection: 'NorthPolarStereo'  # Проекция карты
  map_resolution: 'intermediate'          # Разрешение картографических элементов
  output_dir: 'output/figures'            # Директория для сохранения изображений
  default_dpi: 300                        # Разрешение изображений
  cyclone_marker_size: 30                 # Размер маркера циклона
  track_line_width: 1.5                   # Толщина линии трека
  color_map: 'viridis'                    # Цветовая схема по умолчанию
  map_features:
    coastlines: true                      # Отображать береговые линии
    countries: true                       # Отображать границы стран
    grid_lines: true                      # Отображать линии сетки
    grid_labels: true                     # Отображать подписи сетки
```

### Программная настройка

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Настройка параметров matplotlib
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

# Создание карты с пользовательскими параметрами
fig = plt.figure(figsize=(15, 12))
ax = plt.axes(projection=ccrs.NorthPolarStereo())

# Настройка проекции
ax.set_extent([-180, 180, 65, 90], ccrs.PlateCarree())

# Добавление картографических элементов
ax.coastlines(resolution='50m', linewidth=0.8)
ax.gridlines(draw_labels=True, linewidth=0.5)

# Настройка цветовой схемы
import matplotlib.colors as mcolors
cmap = plt.get_cmap('coolwarm_r')
norm = mcolors.Normalize(vmin=950, vmax=1020)
```

### Пользовательские стили

```python
# Создание пользовательского стиля
custom_style = {
    'figure.figsize': (14, 10),
    'figure.dpi': 300,
    'font.size': 14,
    'axes.labelsize': 12,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 8
}

# Применение стиля
plt.rcParams.update(custom_style)
```

## Пользовательские визуализации

Система поддерживает создание пользовательских визуализаций.

### Создание пользовательской функции визуализации

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from typing import List, Dict
from pathlib import Path

def plot_cyclone_intensity_map(cyclones: List, 
                              region: Dict[str, float],
                              output_file: str = None,
                              grid_size: float = 1.0) -> plt.Figure:
    """
    Создает карту интенсивности циклонов для арктического региона.
    
    Аргументы:
        cyclones: Список объектов циклонов для визуализации
        region: Географический регион (север, юг, запад, восток)
        output_file: Путь для сохранения изображения
        grid_size: Размер ячейки сетки в градусах
        
    Возвращает:
        Matplotlib Figure объект
    """
    # Создаем фигуру с проекцией
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    
    # Устанавливаем границы карты
    ax.set_extent([region['west'], region['east'], 
                  region['south'], region['north']], 
                 ccrs.PlateCarree())
    
    # Добавляем фоновые элементы
    ax.coastlines(resolution='50m', linewidth=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    
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
            
            # Добавляем интенсивность (инвертируем давление)
            intensity_value = 1020 - cyclone.central_pressure
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
    cmap = plt.get_cmap('hot_r')
    mesh = ax.pcolormesh(lon_mesh, lat_mesh, avg_intensity, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, alpha=0.7)
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(mesh, ax=ax, pad=0.1)
    cbar.set_label('Интенсивность циклонов (инвертированное давление)')
    
    # Добавляем заголовок
    plt.title('Карта интенсивности арктических циклонов', fontsize=16)
    
    # Сохраняем изображение
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig

# Использование пользовательской визуализации
region = {
    'north': 90.0,
    'south': 70.0,
    'east': 180.0,
    'west': -180.0
}

fig = plot_cyclone_intensity_map(
    cyclones=all_cyclones,
    region=region,
    output_file='cyclone_intensity_map.png',
    grid_size=2.0
)
```

### Расширение существующих функций

```python
from visualization.tracks import plot_cyclone_tracks

def plot_cyclone_tracks_with_seasons(tracks: List, 
                                   output_file: str = None) -> plt.Figure:
    """
    Отображает треки циклонов с цветовой кодировкой по сезонам.
    """
    # Определяем сезон для каждого трека
    seasons = []
    for track in tracks:
        if track:
            # Берем время первой точки трека
            month = sorted(track, key=lambda c: c.time)[0].time.month
            if 3 <= month <= 5:
                seasons.append('spring')
            elif 6 <= month <= 8:
                seasons.append('summer')
            elif 9 <= month <= 11:
                seasons.append('autumn')
            else:
                seasons.append('winter')
    
    # Цвета для сезонов
    season_colors = {
        'winter': 'blue',
        'spring': 'green',
        'summer': 'red',
        'autumn': 'orange'
    }
    
    # Создаем базовую карту
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.gridlines(draw_labels=True)
    
    # Отображаем треки с цветовой кодировкой по сезонам
    for i, track in enumerate(tracks):
        if track:
            track_sorted = sorted(track, key=lambda c: c.time)
            lats = [c.latitude for c in track_sorted]
            lons = [c.longitude for c in track_sorted]
            
            season = seasons[i]
            color = season_colors[season]
            
            ax.plot(lons, lats, color=color, transform=ccrs.PlateCarree(),
                   linewidth=1.5, alpha=0.7)
    
    # Добавляем легенду
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color=color, label=season)
        for season, color in season_colors.items()
    ]
    ax.legend(handles=legend_elements, loc='lower right', title='Сезон')
    
    # Устанавливаем заголовок
    ax.set_title('Треки арктических циклонов по сезонам', fontsize=14)
    
    # Сохраняем изображение
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig
```

## Сохранение и экспорт графиков

Система поддерживает экспорт графиков в различные форматы.

### Поддерживаемые форматы

1. **PNG**: Основной формат для веб-публикаций
2. **PDF**: Векторный формат для научных публикаций
3. **SVG**: Векторный формат для веб-публикаций
4. **EPS**: Векторный формат для LaTeX

### Настройка экспорта

```python
# Сохранение в различных форматах
fig = plot_cyclone_tracks(cyclone_tracks)

# PNG
fig.savefig('tracks.png', dpi=300, bbox_inches='tight')

# PDF
fig.savefig('tracks.pdf', bbox_inches='tight')

# SVG
fig.savefig('tracks.svg', bbox_inches='tight')

# EPS
fig.savefig('tracks.eps', bbox_inches='tight')
```

### Оптимизация для публикаций

```python
# Настройка для научных публикаций
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Высокое разрешение
fig = plot_cyclone_tracks(cyclone_tracks)
fig.savefig('publication_tracks.png', dpi=600, bbox_inches='tight')
```

### Создание анимаций

```python
from visualization.tracks import animate_cyclone_track

# Создание анимации трека циклона
ani = animate_cyclone_track(
    cyclone_track=track,
    min_latitude=65.0,
    figsize=(10, 8),
    fps=5,
    dpi=150,
    output_file='cyclone_track_animation.gif'
)
```

### Интерактивные визуализации

```python
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def create_interactive_track_plot(tracks: List):
    """
    Создает интерактивный график треков с возможностью фильтрации по времени.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)
    
    # Создаем слайдер для выбора времени
    ax_time = plt.axes([0.2, 0.1, 0.5, 0.03])
    time_slider = Slider(ax_time, 'Time', 0, len(tracks)-1, valinit=0, valfmt='%d')
    
    def update(val):
        ax.clear()
        track_idx = int(time_slider.val)
        if track_idx < len(tracks):
            track = tracks[track_idx]
            lats = [c.latitude for c in track]
            lons = [c.longitude for c in track]
            ax.plot(lons, lats, 'b-', marker='o')
        plt.draw()
    
    time_slider.on_changed(update)
    plt.show()
```