# Экспорт данных в ArcticCyclone

## Содержание

1. [Обзор системы экспорта](#обзор-системы-экспорта)
2. [Архитектура экспорта](#архитектура-экспорта)
3. [Поддерживаемые форматы](#поддерживаемые-форматы)
   - [CSV](#csv)
   - [NetCDF](#netcdf)
   - [GeoJSON](#geojson)
4. [Настройка экспорта](#настройка-экспорта)
5. [Пользовательские форматы экспорта](#пользовательские-форматы-экспорта)
6. [Импорт данных](#импорт-данных)

## Обзор системы экспорта

Система экспорта ArcticCyclone предоставляет возможность сохранять результаты анализа циклонов в различных форматах для дальнейшего использования, анализа и обмена данными. Экспорт поддерживает как табличные, так и научные форматы данных.

Основные возможности системы экспорта:

- Экспорт треков циклонов в различные форматы
- Экспорт статистики циклонов
- Экспорт параметров циклонов
- Настройка форматов экспорта через конфигурацию
- Поддержка стандартных научных форматов (NetCDF, GeoJSON)
- Возможность импорта ранее экспортированных данных

## Архитектура экспорта

Система экспорта организована в модульную структуру:

### Основные модули

1. **formats/**: Директория с форматами экспорта
   - `csv_exporter.py`: Экспорт в CSV
   - `netcdf_exporter.py`: Экспорт в NetCDF
   - `geojson_exporter.py`: Экспорт в GeoJSON
2. **publishers.py**: Публикация данных в различные системы

### Основные классы

- `CycloneCSVExporter`: Экспорт в формат CSV
- `CycloneNetCDFExporter`: Экспорт в формат NetCDF
- `CycloneGeoJSONExporter`: Экспорт в формат GeoJSON

## Поддерживаемые форматы

### CSV

Формат CSV (Comma-Separated Values) представляет собой текстовый формат для хранения табличных данных. Это самый простой и универсальный формат для обмена данными.

#### Структура данных

CSV файл содержит следующие поля для каждого циклона:

- `track_id`: Идентификатор трека
- `time`: Время наблюдения
- `latitude`: Широта центра циклона
- `longitude`: Долгота центра циклона
- `central_pressure`: Центральное давление (гПа)
- `age_hours`: Возраст циклона в часах
- `vorticity_850hPa`: Завихренность на уровне 850 гПа
- `max_wind_speed`: Максимальная скорость ветра (м/с)
- `radius_km`: Радиус циклона (км)
- `thermal_type`: Термический тип циклона
- `temperature_anomaly`: Температурная аномалия

#### Пример использования

```python
from export.formats.csv_exporter import CycloneCSVExporter

# Создание экспортера
exporter = CycloneCSVExporter(
    delimiter=',',
    encoding='utf-8',
    include_header=True
)

# Экспорт треков циклонов
exporter.export_cyclone_tracks(
    cyclones=cyclone_tracks,
    filename='cyclone_tracks.csv'
)

# Экспорт статистики
exporter.export_cyclone_statistics(
    cyclones=all_cyclones,
    filename='cyclone_statistics.csv'
)
```

#### Преимущества

- Простота использования и понимания
- Совместимость с большинством программ анализа данных
- Человекочитаемый формат
- Легко редактируется в текстовых редакторах

#### Ограничения

- Не содержит метаданных
- Не поддерживает сложные структуры данных
- Большие файлы могут быть неэффективны для обработки

### NetCDF

Формат NetCDF (Network Common Data Form) - это самоописательный, платформонезависимый формат для хранения научных данных. Широко используется в метеорологии и океанографии.

#### Структура данных

NetCDF файл содержит:

- **Измерения**:
  - `time`: Временные шаги
  - `track`: Идентифаторы треков
  - `parameter`: Параметры циклонов

- **Переменные**:
  - `latitude`: Широта циклонов
  - `longitude`: Долгота циклонов
  - `central_pressure`: Центральное давление
  - `vorticity`: Завихренность
  - `wind_speed`: Скорость ветра
  - `time`: Временные метки
  - `track_id`: Идентификаторы треков

- **Атрибуты**:
  - Метаданные о данных
  - Информация о единицах измерения
  - Источник данных
  - Дата создания

#### Пример использования

```python
from export.formats.netcdf_exporter import CycloneNetCDFExporter

# Создание экспортера
exporter = CycloneNetCDFExporter(
    compression_level=4,
    include_metadata=True
)

# Экспорт данных
exporter.export_cyclone_data(
    cyclones=cyclone_tracks,
    filename='cyclone_data.nc'
)
```

#### Преимущества

- Эффективное хранение больших объемов данных
- Поддержка многомерных массивов
- Богатые метаданные
- Стандарт в научном сообществе
- Поддержка сжатия данных

#### Ограничения

- Требует специализированных инструментов для просмотра
- Более сложная структура по сравнению с CSV
- Не так легко редактируется вручную

### GeoJSON

Формат GeoJSON - это формат на основе JSON для кодирования различных географических структур данных. Подходит для использования в веб-приложениях и GIS-системах.

#### Структура данных

GeoJSON файл содержит:

- **FeatureCollection**: Коллекция географических объектов
- **Feature**: Отдельный географический объект (циклон)
  - `geometry`: Геометрия (точка, линия)
  - `properties`: Свойства объекта (параметры циклона)
  - `type`: Тип объекта

#### Типы объектов

1. **Point**: Точки для отдельных наблюдений циклонов
2. **LineString**: Линии для представления треков циклонов
3. **Polygon**: Полигоны для представления областей циклонов

#### Пример использования

```python
from export.formats.geojson_exporter import CycloneGeoJSONExporter

# Создание экспортера
exporter = CycloneGeoJSONExporter(
    include_properties=True,
    geometry_type='LineString'
)

# Экспорт треков
exporter.export_cyclone_tracks(
    tracks=cyclone_tracks,
    filename='cyclone_tracks.geojson'
)
```

#### Преимущества

- Совместимость с веб-картографическими библиотеками
- Человекочитаемый формат (JSON)
- Поддержка сложных географических структур
- Легко интегрируется с веб-приложениями

#### Ограничения

- Более объемный по сравнению с бинарными форматами
- Ограниченная поддержка в традиционных научных приложениях
- Может быть неэффективен для очень больших наборов данных

## Настройка экспорта

Экспорт настраивается через конфигурационный файл `config.yaml` и программно.

### Через конфигурационный файл

```yaml
export:
  output_dir: 'output/data'           # Директория для экспорта
  formats: ['csv', 'netcdf']          # Список форматов для экспорта
  default_format: 'csv'               # Формат по умолчанию
  csv:
    delimiter: ','                    # Разделитель полей
    include_header: true              # Включать заголовок
  netcdf:
    compression_level: 4              # Уровень сжатия (0-9)
    include_metadata: true            # Включать метаданные
  geojson:
    include_properties: true          # Включать свойства
    geometry_type: 'LineString'       # Тип геометрии
```

### Программная настройка

```python
# Настройка экспорта в основной программе
from export.formats.csv_exporter import CycloneCSVExporter
from export.formats.netcdf_exporter import CycloneNetCDFExporter

# Создание экспортеров с пользовательскими настройками
csv_exporter = CycloneCSVExporter(
    delimiter=';',
    encoding='utf-8',
    include_header=True
)

netcdf_exporter = CycloneNetCDFExporter(
    compression_level=6,
    include_metadata=True
)

# Экспорт данных в нескольких форматах
output_dir = 'results'

# Экспорт в CSV
csv_exporter.export_cyclone_tracks(
    cyclones=filtered_tracks,
    filename=f'{output_dir}/cyclone_tracks.csv'
)

# Экспорт в NetCDF
netcdf_exporter.export_cyclone_data(
    cyclones=filtered_tracks,
    filename=f'{output_dir}/cyclone_data.nc'
)
```

### Автоматический экспорт

```python
def export_results(tracks, output_dir, formats=['csv', 'netcdf']):
    """
    Автоматический экспорт результатов в указанные форматы.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = []
    
    if 'csv' in formats:
        from export.formats.csv_exporter import CycloneCSVExporter
        exporter = CycloneCSVExporter()
        file_path = exporter.export_cyclone_tracks(tracks, f'{output_dir}/tracks.csv')
        exported_files.append(file_path)
        print(f"Экспортировано в CSV: {file_path}")
    
    if 'netcdf' in formats:
        from export.formats.netcdf_exporter import CycloneNetCDFExporter
        exporter = CycloneNetCDFExporter()
        file_path = exporter.export_cyclone_data(tracks, f'{output_dir}/tracks.nc')
        exported_files.append(file_path)
        print(f"Экспортировано в NetCDF: {file_path}")
    
    if 'geojson' in formats:
        from export.formats.geojson_exporter import CycloneGeoJSONExporter
        exporter = CycloneGeoJSONExporter()
        file_path = exporter.export_cyclone_tracks(tracks, f'{output_dir}/tracks.geojson')
        exported_files.append(file_path)
        print(f"Экспортировано в GeoJSON: {file_path}")
    
    return exported_files
```

## Пользовательские форматы экспорта

Система поддерживает создание пользовательских форматов экспорта.

### Создание пользовательского экспортера

```python
from core.exceptions import ExportError
import json
from typing import List, Dict, Any

class CustomJSONExporter:
    """
    Пользовательский экспортер в формат JSON.
    """
    
    def __init__(self, indent: int = 2, ensure_ascii: bool = False):
        """
        Инициализация экспортера.
        
        Аргументы:
            indent: Отступ для форматирования JSON
            ensure_ascii: Ограничивать символы только ASCII
        """
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    def export_cyclone_summary(self, tracks: List[List], 
                              filename: str) -> str:
        """
        Экспорт сводной информации о циклонах в JSON.
        
        Аргументы:
            tracks: Список треков циклонов
            filename: Путь к выходному файлу
            
        Возвращает:
            Путь к созданному файлу
        """
        try:
            # Подготовка данных для экспорта
            summary_data = {
                'total_tracks': len(tracks),
                'total_cyclones': sum(len(track) for track in tracks),
                'tracks': []
            }
            
            # Обработка каждого трека
            for i, track in enumerate(tracks):
                if not track:
                    continue
                    
                # Сортировка трека по времени
                track_sorted = sorted(track, key=lambda c: c.time)
                
                # Расчет метрик трека
                metrics = track_sorted[0].calculate_lifecycle_metrics()
                
                # Создание записи трека
                track_data = {
                    'track_id': track_sorted[0].track_id,
                    'start_time': track_sorted[0].time.isoformat(),
                    'end_time': track_sorted[-1].time.isoformat(),
                    'duration_hours': metrics.get('lifespan_hours', 0),
                    'min_pressure': min(c.central_pressure for c in track_sorted),
                    'cyclones_count': len(track_sorted),
                    'coordinates': [
                        {
                            'time': c.time.isoformat(),
                            'latitude': c.latitude,
                            'longitude': c.longitude,
                            'pressure': c.central_pressure
                        }
                        for c in track_sorted
                    ]
                }
                
                summary_data['tracks'].append(track_data)
            
            # Сохранение в файл
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=self.indent, 
                         ensure_ascii=self.ensure_ascii, default=str)
            
            print(f"Сводные данные экспортированы в JSON: {filename}")
            return filename
            
        except Exception as e:
            raise ExportError(f"Ошибка при экспорте в JSON: {str(e)}")

# Использование пользовательского экспортера
custom_exporter = CustomJSONExporter(indent=4)
custom_exporter.export_cyclone_summary(
    tracks=cyclone_tracks,
    filename='cyclone_summary.json'
)
```

### Расширение существующих экспортеров

```python
from export.formats.csv_exporter import CycloneCSVExporter

class ExtendedCSVExporter(CycloneCSVExporter):
    """
    Расширенный экспортер CSV с дополнительными возможностями.
    """
    
    def export_with_custom_fields(self, tracks: List[List], 
                                 filename: str,
                                 additional_fields: List[str] = None) -> str:
        """
        Экспорт с пользовательскими полями.
        """
        if additional_fields is None:
            additional_fields = ['lifespan_hours', 'deepening_rate', 'displacement_km']
        
        # Подготовка расширенных данных
        extended_data = []
        
        for track in tracks:
            for cyclone in track:
                # Получаем базовые данные
                base_data = self._cyclone_to_dict(cyclone)
                
                # Добавляем пользовательские поля
                metrics = cyclone.calculate_lifecycle_metrics()
                for field in additional_fields:
                    if field in metrics:
                        base_data[field] = metrics[field]
                
                extended_data.append(base_data)
        
        # Создание и сохранение DataFrame
        import pandas as pd
        df = pd.DataFrame(extended_data)
        df.to_csv(filename, sep=self.delimiter, encoding=self.encoding, 
                 index=False, header=self.include_header)
        
        return filename

# Использование расширенного экспортера
extended_exporter = ExtendedCSVExporter()
extended_exporter.export_with_custom_fields(
    tracks=cyclone_tracks,
    filename='extended_cyclone_data.csv',
    additional_fields=['lifespan_hours', 'deepening_rate', 'mean_speed']
)
```

## Импорт данных

Система поддерживает импорт ранее экспортированных данных.

### Импорт из CSV

```python
from export.formats.csv_exporter import CycloneCSVExporter

# Создание экспортера
exporter = CycloneCSVExporter()

# Импорт треков из CSV
imported_tracks = exporter.import_from_csv('cyclone_tracks.csv')

print(f"Импортировано {len(imported_tracks)} треков")
```

### Импорт из NetCDF

```python
from export.formats.netcdf_exporter import CycloneNetCDFExporter

# Создание экспортера
exporter = CycloneNetCDFExporter()

# Импорт данных из NetCDF
imported_data = exporter.import_from_netcdf('cyclone_data.nc')

print(f"Импортированы данные: {type(imported_data)}")
```

### Импорт из GeoJSON

```python
from export.formats.geojson_exporter import CycloneGeoJSONExporter

# Создание экспортера
exporter = CycloneGeoJSONExporter()

# Импорт данных из GeoJSON
imported_features = exporter.import_from_geojson('cyclone_tracks.geojson')

print(f"Импортировано {len(imported_features)} географических объектов")
```

### Преимущества импорта/экспорта

1. **Воспроизводимость**: Возможность восстановить результаты анализа
2. **Обмен данными**: Легкий обмен результатами с другими исследователями
3. **Интеграция**: Совместимость с другими инструментами анализа
4. **Архивирование**: Долгосрочное хранение результатов
5. **Валидация**: Проверка корректности результатов через повторный импорт