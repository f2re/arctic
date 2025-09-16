# API документация ArcticCyclone

## Содержание

1. [Основные модули](#основные-модули)
2. [Модуль core](#модуль-core)
   - [ConfigManager](#configmanager)
   - [ exceptions](#исключения)
3. [Модуль data](#модуль-data)
   - [DataSourceManager](#datasourcemanager)
   - [CredentialManager](#credentialmanager)
   - [ERA5Adapter](#era5adapter)
4. [Модуль detection](#модуль-detection)
   - [CycloneDetector](#cyclonedetector)
   - [CycloneTracker](#cyclonetracker)
   - [CriteriaManager](#criteriamanager)
   - [BaseCriterion](#basecriterion)
5. [Модуль models](#модуль-models)
   - [Cyclone](#cyclone)
   - [CycloneParameters](#cycloneparameters)
6. [Модуль visualization](#модуль-visualization)
   - [Основные функции визуализации](#основные-функции-визуализации)
7. [Модуль export](#модуль-export)
   - [CycloneCSVExporter](#cyclonecsvexporter)
   - [CycloneNetCDFExporter](#cyclonenetcdfexporter)

## Основные модули

ArcticCyclone организован в модульную структуру с четким разделением ответственности:

- **core**: Базовые компоненты системы (конфигурация, логирование, исключения)
- **data**: Получение и обработка метеорологических данных
- **detection**: Алгоритмы обнаружения и отслеживания циклонов
- **models**: Модели данных циклонов и параметров
- **visualization**: Функции визуализации результатов
- **export**: Экспорт данных в различные форматы

## Модуль core

### ConfigManager

Менеджер конфигурации для управления настройками системы.

#### Конструктор

```python
ConfigManager(config_path: Optional[Path] = None)
```

**Параметры:**
- `config_path`: Путь к файлу конфигурации. Если не указан, используется 'config.yaml'

#### Методы

##### get()

Получает значение параметра конфигурации.

```python
def get(self, section: str, key: Optional[str] = None) -> Any
```

**Параметры:**
- `section`: Раздел конфигурации (например, 'data', 'detection')
- `key`: Ключ параметра в разделе. Если не указан, возвращается весь раздел

**Возвращает:**
- Значение параметра или словарь с параметрами раздела

**Исключения:**
- `ValueError`: Если указанный раздел или ключ не найден

**Пример:**
```python
config = ConfigManager('config.yaml')
min_latitude = config.get('detection', 'min_latitude')
data_config = config.get('data')
```

##### update()

Обновляет значение параметра конфигурации и сохраняет в файл.

```python
def update(self, section: str, key: str, value: Any) -> None
```

**Параметры:**
- `section`: Раздел конфигурации
- `key`: Ключ параметра в разделе
- `value`: Новое значение параметра

##### set_data_source()

Добавляет или обновляет конфигурацию источника данных.

```python
def set_data_source(self, source_name: str, source_config: Dict[str, Any]) -> None
```

**Параметры:**
- `source_name`: Имя источника данных
- `source_config`: Словарь с параметрами источника

##### set_detection_criteria()

Обновляет критерии обнаружения циклонов.

```python
def set_detection_criteria(self, criteria: Dict[str, Any]) -> None
```

**Параметры:**
- `criteria`: Словарь с критериями обнаружения

### Исключения

#### DataSourceError

Ошибка при работе с источниками данных.

```python
class DataSourceError(Exception)
```

#### CredentialError

Ошибка аутентификации или работы с учетными данными.

```python
class CredentialError(Exception)
```

#### DetectionError

Ошибка в процессе обнаружения циклонов.

```python
class DetectionError(Exception)
```

#### TrackingError

Ошибка в процессе отслеживания циклонов.

```python
class TrackingError(Exception)
```

#### ExportError

Ошибка при экспорте данных.

```python
class ExportError(Exception)
```

#### VisualizationError

Ошибка при визуализации данных.

```python
class VisualizationError(Exception)
```

## Модуль data

### DataSourceManager

Управляет подключениями к различным источникам метеорологических данных.

#### Конструктор

```python
DataSourceManager(config_path: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 credentials: Optional[CredentialManager] = None)
```

**Параметры:**
- `config_path`: Путь к файлу конфигурации
- `cache_dir`: Директория для кэширования данных
- `credentials`: Экземпляр CredentialManager

#### Методы

##### get_data()

Получает метеорологические данные из указанного источника.

```python
def get_data(self, source: str, parameters: Dict, 
            region: Dict[str, float], timeframe: Dict,
            use_cache: bool = True) -> xr.Dataset
```

**Параметры:**
- `source`: Имя источника данных
- `parameters`: Параметры запроса (переменные, уровни и т.д.)
- `region`: Географический регион (север, юг, восток, запад)
- `timeframe`: Временные рамки запроса (годы, месяцы, дни, часы)
- `use_cache`: Использовать ли кэширование данных

**Возвращает:**
- Набор данных xarray с запрошенными метеорологическими данными

**Исключения:**
- `ValueError`: Если указан неподдерживаемый источник данных
- `DataSourceError`: При ошибке получения данных

**Пример:**
```python
data_manager = DataSourceManager()
dataset = data_manager.get_data(
    source="ERA5",
    parameters={'dataset_type': 'pressure_levels', 'variables': ['z', 'u', 'v']},
    region={'north': 90.0, 'south': 70.0, 'east': 180.0, 'west': -180.0},
    timeframe={'years': ['2020'], 'months': ['01'], 'days': ['01'], 'hours': ['00:00']}
)
```

##### register_custom_source()

Регистрирует пользовательский источник данных.

```python
def register_custom_source(self, name: str, adapter_class: type) -> None
```

**Параметры:**
- `name`: Имя источника данных
- `adapter_class`: Класс адаптера для источника

##### clear_cache()

Очищает кэш данных.

```python
def clear_cache(self, source: Optional[str] = None) -> None
```

**Параметры:**
- `source`: Имя источника данных для очистки кэша. Если не указано, очищается весь кэш

### CredentialManager

Управление учетными данными для доступа к источникам данных.

#### Конструктор

```python
CredentialManager(credentials_file: Optional[Path] = None)
```

**Параметры:**
- `credentials_file`: Путь к файлу с учетными данными

#### Методы

##### set()

Устанавливает учетные данные для источника.

```python
def set(self, source: str, **kwargs) -> None
```

**Параметры:**
- `source`: Имя источника данных
- `**kwargs`: Учетные данные (например, api_key)

**Пример:**
```python
credentials = CredentialManager()
credentials.set('ERA5', api_key='your_api_key_here')
```

##### get()

Получает учетные данные для источника.

```python
def get(self, source: str) -> Optional[Dict]
```

**Параметры:**
- `source`: Имя источника данных

**Возвращает:**
- Словарь с учетными данными или None, если данные отсутствуют

### ERA5Adapter

Адаптер для данных реанализа ERA5.

#### Конструктор

```python
ERA5Adapter(cache_dir: Path)
```

**Параметры:**
- `cache_dir`: Директория для кэширования данных

#### Методы

##### fetch()

Получает данные реанализа ERA5.

```python
def fetch(self, parameters: Dict, region: Dict, 
         timeframe: Dict, credentials: Dict) -> xr.Dataset
```

**Параметры:**
- `parameters`: Параметры запроса (переменные, уровни и т.д.)
- `region`: Географический регион
- `timeframe`: Временные рамки запроса
- `credentials`: Учетные данные для доступа к CDS API

**Возвращает:**
- Набор данных xarray с запрошенными метеорологическими данными

**Исключения:**
- `DataSourceError`: При ошибке получения данных
- `CredentialError`: При ошибке аутентификации

## Модуль detection

### CycloneDetector

Обнаруживает арктические мезоциклоны на основе настраиваемых критериев.

#### Конструктор

```python
CycloneDetector(min_latitude: float = 65.0, 
               config: Optional[Dict] = None, 
               debug_plot: bool = False)
```

**Параметры:**
- `min_latitude`: Минимальная широта для Арктического региона (по умолчанию 65°N)
- `config`: Конфигурация обнаружения циклонов из config.yaml
- `debug_plot`: Флаг для визуализации полей критериев обнаружения

#### Методы

##### detect()

Обнаруживает циклоны для заданного временного шага.

```python
def detect(self, dataset: xr.Dataset, time_step: Any) -> List[Cyclone]
```

**Параметры:**
- `dataset`: Набор метеорологических данных xarray
- `time_step`: Временной шаг для анализа

**Возвращает:**
- Список объектов Cyclone

**Пример:**
```python
detector = CycloneDetector(min_latitude=70.0)
cyclones = detector.detect(dataset, time_step)
```

##### set_criteria()

Устанавливает активные критерии обнаружения циклонов.

```python
def set_criteria(self, criterion_names: List[str]) -> None
```

**Параметры:**
- `criterion_names`: Список имен активных критериев

**Пример:**
```python
detector.set_criteria(['pressure_minimum', 'vorticity'])
```

##### detect_all_timesteps()

Обнаруживает циклоны для всех временных шагов в наборе данных.

```python
def detect_all_timesteps(self, dataset: xr.Dataset) -> Dict[Any, List[Cyclone]]
```

**Параметры:**
- `dataset`: Набор метеорологических данных xarray

**Возвращает:**
- Словарь с временными шагами в качестве ключей и списками циклонов в качестве значений

### CycloneTracker

Система отслеживания треков арктических циклонов.

#### Конструктор

```python
CycloneTracker(max_distance: float = 500.0,
              max_time_gap: float = 12.0,
              max_pressure_change: float = 20.0,
              min_track_duration: float = 9.0,
              min_track_points: int = 3,
              cluster_distance: float = 100.0,
              cluster_pressure_diff: float = 5.0,
              max_cyclone_speed: float = 120.0,
              debug_save_csv: bool = False,
              debug_dir: str = 'debug')
```

**Параметры:**
- `max_distance`: Максимальное расстояние между точками (км)
- `max_time_gap`: Максимальный временной разрыв (часы)
- `max_pressure_change`: Максимальное изменение давления (гПа)
- `min_track_duration`: Минимальная длительность трека (часы)
- `min_track_points`: Минимальное количество точек в треке
- `cluster_distance`: Расстояние для кластеризации (км)
- `cluster_pressure_diff`: Разность давления для кластеризации (гПа)
- `max_cyclone_speed`: Максимальная физическая скорость (км/ч)
- `debug_save_csv`: Сохранять ли debug CSV файлы
- `debug_dir`: Директория для debug файлов

#### Методы

##### track()

Основной метод трекинга с кластеризацией.

```python
def track(self, all_cyclones: Dict[Any, List[Cyclone]]) -> List[List[Cyclone]]
```

**Параметры:**
- `all_cyclones`: Словарь с циклонами для каждого временного шага

**Возвращает:**
- Список треков циклонов

**Пример:**
```python
tracker = CycloneTracker()
cyclone_tracks = tracker.track(all_cyclones)
```

##### filter_tracks()

Фильтрация треков с физическими критериями.

```python
def filter_tracks(self, tracks: List[List[Cyclone]],
                 min_duration: float = None,
                 min_points: int = None) -> List[List[Cyclone]]
```

**Параметры:**
- `tracks`: Список треков циклонов
- `min_duration`: Минимальная продолжительность трека (часы)
- `min_points`: Минимальное количество точек в треке

**Возвращает:**
- Отфильтрованный список треков

### CriteriaManager

Менеджер критериев обнаружения циклонов.

#### Конструктор

```python
CriteriaManager()
```

#### Методы

##### register_criterion()

Регистрирует критерий обнаружения.

```python
def register_criterion(self, name: str, criterion_class: Union[type, Callable])
```

**Параметры:**
- `name`: Имя критерия
- `criterion_class`: Класс или функция для создания критерия

##### set_active_criteria()

Устанавливает активные критерии обнаружения.

```python
def set_active_criteria(self, criterion_names: List[str])
```

**Параметры:**
- `criterion_names`: Список имен активных критериев

##### get_active_criteria()

Получает активные критерии обнаружения.

```python
def get_active_criteria(self) -> Dict[str, Union[BaseCriterion, Callable]]
```

**Возвращает:**
- Словарь с активными критериями

### BaseCriterion

Базовый класс для всех критериев обнаружения.

#### Методы

##### apply()

Применяет критерий к набору данных.

```python
def apply(self, dataset: xr.Dataset, time_step: Any, 
         debug_plot: bool = False, 
         output_dir: Optional[str] = None) -> List[Dict]
```

**Параметры:**
- `dataset`: Набор метеорологических данных xarray
- `time_step`: Временной шаг для анализа
- `debug_plot`: Если True, включает построение графиков полей критериев для отладки
- `output_dir`: Каталог для сохранения графиков, если debug_plot=True

**Возвращает:**
- Список кандидатов в циклоны (словари с координатами и свойствами)

**Исключения:**
- `DetectionError`: При ошибке обнаружения циклонов

## Модуль models

### Cyclone

Комплексное представление арктического мезоциклона.

#### Конструктор

```python
Cyclone(latitude: float, longitude: float, 
       time: Union[str, datetime, pd.Timestamp],
       central_pressure: float, 
       dataset: Optional[xr.Dataset] = None,
       detector: Optional[CycloneDetector] = None)
```

**Параметры:**
- `latitude`: Широта центра циклона (градусы)
- `longitude`: Долгота центра циклона (градусы)
- `time`: Время наблюдения
- `central_pressure`: Центральное давление на уровне моря (гПа)
- `dataset`: Исходный набор метеорологических данных
- `detector`: Детектор циклонов (необязательный)

#### Атрибуты

- `latitude`: Широта центра циклона
- `longitude`: Долгота центра циклона
- `time`: Время наблюдения
- `central_pressure`: Центральное давление
- `track_id`: Идентификатор трека
- `parameters`: Параметры циклона (CycloneParameters)
- `age`: Возраст циклона в часах
- `track`: Список координат трека
- `life_stage`: Стадия жизненного цикла

#### Методы

##### update()

Обновляет циклон новым наблюдением.

```python
def update(self, new_latitude: float, new_longitude: float, 
          new_time: Union[str, datetime, pd.Timestamp], new_pressure: float,
          dataset: Optional[xr.Dataset] = None)
```

##### calculate_lifecycle_metrics()

Рассчитывает метрики жизненного цикла циклона.

```python
def calculate_lifecycle_metrics(self) -> Dict[str, float]
```

**Возвращает:**
- Словарь с метриками:
  - `lifespan_hours`: Продолжительность жизни (часы)
  - `deepening_rate`: Скорость углубления (гПа/ч)
  - `displacement`: Смещение (км)
  - `mean_speed`: Средняя скорость (км/ч)

##### calculate_intensity()

Определяет интенсивность циклона.

```python
def calculate_intensity(self) -> CycloneIntensity
```

**Возвращает:**
- Категория интенсивности циклона

##### get_current_parameters()

Возвращает текущие параметры циклона в виде словаря.

```python
def get_current_parameters(self) -> Dict[str, Any]
```

**Возвращает:**
- Словарь с текущими параметрами циклона

### CycloneParameters

Параметры циклона.

#### Атрибуты

- `central_pressure`: Центральное давление (гПа)
- `vorticity_850hPa`: Завихренность на уровне 850 гПа (1/с)
- `max_wind_speed`: Максимальная скорость ветра (м/с)
- `radius`: Радиус циклона (км)
- `thermal_type`: Термический тип циклона
- `temperature_anomaly`: Температурная аномалия (K)
- `pressure_gradient`: Градиент давления (гПа/100км)

## Модуль visualization

### Основные функции визуализации

#### plot_cyclone_tracks()

Визуализирует треки циклонов на карте.

```python
def plot_cyclone_tracks(tracks: List[List[Cyclone]], 
                      region: Optional[Dict[str, float]] = None,
                      output_file: Optional[Union[str, Path]] = None,
                      figsize: Tuple[float, float] = (12, 10),
                      show_intensity: bool = True,
                      title: Optional[str] = None) -> plt.Figure
```

**Параметры:**
- `tracks`: Список треков циклонов
- `region`: Словарь с границами региона
- `output_file`: Путь для сохранения изображения
- `figsize`: Размер фигуры
- `show_intensity`: Цветовая кодировка по интенсивности
- `title`: Заголовок графика

**Возвращает:**
- Matplotlib Figure объект

#### create_cyclone_frequency_map()

Создает тепловую карту плотности циклонов.

```python
def create_cyclone_frequency_map(cyclones: List[Cyclone],
                               min_latitude: float = 60.0,
                               grid_resolution: float = 1.0,
                               smoothing_sigma: float = 1.5) -> Tuple[plt.Figure, plt.Axes]
```

**Параметры:**
- `cyclones`: Список циклонов
- `min_latitude`: Минимальная широта для анализа
- `grid_resolution`: Разрешение сетки (градусы)
- `smoothing_sigma`: Параметр сглаживания

**Возвращает:**
- Кортеж (figure, axis) с созданной картой

#### plot_cyclone_parameters()

Создает графики параметров циклона.

```python
def plot_cyclone_parameters(cyclone_track: List[Cyclone],
                          variables: List[str] = None,
                          output_file: Optional[Union[str, Path]] = None,
                          figsize: Tuple[float, float] = (12, 8)) -> plt.Figure
```

**Параметры:**
- `cyclone_track`: Трек циклона
- `variables`: Список переменных для отображения
- `output_file`: Путь для сохранения изображения
- `figsize`: Размер фигуры

**Возвращает:**
- Matplotlib Figure объект

## Модуль export

### CycloneCSVExporter

Экспортер данных о циклонах в формат CSV.

#### Конструктор

```python
CycloneCSVExporter(delimiter: str = ',', 
                  encoding: str = 'utf-8',
                  include_header: bool = True)
```

**Параметры:**
- `delimiter`: Разделитель полей в CSV
- `encoding`: Кодировка файла CSV
- `include_header`: Включать ли заголовок с именами полей

#### Методы

##### export_cyclone_tracks()

Экспортирует треки циклонов в файл CSV.

```python
def export_cyclone_tracks(self, cyclones: Union[List[Cyclone], List[List[Cyclone]]], 
                        filename: Union[str, Path]) -> Path
```

**Параметры:**
- `cyclones`: Список циклонов или список треков циклонов
- `filename`: Путь к выходному файлу CSV

**Возвращает:**
- Путь к созданному файлу

##### export_cyclone_statistics()

Экспортирует статистику циклонов в файл CSV.

```python
def export_cyclone_statistics(self, cyclones: List[Cyclone],
                            filename: Union[str, Path]) -> Path
```

**Параметры:**
- `cyclones`: Список циклонов для экспорта статистики
- `filename`: Путь к выходному файлу CSV

**Возвращает:**
- Путь к созданному файлу

##### import_from_csv()

Импортирует треки циклонов из файла CSV.

```python
def import_from_csv(self, filename: Union[str, Path]) -> List[List[Cyclone]]
```

**Параметры:**
- `filename`: Путь к файлу CSV

**Возвращает:**
- Список треков циклонов

**Исключения:**
- `ExportError`: При ошибке импорта данных

### CycloneNetCDFExporter

Экспортер данных о циклонах в формат NetCDF.

#### Конструктор

```python
CycloneNetCDFExporter(compression_level: int = 4,
                     include_metadata: bool = True)
```

**Параметры:**
- `compression_level`: Уровень сжатия (0-9)
- `include_metadata`: Включать ли метаданные

#### Методы

##### export_cyclone_data()

Экспортирует данные циклонов в файл NetCDF.

```python
def export_cyclone_data(self, cyclones: Union[List[Cyclone], List[List[Cyclone]]], 
                      filename: Union[str, Path]) -> Path
```

**Параметры:**
- `cyclones`: Список циклонов или список треков циклонов
- `filename`: Путь к выходному файлу NetCDF

**Возвращает:**
- Путь к созданному файлу

##### import_from_netcdf()

Импортирует данные циклонов из файла NetCDF.

```python
def import_from_netcdf(self, filename: Union[str, Path]) -> xr.Dataset
```

**Параметры:**
- `filename`: Путь к файлу NetCDF

**Возвращает:**
- Набор данных xarray