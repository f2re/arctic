# Работа с источниками данных в ArcticCyclone

## Содержание

1. [Поддерживаемые источники данных](#поддерживаемые-источники-данных)
2. [Настройка учетных данных ERA5](#настройка-учетных-данных-era5)
3. [Структура запросов к данным](#структура-запросов-к-данным)
4. [Кэширование данных](#кэширование-данных)
5. [Добавление новых источников данных](#добавление-новых-источников-данных)

## Поддерживаемые источники данных

Система ArcticCyclone поддерживает различные источники метеорологических данных. В текущей реализации основным источником является ERA5 от Copernicus Climate Data Store.

### ERA5 (European ReAnalysis 5)

ERA5 - это пятая версия глобального реанализа Европейского центра среднесрочных прогнозов погоды (ECMWF). Он предоставляет данные о состоянии атмосферы, океана и суши с высоким пространственно-временным разрешением.

#### Особенности ERA5:

- **Пространственное разрешение**: 0.25° × 0.25° (~28 км)
- **Временное разрешение**: Ежечасные данные
- **Вертикальное разрешение**: 137 уровней от поверхности до 0.01 гПа
- **Период данных**: С 1950 года по настоящее время (с задержкой 5 дней)
- **Параметры**: Более 100 метеорологических параметров

#### Типы данных ERA5:

1. **Данные на уровнях давления**:
   - Геопотенциал (z)
   - Температура (t)
   - Компоненты ветра (u, v)
   - Удельная влажность (q)
   - Относительная завихренность (vo)
   - Дивергенция (d)
   - Относительная влажность (r)

2. **Поверхностные данные**:
   - Давление на уровне моря (msl)
   - Давление на поверхности (sp)
   - Температура на высоте 2 м (2t)
   - Компоненты ветра на высоте 10 м (10u, 10v)
   - Общее количество осадков (tp)
   - Общая облачность (tcc)
   - Температура поверхности (skt)
   - Высота пограничного слоя (blh)

## Настройка учетных данных ERA5

Для получения данных ERA5 необходимо зарегистрироваться в Climate Data Store (CDS) и получить API-ключ.

### Регистрация в CDS

1. Перейдите на сайт https://cds.climate.copernicus.eu/
2. Нажмите "Login/Register" в правом верхнем углу
3. Зарегистрируйтесь или войдите в существующую учетную запись
4. Перейдите в раздел "Your profile"
5. Найдите раздел "API key" и скопируйте ваш ключ

### Настройка файла конфигурации

Создайте файл `~/.cdsapirc` в вашей домашней директории со следующим содержимым:

```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_API_KEY_HERE
```

Замените `YOUR_API_KEY_HERE` на ваш реальный API-ключ.

### Настройка через код

Также можно настроить учетные данные программно:

```python
from data.credentials import CredentialManager

# Создание менеджера учетных данных
credentials = CredentialManager()

# Установка API-ключа для ERA5
credentials.set('ERA5', api_key='ваш_ключ_API_ERA5')

# Использование в DataSourceManager
from data.acquisition import DataSourceManager
data_manager = DataSourceManager(credentials=credentials)
```

### Проверка подключения

После настройки можно проверить подключение:

```python
from data.acquisition import DataSourceManager
from data.credentials import CredentialManager

# Создание менеджера учетных данных
credentials = CredentialManager()

# Создание менеджера источников данных
data_manager = DataSourceManager(credentials=credentials)

# Проверка наличия учетных данных
if credentials.get('ERA5'):
    print("Учетные данные ERA5 настроены корректно")
else:
    print("Учетные данные ERA5 отсутствуют")
```

## Структура запросов к данным

Запросы к данным ERA5 осуществляются через `DataSourceManager` с использованием определенной структуры параметров.

### Параметры запроса

#### parameters

Словарь с параметрами запроса:

```python
parameters = {
    'dataset_type': 'pressure_levels',  # или 'surface'
    'variables': ['z', 'u', 'v', 't', 'vo'],
    'levels': [1000, 925, 850, 700, 500]  # только для pressure_levels
}
```

#### region

Словарь с географическим регионом:

```python
region = {
    'north': 90.0,   # Северная граница
    'south': 65.0,   # Южная граница
    'east': 180.0,   # Восточная граница
    'west': -180.0   # Западная граница
}
```

#### timeframe

Словарь с временным диапазоном:

```python
timeframe = {
    'years': ['2020'],
    'months': ['01', '02', '03'],
    'days': ['01', '02', '03', '04', '05'],
    'hours': ['00:00', '06:00', '12:00', '18:00']
}
```

### Пример запроса данных

```python
from data.acquisition import DataSourceManager
from data.credentials import CredentialManager

# Настройка учетных данных
credentials = CredentialManager()
credentials.set('ERA5', api_key='ваш_ключ_API_ERA5')

# Создание менеджера данных
data_manager = DataSourceManager(credentials=credentials)

# Определение параметров запроса
region = {
    'north': 90.0,
    'south': 70.0,
    'east': 180.0,
    'west': -180.0
}

timeframe = {
    'years': ['2020'],
    'months': ['01'],
    'days': ['01', '02', '03'],
    'hours': ['00:00', '06:00', '12:00', '18:00']
}

parameters = {
    'dataset_type': 'pressure_levels',
    'variables': ['z', 'u', 'v', 't', 'vo'],
    'levels': [850, 700, 500]
}

# Запрос данных
dataset = data_manager.get_data(
    source="ERA5",
    parameters=parameters,
    region=region,
    timeframe=timeframe,
    use_cache=True
)

print(f"Получен набор данных с измерениями: {dataset.dims}")
```

## Кэширование данных

Система автоматически кэширует загруженные данные для повышения эффективности и уменьшения количества запросов к серверам.

### Настройка кэширования

Кэширование настраивается в конфигурационном файле:

```yaml
data:
  cache_dir: 'data/cache'
```

### Как работает кэширование

1. При каждом запросе данных система вычисляет хеш на основе параметров запроса
2. Проверяется наличие файла с таким хешем в директории кэша
3. Если файл существует, данные загружаются из кэша
4. Если файл отсутствует, данные загружаются с сервера и сохраняются в кэш

### Управление кэшем

```python
from data.acquisition import DataSourceManager

# Очистка всего кэша
data_manager.clear_cache()

# Очистка кэша для конкретного источника
data_manager.clear_cache(source='ERA5')
```

## Добавление новых источников данных

Система поддерживает расширение новыми источниками данных через механизм адаптеров.

### Создание адаптера

Для добавления нового источника данных необходимо создать класс, наследующийся от `BaseDataAdapter`:

```python
from data.base import BaseDataAdapter
import xarray as xr

class MyCustomDataAdapter(BaseDataAdapter):
    def __init__(self, cache_dir):
        super().__init__(cache_dir)
        # Инициализация адаптера
    
    def fetch(self, parameters, region, timeframe, credentials):
        # Реализация получения данных
        # Возвращаем xarray.Dataset
        pass
    
    def _validate_region(self, region):
        # Проверка корректности региона
        pass
    
    def _validate_timeframe(self, timeframe):
        # Проверка корректности временного диапазона
        pass
```

### Регистрация адаптера

```python
from data.acquisition import DataSourceManager

# Регистрация пользовательского адаптера
data_manager.register_custom_source('MySource', MyCustomDataAdapter)

# Использование нового источника
dataset = data_manager.get_data(
    source='MySource',
    parameters={},
    region={},
    timeframe={}
)
```

### Пример адаптера для локальных данных

```python
from data.base import BaseDataAdapter
import xarray as xr
import os

class LocalDataAdapter(BaseDataAdapter):
    def __init__(self, cache_dir, data_directory='local_data'):
        super().__init__(cache_dir)
        self.data_directory = data_directory
    
    def fetch(self, parameters, region, timeframe, credentials):
        # Пример: загрузка данных из локального NetCDF файла
        variable = parameters.get('variable', 'msl')
        year = timeframe.get('years', ['2020'])[0]
        
        # Формирование пути к файлу
        filename = f"{variable}_{year}.nc"
        filepath = os.path.join(self.data_directory, filename)
        
        # Загрузка данных
        if os.path.exists(filepath):
            dataset = xr.open_dataset(filepath)
            
            # Применение региональной маски
            dataset = dataset.where(
                (dataset.latitude >= region['south']) & 
                (dataset.latitude <= region['north']) &
                (dataset.longitude >= region['west']) & 
                (dataset.longitude <= region['east']),
                drop=True
            )
            
            return dataset
        else:
            raise FileNotFoundError(f"Файл {filepath} не найден")
    
    def _validate_region(self, region):
        required_keys = ['north', 'south', 'east', 'west']
        return all(key in region for key in required_keys)
    
    def _validate_timeframe(self, timeframe):
        required_keys = ['years', 'months', 'days', 'hours']
        return all(key in timeframe for key in required_keys)
```