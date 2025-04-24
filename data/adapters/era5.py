"""
Модуль адаптера ERA5 для системы ArcticCyclone.

Обеспечивает специализированный доступ к данным реанализа ERA5 
через API Copernicus Climate Data Store (CDS).
"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List, Any
from datetime import datetime
import logging
import os

from core.exceptions import DataSourceError, CredentialError
from data.base import BaseDataAdapter

# Инициализация логгера
logger = logging.getLogger(__name__)

class ERA5Adapter(BaseDataAdapter):
    """
    Адаптер для данных реанализа ERA5.
    
    Обеспечивает специализированный доступ к данным ERA5 через CDS API с поддержкой
    различных типов данных: на уровнях давления, на поверхности и т.д.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Инициализирует адаптер ERA5.
        
        Аргументы:
            cache_dir: Директория для кэширования данных.
        """
        super().__init__(cache_dir)
        # Updated API URL to the standard CDS endpoint without /api/v2
        self.base_url = "https://cds.climate.copernicus.eu"
        self.available_datasets = {
            "pressure_levels": "reanalysis-era5-pressure-levels",
            "surface": "reanalysis-era5-single-levels",
            "land": "reanalysis-era5-land",
            "ensemble_members": "reanalysis-era5-complete",
            "monthly": "reanalysis-era5-pressure-levels-monthly-means"
        }
        
    def fetch(self, parameters: Dict, region: Dict, 
             timeframe: Dict, credentials: Dict) -> xr.Dataset:
        """
        Получает данные реанализа ERA5.
        
        Аргументы:
            parameters: Параметры запроса (переменные, уровни и т.д.).
            region: Географический регион (север, юг, восток, запад).
            timeframe: Временные рамки запроса (годы, месяцы, дни, часы).
            credentials: Учетные данные для доступа к CDS API.
            
        Возвращает:
            Набор данных xarray с запрошенными метеорологическими данными.
            
        Вызывает:
            DataSourceError: При ошибке получения данных.
            CredentialError: При ошибке аутентификации.
        """
        try:
            # Проверяем учетные данные
            if not credentials or 'api_key' not in credentials:
                raise CredentialError("Отсутствуют учетные данные для доступа к ERA5")
            
            # Проверяем параметры запроса
            if not self._validate_region(region) or not self._validate_timeframe(timeframe):
                raise ValueError("Некорректные параметры региона или временных рамок")
            
            # Определяем тип данных ERA5
            dataset_type = parameters.get('dataset_type', 'pressure_levels')
            if dataset_type not in self.available_datasets:
                raise ValueError(f"Неизвестный тип данных ERA5: {dataset_type}")
            
            # Проверяем, что переменные соответствуют типу данных
            if dataset_type == 'surface':
                # Проверка переменных для single-levels (surface)
                for var in parameters.get('variables', []):
                    if var in ['z', 'u', 'v', 't', 'q', 'vo', 'd', 'r']:
                        logger.warning(f"Переменная '{var}' является переменной уровня давления, но запрашивается как поверхностная. Это может привести к ошибке.")
            
            if dataset_type == 'pressure_levels':
                # Проверка переменных для pressure-levels
                for var in parameters.get('variables', []):
                    if var in ['msl', 'sp', 'tp', '2t', '10u', '10v', 'skt', 'tcc', 'blh']:
                        logger.warning(f"Переменная '{var}' является поверхностной, но запрашивается на уровнях давления. Это может привести к ошибке.")
            
            # Импортируем CDS API
            import cdsapi
            
            logger.info(f"Инициализация клиента CDS API для получения данных ERA5 ({dataset_type})")
            # Let the cdsapi module use default URL configuration from ~/.cdsapirc
            # Just pass the API key and let the library handle proper URL construction
            client = cdsapi.Client(key=credentials.get('api_key'))
            
            # Подготовка параметров запроса в соответствии с типом данных
            request_params = self._prepare_request_params(parameters, region, timeframe, dataset_type)
            
            # Запрос и загрузка данных
            temp_file = self.cache_dir / f"temp_era5_{dataset_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.nc"
            
            logger.info(f"Запрос данных ERA5 ({dataset_type}): {request_params}")
            
            # Get the correct dataset name for the API
            dataset_name = self.available_datasets[dataset_type]
            
            # Double-check and remove 'levtype': 'pl' for surface data
            if dataset_type == 'surface':
                # This is the most important fix - ensure surface vars don't use pressure levels
                if 'levtype' in request_params and request_params['levtype'] != 'sfc':
                    logger.warning(f"Correcting level type for surface data request")
                    request_params['levtype'] = 'sfc'
            
            client.retrieve(dataset_name, request_params, temp_file)
            
            logger.info(f"Данные успешно загружены: {temp_file}")
            dataset = xr.open_dataset(temp_file)
            # Переименование координаты времени для обеспечения совместимости
            if 'valid_time' in dataset.dims and 'time' not in dataset.dims:
                logger.info("Переименование координаты 'valid_time' в 'time' для совместимости")
                dataset = dataset.rename({'valid_time': 'time'})

            # Применяем постобработку в зависимости от типа данных
            dataset = self._postprocess_dataset(dataset, dataset_type)
            
            # Для чистоты удаляем временный файл
            temp_file.unlink()
            
            # Добавляем атрибуты для отслеживания
            dataset.attrs['source'] = 'ERA5'
            dataset.attrs['dataset_type'] = dataset_type
            dataset.attrs['retrieved_at'] = datetime.now().isoformat()
            dataset.attrs['request_parameters'] = str(request_params)
            
            return dataset
            
        except CredentialError as e:
            logger.error(f"Ошибка аутентификации ERA5: {str(e)}")
            raise
            
        except Exception as e:
            error_msg = f"Непредвиденная ошибка при получении данных ERA5: {str(e)}"
            logger.error(error_msg)
            raise DataSourceError(error_msg)
    
    def _prepare_request_params(self, parameters: Dict, region: Dict, 
                              timeframe: Dict, dataset_type: str) -> Dict:
        """
        Подготавливает параметры запроса в соответствии с типом данных ERA5.
        
        Аргументы:
            parameters: Параметры запроса.
            region: Географический регион.
            timeframe: Временные рамки.
            dataset_type: Тип данных ERA5.
            
        Возвращает:
            Словарь с параметрами запроса для CDS API.
        """
        # Базовые параметры для всех типов данных
        request_params = {
            'format': 'netcdf',
            'year': timeframe.get('years', []),
            'month': timeframe.get('months', []),
            'day': timeframe.get('days', []),
            'time': timeframe.get('hours', []),
            'area': [
                region.get('north', 90), 
                region.get('west', -180), 
                region.get('south', -90), 
                region.get('east', 180)
            ],
            'product_type': 'reanalysis',
        }
        
        # Добавляем специфические параметры в зависимости от типа данных
        if dataset_type == 'pressure_levels':
            request_params['variable'] = parameters.get('variables', [])
            request_params['pressure_level'] = parameters.get('levels', [])
            
        elif dataset_type == 'surface':
            request_params['variable'] = parameters.get('variables', [])
            # Fixed: Set correct level type for surface data
            # This is the critical fix - ERA5 API requires this
            request_params['levtype'] = 'sfc'
            
        elif dataset_type == 'land':
            request_params['variable'] = parameters.get('variables', [])
            
        elif dataset_type == 'ensemble_members':
            request_params['variable'] = parameters.get('variables', [])
            request_params['pressure_level'] = parameters.get('levels', [])
            request_params['product_type'] = 'ensemble_members'
            request_params['number'] = parameters.get('ensemble_members', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            
        elif dataset_type == 'monthly':
            request_params['variable'] = parameters.get('variables', [])
            request_params['pressure_level'] = parameters.get('levels', [])
            request_params['product_type'] = 'monthly_averaged_reanalysis'
            # Для месячных данных не используются параметры day и time
            request_params.pop('day', None)
            request_params.pop('time', None)
        
        # Фильтруем пустые параметры
        return {k: v for k, v in request_params.items() if v}
    
    def _postprocess_dataset(self, dataset: xr.Dataset, dataset_type: str) -> xr.Dataset:
        """
        Применяет постобработку к набору данных в зависимости от типа.
        
        Аргументы:
            dataset: Исходный набор данных.
            dataset_type: Тип данных ERA5.
            
        Возвращает:
            Обработанный набор данных.
        """
        # Переименовываем стандартные переменные для единообразия
        renaming_map = {
            'z': 'geopotential',
            't': 'temperature',
            'u': 'u_component_of_wind',
            'v': 'v_component_of_wind',
            'q': 'specific_humidity',
            'vo': 'vorticity',
            'msl': 'mean_sea_level_pressure',
            'd': 'divergence',
            'r': 'relative_humidity',
            'blh': 'boundary_layer_height',
            'tcc': 'total_cloud_cover',
            'sp': 'surface_pressure',
            'tp': 'total_precipitation',
            'skt': 'skin_temperature',
            '2t': '2m_temperature',
            '10u': '10m_u_component_of_wind',
            '10v': '10m_v_component_of_wind'
        }
        
        # Применяем переименование только для присутствующих переменных
        rename_dict = {old: new for old, new in renaming_map.items() if old in dataset}
        if rename_dict:
            dataset = dataset.rename(rename_dict)
        
        # Специфическая обработка в зависимости от типа данных
        if dataset_type == 'pressure_levels':
            # Переименовываем измерения для единообразия
            if 'level' in dataset.dims:
                dataset = dataset.rename({'level': 'pressure_level'})
            
            # Преобразуем единицы измерения, если необходимо
            if 'geopotential' in dataset:
                # Преобразуем геопотенциал в геопотенциальную высоту (м)
                if dataset.geopotential.max() > 100000:  # Проверка, что единицы - м²/с²
                    dataset['geopotential_height'] = dataset.geopotential / 9.80665
                    dataset.geopotential_height.attrs['units'] = 'm'
                    dataset.geopotential_height.attrs['long_name'] = 'Geopotential height'
            
        elif dataset_type == 'surface':
            # Преобразования для данных на поверхности
            if 'mean_sea_level_pressure' in dataset and dataset.mean_sea_level_pressure.max() > 50000:
                # Преобразуем Па в гПа
                dataset['mean_sea_level_pressure'] = dataset.mean_sea_level_pressure / 100
                dataset.mean_sea_level_pressure.attrs['units'] = 'hPa'
        
        # Добавляем координаты широты и долготы в градусах, если они в другом формате
        if 'latitude' in dataset.coords and 'longitude' in dataset.coords:
            if dataset.latitude.max() > 90 or dataset.longitude.max() > 180:
                # Преобразуем координаты, если они в нестандартном формате
                dataset = dataset.assign_coords(
                    latitude=dataset.latitude % 180 - 90 * (dataset.latitude // 180 % 2),
                    longitude=dataset.longitude % 360 - 180 * (dataset.longitude // 180 % 2)
                )
        
        return dataset
    
    def fetch_variables_info(self, dataset_type: str = 'pressure_levels', 
                           credentials: Dict = None) -> Dict[str, Dict]:
        """
        Получает информацию о доступных переменных для заданного типа данных ERA5.
        
        Аргументы:
            dataset_type: Тип данных ERA5.
            credentials: Учетные данные для доступа к CDS API.
            
        Возвращает:
            Словарь с информацией о переменных.
            
        Примечание:
            Эта функция не выполняет запрос к API, а возвращает статическую информацию.
        """
        # Словарь с информацией о переменных для разных типов данных
        variables_info = {
            'pressure_levels': {
                'geopotential': {
                    'code': 'z',
                    'units': 'm²/s²',
                    'description': 'Геопотенциал'
                },
                'temperature': {
                    'code': 't',
                    'units': 'K',
                    'description': 'Температура'
                },
                'u_component_of_wind': {
                    'code': 'u',
                    'units': 'm/s',
                    'description': 'U-компонента ветра'
                },
                'v_component_of_wind': {
                    'code': 'v',
                    'units': 'm/s',
                    'description': 'V-компонента ветра'
                },
                'specific_humidity': {
                    'code': 'q',
                    'units': 'kg/kg',
                    'description': 'Удельная влажность'
                },
                'vorticity': {
                    'code': 'vo',
                    'units': 's⁻¹',
                    'description': 'Относительная завихренность'
                },
                'divergence': {
                    'code': 'd',
                    'units': 's⁻¹',
                    'description': 'Дивергенция'
                },
                'relative_humidity': {
                    'code': 'r',
                    'units': '%',
                    'description': 'Относительная влажность'
                },
            },
            'surface': {
                'mean_sea_level_pressure': {
                    'code': 'msl',
                    'units': 'Pa',
                    'description': 'Давление на уровне моря'
                },
                'total_precipitation': {
                    'code': 'tp',
                    'units': 'm',
                    'description': 'Общее количество осадков'
                },
                '2m_temperature': {
                    'code': '2t',
                    'units': 'K',
                    'description': 'Температура на высоте 2 м'
                },
                '10m_u_component_of_wind': {
                    'code': '10u',
                    'units': 'm/s',
                    'description': 'U-компонента ветра на высоте 10 м'
                },
                '10m_v_component_of_wind': {
                    'code': '10v',
                    'units': 'm/s',
                    'description': 'V-компонента ветра на высоте 10 м'
                },
                'boundary_layer_height': {
                    'code': 'blh',
                    'units': 'm',
                    'description': 'Высота пограничного слоя'
                },
                'total_cloud_cover': {
                    'code': 'tcc',
                    'units': '(0-1)',
                    'description': 'Общая облачность'
                },
                'surface_pressure': {
                    'code': 'sp',
                    'units': 'Pa',
                    'description': 'Давление на поверхности'
                },
                'skin_temperature': {
                    'code': 'skt',
                    'units': 'K',
                    'description': 'Температура поверхности'
                },
            }
        }
        
        # Проверяем наличие информации для запрошенного типа данных
        if dataset_type not in variables_info:
            raise ValueError(f"Информация о переменных для типа данных {dataset_type} недоступна")
            
        return variables_info[dataset_type]
    
    def get_available_levels(self, dataset_type: str = 'pressure_levels') -> List[int]:
        """
        Возвращает список доступных уровней давления для данных ERA5.
        
        Аргументы:
            dataset_type: Тип данных ERA5.
            
        Возвращает:
            Список доступных уровней давления в гПа.
        """
        if dataset_type != 'pressure_levels' and dataset_type != 'monthly':
            return []
            
        # Стандартные уровни давления в ERA5 (гПа)
        return [
            1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 
            200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 
            700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000
        ]