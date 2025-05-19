"""
Модуль получения метеорологических данных для системы ArcticCyclone.

Обеспечивает унифицированный интерфейс для получения данных из различных источников,
включая службы реанализа, спутниковые данные и наблюдения наземных станций.
"""

import os
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import logging
import hashlib

from core.exceptions import DataSourceError, CredentialError
from .credentials import CredentialManager
from .adapters.era5 import ERA5Adapter

# Инициализация логгера
logger = logging.getLogger(__name__)


class DataSourceManager:
    """
    Управляет подключениями к различным источникам метеорологических данных.
    
    Предоставляет единый интерфейс для получения данных из различных источников
    с поддержкой кэширования и учета аутентификационных данных.
    """
    
    def __init__(self, config_path: Optional[Path] = None,
             cache_dir: Optional[Path] = None,
             credentials: Optional[CredentialManager] = None):
        """
        Инициализирует менеджер источников данных.

        Аргументы:
            config_path: Путь к файлу конфигурации. Если не указан, используется
                        конфигурация по умолчанию.
            cache_dir: Директория для кэширования данных. Если не указана, используется
                    директория из конфигурации.
            credentials: Экземпляр CredentialManager. Если не указан, создается новый.
        """
        from core.config import ConfigManager
        self.config = ConfigManager(config_path)
        self.credentials = credentials or CredentialManager()
        self.cache_dir = cache_dir or Path(self.config.get('data', 'cache_dir'))
        self.adapters = self._initialize_adapters()
        # Создаем директорию кэша, если она не существует
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Инициализирован менеджер источников данных с {len(self.adapters)} адаптерами")

    
    def _initialize_adapters(self) -> Dict[str, Any]:
        """
        Инициализирует адаптеры для доступных источников данных.
        
        Возвращает:
            Словарь с адаптерами, где ключ - имя источника данных.
        """
        adapters = {}
        
        # Инициализация адаптера ERA5 по умолчанию
        adapters['ERA5'] = ERA5Adapter(self.cache_dir)
        
        # Инициализация дополнительных адаптеров из конфигурации
        sources_config = self.config.get('data', 'sources')
        for source_name, source_config in sources_config.items():
            if source_name not in adapters:
                adapter_class = self._get_adapter_class(source_config.get('type', 'unknown'))
                if adapter_class:
                    adapters[source_name] = adapter_class(self.cache_dir)
        
        return adapters
    
    def _get_adapter_class(self, source_type: str) -> Optional[type]:
        """
        Определяет класс адаптера на основе типа источника данных.
        
        Аргументы:
            source_type: Тип источника данных.
            
        Возвращает:
            Класс адаптера или None, если подходящий адаптер не найден.
        """
        adapter_mapping = {
            'reanalysis': ERA5Adapter,
            'satellite': None,  # Заглушка для будущих реализаций
            'station': None,    # Заглушка для будущих реализаций
        }
        
        return adapter_mapping.get(source_type)
    
    def get_data(self, source: str, parameters: Dict, 
                region: Dict[str, float], timeframe: Dict,
                use_cache: bool = True) -> xr.Dataset:
        """
        Получает метеорологические данные из указанного источника.
        
        Аргументы:
            source: Имя источника данных.
            parameters: Параметры запроса (переменные, уровни и т.д.).
            region: Географический регион (север, юг, восток, запад).
            timeframe: Временные рамки запроса (годы, месяцы, дни, часы).
            use_cache: Использовать ли кэширование данных.
            
        Возвращает:
            Набор данных xarray с запрошенными метеорологическими данными.
            
        Вызывает:
            ValueError: Если указан неподдерживаемый источник данных.
            DataSourceError: При ошибке получения данных.
        """
        if source not in self.adapters:
            raise ValueError(f"Неподдерживаемый источник данных: {source}")
        
        adapter = self.adapters[source]
        
        try:
            # Получаем учетные данные для источника
            credentials = self.credentials.get(source)
            
            # Формируем хеш запроса для кэширования
            cache_key = self._generate_cache_key(source, parameters, region, timeframe)
            cache_file = self.cache_dir / f"{cache_key}.nc"
            
            # Проверяем наличие данных в кэше
            if use_cache and cache_file.exists():
                logger.info(f"Загрузка данных из кэша: {cache_file}")
                return xr.open_dataset(cache_file)
            
            # Получаем данные от адаптера
            dataset = adapter.fetch(parameters, region, timeframe, credentials)
            
            # Сохраняем данные в кэш
            if use_cache:
                logger.info(f"Сохранение данных в кэш: {cache_file}")
                dataset.to_netcdf(cache_file)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Ошибка при получении данных из источника {source}: {str(e)}")
            raise DataSourceError(f"Не удалось получить данные из {source}: {str(e)}")
    
    def _generate_cache_key(self, source: str, parameters: Dict, 
                          region: Dict, timeframe: Dict) -> str:
        """
        Генерирует уникальный ключ для кэширования запроса.
        
        Аргументы:
            source: Имя источника данных.
            parameters: Параметры запроса.
            region: Географический регион.
            timeframe: Временные рамки запроса.
            
        Возвращает:
            Строка с хешем запроса.
        """
        # Преобразуем параметры в строку для хеширования
        params_str = f"{source}_{str(parameters)}_{str(region)}_{str(timeframe)}"
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def register_custom_source(self, name: str, adapter_class: type) -> None:
        """
        Регистрирует пользовательский источник данных.
        
        Аргументы:
            name: Имя источника данных.
            adapter_class: Класс адаптера для источника.
            
        Примечание:
            Это позволяет добавлять новые источники данных без изменения кода системы.
        """
        self.adapters[name] = adapter_class(self.cache_dir)
        logger.info(f"Зарегистрирован пользовательский источник данных: {name}")
    
    def clear_cache(self, source: Optional[str] = None) -> None:
        """
        Очищает кэш данных.
        
        Аргументы:
            source: Имя источника данных для очистки кэша. Если не указано,
                   очищается весь кэш.
        """
        if source:
            # Очистка кэша для конкретного источника
            pattern = f"{source}_*.nc"
            for file in self.cache_dir.glob(pattern):
                file.unlink()
            logger.info(f"Кэш для источника {source} очищен")
        else:
            # Очистка всего кэша
            for file in self.cache_dir.glob("*.nc"):
                file.unlink()
            logger.info("Весь кэш данных очищен")


class DataPreprocessor:
    """
    Обработчик метеорологических данных.
    
    Предоставляет методы для предварительной обработки данных:
    интерполяция, фильтрация, вычисление производных переменных и т.д.
    """
    
    def __init__(self, dataset: xr.Dataset):
        """
        Инициализирует обработчик данных.
        
        Аргументы:
            dataset: Набор данных xarray для обработки.
        """
        self.dataset = dataset
        
    def interpolate_grid(self, target_resolution: float) -> xr.Dataset:
        """
        Интерполирует данные на регулярную сетку с заданным разрешением.
        
        Аргументы:
            target_resolution: Желаемое разрешение сетки в градусах.
            
        Возвращает:
            Интерполированный набор данных.
        """
        # Получаем текущие границы данных
        lat_min, lat_max = float(self.dataset.latitude.min()), float(self.dataset.latitude.max())
        lon_min, lon_max = float(self.dataset.longitude.min()), float(self.dataset.longitude.max())
        
        # Создаем новую равномерную сетку
        new_lats = np.arange(lat_min, lat_max + target_resolution, target_resolution)
        new_lons = np.arange(lon_min, lon_max + target_resolution, target_resolution)
        
        # Интерполируем данные на новую сетку
        interpolated = self.dataset.interp(latitude=new_lats, longitude=new_lons)
        
        return interpolated
    
    def filter_polar_region(self, min_latitude: float = 65.0) -> xr.Dataset:
        """
        Фильтрует данные для полярного региона.
        
        Аргументы:
            min_latitude: Минимальная широта для фильтрации (по умолчанию 70° с.ш.).
            
        Возвращает:
            Отфильтрованный набор данных.
        """
        return self.dataset.where(self.dataset.latitude >= min_latitude, drop=True)
    
    def calculate_vorticity(self) -> xr.Dataset:
        """
        Вычисляет относительную завихренность.
        
        Возвращает:
            Набор данных с добавленной переменной завихренности.
        """
        # Проверяем наличие необходимых компонентов ветра
        if 'u_component_of_wind' not in self.dataset or 'v_component_of_wind' not in self.dataset:
            raise ValueError("Для расчета завихренности необходимы компоненты ветра u и v")
        
        # Рассчитываем градиенты
        dy = self.dataset.latitude.diff('latitude') * 111000  # примерно 111 км на градус широты
        dx = self.dataset.longitude.diff('longitude') * 111000 * np.cos(np.radians(self.dataset.latitude))
        
        # Рассчитываем завихренность как curl(V) = dv/dx - du/dy
        dvdx = self.dataset.v_component_of_wind.differentiate('longitude') / dx
        dudy = self.dataset.u_component_of_wind.differentiate('latitude') / dy
        
        vorticity = dvdx - dudy
        
        # Добавляем завихренность к набору данных
        result = self.dataset.copy()
        result['vorticity'] = vorticity
        result.vorticity.attrs['units'] = 's^-1'
        result.vorticity.attrs['long_name'] = 'Relative vorticity'
        
        return result
    
    def calculate_geostrophic_wind(self) -> xr.Dataset:
        """
        Вычисляет геострофический ветер на основе геопотенциала.
        
        Возвращает:
            Набор данных с добавленными компонентами геострофического ветра.
        """
        # Проверяем наличие геопотенциала
        if 'geopotential' not in self.dataset:
            raise ValueError("Для расчета геострофического ветра необходим геопотенциал")
        
        # Константы
        f = 2 * 7.29e-5 * np.sin(np.radians(self.dataset.latitude))  # параметр Кориолиса
        g = 9.80665  # ускорение свободного падения, м/с²
        
        # Рассчитываем градиенты
        dy = self.dataset.latitude.diff('latitude') * 111000  # примерно 111 км на градус широты
        dx = self.dataset.longitude.diff('longitude') * 111000 * np.cos(np.radians(self.dataset.latitude))
        
        # Рассчитываем компоненты геострофического ветра
        dZdy = self.dataset.geopotential.differentiate('latitude') / dy
        dZdx = self.dataset.geopotential.differentiate('longitude') / dx
        
        u_geo = -g / f * dZdy
        v_geo = g / f * dZdx
        
        # Добавляем компоненты геострофического ветра к набору данных
        result = self.dataset.copy()
        result['u_geo'] = u_geo
        result.u_geo.attrs['units'] = 'm s^-1'
        result.u_geo.attrs['long_name'] = 'Geostrophic wind U component'
        
        result['v_geo'] = v_geo
        result.v_geo.attrs['units'] = 'm s^-1'
        result.v_geo.attrs['long_name'] = 'Geostrophic wind V component'
        
        return result