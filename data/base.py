"""
Модуль базовых классов для пакета data системы ArcticCyclone.

Содержит базовые классы и интерфейсы, используемые в других модулях пакета,
для предотвращения циклических импортов и обеспечения единого интерфейса.
"""

import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Union, List, Any
import logging

# Инициализация логгера
logger = logging.getLogger(__name__)

class BaseDataAdapter:
    """
    Базовый класс для адаптеров источников данных.
    
    Определяет общий интерфейс для всех адаптеров и предоставляет
    базовую реализацию общих методов.
    """
    
    def __init__(self, cache_dir: Path):
        """
        Инициализирует базовый адаптер.
        
        Аргументы:
            cache_dir: Директория для кэширования данных.
        """
        self.cache_dir = cache_dir
    
    def fetch(self, parameters: Dict, region: Dict, 
             timeframe: Dict, credentials: Dict) -> xr.Dataset:
        """
        Получает данные из источника.
        
        Аргументы:
            parameters: Параметры запроса (переменные, уровни и т.д.).
            region: Географический регион (север, юг, восток, запад).
            timeframe: Временные рамки запроса (годы, месяцы, дни, часы).
            credentials: Учетные данные для доступа к API.
            
        Возвращает:
            Набор данных xarray с метеорологическими данными.
        
        Примечание:
            Этот метод должен быть переопределен в дочерних классах.
        """
        raise NotImplementedError("Метод fetch() должен быть переопределен в дочернем классе")
    
    def _validate_region(self, region: Dict[str, float]) -> bool:
        """
        Проверяет корректность указанного региона.
        
        Аргументы:
            region: Словарь с границами региона (север, юг, восток, запад).
            
        Возвращает:
            True, если регион корректен, иначе False.
        """
        required_keys = ['north', 'south', 'east', 'west']
        
        # Проверяем наличие всех необходимых ключей
        if not all(key in region for key in required_keys):
            return False
        
        # Проверяем диапазоны значений
        lat_valid = -90 <= region['south'] <= region['north'] <= 90
        lon_valid = -180 <= region['west'] <= 180 and -180 <= region['east'] <= 180
        
        return lat_valid and lon_valid
    
    def _validate_timeframe(self, timeframe: Dict) -> bool:
        """
        Проверяет корректность указанных временных рамок.
        
        Аргументы:
            timeframe: Словарь с временными рамками (годы, месяцы, дни, часы).
            
        Возвращает:
            True, если временные рамки корректны, иначе False.
        """
        required_keys = ['years', 'months', 'days', 'hours']
        
        # Проверяем наличие всех необходимых ключей
        return all(key in timeframe for key in required_keys)
