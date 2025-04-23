"""
Модуль фабрики алгоритмов обнаружения циклонов.

Предоставляет класс-фабрику для создания экземпляров
различных алгоритмов обнаружения циклонов.
"""

from typing import Dict, Optional, Any, Type
import logging

from .base_algorithm import BaseDetectionAlgorithm
from .pressure_minima import PressureMinimaAlgorithm
from .multi_parameter import MultiParameterAlgorithm
from .arctic_mesocyclone import ArcticMesocycloneAlgorithm
from .serreze import SerrezeAlgorithm
from core.exceptions import DetectionError

# Инициализация логгера
logger = logging.getLogger(__name__)

class AlgorithmFactory:
    """
    Фабрика для создания алгоритмов обнаружения циклонов.
    
    Позволяет создавать экземпляры различных алгоритмов обнаружения
    циклонов с настраиваемыми параметрами.
    """
    
    # Словарь зарегистрированных алгоритмов
    _algorithms: Dict[str, Type[BaseDetectionAlgorithm]] = {
        'pressure_minima': PressureMinimaAlgorithm,
        'multi_parameter': MultiParameterAlgorithm,
        'arctic_mesocyclone': ArcticMesocycloneAlgorithm,
        'serreze': SerrezeAlgorithm
    }
    
    @classmethod
    def create(cls, algorithm_name: str, **kwargs) -> BaseDetectionAlgorithm:
        """
        Создает экземпляр алгоритма обнаружения циклонов.
        
        Аргументы:
            algorithm_name: Имя алгоритма для создания.
            **kwargs: Параметры для инициализации алгоритма.
            
        Возвращает:
            Экземпляр алгоритма обнаружения.
            
        Вызывает:
            DetectionError: Если алгоритм с указанным именем не зарегистрирован.
        """
        algorithm_class = cls.get_algorithm_class(algorithm_name)
        
        if algorithm_class is None:
            available_algorithms = list(cls._algorithms.keys())
            raise DetectionError(f"Алгоритм '{algorithm_name}' не зарегистрирован. "
                               f"Доступные алгоритмы: {available_algorithms}")
        
        try:
            return algorithm_class(**kwargs)
        except Exception as e:
            raise DetectionError(f"Ошибка при создании алгоритма '{algorithm_name}': {str(e)}")
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type[BaseDetectionAlgorithm]) -> None:
        """
        Регистрирует новый алгоритм обнаружения циклонов.
        
        Аргументы:
            name: Имя алгоритма для регистрации.
            algorithm_class: Класс алгоритма.
            
        Вызывает:
            ValueError: Если алгоритм с таким именем уже зарегистрирован.
        """
        if name in cls._algorithms:
            raise ValueError(f"Алгоритм с именем '{name}' уже зарегистрирован")
            
        cls._algorithms[name] = algorithm_class
        logger.info(f"Зарегистрирован алгоритм обнаружения: {name}")
    
    @classmethod
    def get_algorithm_class(cls, name: str) -> Optional[Type[BaseDetectionAlgorithm]]:
        """
        Возвращает класс алгоритма по имени.
        
        Аргументы:
            name: Имя алгоритма.
            
        Возвращает:
            Класс алгоритма или None, если алгоритм не зарегистрирован.
        """
        return cls._algorithms.get(name)
    
    @classmethod
    def list_algorithms(cls) -> Dict[str, Type[BaseDetectionAlgorithm]]:
        """
        Возвращает словарь всех зарегистрированных алгоритмов.
        
        Возвращает:
            Словарь с именами алгоритмов в качестве ключей и классами алгоритмов в качестве значений.
        """
        return cls._algorithms.copy()
    
    @classmethod
    def create_all(cls, **kwargs) -> Dict[str, BaseDetectionAlgorithm]:
        """
        Создает экземпляры всех зарегистрированных алгоритмов.
        
        Аргументы:
            **kwargs: Общие параметры для инициализации всех алгоритмов.
            
        Возвращает:
            Словарь с именами алгоритмов в качестве ключей и экземплярами алгоритмов в качестве значений.
        """
        result = {}
        
        for name, algorithm_class in cls._algorithms.items():
            try:
                result[name] = algorithm_class(**kwargs)
            except Exception as e:
                logger.error(f"Ошибка при создании алгоритма '{name}': {str(e)}")
        
        return result