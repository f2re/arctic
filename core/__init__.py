"""
Основной модуль ядра системы ArcticCyclone для обнаружения и анализа мезоциклонов в Арктике.

Этот модуль содержит базовые компоненты, необходимые для работы всей системы, включая
конфигурацию, обработку исключений и настройку логирования.
"""

from .config import ConfigManager
from .exceptions import (
    ArcticCycloneError,
    ConfigurationError,
    DataSourceError,
    DetectionError
)
from .logging_setup import setup_logging

__all__ = [
    'ConfigManager',
    'ArcticCycloneError',
    'ConfigurationError',
    'DataSourceError',
    'DetectionError',
    'setup_logging'
]