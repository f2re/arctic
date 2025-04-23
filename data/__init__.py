"""
Пакет управления данными для системы ArcticCyclone.

Этот пакет содержит модули для получения, обработки и управления метеорологическими 
данными из различных источников, с акцентом на данные реанализа для арктического региона.
"""

from .acquisition import DataSourceManager
from .credentials import CredentialManager
from .catalog import DataCatalog, DatasetEntry
from .adapters.era5 import ERA5Adapter
from .processors.era5_processor import ERA5Processor

__all__ = [
    'DataSourceManager',
    'ERA5Adapter',
    'CredentialManager',
    'DataCatalog',
    'DatasetEntry',
    'ERA5Processor'
]