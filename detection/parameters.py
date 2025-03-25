# Параметры для разных типов циклонов
"""
Функции для получения параметров обнаружения различных типов циклонов.
"""

import sys
import os

# Добавляем корневой каталог в путь для импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config import SYNOPTIC_CYCLONE_PARAMS, MESOSCALE_CYCLONE_PARAMS, POLAR_LOW_PARAMS
except ImportError:
    # Альтернативный способ импорта, если стандартный не работает
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", 
             os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.py')))
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    SYNOPTIC_CYCLONE_PARAMS = config.SYNOPTIC_CYCLONE_PARAMS
    MESOSCALE_CYCLONE_PARAMS = config.MESOSCALE_CYCLONE_PARAMS
    POLAR_LOW_PARAMS = config.POLAR_LOW_PARAMS

def get_cyclone_params(cyclone_type):
    """
    Возвращает параметры обнаружения для указанного типа циклонов.
    
    Параметры:
    ----------
    cyclone_type : str
        Тип циклонов: 'synoptic', 'mesoscale' или 'polar_low'
        
    Возвращает:
    -----------
    dict
        Словарь с параметрами обнаружения
    """
    if cyclone_type.lower() == 'synoptic':
        return SYNOPTIC_CYCLONE_PARAMS
    elif cyclone_type.lower() == 'mesoscale':
        return MESOSCALE_CYCLONE_PARAMS
    elif cyclone_type.lower() in ['polar_low', 'polar']:
        return POLAR_LOW_PARAMS
    else:
        # По умолчанию используем параметры для мезомасштабных циклонов
        print(f"Предупреждение: неизвестный тип циклонов '{cyclone_type}'. "
              f"Используются параметры для мезомасштабных циклонов.")
        return MESOSCALE_CYCLONE_PARAMS