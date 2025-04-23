"""
Модуль управления конфигурацией для системы ArcticCyclone.

Предоставляет классы и функции для загрузки, сохранения и доступа к настройкам
системы в унифицированном формате.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

class ConfigManager:
    """
    Управляет конфигурацией для научных рабочих процессов исследования.
    
    Класс обеспечивает загрузку настроек из YAML-файла и предоставляет унифицированный
    интерфейс для доступа к параметрам конфигурации в разных частях системы.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Инициализирует менеджер конфигурации с опциональным путем к файлу.
        
        Аргументы:
            config_path: Путь к файлу конфигурации. Если не указан, используется 'config.yaml'
                         в текущей директории.
        """
        self.config_path = config_path or Path("config.yaml")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Загружает конфигурацию из файла.
        
        Возвращает:
            Словарь с параметрами конфигурации.
        
        Примечание:
            Если файл конфигурации отсутствует, создается файл с настройками по умолчанию.
        """
        if not self.config_path.exists():
            return self._create_default_config()
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Создает конфигурацию по умолчанию и сохраняет ее в файл.
        
        Возвращает:
            Словарь с параметрами конфигурации по умолчанию.
            
        Примечание:
            Автоматически создает необходимые директории, указанные в конфигурации.
        """
        default_config = {
            'data': {
                'default_source': 'ERA5',
                'cache_dir': 'data/cache',
                'sources': {
                    'ERA5': {
                        'type': 'reanalysis',
                        'variables': ['z', 'u', 'v', 't', 'q', 'vo'],
                        'levels': [1000, 925, 850, 700, 500, 300],
                    }
                }
            },
            'detection': {
                'min_latitude': 70.0,
                'criteria': {
                    'pressure_minimum': True,
                    'vorticity_threshold': 1e-5,
                    'wind_threshold': 15.0,
                    'closed_contour': True
                },
                'tracking': {
                    'max_distance': 300.0,
                    'max_pressure_change': 10.0
                }
            },
            'visualization': {
                'default_projection': 'NorthPolarStereo',
                'map_resolution': 'intermediate',
                'output_dir': 'output/figures'
            },
            'export': {
                'output_dir': 'output/data',
                'formats': ['csv', 'netcdf']
            }
        }
        
        # Создаем директории
        os.makedirs(Path(default_config['data']['cache_dir']), exist_ok=True)
        os.makedirs(Path(default_config['visualization']['output_dir']), exist_ok=True)
        os.makedirs(Path(default_config['export']['output_dir']), exist_ok=True)
        
        # Сохраняем конфигурацию по умолчанию
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
            
        return default_config
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Получает значение параметра конфигурации.
        
        Аргументы:
            section: Раздел конфигурации (например, 'data', 'detection').
            key: Ключ параметра в разделе. Если не указан, возвращается весь раздел.
            
        Возвращает:
            Значение параметра или словарь с параметрами раздела.
            
        Вызывает:
            ValueError: Если указанный раздел или ключ не найден.
        """
        if section not in self.config:
            raise ValueError(f"Раздел конфигурации '{section}' не найден")
            
        if key is None:
            return self.config[section]
            
        if key not in self.config[section]:
            raise ValueError(f"Ключ '{key}' не найден в разделе '{section}'")
            
        return self.config[section][key]
    
    def update(self, section: str, key: str, value: Any) -> None:
        """
        Обновляет значение параметра конфигурации и сохраняет в файл.
        
        Аргументы:
            section: Раздел конфигурации.
            key: Ключ параметра в разделе.
            value: Новое значение параметра.
            
        Примечание:
            Если указанный раздел не существует, он будет создан.
        """
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section][key] = value
        
        # Сохраняем обновленную конфигурацию
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def set_data_source(self, source_name: str, source_config: Dict[str, Any]) -> None:
        """
        Добавляет или обновляет конфигурацию источника данных.
        
        Аргументы:
            source_name: Имя источника данных.
            source_config: Словарь с параметрами источника.
        """
        if 'sources' not in self.config['data']:
            self.config['data']['sources'] = {}
            
        self.config['data']['sources'][source_name] = source_config
        
        # Сохраняем обновленную конфигурацию
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def set_detection_criteria(self, criteria: Dict[str, Any]) -> None:
        """
        Обновляет критерии обнаружения циклонов.
        
        Аргументы:
            criteria: Словарь с критериями обнаружения.
        """
        self.config['detection']['criteria'] = criteria
        
        # Сохраняем обновленную конфигурацию
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


def load_user_config(config_path: Optional[Path] = None) -> ConfigManager:
    """
    Вспомогательная функция для загрузки пользовательской конфигурации.
    
    Аргументы:
        config_path: Путь к файлу конфигурации.
        
    Возвращает:
        Экземпляр ConfigManager с загруженной конфигурацией.
    """
    return ConfigManager(config_path)