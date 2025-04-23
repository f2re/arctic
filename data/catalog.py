"""
Модуль каталогизации данных для системы ArcticCyclone.

Обеспечивает систематизацию и поиск наборов метеорологических данных.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging
import xarray as xr

# Инициализация логгера
logger = logging.getLogger(__name__)

class DatasetEntry:
    """
    Запись о наборе данных в каталоге.
    
    Содержит метаданные о наборе данных, включая источник, временной период,
    географический регион, переменные и другие атрибуты.
    """
    
    def __init__(self, 
                dataset_id: str,
                source: str,
                path: Path,
                variables: List[str],
                region: Dict[str, float],
                time_range: Dict[str, str],
                levels: Optional[List[int]] = None,
                attributes: Optional[Dict[str, Any]] = None,
                created_at: Optional[datetime] = None,
                updated_at: Optional[datetime] = None,
                description: Optional[str] = None):
        """
        Инициализирует запись о наборе данных.
        
        Аргументы:
            dataset_id: Уникальный идентификатор набора данных.
            source: Источник данных (например, 'ERA5').
            path: Путь к файлу с данными.
            variables: Список переменных в наборе данных.
            region: Географические границы региона.
            time_range: Временной диапазон данных (начало и конец).
            levels: Список уровней давления, если применимо.
            attributes: Дополнительные атрибуты набора данных.
            created_at: Дата и время создания записи.
            updated_at: Дата и время последнего обновления записи.
            description: Текстовое описание набора данных.
        """
        self.dataset_id = dataset_id
        self.source = source
        self.path = Path(path)
        self.variables = variables
        self.region = region
        self.time_range = time_range
        self.levels = levels or []
        self.attributes = attributes or {}
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.description = description or ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует запись в словарь для сериализации.
        
        Возвращает:
            Словарь с атрибутами записи.
        """
        return {
            "dataset_id": self.dataset_id,
            "source": self.source,
            "path": str(self.path),
            "variables": self.variables,
            "region": self.region,
            "time_range": self.time_range,
            "levels": self.levels,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetEntry':
        """
        Создает запись из словаря.
        
        Аргументы:
            data: Словарь с атрибутами записи.
            
        Возвращает:
            Экземпляр DatasetEntry.
        """
        return cls(
            dataset_id=data["dataset_id"],
            source=data["source"],
            path=Path(data["path"]),
            variables=data["variables"],
            region=data["region"],
            time_range=data["time_range"],
            levels=data.get("levels"),
            attributes=data.get("attributes"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            description=data.get("description", "")
        )
    
    def open_dataset(self) -> xr.Dataset:
        """
        Открывает набор данных, связанный с записью.
        
        Возвращает:
            Набор данных xarray.
            
        Вызывает:
            FileNotFoundError: Если файл с данными не найден.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Файл набора данных не найден: {self.path}")
            
        return xr.open_dataset(self.path)
    
    def update_from_dataset(self, dataset: xr.Dataset) -> None:
        """
        Обновляет метаданные записи из набора данных.
        
        Аргументы:
            dataset: Набор данных xarray.
        """
        # Обновляем временной диапазон
        time_min = dataset.time.min().values
        time_max = dataset.time.max().values
        self.time_range = {
            "start": str(time_min),
            "end": str(time_max)
        }
        
        # Обновляем список переменных
        self.variables = list(dataset.data_vars)
        
        # Обновляем уровни давления, если применимо
        if 'level' in dataset.dims:
            self.levels = sorted(dataset.level.values.tolist())
        
        # Обновляем регион
        self.region = {
            "north": float(dataset.latitude.max().values),
            "south": float(dataset.latitude.min().values),
            "east": float(dataset.longitude.max().values),
            "west": float(dataset.longitude.min().values)
        }
        
        # Обновляем атрибуты
        self.attributes = dict(dataset.attrs)
        
        # Обновляем время изменения
        self.updated_at = datetime.now()


class DataCatalog:
    """
    Каталог наборов метеорологических данных.
    
    Обеспечивает управление и поиск наборов данных по различным критериям.
    """
    
    def __init__(self, catalog_file: Optional[Path] = None, data_dir: Optional[Path] = None):
        """
        Инициализирует каталог данных.
        
        Аргументы:
            catalog_file: Путь к файлу каталога. Если None, используется 
                         ~/.config/arctic_cyclone/catalog.json.
            data_dir: Директория для хранения данных. Если None, используется
                     ./data/catalog.
        """
        self.catalog_file = catalog_file or Path.home() / ".config" / "arctic_cyclone" / "catalog.json"
        self.data_dir = data_dir or Path("data/catalog")
        self.entries = {}
        
        # Создаем директории, если они не существуют
        os.makedirs(self.catalog_file.parent, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Загружаем каталог, если он существует
        self._load()
        
    def _load(self) -> None:
        """
        Загружает каталог из файла.
        """
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, 'r') as f:
                    data = json.load(f)
                    
                self.entries = {
                    entry_id: DatasetEntry.from_dict(entry_data)
                    for entry_id, entry_data in data.items()
                }
                
                logger.info(f"Загружен каталог с {len(self.entries)} записями из {self.catalog_file}")
                
            except Exception as e:
                logger.error(f"Ошибка при загрузке каталога: {str(e)}")
                self.entries = {}
        else:
            logger.info(f"Файл каталога {self.catalog_file} не найден, создан пустой каталог")
            self.entries = {}
    
    def _save(self) -> None:
        """
        Сохраняет каталог в файл.
        """
        try:
            # Преобразуем записи в словари
            data = {
                entry_id: entry.to_dict()
                for entry_id, entry in self.entries.items()
            }
            
            # Сохраняем в файл
            with open(self.catalog_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Каталог с {len(self.entries)} записями сохранен в {self.catalog_file}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении каталога: {str(e)}")
    
    def add(self, entry: DatasetEntry) -> None:
        """
        Добавляет запись в каталог.
        
        Аргументы:
            entry: Запись о наборе данных.
        """
        self.entries[entry.dataset_id] = entry
        self._save()
        logger.info(f"Добавлена запись {entry.dataset_id} в каталог")
    
    def get(self, dataset_id: str) -> Optional[DatasetEntry]:
        """
        Получает запись по идентификатору.
        
        Аргументы:
            dataset_id: Идентификатор набора данных.
            
        Возвращает:
            Запись о наборе данных или None, если запись не найдена.
        """
        return self.entries.get(dataset_id)
    
    def update(self, dataset_id: str, **kwargs) -> bool:
        """
        Обновляет запись в каталоге.
        
        Аргументы:
            dataset_id: Идентификатор набора данных.
            **kwargs: Атрибуты для обновления.
            
        Возвращает:
            True, если запись обновлена успешно, иначе False.
        """
        if dataset_id not in self.entries:
            logger.warning(f"Невозможно обновить запись {dataset_id}: не найдена в каталоге")
            return False
            
        entry = self.entries[dataset_id]
        
        # Обновляем атрибуты
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
                
        # Обновляем время изменения
        entry.updated_at = datetime.now()
        
        self._save()
        logger.info(f"Обновлена запись {dataset_id} в каталоге")
        return True
    
    def remove(self, dataset_id: str, delete_file: bool = False) -> bool:
        """
        Удаляет запись из каталога.
        
        Аргументы:
            dataset_id: Идентификатор набора данных.
            delete_file: Удалить ли файл с данными.
            
        Возвращает:
            True, если запись удалена успешно, иначе False.
        """
        if dataset_id not in self.entries:
            logger.warning(f"Невозможно удалить запись {dataset_id}: не найдена в каталоге")
            return False
            
        entry = self.entries[dataset_id]
        
        # Удаляем файл с данными, если требуется
        if delete_file and entry.path.exists():
            try:
                entry.path.unlink()
                logger.info(f"Удален файл набора данных: {entry.path}")
            except Exception as e:
                logger.error(f"Ошибка при удалении файла {entry.path}: {str(e)}")
        
        # Удаляем запись из каталога
        del self.entries[dataset_id]
        self._save()
        
        logger.info(f"Удалена запись {dataset_id} из каталога")
        return True
    
    def search(self, 
              source: Optional[str] = None,
              variables: Optional[List[str]] = None,
              region: Optional[Dict[str, float]] = None,
              time_range: Optional[Dict[str, str]] = None,
              levels: Optional[List[int]] = None) -> List[DatasetEntry]:
        """
        Ищет записи в каталоге по заданным критериям.
        
        Аргументы:
            source: Источник данных.
            variables: Список требуемых переменных.
            region: Географический регион.
            time_range: Временной диапазон.
            levels: Список уровней давления.
            
        Возвращает:
            Список записей, соответствующих критериям.
        """
        results = []
        
        for entry in self.entries.values():
            # Фильтр по источнику
            if source and entry.source != source:
                continue
                
            # Фильтр по переменным
            if variables and not all(var in entry.variables for var in variables):
                continue
                
            # Фильтр по региону
            if region:
                if not (
                    entry.region["north"] >= region.get("north", -90) and
                    entry.region["south"] <= region.get("south", 90) and
                    entry.region["east"] >= region.get("east", -180) and
                    entry.region["west"] <= region.get("west", 180)
                ):
                    continue
                    
            # Фильтр по временному диапазону
            if time_range:
                entry_start = datetime.fromisoformat(entry.time_range["start"])
                entry_end = datetime.fromisoformat(entry.time_range["end"])
                
                if time_range.get("start"):
                    request_start = datetime.fromisoformat(time_range["start"])
                    if entry_start > request_start:
                        continue
                        
                if time_range.get("end"):
                    request_end = datetime.fromisoformat(time_range["end"])
                    if entry_end < request_end:
                        continue
            
            # Фильтр по уровням
            if levels and not all(level in entry.levels for level in levels):
                continue
                
            # Если прошли все фильтры, добавляем запись в результаты
            results.append(entry)
            
        return results
    
    def register_dataset(self, 
                        dataset: xr.Dataset,
                        dataset_id: str,
                        source: str,
                        path: Optional[Path] = None,
                        description: str = "") -> DatasetEntry:
        """
        Регистрирует набор данных в каталоге.
        
        Аргументы:
            dataset: Набор данных xarray.
            dataset_id: Идентификатор набора данных.
            source: Источник данных.
            path: Путь для сохранения набора данных. Если None, генерируется автоматически.
            description: Описание набора данных.
            
        Возвращает:
            Созданную запись о наборе данных.
        """
        # Генерируем путь, если не указан
        if path is None:
            path = self.data_dir / f"{dataset_id}.nc"
            
        # Сохраняем набор данных
        dataset.to_netcdf(path)
        logger.info(f"Набор данных сохранен в {path}")
        
        # Извлекаем метаданные из набора данных
        variables = list(dataset.data_vars)
        
        # Определяем регион
        region = {
            "north": float(dataset.latitude.max().values),
            "south": float(dataset.latitude.min().values),
            "east": float(dataset.longitude.max().values),
            "west": float(dataset.longitude.min().values)
        }
        
        # Определяем временной диапазон
        time_range = {
            "start": str(dataset.time.min().values),
            "end": str(dataset.time.max().values)
        }
        
        # Определяем уровни давления, если применимо
        levels = []
        if 'level' in dataset.dims:
            levels = sorted(dataset.level.values.tolist())
        
        # Создаем запись
        entry = DatasetEntry(
            dataset_id=dataset_id,
            source=source,
            path=path,
            variables=variables,
            region=region,
            time_range=time_range,
            levels=levels,
            attributes=dict(dataset.attrs),
            description=description
        )
        
        # Добавляем запись в каталог
        self.add(entry)
        
        return entry
    
    def scan_directory(self, directory: Path, recursive: bool = False) -> int:
        """
        Сканирует директорию на наличие NetCDF файлов и добавляет их в каталог.
        
        Аргументы:
            directory: Директория для сканирования.
            recursive: Сканировать ли вложенные директории.
            
        Возвращает:
            Количество добавленных записей.
        """
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Директория {directory} не существует или не является директорией")
            return 0
            
        # Функция для поиска файлов
        def find_nc_files(path, rec=False):
            files = []
            for item in path.iterdir():
                if item.is_file() and item.suffix.lower() in ['.nc', '.netcdf']:
                    files.append(item)
                elif rec and item.is_dir():
                    files.extend(find_nc_files(item, True))
            return files
            
        # Находим все NetCDF файлы
        nc_files = find_nc_files(directory, recursive)
        logger.info(f"Найдено {len(nc_files)} NetCDF файлов в {directory}")
        
        # Регистрируем файлы в каталоге
        added_count = 0
        
        for file_path in nc_files:
            try:
                # Открываем набор данных
                dataset = xr.open_dataset(file_path)
                
                # Генерируем идентификатор
                dataset_id = f"{file_path.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Определяем источник из атрибутов или имени файла
                source = dataset.attrs.get('source', file_path.stem.split('_')[0])
                
                # Регистрируем набор данных
                self.register_dataset(
                    dataset=dataset,
                    dataset_id=dataset_id,
                    source=source,
                    path=file_path,
                    description=f"Автоматически добавлено при сканировании {directory}"
                )
                
                added_count += 1
                logger.info(f"Добавлен набор данных {dataset_id} из файла {file_path}")
                
            except Exception as e:
                logger.error(f"Ошибка при регистрации файла {file_path}: {str(e)}")
                
        return added_count