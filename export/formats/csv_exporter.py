"""
Модуль экспорта данных о циклонах в формат CSV.

Предоставляет функциональность для экспорта информации о циклонах,
их треках и характеристиках в файлы формата CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

from models.cyclone import Cyclone
from core.exceptions import ExportError

# Инициализация логгера
logger = logging.getLogger(__name__)

class CycloneCSVExporter:
    """
    Экспортер данных о циклонах в формат CSV.
    
    Предоставляет методы для экспорта данных о циклонах,
    их треках и статистике в файлы CSV.
    """
    
    def __init__(self, delimiter: str = ',', 
                encoding: str = 'utf-8',
                include_header: bool = True):
        """
        Инициализирует экспортер CSV.
        
        Аргументы:
            delimiter: Разделитель полей в CSV.
            encoding: Кодировка файла CSV.
            include_header: Включать ли заголовок с именами полей.
        """
        self.delimiter = delimiter
        self.encoding = encoding
        self.include_header = include_header
        
        logger.debug(f"Инициализирован экспортер CSV с параметрами: "
                    f"delimiter='{delimiter}', encoding='{encoding}', "
                    f"include_header={include_header}")
    
    def export_cyclone_tracks(self, cyclones: Union[List[Cyclone], List[List[Cyclone]]], 
                            filename: Union[str, Path]) -> Path:
        """
        Экспортирует треки циклонов в файл CSV.
        
        Аргументы:
            cyclones: Список циклонов или список треков циклонов.
            filename: Путь к выходному файлу CSV.
            
        Возвращает:
            Путь к созданному файлу.
            
        Вызывает:
            ExportError: При ошибке экспорта данных.
        """
        try:
            # Проверяем, является ли входной список списком треков или списком циклонов
            is_track_list = False
            if cyclones and isinstance(cyclones[0], list):
                is_track_list = True
            
            # Готовим данные для экспорта
            data = []
            
            if is_track_list:
                # Обрабатываем список треков
                for track_idx, track in enumerate(cyclones):
                    for cyclone in track:
                        row = self._cyclone_to_dict(cyclone, track_idx)
                        data.append(row)
            else:
                # Обрабатываем список циклонов
                for cyclone in cyclones:
                    row = self._cyclone_to_dict(cyclone)
                    data.append(row)
            
            # Создаем DataFrame и сохраняем в CSV
            if data:
                df = pd.DataFrame(data)
                
                # Конвертируем путь к Path
                file_path = Path(filename)
                
                # Создаем директорию, если не существует
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Сохраняем в CSV
                df.to_csv(
                    file_path,
                    sep=self.delimiter,
                    encoding=self.encoding,
                    index=False,
                    header=self.include_header
                )
                
                logger.info(f"Треки циклонов успешно экспортированы в CSV: {file_path}")
                return file_path
            else:
                logger.warning("Нет данных для экспорта в CSV")
                return Path(filename)
            
        except Exception as e:
            error_msg = f"Ошибка при экспорте треков циклонов в CSV: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def export_cyclone_statistics(self, cyclones: List[Cyclone],
                                filename: Union[str, Path]) -> Path:
        """
        Экспортирует статистику циклонов в файл CSV.
        
        Аргументы:
            cyclones: Список циклонов для экспорта статистики.
            filename: Путь к выходному файлу CSV.
            
        Возвращает:
            Путь к созданному файлу.
            
        Вызывает:
            ExportError: При ошибке экспорта данных.
        """
        try:
            # Готовим данные для экспорта
            data = []
            
            for cyclone in cyclones:
                # Получаем метрики жизненного цикла
                metrics = cyclone.calculate_lifecycle_metrics()
                
                # Создаем запись с статистикой
                row = {
                    'track_id': cyclone.track_id,
                    'genesis_time': cyclone.intensity_history[0][0] if cyclone.intensity_history else None,
                    'genesis_latitude': cyclone.track[0][0] if cyclone.track else None,
                    'genesis_longitude': cyclone.track[0][1] if cyclone.track else None,
                    'min_pressure': min(p for _, p in cyclone.intensity_history) if cyclone.intensity_history else None,
                    'lifespan_hours': metrics.get('lifespan_hours', 0),
                    'deepening_rate': metrics.get('deepening_rate', 0),
                    'displacement_km': metrics.get('displacement', 0),
                    'mean_speed_kmh': metrics.get('mean_speed', 0),
                    'thermal_type': cyclone.parameters.thermal_type.value if hasattr(cyclone.parameters, 'thermal_type') else None,
                }
                
                data.append(row)
            
            # Создаем DataFrame и сохраняем в CSV
            if data:
                df = pd.DataFrame(data)
                
                # Конвертируем путь к Path
                file_path = Path(filename)
                
                # Создаем директорию, если не существует
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Сохраняем в CSV
                df.to_csv(
                    file_path,
                    sep=self.delimiter,
                    encoding=self.encoding,
                    index=False,
                    header=self.include_header
                )
                
                logger.info(f"Статистика циклонов успешно экспортирована в CSV: {file_path}")
                return file_path
            else:
                logger.warning("Нет данных для экспорта статистики в CSV")
                return Path(filename)
            
        except Exception as e:
            error_msg = f"Ошибка при экспорте статистики циклонов в CSV: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def _cyclone_to_dict(self, cyclone: Cyclone, track_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Преобразует объект циклона в словарь для экспорта.
        
        Аргументы:
            cyclone: Объект циклона для преобразования.
            track_idx: Индекс трека (если применимо).
            
        Возвращает:
            Словарь с характеристиками циклона.
        """
        # Базовые характеристики циклона
        result = {
            'track_id': cyclone.track_id or (f"track_{track_idx}" if track_idx is not None else None),
            'time': cyclone.time,
            'latitude': cyclone.latitude,
            'longitude': cyclone.longitude,
            'central_pressure': cyclone.central_pressure,
            'age_hours': cyclone.age,
        }
        
        # Добавляем параметры, если доступны
        if hasattr(cyclone, 'parameters'):
            if hasattr(cyclone.parameters, 'vorticity_850hPa'):
                result['vorticity_850hPa'] = cyclone.parameters.vorticity_850hPa
            
            if hasattr(cyclone.parameters, 'max_wind_speed'):
                result['max_wind_speed'] = cyclone.parameters.max_wind_speed
            
            if hasattr(cyclone.parameters, 'radius'):
                result['radius_km'] = cyclone.parameters.radius
            
            if hasattr(cyclone.parameters, 'thermal_type'):
                result['thermal_type'] = cyclone.parameters.thermal_type.value
            
            if hasattr(cyclone.parameters, 'temperature_anomaly'):
                result['temperature_anomaly'] = cyclone.parameters.temperature_anomaly
        
        # Добавляем другие атрибуты, если есть
        for attr in dir(cyclone):
            if (not attr.startswith('_') and
                attr not in result and 
                attr not in ['track', 'intensity_history', 'parameters', 'update'] and
                not callable(getattr(cyclone, attr))):
                
                value = getattr(cyclone, attr)
                # Пропускаем сложные объекты
                if isinstance(value, (int, float, str, bool)) or value is None:
                    result[attr] = value
        
        return result
    
    def import_from_csv(self, filename: Union[str, Path]) -> List[List[Cyclone]]:
        """
        Импортирует треки циклонов из файла CSV.
        
        Аргументы:
            filename: Путь к файлу CSV.
            
        Возвращает:
            Список треков циклонов.
            
        Вызывает:
            ExportError: При ошибке импорта данных.
        """
        try:
            # Загружаем CSV в DataFrame
            df = pd.read_csv(
                filename,
                sep=self.delimiter,
                encoding=self.encoding
            )
            
            # Группируем данные по track_id
            track_groups = df.groupby('track_id')
            
            # Создаем треки циклонов
            tracks = []
            
            for track_id, group in track_groups:
                track = []
                
                # Сортируем группу по времени
                group = group.sort_values('time')
                
                for _, row in group.iterrows():
                    # Создаем объект циклона из строки CSV
                    from models.cyclone import Cyclone, CycloneParameters, CycloneType
                    
                    # Преобразуем время в datetime
                    time = pd.to_datetime(row['time'])
                    
                    # Создаем базовый объект циклона
                    cyclone = Cyclone(
                        latitude=row['latitude'],
                        longitude=row['longitude'],
                        time=time,
                        central_pressure=row['central_pressure'],
                        dataset=None  # При импорте из CSV данные недоступны
                    )
                    
                    # Устанавливаем track_id
                    cyclone.track_id = str(track_id)
                    
                    # Устанавливаем возраст
                    if 'age_hours' in row:
                        cyclone.age = row['age_hours']
                    
                    # Создаем объект параметров
                    params = {}
                    
                    if 'vorticity_850hPa' in row:
                        params['vorticity_850hPa'] = row['vorticity_850hPa']
                    
                    if 'max_wind_speed' in row:
                        params['max_wind_speed'] = row['max_wind_speed']
                    
                    if 'radius_km' in row:
                        params['radius'] = row['radius_km']
                    
                    if 'thermal_type' in row:
                        thermal_type_str = row['thermal_type']
                        if thermal_type_str:
                            try:
                                params['thermal_type'] = CycloneType(thermal_type_str)
                            except:
                                params['thermal_type'] = CycloneType.UNCLASSIFIED
                    
                    if 'temperature_anomaly' in row:
                        params['temperature_anomaly'] = row['temperature_anomaly']
                    
                    # Создаем параметры циклона
                    cyclone.parameters = CycloneParameters(**params)
                    
                    # Добавляем дополнительные атрибуты
                    for col in row.index:
                        if col not in ['track_id', 'time', 'latitude', 'longitude', 
                                      'central_pressure', 'age_hours', 'vorticity_850hPa', 
                                      'max_wind_speed', 'radius_km', 'thermal_type', 
                                      'temperature_anomaly']:
                            setattr(cyclone, col, row[col])
                    
                    track.append(cyclone)
                
                tracks.append(track)
            
            logger.info(f"Успешно импортировано {len(tracks)} треков циклонов из CSV: {filename}")
            return tracks
            
        except Exception as e:
            error_msg = f"Ошибка при импорте треков циклонов из CSV: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)