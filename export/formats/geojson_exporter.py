"""
Модуль экспорта данных о циклонах в формат GeoJSON.

Предоставляет функциональность для экспорта информации о циклонах,
их треках и характеристиках в файлы формата GeoJSON.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime

from models.cyclone import Cyclone
from core.exceptions import ExportError

# Инициализация логгера
logger = logging.getLogger(__name__)

class CycloneGeoJSONExporter:
    """
    Экспортер данных о циклонах в формат GeoJSON.
    
    Предоставляет методы для экспорта данных о циклонах и их треках
    в формат GeoJSON для использования в геоинформационных системах.
    """
    
    def __init__(self, indent: Optional[int] = 2,
                simplify_tracks: bool = False,
                simplify_tolerance: float = 0.01):
        """
        Инициализирует экспортер GeoJSON.
        
        Аргументы:
            indent: Отступ для форматирования JSON. None для минимизации.
            simplify_tracks: Упрощать ли геометрию треков для уменьшения размера файла.
            simplify_tolerance: Допуск при упрощении геометрии (в градусах).
        """
        self.indent = indent
        self.simplify_tracks = simplify_tracks
        self.simplify_tolerance = simplify_tolerance
        
        logger.debug(f"Инициализирован экспортер GeoJSON с параметрами: "
                    f"indent={indent}, simplify_tracks={simplify_tracks}, "
                    f"simplify_tolerance={simplify_tolerance}")
    
    def export_to_geojson(self, cyclones: Union[List[Cyclone], List[List[Cyclone]]],
                        filename: Union[str, Path],
                        metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Экспортирует данные о циклонах в файл GeoJSON.
        
        Аргументы:
            cyclones: Список циклонов или список треков циклонов.
            filename: Путь к выходному файлу GeoJSON.
            metadata: Дополнительные метаданные для включения в файл.
            
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
            
            # Подготавливаем GeoJSON
            geojson = self._prepare_geojson(cyclones, is_track_list, metadata)
            
            # Конвертируем путь к Path
            file_path = Path(filename)
            
            # Создаем директорию, если не существует
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем в файл
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=self.indent, ensure_ascii=False)
            
            logger.info(f"Данные о циклонах успешно экспортированы в GeoJSON: {file_path}")
            return file_path
            
        except Exception as e:
            error_msg = f"Ошибка при экспорте данных о циклонах в GeoJSON: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def _prepare_geojson(self, cyclones: Union[List[Cyclone], List[List[Cyclone]]],
                       is_track_list: bool,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Подготавливает данные о циклонах в формате GeoJSON.
        
        Аргументы:
            cyclones: Список циклонов или список треков циклонов.
            is_track_list: Флаг, указывающий, что входные данные - список треков.
            metadata: Дополнительные метаданные для включения в файл.
            
        Возвращает:
            Словарь в формате GeoJSON.
        """
        features = []
        
        if is_track_list:
            # Обрабатываем треки циклонов
            for track_idx, track in enumerate(cyclones):
                if not track:
                    continue
                
                track_id = track[0].track_id or f"track_{track_idx}"
                
                # Получаем координаты трека
                coordinates = [(c.longitude, c.latitude) for c in track]
                
                # Упрощаем геометрию, если требуется
                if self.simplify_tracks and len(coordinates) > 3:
                    coordinates = self._simplify_line(coordinates)
                
                # Свойства трека
                properties = {
                    'track_id': track_id,
                    'points_count': len(track),
                    'genesis_time': self._format_datetime(track[0].time),
                    'lysis_time': self._format_datetime(track[-1].time),
                }
                
                # Добавляем метрики жизненного цикла, если первый циклон имеет соответствующий метод
                if hasattr(track[0], 'calculate_lifecycle_metrics'):
                    metrics = track[0].calculate_lifecycle_metrics()
                    properties.update(metrics)
                
                # Добавляем информацию о типе циклона
                if hasattr(track[0], 'parameters') and hasattr(track[0].parameters, 'thermal_type'):
                    properties['thermal_type'] = track[0].parameters.thermal_type.value
                
                # Создаем объект Feature для трека
                track_feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': coordinates
                    },
                    'properties': properties
                }
                
                features.append(track_feature)
                
                # Добавляем точки трека как отдельные Features
                for point_idx, cyclone in enumerate(track):
                    point_properties = {
                        'track_id': track_id,
                        'point_idx': point_idx,
                        'time': self._format_datetime(cyclone.time),
                        'pressure': cyclone.central_pressure,
                        'age_hours': cyclone.age,
                    }
                    
                    # Добавляем параметры циклона, если доступны
                    if hasattr(cyclone, 'parameters'):
                        if hasattr(cyclone.parameters, 'vorticity_850hPa'):
                            point_properties['vorticity_850hPa'] = cyclone.parameters.vorticity_850hPa
                        
                        if hasattr(cyclone.parameters, 'max_wind_speed'):
                            point_properties['max_wind_speed'] = cyclone.parameters.max_wind_speed
                        
                        if hasattr(cyclone.parameters, 'radius'):
                            point_properties['radius_km'] = cyclone.parameters.radius
                        
                        if hasattr(cyclone.parameters, 'thermal_type'):
                            point_properties['thermal_type'] = cyclone.parameters.thermal_type.value
                    
                    # Создаем объект Feature для точки
                    point_feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [cyclone.longitude, cyclone.latitude]
                        },
                        'properties': point_properties
                    }
                    
                    features.append(point_feature)
        else:
            # Обрабатываем список циклонов
            for cyclone_idx, cyclone in enumerate(cyclones):
                properties = {
                    'id': cyclone_idx,
                    'track_id': cyclone.track_id or f"track_{cyclone_idx}",
                    'time': self._format_datetime(cyclone.time),
                    'pressure': cyclone.central_pressure,
                    'age_hours': cyclone.age,
                }
                
                # Добавляем параметры циклона, если доступны
                if hasattr(cyclone, 'parameters'):
                    if hasattr(cyclone.parameters, 'vorticity_850hPa'):
                        properties['vorticity_850hPa'] = cyclone.parameters.vorticity_850hPa
                    
                    if hasattr(cyclone.parameters, 'max_wind_speed'):
                        properties['max_wind_speed'] = cyclone.parameters.max_wind_speed
                    
                    if hasattr(cyclone.parameters, 'radius'):
                        properties['radius_km'] = cyclone.parameters.radius
                    
                    if hasattr(cyclone.parameters, 'thermal_type'):
                        properties['thermal_type'] = cyclone.parameters.thermal_type.value
                
                # Создаем объект Feature для циклона
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [cyclone.longitude, cyclone.latitude]
                    },
                    'properties': properties
                }
                
                features.append(feature)
        
        # Создаем корневой объект GeoJSON
        geojson = {
            'type': 'FeatureCollection',
            'features': features,
        }
        
        # Добавляем метаданные
        if metadata:
            geojson['metadata'] = metadata
        
        # Добавляем стандартные метаданные
        if 'metadata' not in geojson:
            geojson['metadata'] = {}
        
        if 'created_at' not in geojson['metadata']:
            geojson['metadata']['created_at'] = datetime.now().isoformat()
        
        if 'generator' not in geojson['metadata']:
            geojson['metadata']['generator'] = 'ArcticCyclone GeoJSON Exporter'
        
        return geojson
    
    def _format_datetime(self, dt) -> str:
        """
        Форматирует объект datetime в строку ISO 8601.
        
        Аргументы:
            dt: Объект datetime для форматирования.
            
        Возвращает:
            Строка в формате ISO 8601.
        """
        if isinstance(dt, pd.Timestamp):
            return dt.isoformat()
        elif isinstance(dt, datetime):
            return dt.isoformat()
        else:
            return str(dt)
    
    def _simplify_line(self, coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Упрощает линию с использованием алгоритма Дугласа-Пойкера.
        
        Аргументы:
            coordinates: Список координат линии [(lon1, lat1), (lon2, lat2), ...].
            
        Возвращает:
            Упрощенный список координат.
        """
        try:
            from shapely.geometry import LineString
            from shapely.simplify import simplify
            
            # Создаем линию из координат
            line = LineString(coordinates)
            
            # Упрощаем линию
            simplified = simplify(line, tolerance=self.simplify_tolerance)
            
            # Возвращаем координаты упрощенной линии
            return list(simplified.coords)
            
        except ImportError:
            logger.warning("Библиотека shapely не установлена, упрощение геометрии недоступно")
            return coordinates
    
    def import_from_geojson(self, filename: Union[str, Path]) -> List[List[Cyclone]]:
        """
        Импортирует треки циклонов из файла GeoJSON.
        
        Аргументы:
            filename: Путь к файлу GeoJSON.
            
        Возвращает:
            Список треков циклонов.
            
        Вызывает:
            ExportError: При ошибке импорта данных.
        """
        try:
            # Открываем файл GeoJSON
            with open(filename, 'r', encoding='utf-8') as f:
                geojson = json.load(f)
            
            # Проверяем, что это FeatureCollection
            if geojson.get('type') != 'FeatureCollection':
                raise ExportError("Файл GeoJSON не содержит FeatureCollection")
            
            # Получаем список фич
            features = geojson.get('features', [])
            
            # Словарь для хранения информации о треках
            tracks_dict = {}
            
            # Сначала собираем все точки
            for feature in features:
                geometry_type = feature.get('geometry', {}).get('type')
                properties = feature.get('properties', {})
                
                # Пропускаем треки (LineString), обрабатываем только точки
                if geometry_type == 'Point' and 'track_id' in properties:
                    track_id = properties['track_id']
                    
                    if track_id not in tracks_dict:
                        tracks_dict[track_id] = []
                    
                    # Получаем координаты
                    lon, lat = feature['geometry']['coordinates']
                    
                    # Получаем время
                    time_str = properties.get('time')
                    if time_str:
                        try:
                            time = pd.to_datetime(time_str)
                        except:
                            time = datetime.now()  # Заглушка, если не удалось распарсить время
                    else:
                        time = datetime.now()  # Заглушка, если нет времени
                    
                    # Получаем давление
                    pressure = properties.get('pressure', 1000.0)
                    
                    # Создаем объект циклона
                    from models.cyclone import Cyclone, CycloneParameters, CycloneType
                    
                    cyclone = Cyclone(
                        latitude=lat,
                        longitude=lon,
                        time=time,
                        central_pressure=pressure,
                        dataset=None  # При импорте из GeoJSON данные недоступны
                    )
                    
                    # Устанавливаем track_id
                    cyclone.track_id = track_id
                    
                    # Устанавливаем возраст
                    if 'age_hours' in properties:
                        cyclone.age = properties['age_hours']
                    
                    # Создаем объект параметров
                    params = {
                        'central_pressure': pressure
                    }
                    
                    if 'vorticity_850hPa' in properties:
                        params['vorticity_850hPa'] = properties['vorticity_850hPa']
                    
                    if 'max_wind_speed' in properties:
                        params['max_wind_speed'] = properties['max_wind_speed']
                    
                    if 'radius_km' in properties:
                        params['radius'] = properties['radius_km']
                    
                    if 'thermal_type' in properties:
                        thermal_type_str = properties['thermal_type']
                        if thermal_type_str:
                            try:
                                params['thermal_type'] = CycloneType(thermal_type_str)
                            except:
                                params['thermal_type'] = CycloneType.UNCLASSIFIED
                    
                    # Создаем параметры циклона
                    cyclone.parameters = CycloneParameters(**params)
                    
                    # Добавляем дополнительные атрибуты
                    for key, value in properties.items():
                        if key not in ['track_id', 'time', 'pressure', 'age_hours', 
                                      'vorticity_850hPa', 'max_wind_speed', 'radius_km', 
                                      'thermal_type']:
                            setattr(cyclone, key, value)
                    
                    # Добавляем точку в трек
                    point_idx = properties.get('point_idx')
                    if point_idx is not None:
                        # Если есть индекс точки, вставляем в нужную позицию
                        while len(tracks_dict[track_id]) <= point_idx:
                            tracks_dict[track_id].append(None)
                        tracks_dict[track_id][point_idx] = cyclone
                    else:
                        # Иначе просто добавляем в конец
                        tracks_dict[track_id].append(cyclone)
            
            # Преобразуем словарь треков в список, удаляя пустые значения
            tracks = []
            for track_id, track in tracks_dict.items():
                # Удаляем None из трека
                track = [c for c in track if c is not None]
                
                if track:
                    # Сортируем по времени
                    track.sort(key=lambda c: c.time)
                    tracks.append(track)
            
            logger.info(f"Успешно импортировано {len(tracks)} треков циклонов из GeoJSON: {filename}")
            return tracks
            
        except Exception as e:
            error_msg = f"Ошибка при импорте треков циклонов из GeoJSON: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)