"""
Модуль экспорта данных о циклонах в формат NetCDF.

Предоставляет функциональность для экспорта информации о циклонах,
их треках и характеристиках в файлы формата NetCDF.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import os

from models.cyclone import Cyclone
from core.exceptions import ExportError

# Инициализация логгера
logger = logging.getLogger(__name__)

class CycloneNetCDFExporter:
    """
    Экспортер данных о циклонах в формат NetCDF.
    
    Предоставляет методы для экспорта данных о циклонах,
    их треках и характеристиках в файлы NetCDF.
    """
    
    def __init__(self, compression_level: int = 4,
                format: str = 'NETCDF4'):
        """
        Инициализирует экспортер NetCDF.
        
        Аргументы:
            compression_level: Уровень сжатия данных (0-9).
            format: Формат NetCDF ('NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT', 'NETCDF3_CLASSIC').
        """
        self.compression_level = compression_level
        self.format = format
        
        logger.debug(f"Инициализирован экспортер NetCDF с параметрами: "
                    f"compression_level={compression_level}, format='{format}'")
    
    def export_to_netcdf(self, cyclones: Union[List[Cyclone], List[List[Cyclone]]],
                       filename: Union[str, Path],
                       metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Экспортирует данные о циклонах в файл NetCDF.
        
        Аргументы:
            cyclones: Список циклонов или список треков циклонов.
            filename: Путь к выходному файлу NetCDF.
            metadata: Дополнительные метаданные для включения в файл.
            
        Возвращает:
            Путь к созданному файлу.
            
        Вызывает:
            ExportError: При ошибке экспорта данных.
        """
        try:
            # Проверяем наличие зависимостей
            try:
                import xarray as xr
                import netCDF4
            except ImportError as e:
                raise ExportError(f"Для экспорта в NetCDF требуются библиотеки xarray и netCDF4: {str(e)}")
            
            # Конвертируем путь к Path
            file_path = Path(filename)
            
            # Создаем директорию, если не существует
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Подготавливаем данные в формате для xarray
            ds = self._prepare_dataset(cyclones, metadata)
            
            # Сохраняем в файл NetCDF
            encoding = {var: {'zlib': True, 'complevel': self.compression_level} 
                       for var in ds.data_vars}
            
            ds.to_netcdf(
                file_path,
                format=self.format,
                encoding=encoding
            )
            
            logger.info(f"Данные о циклонах успешно экспортированы в NetCDF: {file_path}")
            return file_path
            
        except Exception as e:
            error_msg = f"Ошибка при экспорте данных о циклонах в NetCDF: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)
    
    def _prepare_dataset(self, cyclones: Union[List[Cyclone], List[List[Cyclone]]],
                       metadata: Optional[Dict[str, Any]] = None) -> 'xr.Dataset':
        """
        Подготавливает данные о циклонах в формате xarray Dataset.
        
        Аргументы:
            cyclones: Список циклонов или список треков циклонов.
            metadata: Дополнительные метаданные для включения в файл.
            
        Возвращает:
            Объект xarray.Dataset.
        """
        import xarray as xr
        
        # Проверяем, является ли входной список списком треков или списком циклонов
        is_track_list = False
        if cyclones and isinstance(cyclones[0], list):
            is_track_list = True
        
        # Подготавливаем данные для треков
        if is_track_list:
            # Определяем максимальную длину трека
            max_track_length = max(len(track) for track in cyclones)
            
            # Создаем массивы для данных
            track_ids = []
            track_lengths = []
            track_data = {
                'time': np.full((len(cyclones), max_track_length), np.datetime64('NaT')),
                'latitude': np.full((len(cyclones), max_track_length), np.nan),
                'longitude': np.full((len(cyclones), max_track_length), np.nan),
                'pressure': np.full((len(cyclones), max_track_length), np.nan),
                'age': np.full((len(cyclones), max_track_length), np.nan),
                'vorticity': np.full((len(cyclones), max_track_length), np.nan),
                'wind_speed': np.full((len(cyclones), max_track_length), np.nan),
                'radius': np.full((len(cyclones), max_track_length), np.nan),
                'thermal_type': np.full((len(cyclones), max_track_length), '', dtype='U20')
            }
            
            # Заполняем массивы данными
            for track_idx, track in enumerate(cyclones):
                track_id = track[0].track_id if track else f"track_{track_idx}"
                track_ids.append(track_id)
                track_lengths.append(len(track))
                
                for point_idx, cyclone in enumerate(track):
                    # Заполняем основные параметры
                    track_data['time'][track_idx, point_idx] = np.datetime64(cyclone.time)
                    track_data['latitude'][track_idx, point_idx] = cyclone.latitude
                    track_data['longitude'][track_idx, point_idx] = cyclone.longitude
                    track_data['pressure'][track_idx, point_idx] = cyclone.central_pressure
                    track_data['age'][track_idx, point_idx] = cyclone.age
                    
                    # Заполняем дополнительные параметры, если доступны
                    if hasattr(cyclone, 'parameters'):
                        if hasattr(cyclone.parameters, 'vorticity_850hPa'):
                            track_data['vorticity'][track_idx, point_idx] = cyclone.parameters.vorticity_850hPa
                        
                        if hasattr(cyclone.parameters, 'max_wind_speed'):
                            track_data['wind_speed'][track_idx, point_idx] = cyclone.parameters.max_wind_speed
                        
                        if hasattr(cyclone.parameters, 'radius'):
                            track_data['radius'][track_idx, point_idx] = cyclone.parameters.radius
                        
                        if hasattr(cyclone.parameters, 'thermal_type'):
                            track_data['thermal_type'][track_idx, point_idx] = cyclone.parameters.thermal_type.value
            
            # Создаем координаты
            coords = {
                'track': np.arange(len(cyclones)),
                'track_id': ('track', track_ids),
                'track_length': ('track', track_lengths),
                'point': np.arange(max_track_length)
            }
            
            # Создаем переменные данных
            data_vars = {
                'time': xr.DataArray(
                    track_data['time'],
                    dims=('track', 'point'),
                    attrs={'units': 'seconds since 1970-01-01', 
                          'long_name': 'Time of observation'}
                ),
                'latitude': xr.DataArray(
                    track_data['latitude'],
                    dims=('track', 'point'),
                    attrs={'units': 'degrees_north', 
                          'long_name': 'Latitude', 
                          'standard_name': 'latitude'}
                ),
                'longitude': xr.DataArray(
                    track_data['longitude'],
                    dims=('track', 'point'),
                    attrs={'units': 'degrees_east', 
                          'long_name': 'Longitude', 
                          'standard_name': 'longitude'}
                ),
                'pressure': xr.DataArray(
                    track_data['pressure'],
                    dims=('track', 'point'),
                    attrs={'units': 'hPa', 
                          'long_name': 'Central pressure', 
                          'standard_name': 'air_pressure_at_sea_level'}
                ),
                'age': xr.DataArray(
                    track_data['age'],
                    dims=('track', 'point'),
                    attrs={'units': 'hours', 
                          'long_name': 'Cyclone age'}
                ),
                'vorticity': xr.DataArray(
                    track_data['vorticity'],
                    dims=('track', 'point'),
                    attrs={'units': 's^-1', 
                          'long_name': 'Relative vorticity at 850 hPa'}
                ),
                'wind_speed': xr.DataArray(
                    track_data['wind_speed'],
                    dims=('track', 'point'),
                    attrs={'units': 'm s^-1', 
                          'long_name': 'Maximum wind speed'}
                ),
                'radius': xr.DataArray(
                    track_data['radius'],
                    dims=('track', 'point'),
                    attrs={'units': 'km', 
                          'long_name': 'Cyclone radius'}
                ),
                'thermal_type': xr.DataArray(
                    track_data['thermal_type'],
                    dims=('track', 'point'),
                    attrs={'long_name': 'Thermal structure type'}
                )
            }
            
        else:
            # Обрабатываем список циклонов
            data = {
                'time': [],
                'latitude': [],
                'longitude': [],
                'pressure': [],
                'track_id': [],
                'age': [],
                'vorticity': [],
                'wind_speed': [],
                'radius': [],
                'thermal_type': []
            }
            
            for cyclone in cyclones:
                # Заполняем основные параметры
                data['time'].append(cyclone.time)
                data['latitude'].append(cyclone.latitude)
                data['longitude'].append(cyclone.longitude)
                data['pressure'].append(cyclone.central_pressure)
                data['track_id'].append(cyclone.track_id or "")
                data['age'].append(cyclone.age)
                
                # Заполняем дополнительные параметры, если доступны
                if hasattr(cyclone, 'parameters'):
                    data['vorticity'].append(getattr(cyclone.parameters, 'vorticity_850hPa', np.nan))
                    data['wind_speed'].append(getattr(cyclone.parameters, 'max_wind_speed', np.nan))
                    data['radius'].append(getattr(cyclone.parameters, 'radius', np.nan))
                    
                    if hasattr(cyclone.parameters, 'thermal_type'):
                        data['thermal_type'].append(cyclone.parameters.thermal_type.value)
                    else:
                        data['thermal_type'].append("")
                else:
                    data['vorticity'].append(np.nan)
                    data['wind_speed'].append(np.nan)
                    data['radius'].append(np.nan)
                    data['thermal_type'].append("")
            
            # Создаем координаты
            coords = {
                'cyclone': np.arange(len(cyclones))
            }
            
            # Создаем переменные данных
            data_vars = {
                'time': xr.DataArray(
                    np.array(data['time'], dtype='datetime64[ns]'),
                    dims='cyclone',
                    attrs={'units': 'seconds since 1970-01-01', 
                          'long_name': 'Time of observation'}
                ),
                'latitude': xr.DataArray(
                    np.array(data['latitude']),
                    dims='cyclone',
                    attrs={'units': 'degrees_north', 
                          'long_name': 'Latitude', 
                          'standard_name': 'latitude'}
                ),
                'longitude': xr.DataArray(
                    np.array(data['longitude']),
                    dims='cyclone',
                    attrs={'units': 'degrees_east', 
                          'long_name': 'Longitude', 
                          'standard_name': 'longitude'}
                ),
                'pressure': xr.DataArray(
                    np.array(data['pressure']),
                    dims='cyclone',
                    attrs={'units': 'hPa', 
                          'long_name': 'Central pressure', 
                          'standard_name': 'air_pressure_at_sea_level'}
                ),
                'track_id': xr.DataArray(
                    np.array(data['track_id']),
                    dims='cyclone',
                    attrs={'long_name': 'Track identifier'}
                ),
                'age': xr.DataArray(
                    np.array(data['age']),
                    dims='cyclone',
                    attrs={'units': 'hours', 
                          'long_name': 'Cyclone age'}
                ),
                'vorticity': xr.DataArray(
                    np.array(data['vorticity']),
                    dims='cyclone',
                    attrs={'units': 's^-1', 
                          'long_name': 'Relative vorticity at 850 hPa'}
                ),
                'wind_speed': xr.DataArray(
                    np.array(data['wind_speed']),
                    dims='cyclone',
                    attrs={'units': 'm s^-1', 
                          'long_name': 'Maximum wind speed'}
                ),
                'radius': xr.DataArray(
                    np.array(data['radius']),
                    dims='cyclone',
                    attrs={'units': 'km', 
                          'long_name': 'Cyclone radius'}
                ),
                'thermal_type': xr.DataArray(
                    np.array(data['thermal_type']),
                    dims='cyclone',
                    attrs={'long_name': 'Thermal structure type'}
                )
            }
        
        # Создаем Dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        
        # Добавляем глобальные атрибуты
        ds.attrs['title'] = 'Arctic Mesocyclone Data'
        ds.attrs['source'] = 'ArcticCyclone Detection System'
        ds.attrs['creation_date'] = datetime.now().isoformat()
        ds.attrs['Conventions'] = 'CF-1.8'
        
        # Добавляем пользовательские метаданные
        if metadata:
            for key, value in metadata.items():
                if key not in ds.attrs:
                    ds.attrs[key] = str(value)
        
        return ds
    
    def import_from_netcdf(self, filename: Union[str, Path]) -> List[List[Cyclone]]:
        """
        Импортирует треки циклонов из файла NetCDF.
        
        Аргументы:
            filename: Путь к файлу NetCDF.
            
        Возвращает:
            Список треков циклонов.
            
        Вызывает:
            ExportError: При ошибке импорта данных.
        """
        try:
            # Проверяем наличие зависимостей
            try:
                import xarray as xr
            except ImportError as e:
                raise ExportError(f"Для импорта из NetCDF требуется библиотека xarray: {str(e)}")
            
            # Открываем файл NetCDF
            ds = xr.open_dataset(filename)
            
            # Определяем формат данных (треки или отдельные циклоны)
            has_tracks = 'track' in ds.dims
            
            tracks = []
            
            if has_tracks:
                # Обрабатываем данные с треками
                for track_idx in range(ds.dims['track']):
                    track = []
                    track_data = ds.isel(track=track_idx)
                    track_length = int(track_data.track_length.values)
                    
                    for point_idx in range(track_length):
                        point_data = track_data.isel(point=point_idx)
                        
                        # Пропускаем точки с отсутствующими временами
                        if pd.isna(point_data.time) or point_data.time.values == np.datetime64('NaT'):
                            continue
                        
                        # Создаем объект циклона
                        from models.cyclone import Cyclone, CycloneParameters, CycloneType
                        
                        # Получаем базовые параметры
                        latitude = float(point_data.latitude.values)
                        longitude = float(point_data.longitude.values)
                        time = pd.Timestamp(point_data.time.values).to_pydatetime()
                        central_pressure = float(point_data.pressure.values)
                        
                        # Создаем циклон
                        cyclone = Cyclone(
                            latitude=latitude,
                            longitude=longitude,
                            time=time,
                            central_pressure=central_pressure,
                            dataset=None  # При импорте из NetCDF данные недоступны
                        )
                        
                        # Добавляем track_id
                        cyclone.track_id = str(track_data.track_id.values)
                        
                        # Добавляем возраст
                        cyclone.age = float(point_data.age.values) if not pd.isna(point_data.age) else 0
                        
                        # Создаем параметры циклона
                        cyclone.parameters = CycloneParameters(
                            central_pressure=central_pressure,
                            vorticity_850hPa=float(point_data.vorticity.values) if not pd.isna(point_data.vorticity) else None,
                            max_wind_speed=float(point_data.wind_speed.values) if not pd.isna(point_data.wind_speed) else None,
                            radius=float(point_data.radius.values) if not pd.isna(point_data.radius) else None
                        )
                        
                        # Добавляем термический тип
                        thermal_type_str = str(point_data.thermal_type.values)
                        if thermal_type_str and thermal_type_str != 'nan':
                            try:
                                cyclone.parameters.thermal_type = CycloneType(thermal_type_str)
                            except:
                                cyclone.parameters.thermal_type = CycloneType.UNCLASSIFIED
                        
                        track.append(cyclone)
                    
                    if track:
                        tracks.append(track)
            else:
                # Обрабатываем данные с отдельными циклонами
                # Группируем по track_id
                if 'track_id' in ds:
                    unique_tracks = np.unique(ds.track_id.values)
                    
                    for track_id in unique_tracks:
                        track = []
                        track_data = ds.where(ds.track_id == track_id, drop=True)
                        
                        # Сортируем по времени
                        track_data = track_data.sortby('time')
                        
                        for cyclone_idx in range(track_data.dims['cyclone']):
                            point_data = track_data.isel(cyclone=cyclone_idx)
                            
                            # Создаем объект циклона
                            from models.cyclone import Cyclone, CycloneParameters, CycloneType
                            
                            # Получаем базовые параметры
                            latitude = float(point_data.latitude.values)
                            longitude = float(point_data.longitude.values)
                            time = pd.Timestamp(point_data.time.values).to_pydatetime()
                            central_pressure = float(point_data.pressure.values)
                            
                            # Создаем циклон
                            cyclone = Cyclone(
                                latitude=latitude,
                                longitude=longitude,
                                time=time,
                                central_pressure=central_pressure,
                                dataset=None  # При импорте из NetCDF данные недоступны
                            )
                            
                            # Добавляем track_id
                            cyclone.track_id = str(track_id)
                            
                            # Добавляем возраст
                            cyclone.age = float(point_data.age.values) if not pd.isna(point_data.age) else 0
                            
                            # Создаем параметры циклона
                            cyclone.parameters = CycloneParameters(
                                central_pressure=central_pressure,
                                vorticity_850hPa=float(point_data.vorticity.values) if not pd.isna(point_data.vorticity) else None,
                                max_wind_speed=float(point_data.wind_speed.values) if not pd.isna(point_data.wind_speed) else None,
                                radius=float(point_data.radius.values) if not pd.isna(point_data.radius) else None
                            )
                            
                            # Добавляем термический тип
                            thermal_type_str = str(point_data.thermal_type.values)
                            if thermal_type_str and thermal_type_str != 'nan':
                                try:
                                    cyclone.parameters.thermal_type = CycloneType(thermal_type_str)
                                except:
                                    cyclone.parameters.thermal_type = CycloneType.UNCLASSIFIED
                            
                            track.append(cyclone)
                        
                        if track:
                            tracks.append(track)
                else:
                    # Если нет track_id, создаем отдельные треки для каждого циклона
                    for cyclone_idx in range(ds.dims['cyclone']):
                        point_data = ds.isel(cyclone=cyclone_idx)
                        
                        # Создаем объект циклона
                        from models.cyclone import Cyclone, CycloneParameters, CycloneType
                        
                        # Получаем базовые параметры
                        latitude = float(point_data.latitude.values)
                        longitude = float(point_data.longitude.values)
                        time = pd.Timestamp(point_data.time.values).to_pydatetime()
                        central_pressure = float(point_data.pressure.values)
                        
                        # Создаем циклон
                        cyclone = Cyclone(
                            latitude=latitude,
                            longitude=longitude,
                            time=time,
                            central_pressure=central_pressure,
                            dataset=None  # При импорте из NetCDF данные недоступны
                        )
                        
                        # Добавляем track_id
                        cyclone.track_id = f"track_{cyclone_idx}"
                        
                        # Добавляем возраст
                        cyclone.age = float(point_data.age.values) if not pd.isna(point_data.age) else 0
                        
                        # Создаем параметры циклона
                        cyclone.parameters = CycloneParameters(
                            central_pressure=central_pressure,
                            vorticity_850hPa=float(point_data.vorticity.values) if not pd.isna(point_data.vorticity) else None,
                            max_wind_speed=float(point_data.wind_speed.values) if not pd.isna(point_data.wind_speed) else None,
                            radius=float(point_data.radius.values) if not pd.isna(point_data.radius) else None
                        )
                        
                        # Добавляем термический тип
                        thermal_type_str = str(point_data.thermal_type.values)
                        if thermal_type_str and thermal_type_str != 'nan':
                            try:
                                cyclone.parameters.thermal_type = CycloneType(thermal_type_str)
                            except:
                                cyclone.parameters.thermal_type = CycloneType.UNCLASSIFIED
                        
                        tracks.append([cyclone])
            
            # Закрываем файл
            ds.close()
            
            logger.info(f"Успешно импортировано {len(tracks)} треков циклонов из NetCDF: {filename}")
            return tracks
            
        except Exception as e:
            error_msg = f"Ошибка при импорте треков циклонов из NetCDF: {str(e)}"
            logger.error(error_msg)
            raise ExportError(error_msg)