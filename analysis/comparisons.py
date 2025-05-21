"""
Модуль сравнительного анализа для системы ArcticCyclone.

Предоставляет функции и классы для сравнения различных циклонов,
их характеристик и различных наборов данных о циклонах.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.spatial.distance import cdist
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import logging
from pathlib import Path
from datetime import datetime, timedelta
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.cyclone import Cyclone
from models.classifications import CycloneType, CycloneIntensity
from core.exceptions import ArcticCycloneError

# Инициализация логгера
logger = logging.getLogger(__name__)


class CycloneComparator:
    """
    Класс для сравнения циклонов и их характеристик.
    
    Предоставляет методы для сравнения отдельных циклонов, их треков
    и целых наборов данных о циклонах.
    """
    
    def __init__(self, default_distance_threshold: float = 500.0):
        """
        Инициализирует компаратор циклонов.
        
        Аргументы:
            default_distance_threshold: Пороговое значение расстояния (км) для сопоставления циклонов.
        """
        self.default_distance_threshold = default_distance_threshold
        
        logger.debug(f"Инициализирован CycloneComparator с порогом расстояния {default_distance_threshold} км")
    
    def compare_cyclones(self, cyclone1: Cyclone, cyclone2: Cyclone) -> Dict[str, Any]:
        """
        Сравнивает два циклона по их характеристикам.
        
        Аргументы:
            cyclone1: Первый циклон для сравнения.
            cyclone2: Второй циклон для сравнения.
            
        Возвращает:
            Словарь с результатами сравнения.
        """
        try:
            # Рассчитываем расстояние между циклонами
            distance = self._calculate_distance(
                cyclone1.latitude, cyclone1.longitude,
                cyclone2.latitude, cyclone2.longitude
            )
            
            # Рассчитываем временную разницу
            time_diff = abs((cyclone1.time - cyclone2.time).total_seconds()) / 3600  # в часах
            
            # Рассчитываем разницу в давлении
            pressure_diff = abs(cyclone1.central_pressure - cyclone2.central_pressure)
            
            # Сравниваем параметры циклонов
            param_diffs = {}
            params_to_compare = [
                'vorticity_850hPa', 'max_wind_speed', 'radius', 'pressure_gradient',
                'temperature_anomaly', 'thickness_500_850'
            ]
            
            for param in params_to_compare:
                # Проверяем наличие параметра у обоих циклонов
                if (hasattr(cyclone1.parameters, param) and
                    hasattr(cyclone2.parameters, param) and
                    getattr(cyclone1.parameters, param) is not None and
                    getattr(cyclone2.parameters, param) is not None):
                    
                    value1 = getattr(cyclone1.parameters, param)
                    value2 = getattr(cyclone2.parameters, param)
                    
                    # Рассчитываем абсолютную и относительную разницу
                    abs_diff = abs(value1 - value2)
                    if abs(value1) > 0 and abs(value2) > 0:
                        rel_diff = abs_diff / ((value1 + value2) / 2) * 100  # в процентах
                    else:
                        rel_diff = np.nan
                    
                    param_diffs[param] = {
                        'absolute_diff': abs_diff,
                        'relative_diff': rel_diff,
                        'value1': value1,
                        'value2': value2
                    }
            
            # Проверяем совпадение типов
            type_match = False
            if (hasattr(cyclone1.parameters, 'thermal_type') and
                hasattr(cyclone2.parameters, 'thermal_type')):
                type_match = cyclone1.parameters.thermal_type == cyclone2.parameters.thermal_type
            
            # Создаем результат сравнения
            comparison = {
                'distance_km': distance,
                'time_difference_hours': time_diff,
                'pressure_difference_hPa': pressure_diff,
                'parameter_differences': param_diffs,
                'type_match': type_match,
                'cyclone1_id': getattr(cyclone1, 'track_id', None),
                'cyclone2_id': getattr(cyclone2, 'track_id', None),
                'cyclone1_time': cyclone1.time,
                'cyclone2_time': cyclone2.time
            }
            
            logger.debug(f"Проведено сравнение циклонов: "
                      f"расстояние={distance:.1f} км, "
                      f"разница времени={time_diff:.1f} ч, "
                      f"разница давления={pressure_diff:.1f} гПа")
            
            return comparison
            
        except Exception as e:
            error_msg = f"Ошибка при сравнении циклонов: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Рассчитывает расстояние между двумя точками на сфере (формула гаверсинуса).
        
        Аргументы:
            lat1: Широта первой точки (градусы).
            lon1: Долгота первой точки (градусы).
            lat2: Широта второй точки (градусы).
            lon2: Долгота второй точки (градусы).
            
        Возвращает:
            Расстояние в километрах.
        """
        from math import radians, sin, cos, sqrt, atan2
        
        # Конвертируем координаты из градусов в радианы
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Формула гаверсинуса
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Радиус Земли в километрах
        
        return r * c
    
    def compare_tracks(self, track1: List[Cyclone], track2: List[Cyclone]) -> Dict[str, Any]:
        """
        Сравнивает два трека циклонов.
        
        Аргументы:
            track1: Первый трек для сравнения (список циклонов).
            track2: Второй трек для сравнения (список циклонов).
            
        Возвращает:
            Словарь с результатами сравнения.
        """
        try:
            if not track1 or not track2:
                raise ValueError("Невозможно сравнить пустые треки")
            
            # Сортируем треки по времени
            track1 = sorted(track1, key=lambda c: c.time)
            track2 = sorted(track2, key=lambda c: c.time)
            
            # Рассчитываем базовые характеристики треков
            track1_duration = (track1[-1].time - track1[0].time).total_seconds() / 3600  # часы
            track2_duration = (track2[-1].time - track2[0].time).total_seconds() / 3600  # часы
            
            # Рассчитываем пройденное расстояние
            track1_distance = sum([
                self._calculate_distance(
                    track1[i-1].latitude, track1[i-1].longitude,
                    track1[i].latitude, track1[i].longitude
                )
                for i in range(1, len(track1))
            ])
            
            track2_distance = sum([
                self._calculate_distance(
                    track2[i-1].latitude, track2[i-1].longitude,
                    track2[i].latitude, track2[i].longitude
                )
                for i in range(1, len(track2))
            ])
            
            # Рассчитываем среднюю скорость движения
            track1_speed = track1_distance / max(1, track1_duration)
            track2_speed = track2_distance / max(1, track2_duration)
            
            # Находим минимальное давление в треках
            track1_min_pressure = min([c.central_pressure for c in track1])
            track2_min_pressure = min([c.central_pressure for c in track2])
            
            # Рассчитываем максимальную интенсивность (инверсное давление)
            track1_max_intensity = 1020 - track1_min_pressure
            track2_max_intensity = 1020 - track2_min_pressure
            
            # Сравниваем направление движения
            # Упрощенно - по начальной и конечной точкам
            track1_direction = np.arctan2(
                track1[-1].longitude - track1[0].longitude,
                track1[-1].latitude - track1[0].latitude
            ) * 180 / np.pi
            
            track2_direction = np.arctan2(
                track2[-1].longitude - track2[0].longitude,
                track2[-1].latitude - track2[0].latitude
            ) * 180 / np.pi
            
            # Нормализуем углы в диапазоне 0-360
            track1_direction = (track1_direction + 360) % 360
            track2_direction = (track2_direction + 360) % 360
            
            # Рассчитываем разницу направлений (0-180)
            direction_diff = min(
                abs(track1_direction - track2_direction),
                360 - abs(track1_direction - track2_direction)
            )
            
            # Рассчитываем метрику схожести треков
            track_similarity = self.calculate_track_similarity(track1, track2)
            
            # Создаем результат сравнения
            comparison = {
                'track1_id': getattr(track1[0], 'track_id', None),
                'track2_id': getattr(track2[0], 'track_id', None),
                'track1_duration_hours': track1_duration,
                'track2_duration_hours': track2_duration,
                'duration_difference_hours': abs(track1_duration - track2_duration),
                'duration_relative_diff': abs(track1_duration - track2_duration) / max(track1_duration, track2_duration) * 100,
                'track1_distance_km': track1_distance,
                'track2_distance_km': track2_distance,
                'distance_difference_km': abs(track1_distance - track2_distance),
                'distance_relative_diff': abs(track1_distance - track2_distance) / max(track1_distance, track2_distance) * 100,
                'track1_speed_kmh': track1_speed,
                'track2_speed_kmh': track2_speed,
                'speed_difference_kmh': abs(track1_speed - track2_speed),
                'track1_min_pressure_hPa': track1_min_pressure,
                'track2_min_pressure_hPa': track2_min_pressure,
                'pressure_difference_hPa': abs(track1_min_pressure - track2_min_pressure),
                'track1_direction_deg': track1_direction,
                'track2_direction_deg': track2_direction,
                'direction_difference_deg': direction_diff,
                'track_similarity': track_similarity,
                'point_count_diff': abs(len(track1) - len(track2))
            }
            
            logger.debug(f"Проведено сравнение треков: "
                      f"similarity={track_similarity:.2f}, "
                      f"duration_diff={comparison['duration_difference_hours']:.1f} ч, "
                      f"direction_diff={direction_diff:.1f}°")
            
            return comparison
            
        except Exception as e:
            error_msg = f"Ошибка при сравнении треков: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def calculate_track_similarity(self, track1: List[Cyclone], track2: List[Cyclone]) -> float:
        """
        Рассчитывает метрику схожести двух треков.
        
        Аргументы:
            track1: Первый трек (список циклонов).
            track2: Второй трек (список циклонов).
            
        Возвращает:
            Значение метрики схожести (0-1, где 1 - идеальное соответствие).
        """
        try:
            if not track1 or not track2:
                return 0.0
            
            # Сортируем треки по времени
            track1 = sorted(track1, key=lambda c: c.time)
            track2 = sorted(track2, key=lambda c: c.time)
            
            # Извлекаем координаты
            lats1 = np.array([c.latitude for c in track1])
            lons1 = np.array([c.longitude for c in track1])
            lats2 = np.array([c.latitude for c in track2])
            lons2 = np.array([c.longitude for c in track2])
            
            # Для сравнения треков разной длины используем интерполяцию
            # Приводим оба трека к одинаковому количеству точек (100)
            num_points = 100
            
            # Создаем параметризацию для интерполяции
            t1 = np.linspace(0, 1, len(track1))
            t2 = np.linspace(0, 1, len(track2))
            t_interp = np.linspace(0, 1, num_points)
            
            # Интерполируем треки
            lats1_interp = np.interp(t_interp, t1, lats1)
            lons1_interp = np.interp(t_interp, t1, lons1)
            lats2_interp = np.interp(t_interp, t2, lats2)
            lons2_interp = np.interp(t_interp, t2, lons2)
            
            # Рассчитываем среднее расстояние между соответствующими точками
            total_distance = 0
            for i in range(num_points):
                distance = self._calculate_distance(
                    lats1_interp[i], lons1_interp[i],
                    lats2_interp[i], lons2_interp[i]
                )
                total_distance += distance
            
            mean_distance = total_distance / num_points
            
            # Преобразуем расстояние в метрику схожести (0-1)
            # Используем экспоненциальную функцию для нормализации:
            # 1 при расстоянии 0, близко к 0 при больших расстояниях
            similarity = np.exp(-mean_distance / self.default_distance_threshold)
            
            return similarity
            
        except Exception as e:
            error_msg = f"Ошибка при расчете схожести треков: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def find_matching_cyclones(self, source_cyclones: List[Cyclone], target_cyclones: List[Cyclone],
                             max_distance: Optional[float] = None,
                             max_time_diff: Optional[float] = 12.0) -> List[Tuple[Cyclone, Cyclone, float]]:
        """
        Находит соответствующие циклоны в двух наборах.
        
        Аргументы:
            source_cyclones: Исходный набор циклонов.
            target_cyclones: Целевой набор циклонов для сопоставления.
            max_distance: Максимальное расстояние (км) для сопоставления.
                        Если None, используется значение по умолчанию.
            max_time_diff: Максимальная разница во времени (часы).
                         Если None, временная разница не учитывается.
            
        Возвращает:
            Список кортежей (исходный_циклон, соответствующий_циклон, расстояние).
        """
        try:
            if not source_cyclones or not target_cyclones:
                return []
            
            # Используем пороговое расстояние по умолчанию, если не указано
            if max_distance is None:
                max_distance = self.default_distance_threshold
            
            matches = []
            
            # Для каждого исходного циклона
            for source in source_cyclones:
                best_match = None
                best_distance = float('inf')
                
                # Для каждого целевого циклона
                for target in target_cyclones:
                    # Проверяем временную разницу, если указан порог
                    if max_time_diff is not None:
                        time_diff = abs((source.time - target.time).total_seconds()) / 3600
                        if time_diff > max_time_diff:
                            continue
                    
                    # Рассчитываем расстояние
                    distance = self._calculate_distance(
                        source.latitude, source.longitude,
                        target.latitude, target.longitude
                    )
                    
                    # Проверяем пороговое расстояние
                    if distance <= max_distance and distance < best_distance:
                        best_match = target
                        best_distance = distance
                
                # Если нашли соответствие
                if best_match is not None:
                    matches.append((source, best_match, best_distance))
            
            logger.info(f"Найдено {len(matches)} соответствий из {len(source_cyclones)} исходных циклонов")
            
            return matches
            
        except Exception as e:
            error_msg = f"Ошибка при поиске соответствующих циклонов: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)
    
    def compare_datasets(self, dataset1: List[Cyclone], dataset2: List[Cyclone],
                       max_distance: Optional[float] = None,
                       max_time_diff: Optional[float] = 12.0) -> Dict[str, Any]:
        """
        Сравнивает два набора данных о циклонах.
        
        Аргументы:
            dataset1: Первый набор циклонов.
            dataset2: Второй набор циклонов.
            max_distance: Максимальное расстояние (км) для сопоставления.
                        Если None, используется значение по умолчанию.
            max_time_diff: Максимальная разница во времени (часы).
                         Если None, временная разница не учитывается.
            
        Возвращает:
            Словарь с результатами сравнения.
        """
        try:
            if not dataset1 or not dataset2:
                raise ValueError("Невозможно сравнить пустые наборы данных")
            
            # Находим соответствия в обоих направлениях
            matches1to2 = self.find_matching_cyclones(dataset1, dataset2, max_distance, max_time_diff)
            matches2to1 = self.find_matching_cyclones(dataset2, dataset1, max_distance, max_time_diff)
            
            # Рассчитываем статистики
            dataset1_size = len(dataset1)
            dataset2_size = len(dataset2)
            matches_count = len(matches1to2)
            
            detection_rate = matches_count / dataset1_size
            false_alarm_rate = (dataset2_size - len(matches2to1)) / dataset2_size
            
            # Рассчитываем ошибки позиционирования
            distances = [dist for _, _, dist in matches1to2]
            mean_distance = np.mean(distances) if distances else np.nan
            median_distance = np.median(distances) if distances else np.nan
            max_distance_found = max(distances) if distances else np.nan
            
            # Сравниваем давление в соответствующих циклонах
            pressure_diffs = [
                abs(source.central_pressure - target.central_pressure)
                for source, target, _ in matches1to2
            ]
            
            mean_pressure_diff = np.mean(pressure_diffs) if pressure_diffs else np.nan
            median_pressure_diff = np.median(pressure_diffs) if pressure_diffs else np.nan
            max_pressure_diff = max(pressure_diffs) if pressure_diffs else np.nan
            
            # Рассчитываем статистику треков, если есть информация о треках
            track_stats = {}
            if all(hasattr(c, 'track_id') for c in dataset1 + dataset2):
                # Группируем циклоны по трекам
                tracks1 = {}
                for cyclone in dataset1:
                    if cyclone.track_id not in tracks1:
                        tracks1[cyclone.track_id] = []
                    tracks1[cyclone.track_id].append(cyclone)
                
                tracks2 = {}
                for cyclone in dataset2:
                    if cyclone.track_id not in tracks2:
                        tracks2[cyclone.track_id] = []
                    tracks2[cyclone.track_id].append(cyclone)
                
                # Считаем количество треков
                track_stats['track_count1'] = len(tracks1)
                track_stats['track_count2'] = len(tracks2)
                
                # Оцениваем соответствие треков
                # Трек A считается соответствующим треку B, если большинство
                # циклонов из A имеют соответствующие циклоны в B
                track_matches = 0
                
                for track_id, track_cyclones in tracks1.items():
                    matching_count = 0
                    
                    for cyclone in track_cyclones:
                        # Проверяем, есть ли циклон в списке соответствий
                        for source, _, _ in matches1to2:
                            if source == cyclone:
                                matching_count += 1
                                break
                    
                    # Если большинство циклонов имеют соответствие,
                    # считаем трек соответствующим
                    if matching_count / len(track_cyclones) > 0.5:
                        track_matches += 1
                
                track_stats['matching_tracks'] = track_matches
                track_stats['track_detection_rate'] = track_matches / len(tracks1)
            
            # Формируем результат
            result = {
                'dataset1_size': dataset1_size,
                'dataset2_size': dataset2_size,
                'matching_cyclones': matches_count,
                'detection_rate': detection_rate,
                'false_alarm_rate': false_alarm_rate,
                'mean_position_error_km': mean_distance,
                'median_position_error_km': median_distance,
                'max_position_error_km': max_distance_found,
                'mean_pressure_error_hPa': mean_pressure_diff,
                'median_pressure_error_hPa': median_pressure_diff,
                'max_pressure_error_hPa': max_pressure_diff,
                'track_statistics': track_stats
            }
            
            logger.info(f"Проведено сравнение наборов данных: "
                      f"detection_rate={detection_rate:.2f}, "
                      f"false_alarm_rate={false_alarm_rate:.2f}, "
                      f"mean_position_error={mean_distance:.1f} км")
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка при сравнении наборов данных: {str(e)}"
            logger.error(error_msg)
            raise ArcticCycloneError(error_msg)


def compare_cyclone_tracks(track1: List[Cyclone], track2: List[Cyclone]) -> Dict[str, Any]:
    """
    Сравнивает два трека циклонов.
    
    Аргументы:
        track1: Первый трек для сравнения (список циклонов).
        track2: Второй трек для сравнения (список циклонов).
        
    Возвращает:
        Словарь с результатами сравнения.
    """
    comparator = CycloneComparator()
    return comparator.compare_tracks(track1, track2)


def compare_cyclone_parameters(cyclone1: Cyclone, cyclone2: Cyclone) -> Dict[str, Any]:
    """
    Сравнивает параметры двух циклонов.
    
    Аргументы:
        cyclone1: Первый циклон для сравнения.
        cyclone2: Второй циклон для сравнения.
        
    Возвращает:
        Словарь с результатами сравнения.
    """
    comparator = CycloneComparator()
    return comparator.compare_cyclones(cyclone1, cyclone2)


def compare_datasets(dataset1: List[Cyclone], dataset2: List[Cyclone],
                   max_distance: float = 500.0,
                   max_time_diff: float = 12.0) -> Dict[str, Any]:
    """
    Сравнивает два набора данных о циклонах.
    
    Аргументы:
        dataset1: Первый набор циклонов.
        dataset2: Второй набор циклонов.
        max_distance: Максимальное расстояние (км) для сопоставления.
        max_time_diff: Максимальная разница во времени (часы).
        
    Возвращает:
        Словарь с результатами сравнения.
    """
    comparator = CycloneComparator(default_distance_threshold=max_distance)
    return comparator.compare_datasets(dataset1, dataset2, max_distance, max_time_diff)


def calculate_track_similarity(track1: List[Cyclone], track2: List[Cyclone]) -> float:
    """
    Рассчитывает метрику схожести двух треков.
    
    Аргументы:
        track1: Первый трек (список циклонов).
        track2: Второй трек (список циклонов).
        
    Возвращает:
        Значение метрики схожести (0-1, где 1 - идеальное соответствие).
    """
    comparator = CycloneComparator()
    return comparator.calculate_track_similarity(track1, track2)


def compare_spatial_distributions(dataset1: List[Cyclone], dataset2: List[Cyclone],
                                grid_resolution: float = 2.0) -> xr.Dataset:
    """
    Сравнивает пространственные распределения циклонов из двух наборов данных.
    
    Аргументы:
        dataset1: Первый набор циклонов.
        dataset2: Второй набор циклонов.
        grid_resolution: Разрешение сетки в градусах.
        
    Возвращает:
        Набор данных xarray с результатами сравнения.
    """
    try:
        # Создаем сетку
        lon_bins = np.arange(-180, 180 + grid_resolution, grid_resolution)
        lat_bins = np.arange(60, 90 + grid_resolution, grid_resolution)
        
        # Инициализируем массивы для подсчета
        count1 = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
        count2 = np.zeros_like(count1)
        
        # Заполняем гриды данными о циклонах
        for cyclone in dataset1:
            lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
            lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
            
            if (0 <= lat_idx < len(lat_bins)-1 and 
                0 <= lon_idx < len(lon_bins)-1):
                count1[lat_idx, lon_idx] += 1
        
        for cyclone in dataset2:
            lon_idx = np.searchsorted(lon_bins, cyclone.longitude) - 1
            lat_idx = np.searchsorted(lat_bins, cyclone.latitude) - 1
            
            if (0 <= lat_idx < len(lat_bins)-1 and 
                0 <= lon_idx < len(lon_bins)-1):
                count2[lat_idx, lon_idx] += 1
        
        # Рассчитываем разницу и относительную разницу
        count_diff = count1 - count2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.where(
                (count1 + count2) > 0,
                count_diff / ((count1 + count2) / 2) * 100,
                np.nan
            )
        
        # Рассчитываем корреляцию и метрики сходства
        flat_count1 = count1.flatten()
        flat_count2 = count2.flatten()
        
        # Корреляция Пирсона
        if np.any(flat_count1) and np.any(flat_count2):
            correlation = np.corrcoef(flat_count1, flat_count2)[0, 1]
        else:
            correlation = np.nan
        
        # Средняя квадратическая ошибка (MSE)
        mse = mean_squared_error(flat_count1, flat_count2)
        
        # Средняя абсолютная ошибка (MAE)
        mae = mean_absolute_error(flat_count1, flat_count2)
        
        # Средние координаты ячеек сетки
        lon_centers = lon_bins[:-1] + grid_resolution / 2
        lat_centers = lat_bins[:-1] + grid_resolution / 2
        
        # Создаем набор данных
        ds = xr.Dataset(
            data_vars={
                'count1': (
                    ['latitude', 'longitude'], 
                    count1
                ),
                'count2': (
                    ['latitude', 'longitude'], 
                    count2
                ),
                'difference': (
                    ['latitude', 'longitude'], 
                    count_diff
                ),
                'relative_difference': (
                    ['latitude', 'longitude'], 
                    rel_diff
                )
            },
            coords={
                'latitude': lat_centers,
                'longitude': lon_centers
            },
            attrs={
                'description': 'Comparison of spatial distributions of cyclones',
                'grid_resolution': f'{grid_resolution} degrees',
                'dataset1_size': len(dataset1),
                'dataset2_size': len(dataset2),
                'correlation': correlation,
                'mse': mse,
                'mae': mae,
                'creation_date': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Проведено сравнение пространственных распределений: "
                  f"correlation={correlation:.2f}, MSE={mse:.2f}, MAE={mae:.2f}")
        
        return ds
        
    except Exception as e:
        error_msg = f"Ошибка при сравнении пространственных распределений: {str(e)}"
        logger.error(error_msg)
        raise ArcticCycloneError(error_msg)


def compare_seasonal_distributions(dataset1: List[Cyclone], dataset2: List[Cyclone]) -> Dict[str, Any]:
    """
    Сравнивает сезонные распределения циклонов из двух наборов данных.
    
    Аргументы:
        dataset1: Первый набор циклонов.
        dataset2: Второй набор циклонов.
        
    Возвращает:
        Словарь с результатами сравнения.
    """
    try:
        # Инициализируем счетчики для сезонов
        seasons = ['DJF', 'MAM', 'JJA', 'SON']
        months_to_season = {
            12: 0, 1: 0, 2: 0,  # Зима (ДЯФ)
            3: 1, 4: 1, 5: 1,   # Весна (МАМ)
            6: 2, 7: 2, 8: 2,   # Лето (ИИА)
            9: 3, 10: 3, 11: 3  # Осень (СОН)
        }
        
        # Счетчики для сезонов и месяцев
        seasonal_counts1 = np.zeros(4)
        seasonal_counts2 = np.zeros(4)
        monthly_counts1 = np.zeros(12)
        monthly_counts2 = np.zeros(12)
        
        # Считаем циклоны по сезонам и месяцам
        for cyclone in dataset1:
            month = cyclone.time.month
            seasonal_counts1[months_to_season[month]] += 1
            monthly_counts1[month - 1] += 1
        
        for cyclone in dataset2:
            month = cyclone.time.month
            seasonal_counts2[months_to_season[month]] += 1
            monthly_counts2[month - 1] += 1
        
        # Рассчитываем разницу и относительную разницу
        seasonal_diff = seasonal_counts1 - seasonal_counts2
        monthly_diff = monthly_counts1 - monthly_counts2
        
        with np.errstate(divide='ignore', invalid='ignore'):
            seasonal_rel_diff = np.where(
                (seasonal_counts1 + seasonal_counts2) > 0,
                seasonal_diff / ((seasonal_counts1 + seasonal_counts2) / 2) * 100,
                np.nan
            )
            
            monthly_rel_diff = np.where(
                (monthly_counts1 + monthly_counts2) > 0,
                monthly_diff / ((monthly_counts1 + monthly_counts2) / 2) * 100,
                np.nan
            )
        
        # Рассчитываем корреляцию
        if np.any(seasonal_counts1) and np.any(seasonal_counts2):
            seasonal_correlation = np.corrcoef(seasonal_counts1, seasonal_counts2)[0, 1]
        else:
            seasonal_correlation = np.nan
            
        if np.any(monthly_counts1) and np.any(monthly_counts2):
            monthly_correlation = np.corrcoef(monthly_counts1, monthly_counts2)[0, 1]
        else:
            monthly_correlation = np.nan
        
        # Рассчитываем метрики сходства
        seasonal_mse = mean_squared_error(seasonal_counts1, seasonal_counts2)
        seasonal_mae = mean_absolute_error(seasonal_counts1, seasonal_counts2)
        monthly_mse = mean_squared_error(monthly_counts1, monthly_counts2)
        monthly_mae = mean_absolute_error(monthly_counts1, monthly_counts2)
        
        # Находим сезоны с максимальной разницей
        max_diff_season_idx = np.argmax(np.abs(seasonal_diff))
        max_diff_season = seasons[max_diff_season_idx]
        max_diff_value = seasonal_diff[max_diff_season_idx]
        
        # Находим месяц с максимальной разницей
        month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        max_diff_month_idx = np.argmax(np.abs(monthly_diff))
        max_diff_month = month_names[max_diff_month_idx]
        max_diff_month_value = monthly_diff[max_diff_month_idx]
        
        # Формируем результат
        result = {
            'dataset1_size': len(dataset1),
            'dataset2_size': len(dataset2),
            'seasonal_counts1': seasonal_counts1.tolist(),
            'seasonal_counts2': seasonal_counts2.tolist(),
            'seasonal_diff': seasonal_diff.tolist(),
            'seasonal_rel_diff': seasonal_rel_diff.tolist(),
            'seasonal_correlation': seasonal_correlation,
            'seasonal_mse': seasonal_mse,
            'seasonal_mae': seasonal_mae,
            'monthly_counts1': monthly_counts1.tolist(),
            'monthly_counts2': monthly_counts2.tolist(),
            'monthly_diff': monthly_diff.tolist(),
            'monthly_rel_diff': monthly_rel_diff.tolist(),
            'monthly_correlation': monthly_correlation,
            'monthly_mse': monthly_mse,
            'monthly_mae': monthly_mae,
            'max_diff_season': max_diff_season,
            'max_diff_season_value': float(max_diff_value),
            'max_diff_month': max_diff_month,
            'max_diff_month_value': float(max_diff_month_value),
            'seasons': seasons,
            'months': month_names
        }
        
        logger.info(f"Проведено сравнение сезонных распределений: "
                  f"seasonal_correlation={seasonal_correlation:.2f}, "
                  f"monthly_correlation={monthly_correlation:.2f}")
        
        return result
        
    except Exception as e:
        error_msg = f"Ошибка при сравнении сезонных распределений: {str(e)}"
        logger.error(error_msg)
        raise ArcticCycloneError(error_msg)