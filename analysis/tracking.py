# Отслеживание циклонов во времени
"""
Функции для отслеживания циклонов во времени.
"""

import numpy as np
from analysis.metrics import haversine_distance

def track_cyclones(cyclone_centers, previous_tracks, max_distance=300, hours_per_step=1):
    """
    Отслеживает циклоны во времени, связывая текущие центры с предыдущими треками.
    
    Параметры:
    ----------
    cyclone_centers : list
        Список центров циклонов на текущем временном шаге
    previous_tracks : dict
        Словарь с информацией о предыдущих треках
    max_distance : float
        Максимальное расстояние для связывания центров (км)
    hours_per_step : int
        Количество часов между временными шагами
        
    Возвращает:
    -----------
    dict
        Обновленный словарь треков
    dict
        Словарь связей между текущими центрами и треками
    """
    current_tracked = {}  # Словарь для текущего шага
    updated_tracks = previous_tracks.copy()
    
    # Для каждого обнаруженного центра
    for cyclone_center in cyclone_centers:
        lat, lon = cyclone_center[:2]
        
        # Проверяем, есть ли уже такой циклон в списке отслеживания
        is_tracked = False
        track_id = None
        
        for tid, track_info in previous_tracks.items():
            last_lat = track_info['last_lat']
            last_lon = track_info['last_lon']
            
            # Вычисляем расстояние между последним положением и текущим
            distance = haversine_distance(lat, lon, last_lat, last_lon)
            
            # Если расстояние меньше допустимого порога, считаем циклоны одним и тем же
            if distance < max_distance:
                track_id = tid
                is_tracked = True
                break
        
        # Создаем новый идентификатор циклона если не найден существующий
        if not is_tracked:
            track_id = f"{lat:.2f}_{lon:.2f}"
            
            # Добавляем новый трек
            updated_tracks[track_id] = {
                'start_time': 'current',  # будет заменено в вызывающей функции
                'last_time': 'current',   # будет заменено в вызывающей функции
                'first_lat': lat,
                'first_lon': lon,
                'last_lat': lat,
                'last_lon': lon,
                'min_pressure': cyclone_center[2] if len(cyclone_center) > 2 else None,
                'max_depth': cyclone_center[3] if len(cyclone_center) > 3 else None,
                'positions': [(lat, lon)],
                'times': ['current'],     # будет заменено в вызывающей функции
                'duration': hours_per_step
            }
        else:
            # Обновляем существующий трек
            updated_tracks[track_id]['last_lat'] = lat
            updated_tracks[track_id]['last_lon'] = lon
            updated_tracks[track_id]['positions'].append((lat, lon))
            updated_tracks[track_id]['times'].append('current')  # будет заменено
            updated_tracks[track_id]['duration'] += hours_per_step
            
            # Обновляем минимальное давление если текущее меньше
            if len(cyclone_center) > 2:
                pressure = cyclone_center[2]
                if pressure < updated_tracks[track_id]['min_pressure']:
                    updated_tracks[track_id]['min_pressure'] = pressure
                    if len(cyclone_center) > 3:
                        updated_tracks[track_id]['max_depth'] = cyclone_center[3]
        
        # Добавляем циклон в текущий список отслеживания
        current_tracked[track_id] = cyclone_center
    
    return updated_tracks, current_tracked