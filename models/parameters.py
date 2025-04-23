"""
Модуль параметров циклонов для системы ArcticCyclone.

Предоставляет классы для представления и обработки
метеорологических параметров циклонов.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np

from .classifications import CycloneType

@dataclass
class CycloneParameters:
    """
    Контейнер для метеорологических параметров циклона.
    
    Содержит параметры, характеризующие физические свойства
    и состояние циклона.
    """
    # Основные параметры
    central_pressure: float  # гПа
    vorticity_850hPa: Optional[float] = None  # с^-1
    max_wind_speed: Optional[float] = None  # м/с
    radius: Optional[float] = None  # км
    
    # Термическая структура
    thermal_type: CycloneType = CycloneType.UNCLASSIFIED
    temperature_anomaly: Optional[float] = None  # K
    thickness_anomaly: Optional[Dict[str, float]] = None  # м
    
    # Производные параметры
    pressure_gradient: Optional[float] = None  # гПа/100км
    geostrophic_wind: Optional[float] = None  # м/с
    
    # Спектральные характеристики
    energy_spectrum: Optional[Dict[str, Any]] = None
    spatial_scale: Optional[float] = None  # км
    
    # Динамические характеристики
    vertical_motion: Optional[Dict[str, float]] = None  # Па/с
    divergence: Optional[float] = None  # с^-1
    
    # Влагосодержание
    total_precipitation: Optional[float] = None  # мм
    precipitable_water: Optional[float] = None  # кг/м²
    
    def __post_init__(self):
        """
        Валидация и обработка параметров после инициализации.
        """
        # Проверяем тип термической структуры
        if not isinstance(self.thermal_type, CycloneType):
            try:
                self.thermal_type = CycloneType(self.thermal_type)
            except:
                self.thermal_type = CycloneType.UNCLASSIFIED
        
        # Проверяем и инициализируем словари
        if self.thickness_anomaly is None:
            self.thickness_anomaly = {}
            
        if self.energy_spectrum is None:
            self.energy_spectrum = {}
            
        if self.vertical_motion is None:
            self.vertical_motion = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует параметры в словарь.
        
        Возвращает:
            Словарь с параметрами циклона.
        """
        result = {
            'central_pressure': self.central_pressure
        }
        
        # Добавляем остальные параметры, если они не None
        if self.vorticity_850hPa is not None:
            result['vorticity_850hPa'] = self.vorticity_850hPa
        
        if self.max_wind_speed is not None:
            result['max_wind_speed'] = self.max_wind_speed
        
        if self.radius is not None:
            result['radius'] = self.radius
        
        if self.thermal_type != CycloneType.UNCLASSIFIED:
            result['thermal_type'] = self.thermal_type.value
        
        if self.temperature_anomaly is not None:
            result['temperature_anomaly'] = self.temperature_anomaly
        
        if self.thickness_anomaly:
            result['thickness_anomaly'] = self.thickness_anomaly
        
        if self.pressure_gradient is not None:
            result['pressure_gradient'] = self.pressure_gradient
        
        if self.geostrophic_wind is not None:
            result['geostrophic_wind'] = self.geostrophic_wind
        
        if self.energy_spectrum:
            result['energy_spectrum'] = self.energy_spectrum
        
        if self.spatial_scale is not None:
            result['spatial_scale'] = self.spatial_scale
        
        if self.vertical_motion:
            result['vertical_motion'] = self.vertical_motion
        
        if self.divergence is not None:
            result['divergence'] = self.divergence
        
        if self.total_precipitation is not None:
            result['total_precipitation'] = self.total_precipitation
        
        if self.precipitable_water is not None:
            result['precipitable_water'] = self.precipitable_water
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CycloneParameters':
        """
        Создает объект параметров из словаря.
        
        Аргументы:
            data: Словарь с параметрами циклона.
            
        Возвращает:
            Объект CycloneParameters.
        """
        # Создаем объект с обязательными параметрами
        params = cls(
            central_pressure=data['central_pressure']
        )
        
        # Устанавливаем остальные параметры, если они есть в словаре
        if 'vorticity_850hPa' in data:
            params.vorticity_850hPa = data['vorticity_850hPa']
        
        if 'max_wind_speed' in data:
            params.max_wind_speed = data['max_wind_speed']
        
        if 'radius' in data:
            params.radius = data['radius']
        
        if 'thermal_type' in data:
            try:
                params.thermal_type = CycloneType(data['thermal_type'])
            except:
                params.thermal_type = CycloneType.UNCLASSIFIED
        
        if 'temperature_anomaly' in data:
            params.temperature_anomaly = data['temperature_anomaly']
        
        if 'thickness_anomaly' in data:
            params.thickness_anomaly = data['thickness_anomaly']
        
        if 'pressure_gradient' in data:
            params.pressure_gradient = data['pressure_gradient']
        
        if 'geostrophic_wind' in data:
            params.geostrophic_wind = data['geostrophic_wind']
        
        if 'energy_spectrum' in data:
            params.energy_spectrum = data['energy_spectrum']
        
        if 'spatial_scale' in data:
            params.spatial_scale = data['spatial_scale']
        
        if 'vertical_motion' in data:
            params.vertical_motion = data['vertical_motion']
        
        if 'divergence' in data:
            params.divergence = data['divergence']
        
        if 'total_precipitation' in data:
            params.total_precipitation = data['total_precipitation']
        
        if 'precipitable_water' in data:
            params.precipitable_water = data['precipitable_water']
        
        return params
    
    def is_mesocyclone(self) -> bool:
        """
        Проверяет, является ли циклон мезоциклоном на основе параметров.
        
        Возвращает:
            True, если циклон соответствует критериям мезоциклона.
        """
        # Основные критерии для арктических мезоциклонов:
        # 1. Радиус менее 1000 км
        # 2. Высокая завихренность
        # 3. Обычно холодный верх, теплый низ (гибридная структура)
        
        if self.radius is not None and self.radius > 1000:
            return False
            
        if self.vorticity_850hPa is not None and self.vorticity_850hPa < 1e-5:
            return False
            
        if self.thermal_type == CycloneType.HYBRID:
            return True
            
        # Если не все параметры доступны, но имеющиеся соответствуют
        if (self.radius is not None and self.radius < 1000) or \
           (self.vorticity_850hPa is not None and self.vorticity_850hPa >= 1e-5):
            return True
            
        return False
    
    def calculate_intensity_index(self) -> float:
        """
        Рассчитывает индекс интенсивности циклона на основе комбинации параметров.
        
        Возвращает:
            Индекс интенсивности от 0 (слабый) до 10 (очень сильный).
        """
        # Базовый индекс на основе центрального давления
        # Примерный диапазон: 950-1010 гПа
        pressure_index = max(0, min(10, (1010 - self.central_pressure) / 6))
        
        # Модификаторы на основе других параметров
        modifiers = 0
        
        # Модификатор по завихренности
        if self.vorticity_850hPa is not None:
            # Типичный диапазон: 1e-5 - 1e-4 с^-1
            vorticity_modifier = max(0, min(3, (self.vorticity_850hPa - 1e-5) * 3e4))
            modifiers += vorticity_modifier
        
        # Модификатор по скорости ветра
        if self.max_wind_speed is not None:
            # Типичный диапазон: 10-40 м/с
            wind_modifier = max(0, min(3, (self.max_wind_speed - 10) / 10))
            modifiers += wind_modifier
        
        # Модификатор по градиенту давления
        if self.pressure_gradient is not None:
            # Типичный диапазон: 0.5-3 гПа/100км
            gradient_modifier = max(0, min(2, (self.pressure_gradient - 0.5) / 1.25))
            modifiers += gradient_modifier
        
        # Модификатор по размеру (меньшие циклоны обычно менее интенсивны)
        if self.radius is not None:
            # Типичный диапазон: 100-1000 км
            size_modifier = max(0, min(2, (self.radius - 100) / 450))
            modifiers += size_modifier
        
        # Финальный индекс: 60% по давлению, 40% по модификаторам
        intensity_index = 0.6 * pressure_index + 0.4 * (modifiers / max(1, 4))
        
        # Нормализуем к диапазону 0-10
        intensity_index = max(0, min(10, intensity_index))
        
        return intensity_index