"""
Модуль классификаций циклонов для системы ArcticCyclone.

Предоставляет перечисления и классы для классификации
циклонов по различным параметрам.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Any
import json

class CycloneType(Enum):
    """
    Классификация циклонов по термической структуре.
    """
    COLD_CORE = "cold_core"  # Холодный центр (холоднее окружения на всех уровнях)
    WARM_CORE = "warm_core"  # Теплый центр (теплее окружения на всех уровнях)
    HYBRID = "hybrid"  # Гибридная структура (обычно холодный верх, теплый низ)
    UNCLASSIFIED = "unclassified"  # Неопределенная структура
    
    def __str__(self) -> str:
        return self.value
    
    @staticmethod
    def from_string(value: str) -> 'CycloneType':
        """
        Создает объект типа из строки.
        
        Аргументы:
            value: Строковое представление типа.
            
        Возвращает:
            Объект CycloneType.
            
        Вызывает:
            ValueError: Если строка не соответствует ни одному типу.
        """
        try:
            return CycloneType(value)
        except ValueError:
            # Пытаемся найти близкое соответствие
            value_lower = value.lower()
            
            if "cold" in value_lower:
                return CycloneType.COLD_CORE
            elif "warm" in value_lower:
                return CycloneType.WARM_CORE
            elif "hybrid" in value_lower or "mixed" in value_lower:
                return CycloneType.HYBRID
            else:
                return CycloneType.UNCLASSIFIED
    
    def get_description(self) -> str:
        """
        Возвращает описание типа циклона.
        
        Возвращает:
            Строка с описанием типа.
        """
        descriptions = {
            CycloneType.COLD_CORE: "Холодный центр (холоднее окружения на всех уровнях)",
            CycloneType.WARM_CORE: "Теплый центр (теплее окружения на всех уровнях)",
            CycloneType.HYBRID: "Гибридная структура (обычно холодный верх, теплый низ)",
            CycloneType.UNCLASSIFIED: "Неопределенная структура"
        }
        
        return descriptions.get(self, "Неизвестный тип")


class CycloneIntensity(Enum):
    """
    Классификация циклонов по интенсивности.
    """
    WEAK = "weak"  # Слабый циклон
    MODERATE = "moderate"  # Умеренный циклон
    STRONG = "strong"  # Сильный циклон
    VERY_STRONG = "very_strong"  # Очень сильный циклон
    
    def __str__(self) -> str:
        return self.value
    
    @staticmethod
    def from_pressure(pressure: float) -> 'CycloneIntensity':
        """
        Определяет интенсивность циклона по центральному давлению.
        
        Аргументы:
            pressure: Центральное давление циклона (гПа).
            
        Возвращает:
            Категория интенсивности.
        """
        if pressure < 960:
            return CycloneIntensity.VERY_STRONG
        elif pressure < 980:
            return CycloneIntensity.STRONG
        elif pressure < 995:
            return CycloneIntensity.MODERATE
        else:
            return CycloneIntensity.WEAK
    
    def get_description(self) -> str:
        """
        Возвращает описание интенсивности циклона.
        
        Возвращает:
            Строка с описанием интенсивности.
        """
        descriptions = {
            CycloneIntensity.WEAK: "Слабый циклон (давление > 995 гПа)",
            CycloneIntensity.MODERATE: "Умеренный циклон (980-995 гПа)",
            CycloneIntensity.STRONG: "Сильный циклон (960-980 гПа)",
            CycloneIntensity.VERY_STRONG: "Очень сильный циклон (давление < 960 гПа)"
        }
        
        return descriptions.get(self, "Неизвестная интенсивность")


class CycloneLifeStage(Enum):
    """
    Стадии жизненного цикла циклона.
    """
    GENESIS = "genesis"  # Зарождение
    INTENSIFICATION = "intensification"  # Углубление (усиление)
    MATURE = "mature"  # Зрелость (максимальная интенсивность)
    DISSIPATION = "dissipation"  # Заполнение (ослабление)
    UNKNOWN = "unknown"  # Неизвестная стадия
    
    def __str__(self) -> str:
        return self.value
    
    def get_description(self) -> str:
        """
        Возвращает описание стадии жизненного цикла.
        
        Возвращает:
            Строка с описанием стадии.
        """
        descriptions = {
            CycloneLifeStage.GENESIS: "Зарождение циклона",
            CycloneLifeStage.INTENSIFICATION: "Стадия углубления (усиления) циклона",
            CycloneLifeStage.MATURE: "Стадия зрелости (максимальной интенсивности) циклона",
            CycloneLifeStage.DISSIPATION: "Стадия заполнения (ослабления) циклона",
            CycloneLifeStage.UNKNOWN: "Неизвестная стадия жизненного цикла"
        }
        
        return descriptions.get(self, "Неизвестная стадия")


class MesocycloneClassifier:
    """
    Классификатор мезоциклонов на основе их характеристик.
    """
    
    def __init__(self):
        """
        Инициализирует классификатор.
        """
        pass
    
    def classify_by_size(self, radius: float) -> str:
        """
        Классифицирует мезоциклон по размеру.
        
        Аргументы:
            radius: Радиус циклона (км).
            
        Возвращает:
            Категория размера.
        """
        if radius < 200:
            return "micro"
        elif radius < 500:
            return "meso-beta"
        elif radius < 1000:
            return "meso-alpha"
        else:
            return "synoptic"
    
    def classify_by_thermal_structure(self, 
                                    t500_anomaly: float, 
                                    t850_anomaly: float) -> CycloneType:
        """
        Классифицирует мезоциклон по термической структуре.
        
        Аргументы:
            t500_anomaly: Аномалия температуры на 500 гПа (K).
            t850_anomaly: Аномалия температуры на 850 гПа (K).
            
        Возвращает:
            Тип термической структуры.
        """
        if t500_anomaly > 1.0 and t850_anomaly > 1.0:
            return CycloneType.WARM_CORE
        elif t500_anomaly < -1.0 and t850_anomaly < -1.0:
            return CycloneType.COLD_CORE
        elif t500_anomaly < 0 and t850_anomaly > 0:
            return CycloneType.HYBRID
        else:
            return CycloneType.UNCLASSIFIED
    
    def classify_by_intensity(self, 
                           central_pressure: float,
                           pressure_gradient: Optional[float] = None,
                           vorticity: Optional[float] = None,
                           wind_speed: Optional[float] = None) -> CycloneIntensity:
        """
        Классифицирует мезоциклон по интенсивности.
        
        Аргументы:
            central_pressure: Центральное давление (гПа).
            pressure_gradient: Градиент давления (гПа/100км).
            vorticity: Завихренность на 850 гПа (1/с).
            wind_speed: Максимальная скорость ветра (м/с).
            
        Возвращает:
            Категория интенсивности.
        """
        # Базовая классификация по давлению
        intensity = CycloneIntensity.from_pressure(central_pressure)
        
        # Корректируем категорию на основе дополнительных параметров
        intensity_value = {
            CycloneIntensity.WEAK: 1,
            CycloneIntensity.MODERATE: 2,
            CycloneIntensity.STRONG: 3,
            CycloneIntensity.VERY_STRONG: 4
        }[intensity]
        
        modifiers = 0
        modifier_count = 0
        
        # Модификатор по градиенту давления
        if pressure_gradient is not None:
            if pressure_gradient > 2.0:
                modifiers += 1
            elif pressure_gradient < 0.5:
                modifiers -= 1
            modifier_count += 1
        
        # Модификатор по завихренности
        if vorticity is not None:
            if vorticity > 5e-5:
                modifiers += 1
            elif vorticity < 1e-5:
                modifiers -= 1
            modifier_count += 1
        
        # Модификатор по скорости ветра
        if wind_speed is not None:
            if wind_speed > 25:
                modifiers += 1
            elif wind_speed < 10:
                modifiers -= 1
            modifier_count += 1
        
        # Применяем модификаторы, если есть хотя бы один
        if modifier_count > 0:
            # Средний модификатор
            avg_modifier = modifiers / modifier_count
            
            # Корректируем категорию
            if avg_modifier > 0.5:
                intensity_value += 1
            elif avg_modifier < -0.5:
                intensity_value -= 1
            
            # Ограничиваем диапазон
            intensity_value = max(1, min(4, intensity_value))
            
            # Преобразуем обратно в категорию
            intensity_map = {
                1: CycloneIntensity.WEAK,
                2: CycloneIntensity.MODERATE,
                3: CycloneIntensity.STRONG,
                4: CycloneIntensity.VERY_STRONG
            }
            
            intensity = intensity_map[intensity_value]
        
        return intensity