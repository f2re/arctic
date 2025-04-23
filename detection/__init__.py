"""
Пакет обнаружения и отслеживания циклонов для системы ArcticCyclone.

Содержит модули и классы для обнаружения арктических мезоциклонов,
отслеживания их жизненного цикла и анализа их характеристик.
"""

from .tracker import CycloneDetector, CycloneTracker
from .validators import DetectionValidator
from .criteria import CriteriaManager, BaseCriterion

__all__ = [
    'CycloneDetector',
    'CycloneTracker',
    'DetectionValidator',
    'CriteriaManager',
    'BaseCriterion'
]