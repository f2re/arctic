"""
Модуль визуализации параметров циклонов для системы ArcticCyclone.

Предоставляет функции для создания графиков, диаграмм и других
визуализаций параметров арктических циклонов.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import logging
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import gaussian_kde
import xarray as xr

from models.cyclone import Cyclone
from models.classifications import CycloneType, CycloneIntensity, CycloneLifeStage
from core.exceptions import VisualizationError
from .mappers import create_arctic_map, plot_cyclone_centers, add_map_features

# Инициализация логгера
logger = logging.getLogger(__name__)


def plot_cyclone_parameters(cyclone: Cyclone,
                         parameters: List[str] = None,
                         figsize: Tuple[float, float] = (10, 6)) -> Tuple[Figure, List[Axes]]:
    """
    Создает визуализацию основных параметров циклона.
    
    Аргументы:
        cyclone: Циклон для визуализации.
        parameters: Список параметров для отображения. Если None, использует
                  стандартный набор параметров.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        
    Возвращает:
        Кортеж (figure, list_of_axes) с созданной визуализацией.
    """
    try:
        # Определяем стандартный набор параметров, если не указано
        if parameters is None:
            parameters = [
                'central_pressure',         # Центральное давление
                'vorticity_850hPa',         # Завихренность на 850 гПа
                'max_wind_speed',           # Максимальная скорость ветра
                'radius',                    # Радиус циклона
                'pressure_gradient'          # Градиент давления
            ]
        
        # Создаем словарь значений параметров и их атрибутов
        param_values = {}
        param_labels = {
            'central_pressure': 'Центральное давление',
            'vorticity_850hPa': 'Завихренность на 850 гПа',
            'max_wind_speed': 'Максимальная скорость ветра',
            'radius': 'Радиус циклона',
            'pressure_gradient': 'Градиент давления',
            'temperature_anomaly': 'Аномалия температуры',
            'thickness_500_850': 'Толщина слоя 500-850 гПа',
            'thermal_type': 'Термический тип',
            'age': 'Возраст циклона',
            'intensity': 'Интенсивность',
            'life_stage': 'Стадия жизненного цикла'
        }
        param_units = {
            'central_pressure': 'гПа',
            'vorticity_850hPa': 'с⁻¹',
            'max_wind_speed': 'м/с',
            'radius': 'км',
            'pressure_gradient': 'гПа/100км',
            'temperature_anomaly': 'K',
            'thickness_500_850': 'м',
            'age': 'ч'
        }
        
        # Извлекаем значения параметров
        for param in parameters:
            if param == 'central_pressure':
                param_values[param] = cyclone.central_pressure
            elif param == 'age':
                param_values[param] = cyclone.age
            elif param == 'intensity':
                param_values[param] = cyclone.calculate_intensity()
            elif param == 'life_stage':
                param_values[param] = cyclone.life_stage
            elif param == 'thermal_type':
                param_values[param] = cyclone.parameters.thermal_type if hasattr(cyclone.parameters, 'thermal_type') else CycloneType.UNCLASSIFIED
            elif hasattr(cyclone.parameters, param):
                param_values[param] = getattr(cyclone.parameters, param)
            else:
                param_values[param] = None
        
        # Создаем фигуру
        fig, axes = plt.subplots(len(parameters), 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Отображаем параметры
        for i, param in enumerate(parameters):
            ax = axes[i]
            
            # Получаем значение параметра
            value = param_values[param]
            
            if value is None:
                # Если значение отсутствует, показываем сообщение
                ax.text(0.5, 0.5, "Данные недоступны", ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            elif param in ['thermal_type', 'intensity', 'life_stage']:
                # Для перечислений создаем цветной прямоугольник
                if param == 'thermal_type':
                    colors = {
                        CycloneType.COLD_CORE: 'blue',
                        CycloneType.WARM_CORE: 'red',
                        CycloneType.HYBRID: 'purple',
                        CycloneType.UNCLASSIFIED: 'gray'
                    }
                    color = colors.get(value, 'gray')
                    label = value.name if hasattr(value, 'name') else 'UNKNOWN'
                elif param == 'intensity':
                    colors = {
                        CycloneIntensity.WEAK: 'green',
                        CycloneIntensity.MODERATE: 'yellow',
                        CycloneIntensity.STRONG: 'orange',
                        CycloneIntensity.VERY_STRONG: 'red'
                    }
                    color = colors.get(value, 'gray')
                    label = value.name if hasattr(value, 'name') else 'UNKNOWN'
                else:  # life_stage
                    colors = {
                        CycloneLifeStage.GENESIS: 'skyblue',
                        CycloneLifeStage.INTENSIFICATION: 'blue',
                        CycloneLifeStage.MATURE: 'darkblue',
                        CycloneLifeStage.DISSIPATION: 'lightblue',
                        CycloneLifeStage.UNKNOWN: 'gray'
                    }
                    color = colors.get(value, 'gray')
                    label = value.name if hasattr(value, 'name') else 'UNKNOWN'
                
                ax.add_patch(plt.Rectangle((0.25, 0.25), 0.5, 0.5, color=color))
                ax.text(0.5, 0.5, label, ha='center', va='center', 
                      color='white' if color in ['blue', 'darkblue', 'purple'] else 'black',
                      fontsize=12, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Для числовых параметров создаем гистограмму с выделенным значением
                if param == 'vorticity_850hPa':
                    # Преобразуем завихренность в 10⁻⁵ с⁻¹ для читаемости
                    value_to_show = value * 1e5
                    units = '10⁻⁵ с⁻¹'
                else:
                    value_to_show = value
                    units = param_units.get(param, '')
                
                # Определяем диапазон и создаем синтетическую гистограмму
                if param == 'central_pressure':
                    x_range = np.linspace(950, 1020, 100)
                    center = 985
                    std = 15
                elif param == 'vorticity_850hPa':
                    x_range = np.linspace(0, 10, 100)  # в 10⁻⁵ с⁻¹
                    center = 3
                    std = 1.5
                elif param == 'max_wind_speed':
                    x_range = np.linspace(0, 40, 100)
                    center = 20
                    std = 8
                elif param == 'radius':
                    x_range = np.linspace(0, 1000, 100)
                    center = 500
                    std = 200
                elif param == 'pressure_gradient':
                    x_range = np.linspace(0, 5, 100)
                    center = 2
                    std = 1
                else:
                    x_range = np.linspace(0, value * 2, 100)
                    center = value
                    std = value / 3
                
                # Создаем синтетическое распределение
                if param == 'central_pressure':
                    # Для давления - чем ниже, тем интенсивнее
                    y = np.exp(-(x_range - center)**2 / (2 * std**2))
                else:
                    # Для других параметров - чем выше, тем интенсивнее
                    y = np.exp(-(x_range - center)**2 / (2 * std**2))
                
                # Отображаем гистограмму
                ax.fill_between(x_range, 0, y, alpha=0.3, color='gray')
                
                # Выделяем текущее значение
                if param == 'central_pressure':
                    color = plt.cm.coolwarm_r(plt.Normalize(950, 1020)(value_to_show))
                else:
                    # Для других параметров - более высокие значения более интенсивные
                    if param == 'vorticity_850hPa':
                        color = plt.cm.viridis(plt.Normalize(0, 10)(value_to_show))
                    elif param == 'max_wind_speed':
                        color = plt.cm.viridis(plt.Normalize(0, 40)(value_to_show))
                    elif param == 'radius':
                        color = plt.cm.viridis(plt.Normalize(0, 1000)(value_to_show))
                    elif param == 'pressure_gradient':
                        color = plt.cm.viridis(plt.Normalize(0, 5)(value_to_show))
                    else:
                        color = 'red'
                
                # Отображаем вертикальную линию на значении параметра
                ax.axvline(x=value_to_show, color=color, linestyle='-', linewidth=2)
                ax.text(value_to_show, np.max(y), f"{value_to_show:.2f} {units}",
                      ha='center', va='bottom', fontsize=10, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                ax.set_xlim(np.min(x_range), np.max(x_range))
                ax.set_ylim(0, np.max(y) * 1.2)
                ax.set_yticks([])
            
            # Добавляем заголовок параметра
            ax.set_title(param_labels.get(param, param), fontsize=12)
        
        # Настраиваем расположение
        plt.tight_layout()
        
        # Добавляем общий заголовок
        fig.suptitle(
            f"Параметры циклона (ID: {cyclone.track_id}, Дата: {cyclone.time.strftime('%Y-%m-%d %H:%M')})",
            fontsize=14, y=1.05
        )
        
        return fig, axes
        
    except Exception as e:
        error_msg = f"Ошибка при визуализации параметров циклона: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def plot_parameter_correlation(cyclones: List[Cyclone],
                            x_param: str,
                            y_param: str,
                            color_by: Optional[str] = None,
                            figsize: Tuple[float, float] = (10, 8)) -> Tuple[Figure, Axes]:
    """
    Создает диаграмму рассеяния для анализа корреляции между параметрами циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        x_param: Параметр для оси X.
        y_param: Параметр для оси Y.
        color_by: Параметр для цветового кодирования точек. Если None, все точки одного цвета.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        
    Возвращает:
        Кортеж (figure, axis) с созданной диаграммой.
    """
    try:
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=figsize)
        
        # Словарь с метками осей
        param_labels = {
            'central_pressure': 'Центральное давление (гПа)',
            'vorticity_850hPa': 'Завихренность на 850 гПа (10⁻⁵ с⁻¹)',
            'max_wind_speed': 'Максимальная скорость ветра (м/с)',
            'radius': 'Радиус циклона (км)',
            'pressure_gradient': 'Градиент давления (гПа/100км)',
            'temperature_anomaly': 'Аномалия температуры (K)',
            'thickness_500_850': 'Толщина слоя 500-850 гПа (м)',
            'age': 'Возраст циклона (ч)',
            'intensity_index': 'Индекс интенсивности'
        }
        
        # Проверяем допустимость параметров
        for param in [x_param, y_param]:
            if param not in param_labels and param not in ['thermal_type', 'intensity', 'life_stage']:
                logger.warning(f"Неизвестный параметр: {param}")
        
        # Извлекаем значения параметров
        x_values = []
        y_values = []
        colors = []
        
        for cyclone in cyclones:
            # Извлекаем значение X
            x_val = None
            if x_param == 'central_pressure':
                x_val = cyclone.central_pressure
            elif x_param == 'age':
                x_val = cyclone.age
            elif x_param == 'intensity_index':
                if hasattr(cyclone, 'calculate_intensity_index'):
                    x_val = cyclone.calculate_intensity_index()
            elif hasattr(cyclone.parameters, x_param):
                x_val = getattr(cyclone.parameters, x_param)
            
            # Извлекаем значение Y
            y_val = None
            if y_param == 'central_pressure':
                y_val = cyclone.central_pressure
            elif y_param == 'age':
                y_val = cyclone.age
            elif y_param == 'intensity_index':
                if hasattr(cyclone, 'calculate_intensity_index'):
                    y_val = cyclone.calculate_intensity_index()
            elif hasattr(cyclone.parameters, y_param):
                y_val = getattr(cyclone.parameters, y_param)
            
            # Пропускаем записи с отсутствующими значениями
            if x_val is None or y_val is None:
                continue
            
            # Преобразуем завихренность в 10⁻⁵ с⁻¹ для читаемости
            if x_param == 'vorticity_850hPa':
                x_val *= 1e5
            if y_param == 'vorticity_850hPa':
                y_val *= 1e5
            
            x_values.append(x_val)
            y_values.append(y_val)
            
            # Определяем цвет точки, если требуется
            if color_by is not None:
                if color_by == 'thermal_type':
                    if hasattr(cyclone.parameters, 'thermal_type'):
                        colors.append(cyclone.parameters.thermal_type.value)
                    else:
                        colors.append('unknown')
                elif color_by == 'intensity':
                    colors.append(cyclone.calculate_intensity().value)
                elif color_by == 'life_stage':
                    colors.append(cyclone.life_stage.value)
                elif color_by == 'central_pressure':
                    colors.append(cyclone.central_pressure)
                elif hasattr(cyclone.parameters, color_by):
                    color_val = getattr(cyclone.parameters, color_by)
                    if color_by == 'vorticity_850hPa':
                        color_val *= 1e5
                    colors.append(color_val)
                else:
                    colors.append('unknown')
        
        # Проверяем, что есть данные для визуализации
        if len(x_values) == 0 or len(y_values) == 0:
            ax.text(0.5, 0.5, "Недостаточно данных для визуализации", 
                  ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Определяем цветовую кодировку
            if color_by is not None:
                if color_by in ['thermal_type', 'intensity', 'life_stage']:
                    # Для категориальных переменных используем дискретную цветовую карту
                    unique_colors = list(set(colors))
                    color_map = {}
                    
                    if color_by == 'thermal_type':
                        thermal_colors = {
                            'cold_core': 'blue',
                            'warm_core': 'red',
                            'hybrid': 'purple',
                            'unclassified': 'gray',
                            'unknown': 'lightgray'
                        }
                        color_map = {color: thermal_colors.get(color, 'black') for color in unique_colors}
                    elif color_by == 'intensity':
                        intensity_colors = {
                            'weak': 'green',
                            'moderate': 'yellow',
                            'strong': 'orange',
                            'very_strong': 'red',
                            'unknown': 'lightgray'
                        }
                        color_map = {color: intensity_colors.get(color, 'black') for color in unique_colors}
                    elif color_by == 'life_stage':
                        stage_colors = {
                            'genesis': 'skyblue',
                            'intensification': 'blue',
                            'mature': 'darkblue',
                            'dissipation': 'lightblue',
                            'unknown': 'gray'
                        }
                        color_map = {color: stage_colors.get(color, 'black') for color in unique_colors}
                    
                    # Отображаем точки с цветовой кодировкой
                    for i in range(len(x_values)):
                        ax.scatter(x_values[i], y_values[i], color=color_map.get(colors[i], 'black'),
                                 alpha=0.7, edgecolor='none')
                    
                    # Добавляем легенду
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                              markersize=10, label=label)
                        for label, color in color_map.items()
                    ]
                    ax.legend(handles=legend_elements, title=param_labels.get(color_by, color_by))
                    
                else:
                    # Для непрерывных переменных используем цветовую карту
                    if color_by == 'central_pressure':
                        # Для давления - чем ниже, тем интенсивнее
                        scatter = ax.scatter(x_values, y_values, c=colors, cmap='coolwarm_r',
                                          alpha=0.7, edgecolor='none')
                    else:
                        scatter = ax.scatter(x_values, y_values, c=colors, cmap='viridis',
                                          alpha=0.7, edgecolor='none')
                        
                    # Добавляем цветовую шкалу
                    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
                    cbar.set_label(param_labels.get(color_by, color_by))
            else:
                # Отображаем точки без цветовой кодировки
                ax.scatter(x_values, y_values, color='blue', alpha=0.7, edgecolor='none')
            
            # Добавляем линию тренда
            if len(x_values) > 1:
                z = np.polyfit(x_values, y_values, 1)
                p = np.poly1d(z)
                ax.plot(np.array([min(x_values), max(x_values)]), 
                      p(np.array([min(x_values), max(x_values)])), 
                      "r--", alpha=0.8)
                
                # Рассчитываем и отображаем коэффициент корреляции
                corr_coef = np.corrcoef(x_values, y_values)[0, 1]
                ax.text(0.05, 0.95, f"r = {corr_coef:.2f}",
                      transform=ax.transAxes, fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Настраиваем оси
            ax.set_xlabel(param_labels.get(x_param, x_param), fontsize=12)
            ax.set_ylabel(param_labels.get(y_param, y_param), fontsize=12)
            
            # Добавляем сетку
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Устанавливаем заголовок
        ax.set_title(f"Корреляция между {param_labels.get(x_param, x_param)} и {param_labels.get(y_param, y_param)}")
        
        # Настраиваем расположение
        plt.tight_layout()
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при создании диаграммы корреляции: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def plot_parameter_histogram(cyclones: List[Cyclone],
                          parameter: str,
                          bins: int = 20,
                          kde: bool = True,
                          color: str = 'blue',
                          figsize: Tuple[float, float] = (10, 6)) -> Tuple[Figure, Axes]:
    """
    Создает гистограмму распределения параметра циклонов.
    
    Аргументы:
        cyclones: Список циклонов для анализа.
        parameter: Параметр для анализа.
        bins: Количество бинов для гистограммы.
        kde: Добавить оценку плотности распределения.
        color: Цвет гистограммы.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        
    Возвращает:
        Кортеж (figure, axis) с созданной гистограммой.
    """
    try:
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=figsize)
        
        # Словарь с метками параметров
        param_labels = {
            'central_pressure': 'Центральное давление (гПа)',
            'vorticity_850hPa': 'Завихренность на 850 гПа (10⁻⁵ с⁻¹)',
            'max_wind_speed': 'Максимальная скорость ветра (м/с)',
            'radius': 'Радиус циклона (км)',
            'pressure_gradient': 'Градиент давления (гПа/100км)',
            'temperature_anomaly': 'Аномалия температуры (K)',
            'thickness_500_850': 'Толщина слоя 500-850 гПа (м)',
            'age': 'Возраст циклона (ч)',
            'intensity_index': 'Индекс интенсивности'
        }
        
        # Извлекаем значения параметра
        values = []
        
        for cyclone in cyclones:
            value = None
            if parameter == 'central_pressure':
                value = cyclone.central_pressure
            elif parameter == 'age':
                value = cyclone.age
            elif parameter == 'intensity_index':
                if hasattr(cyclone, 'calculate_intensity_index'):
                    value = cyclone.calculate_intensity_index()
            elif hasattr(cyclone.parameters, parameter):
                value = getattr(cyclone.parameters, parameter)
            
            if value is not None:
                # Преобразуем завихренность в 10⁻⁵ с⁻¹ для читаемости
                if parameter == 'vorticity_850hPa':
                    value *= 1e5
                values.append(value)
        
        # Проверяем, что есть данные для визуализации
        if len(values) == 0:
            ax.text(0.5, 0.5, "Недостаточно данных для визуализации", 
                  ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # Определяем диапазон данных
            if parameter == 'central_pressure':
                # Для давления задаем диапазон примерно 950-1020 гПа
                x_range = (max(950, min(values) - 5), min(1020, max(values) + 5))
                binrange = np.linspace(x_range[0], x_range[1], bins + 1)
            else:
                # Для других параметров - от 0 до максимального значения с запасом
                x_range = (0, max(values) * 1.1)
                binrange = np.linspace(x_range[0], x_range[1], bins + 1)
            
            # Создаем гистограмму
            sns.histplot(values, bins=binrange, kde=kde, color=color, alpha=0.7, ax=ax)
            
            # Добавляем вертикальную линию для среднего значения
            mean_value = np.mean(values)
            median_value = np.median(values)
            
            ax.axvline(x=mean_value, color='red', linestyle='-', linewidth=2, 
                      label=f'Среднее: {mean_value:.2f}')
            ax.axvline(x=median_value, color='green', linestyle='--', linewidth=2, 
                      label=f'Медиана: {median_value:.2f}')
            
            # Добавляем статистические показатели
            stats_text = (
                f"Среднее: {mean_value:.2f}\n"
                f"Медиана: {median_value:.2f}\n"
                f"СКО: {np.std(values):.2f}\n"
                f"Мин: {min(values):.2f}\n"
                f"Макс: {max(values):.2f}\n"
                f"Количество: {len(values)}"
            )
            
            # Размещаем блок статистики
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                  va='top', ha='right',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Настраиваем оси
            ax.set_xlabel(param_labels.get(parameter, parameter), fontsize=12)
            ax.set_ylabel('Количество циклонов', fontsize=12)
            
            # Настраиваем подписи осей
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Добавляем сетку
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Добавляем легенду
            ax.legend()
        
        # Устанавливаем заголовок
        ax.set_title(f"Распределение {param_labels.get(parameter, parameter)}")
        
        # Настраиваем расположение
        plt.tight_layout()
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при создании гистограммы параметра: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def plot_parameter_evolution(cyclone_track: List[Cyclone],
                          parameters: List[str] = None,
                          figsize: Tuple[float, float] = (12, 8)) -> Tuple[Figure, List[Axes]]:
    """
    Создает график эволюции параметров циклона во времени.
    
    Аргументы:
        cyclone_track: Список циклонов, представляющих трек (отсортированный по времени).
        parameters: Список параметров для отображения. Если None, отображаются стандартные параметры.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        
    Возвращает:
        Кортеж (figure, list_of_axes) с созданными графиками.
    """
    try:
        # Определяем стандартные параметры, если не указаны
        if parameters is None:
            parameters = ['central_pressure', 'vorticity_850hPa', 'max_wind_speed']
        
        # Словарь с метками параметров и единицами измерения
        param_labels = {
            'central_pressure': 'Центральное давление',
            'vorticity_850hPa': 'Завихренность на 850 гПа',
            'max_wind_speed': 'Максимальная скорость ветра',
            'radius': 'Радиус циклона',
            'pressure_gradient': 'Градиент давления',
            'temperature_anomaly': 'Аномалия температуры',
            'thickness_500_850': 'Толщина слоя 500-850 гПа'
        }
        param_units = {
            'central_pressure': 'гПа',
            'vorticity_850hPa': '10⁻⁵ с⁻¹',
            'max_wind_speed': 'м/с',
            'radius': 'км',
            'pressure_gradient': 'гПа/100км',
            'temperature_anomaly': 'K',
            'thickness_500_850': 'м'
        }
        
        # Сортируем циклоны по времени
        cyclones_sorted = sorted(cyclone_track, key=lambda c: c.time)
        
        if len(cyclones_sorted) < 2:
            raise ValueError("Недостаточно точек для построения эволюции параметров")
        
        # Создаем временной массив
        times = [c.time for c in cyclones_sorted]
        time_hours = [(t - times[0]).total_seconds() / 3600 for t in times]
        
        # Создаем фигуру
        fig, axes = plt.subplots(len(parameters), 1, figsize=figsize, sharex=True)
        
        # Если только один параметр, оборачиваем оси в список
        if len(parameters) == 1:
            axes = [axes]
        
        # Отображаем график для каждого параметра
        for i, param in enumerate(parameters):
            ax = axes[i]
            
            # Извлекаем значения параметра
            values = []
            for cyclone in cyclones_sorted:
                value = None
                if param == 'central_pressure':
                    value = cyclone.central_pressure
                elif hasattr(cyclone.parameters, param):
                    value = getattr(cyclone.parameters, param)
                
                if value is not None:
                    # Преобразуем завихренность в 10⁻⁵ с⁻¹ для читаемости
                    if param == 'vorticity_850hPa':
                        value *= 1e5
                    values.append(value)
                else:
                    values.append(np.nan)
            
            # Отображаем график
            line = ax.plot(time_hours, values, marker='o', linestyle='-', 
                         color='blue', markersize=8, markerfacecolor='white')
            
            # Настраиваем оси
            ax.set_ylabel(f"{param_labels.get(param, param)}\n({param_units.get(param, '')})", 
                        fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Выделяем точки, соответствующие стадиям жизненного цикла
            stage_colors = {
                CycloneLifeStage.GENESIS: 'skyblue',
                CycloneLifeStage.INTENSIFICATION: 'blue',
                CycloneLifeStage.MATURE: 'darkblue',
                CycloneLifeStage.DISSIPATION: 'lightblue',
                CycloneLifeStage.UNKNOWN: 'gray'
            }
            
            for j, cyclone in enumerate(cyclones_sorted):
                if j < len(time_hours) and j < len(values):
                    stage = cyclone.life_stage
                    ax.plot(time_hours[j], values[j], 'o', 
                          color=stage_colors.get(stage, 'gray'), 
                          markersize=10)
            
            # Добавляем аннотации для экстремальных значений
            if not all(np.isnan(values)):
                max_idx = np.nanargmax(values)
                min_idx = np.nanargmin(values)
                
                if param == 'central_pressure':
                    # Для давления минимум - наиболее важная точка
                    ax.annotate(f"{values[min_idx]:.1f}",
                              xy=(time_hours[min_idx], values[min_idx]),
                              xytext=(10, -20), textcoords='offset points',
                              arrowprops=dict(arrowstyle="->", color='red'),
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                else:
                    # Для других параметров максимум - наиболее важная точка
                    ax.annotate(f"{values[max_idx]:.1f}",
                              xy=(time_hours[max_idx], values[max_idx]),
                              xytext=(10, 20), textcoords='offset points',
                              arrowprops=dict(arrowstyle="->", color='red'),
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Добавляем метки стадий жизненного цикла на нижнюю ось
        ax = axes[-1]
        ax.set_xlabel("Время с момента обнаружения (ч)", fontsize=12)
        
        # Отмечаем стадии жизненного цикла
        stage_changes = [(0, cyclones_sorted[0].life_stage)]
        
        for j in range(1, len(cyclones_sorted)):
            if cyclones_sorted[j].life_stage != cyclones_sorted[j-1].life_stage:
                stage_changes.append((time_hours[j], cyclones_sorted[j].life_stage))
        
        # Добавляем цветовые области для стадий
        for k in range(len(stage_changes)-1):
            start_time = stage_changes[k][0]
            end_time = stage_changes[k+1][0]
            stage = stage_changes[k][1]
            
            ax.axvspan(start_time, end_time, alpha=0.2,
                     color=stage_colors.get(stage, 'gray'))
            
            # Добавляем метку стадии
            mid_time = (start_time + end_time) / 2
            ax.text(mid_time, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                  stage.name, ha='center', va='top', fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
        
        # Если есть только одна стадия
        if len(stage_changes) == 1:
            stage = stage_changes[0][1]
            ax.axvspan(time_hours[0], time_hours[-1], alpha=0.2,
                     color=stage_colors.get(stage, 'gray'))
            ax.text(np.mean(time_hours), ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                  stage.name, ha='center', va='top', fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
        
        # Устанавливаем общий заголовок
        if hasattr(cyclones_sorted[0], 'track_id') and cyclones_sorted[0].track_id:
            track_id = cyclones_sorted[0].track_id
            start_time = cyclones_sorted[0].time.strftime('%Y-%m-%d %H:%M')
            end_time = cyclones_sorted[-1].time.strftime('%Y-%m-%d %H:%M')
            duration = time_hours[-1]
            
            fig.suptitle(
                f"Эволюция циклона (ID: {track_id})\n"
                f"Период: {start_time} - {end_time} ({duration:.1f} ч)",
                fontsize=14
            )
        else:
            fig.suptitle("Эволюция параметров циклона", fontsize=14)
        
        # Настраиваем расположение
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig, axes
        
    except Exception as e:
        error_msg = f"Ошибка при визуализации эволюции параметров: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def create_pressure_profile(cyclone: Cyclone,
                         dataset: Optional[xr.Dataset] = None,
                         radius_km: float = 500.0,
                         num_points: int = 100,
                         directions: int = 8,
                         figsize: Tuple[float, float] = (12, 10)) -> Tuple[Figure, List[Axes]]:
    """
    Создает визуализацию профиля давления вокруг циклона.
    
    Аргументы:
        cyclone: Циклон для анализа.
        dataset: Набор данных с полем давления. Если None, используется dataset из циклона.
        radius_km: Радиус профиля в километрах.
        num_points: Количество точек для расчета профиля.
        directions: Количество направлений для расчета профиля.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        
    Возвращает:
        Кортеж (figure, [map_ax, profile_ax]) с картой и графиком профиля.
    """
    try:
        # Получаем набор данных
        if dataset is None and hasattr(cyclone, 'dataset'):
            dataset = cyclone.dataset
        
        if dataset is None:
            raise ValueError("Необходим набор данных для создания профиля давления")
        
        # Проверяем наличие поля давления
        pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
        pressure_var = None
        
        for var in pressure_vars:
            if var in dataset:
                pressure_var = var
                break
        
        if pressure_var is None:
            raise ValueError("В наборе данных отсутствует поле давления")
        
        # Создаем фигуру с двумя графиками
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
        map_ax = fig.add_subplot(gs[0], projection=ccrs.NorthPolarStereo())
        profile_ax = fig.add_subplot(gs[1])
        
        # Настраиваем карту
        map_ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
        map_ax.coastlines(resolution='50m')
        map_ax.gridlines()
        
        # Преобразуем радиус в градусы (приблизительно)
        radius_deg = radius_km / 111.0  # 1 градус ≈ 111 км
        
        # Вычисляем координаты центра и создаем круг для визуализации
        center_lat = cyclone.latitude
        center_lon = cyclone.longitude
        
        # Создаем круг для отображения области профиля
        theta = np.linspace(0, 2*np.pi, 100)
        circle_lats = center_lat + radius_deg * np.cos(theta)
        circle_lons = center_lon + radius_deg * np.sin(theta) / np.cos(np.radians(center_lat))
        
        map_ax.plot(circle_lons, circle_lats, color='blue', linestyle='--', 
                  transform=ccrs.PlateCarree())
        
        # Отображаем центр циклона
        map_ax.scatter(center_lon, center_lat, color='red', s=100, 
                     transform=ccrs.PlateCarree(), zorder=10)
        
        # Отображаем поле давления
        pressure_field = dataset[pressure_var]
        
        # Определяем диапазон данных для отображения
        vmin = max(950, float(pressure_field.min().values))
        vmax = min(1020, float(pressure_field.max().values))
        
        # Отображаем поле как контурную карту
        contour = map_ax.contourf(
            dataset.longitude, dataset.latitude, pressure_field,
            transform=ccrs.PlateCarree(),
            levels=np.linspace(vmin, vmax, 21),
            cmap='coolwarm_r',
            extend='both'
        )
        
        cbar = fig.colorbar(contour, ax=map_ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Давление на уровне моря (гПа)')
        
        # Отображаем контурные линии
        contour_lines = map_ax.contour(
            dataset.longitude, dataset.latitude, pressure_field,
            transform=ccrs.PlateCarree(),
            levels=np.linspace(vmin, vmax, 11),
            colors='black',
            linewidths=0.5
        )
        
        # Подписываем контурные линии
        map_ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.0f')
        
        # Создаем профили для разных направлений
        angles = np.linspace(0, 2*np.pi, directions, endpoint=False)
        angle_labels = ['Восток', 'Северо-восток', 'Север', 'Северо-запад', 
                       'Запад', 'Юго-запад', 'Юг', 'Юго-восток']
        
        # Цвета для разных направлений
        colors = plt.cm.tab10(np.linspace(0, 1, directions))
        
        # Рассчитываем и отображаем профили
        distances = np.linspace(0, radius_km, num_points)
        
        # Рассчитываем координаты для каждого направления и расстояния
        for i, angle in enumerate(angles):
            # Рассчитываем координаты
            lats = center_lat + distances * np.cos(angle) / 111.0
            # Учитываем сужение долготных кругов с широтой
            lons = center_lon + distances * np.sin(angle) / (111.0 * np.cos(np.radians(center_lat)))
            
            # Извлекаем давление в этих точках
            pressures = []
            for j in range(len(distances)):
                try:
                    # Находим ближайшую точку сетки
                    pressure = float(pressure_field.sel(
                        latitude=lats[j], longitude=lons[j], method='nearest').values)
                    pressures.append(pressure)
                except:
                    pressures.append(np.nan)
            
            # Отображаем профиль
            profile_ax.plot(distances, pressures, color=colors[i], 
                          label=angle_labels[i % len(angle_labels)], 
                          marker='o', markersize=4)
            
            # Отображаем линию на карте
            map_ax.plot(lons, lats, color=colors[i], linewidth=1.5,
                      transform=ccrs.PlateCarree())
        
        # Добавляем сетку и легенду на график профиля
        profile_ax.grid(True, linestyle='--', alpha=0.7)
        profile_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                        ncol=4, fontsize=10)
        
        # Настраиваем оси графика профиля
        profile_ax.set_xlabel('Расстояние от центра циклона (км)', fontsize=12)
        profile_ax.set_ylabel('Давление (гПа)', fontsize=12)
        profile_ax.set_xlim(0, radius_km)
        
        # Инвертируем ось Y для давления (ниже давление - интенсивнее циклон)
        y_min = max(950, min(pressures) - 5)
        y_max = min(1020, max(pressures) + 5)
        profile_ax.set_ylim(y_max, y_min)
        
        # Добавляем заголовки
        map_ax.set_title(f"Поле давления вокруг циклона", fontsize=12)
        profile_ax.set_title(f"Профили давления в разных направлениях", fontsize=12)
        
        # Общий заголовок
        fig.suptitle(
            f"Профиль давления циклона (ID: {cyclone.track_id}, "
            f"Дата: {cyclone.time.strftime('%Y-%m-%d %H:%M')})",
            fontsize=14, y=0.98
        )
        
        # Настраиваем расположение
        fig.tight_layout()
        
        return fig, [map_ax, profile_ax]
        
    except Exception as e:
        error_msg = f"Ошибка при создании профиля давления: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def create_wind_profile(cyclone: Cyclone,
                      dataset: Optional[xr.Dataset] = None,
                      pressure_level: int = 850,
                      radius_km: float = 500.0,
                      figsize: Tuple[float, float] = (12, 10)) -> Tuple[Figure, Axes]:
    """
    Создает визуализацию поля ветра вокруг циклона.
    
    Аргументы:
        cyclone: Циклон для анализа.
        dataset: Набор данных с полем ветра. Если None, используется dataset из циклона.
        pressure_level: Уровень давления для анализа ветра (гПа).
        radius_km: Радиус профиля в километрах.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        
    Возвращает:
        Кортеж (figure, axes) с картой ветра.
    """
    try:
        # Получаем набор данных
        if dataset is None and hasattr(cyclone, 'dataset'):
            dataset = cyclone.dataset
        
        if dataset is None:
            raise ValueError("Необходим набор данных для визуализации поля ветра")
        
        # Проверяем наличие компонентов ветра
        u_vars = ['u_component_of_wind', 'u', 'uwnd']
        v_vars = ['v_component_of_wind', 'v', 'vwnd']
        
        u_var, v_var = None, None
        
        for u, v in zip(u_vars, v_vars):
            if u in dataset and v in dataset:
                u_var, v_var = u, v
                break
        
        if u_var is None or v_var is None:
            raise ValueError("В наборе данных отсутствуют компоненты ветра")
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.NorthPolarStereo()})
        
        # Настраиваем карту
        center_lat = cyclone.latitude
        center_lon = cyclone.longitude
        
        # Определяем границы региона вокруг циклона
        radius_deg = radius_km / 111.0  # 1 градус ≈ 111 км
        
        # Устанавливаем экстент
        lon_min = center_lon - radius_deg / np.cos(np.radians(center_lat))
        lon_max = center_lon + radius_deg / np.cos(np.radians(center_lat))
        lat_min = max(60, center_lat - radius_deg)
        lat_max = min(90, center_lat + radius_deg)
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
        ax.coastlines(resolution='50m')
        ax.gridlines()
        
        # Выбираем данные о ветре на нужном уровне
        if 'level' in dataset.dims and pressure_level:
            # Если есть уровни давления, выбираем ближайший к запрошенному
            if hasattr(dataset, 'level'):
                levels = dataset.level.values
                closest_level = levels[np.abs(levels - pressure_level).argmin()]
                
                if closest_level != pressure_level:
                    logger.warning(f"Уровень {pressure_level} гПа недоступен, используется ближайший: {closest_level} гПа")
                
                u_wind = dataset[u_var].sel(level=closest_level)
                v_wind = dataset[v_var].sel(level=closest_level)
                level_str = f" на уровне {closest_level} гПа"
            else:
                # Если нет атрибута level, пробуем искать по имени координаты
                for level_name in ['level', 'lev', 'plev', 'pressure_level']:
                    if level_name in dataset.dims:
                        levels = dataset[level_name].values
                        closest_level = levels[np.abs(levels - pressure_level).argmin()]
                        
                        u_wind = dataset[u_var].sel({level_name: closest_level})
                        v_wind = dataset[v_var].sel({level_name: closest_level})
                        level_str = f" на уровне {closest_level} гПа"
                        break
                else:
                    # Если не нашли ни одного подходящего имени
                    u_wind = dataset[u_var]
                    v_wind = dataset[v_var]
                    level_str = ""
        else:
            # Если нет уровней, используем данные как есть
            u_wind = dataset[u_var]
            v_wind = dataset[v_var]
            level_str = ""
        
        # Рассчитываем скорость ветра
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        
        # Определяем диапазон данных для отображения
        vmin = 0
        vmax = min(40, float(wind_speed.max().values))
        
        # Отображаем скорость ветра как заливку
        contour = ax.contourf(
            dataset.longitude, dataset.latitude, wind_speed,
            transform=ccrs.PlateCarree(),
            levels=np.linspace(vmin, vmax, 21),
            cmap='viridis',
            extend='max'
        )
        
        cbar = fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Скорость ветра (м/с)')
        
        # Отображаем векторы ветра
        stride = 5  # Шаг для векторов (чтобы не перегружать карту)
        quiver = ax.quiver(
            dataset.longitude.values[::stride], dataset.latitude.values[::stride],
            u_wind.values[::stride, ::stride], v_wind.values[::stride, ::stride],
            transform=ccrs.PlateCarree(),
            scale=700,
            color='white',
            width=0.002,
            headwidth=4,
            headlength=4
        )
        
        # Добавляем легенду для векторов
        ax.quiverkey(quiver, 0.9, 0.95, 10, '10 м/с', labelpos='E',
                    coordinates='axes', color='white', fontproperties={'size': 10})
        
        # Отображаем центр циклона
        ax.scatter(center_lon, center_lat, color='red', s=100,
                 transform=ccrs.PlateCarree(), zorder=10,
                 edgecolor='black', linewidth=1)
        
        # Создаем круг для отображения области радиуса циклона (если известен)
        if hasattr(cyclone.parameters, 'radius') and cyclone.parameters.radius:
            cyclone_radius_deg = cyclone.parameters.radius / 111.0
            theta = np.linspace(0, 2*np.pi, 100)
            circle_lats = center_lat + cyclone_radius_deg * np.cos(theta)
            circle_lons = center_lon + cyclone_radius_deg * np.sin(theta) / np.cos(np.radians(center_lat))
            
            ax.plot(circle_lons, circle_lats, color='red', linestyle='-',
                  transform=ccrs.PlateCarree(), linewidth=2, alpha=0.7)
        
        # Добавляем заголовок
        if hasattr(cyclone, 'track_id') and cyclone.track_id:
            title = (
                f"Поле ветра{level_str} вокруг циклона\n"
                f"ID: {cyclone.track_id}, Дата: {cyclone.time.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            title = f"Поле ветра{level_str} вокруг циклона ({cyclone.time.strftime('%Y-%m-%d %H:%M')})"
            
        ax.set_title(title, fontsize=14)
        
        # Настраиваем расположение
        plt.tight_layout()
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при создании визуализации поля ветра: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)