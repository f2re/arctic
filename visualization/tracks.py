"""
Модуль визуализации треков циклонов для системы ArcticCyclone.

Предоставляет функции для отображения треков арктических циклонов,
их перемещения и эволюции во времени.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
import matplotlib.patches as mpatches
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import logging
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from datetime import datetime, timedelta
import pandas as pd
from matplotlib.collections import LineCollection

from models.cyclone import Cyclone
from models.classifications import CycloneType, CycloneIntensity, CycloneLifeStage
from core.exceptions import VisualizationError
from .mappers import create_arctic_map, plot_cyclone_centers, save_figure

# Инициализация логгера
logger = logging.getLogger(__name__)


def plot_cyclone_track(cyclone_track: List[Cyclone],
                     show_points: bool = True,
                     color_by: str = 'time',
                     min_latitude: float = 60.0,
                     figsize: Tuple[float, float] = (10, 8),
                     projection: str = 'NorthPolarStereo',
                     add_map_features: bool = True,
                     add_legend: bool = True) -> Tuple[Figure, Axes]:
    """
    Визуализирует трек циклона на карте.
    
    Аргументы:
        cyclone_track: Список циклонов, представляющих трек (отсортированный по времени).
        show_points: Отображать ли точки трека.
        color_by: Параметр для цветового кодирования ('time', 'pressure', 'intensity', 'type').
        min_latitude: Минимальная широта для отображения.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип картографической проекции.
        add_map_features: Добавлять ли географические объекты (береговые линии, границы и т.д.).
        add_legend: Добавлять ли легенду на карту.
        
    Возвращает:
        Кортеж (figure, axis) с созданной картой.
    """
    try:
        # Сортируем циклоны по времени
        cyclones_sorted = sorted(cyclone_track, key=lambda c: c.time)
        
        if len(cyclones_sorted) < 2:
            raise ValueError("Недостаточно точек для построения трека")
        
        # Создаем базовую карту
        fig, ax = create_arctic_map(
            min_latitude=min_latitude, 
            figsize=figsize,
            projection=projection
        )
        
        # Извлекаем координаты и характеристики
        lats = [c.latitude for c in cyclones_sorted]
        lons = [c.longitude for c in cyclones_sorted]
        pressures = [c.central_pressure for c in cyclones_sorted]
        times = [c.time for c in cyclones_sorted]
        
        # Настраиваем цветовую схему
        if color_by == 'time':
            # Нормализуем время для цветовой карты
            time_diffs = [(t - times[0]).total_seconds() / 3600 for t in times]
            norm = plt.Normalize(0, max(time_diffs))
            cmap = plt.cm.plasma
            colors = [cmap(norm(td)) for td in time_diffs]
            
            # Создаем цветную линию
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors, transform=ccrs.PlateCarree())
            line = ax.add_collection(lc)
            
            # Добавляем цветовую шкалу
            if add_legend:
                # Создаем искусственные данные для colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                
                # Преобразуем часы в более читаемый формат
                hours_max = max(time_diffs)
                if hours_max > 72:
                    # Дни
                    days_max = hours_max / 24
                    cbar.set_label('Время с начала трека (дни)')
                    ticks = np.linspace(0, hours_max, 5)
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels([f"{t/24:.1f}" for t in ticks])
                else:
                    # Часы
                    cbar.set_label('Время с начала трека (часы)')
        
        elif color_by == 'pressure':
            # Нормализуем давление для цветовой карты
            norm = plt.Normalize(min(950, min(pressures) - 5), max(1020, max(pressures) + 5))
            cmap = plt.cm.coolwarm_r  # Инвертированная, так как низкое давление = сильный циклон
            colors = [cmap(norm(p)) for p in pressures]
            
            # Создаем цветную линию
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors, transform=ccrs.PlateCarree())
            line = ax.add_collection(lc)
            
            # Добавляем цветовую шкалу
            if add_legend:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
                cbar.set_label('Центральное давление (гПа)')
        
        elif color_by == 'intensity':
            # Определяем интенсивность каждого циклона
            intensity_values = [c.calculate_intensity() for c in cyclones_sorted]
            intensity_codes = [intensity.value for intensity in intensity_values]
            
            # Определяем цвета для каждой категории интенсивности
            intensity_colors = {
                'weak': 'green',
                'moderate': 'yellow',
                'strong': 'orange',
                'very_strong': 'red'
            }
            
            line_colors = []
            
            # Создаем отдельные линии для каждого сегмента с одинаковой интенсивностью
            current_intensity = intensity_codes[0]
            current_segment_lats = [lats[0]]
            current_segment_lons = [lons[0]]
            
            for i in range(1, len(cyclones_sorted)):
                if intensity_codes[i] == current_intensity:
                    # Продолжаем текущий сегмент
                    current_segment_lats.append(lats[i])
                    current_segment_lons.append(lons[i])
                else:
                    # Завершаем текущий сегмент и рисуем его
                    ax.plot(current_segment_lons, current_segment_lats, 
                          color=intensity_colors.get(current_intensity, 'blue'),
                          transform=ccrs.PlateCarree(), linewidth=2)
                    
                    # Начинаем новый сегмент
                    current_intensity = intensity_codes[i]
                    current_segment_lats = [lats[i-1], lats[i]]
                    current_segment_lons = [lons[i-1], lons[i]]
                    
            # Добавляем последний сегмент
            ax.plot(current_segment_lons, current_segment_lats, 
                  color=intensity_colors.get(current_intensity, 'blue'),
                  transform=ccrs.PlateCarree(), linewidth=2)
            
            # Добавляем легенду
            if add_legend:
                legend_elements = [
                    mpatches.Patch(color=color, label=intensity)
                    for intensity, color in intensity_colors.items()
                ]
                ax.legend(handles=legend_elements, loc='lower right', 
                        title='Интенсивность циклона')
                
        elif color_by == 'type':
            # Определяем тип каждого циклона
            type_values = []
            for c in cyclones_sorted:
                if hasattr(c.parameters, 'thermal_type'):
                    type_values.append(c.parameters.thermal_type)
                else:
                    type_values.append(CycloneType.UNCLASSIFIED)
            
            type_codes = [type_val.value for type_val in type_values]
            
            # Определяем цвета для каждого типа
            type_colors = {
                'cold_core': 'blue',
                'warm_core': 'red',
                'hybrid': 'purple',
                'unclassified': 'gray'
            }
            
            # Создаем отдельные линии для каждого сегмента с одинаковым типом
            current_type = type_codes[0]
            current_segment_lats = [lats[0]]
            current_segment_lons = [lons[0]]
            
            for i in range(1, len(cyclones_sorted)):
                if type_codes[i] == current_type:
                    # Продолжаем текущий сегмент
                    current_segment_lats.append(lats[i])
                    current_segment_lons.append(lons[i])
                else:
                    # Завершаем текущий сегмент и рисуем его
                    ax.plot(current_segment_lons, current_segment_lats, 
                          color=type_colors.get(current_type, 'gray'),
                          transform=ccrs.PlateCarree(), linewidth=2)
                    
                    # Начинаем новый сегмент
                    current_type = type_codes[i]
                    current_segment_lats = [lats[i-1], lats[i]]
                    current_segment_lons = [lons[i-1], lons[i]]
                    
            # Добавляем последний сегмент
            ax.plot(current_segment_lons, current_segment_lats, 
                  color=type_colors.get(current_type, 'gray'),
                  transform=ccrs.PlateCarree(), linewidth=2)
            
            # Добавляем легенду
            if add_legend:
                legend_elements = [
                    mpatches.Patch(color=color, label=type_name)
                    for type_name, color in type_colors.items()
                ]
                ax.legend(handles=legend_elements, loc='lower right', 
                        title='Тип циклона')
                
        else:
            # По умолчанию - сплошная линия
            ax.plot(lons, lats, color='blue', transform=ccrs.PlateCarree(), linewidth=2)
        
        # Отображаем точки, если требуется
        if show_points:
            for i, cyclone in enumerate(cyclones_sorted):
                marker_size = 30
                marker_color = 'blue'
                marker_edge_color = 'white'
                
                # Выделяем начальную и конечную точки
                if i == 0:  # Начальная точка
                    marker_size = 100
                    marker_color = 'green'
                    marker_edge_color = 'black'
                elif i == len(cyclones_sorted) - 1:  # Конечная точка
                    marker_size = 100
                    marker_color = 'red'
                    marker_edge_color = 'black'
                
                ax.scatter(cyclone.longitude, cyclone.latitude, 
                         s=marker_size, color=marker_color, edgecolor=marker_edge_color,
                         transform=ccrs.PlateCarree(), zorder=10, linewidth=1)
        
        # Добавляем информацию о треке
        if hasattr(cyclones_sorted[0], 'track_id') and cyclones_sorted[0].track_id:
            track_id = cyclones_sorted[0].track_id
            start_time = cyclones_sorted[0].time.strftime('%Y-%m-%d %H:%M')
            end_time = cyclones_sorted[-1].time.strftime('%Y-%m-%d %H:%M')
            duration = (cyclones_sorted[-1].time - cyclones_sorted[0].time).total_seconds() / 3600
            
            # Размещаем информацию в левом верхнем углу карты
            track_info = (
                f"Трек: {track_id}\n"
                f"Начало: {start_time}\n"
                f"Конец: {end_time}\n"
                f"Длительность: {duration:.1f} ч"
            )
            
            ax.text(0.02, 0.98, track_info, transform=ax.transAxes, fontsize=10,
                  va='top', ha='left',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Устанавливаем заголовок
        ax.set_title('Трек арктического циклона', fontsize=14)
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при визуализации трека циклона: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def plot_multiple_tracks(tracks: List[List[Cyclone]],
                       min_latitude: float = 60.0,
                       figsize: Tuple[float, float] = (12, 10),
                       projection: str = 'NorthPolarStereo',
                       color_method: str = 'unique',
                       max_tracks: int = 50,
                       alpha: float = 0.7,
                       line_width: float = 1.5,
                       show_points: bool = False) -> Tuple[Figure, Axes]:
    """
    Визуализирует несколько треков циклонов на одной карте.
    
    Аргументы:
        tracks: Список треков циклонов, где каждый трек - список циклонов.
        min_latitude: Минимальная широта для отображения.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип картографической проекции.
        color_method: Метод раскраски треков ('unique', 'pressure', 'intensity', 'season').
        max_tracks: Максимальное количество треков для отображения.
        alpha: Прозрачность линий треков.
        line_width: Ширина линий треков.
        show_points: Отображать ли точки начала и конца треков.
        
    Возвращает:
        Кортеж (figure, axis) с созданной картой.
    """
    try:
        # Создаем базовую карту
        fig, ax = create_arctic_map(
            min_latitude=min_latitude, 
            figsize=figsize,
            projection=projection
        )
        
        # Ограничиваем количество треков для отображения
        if len(tracks) > max_tracks:
            logger.warning(f"Слишком много треков ({len(tracks)}), "
                         f"отображение ограничено до {max_tracks}")
            tracks = tracks[:max_tracks]
        
        # Определяем цвета для треков
        if color_method == 'unique':
            # Уникальный цвет для каждого трека
            colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(tracks))))
            # Если треков больше 20, повторяем цвета
            if len(tracks) > 20:
                colors = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, min(20, len(tracks) - 20)))])
            if len(tracks) > 40:
                colors = np.vstack([colors, plt.cm.tab20c(np.linspace(0, 1, min(20, len(tracks) - 40)))])
        elif color_method == 'season':
            # Определяем сезон для каждого трека
            seasons = []
            for track in tracks:
                if track:
                    # Берем время первой точки трека
                    month = sorted(track, key=lambda c: c.time)[0].time.month
                    if 3 <= month <= 5:
                        seasons.append('spring')
                    elif 6 <= month <= 8:
                        seasons.append('summer')
                    elif 9 <= month <= 11:
                        seasons.append('autumn')
                    else:
                        seasons.append('winter')
                else:
                    seasons.append('unknown')
            
            # Цвета для сезонов
            season_colors = {
                'winter': 'blue',
                'spring': 'green',
                'summer': 'red',
                'autumn': 'orange',
                'unknown': 'gray'
            }
            
            # Создаем легенду
            legend_elements = [
                mpatches.Patch(color=color, label=season)
                for season, color in season_colors.items()
                if season in seasons
            ]
            ax.legend(handles=legend_elements, loc='lower right', 
                    title='Сезон')
        
        # Отображаем каждый трек
        for i, track in enumerate(tracks):
            if not track:
                continue
                
            # Сортируем точки трека по времени
            track_sorted = sorted(track, key=lambda c: c.time)
            
            # Извлекаем координаты и характеристики
            lats = [c.latitude for c in track_sorted]
            lons = [c.longitude for c in track_sorted]
            pressures = [c.central_pressure for c in track_sorted]
            times = [c.time for c in track_sorted]
            
            # Определяем цвет трека
            if color_method == 'unique':
                track_color = colors[i % len(colors)]
            elif color_method == 'season':
                # Берем время первой точки трека
                month = track_sorted[0].time.month
                if 3 <= month <= 5:
                    track_color = season_colors['spring']
                elif 6 <= month <= 8:
                    track_color = season_colors['summer']
                elif 9 <= month <= 11:
                    track_color = season_colors['autumn']
                else:
                    track_color = season_colors['winter']
            elif color_method == 'pressure':
                # Используем среднее давление трека для определения цвета
                mean_pressure = np.mean(pressures)
                # Нормализуем в диапазоне 950-1020 гПа
                norm = plt.Normalize(950, 1020)
                cmap = plt.cm.coolwarm_r
                track_color = cmap(norm(mean_pressure))
            elif color_method == 'intensity':
                # Используем минимальное давление трека для определения интенсивности
                min_pressure = min(pressures)
                if min_pressure < 960:
                    track_color = 'red'  # Очень сильный
                elif min_pressure < 980:
                    track_color = 'orange'  # Сильный
                elif min_pressure < 995:
                    track_color = 'yellow'  # Умеренный
                else:
                    track_color = 'green'  # Слабый
            else:
                track_color = 'blue'
            
            # Отображаем линию трека
            ax.plot(lons, lats, color=track_color, transform=ccrs.PlateCarree(),
                  linewidth=line_width, alpha=alpha)
            
            # Отображаем точки начала и конца, если требуется
            if show_points:
                # Начальная точка (зеленая)
                ax.scatter(lons[0], lats[0], s=40, color='green', 
                         transform=ccrs.PlateCarree(), zorder=10)
                
                # Конечная точка (красная)
                ax.scatter(lons[-1], lats[-1], s=40, color='red', 
                         transform=ccrs.PlateCarree(), zorder=10)
        
        # Устанавливаем заголовок
        ax.set_title(f'Треки арктических циклонов (всего {len(tracks)})', fontsize=14)
        
        # Если использовалась цветовая схема по давлению, добавляем шкалу
        if color_method == 'pressure':
            sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm_r, norm=plt.Normalize(950, 1020))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label('Среднее давление (гПа)')
        
        # Если использовалась цветовая схема по интенсивности, добавляем легенду
        if color_method == 'intensity':
            legend_elements = [
                mpatches.Patch(color='red', label='Очень сильный (<960 гПа)'),
                mpatches.Patch(color='orange', label='Сильный (960-980 гПа)'),
                mpatches.Patch(color='yellow', label='Умеренный (980-995 гПа)'),
                mpatches.Patch(color='green', label='Слабый (>995 гПа)')
            ]
            ax.legend(handles=legend_elements, loc='lower right', 
                    title='Интенсивность циклона')
        
        return fig, ax
        
    except Exception as e:
        error_msg = f"Ошибка при визуализации множественных треков: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def animate_cyclone_track(cyclone_track: List[Cyclone],
                        min_latitude: float = 60.0,
                        figsize: Tuple[float, float] = (10, 8),
                        projection: str = 'NorthPolarStereo',
                        show_pressure: bool = True,
                        fps: int = 5,
                        dpi: int = 100,
                        output_file: Optional[str] = None) -> animation.FuncAnimation:
    """
    Создает анимацию движения циклона по его треку.
    
    Аргументы:
        cyclone_track: Список циклонов, представляющих трек (отсортированный по времени).
        min_latitude: Минимальная широта для отображения.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип картографической проекции.
        show_pressure: Отображать ли давление в центре циклона.
        fps: Кадров в секунду для анимации.
        dpi: Разрешение изображения.
        output_file: Путь для сохранения анимации. Если None, анимация не сохраняется.
        
    Возвращает:
        Объект анимации.
    """
    try:
        # Сортируем циклоны по времени
        cyclones_sorted = sorted(cyclone_track, key=lambda c: c.time)
        
        if len(cyclones_sorted) < 2:
            raise ValueError("Недостаточно точек для анимации трека")
        
        # Создаем базовую карту
        fig, ax = create_arctic_map(
            min_latitude=min_latitude, 
            figsize=figsize,
            projection=projection
        )
        
        # Извлекаем координаты и характеристики
        lats = [c.latitude for c in cyclones_sorted]
        lons = [c.longitude for c in cyclones_sorted]
        pressures = [c.central_pressure for c in cyclones_sorted]
        times = [c.time for c in cyclones_sorted]
        
        # Рассчитываем диапазон давления для цветовой шкалы
        p_min = min(pressures)
        p_max = max(pressures)
        norm = plt.Normalize(p_min, p_max)
        cmap = plt.cm.coolwarm_r
        
        # Создаем линию трека
        track_line, = ax.plot([], [], color='blue', transform=ccrs.PlateCarree(), linewidth=2)
        
        # Создаем маркер для текущего положения циклона
        cyclone_marker = ax.scatter([], [], s=200, color='red', edgecolor='black',
                                  transform=ccrs.PlateCarree(), zorder=10)
        
        # Создаем текстовое поле для информации о циклоне
        info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=10,
                          va='top', ha='left',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Инициализируем кадр
        def init():
            track_line.set_data([], [])
            cyclone_marker.set_offsets(np.array([[], []]).T)
            info_text.set_text("")
            return track_line, cyclone_marker, info_text
        
        # Функция обновления для анимации
        def update(frame):
            # Обновляем линию трека до текущей точки
            track_line.set_data(lons[:frame+1], lats[:frame+1])
            
            # Обновляем положение маркера
            cyclone_marker.set_offsets(np.array([[lons[frame], lats[frame]]]))
            
            # Цвет маркера зависит от давления
            cyclone_marker.set_color(cmap(norm(pressures[frame])))
            
            # Обновляем информацию о циклоне
            time_str = times[frame].strftime('%Y-%m-%d %H:%M')
            duration = (times[frame] - times[0]).total_seconds() / 3600
            
            if show_pressure:
                info_text.set_text(
                    f"Время: {time_str}\n"
                    f"Длительность: {duration:.1f} ч\n"
                    f"Давление: {pressures[frame]:.1f} гПа"
                )
            else:
                info_text.set_text(
                    f"Время: {time_str}\n"
                    f"Длительность: {duration:.1f} ч"
                )
            
            return track_line, cyclone_marker, info_text
        
        # Создаем анимацию
        ani = animation.FuncAnimation(
            fig, update, frames=len(cyclones_sorted),
            init_func=init, blit=True, interval=1000/fps
        )
        
        # Устанавливаем заголовок
        if hasattr(cyclones_sorted[0], 'track_id') and cyclones_sorted[0].track_id:
            ax.set_title(f'Анимация движения циклона (ID: {cyclones_sorted[0].track_id})', fontsize=14)
        else:
            ax.set_title('Анимация движения циклона', fontsize=14)
        
        # Добавляем цветовую шкалу для давления
        if show_pressure:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label('Центральное давление (гПа)')
        
        # Сохраняем анимацию, если указан файл
        if output_file:
            ani.save(output_file, fps=fps, dpi=dpi)
            logger.info(f"Анимация сохранена в файл: {output_file}")
        
        return ani
        
    except Exception as e:
        error_msg = f"Ошибка при создании анимации трека: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def plot_track_parameters(cyclone_track: List[Cyclone],
                        parameters: List[str] = None,
                        map_view: bool = True,
                        min_latitude: float = 60.0,
                        figsize: Tuple[float, float] = (14, 10)) -> Tuple[Figure, List[Axes]]:
    """
    Создает комбинированную визуализацию трека циклона и его параметров.
    
    Аргументы:
        cyclone_track: Список циклонов, представляющих трек (отсортированный по времени).
        parameters: Список параметров для отображения. Если None, использует стандартный набор.
        map_view: Включать ли карту с треком.
        min_latitude: Минимальная широта для отображения на карте.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        
    Возвращает:
        Кортеж (figure, list_of_axes) с созданной визуализацией.
    """
    try:
        # Определяем стандартный набор параметров, если не указан
        if parameters is None:
            parameters = ['central_pressure', 'vorticity_850hPa', 'max_wind_speed']
        
        # Сортируем циклоны по времени
        cyclones_sorted = sorted(cyclone_track, key=lambda c: c.time)
        
        if len(cyclones_sorted) < 2:
            raise ValueError("Недостаточно точек для анализа трека")
        
        # Определяем расположение графиков
        if map_view:
            # Карта слева, графики параметров справа
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(len(parameters), 2, width_ratios=[1, 1.5])
            
            # Создаем карту
            map_ax = fig.add_subplot(gs[:, 0], projection=ccrs.NorthPolarStereo())
            map_ax.set_extent([-180, 180, min_latitude, 90], ccrs.PlateCarree())
            map_ax.coastlines(resolution='50m')
            map_ax.gridlines()
            
            # Добавляем трек на карту
            lats = [c.latitude for c in cyclones_sorted]
            lons = [c.longitude for c in cyclones_sorted]
            
            # Цветовая схема по времени
            times = [c.time for c in cyclones_sorted]
            time_diffs = [(t - times[0]).total_seconds() / 3600 for t in times]
            norm = plt.Normalize(0, max(time_diffs))
            cmap = plt.cm.plasma
            
            # Создаем цветную линию для трека
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            lc.set_array(np.array(time_diffs[:-1]))
            line = map_ax.add_collection(lc)
            
            # Отмечаем начальную и конечную точки
            map_ax.scatter(lons[0], lats[0], s=100, color='green', edgecolor='black',
                         transform=ccrs.PlateCarree(), zorder=10)
            map_ax.scatter(lons[-1], lats[-1], s=100, color='red', edgecolor='black',
                         transform=ccrs.PlateCarree(), zorder=10)
            
            # Добавляем цветовую шкалу
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=map_ax, orientation='horizontal', pad=0.05, shrink=0.8)
            
            # Преобразуем часы в более читаемый формат
            hours_max = max(time_diffs)
            if hours_max > 72:
                # Дни
                days_max = hours_max / 24
                cbar.set_label('Время с начала трека (дни)')
                ticks = np.linspace(0, hours_max, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{t/24:.1f}" for t in ticks])
            else:
                # Часы
                cbar.set_label('Время с начала трека (часы)')
            
            # Добавляем заголовок для карты
            map_ax.set_title('Трек циклона', fontsize=12)
            
            # Создаем графики параметров
            param_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(parameters))]
        else:
            # Только графики параметров
            fig, param_axes = plt.subplots(len(parameters), 1, figsize=figsize, sharex=True)
            
            # Если только один параметр, оборачиваем оси в список
            if len(parameters) == 1:
                param_axes = [param_axes]
            
            map_ax = None
        
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
        
        # Создаем временной массив
        times = [c.time for c in cyclones_sorted]
        time_hours = [(t - times[0]).total_seconds() / 3600 for t in times]
        
        # Отображаем график для каждого параметра
        for i, param in enumerate(parameters):
            ax = param_axes[i]
            
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
            line, = ax.plot(time_hours, values, marker='o', linestyle='-', 
                          color='blue', markersize=6)
            
            # Настраиваем оси
            ax.set_ylabel(f"{param_labels.get(param, param)}\n({param_units.get(param, '')})", 
                        fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
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
        
        # Настраиваем общую ось X
        if len(parameters) > 0:
            param_axes[-1].set_xlabel("Время с момента обнаружения (ч)", fontsize=12)
        
        # Устанавливаем общий заголовок
        if hasattr(cyclones_sorted[0], 'track_id') and cyclones_sorted[0].track_id:
            track_id = cyclones_sorted[0].track_id
            start_time = cyclones_sorted[0].time.strftime('%Y-%m-%d %H:%M')
            end_time = cyclones_sorted[-1].time.strftime('%Y-%m-%d %H:%M')
            duration = time_hours[-1]
            
            fig.suptitle(
                f"Параметры циклона (ID: {track_id})\n"
                f"Период: {start_time} - {end_time} ({duration:.1f} ч)",
                fontsize=14
            )
        else:
            fig.suptitle("Параметры циклона", fontsize=14)
        
        # Настраиваем расположение
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        if map_view:
            return fig, [map_ax] + param_axes
        else:
            return fig, param_axes
        
    except Exception as e:
        error_msg = f"Ошибка при визуализации параметров трека: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)


def plot_seasonal_tracks(tracks: List[List[Cyclone]],
                       min_latitude: float = 60.0,
                       figsize: Tuple[float, float] = (16, 12),
                       projection: str = 'NorthPolarStereo',
                       seasons: Dict[str, Tuple[int, int]] = None) -> Figure:
    """
    Создает карты треков циклонов для каждого сезона.
    
    Аргументы:
        tracks: Список треков циклонов, где каждый трек - список циклонов.
        min_latitude: Минимальная широта для отображения.
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип картографической проекции.
        seasons: Словарь сезонов в формате {'name': (start_month, end_month)}.
        
    Возвращает:
        Фигура с картами треков для каждого сезона.
    """
    try:
        # Определяем сезоны, если не указаны
        if seasons is None:
            seasons = {
                'Зима': (12, 2),
                'Весна': (3, 5),
                'Лето': (6, 8),
                'Осень': (9, 11)
            }
        
        # Создаем фигуру с подграфиками
        fig = plt.figure(figsize=figsize)
        n_seasons = len(seasons)
        
        # Определяем сетку для размещения подграфиков
        n_cols = 2
        n_rows = (n_seasons + 1) // 2
        
        # Группируем треки по сезонам
        seasonal_tracks = {season: [] for season in seasons}
        
        for track in tracks:
            if not track:
                continue
                
            # Берем время первой точки трека (сортируем по времени)
            track_sorted = sorted(track, key=lambda c: c.time)
            month = track_sorted[0].time.month
            
            for season, (start, end) in seasons.items():
                if start <= end:
                    # Обычный сезон (например, весна: 3-5)
                    if start <= month <= end:
                        seasonal_tracks[season].append(track)
                else:
                    # Сезон, переходящий через год (например, зима: 12-2)
                    if month >= start or month <= end:
                        seasonal_tracks[season].append(track)
        
        # Создаем карту для каждого сезона
        for i, (season, season_tracks) in enumerate(seasonal_tracks.items()):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, 
                               projection=ccrs.NorthPolarStereo())
            
            # Настраиваем карту
            ax.set_extent([-180, 180, min_latitude, 90], ccrs.PlateCarree())
            ax.coastlines(resolution='50m')
            ax.gridlines()
            
            # Отображаем треки этого сезона
            for track in season_tracks:
                track_sorted = sorted(track, key=lambda c: c.time)
                lats = [c.latitude for c in track_sorted]
                lons = [c.longitude for c in track_sorted]
                
                # Отображаем линию трека
                ax.plot(lons, lats, color='blue', transform=ccrs.PlateCarree(),
                      linewidth=1, alpha=0.7)
                
                # Отмечаем начальную и конечную точки
                ax.scatter(lons[0], lats[0], s=20, color='green', 
                         transform=ccrs.PlateCarree(), zorder=10, alpha=0.7)
                ax.scatter(lons[-1], lats[-1], s=20, color='red', 
                         transform=ccrs.PlateCarree(), zorder=10, alpha=0.7)
            
            # Устанавливаем заголовок для сезона
            ax.set_title(f"{season} ({len(season_tracks)} треков)", fontsize=12)
        
        # Добавляем общий заголовок
        fig.suptitle("Сезонное распределение треков арктических циклонов", fontsize=16, y=0.98)
        
        # Регулируем расстояние между подграфиками
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
        
    except Exception as e:
        error_msg = f"Ошибка при создании сезонной карты треков: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)




def plot_cyclone_tracks(tracks: List[List[Cyclone]], 
                       region: Optional[Dict[str, float]] = None,
                       output_file: Optional[Union[str, Path]] = None,
                       figsize: Tuple[float, float] = (12, 10),
                       show_intensity: bool = True,
                       title: Optional[str] = None) -> plt.Figure:
    """
    Plots cyclone tracks on a map of the Arctic region.
    
    Arguments:
        tracks: List of cyclone tracks, where each track is a list of Cyclone objects.
        region: Dictionary with region boundaries {'north': float, 'south': float, 
                'east': float, 'west': float}. If None, uses default Arctic region.
        output_file: Path to save the figure. If None, the figure is not saved.
        figsize: Figure size as (width, height) in inches.
        show_intensity: Whether to color tracks by cyclone intensity.
        title: Custom title for the plot. If None, a default title is used.
        
    Returns:
        Matplotlib Figure object with the plotted tracks.
    """
    try:
        # Set default region if not provided
        if region is None:
            region = {'north': 90.0, 'south': 60.0, 'east': 180.0, 'west': -180.0}
            
        # Create figure with North Polar Stereographic projection
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        
        # Set map extent to focus on Arctic region
        ax.set_extent([region['west'], region['east'], region['south'], region['north']], 
                     crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
        gl.top_labels = False
        gl.right_labels = False
        
        # Create colormap for intensity if needed
        if show_intensity:
            cmap = cm.get_cmap('coolwarm_r')
            
            # Find global min/max pressure for normalization
            all_pressures = []
            for track in tracks:
                if track:
                    all_pressures.extend([c.central_pressure for c in track])
            
            if all_pressures:
                vmin = min(all_pressures)
                vmax = max(all_pressures)
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=950, vmax=1010)
        
        # Plot each track
        for i, track in enumerate(tracks):
            if not track:
                continue
                
            # Sort track by time
            track_sorted = sorted(track, key=lambda c: c.time)
            
            # Extract coordinates
            lats = [c.latitude for c in track_sorted]
            lons = [c.longitude for c in track_sorted]
            
            if show_intensity:
                # Color by pressure
                pressures = [c.central_pressure for c in track_sorted]
                
                # Plot track segments with color based on pressure
                for j in range(len(track_sorted) - 1):
                    segment_lons = [lons[j], lons[j+1]]
                    segment_lats = [lats[j], lats[j+1]]
                    color = cmap(norm(pressures[j]))
                    ax.plot(segment_lons, segment_lats, transform=ccrs.PlateCarree(),
                           color=color, linewidth=1.5, marker='o', markersize=3)
                    
                # Add track ID label at starting point
                if hasattr(track_sorted[0], 'track_id') and track_sorted[0].track_id:
                    track_id = track_sorted[0].track_id
                else:
                    track_id = f"Track {i+1}"
                
                ax.text(lons[0], lats[0], track_id, transform=ccrs.PlateCarree(),
                       fontsize=7, ha='right', va='bottom', bbox=dict(
                           boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
            else:
                # Use a predefined set of colors
                colors_list = plt.cm.tab10.colors
                color = colors_list[i % len(colors_list)]
                
                # Plot entire track with consistent color
                ax.plot(lons, lats, transform=ccrs.PlateCarree(),
                       color=color, linewidth=1.5, marker='o', markersize=3, label=f"Track {i+1}")
                
                # Mark start and end points
                ax.plot(lons[0], lats[0], 'go', transform=ccrs.PlateCarree(), markersize=6)
                ax.plot(lons[-1], lats[-1], 'ro', transform=ccrs.PlateCarree(), markersize=6)
        
        # Add colorbar if showing intensity
        if show_intensity and all_pressures:
            cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                             orientation='horizontal', pad=0.05, aspect=40)
            cb.set_label('Central Pressure (hPa)')
        
        # Add title
        if title is None:
            title = f"Arctic Cyclone Tracks ({len(tracks)} tracks)"
        ax.set_title(title, fontsize=14)
        
        # Add timestamp
        plt.figtext(0.02, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                   fontsize=8, color='gray')
        
        # Add legend if not showing intensity
        if not show_intensity:
            ax.legend(loc='lower right')
        
        # Save figure if output file is specified
        if output_file is not None:
            output_path = Path(output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cyclone tracks plot saved to {output_path}")
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting cyclone tracks: {str(e)}")
        # Create a simple error figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error plotting tracks: {str(e)}", 
               ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        
        if output_file is not None:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
        return fig


def plot_track_evolution(track: List[Cyclone],
                        variables: List[str] = None,
                        output_file: Optional[Union[str, Path]] = None,
                        figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """
    Plots the evolution of a cyclone track over time.
    
    Arguments:
        track: List of Cyclone objects representing a single track.
        variables: List of variables to plot. If None, plots pressure and vorticity.
        output_file: Path to save the figure. If None, the figure is not saved.
        figsize: Figure size as (width, height) in inches.
        
    Returns:
        Matplotlib Figure object with the track evolution plot.
    """
    if not track:
        logger.warning("Empty track provided for plotting")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No track data available", ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return fig
    
    # Sort track by time
    track_sorted = sorted(track, key=lambda c: c.time)
    
    # Default variables to plot
    if variables is None:
        variables = ['central_pressure', 'vorticity_850hPa']
    
    # Set up multiple subplots - one for each variable
    fig, axes = plt.subplots(len(variables), 1, figsize=figsize, sharex=True)
    if len(variables) == 1:
        axes = [axes]
    
    # Format x-axis with time
    times = [c.time for c in track_sorted]
    
    for i, variable in enumerate(variables):
        ax = axes[i]
        
        if variable == 'central_pressure':
            # Extract pressure data
            values = [c.central_pressure for c in track_sorted]
            ax.plot(times, values, 'b-', marker='o', linewidth=2)
            ax.set_ylabel('Pressure (hPa)')
            ax.set_title('Central Pressure Evolution')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Invert y-axis for pressure
            ax.invert_yaxis()
            
        elif variable == 'vorticity_850hPa':
            # Extract vorticity data (if available)
            values = []
            for c in track_sorted:
                if hasattr(c.parameters, 'vorticity_850hPa') and c.parameters.vorticity_850hPa is not None:
                    values.append(c.parameters.vorticity_850hPa)
                else:
                    values.append(np.nan)
            
            # Only plot if we have valid vorticity data
            if any(~np.isnan(values)):
                ax.plot(times, values, 'r-', marker='s', linewidth=2)
                ax.set_ylabel('Vorticity (s⁻¹)')
                ax.set_title('850 hPa Vorticity Evolution')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Scientific notation for vorticity
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            else:
                ax.text(0.5, 0.5, "No vorticity data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel('Vorticity (s⁻¹)')
                ax.set_title('850 hPa Vorticity Evolution')
        
        elif variable == 'radius':
            # Extract radius data (if available)
            values = []
            for c in track_sorted:
                if hasattr(c.parameters, 'radius') and c.parameters.radius is not None:
                    values.append(c.parameters.radius)
                else:
                    values.append(np.nan)
            
            if any(~np.isnan(values)):
                ax.plot(times, values, 'g-', marker='^', linewidth=2)
                ax.set_ylabel('Radius (km)')
                ax.set_title('Cyclone Radius Evolution')
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No radius data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel('Radius (km)')
                ax.set_title('Cyclone Radius Evolution')
        
        elif variable == 'max_wind_speed':
            # Extract wind speed data (if available)
            values = []
            for c in track_sorted:
                if hasattr(c.parameters, 'max_wind_speed') and c.parameters.max_wind_speed is not None:
                    values.append(c.parameters.max_wind_speed)
                else:
                    values.append(np.nan)
            
            if any(~np.isnan(values)):
                ax.plot(times, values, 'm-', marker='*', linewidth=2)
                ax.set_ylabel('Wind Speed (m/s)')
                ax.set_title('Maximum Wind Speed Evolution')
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No wind speed data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel('Wind Speed (m/s)')
                ax.set_title('Maximum Wind Speed Evolution')
        
        else:
            # Try to extract any other parameter data
            values = []
            for c in track_sorted:
                if hasattr(c, variable):
                    values.append(getattr(c, variable))
                elif hasattr(c.parameters, variable):
                    values.append(getattr(c.parameters, variable))
                else:
                    values.append(np.nan)
            
            if any(~np.isnan(values)):
                ax.plot(times, values, 'k-', marker='d', linewidth=2)
                ax.set_ylabel(variable)
                ax.set_title(f'{variable} Evolution')
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, f"No {variable} data available", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(variable)
                ax.set_title(f'{variable} Evolution')
    
    # Format x-axis
    axes[-1].set_xlabel('Time')
    axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    # Add track information in the figure title
    track_id = getattr(track_sorted[0], 'track_id', 'Unknown')
    duration_hours = (track_sorted[-1].time - track_sorted[0].time).total_seconds() / 3600
    plt.suptitle(f"Track ID: {track_id}, Duration: {duration_hours:.1f} hours, Points: {len(track)}")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save figure if output file is specified
    if output_file is not None:
        output_path = Path(output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Track evolution plot saved to {output_path}")
    
    return fig


def plot_tracks_statistics(tracks: List[List[Cyclone]],
                          output_file: Optional[Union[str, Path]] = None,
                          figsize: Tuple[float, float] = (15, 10)) -> plt.Figure:
    """
    Plots statistical information about cyclone tracks.
    
    Arguments:
        tracks: List of cyclone tracks, where each track is a list of Cyclone objects.
        output_file: Path to save the figure. If None, the figure is not saved.
        figsize: Figure size as (width, height) in inches.
        
    Returns:
        Matplotlib Figure object with the statistics plots.
    """
    if not tracks:
        logger.warning("No tracks provided for plotting statistics")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No track data available", ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        return fig
    
    # Extract track statistics
    durations = []
    min_pressures = []
    displacements = []
    speeds = []
    deepening_rates = []
    
    for track in tracks:
        if not track:
            continue
            
        # Sort track by time
        track_sorted = sorted(track, key=lambda c: c.time)
        
        # Calculate duration
        start_time = track_sorted[0].time
        end_time = track_sorted[-1].time
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        durations.append(duration)
        
        # Find minimum pressure
        min_pressure = min(c.central_pressure for c in track_sorted)
        min_pressures.append(min_pressure)
        
        # Calculate displacement (distance between start and end)
        start_lat, start_lon = track_sorted[0].latitude, track_sorted[0].longitude
        end_lat, end_lon = track_sorted[-1].latitude, track_sorted[-1].longitude
        
        # Simple distance calculation in km using the haversine formula
        from math import radians, sin, cos, sqrt, atan2
        R = 6371  # Earth radius in km
        
        lat1, lon1 = radians(start_lat), radians(start_lon)
        lat2, lon2 = radians(end_lat), radians(end_lon)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        displacement = R * c
        displacements.append(displacement)
        
        # Calculate average speed
        speed = displacement / max(1, duration)  # km/h
        speeds.append(speed)
        
        # Calculate deepening rate (pressure change per hour during intensification)
        initial_pressure = track_sorted[0].central_pressure
        deepening = initial_pressure - min_pressure
        time_to_min = None
        
        for i, c in enumerate(track_sorted):
            if c.central_pressure == min_pressure:
                time_to_min = (c.time - start_time).total_seconds() / 3600
                break
        
        if time_to_min and time_to_min > 0:
            deepening_rate = deepening / time_to_min  # hPa/hour
        else:
            deepening_rate = 0
            
        deepening_rates.append(deepening_rate)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Histogram of track durations
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(durations, bins=10, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Duration (hours)')
    ax1.set_ylabel('Number of tracks')
    ax1.set_title('Track Duration Distribution')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Histogram of minimum pressures
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(min_pressures, bins=10, color='salmon', edgecolor='black')
    ax2.set_xlabel('Minimum Pressure (hPa)')
    ax2.set_ylabel('Number of tracks')
    ax2.set_title('Minimum Pressure Distribution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Scatter plot of duration vs. minimum pressure
    ax3 = plt.subplot(2, 3, 3)
    sc = ax3.scatter(durations, min_pressures, c=deepening_rates, cmap='viridis', 
                   alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Duration (hours)')
    ax3.set_ylabel('Minimum Pressure (hPa)')
    ax3.set_title('Duration vs. Minimum Pressure')
    ax3.grid(True, linestyle='--', alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('Deepening Rate (hPa/h)')
    
    # 4. Histogram of track displacements
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(displacements, bins=10, color='lightgreen', edgecolor='black')
    ax4.set_xlabel('Displacement (km)')
    ax4.set_ylabel('Number of tracks')
    ax4.set_title('Track Displacement Distribution')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Histogram of average speeds
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(speeds, bins=10, color='plum', edgecolor='black')
    ax5.set_xlabel('Average Speed (km/h)')
    ax5.set_ylabel('Number of tracks')
    ax5.set_title('Track Speed Distribution')
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # 6. Scatter plot of displacement vs. duration
    ax6 = plt.subplot(2, 3, 6)
    sc2 = ax6.scatter(displacements, durations, c=min_pressures, cmap='coolwarm', 
                    alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Displacement (km)')
    ax6.set_ylabel('Duration (hours)')
    ax6.set_title('Displacement vs. Duration')
    ax6.grid(True, linestyle='--', alpha=0.7)
    cbar2 = plt.colorbar(sc2, ax=ax6)
    cbar2.set_label('Minimum Pressure (hPa)')
    
    # Add summary statistics
    summary = (f"Total Tracks: {len(tracks)}\n"
              f"Mean Duration: {np.mean(durations):.1f} h\n"
              f"Mean Min Pressure: {np.mean(min_pressures):.1f} hPa\n"
              f"Mean Displacement: {np.mean(displacements):.1f} km\n"
              f"Mean Speed: {np.mean(speeds):.1f} km/h\n"
              f"Mean Deepening Rate: {np.mean(deepening_rates):.2f} hPa/h")
    
    plt.figtext(0.5, 0.01, summary, ha='center', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add title
    plt.suptitle('Arctic Cyclone Tracks Statistics', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)
    
    # Save figure if output file is specified
    if output_file is not None:
        output_path = Path(output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Tracks statistics plot saved to {output_path}")
    
    return fig