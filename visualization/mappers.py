"""
Модуль базовых инструментов картографии для системы ArcticCyclone.

Предоставляет функции и классы для создания базовых карт и
визуализации пространственных данных арктических циклонов.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import matplotlib.ticker as mticker
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import logging
from datetime import datetime

from models.cyclone import Cyclone
from models.classifications import CycloneType, CycloneIntensity
from core.exceptions import VisualizationError

# Инициализация логгера
logger = logging.getLogger(__name__)

class MapManager:
    """
    Управляет созданием и настройкой карт для визуализации арктических данных.
    
    Предоставляет интерфейс для создания специализированных карт,
    оптимизированных для арктического региона.
    """
    
    def __init__(self, central_longitude: float = 0.0, 
                min_latitude: float = 60.0,
                resolution: str = 'intermediate',
                map_projection: str = 'NorthPolarStereo',
                feature_set: List[str] = None):
        """
        Инициализирует менеджер карт.
        
        Аргументы:
            central_longitude: Центральная долгота проекции (градусы).
            min_latitude: Минимальная широта для визуализации (градусы с.ш.).
            resolution: Разрешение географических данных ('low', 'intermediate', 'high').
            map_projection: Тип проекции карты ('NorthPolarStereo', 'PlateCarree', 'LambertConformal').
            feature_set: Список географических объектов для отображения.
        """
        self.central_longitude = central_longitude
        self.min_latitude = min_latitude
        self.resolution = resolution
        self.map_projection = map_projection
        
        # Набор географических объектов по умолчанию
        self.feature_set = feature_set or ['coastline', 'borders', 'lakes']
        
        # Словарь доступных проекций
        self.projections = {
            'NorthPolarStereo': ccrs.NorthPolarStereo(central_longitude=central_longitude),
            'PlateCarree': ccrs.PlateCarree(central_longitude=central_longitude),
            'LambertConformal': ccrs.LambertConformal(central_longitude=central_longitude, 
                                                     central_latitude=90.0),
            'Orthographic': ccrs.Orthographic(central_longitude=central_longitude, 
                                            central_latitude=90.0),
            'AzimuthalEquidistant': ccrs.AzimuthalEquidistant(central_longitude=central_longitude, 
                                                             central_latitude=90.0)
        }
        
        # Словарь географических объектов
        self.features = {
            'coastline': cfeature.COASTLINE.with_scale(resolution),
            'borders': cfeature.BORDERS.with_scale(resolution),
            'lakes': cfeature.LAKES.with_scale(resolution),
            'rivers': cfeature.RIVERS.with_scale(resolution),
            'land': cfeature.LAND.with_scale(resolution),
            'ocean': cfeature.OCEAN.with_scale(resolution),
            'states': cfeature.STATES.with_scale(resolution)
        }
        
        # Цветовая схема для объектов по умолчанию
        self.feature_colors = {
            'coastline': 'black',
            'borders': 'gray',
            'lakes': 'lightblue',
            'rivers': 'blue',
            'land': 'tan',
            'ocean': 'lightblue',
            'states': 'gray'
        }
        
        logger.debug(f"Инициализирован MapManager с проекцией {map_projection}, "
                    f"центральной долготой {central_longitude}°, "
                    f"и минимальной широтой {min_latitude}°N")
    
    def create_map(self, figsize: Tuple[float, float] = (10, 8),
                 projection: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Создает новую карту с выбранной проекцией.
        
        Аргументы:
            figsize: Размер фигуры (ширина, высота) в дюймах.
            projection: Тип проекции. Если None, используется установленная по умолчанию.
            
        Возвращает:
            Кортеж (figure, axis) с созданной картой.
            
        Вызывает:
            VisualizationError: Если указана неподдерживаемая проекция.
        """
        try:
            # Определяем проекцию
            proj_name = projection or self.map_projection
            
            if proj_name not in self.projections:
                raise VisualizationError(f"Неподдерживаемая проекция: {proj_name}")
            
            # Создаем фигуру с указанной проекцией
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection=self.projections[proj_name])
            
            # Для полярной стереографической проекции устанавливаем границы по широте
            if proj_name == 'NorthPolarStereo':
                ax.set_extent([-180, 180, self.min_latitude, 90], ccrs.PlateCarree())
            
            # Добавляем стандартные географические объекты
            self._add_features(ax)
            
            # Добавляем сетку координат
            self._add_grid(ax, proj_name)
            
            # Устанавливаем заголовок
            ax.set_title(f"Арктический регион (> {self.min_latitude}°N)")
            
            return fig, ax
            
        except Exception as e:
            error_msg = f"Ошибка при создании карты: {str(e)}"
            logger.error(error_msg)
            raise VisualizationError(error_msg)
    
    def _add_features(self, ax: plt.Axes) -> None:
        """
        Добавляет географические объекты на карту.
        
        Аргументы:
            ax: Оси для отображения объектов.
        """
        for feature_name in self.feature_set:
            if feature_name in self.features:
                ax.add_feature(self.features[feature_name], 
                              edgecolor=self.feature_colors.get(feature_name, 'black'),
                              facecolor='none' if feature_name not in ['land', 'ocean'] else self.feature_colors.get(feature_name))
    
    def _add_grid(self, ax: plt.Axes, projection: str) -> None:
        """
        Добавляет координатную сетку на карту.
        
        Аргументы:
            ax: Оси для отображения сетки.
            projection: Тип проекции карты.
        """
        if projection in ['PlateCarree', 'LambertConformal']:
            # Стандартные линии сетки для нормальных проекций
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                            linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        else:
            # Круговые линии сетки для полярных проекций
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpl.path.Path(verts * radius + center)
            
            for lat in range(int(self.min_latitude), 90, 10):
                ax.gridlines(ccrs.PlateCarree(), ylimits=(lat, lat),
                            linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
            
            # Добавляем линии долготы
            for lon in range(-180, 180, 30):
                ax.gridlines(ccrs.PlateCarree(), xlimits=(lon, lon),
                            linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
    
    def set_map_features(self, feature_set: List[str]) -> None:
        """
        Устанавливает набор географических объектов для отображения.
        
        Аргументы:
            feature_set: Список названий объектов для отображения.
            
        Вызывает:
            ValueError: Если указан неподдерживаемый объект.
        """
        for feature in feature_set:
            if feature not in self.features:
                raise ValueError(f"Неподдерживаемый географический объект: {feature}")
        
        self.feature_set = feature_set
        logger.debug(f"Установлен набор географических объектов: {feature_set}")
    
    def set_feature_colors(self, color_map: Dict[str, str]) -> None:
        """
        Устанавливает цвета для географических объектов.
        
        Аргументы:
            color_map: Словарь сопоставления объектов и цветов.
        """
        self.feature_colors.update(color_map)
        logger.debug(f"Обновлена цветовая схема объектов: {color_map}")


def create_arctic_map(central_longitude: float = 0.0, 
                    min_latitude: float = 60.0, 
                    figsize: Tuple[float, float] = (10, 8),
                    projection: str = 'NorthPolarStereo',
                    resolution: str = 'intermediate',
                    features: List[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Создает карту арктического региона.
    
    Аргументы:
        central_longitude: Центральная долгота проекции (градусы).
        min_latitude: Минимальная широта для визуализации (градусы с.ш.).
        figsize: Размер фигуры (ширина, высота) в дюймах.
        projection: Тип проекции карты.
        resolution: Разрешение географических данных.
        features: Список географических объектов для отображения.
        
    Возвращает:
        Кортеж (figure, axis) с созданной картой.
    """
    # Создаем менеджер карт
    manager = MapManager(
        central_longitude=central_longitude,
        min_latitude=min_latitude,
        resolution=resolution,
        map_projection=projection,
        feature_set=features
    )
    
    # Создаем карту
    return manager.create_map(figsize=figsize)


def set_map_projection(ax: plt.Axes, projection: str, 
                     central_longitude: float = 0.0) -> plt.Axes:
    """
    Устанавливает проекцию для существующих осей.
    
    Аргументы:
        ax: Оси для изменения проекции.
        projection: Тип проекции карты.
        central_longitude: Центральная долгота проекции.
        
    Возвращает:
        Оси с новой проекцией.
        
    Вызывает:
        VisualizationError: Если указана неподдерживаемая проекция.
    """
    # Словарь доступных проекций
    projections = {
        'NorthPolarStereo': ccrs.NorthPolarStereo(central_longitude=central_longitude),
        'PlateCarree': ccrs.PlateCarree(central_longitude=central_longitude),
        'LambertConformal': ccrs.LambertConformal(central_longitude=central_longitude, 
                                                central_latitude=90.0),
        'Orthographic': ccrs.Orthographic(central_longitude=central_longitude, 
                                        central_latitude=90.0),
        'AzimuthalEquidistant': ccrs.AzimuthalEquidistant(central_longitude=central_longitude, 
                                                         central_latitude=90.0)
    }
    
    if projection not in projections:
        raise VisualizationError(f"Неподдерживаемая проекция: {projection}")
    
    # Создаем новую фигуру с новой проекцией
    fig = plt.gcf()
    new_ax = fig.add_subplot(1, 1, 1, projection=projections[projection])
    
    # Копируем данные из старых осей
    for artist in ax.get_children():
        if hasattr(artist, 'get_transform'):
            try:
                # Создаем копию артиста с новой проекцией
                new_artist = artist.copy()
                new_artist.set_transform(projections[projection])
                new_ax.add_artist(new_artist)
            except:
                pass
    
    # Удаляем старые оси
    fig.delaxes(ax)
    
    return new_ax


def add_map_features(ax: plt.Axes, features: List[str], 
                   resolution: str = 'intermediate',
                   colors: Dict[str, str] = None) -> None:
    """
    Добавляет географические объекты на карту.
    
    Аргументы:
        ax: Оси для отображения объектов.
        features: Список названий объектов для отображения.
        resolution: Разрешение географических данных.
        colors: Словарь цветов для объектов.
    """
    # Словарь географических объектов
    feature_dict = {
        'coastline': cfeature.COASTLINE.with_scale(resolution),
        'borders': cfeature.BORDERS.with_scale(resolution),
        'lakes': cfeature.LAKES.with_scale(resolution),
        'rivers': cfeature.RIVERS.with_scale(resolution),
        'land': cfeature.LAND.with_scale(resolution),
        'ocean': cfeature.OCEAN.with_scale(resolution),
        'states': cfeature.STATES.with_scale(resolution)
    }
    
    # Цветовая схема для объектов по умолчанию
    default_colors = {
        'coastline': 'black',
        'borders': 'gray',
        'lakes': 'lightblue',
        'rivers': 'blue',
        'land': 'tan',
        'ocean': 'lightblue',
        'states': 'gray'
    }
    
    # Обновляем цвета, если указаны
    if colors:
        default_colors.update(colors)
    
    # Добавляем объекты
    for feature_name in features:
        if feature_name in feature_dict:
            ax.add_feature(feature_dict[feature_name], 
                          edgecolor=default_colors.get(feature_name, 'black'),
                          facecolor='none' if feature_name not in ['land', 'ocean'] else default_colors.get(feature_name))
        else:
            logger.warning(f"Неподдерживаемый географический объект: {feature_name}")


def plot_cyclone_centers(ax: plt.Axes, cyclones: List[Cyclone], 
                       color_by: str = 'type',
                       marker_size: float = 50.0,
                       add_labels: bool = True,
                       label_offset: Tuple[float, float] = (0.1, 0.1),
                       transform: Optional[ccrs.Projection] = None) -> None:
    """
    Отображает центры циклонов на карте.
    
    Аргументы:
        ax: Оси для отображения центров циклонов.
        cyclones: Список циклонов для отображения.
        color_by: Параметр для определения цвета ('type', 'intensity', 'pressure').
        marker_size: Размер маркера в точках.
        add_labels: Добавлять ли метки с идентификаторами циклонов.
        label_offset: Смещение меток (x, y) относительно центра циклона.
        transform: Проекция для преобразования координат.
    """
    # Устанавливаем проекцию по умолчанию, если не указана
    if transform is None:
        transform = ccrs.PlateCarree()
    
    # Цветовые схемы для разных параметров
    color_schemes = {
        'type': {
            CycloneType.COLD_CORE: 'blue',
            CycloneType.WARM_CORE: 'red',
            CycloneType.HYBRID: 'purple',
            CycloneType.UNCLASSIFIED: 'gray'
        },
        'intensity': {
            CycloneIntensity.WEAK: 'green',
            CycloneIntensity.MODERATE: 'yellow',
            CycloneIntensity.STRONG: 'orange',
            CycloneIntensity.VERY_STRONG: 'red'
        }
    }
    
    # Отображаем каждый циклон
    for cyclone in cyclones:
        # Определяем цвет
        if color_by == 'type' and hasattr(cyclone.parameters, 'thermal_type'):
            color = color_schemes['type'].get(cyclone.parameters.thermal_type, 'gray')
        elif color_by == 'intensity':
            intensity = cyclone.calculate_intensity()
            color = color_schemes['intensity'].get(intensity, 'gray')
        elif color_by == 'pressure':
            # Цвет по давлению (от красного к синему)
            norm = plt.Normalize(950, 1020)
            cmap = plt.cm.coolwarm_r
            color = cmap(norm(cyclone.central_pressure))
        else:
            # По умолчанию - черный
            color = 'black'
        
        # Отображаем центр циклона
        ax.scatter(cyclone.longitude, cyclone.latitude, 
                 color=color, s=marker_size, 
                 transform=transform, 
                 edgecolor='black', linewidth=0.5, 
                 zorder=5)
        
        # Добавляем метку, если требуется
        if add_labels and hasattr(cyclone, 'track_id') and cyclone.track_id:
            # Создаем короткий идентификатор, если слишком длинный
            label = cyclone.track_id
            if len(label) > 8:
                label = label[:8]
            
            # Добавляем метку с эффектом обводки для читаемости
            text = ax.text(cyclone.longitude + label_offset[0], 
                         cyclone.latitude + label_offset[1], 
                         label, transform=transform,
                         fontsize=8, ha='center', va='center',
                         zorder=6)
            
            # Добавляем обводку для улучшения читаемости
            text.set_path_effects([
                patheffects.withStroke(linewidth=2, foreground='white')
            ])


def save_figure(fig: plt.Figure, filename: Union[str, Path], 
              dpi: int = 300, 
              bbox_inches: str = 'tight',
              create_dirs: bool = True) -> None:
    """
    Сохраняет фигуру в файл.
    
    Аргументы:
        fig: Фигура для сохранения.
        filename: Имя файла или путь для сохранения.
        dpi: Разрешение изображения (точек на дюйм).
        bbox_inches: Параметр обрезки полей ('tight' или None).
        create_dirs: Создавать ли директории, если они не существуют.
    """
    try:
        # Преобразуем в Path
        path = Path(filename)
        
        # Создаем директории, если требуется
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем фигуру
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Фигура сохранена в файл: {path}")
        
    except Exception as e:
        error_msg = f"Ошибка при сохранении фигуры: {str(e)}"
        logger.error(error_msg)
        raise VisualizationError(error_msg)