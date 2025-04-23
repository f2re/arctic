"""
Пакет визуализации для системы ArcticCyclone.

Предоставляет функции и классы для визуализации арктических циклонов,
их треков, параметров и пространственного распределения.
"""

from .mappers import (
    create_arctic_map, 
    plot_cyclone_centers, 
    save_figure, 
    set_map_projection,
    add_map_features
)
from .tracks import (
    plot_cyclone_track, 
    plot_multiple_tracks, 
    animate_cyclone_track,
    plot_track_parameters
)
from .parameters import (
    plot_cyclone_parameters, 
    plot_parameter_correlation, 
    plot_parameter_histogram,
    plot_parameter_evolution,
    create_pressure_profile,
    create_wind_profile
)
from .heatmaps import (
    create_genesis_density_map, 
    create_track_density_map, 
    create_pressure_intensity_map,
    create_cyclone_frequency_map,
    create_parameter_distribution
)

__all__ = [
    # Base mapping functions
    'create_arctic_map',
    'plot_cyclone_centers',
    'save_figure',
    'set_map_projection',
    'add_map_features',
    
    # Track visualization
    'plot_cyclone_track',
    'plot_multiple_tracks',
    'animate_cyclone_track',
    'plot_track_parameters',
    
    # Parameter visualization
    'plot_cyclone_parameters',
    'plot_parameter_correlation',
    'plot_parameter_histogram',
    'plot_parameter_evolution',
    'create_pressure_profile',
    'create_wind_profile',
    
    # Heatmap visualization
    'create_genesis_density_map',
    'create_track_density_map',
    'create_pressure_intensity_map',
    'create_cyclone_frequency_map',
    'create_parameter_distribution'
]