"""
Пакет анализа для системы ArcticCyclone.

Предоставляет инструменты и классы для анализа циклонов, их сравнения,
статистической обработки и климатологических исследований.
"""

from .climatology import (
    ClimateAnalyzer,
    calculate_monthly_frequencies,
    calculate_annual_cycle,
    calculate_interannual_variability,
    create_climatology,
    analyze_seasonal_patterns,
    analyze_climate_trends
)

from .comparisons import (
    CycloneComparator,
    compare_cyclone_tracks,
    compare_cyclone_parameters,
    compare_datasets,
    calculate_track_similarity,
    compare_spatial_distributions,
    compare_seasonal_distributions
)

from .statistics import (
    CycloneStatistics,
    calculate_basic_statistics,
    calculate_spatial_statistics,
    calculate_temporal_statistics,
    calculate_ensemble_statistics,
    estimate_uncertainties,
    calculate_cyclone_flux,
    perform_trend_analysis
)

__all__ = [
    # Climatology functions
    'ClimateAnalyzer',
    'calculate_monthly_frequencies',
    'calculate_annual_cycle',
    'calculate_interannual_variability',
    'create_climatology',
    'analyze_seasonal_patterns',
    'analyze_climate_trends',
    
    # Comparison functions
    'CycloneComparator',
    'compare_cyclone_tracks',
    'compare_cyclone_parameters',
    'compare_datasets',
    'calculate_track_similarity',
    'compare_spatial_distributions',
    'compare_seasonal_distributions',
    
    # Statistics functions
    'CycloneStatistics',
    'calculate_basic_statistics',
    'calculate_spatial_statistics',
    'calculate_temporal_statistics',
    'calculate_ensemble_statistics',
    'estimate_uncertainties',
    'calculate_cyclone_flux',
    'perform_trend_analysis'
]