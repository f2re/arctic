"""
Module for pressure Laplacian detection criterion for the ArcticCyclone system.

Provides a criterion for detecting cyclones based on calculating the Laplacian of pressure field.
The Laplacian of pressure (∇²p) is proportional to geostrophic relative vorticity,
making it effective for identifying cyclonic features.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import scipy.ndimage as ndimage
from scipy import signal
from matplotlib.lines import Line2D 

from . import BaseCriterion
from core.exceptions import DetectionError
from visualization.criteria import plot_laplacian_field

# Initialize logger
logger = logging.getLogger(__name__)

class PressureLaplacianCriterion(BaseCriterion):
    """
    Criterion for detecting cyclones based on pressure Laplacian.
    
    Calculates the Laplacian (∇²p) of sea level pressure field to identify
    areas with strong cyclonic properties. Positive values correspond to
    cyclonic systems in the Northern Hemisphere.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                laplacian_threshold: float = 0.10,  # Pa/km²
                window_size: int = 3,
                smooth_sigma: float = 1.5):
        """
        Initializes the pressure Laplacian criterion.
        
        Arguments:
            min_latitude: Minimum latitude for detection (degrees N).
            laplacian_threshold: Threshold value for Laplacian (Pa/km²).
            window_size: Window size for finding local maxima.
            smooth_sigma: Smoothing parameter for pressure field.
        """
        self.min_latitude = min_latitude
        self.laplacian_threshold = laplacian_threshold
        self.window_size = window_size
        self.smooth_sigma = smooth_sigma
        
        logger.debug(f"Initialized pressure Laplacian criterion: "
                    f"min_latitude={min_latitude}, "
                    f"laplacian_threshold={laplacian_threshold}, "
                    f"window_size={window_size}, "
                    f"smooth_sigma={smooth_sigma}")
    
    def apply(self, dataset: xr.Dataset, time_step: Any, debug_plot: bool = False, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Applies the criterion to the dataset.
        
        Аргументы:
            dataset: Набор метеорологических данных xarray.
            time_step: Временной шаг для анализа.
            debug_plot: Если True, включает построение графиков полей критериев для отладки.
            output_dir: Каталог для сохранения графиков, если debug_plot=True.
            
        Returns:
            List of cyclone candidates (dictionaries with coordinates and properties).
            
        Raises:
            DetectionError: On error during detection process.
        """
        try:
            # Select time step
            time_data = dataset.sel(time=time_step)
            
            # Apply mask for Arctic region
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Determine pressure variable in the dataset
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            
            for var in pressure_vars:
                if var in arctic_data:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                available_vars = list(arctic_data.variables)
                logger.error(f"Cannot determine pressure variable. Available: {available_vars}")
                return []
            
            # Get pressure field
            pressure_data = arctic_data[pressure_var]
            
            # Extract coordinates
            lats = arctic_data.latitude.values
            lons = arctic_data.longitude.values
            
            # Calculate distances properly for polar regions
            # Earth radius in kilometers
            R = 6371.0
            
            # Create meshgrids for the coordinates
            lon_mesh, lat_mesh = np.meshgrid(lons, lats)
            lat_rad = np.radians(lat_mesh)
            
            # Calculate grid spacing in kilometers
            dx = np.zeros_like(lon_mesh)
            dy = np.zeros_like(lat_mesh)
            
            # Y-spacing (latitude)
            if len(lats) > 1:
                dlat = np.abs(lats[1] - lats[0])  # Assuming uniform spacing
                dy = R * np.radians(dlat) * np.ones_like(lat_mesh)
            else:
                dy = np.ones_like(lat_mesh)
            
            # X-spacing (longitude) - varies with latitude
            if len(lons) > 1:
                dlon = np.abs(lons[1] - lons[0])  # Assuming uniform spacing
                # Adjust for latitude (distances shrink near poles)
                dx = R * np.cos(lat_rad) * np.radians(dlon)
            else:
                dx = np.ones_like(lon_mesh)
            
            # Ensure no zeros in spacing
            dx = np.where(dx < 1e-10, 1.0, dx)
            dy = np.where(dy < 1e-10, 1.0, dy)
            
            # Get pressure values and handle units
            pressure_values = pressure_data.values
            
            # If pressure is in Pa, we don't need to convert it for the Laplacian calculation
            # The Laplacian's shape stays the same regardless of units
            
            # Apply smoothing to reduce noise
            if self.smooth_sigma > 0:
                pressure_values = ndimage.gaussian_filter(pressure_values, sigma=self.smooth_sigma)
            
            # Calculate Laplacian using better numerical approximation
            # Create convolution kernel for Laplacian
            laplacian_kernel = np.array([[0, 1, 0], 
                                        [1, -4, 1], 
                                        [0, 1, 0]])
            
            # Apply convolution to get approximate Laplacian
            # This works better than direct second derivatives for discrete fields
            laplacian_raw = ndimage.convolve(pressure_values, laplacian_kernel)
            
            # Scale by grid distances (accounting for polar grid)
            # Average dx and dy for each point to approximate local grid size
            grid_size = (dx**2 + dy**2) / 2
            
            # Scale the Laplacian
            laplacian = laplacian_raw / grid_size
            
            # Find local maxima above threshold
            # Positive Laplacian values indicate cyclonic features in Northern Hemisphere
            max_filter = ndimage.maximum_filter(laplacian, size=self.window_size)
            local_maxima = (laplacian == max_filter) & (laplacian > self.laplacian_threshold)
            
            # Get coordinates of maxima
            maxima_indices = np.where(local_maxima)
            
            # Create candidate list
            candidates = []
            
            for i in range(len(maxima_indices[0])):
                lat_idx = maxima_indices[0][i]
                lon_idx = maxima_indices[1][i]
                
                if lat_idx < len(lats) and lon_idx < len(lons):
                    latitude = float(lats[lat_idx])
                    longitude = float(lons[lon_idx])
                    laplacian_value = float(laplacian[lat_idx, lon_idx])
                    
                    # Get pressure value
                    pressure_value = float(pressure_values[lat_idx, lon_idx])
                    
                    # Convert to hPa if needed
                    if pressure_value > 10000:  # If in Pa
                        pressure_value /= 100.0  # Convert to hPa
                    
                    # Create candidate dictionary
                    candidate = {
                        'latitude': latitude,
                        'longitude': longitude,
                        'laplacian': laplacian_value,
                        'pressure': pressure_value,
                        'criterion': 'pressure_laplacian'
                    }
                    
                    candidates.append(candidate)
            
            # Получаем размеры сетки
            logger.debug(f"Размеры сетки: lat={len(lat_rad)}, lon={len(lon_rad)}")
            
            # Создаем сетку расстояний с учетом размерности
            # Используем скалярные значения для упрощения расчетов
            dlat = np.abs(np.mean(np.gradient(lat_rad)))  # Средний шаг по широте в радианах
            dlon = np.abs(np.mean(np.gradient(lon_rad)))  # Средний шаг по долготе в радианах
            
            # Переводим в километры
            dlat_km = R * dlat  # Шаг по широте в км
            dlon_km = R * np.mean(np.cos(lat_rad)) * dlon  # Шаг по долготе в км
            
            logger.debug(f"Шаг сетки: dlat_km={dlat_km}, dlon_km={dlon_km}")
            
            # Create diagnostic visualization (within existing code, not a new function)
            try:
                import matplotlib.pyplot as plt
                import cartopy.crs as ccrs
                from pathlib import Path
                
                # Create output directory
                output_dir = Path("output/diagnostics")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create polar stereographic map
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
                
                # Рассчитываем вторые производные
                # Рассчитываем вторые производные с учетом шага сетки
                # Используем скалярные значения шагов, которые уже рассчитаны выше
                # Убедимся, что dlat_km и dlon_km - скалярные значения
                dlat_km_scalar = float(dlat_km)
                dlon_km_scalar = float(dlon_km)
                
                # Вычисляем вторые производные
                d2p_dy2 = np.gradient(np.gradient(pressure_values, axis=0, edge_order=2), axis=0, edge_order=2) / (dlat_km_scalar**2)
                d2p_dx2 = np.gradient(np.gradient(pressure_values, axis=1, edge_order=2), axis=1, edge_order=2) / (dlon_km_scalar**2)
                
                # Суммируем для получения лапласиана и убедимся, что результат - numpy массив
                laplacian = np.array(d2p_dy2 + d2p_dx2)
                
                # Лапласиан в Па/км²
                # Положительные значения соответствуют циклонам (минимумам давления)

                if debug_plot and output_dir:
                    try:
                        # Ensure lats and lons are 2D for plotting if they are 1D
                        plot_lons, plot_lats = np.meshgrid(arctic_data.longitude.values, arctic_data.latitude.values)
                        plot_laplacian_field(
                            laplacian=laplacian, 
                            lats=plot_lats, 
                            lons=plot_lons,
                            threshold=self.laplacian_threshold,
                            time_step=time_step,
                            output_dir=output_dir
                        )
                        logger.debug(f"Saved pressure_laplacian plot for {time_step} to {output_dir}")
                    except Exception as plot_e:
                        logger.error(f"Error plotting pressure_laplacian field for {time_step}: {plot_e}")
                
                is_pascal = np.mean(pressure_values) > 10000  # Check if values are likely in Pa
    
                # Create conversion factor for display
                conversion = 100.0 if is_pascal else 1.0  # For Pa to hPa conversion
                
                # Calculate pressure range
                pressure_min = np.floor(np.nanmin(pressure_values) / (500 if is_pascal else 5)) * (500 if is_pascal else 5)
                pressure_max = np.ceil(np.nanmax(pressure_values) / (500 if is_pascal else 5)) * (500 if is_pascal else 5)
                
                # Create levels for 5 hPa intervals (solid lines)
                major_levels = np.arange(pressure_min, pressure_max + (500 if is_pascal else 5), (500 if is_pascal else 5))
                
                # Create levels for 2.5 hPa intervals (dotted lines) - excluding the major levels
                minor_levels = np.arange(pressure_min + (250 if is_pascal else 2.5), pressure_max, (500 if is_pascal else 5))
                
                # Add major isobars (5 hPa intervals) as solid lines
                major_contours = ax.contour(lon_mesh, lat_mesh, pressure_values, 
                                        levels=major_levels,
                                        colors='black', 
                                        linewidths=0.7,
                                        transform=ccrs.PlateCarree())
                
                # Add minor isobars (2.5 hPa intervals) as dotted lines
                minor_contours = ax.contour(lon_mesh, lat_mesh, pressure_values, 
                                        levels=minor_levels,
                                        colors='black', 
                                        linewidths=0.4,
                                        linestyles='dotted',
                                        transform=ccrs.PlateCarree())
                
                # Label only the major isobars
                ax.clabel(major_contours, inline=True, fontsize=8, 
                        fmt=lambda x: f"{x/conversion:.0f}", 
                        manual=False,
                        inline_spacing=10)
                
                # Add a legend for the isobars
                legend_elements = [
                    Line2D([0], [0], color='black', linewidth=0.7, label='5 hPa'),
                    Line2D([0], [0], color='black', linewidth=0.4, linestyle='dotted', label='2.5 hPa')
                ]
                ax.legend(handles=legend_elements, loc='lower right', 
                        fontsize=8, framealpha=0.7, title="Isobars")

                # Add title
                ts_str = str(time_step).replace('T', ' ')
                ax.set_title(f'Pressure Laplacian Field with Detected Cyclones\n{ts_str}')
                
                # Save figure
                time_str = str(time_step).replace(':', '').replace(' ', '_').replace('-', '')
                filepath = output_dir / f"laplacian_{time_str}.png"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                logger.info(f"Saved Laplacian visualization to {filepath}")
                
            except Exception as e:
                logger.warning(f"Could not create diagnostic visualization: {str(e)}")
            
            return candidates
                
        except Exception as e:
            error_msg = f"Error applying Laplacian criterion: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)