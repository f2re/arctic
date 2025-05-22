"""
Module for closed contour detection criterion for the ArcticCyclone system.

Provides a criterion for detecting cyclones based on the presence of
closed isobars around low-pressure centers.
"""

import xarray as xr
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import scipy.ndimage as ndimage

from . import BaseCriterion
from core.exceptions import DetectionError
from visualization.criteria import plot_pressure_field

# Initialize logger
logger = logging.getLogger(__name__)

class ClosedContourCriterion(BaseCriterion):
    """
    Criterion for detecting cyclones based on closed pressure contours.
    
    Identifies cyclones by finding areas enclosed by complete isobars,
    which is a defining characteristic of cyclonic systems.
    """
    
    def __init__(self, min_latitude: float = 70.0,
                contour_interval: float = 2.0,  # hPa
                min_contours: int = 1,
                max_area: float = 2e6,  # km²
                smooth_data: bool = True,
                smooth_sigma: float = 1.0):
        """
        Initializes the closed contour criterion.
        
        Arguments:
            min_latitude: Minimum latitude for detection (degrees N).
            contour_interval: Pressure interval between contours (hPa).
            min_contours: Minimum number of closed contours required.
            max_area: Maximum area enclosed by the outermost contour (km²).
            smooth_data: Whether to apply smoothing to pressure data.
            smooth_sigma: Smoothing parameter when smooth_data is True.
        """
        self.min_latitude = min_latitude
        self.contour_interval = contour_interval
        self.min_contours = min_contours
        self.max_area = max_area
        self.smooth_data = smooth_data
        self.smooth_sigma = smooth_sigma
        
        logger.debug(f"Initialized closed contour criterion: "
                   f"min_latitude={min_latitude}, "
                   f"contour_interval={contour_interval}, "
                   f"min_contours={min_contours}, "
                   f"max_area={max_area}")
    
    def apply(self, dataset: xr.Dataset, time_step: Any, debug_plot: bool = False, output_dir: Optional[str] = None) -> List[Dict]:
        """
        Applies the criterion to the dataset.
        
        Arguments:
            dataset: Dataset of meteorological data.
            time_step: Time step for analysis.
            debug_plot: If True, enables plotting of criterion fields for debugging.
            output_dir: Directory for saving plots, if debug_plot=True.
            
        Returns:
            List of cyclone candidates (dictionaries with coordinates and properties).
            
        Raises:
            DetectionError: On error during detection process.
        """
        try:
            # Determine pressure variable in the dataset
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            
            for var in pressure_vars:
                if var in dataset:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                raise ValueError("Cannot determine pressure variable in dataset")
                
            # Select time step and apply regional mask
            time_data = dataset.sel(time=time_step)
            arctic_data = time_data.where(time_data.latitude >= self.min_latitude, drop=True)
            
            # Get pressure field
            pressure_field = arctic_data[pressure_var].values
            
            # Apply smoothing if needed
            if self.smooth_data:
                pressure_field = ndimage.gaussian_filter(pressure_field, sigma=self.smooth_sigma)
            # Store the (smoothed) pressure field for combined visualization
            self.pressure_field = pressure_field
            # Placeholder mask for closed contour visualization
            self.contour_mask = np.zeros_like(pressure_field, dtype=bool)
            
            # Find local minima (potential cyclone centers)
            min_filter = ndimage.minimum_filter(pressure_field, size=3)
            local_minima = (pressure_field == min_filter)
            
            # Get coordinates of minima
            minima_indices = np.where(local_minima)
            
            candidates = []
            
            # Check each minimum for closed contours
            for i in range(len(minima_indices[0])):
                lat_idx = minima_indices[0][i]
                lon_idx = minima_indices[1][i]
                
                center_pressure = pressure_field[lat_idx, lon_idx]
                latitude = float(arctic_data.latitude.values[lat_idx])
                longitude = float(arctic_data.longitude.values[lon_idx])
                
                # Check for closed contours around this minimum
                closed_contours = self._check_closed_contours(
                    pressure_field, lat_idx, lon_idx, center_pressure)
                
                if closed_contours >= self.min_contours:
                    # Calculate enclosed area
                    area = self._calculate_enclosed_area(
                        pressure_field, lat_idx, lon_idx, center_pressure,
                        arctic_data.latitude.values, arctic_data.longitude.values)
                    
                    # If area is within limits
                    if area <= self.max_area:
                        candidate = {
                            'latitude': latitude,
                            'longitude': longitude,
                            'pressure': float(center_pressure),
                            'closed_contours': closed_contours,
                            'enclosed_area_km2': float(area),
                            'criterion': 'closed_contour'
                        }
                        
                        candidates.append(candidate)
            
            if debug_plot and output_dir:
                try:
                    lons_2d, lats_2d = np.meshgrid(arctic_data.longitude.values, arctic_data.latitude.values)
                    # Use pressure_field which is already smoothed if self.smooth_data is True
                    plot_pressure_field(
                        pressure=self.pressure_field, # Use the (potentially smoothed) pressure field used for detection
                        lats=lats_2d,
                        lons=lons_2d,
                        time_step=time_step,
                        output_dir=output_dir
                    )
                    logger.debug(f"Saved pressure field plot (for closed_contour context) for {time_step} to {output_dir}")
                except Exception as plot_e:
                    logger.error(f"Error plotting pressure field (for closed_contour context) for {time_step}: {plot_e}")
        
            logger.debug(f"Closed contour criterion found {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            error_msg = f"Error applying closed contour criterion: {str(e)}"
            logger.error(error_msg)
            raise DetectionError(error_msg)
    
    def _check_closed_contours(self, pressure_field: np.ndarray, 
                             lat_idx: int, lon_idx: int, 
                             center_pressure: float) -> int:
        """
        Checks for closed contours around a pressure minimum.
        
        Arguments:
            pressure_field: 2D pressure field array.
            lat_idx: Latitude index of the center.
            lon_idx: Longitude index of the center.
            center_pressure: Pressure at the center.
            
        Returns:
            Number of closed contours detected.
        """
        # Define a region around the minimum
        window_size = min(20, min(pressure_field.shape) // 4)
        
        lat_min = max(0, lat_idx - window_size)
        lat_max = min(pressure_field.shape[0], lat_idx + window_size + 1)
        lon_min = max(0, lon_idx - window_size)
        lon_max = min(pressure_field.shape[1], lon_idx + window_size + 1)
        
        region = pressure_field[lat_min:lat_max, lon_min:lon_max]
        
        # Adjust center indices for the region
        local_lat_idx = lat_idx - lat_min
        local_lon_idx = lon_idx - lon_min
        
        # Count closed contours
        num_contours = 0
        current_level = center_pressure
        max_contours = 10  # Limit to avoid excessive computation
        
        for _ in range(max_contours):
            current_level += self.contour_interval
            
            # Create binary mask for points below current level
            mask = region < current_level
            
            # Label connected components
            labeled_mask, num_components = ndimage.label(mask)
            
            # Check if center is in a labeled component
            if 0 <= local_lat_idx < mask.shape[0] and 0 <= local_lon_idx < mask.shape[1]:
                center_label = labeled_mask[local_lat_idx, local_lon_idx]
                
                if center_label > 0:
                    # Check if component touches border (not closed)
                    component_mask = labeled_mask == center_label
                    border_touch = (
                        np.any(component_mask[0, :]) or  # Top
                        np.any(component_mask[-1, :]) or  # Bottom
                        np.any(component_mask[:, 0]) or  # Left
                        np.any(component_mask[:, -1])     # Right
                    )
                    
                    if not border_touch:
                        num_contours += 1
                    else:
                        break  # Stop counting if we find a non-closed contour
                else:
                    break  # Stop counting if center is not in a labeled component
            else:
                break  # Stop if indices are out of bounds
        
        return num_contours
    
    def _calculate_enclosed_area(self, pressure_field: np.ndarray,
                              lat_idx: int, lon_idx: int,
                              center_pressure: float,
                              latitudes: np.ndarray,
                              longitudes: np.ndarray) -> float:
        """
        Calculates the area enclosed by the outermost closed contour.
        
        Arguments:
            pressure_field: 2D pressure field array.
            lat_idx: Latitude index of the center.
            lon_idx: Longitude index of the center.
            center_pressure: Pressure at the center.
            latitudes: Array of latitude values.
            longitudes: Array of longitude values.
            
        Returns:
            Area in square kilometers.
        """
        # Define a region around the minimum
        window_size = min(20, min(pressure_field.shape) // 4)
        
        lat_min = max(0, lat_idx - window_size)
        lat_max = min(pressure_field.shape[0], lat_idx + window_size + 1)
        lon_min = max(0, lon_idx - window_size)
        lon_max = min(pressure_field.shape[1], lon_idx + window_size + 1)
        
        region = pressure_field[lat_min:lat_max, lon_min:lon_max]
        region_lats = latitudes[lat_min:lat_max]
        region_lons = longitudes[lon_min:lon_max]
        
        # Adjust center indices for the region
        local_lat_idx = lat_idx - lat_min
        local_lon_idx = lon_idx - lon_min
        
        # Find outermost closed contour
        current_level = center_pressure
        max_contours = 10
        last_closed_mask = None
        
        for _ in range(max_contours):
            current_level += self.contour_interval
            
            # Create binary mask for points below current level
            mask = region < current_level
            
            # Label connected components
            labeled_mask, _ = ndimage.label(mask)
            
            # Check if center is in a labeled component
            if 0 <= local_lat_idx < mask.shape[0] and 0 <= local_lon_idx < mask.shape[1]:
                center_label = labeled_mask[local_lat_idx, local_lon_idx]
                
                if center_label > 0:
                    # Check if component touches border (not closed)
                    component_mask = labeled_mask == center_label
                    border_touch = (
                        np.any(component_mask[0, :]) or  # Top
                        np.any(component_mask[-1, :]) or  # Bottom
                        np.any(component_mask[:, 0]) or  # Left
                        np.any(component_mask[:, -1])     # Right
                    )
                    
                    if not border_touch:
                        last_closed_mask = component_mask
                    else:
                        break  # Stop if we find a non-closed contour
                else:
                    break  # Stop if center is not in a labeled component
            else:
                break  # Stop if indices are out of bounds
        
        # Calculate area if we found closed contours
        if last_closed_mask is not None:
            # Count grid cells in the closed contour
            num_cells = np.sum(last_closed_mask)
            
            # Calculate average cell area at this latitude
            avg_lat = np.mean(region_lats)
            lat_km = 111.0  # 1 degree latitude ≈ 111 km
            lon_km = np.cos(np.radians(avg_lat)) * 111.0  # 1 degree longitude dependent on latitude
            
            # Estimate average cell spacing
            lat_spacing = np.mean(np.diff(region_lats)) if len(region_lats) > 1 else 1.0
            lon_spacing = np.mean(np.diff(region_lons)) if len(region_lons) > 1 else 1.0
            
            # Calculate area in km²
            cell_area_km2 = lat_spacing * lon_spacing * lat_km * lon_km
            total_area_km2 = num_cells * cell_area_km2
            
            return total_area_km2
        
        return 0.0  # Return zero area if no closed contours found