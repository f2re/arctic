"""
Multi-criteria validator for Arctic cyclone detection.

Provides a system for validating cyclone candidates using multiple weighted criteria
to reduce false positives while maintaining sensitivity.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
import xarray as xr

# Initialize logger
logger = logging.getLogger(__name__)

class MultiCriteriaValidator:
    """
    Multi-criteria validation system for Arctic cyclone detection.
    
    Evaluates candidates based on multiple weighted parameters to reduce false positives.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize with configurable weights for each criterion.
        
        Args:
            weights: Dictionary of weights for each criterion
        """
        self.weights = weights or {
            'pressure_minimum': 0.25,
            'vorticity': 0.30,
            'size_filter': 0.15,
            'wind_threshold': 0.20,
            'pressure_gradient': 0.10
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.debug(f"Initialized multi-criteria validator with weights: {self.weights}")

    def validate_candidate(self, candidate: Dict[str, Any], dataset: xr.Dataset) -> Tuple[bool, float, Dict[str, float]]:
        """
        Validate a candidate using multiple criteria with weighted scoring.
        
        Args:
            candidate: Dictionary with candidate properties
            dataset: xarray.Dataset with meteorological data
            
        Returns:
            Tuple of (is_valid, total_score, individual_scores) where score is 0-1
        """
        scores = {}
        
        # Pressure minimum criterion (0-1 score)
        scores['pressure_minimum'] = self._validate_pressure_minimum(candidate, dataset)
        
        # Vorticity criterion (0-1 score)
        scores['vorticity'] = self._validate_vorticity(candidate, dataset)
        
        # Size filtering criterion (0-1 score)
        scores['size_filter'] = self._validate_size(candidate, dataset)
        
        # Wind threshold criterion (0-1 score)
        scores['wind_threshold'] = self._validate_wind(candidate, dataset)
        
        # Pressure gradient criterion (0-1 score)
        scores['pressure_gradient'] = self._validate_pressure_gradient(candidate, dataset)
        
        # Calculate weighted score
        total_score = sum(scores[criterion] * self.weights[criterion] 
                         for criterion in self.weights)
        
        # Candidate is valid if score exceeds threshold (e.g., 0.6)
        is_valid = total_score >= 0.6
        
        return is_valid, total_score, scores

    def _validate_pressure_minimum(self, candidate: Dict[str, Any], dataset: xr.Dataset) -> float:
        """
        Validate pressure minimum criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            
        Returns:
            Score between 0-1
        """
        try:
            if 'pressure' not in candidate:
                # Try to extract pressure from dataset
                pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
                pressure_var = None
                
                for var in pressure_vars:
                    if var in dataset:
                        pressure_var = var
                        break
                
                if pressure_var is None:
                    return 0.5  # Neutral score if pressure data unavailable
                
                lat, lon = candidate['latitude'], candidate['longitude']
                pressure = float(dataset[pressure_var].sel(
                    latitude=lat, longitude=lon, method='nearest').values)
                candidate['pressure'] = pressure
            else:
                pressure = candidate['pressure']
            
            # Score based on pressure value (lower pressure = higher score)
            # Typical Arctic mesocyclones have pressures 980-1010 hPa
            if pressure <= 980:
                return 1.0
            elif pressure <= 990:
                return 0.8
            elif pressure <= 1000:
                return 0.6
            elif pressure <= 1010:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating pressure minimum: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_vorticity(self, candidate: Dict[str, Any], dataset: xr.Dataset) -> float:
        """
        Validate vorticity criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            
        Returns:
            Score between 0-1
        """
        try:
            if 'vorticity' not in candidate:
                # Try to extract vorticity from dataset
                vorticity_vars = ['vorticity', 'vo', 'relative_vorticity']
                vorticity_var = None
                
                for var in vorticity_vars:
                    if var in dataset:
                        vorticity_var = var
                        break
                
                if vorticity_var is None:
                    return 0.5  # Neutral score if vorticity data unavailable
                
                lat, lon = candidate['latitude'], candidate['longitude']
                vorticity = float(dataset[vorticity_var].sel(
                    latitude=lat, longitude=lon, method='nearest').values)
                candidate['vorticity'] = vorticity
            else:
                vorticity = candidate['vorticity']
            
            # Score based on vorticity value (higher positive vorticity = higher score)
            # Typical Arctic mesocyclones have vorticity 1e-5 to 1e-4 1/s
            abs_vorticity = abs(vorticity)
            if abs_vorticity >= 5e-5:
                return 1.0
            elif abs_vorticity >= 3e-5:
                return 0.8
            elif abs_vorticity >= 1e-5:
                return 0.6
            elif abs_vorticity >= 5e-6:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating vorticity: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_size(self, candidate: Dict[str, Any], dataset: xr.Dataset) -> float:
        """
        Validate size filtering criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            
        Returns:
            Score between 0-1
        """
        try:
            # If diameter already calculated, use it
            if 'diameter_km' in candidate:
                diameter_km = candidate['diameter_km']
            else:
                # Estimate size based on pressure field
                lat, lon = candidate['latitude'], candidate['longitude']
                
                # Extract region around candidate
                radius_deg = 3.0  # ~300km at mid-latitudes
                lat_min, lat_max = float(dataset.latitude.min()), float(dataset.latitude.max())
                lon_min, lon_max = float(dataset.longitude.min()), float(dataset.longitude.max())
                
                region_lat_min = max(lat - radius_deg, lat_min)
                region_lat_max = min(lat + radius_deg, lat_max)
                region_lon_min = max(lon - radius_deg, lon_min)
                region_lon_max = min(lon + radius_deg, lon_max)
                
                # Check if we have a reasonable region
                if (region_lat_max - region_lat_min) < 0.1 or (region_lon_max - region_lon_min) < 0.1:
                    return 0.5  # Neutral score for edge cases
                
                region = dataset.sel(
                    latitude=slice(region_lat_min, region_lat_max),
                    longitude=slice(region_lon_min, region_lon_max)
                )
                
                # Get pressure at center
                pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
                pressure_var = None
                for var in pressure_vars:
                    if var in region:
                        pressure_var = var
                        break
                
                if pressure_var is None:
                    return 0.5  # Neutral score if pressure data unavailable
                
                central_pressure = candidate.get('pressure', 
                                               float(region[pressure_var].sel(
                                                   latitude=lat, longitude=lon, 
                                                   method='nearest').values))
                
                # Define contour threshold (typically 2-4 hPa above center)
                contour_threshold = central_pressure + 3.0
                
                # Create mask for areas below threshold
                pressure_field = region[pressure_var].values
                mask = pressure_field < contour_threshold
                
                # Estimate connected area
                # Simple approach: count pixels and convert to approximate diameter
                area_pixels = np.sum(mask)
                
                if area_pixels == 0:
                    return 0.0  # No area
                
                # Convert to approximate diameter in km
                avg_lat = (region_lat_min + region_lat_max) / 2
                km_per_degree = 111.0 * np.cos(np.radians(avg_lat))
                pixel_area_km2 = (km_per_degree ** 2) / (len(region.latitude) * len(region.longitude))
                area_km2 = area_pixels * pixel_area_km2
                diameter_km = 2 * np.sqrt(area_km2 / np.pi)
                
                # Store for future use
                candidate['diameter_km'] = float(diameter_km)
                candidate['area_km2'] = float(area_km2)
            
            # Score based on size (mesoscale systems are 100-800 km)
            if 100 <= diameter_km <= 800:
                return 1.0  # Perfect size
            elif 50 <= diameter_km <= 1000:
                # Score decreases as we move away from optimal range
                if diameter_km < 100:
                    # Too small but within extended range
                    return 0.5 + 0.5 * (diameter_km - 50) / 50
                else:
                    # Too large but within extended range
                    return 0.5 + 0.5 * (1000 - diameter_km) / 200
            else:
                return 0.0  # Outside acceptable range
                
        except Exception as e:
            logger.debug(f"Error validating size: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_wind(self, candidate: Dict[str, Any], dataset: xr.Dataset) -> float:
        """
        Validate wind threshold criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            
        Returns:
            Score between 0-1
        """
        try:
            if 'wind_speed' not in candidate:
                # Try to extract wind from dataset
                lat, lon = candidate['latitude'], candidate['longitude']
                
                # Check for 10m wind data first
                wind_var_pairs = [
                    ('u10', 'v10'),
                    ('10u', '10v'),
                    ('u_component_of_wind_10m', 'v_component_of_wind_10m'),
                    ('10m_u_component_of_wind', '10m_v_component_of_wind')
                ]
                
                u_wind_var = None
                v_wind_var = None
                
                # First try to find 10m wind data
                for u_var, v_var in wind_var_pairs:
                    if u_var in dataset and v_var in dataset:
                        u_wind_var, v_wind_var = u_var, v_var
                        break
                
                # If 10m wind not found, try pressure level wind
                if u_wind_var is None or v_wind_var is None:
                    wind_var_pairs = [
                        ('u', 'v'),
                        ('u_component_of_wind', 'v_component_of_wind')
                    ]
                    
                    for u_var, v_var in wind_var_pairs:
                        if u_var in dataset and v_var in dataset:
                            u_wind_var, v_wind_var = u_var, v_var
                            break
                
                if u_wind_var is None or v_wind_var is None:
                    return 0.5  # Neutral score if wind data unavailable
                
                u_point = float(dataset[u_wind_var].sel(
                    latitude=lat, longitude=lon, method='nearest').values)
                v_point = float(dataset[v_wind_var].sel(
                    latitude=lat, longitude=lon, method='nearest').values)
                
                wind_speed = np.sqrt(u_point**2 + v_point**2)
                candidate['wind_speed'] = wind_speed
                candidate['u_wind'] = u_point
                candidate['v_wind'] = v_point
            else:
                wind_speed = candidate['wind_speed']
            
            # Score based on wind speed (higher wind = higher score)
            # Typical Arctic mesocyclones have wind speeds 10-25 m/s
            if wind_speed >= 20:
                return 1.0
            elif wind_speed >= 15:
                return 0.8
            elif wind_speed >= 12:
                return 0.6
            elif wind_speed >= 10:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating wind: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_pressure_gradient(self, candidate: Dict[str, Any], dataset: xr.Dataset) -> float:
        """
        Validate pressure gradient criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            
        Returns:
            Score between 0-1
        """
        try:
            # Extract pressure gradient information
            lat, lon = candidate['latitude'], candidate['longitude']
            
            # Get pressure at center
            pressure_vars = ['mean_sea_level_pressure', 'msl', 'psl', 'slp']
            pressure_var = None
            for var in pressure_vars:
                if var in dataset:
                    pressure_var = var
                    break
            
            if pressure_var is None:
                return 0.5  # Neutral score if pressure data unavailable
            
            # Extract region around candidate
            radius_deg = 1.0  # ~100km at mid-latitudes
            lat_min, lat_max = float(dataset.latitude.min()), float(dataset.latitude.max())
            lon_min, lon_max = float(dataset.longitude.min()), float(dataset.longitude.max())
            
            region_lat_min = max(lat - radius_deg, lat_min)
            region_lat_max = min(lat + radius_deg, lat_max)
            region_lon_min = max(lon - radius_deg, lon_min)
            region_lon_max = min(lon + radius_deg, lon_max)
            
            region = dataset.sel(
                latitude=slice(region_lat_min, region_lat_max),
                longitude=slice(region_lon_min, region_lon_max)
            )
            
            # Calculate pressure gradient using numpy gradient
            pressure_field = region[pressure_var].values
            
            if pressure_field.ndim < 2:
                return 0.5  # Cannot calculate gradient for 1D data
            
            # Calculate gradients
            dlat = np.gradient(pressure_field, axis=0)
            dlon = np.gradient(pressure_field, axis=1)
            
            # Calculate magnitude of pressure gradient
            grad_magnitude = np.sqrt(dlat**2 + dlon**2)
            
            # Get maximum gradient in the region
            max_gradient = np.max(grad_magnitude)
            
            # Convert to hPa/100km (approximate conversion)
            # This is a rough approximation - proper calculation would require
            # actual distance calculations based on latitude/longitude
            avg_lat = (region_lat_min + region_lat_max) / 2
            km_per_degree = 111.0 * np.cos(np.radians(avg_lat))
            gradient_hpa_per_100km = max_gradient * km_per_degree / 100.0
            
            # Score based on pressure gradient (higher gradient = higher score)
            # Typical Arctic mesocyclones have gradients 0.5-2.0 hPa/100km
            if gradient_hpa_per_100km >= 1.5:
                return 1.0
            elif gradient_hpa_per_100km >= 1.0:
                return 0.8
            elif gradient_hpa_per_100km >= 0.7:
                return 0.6
            elif gradient_hpa_per_100km >= 0.4:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating pressure gradient: {str(e)}")
            return 0.5  # Neutral score on error