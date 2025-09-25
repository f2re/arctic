"""
Multi-criteria validator for Arctic cyclone detection.

Provides a system for validating cyclone candidates using multiple weighted criteria
to reduce false positives while maintaining sensitivity. Supports dynamic criteria
loading from YAML configuration.
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
    Supports dynamic criteria loading from YAML configuration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configurable criteria and weights from YAML config.
        
        Args:
            config: Configuration dictionary from YAML config file
        """
        # Default criteria configuration
        self.default_criteria_config = {
            'pressure_minimum': {
                'enabled': True,
                'weight': 0.25,
                'thresholds': {
                    'very_strong': 980,
                    'strong': 990,
                    'moderate': 1000,
                    'weak': 1010
                }
            },
            'vorticity': {
                'enabled': True,
                'weight': 0.30,
                'thresholds': {
                    'very_strong': 5e-5,
                    'strong': 3e-5,
                    'moderate': 1e-5,
                    'weak': 5e-6
                }
            },
            'size_filter': {
                'enabled': True,
                'weight': 0.15,
                'thresholds': {
                    'optimal_min': 100,
                    'optimal_max': 800,
                    'extended_min': 50,
                    'extended_max': 1000
                }
            },
            'wind_threshold': {
                'enabled': True,
                'weight': 0.20,
                'thresholds': {
                    'very_strong': 20,
                    'strong': 15,
                    'moderate': 12,
                    'weak': 10
                }
            },
            'pressure_gradient': {
                'enabled': True,
                'weight': 0.10,
                'thresholds': {
                    'very_strong': 1.5,
                    'strong': 1.0,
                    'moderate': 0.7,
                    'weak': 0.4
                }
            },
            'wind_850hPa': {
                'enabled': True,
                'weight': 0.15,
                'thresholds': {
                    'very_strong': 25.0,
                    'strong': 20.0,
                    'moderate': 15.0,
                    'weak': 10.0
                }
            },
            
        }
        
        # Load configuration from YAML
        self.criteria_config = self._load_config(config)
        
        # Extract weights for active criteria
        self.weights = self._extract_weights()
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.debug(f"Initialized multi-criteria validator with config: {self.criteria_config}")
        logger.debug(f"Active criteria weights: {self.weights}")

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load criteria configuration from YAML config.
        
        Args:
            config: Configuration dictionary from YAML config file
            
        Returns:
            Dictionary with criteria configuration
        """
        if not config or 'detection' not in config:
            logger.warning("No detection config found, using defaults")
            return self.default_criteria_config
            
        detection_config = config['detection']
        
        # Check if we have a dedicated multi_criteria_validator section
        if 'multi_criteria_validator' in detection_config:
            validator_config = detection_config['multi_criteria_validator']
            # Use thresholds from multi_criteria_validator section if available
            if 'thresholds' in validator_config:
                thresholds = validator_config['thresholds']
                # Update default criteria config with these thresholds
                for criterion_name in self.default_criteria_config:
                    if criterion_name in thresholds:
                        self.default_criteria_config[criterion_name]['thresholds'] = thresholds[criterion_name]
        
        # Load criteria configuration from the criteria section
        criteria_config = detection_config.get('criteria', {})
        final_config = {}
        
        # Merge with default configuration
        for criterion_name, default_config in self.default_criteria_config.items():
            # Check if criterion exists in config
            if criterion_name in criteria_config:
                criterion_settings = criteria_config[criterion_name]
                
                # Check if criterion is enabled
                is_enabled = criterion_settings.get('enabled', default_config['enabled'])
                
                # Extract weight
                weight = criterion_settings.get('weight', default_config['weight'])
                
                # Extract thresholds
                thresholds = default_config['thresholds'].copy()
                if 'thresholds' in criterion_settings:
                    thresholds.update(criterion_settings['thresholds'])
                elif 'threshold' in criterion_settings:
                    # Handle single threshold case
                    single_threshold = criterion_settings['threshold']
                    # Distribute single threshold to all levels proportionally
                    base_thresholds = default_config['thresholds']
                    if criterion_name == 'pressure_minimum':
                        thresholds['very_strong'] = single_threshold
                        thresholds['strong'] = single_threshold + 10
                        thresholds['moderate'] = single_threshold + 20
                        thresholds['weak'] = single_threshold + 30
                    elif criterion_name == 'vorticity':
                        thresholds['very_strong'] = single_threshold
                        thresholds['strong'] = single_threshold * 0.6
                        thresholds['moderate'] = single_threshold * 0.2
                        thresholds['weak'] = single_threshold * 0.1
                    elif criterion_name == 'wind':
                        thresholds['very_strong'] = single_threshold
                        thresholds['strong'] = single_threshold * 0.75
                        thresholds['moderate'] = single_threshold * 0.6
                        thresholds['weak'] = single_threshold * 0.5
                    elif criterion_name == 'pressure_gradient':
                        thresholds['very_strong'] = single_threshold
                        thresholds['strong'] = single_threshold * 0.66
                        thresholds['moderate'] = single_threshold * 0.46
                        thresholds['weak'] = single_threshold * 0.26
                    elif criterion_name == 'wind_850hPa':
                        thresholds['very_strong'] = single_threshold
                        thresholds['strong'] = single_threshold * 0.75
                        thresholds['moderate'] = single_threshold * 0.6
                        thresholds['weak'] = single_threshold * 0.5
                
                final_config[criterion_name] = {
                    'enabled': is_enabled,
                    'weight': weight,
                    'thresholds': thresholds
                }
            else:
                # Use default configuration
                final_config[criterion_name] = default_config
                
        return final_config

    def _extract_weights(self) -> Dict[str, float]:
        """
        Extract weights for active criteria.
        
        Returns:
            Dictionary of weights for active criteria
        """
        weights = {}
        for criterion_name, config in self.criteria_config.items():
            if config['enabled']:
                weights[criterion_name] = config['weight']
        return weights

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
        active_criteria_count = 0
        
        # Apply each active criterion
        for criterion_name, config in self.criteria_config.items():
            if config['enabled']:
                active_criteria_count += 1
                try:
                    # Call the appropriate validation method
                    method_name = f"_validate_{criterion_name}"
                    if hasattr(self, method_name):
                        method = getattr(self, method_name)
                        scores[criterion_name] = method(candidate, dataset, config['thresholds'])
                    else:
                        logger.warning(f"Validation method {method_name} not found")
                        scores[criterion_name] = 0.5  # Neutral score
                except Exception as e:
                    logger.debug(f"Error validating {criterion_name}: {str(e)}")
                    scores[criterion_name] = 0.5  # Neutral score on error
        
        # If no criteria are active, accept all candidates
        if active_criteria_count == 0:
            logger.warning("No active criteria found, accepting all candidates")
            return True, 1.0, {}
        
        # Calculate weighted score
        total_score = sum(scores[criterion] * self.weights[criterion] 
                         for criterion in scores.keys() if criterion in self.weights)
        
        # Candidate is valid if score exceeds threshold (e.g., 0.6)
        is_valid = total_score >= 0.6
        
        return is_valid, total_score, scores

    def _validate_pressure_minimum(self, candidate: Dict[str, Any], dataset: xr.Dataset, thresholds: Dict[str, float]) -> float:
        """
        Validate pressure minimum criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            thresholds: Threshold values for scoring
            
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
            if pressure <= thresholds['very_strong']:
                return 1.0
            elif pressure <= thresholds['strong']:
                return 0.8
            elif pressure <= thresholds['moderate']:
                return 0.6
            elif pressure <= thresholds['weak']:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating pressure minimum: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_vorticity(self, candidate: Dict[str, Any], dataset: xr.Dataset, thresholds: Dict[str, float]) -> float:
        """
        Validate vorticity criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            thresholds: Threshold values for scoring
            
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
            abs_vorticity = abs(vorticity)
            if abs_vorticity >= thresholds['very_strong']:
                return 1.0
            elif abs_vorticity >= thresholds['strong']:
                return 0.8
            elif abs_vorticity >= thresholds['moderate']:
                return 0.6
            elif abs_vorticity >= thresholds['weak']:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating vorticity: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_size_filter(self, candidate: Dict[str, Any], dataset: xr.Dataset, thresholds: Dict[str, float]) -> float:
        """
        Validate size filtering criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            thresholds: Threshold values for scoring
            
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
            if thresholds['optimal_min'] <= diameter_km <= thresholds['optimal_max']:
                return 1.0  # Perfect size
            elif thresholds['extended_min'] <= diameter_km <= thresholds['extended_max']:
                # Score decreases as we move away from optimal range
                if diameter_km < thresholds['optimal_min']:
                    # Too small but within extended range
                    return 0.5 + 0.5 * (diameter_km - thresholds['extended_min']) / (thresholds['optimal_min'] - thresholds['extended_min'])
                else:
                    # Too large but within extended range
                    return 0.5 + 0.5 * (thresholds['extended_max'] - diameter_km) / (thresholds['extended_max'] - thresholds['optimal_max'])
            else:
                return 0.0  # Outside acceptable range
                
        except Exception as e:
            logger.debug(f"Error validating size: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_wind_threshold(self, candidate: Dict[str, Any], dataset: xr.Dataset, thresholds: Dict[str, float]) -> float:
        """
        Validate wind threshold criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            thresholds: Threshold values for scoring
            
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
            if wind_speed >= thresholds['very_strong']:
                return 1.0
            elif wind_speed >= thresholds['strong']:
                return 0.8
            elif wind_speed >= thresholds['moderate']:
                return 0.6
            elif wind_speed >= thresholds['weak']:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating wind: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_pressure_gradient(self, candidate: Dict[str, Any], dataset: xr.Dataset, thresholds: Dict[str, float]) -> float:
        """
        Validate pressure gradient criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            thresholds: Threshold values for scoring
            
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
            if gradient_hpa_per_100km >= thresholds['very_strong']:
                return 1.0
            elif gradient_hpa_per_100km >= thresholds['strong']:
                return 0.8
            elif gradient_hpa_per_100km >= thresholds['moderate']:
                return 0.6
            elif gradient_hpa_per_100km >= thresholds['weak']:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating pressure gradient: {str(e)}")
            return 0.5  # Neutral score on error

    def _validate_wind_850hPa(self, candidate: Dict[str, Any], dataset: xr.Dataset, thresholds: Dict[str, float]) -> float:
        """
        Validate wind speed at 850 hPa criterion.
        
        Args:
            candidate: Candidate dictionary
            dataset: Meteorological dataset
            thresholds: Threshold values for scoring
            
        Returns:
            Score between 0-1
        """
        try:
            # Check if wind speed at 850 hPa is already calculated in candidate
            if 'wind_speed_850hPa' in candidate and candidate['wind_speed_850hPa'] is not None:
                wind_speed = candidate['wind_speed_850hPa']
            else:
                # Try to extract wind from dataset at 850 hPa
                lat, lon = candidate['latitude'], candidate['longitude']
                
                # Check for pressure level wind data
                wind_var_pairs = [
                    ('u', 'v'),
                    ('u_component_of_wind', 'v_component_of_wind')
                ]
                
                u_wind_var = None
                v_wind_var = None
                
                for u_var, v_var in wind_var_pairs:
                    if u_var in dataset and v_var in dataset:
                        u_wind_var, v_wind_var = u_var, v_var
                        break
                
                if u_wind_var is None or v_wind_var is None:
                    return 0.5  # Neutral score if wind data unavailable
                
                # Look for pressure levels
                pressure_levels = ['level', 'pressure_level', 'lev', 'plev']
                pressure_level_dim = None
                for level_name in pressure_levels:
                    if level_name in dataset.dims:
                        pressure_level_dim = level_name
                        break
                
                if pressure_level_dim is None:
                    # No pressure levels in dataset, use surface wind or return neutral score
                    return 0.5
                
                # Find 850 hPa level or closest available
                levels = dataset[pressure_level_dim].values
                closest_level = min(levels, key=lambda x: abs(x - 850))
                
                # Get wind components at 850 hPa
                u_point = float(dataset[u_wind_var].sel(
                    {pressure_level_dim: closest_level}, 
                    latitude=lat, 
                    longitude=lon, 
                    method='nearest'
                ).values)
                v_point = float(dataset[v_wind_var].sel(
                    {pressure_level_dim: closest_level}, 
                    latitude=lat, 
                    longitude=lon, 
                    method='nearest'
                ).values)
                
                wind_speed = np.sqrt(u_point**2 + v_point**2)
                
                # Store for future use
                candidate['wind_speed_850hPa'] = wind_speed
                candidate['u_wind_850hPa'] = u_point
                candidate['v_wind_850hPa'] = v_point

            # Score based on wind speed (higher wind = higher score)
            if wind_speed >= thresholds['very_strong']:
                return 1.0
            elif wind_speed >= thresholds['strong']:
                return 0.8
            elif wind_speed >= thresholds['moderate']:
                return 0.6
            elif wind_speed >= thresholds['weak']:
                return 0.4
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error validating wind at 850 hPa: {str(e)}")
            return 0.5  # Neutral score on error

    