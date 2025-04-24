"""
Subpackage of detection criteria for the ArcticCyclone system.

Contains base and specialized criteria for detecting
arctic mesocyclones in meteorological data.
"""

from typing import Dict, List, Any, Optional, Type, Union
import xarray as xr
import logging
from abc import ABC, abstractmethod
import importlib
import inspect

# Initialize logger
logger = logging.getLogger(__name__)

class BaseCriterion(ABC):
    """
    Base abstract class for all cyclone detection criteria.
    
    Defines the interface that all concrete criteria must implement.
    """
    
    @abstractmethod
    def apply(self, dataset: xr.Dataset, time_step: Any) -> List[Dict]:
        """
        Applies the criterion to a dataset.
        
        Arguments:
            dataset: xarray meteorological dataset.
            time_step: Time step for analysis.
            
        Returns:
            List of cyclone candidates (dictionaries with coordinates and properties).
        """
        pass
    
    @property
    def name(self) -> str:
        """
        Returns the criterion name.
        
        Returns:
            String with the criterion name.
        """
        return self.__class__.__name__
    
    @property
    def description(self) -> str:
        """
        Returns the criterion description.
        
        Returns:
            String with the criterion description.
        """
        return self.__doc__ or "Cyclone detection criterion"


class CriteriaManager:
    """
    Manages cyclone detection criteria.
    
    Provides methods for registering, activating, and applying
    various cyclone detection criteria.
    """
    
    def __init__(self):
        """
        Initializes the criteria manager.
        """
        self.criteria = {}  # Dictionary of registered criteria
        self.active_criteria = []  # List of active criteria
        
        logger.info("Initialized cyclone detection criteria manager")
    
    def register_criterion(self, name: str, criterion_class) -> None:
        """
        Registers a new detection criterion.
        
        Arguments:
            name: Criterion name for registration.
            criterion_class: Criterion class (not instance).
            
        Raises:
            ValueError: If a criterion with the same name is already registered.
        """
        if name in self.criteria:
            raise ValueError(f"Criterion with name '{name}' is already registered")
            
        self.criteria[name] = criterion_class
        logger.info(f"Registered criterion: {name}")
    
    def unregister_criterion(self, name: str) -> None:
        """
        Removes a criterion registration.
        
        Arguments:
            name: Criterion name to remove.
            
        Raises:
            ValueError: If the criterion with the given name is not registered.
        """
        if name not in self.criteria:
            raise ValueError(f"Criterion with name '{name}' is not registered")
            
        # Remove from active criteria if it's there
        if name in self.active_criteria:
            self.active_criteria.remove(name)
            
        del self.criteria[name]
        logger.info(f"Unregistered criterion: {name}")
    
    def get_criterion(self, name: str) -> BaseCriterion:
        """
        Returns a registered criterion by name.
        
        Arguments:
            name: Criterion name.
            
        Returns:
            Criterion instance.
            
        Raises:
            ValueError: If the criterion with the given name is not registered.
        """
        if name not in self.criteria:
            raise ValueError(f"Criterion with name '{name}' is not registered")
            
        return self.criteria[name]()
    
    def set_active_criteria(self, names: List[str]) -> None:
        """
        Sets active detection criteria.
        
        Arguments:
            names: List of criterion names to activate.
            
        Raises:
            ValueError: If any criterion is not registered.
        """
        # Check that all criteria are registered
        for name in names:
            if name not in self.criteria:
                raise ValueError(f"Criterion with name '{name}' is not registered")
                
        self.active_criteria = names
        logger.info(f"Set active criteria: {', '.join(names)}")
    
    def get_active_criteria(self) -> List:
        """
        Returns a list of active criterion classes.
        
        Returns:
            List of active criterion classes.
        """
        return [self.criteria[name] for name in self.active_criteria]
    
    def get_active_criterion_names(self) -> List[str]:
        """
        Returns a list of active criterion names.
        
        Returns:
            List of active criterion names.
        """
        return self.active_criteria.copy()
    
    def apply_criteria(self, dataset: xr.Dataset, time_step: Any) -> Dict[str, List[Dict]]:
        """
        Applies all active criteria to a dataset.
        
        Arguments:
            dataset: xarray meteorological dataset.
            time_step: Time step for analysis.
            
        Returns:
            Dictionary with results from applying each criterion.
        """
        results = {}
        
        for name in self.active_criteria:
            criterion = self.criteria[name]()
            try:
                candidates = criterion.apply(dataset, time_step)
                results[name] = candidates
                logger.debug(f"Criterion {name} found {len(candidates)} candidates")
            except Exception as e:
                logger.error(f"Error applying criterion {name}: {str(e)}")
                results[name] = []
        
        return results
    
    def list_available_criteria(self) -> Dict[str, str]:
        """
        Returns a list of available criteria with their descriptions.
        
        Returns:
            Dictionary with criterion names and their descriptions.
        """
        return {name: criterion.__doc__ or "Cyclone detection criterion" for name, criterion in self.criteria.items()}


# Import concrete criteria
from .pressure import PressureMinimumCriterion
from .vorticity import VorticityCriterion
from .gradient import PressureGradientCriterion
from .closed_contour import ClosedContourCriterion
from .wind import WindThresholdCriterion
from .laplacian import PressureLaplacianCriterion

__all__ = [
    'BaseCriterion',
    'CriteriaManager',
    'PressureMinimumCriterion',
    'VorticityCriterion',
    'PressureGradientCriterion',
    'ClosedContourCriterion',
    'WindThresholdCriterion',
    'PressureLaplacianCriterion'
]