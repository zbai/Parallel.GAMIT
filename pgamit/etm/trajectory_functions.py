# base_function.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np

# app
from pgamit.etm.etm_config import ParameterVector
from pgamit.etm.etm_config import ETMConfig
from pgamit.etm.solution_data import SolutionData

class EtmFunction(ABC):
    """Enhanced base class for all ETM function objects"""

    def __init__(self, solution_data: SolutionData, config: ETMConfig, **kwargs):
        self.p = ParameterVector(config.network_code, config.station_code)
        self.config = config

        # Core identification
        # @ todo: remove solution data and add to config: no real need for solution data in etm functions
        self.p.soln = solution_data.type
        self.p.stack = solution_data.stack_name

        # Function properties
        self.param_count = 0
        self.column_index = np.array([])
        self.format_str = ''
        self.fit = True
        self.design = np.array([])

        # time vector
        self._time_vector = np.array([])

        # Initialize specific parameters
        self.initialize(**kwargs)

        # Compute hash
        self.rehash()

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize function-specific parameters"""
        pass

    @abstractmethod
    def get_design_ts(self, ts: np.ndarray) -> np.ndarray:
        """Generate design matrix for given time series"""
        pass

    @abstractmethod
    def rehash(self) -> None:
        """Recompute hash value for change detection"""
        pass

    def eval(self, override_params: np.ndarray = np.array([])):
        if override_params:
            return self.design @ override_params
        else:
            return self.design @ self.p.params

    def load_parameters(self, params: np.ndarray, sigmas: np.ndarray, **kwargs) -> None:
        """Load estimated parameters and their uncertainties"""
        if params.ndim == 1:
            params = params.reshape((3, params.shape[0] // 3))
        if sigmas.ndim == 1:
            sigmas = sigmas.reshape((3, sigmas.shape[0] // 3))

        # Handle different parameter sources (LSQ X vector vs database)
        if params.shape[1] > self.param_count:
            # From LSQ X vector - use column indices
            self.p.params = params[:, self.column_index]
            self.p.sigmas = sigmas[:, self.column_index]
        else:
            # From database - direct assignment
            self.p.params = params
            self.p.sigmas = sigmas

    def validate_parameters(self) -> List[str]:
        """Validate parameter values and return issues"""
        issues = []

        if self.p.params.size > 0:
            if np.any(np.isnan(self.p.params)):
                issues.append(f"{self.__class__.__name__}: NaN values in parameters")
            if np.any(np.isinf(self.p.params)):
                issues.append(f"{self.__class__.__name__}: Infinite values in parameters")

        return issues

    def get_parameter_dict(self) -> Dict[str, Any]:
        """Get parameters as dictionary for serialization"""
        return {
            'NetworkCode': self.p.NetworkCode,
            'StationCode': self.p.StationCode,
            'soln': self.p.soln,
            'stack': self.p.stack,
            'object': self.p.object,
            'params': self.p.params.tolist() if self.p.params.size > 0 else [],
            'sigmas': self.p.sigmas.tolist() if self.p.sigmas.size > 0 else [],
            'metadata': self.p.metadata,
            'hash': self.p.hash,
            'param_count': self.param_count,
            'fit': self.fit
        }

    def configure_behavior(self, behavior_config: Dict[str, Any]) -> None:
        """Configure function behavior based on conditions"""
        # This method allows aggregating behavior-changing conditions
        # within each function object as requested

        for condition, value in behavior_config.items():
            if condition == 'fit_enabled':
                self.fit = bool(value)
            elif condition == 'custom_metadata':
                self.p.metadata = value
            elif hasattr(self, condition):
                setattr(self, condition, value)


