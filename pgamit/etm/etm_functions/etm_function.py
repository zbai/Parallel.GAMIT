# base_function.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np

from etm.core.etm_config import JumpType
# app
from etm.core.etm_config import ETMConfig

@dataclass
class ParameterVector:
    # station and solution identification
    NetworkCode: str
    StationCode: str
    # Parameter storage
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    params: List[np.ndarray] = field(default_factory=lambda: [np.array([]), np.array([]), np.array([])])
    sigmas: List[np.ndarray] = field(default_factory=lambda: [np.array([]), np.array([]), np.array([])])
    covar: List[np.ndarray] = field(default_factory=lambda: [np.array([]), np.array([]), np.array([])])

    soln: str = 'ppp'
    stack: str = 'ppp'
    object: str = ''
    metadata: Optional[str] = None
    hash: int = 0
    jump_date: datetime = datetime(1980, 1, 1)
    jump_type: JumpType = JumpType.UNDETERMINED
    t_ref: float = 0


class EtmFunction(ABC):
    """Enhanced base class for all ETM function objects"""

    def __init__(self, config: ETMConfig,
                 metadata: Optional[str] = 'Generic ETM function',
                 **kwargs):

        self.p = ParameterVector(config.network_code, config.station_code)
        self.config = config

        # Core identification
        self.p.soln = config.solution.soln
        self.p.stack = config.solution.stack_name
        self.p.metadata = metadata

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

    def eval(self, component: int,
             override_time_vector: np.ndarray = None,
             override_params: np.ndarray = None):

        if override_time_vector is not None:
            design = self.get_design_ts(override_time_vector)
        else:
            design = self.design

        if override_params is not None:
            return design @ override_params[component]
        else:
            return design @ self.p.params[component]

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

    @abstractmethod
    def print_parameters(self) -> Tuple[list, list, list]:
        pass

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


