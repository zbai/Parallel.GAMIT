import numpy as np
from typing import Dict, Any

# app
from pgamit.Utils import crc32
from pgamit.etm.solution_data import SolutionData
from pgamit.etm.etm_config import ETMConfig
from pgamit.etm.trajectory_functions import EtmFunction


class PolynomialFunction(EtmFunction):
    """Enhanced polynomial function with improved configuration management"""

    def __init__(self, solution_data: SolutionData, config: ETMConfig, **kwargs):
        super().__init__(solution_data, config, **kwargs)
        self.p.object = 'polynomial'

    def initialize(self, models: tuple = (), _time_vector: np.ndarray = np.array([])) -> None:
        """Initialize polynomial-specific parameters"""

        # Set reference time
        if self.config.modeling.reference_epoch == 0:
            self.config.modeling.reference_epoch = np.min(self._time_vector)
            self.p.t_ref = self.config.modeling.reference_epoch

        # Initialize design matrix if time vector available
        self._time_vector = _time_vector
        self.design = self.get_design_ts(self._time_vector)

    def get_design_ts(self, ts: np.ndarray) -> np.ndarray:
        """Generate design matrix for polynomial terms"""
        if ts.size == 0:
            return np.array([])

        a = np.zeros((ts.size, self.config.modeling.poly_terms))
        self.param_count = 0
        for p in range(self.config.modeling.poly_terms):
            a[:, p] = np.power(ts - self.p.t_ref, p)
            self.param_count += 1

        return a

    def rehash(self) -> None:
        """Recompute hash for change detection"""
        hash_input = f"{self.config.modeling.poly_terms}_{self.config.modeling.reference_epoch}"
        self.p.hash = crc32(hash_input)

    def configure_behavior(self, behavior_config: Dict[str, Any]) -> None:
        """Configure polynomial-specific behavior"""
        super().configure_behavior(behavior_config)

        if 'reference_epoch' in behavior_config:
            new_ref = behavior_config['reference_epoch']
            if isinstance(new_ref, (int, float)):
                self.p.t_ref = float(new_ref)
            elif hasattr(new_ref, 'fyear'):  # pyDate.Date object
                self.p.t_ref = new_ref.fyear

        if 'polynomial_terms' in behavior_config:
            new_terms = int(behavior_config['polynomial_terms'])
            if new_terms != self.config.modeling.poly_terms and new_terms > 0:
                self.config.modeling.poly_terms = new_terms
                self.design = self.get_design_ts(self._time_vector)
                self.rehash()