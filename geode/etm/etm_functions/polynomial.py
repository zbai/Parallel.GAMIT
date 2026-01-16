import numpy as np
from typing import Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

# app
from ...Utils import crc32
from ..core.etm_config import EtmConfig
from ..etm_functions.etm_function import EtmFunction


class PolynomialFunction(EtmFunction):
    """Enhanced polynomial function with improved configuration management"""

    def __init__(self, config: EtmConfig, **kwargs):
        super().__init__(config, **kwargs)

    def initialize(self, models: tuple = (),
                   time_vector: np.ndarray = np.array([]),
                   **kwargs) -> None:
        """Initialize polynomial-specific parameters"""

        self.p.object = 'polynomial'

        # Set reference time
        if self.config.modeling.reference_epoch == 0 and time_vector.size > 0:
            self.config.modeling.reference_epoch = float(np.min(time_vector))
        elif self.config.modeling.reference_epoch == 0:
            self.config.modeling.reference_epoch = 2015.0

        self.p.t_ref = self.config.modeling.reference_epoch

        # Initialize design matrix if time vector available
        self._time_vector = time_vector
        self.design = self.get_design_ts(time_vector)

        # fill the parameter and sigma vectors with nans. This helps the creation of
        # empty objects for constraints
        for j in range(3):
            self.p.params[j] = np.array([np.nan] * self.config.modeling.poly_terms)
            self.p.sigmas[j] = np.array([np.nan] * self.config.modeling.poly_terms)

        logger.info(f'Polynomial -> Fitting {self.config.modeling.poly_terms} term(s), conventional '
                    f'epoch {self.p.t_ref:.3f}')

        self.p.metadata = f'polynomial:{self.param_count}'

        params = ','.join([f'p{i}' for i in range(self.param_count)])

        self.p.param_metadata = f'polynomial:[n:[{params}]],[e:[{params}]],[u:[{params}]]]'

    def get_design_ts(self, time_vector: np.ndarray) -> np.ndarray:
        """Generate design matrix for polynomial terms"""
        if time_vector.size == 0:
            return np.array([])

        a = np.zeros((time_vector.size, self.config.modeling.poly_terms))
        self.param_count = 0
        for p in range(self.config.modeling.poly_terms):
            a[:, p] = np.power(time_vector - self.p.t_ref, p)
            self.param_count += 1

        return a

    def print_parameters(self, ce_position: List[np.ndarray] = None) -> Tuple[list, list, list]:
        self.format_str = []

        params = np.array(self.p.params) * 1000.
        sigmas = np.array(self.p.sigmas) * 1000.

        self.format_str = (self.config.get_label('position') +
                           f' ({self.p.t_ref:.3f}) '
                           f'X: {params[0, 0] if ce_position is None else ce_position[0][0]:.3f} '
                           f'Y: {params[1, 0] if ce_position is None else ce_position[1][0]:.3f} '
                           f'Z: {params[2, 0] if ce_position is None else ce_position[2][0]:.3f} [m]')

        if self.param_count > 1:
            self.format_str += ('\n' + self.config.get_label('velocity') + ' '
                                f'N: {params[0, 1]:.2f} $\pm$ {sigmas[0, 1]:.2f} '
                                f'E: {params[1, 1]:.2f} $\pm$ {sigmas[1, 1]:.2f} '
                                f'U: {params[2, 1]:.2f} $\pm$ {sigmas[2, 1]:.2f} [mm/yr]')

        if self.param_count > 2:
            self.format_str += ('\n' + self.config.get_label('acceleration') + ' '
                                f'N: {params[0, 2]:.2f} $\pm$ {sigmas[0, 2]:.2f} '
                                f'E: {params[1, 2]:.2f} $\pm$ {sigmas[1, 2]:.2f} '
                                f'U: {params[2, 2]:.2f} $\pm$ {sigmas[2, 2]:.2f} [mm/yr**2]')

        if self.param_count > 3:
            self.format_str += f' + {self.param_count - 3} ' + self.config.get_label('other')

        return [self.format_str], [], []

    def rehash(self) -> None:
        """Recompute hash for change detection"""
        hash_input = f"{self.config.modeling.poly_terms}_{self.config.modeling.reference_epoch}"
        self.p.hash = crc32(hash_input)

    def configure_behavior(self, behavior_config: Dict[str, Any]) -> None:
        """Configure polynomial-specific behavior"""
        super().configure_behavior(behavior_config)

        logger.debug(f'configuring behavior {behavior_config}')

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

    def short_name(self) -> str:
        name = []
        for p in range(self.param_count):
            name.append(f'{"POLY " + f"P{p:d}":>10}')

        return ' '.join(name)

    def __str__(self) -> str:
        """String representation for debugging"""
        return f"param count: {self.param_count}"

    def __repr__(self) -> str:
        return f"Polynomial({str(self)})"