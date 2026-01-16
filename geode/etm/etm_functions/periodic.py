import numpy as np
from typing import Dict, Any, Tuple, Union, List
import logging

logger = logging.getLogger(__name__)

# app
from ...Utils import crc32
from ..etm_functions.etm_function import EtmFunction
from ..core.etm_config import EtmConfig


class PeriodicFunction(EtmFunction):
    """Enhanced periodic function with improved frequency management"""
    def __init__(self, config: EtmConfig, **kwargs):
        self.dt_max = np.inf
        super().__init__(config, **kwargs)

    def initialize(self, time_vector: np.ndarray = np.array([]), **kwargs) -> None:
        """Initialize periodic-specific parameters"""
        self.p.object = 'periodic'
        self._time_vector = time_vector

        # Load frequencies from configuration or database
        if time_vector.size > 1:
            self._analyze_data_gaps(time_vector)
            self._filter_fittable_frequencies(self.config.modeling.frequencies)
        else:
            self.param_count = 0
            self.fit = False
            self.p.frequencies = np.array([])

        logger.info(f'Periodic -> Frequency count: {len(self.p.frequencies)}; FitPeriodic: {self.fit}')

        self.design = self.get_design_ts(time_vector)

        self.p.metadata = f'periodic:{len(self.p.frequencies)}'

        params = (','.join([f's{i}' for i in range(int(self.param_count/2))]) + ',' +
                  ','.join([f'c{i}' for i in range(int(self.param_count/2))]))

        self.p.param_metadata = f'periodic:[n:[{params}]],[e:[{params}]],[u:[{params}]]]'

    def _analyze_data_gaps(self, time_vector: np.ndarray) -> None:
        """Analyze data gaps to determine fittable frequencies"""
        # wrap around the solutions
        wt = np.sort(np.unique(time_vector - np.fix(time_vector)))
        if wt.size < 2:
            # to handle a call with a single epoch on stations without periodic
            self.dt_max = np.inf
            return
        # analyze the gaps in the data
        dt = np.diff(wt)
        # max dt (internal)
        dtmax = np.max(dt)
        # dt wrapped around
        dt_interyr = 1 - wt[-1] + wt[0]
        if dt_interyr > dtmax:
            dtmax = dt_interyr
        # save the value of the max wrapped delta time
        self.dt_max = dtmax

    def _filter_fittable_frequencies(self, new_freqs: np.ndarray) -> None:
        """Filter frequencies based on data availability"""
        # get the 50 % of Nyquist for each component (and convert to average fyear)
        nyquist_limits = ((1 / new_freqs) / 2.) * 0.5 * 1 / 365.25
        fittable_mask = self.dt_max <= nyquist_limits

        self.param_count = int(np.sum(fittable_mask)) * 2
        self.p.frequencies = self.config.modeling.frequencies[fittable_mask]

        if self.param_count == 0:
            self.fit = False

    def get_design_ts(self, time_vector: np.ndarray) -> np.ndarray:
        """Generate design matrix for periodic terms"""
        if self.param_count == 0:
            if time_vector.size > 0:
                # maybe the object was initialized without time_vector, try analyzing gaps again
                self._analyze_data_gaps(time_vector)
                self._filter_fittable_frequencies(self.config.modeling.frequencies)
            # if still unable to fit, then return nothing
            if self.param_count == 0:
                return np.array([]).reshape(time_vector.size, 0)

        # Create frequency matrix
        f_matrix = np.tile(self.p.frequencies, (time_vector.shape[0], 1))
        t_matrix = np.tile(time_vector[:, np.newaxis], (1, len(self.p.frequencies)))

        # Calculate sin and cos components
        sin_components = np.sin(2 * np.pi * f_matrix * 365.25 * t_matrix)
        cos_components = np.cos(2 * np.pi * f_matrix * 365.25 * t_matrix)

        return np.column_stack((sin_components, cos_components))

    def get_periodic_cols(self, frequency: Union[float, List, np.ndarray] = None,
                          return_col_of_design_matrix: bool = True) -> List:
        """
        method to retrieve the column of a given frequency (or all if None)
        if return_col_of_design_matrix then the returned index if that of the ETM design matrix
        if not return_col_of_design_matrix then the index is that
        returns list of sin cols (cos is sin + params / 2) in the order passed to the method
        """
        if frequency is None:
            frequency = self.p.frequencies
        elif isinstance(frequency, float):
            frequency = [frequency]

        sin_cols = []
        for freq in frequency:
            idx = np.where(np.isin(self.p.frequencies, freq))[0].tolist()

            if len(idx) and self.fit:
                if return_col_of_design_matrix:
                    sin_cols.append(self.column_index[idx[0]])
                else:
                    sin_cols.append(idx[0])

        return sin_cols

    def print_parameters(self) -> Tuple[list, list, list]:

        periods = (1 / (self.p.frequencies * 365.25)).tolist()
        periods_str = ' '.join('%.1f yr' % i for i in periods)

        self.format_str = (self.config.get_label('periodic') + f' ({periods_str})')
        if self.p.params:
            params = np.array(self.p.params) * 1000.

            freq = int(self.param_count / 2)
            amplitude = np.zeros((3, freq))
            for i in range(freq):
                amplitude[:, i] = np.sqrt(np.sum(np.square(params[:, i::freq]), axis=1))

            Na = ' '.join([f'{amp:.2f}' for amp in amplitude[0]])
            Ea = ' '.join([f'{amp:.2f}' for amp in amplitude[1]])
            Ua = ' '.join([f'{amp:.2f}' for amp in amplitude[2]])
        else:
            Na = Ea = Ua = 0

        self.format_str += f' N: ({Na}) E: ({Ea}) U: ({Ua}) [mm]'

        return [self.format_str], [], []

    def rehash(self) -> None:
        """Recompute hash for change detection"""
        freq_str = ','.join(f'{f:.6f}' for f in self.p.frequencies)
        self.p.hash = crc32(freq_str)

    def configure_behavior(self, behavior_config: Dict[str, Any]) -> None:
        """Configure periodic-specific behavior"""
        super().configure_behavior(behavior_config)

        if 'custom_frequencies' in behavior_config:
            new_freqs = np.array(behavior_config['custom_frequencies'])
            if len(new_freqs) != len(self.p.frequencies):
                self._filter_fittable_frequencies(new_freqs)
                self.param_count = self.p.frequencies.size * 2
                self.design = self.get_design_ts(self._time_vector)

        self.rehash()

    def short_name(self) -> str:
        name = []
        if self.p.frequencies is not None:
            for i in range(self.p.frequencies.size):
                name.append(f'{"FREQ " + f"S{i:d}":>10}')
            for i in range(self.p.frequencies.size):
                name.append(f'{"FREQ " + f"C{i:d}":>10}')

        return ' '.join(name)

    def __str__(self) -> str:
        """String representation for debugging"""
        out_str = [f"param count: {self.param_count}"]
        if self.p.frequencies is not None:
            for f in self.p.frequencies:
                out_str.append(f'period: {1/f:7.3f} days')

        return '; '.join(out_str)

    def __repr__(self) -> str:
        return f"Periodic({str(self)})"