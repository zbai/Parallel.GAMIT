import numpy as np
from typing import Dict, Any

# app
from pgamit.Utils import crc32
from pgamit.etm.trajectory_functions import EtmFunction
from pgamit.etm.solution_data import SolutionData
from pgamit.etm.etm_config import ETMConfig


class PeriodicFunction(EtmFunction):
    """Enhanced periodic function with improved frequency management"""
    def __init__(self, solution_data: SolutionData, config: ETMConfig, **kwargs):
        super().__init__(solution_data, config, **kwargs)
        self.p.object = 'periodic'

    def initialize(self, time_vector: np.ndarray, **kwargs) -> None:
        """Initialize periodic-specific parameters"""
        self._time_vector = time_vector

        # Load frequencies from configuration or database
        self.p.frequencies = self.config.modeling.frequencies
        self.requested_frequencies = kwargs.get('requested_frequencies', np.array([]))

        if time_vector.size > 1:
            self._analyze_data_gaps(time_vector)
            self._filter_fittable_frequencies()
        else:
            self.frequency_count = 0
            self.p.frequencies = np.array([])
            self.dt_max = 1.0

        self.param_count = self.frequency_count * 2  # sin and cos for each frequency
        self.design = self.get_design_ts(time_vector) if time_vector.size > 0 else np.array([])
        self._setup_format_string()
        self._setup_metadata()

    def _analyze_data_gaps(self, time_vector: np.ndarray) -> None:
        """Analyze data gaps to determine fittable frequencies"""
        # Wrap time around to find largest gap
        wrapped_time = np.sort(np.unique(time_vector - np.fix(time_vector)))
        dt_internal = np.max(np.diff(wrapped_time))
        dt_wrap = 1 - wrapped_time[-1] + wrapped_time[0]

        self.dt_max = max(dt_internal, dt_wrap)

    def _filter_fittable_frequencies(self) -> None:
        """Filter frequencies based on data availability"""
        # Calculate Nyquist criterion for each frequency
        nyquist_limits = ((1 / self.p.frequencies) / 2.0) * 0.5 * (1 / 365.25)
        fittable_mask = self.dt_max <= nyquist_limits

        self.frequency_count = int(np.sum(fittable_mask))
        self.p.frequencies = self.p.frequencies[fittable_mask]

    def _setup_format_string(self) -> None:
        """Setup format string for parameter printing"""
        if self.frequency_count > 0:
            periods = 1 / (self.p.frequencies * 365.25)
            period_str = ', '.join(f'{p:.1f} yr' for p in periods)
            self.format_str = f"{self.config.get_label('periodic')} ({period_str}) N: %s E: %s U: %s [mm]"
        else:
            self.format_str = f"No {self.config.get_label('periodic').lower()} terms"

    def _setup_metadata(self) -> None:
        """Setup metadata description"""
        if self.frequency_count == 0:
            self.p.metadata = '[[],[],[]]'
            return

        metadata_parts = []
        for comp in ('n', 'e', 'u'):
            comp_meta = []
            for trig_func in ('sin', 'cos'):
                for freq in self.p.frequencies:
                    period = 1 / (freq * 365.25)
                    comp_meta.append(f'{comp}:{trig_func}({period:.1f} yr)')
            metadata_parts.append('[' + ','.join(comp_meta) + ']')

        self.p.metadata = '[' + ','.join(metadata_parts) + ']'

    def get_design_ts(self, ts: np.ndarray) -> np.ndarray:
        """Generate design matrix for periodic terms"""
        if self.frequency_count == 0 or ts.size == 0:
            return np.array([]).reshape(ts.size, 0)

        # Create frequency matrix
        f_matrix = np.tile(self.p.frequencies, (ts.shape[0], 1))
        t_matrix = np.tile(ts[:, np.newaxis], (1, len(self.p.frequencies)))

        # Calculate sin and cos components
        sin_components = np.sin(2 * np.pi * f_matrix * 365.25 * t_matrix)
        cos_components = np.cos(2 * np.pi * f_matrix * 365.25 * t_matrix)

        return np.column_stack((sin_components, cos_components))

    def print_parameters(self) -> str:
        """Generate formatted parameter string"""
        if self.p.params.size == 0 or self.frequency_count == 0:
            return "No periodic parameters estimated"

        # Reshape parameters: [sin_freq1, sin_freq2, ..., cos_freq1, cos_freq2, ...]
        n_params = self.p.params[0, :]
        e_params = self.p.params[1, :]
        u_params = self.p.params[2, :]

        # Split into sin and cos components
        mid_point = self.frequency_count
        n_sin, n_cos = n_params[:mid_point], n_params[mid_point:]
        e_sin, e_cos = e_params[:mid_point], e_params[mid_point:]
        u_sin, u_cos = u_params[:mid_point], u_params[mid_point:]

        # Calculate amplitudes for each frequency
        n_amp = np.sqrt(n_sin ** 2 + n_cos ** 2) * 1000  # Convert to mm
        e_amp = np.sqrt(e_sin ** 2 + e_cos ** 2) * 1000
        u_amp = np.sqrt(u_sin ** 2 + u_cos ** 2) * 1000

        return self.format_str % (
            np.array_str(n_amp, precision=1),
            np.array_str(e_amp, precision=1),
            np.array_str(u_amp, precision=1)
        )

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
                self.p.frequencies = new_freqs
                self._filter_fittable_frequencies()
                self.param_count = self.frequency_count * 2
                self._setup_format_string()
                self._setup_metadata()
                if hasattr(self, '_time_vector'):
                    self.design = self.get_design_ts(self._time_vector)
                self.rehash()
