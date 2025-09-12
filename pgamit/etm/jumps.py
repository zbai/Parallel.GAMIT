from datetime import datetime
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np

# app
from pgamit.Utils import crc32
from pgamit import pyDate
from pgamit.etm.trajectory_functions import EtmFunction
from pgamit.etm.etm_config import JumpType
from pgamit.etm.etm_config import ETMConfig

class JumpFunction(EtmFunction):
    """Enhanced jump function with improved management and validation"""

    def initialize(self, time_vector: np.ndarray, date: Union[pyDate.Date, datetime],
                   jump_type: JumpType = JumpType.MECHANICAL_MANUAL,
                   metadata: str = "", action: str = 'A',
                   relaxation: Optional[np.ndarray] = None,
                   magnitude: float = np.nan, epi_distance: float = 0.0,
                   **kwargs) -> None:
        """Initialize jump-specific parameters"""
        self.p.object = 'jump'
        self._time_vector = time_vector

        # Jump identification
        self.date = date if isinstance(date, pyDate.Date) else pyDate.Date(datetime=date)
        self.p.jump_date = self.date.datetime()
        self.p.jump_type = jump_type
        self.p.metadata = metadata

        # Jump properties
        self.action = action  # 'A'uto, '+'add, '-'remove, 'M'anual
        self.magnitude = magnitude
        self.epi_distance = epi_distance

        # Relaxation parameters for postseismic jumps
        self.p.relaxation = self._setup_relaxation(relaxation, jump_type)

        # Parameter counting and design matrix setup
        self._setup_parameter_count()
        self.design = self._create_design_matrix(time_vector) if self.fit else np.array([])

        # Validation and final setup
        self._validate_jump_configuration(time_vector)

    def _setup_relaxation(self, relaxation: Optional[np.ndarray], jump_type: JumpType) -> np.ndarray:
        """Setup relaxation parameters based on jump type"""
        if jump_type in (JumpType.COSEISMIC_JUMP_DECAY, JumpType.POSTSEISMIC_ONLY):
            if relaxation is None:
                return self.config.modeling.default_relaxation.copy()
            elif isinstance(relaxation, (list, tuple)):
                return np.array(relaxation)
            elif isinstance(relaxation, (int, float)):
                return np.array([relaxation])
            else:
                return relaxation.copy()
        else:
            return np.array([])

    def _setup_parameter_count(self) -> None:
        """Setup parameter count based on jump type and relaxation"""
        self.param_count = 1  # Base jump parameter

        if self.p.jump_type in (JumpType.COSEISMIC_JUMP_DECAY, JumpType.POSTSEISMIC_ONLY):
            self.param_count += len(self.p.relaxation)  # Add relaxation parameters

        if self.p.jump_type == JumpType.POSTSEISMIC_ONLY:
            self.param_count -= 1  # No instantaneous jump, only decay

        self.nr = len(self.p.relaxation)  # Number of relaxation terms

    def _create_design_matrix(self, time_vector: np.ndarray) -> np.ndarray:
        """Create design matrix for the jump"""
        if not self.fit or time_vector.size == 0:
            return np.array([])

        # Basic step function
        step_function = self._create_step_function(time_vector)

        if self.p.jump_type == JumpType.COSEISMIC_ONLY:
            return step_function

        elif self.p.jump_type == JumpType.POSTSEISMIC_ONLY:
            return self._create_logarithmic_decay(time_vector)

        elif self.p.jump_type == JumpType.COSEISMIC_JUMP_DECAY:
            decay_matrix = self._create_logarithmic_decay(time_vector)
            if decay_matrix.size > 0:
                return np.column_stack((step_function, decay_matrix))
            else:
                # Fallback to jump only if decay can't be computed
                self.p.jump_type = JumpType.COSEISMIC_ONLY
                self.param_count = 1
                return step_function

        else:
            return step_function

    def _create_step_function(self, time_vector: np.ndarray) -> np.ndarray:
        """Create step function component"""
        step = np.zeros((time_vector.shape[0], 1))
        step[time_vector > self.date.fyear] = 1.0
        return step

    def _create_logarithmic_decay(self, time_vector: np.ndarray) -> np.ndarray:
        """Create logarithmic decay components for postseismic deformation"""
        if len(self.p.relaxation) == 0:
            return np.array([])

        decay_matrix = np.zeros((time_vector.shape[0], len(self.p.relaxation)))

        for i, relaxation_time in enumerate(self.p.relaxation):
            mask = time_vector > self.date.fyear
            if np.any(mask):
                decay_matrix[mask, i] = np.log10(
                    1.0 + (time_vector[mask] - self.date.fyear) / relaxation_time
                )

        return decay_matrix

    def _validate_jump_configuration(self, time_vector: np.ndarray) -> None:
        """Validate jump configuration and adjust if necessary"""
        if not self.fit:
            return

        # Check if jump is before time series start
        if self.date.fyear < time_vector.min():
            if self.p.jump_type == JumpType.COSEISMIC_JUMP_DECAY:
                self.p.jump_type = JumpType.POSTSEISMIC_ONLY
                self.param_count -= 1
            elif self.p.jump_type == JumpType.COSEISMIC_ONLY:
                self.fit = False
                self.design = np.array([])
                return

        # Validate design matrix
        if self.design.size > 0:
            # Check for valid step function (some but not all values should be 1)
            if self.p.jump_type in (JumpType.COSEISMIC_ONLY, JumpType.COSEISMIC_JUMP_DECAY):
                step_col = self.design[:, 0]
                if np.all(step_col == 0) or np.all(step_col == 1):
                    self.fit = False
                    self.design = np.array([])
                    return

            # Check condition number for combined jump+decay
            if self.p.jump_type == JumpType.COSEISMIC_JUMP_DECAY and self.design.shape[1] > 1:
                condition_num = np.linalg.cond(self.design.T @ self.design)
                if condition_num > self.config.validation.max_condition_number:
                    # Fallback to jump only
                    self.p.jump_type = JumpType.COSEISMIC_ONLY
                    self.param_count = 1
                    self.design = self._create_step_function(time_vector)

    def get_design_ts(self, ts: np.ndarray) -> np.ndarray:
        """Generate design matrix for given time series"""
        return self._create_design_matrix(ts)

    def print_parameters(self) -> Tuple[str, str, str]:
        """Generate formatted parameter strings for N, E, U components"""
        if not self.fit or self.p.params.size == 0:
            return self._format_no_fit_output()

        return self._format_parameter_output()

    def _format_no_fit_output(self) -> Tuple[str, str, str]:
        """Format output when jump is not fitted"""
        mag_str = f"{self.magnitude:.1f}" if not np.isnan(self.magnitude) else "-"
        dist_str = f"{self.epi_distance:.1f}" if self.epi_distance > 0 else ""

        base_format = f"{self.date.yyyyddd()}            - {mag_str} {self.action} {dist_str}"

        if self.p.jump_type == JumpType.POSTSEISMIC_ONLY:
            # Show relaxation terms for postseismic
            output = []
            for r in self.p.relaxation:
                output.append(f"{self.date.yyyyddd()} {r:4.2f}       - {mag_str} {self.action} {dist_str}")
            return '\n'.join(output), '\n'.join(output), '\n'.join(output)
        else:
            return base_format, base_format, base_format

    def _format_parameter_output(self) -> Tuple[str, str, str]:
        """Format output when parameters are available"""
        mag_str = f"{self.magnitude:.1f}" if not np.isnan(self.magnitude) else "-"
        dist_str = f"{self.epi_distance:.1f}" if self.epi_distance > 0 else ""

        output_n, output_e, output_u = [], [], []
        param_idx = 0

        # Format jump component
        if self.p.jump_type != JumpType.POSTSEISMIC_ONLY:
            jump_params = self.p.params[:, param_idx] * 1000  # Convert to mm
            output_n.append(f"{self.date.yyyyddd()}      {jump_params[0]:>7.1f} {mag_str} {self.action} {dist_str}")
            output_e.append(f"{self.date.yyyyddd()}      {jump_params[1]:>7.1f} {mag_str} {self.action} {dist_str}")
            output_u.append(f"{self.date.yyyyddd()}      {jump_params[2]:>7.1f} {mag_str} {self.action} {dist_str}")
            param_idx += 1

        # Format relaxation components
        for r in self.p.relaxation:
            if param_idx < self.p.params.shape[1]:
                relax_params = self.p.params[:, param_idx] * 1000
                output_n.append(
                    f"{self.date.yyyyddd()} {r:4.2f} {relax_params[0]:>7.1f} {mag_str} {self.action} {dist_str}")
                output_e.append(
                    f"{self.date.yyyyddd()} {r:4.2f} {relax_params[1]:>7.1f} {mag_str} {self.action} {dist_str}")
                output_u.append(
                    f"{self.date.yyyyddd()} {r:4.2f} {relax_params[2]:>7.1f} {mag_str} {self.action} {dist_str}")
                param_idx += 1

        return '\n'.join(output_n), '\n'.join(output_e), '\n'.join(output_u)

    def rehash(self) -> None:
        """Recompute hash for change detection"""
        hash_input = (f"{self.date}_{self.fit}_{self.param_count}_"
                      f"{self.p.jump_type}_{','.join(map(str, self.p.relaxation))}")
        self.p.hash = crc32(hash_input)

    def configure_behavior(self, behavior_config: Dict[str, Any]) -> None:
        """Configure jump-specific behavior"""
        super().configure_behavior(behavior_config)

        if 'relaxation_times' in behavior_config:
            new_relaxation = np.array(behavior_config['relaxation_times'])
            if not np.array_equal(new_relaxation, self.p.relaxation):
                self.p.relaxation = new_relaxation
                self._setup_parameter_count()
                if hasattr(self, '_time_vector'):
                    self.design = self._create_design_matrix(self._time_vector)
                self.rehash()

        if 'jump_type' in behavior_config:
            new_type = JumpType(behavior_config['jump_type'])
            if new_type != self.p.jump_type:
                self.p.jump_type = new_type
                self._setup_parameter_count()
                if hasattr(self, '_time_vector'):
                    self.design = self._create_design_matrix(self._time_vector)
                    self._validate_jump_configuration(self._time_vector)
                self.rehash()

        if 'magnitude' in behavior_config:
            self.magnitude = float(behavior_config['magnitude'])

        if 'action' in behavior_config:
            self.action = str(behavior_config['action'])

    def remove_from_fit(self) -> None:
        """Remove jump from fitting process"""
        self.fit = False
        self.design = np.array([])
        self.rehash()

    def validate_parameters(self) -> List[str]:
        """Validate jump parameters"""
        issues = super().validate_parameters()

        if self.fit and self.p.params.size > 0:
            # Check for unrealistic jump amplitudes
            max_amplitude = np.max(np.abs(self.p.params))
            if max_amplitude > self.config.validation.max_jump_amplitude:
                issues.append(
                    f"Jump {self.date.yyyyddd()}: Unrealistic amplitude {max_amplitude:.3f}m"
                )

        # Check relaxation values are positive
        if len(self.p.relaxation) > 0:
            if np.any(self.p.relaxation <= 0):
                issues.append(f"Jump {self.date.yyyyddd()}: Relaxation times must be positive")

        return issues

    def __lt__(self, other: 'JumpFunction') -> bool:
        """Enable sorting of jumps by date"""
        return self.date.fyear < other.date.fyear

    def __eq__(self, other: 'JumpFunction') -> Tuple[bool, Optional['JumpFunction']]:
        """
        Compare two jumps for equivalence and return decision on which to keep
        Returns (is_equivalent, preferred_jump)
        """
        if not isinstance(other, JumpFunction):
            raise ValueError("Can only compare JumpFunction objects")

        # Handle non-fitting jumps
        if not self.fit and other.fit:
            return True, other
        elif self.fit and not other.fit:
            return True, self
        elif not self.fit and not other.fit:
            return False, None

        # Both jumps are fitted - check for conflicts
        time_separation = abs(other.date.fyear - self.date.fyear)
        days_separation = time_separation * 365.25

        # Calculate data overlap
        design_overlap = 0
        if self.design.size > 0 and other.design.size > 0:
            design_overlap = np.sum(np.logical_xor(self.design[:, 0], other.design[:, 0]))

        # Decision logic based on jump types and data availability
        if self._is_coseismic() and other._is_coseismic():
            # Two coseismic jumps
            if design_overlap < max(self.param_count, other.param_count) + 1:
                # Insufficient data separation - choose by magnitude
                if not np.isnan(self.magnitude) and not np.isnan(other.magnitude):
                    return True, self if self.magnitude > other.magnitude else other
                else:
                    return True, other  # Prefer the newer one
            else:
                return False, None  # Can coexist

        elif self._is_coseismic() and not other._is_coseismic():
            if design_overlap < self.param_count + 1:
                return True, self  # Coseismic prevails
            else:
                return False, None

        elif not self._is_coseismic() and other._is_coseismic():
            if design_overlap < other.param_count + 1:
                return True, other  # Coseismic prevails
            else:
                return False, None

        else:
            # Two mechanical/generic jumps
            if design_overlap < max(self.param_count, other.param_count) + 1:
                return True, other  # Prefer the newer one
            else:
                return False, None

    def _is_coseismic(self) -> bool:
        """Check if jump is coseismic type"""
        return self.p.jump_type >= JumpType.COSEISMIC_JUMP_DECAY

    def __str__(self) -> str:
        """String representation for debugging"""
        return (f"JumpFunction(date={self.date.yyyyddd()}, "
                f"type={JumpTypeDict.get_description(self.p.jump_type)}, "
                f"action={self.action}, fit={self.fit}, "
                f"params={self.param_count})")

    def __repr__(self) -> str:
        return f"JumpFunction({str(self)})"


class JumpManager:
    """Manager class for handling collections of jumps"""

    def __init__(self, config: ETMConfig):
        self.config = config
        self.jumps: List[JumpFunction] = []

    def add_jump(self, jump: JumpFunction) -> bool:
        """
        Add jump to collection, handling conflicts automatically
        Returns True if jump was added, False if rejected
        """
        # Check for conflicts with existing jumps
        for existing_jump in self.jumps[::-1]:  # Check in reverse order
            if existing_jump.fit:
                is_equivalent, preferred = existing_jump.__eq__(jump)
                if is_equivalent:
                    if preferred is jump:
                        existing_jump.remove_from_fit()
                    else:
                        jump.remove_from_fit()
                    break

        self.jumps.append(jump)
        self.jumps.sort()  # Keep chronological order
        return jump.fit

    def remove_jump(self, date: pyDate.Date) -> bool:
        """Remove jump by date"""
        for i, jump in enumerate(self.jumps):
            if jump.date == date:
                del self.jumps[i]
                return True
        return False

    def get_active_jumps(self) -> List[JumpFunction]:
        """Get list of jumps that are fitted"""
        return [jump for jump in self.jumps if jump.fit]

    def get_parameter_count(self) -> int:
        """Get total parameter count for active jumps"""
        return sum(jump.param_count for jump in self.jumps if jump.fit)

    def validate_all_jumps(self) -> List[str]:
        """Validate all jumps and return issues"""
        all_issues = []
        for jump in self.jumps:
            issues = jump.validate_parameters()
            all_issues.extend(issues)
        return all_issues

    def print_jump_table(self) -> Tuple[str, str, str]:
        """Generate formatted jump tables for N, E, U components"""
        if not self.jumps:
            return "No jumps", "No jumps", "No jumps"

        # Header
        header = self.config.get_label('table_title')
        output_n = [header]
        output_e = [header]
        output_u = [header]

        # Add each jump's parameters
        for jump in self.jumps:
            n_str, e_str, u_str = jump.print_parameters()
            if n_str:  # Only add non-empty strings
                output_n.extend(n_str.split('\n'))
                output_e.extend(e_str.split('\n'))
                output_u.extend(u_str.split('\n'))

        # Truncate if too long
        max_lines = 22
        if len(output_n) > max_lines:
            truncate_msg = self.config.get_label('table_too_long')
            output_n = output_n[:max_lines] + [truncate_msg]
            output_e = output_e[:max_lines] + [truncate_msg]
            output_u = output_u[:max_lines] + [truncate_msg]

        return '\n'.join(output_n), '\n'.join(output_e), '\n'.join(output_u)

