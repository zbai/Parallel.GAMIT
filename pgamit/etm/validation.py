from typing import List, Tuple, Any
import numpy as np

# app
from pgamit.etm.etm_config import ETMConfig


class DataValidator:
    """Comprehensive data validation utilities"""

    def __init__(self, config: ETMConfig):
        self.config = config

    def validate_coordinates(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> List[str]:
        """Validate coordinate arrays"""
        issues = []

        # Check array lengths match
        if not (len(x) == len(y) == len(z)):
            issues.append("Coordinate arrays have different lengths")

        # Check for NaN or infinite values
        for name, arr in [('X', x), ('Y', y), ('Z', z)]:
            if np.any(np.isnan(arr)):
                issues.append(f"{name} coordinates contain NaN values")
            if np.any(np.isinf(arr)):
                issues.append(f"{name} coordinates contain infinite values")

        # Check coordinate ranges (should be Earth-like)
        if len(x) > 0:
            coord_magnitudes = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            if np.any(coord_magnitudes > 7e6):  # Earth radius ~ 6.37e6
                issues.append("Some coordinates are suspiciously far from Earth center")
            if np.any(coord_magnitudes < 6e6):
                issues.append("Some coordinates are suspiciously close to Earth center")

        return issues

    def validate_time_series(self, time_array: np.ndarray,
                             coordinate_arrays: List[np.ndarray]) -> List[str]:
        """Validate time series consistency"""
        issues = []

        # Check time array length matches coordinates
        coord_length = len(coordinate_arrays[0]) if coordinate_arrays else 0
        if len(time_array) != coord_length:
            issues.append("Time array length doesn't match coordinate arrays")

        # Check for valid time range (reasonable GPS era)
        if len(time_array) > 0:
            min_year = time_array.min()
            max_year = time_array.max()

            if min_year < 1995:  # Before widespread GPS
                issues.append(f"Time series starts suspiciously early: {min_year}")
            if max_year > 2030:  # Far future
                issues.append(f"Time series ends suspiciously late: {max_year}")

        # Check for chronological order
        if len(time_array) > 1 and not np.all(np.diff(time_array) >= 0):
            issues.append("Time series is not in chronological order")

        return issues

    def validate_solution_quality(self, solution_data: SolutionData) -> List[str]:
        """Validate overall solution quality"""
        issues = []

        # Minimum number of solutions
        if solution_data.solutions < self.config.validation.min_solutions_for_etm:
            issues.append(
                f"Insufficient solutions: {solution_data.solutions} < "
                f"{self.config.validation.min_solutions_for_etm}"
            )

        # Check completion percentage
        if solution_data.completion < 10.0:  # Less than 10% data availability
            issues.append(f"Very low data completion: {solution_data.completion:.1f}%")

        # Check time span
        if len(solution_data.t) > 1:
            time_span = solution_data.t.max() - solution_data.t.min()
            if time_span < 0.25:  # Less than 3 months
                issues.append(f"Very short time span: {time_span:.2f} years")

        return issues


class ParameterValidator:
    """Validation for ETM function parameters"""

    def __init__(self, config: ETMConfig):
        self.config = config

    def validate_polynomial_params(self, params: np.ndarray, terms: int) -> List[str]:
        """Validate polynomial parameters"""
        issues = []

        if params.size == 0:
            return ["No polynomial parameters to validate"]

        # Check parameter dimensions
        expected_size = 3 * terms  # 3 components (N,E,U) x number of terms
        if params.size != expected_size:
            issues.append(f"Polynomial parameter size mismatch: {params.size} != {expected_size}")

        # Check for reasonable velocity values (if terms >= 2)
        if terms >= 2 and params.size >= 6:
            velocities = params.reshape((3, -1))[:, 1]  # Second column is velocity
            vel_magnitude = np.sqrt(np.sum(velocities ** 2))

            if vel_magnitude > 0.2:  # 20 cm/year is very fast for most stations
                issues.append(f"Very high velocity: {vel_magnitude:.3f} m/year")

        return issues

    def validate_jump_params(self, params: np.ndarray, jump_type: JumpType) -> List[str]:
        """Validate jump parameters"""
        issues = []

        if params.size == 0:
            return ["No jump parameters to validate"]

        # Check for unrealistic jump amplitudes
        max_amplitude = np.max(np.abs(params))
        if max_amplitude > self.config.validation.max_jump_amplitude:
            issues.append(
                f"Unrealistic jump amplitude: {max_amplitude:.3f} m > "
                f"{self.config.validation.max_jump_amplitude:.3f} m"
            )

        # Check jump type consistency
        if jump_type in (JumpType.COSEISMIC_JUMP_DECAY, JumpType.POSTSEISMIC_ONLY):
            # Should have both jump and decay parameters
            if params.shape[1] < 2:
                issues.append("Postseismic jump should have multiple parameter columns")

        return issues
