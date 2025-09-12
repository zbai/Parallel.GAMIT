import numpy as np

# app
from pgamit.etm.etm_config import ETMConfig
from pgamit.etm.solution_data import SolutionData
from pgamit.etm.polynomial import PolynomialFunction
from pgamit.etm.periodic import PeriodicFunction
from pgamit.etm.jumps import JumpFunction


class FunctionFactory:
    """Factory for creating appropriate function instances"""

    @staticmethod
    def create_polynomial(solution_data: SolutionData, config: ETMConfig,
                          _time_vector: np.ndarray) -> 'PolynomialFunction':
        """Create polynomial function with appropriate configuration"""
        return PolynomialFunction(solution_data, config, _time_vector=_time_vector)

    @staticmethod
    def create_periodic(solution_data: SolutionData, config: ETMConfig,
                        time_vector: np.ndarray) -> 'PeriodicFunction':
        """Create periodic function with appropriate configuration"""
        return PeriodicFunction(solution_data, config, time_vector=time_vector)

    @staticmethod
    def create_jump(solution_data: SolutionData, config: ETMConfig,
                    time_vector: np.ndarray, **jump_params) -> 'JumpFunction':
        """Create jump function with appropriate configuration"""
        return JumpFunction(solution_data, config, time_vector=time_vector, **jump_params)