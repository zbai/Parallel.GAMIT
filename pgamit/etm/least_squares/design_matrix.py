from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from pgamit.etm.core.etm_config import  EtmConfig
from pgamit.etm.etm_functions.etm_function import EtmFunction
from pgamit.etm.core.type_declarations import FitStatus

# ============================================================================
# Enhanced Design Matrix
# ============================================================================

# design_matrix.py
class DesignMatrix:
    """Enhanced design matrix with validation and optimization"""

    def __init__(self, time_vector: np.ndarray, functions: List[EtmFunction],
                 config: EtmConfig):
        self.time_vector = time_vector
        self.functions = functions
        self.config = config

        # Build matrix
        # self.matrix: np.ndarray = self._build_matrix(self.time_vector)
        self.condition_number = self._compute_condition_number()

        # Assign column indices to functions
        self._assign_column_indices()

        # Validate matrix
        self._validate_matrix()

    @property
    def matrix(self) -> np.ndarray:
        # two options:
        # config.modeling.status == PREFIT the size (column count) matrix can change
        # config.modeling.status == POSTFIT the size (column count) matrix cannot change

        self._assign_column_indices()
        return self._build_matrix(self.time_vector)

    def validate_parameters(self) -> List[Tuple[EtmFunction, str]]:
        """call the individual elements in functions and runs validations"""
        validation = []
        for f in self.functions:
            if f.fit:
                validation.extend(f.validate_parameters())

        return validation

    def validate_design(self) -> List[Tuple[EtmFunction, str]]:
        """call the individual elements in functions and runs validations"""
        validation = []
        for f in self.functions:
            if f.fit:
                validation.extend(f.validate_design())

        return validation

    def _build_matrix(self, time_vector: np.ndarray) -> np.ndarray:
        """Build the complete design matrix"""
        # two options:
        # config.modeling.status == PREFIT the size (column count) matrix can change
        # config.modeling.status == POSTFIT the size (column count) matrix cannot change
        #   only the jump function is affected by changing the time window

        if (self.config.modeling.status == FitStatus.PREFIT and
            self.config.modeling.data_model_window is not None):
            # pulling matrix in prefit mode, apply window filter
            mask = self.config.modeling.get_observation_mask(time_vector)
        else:
            # in POSTFIT mode return all observations
            mask = np.ones(time_vector.shape).astype(bool)

        active_functions = [f for f in self.functions if f.fit]

        if not active_functions:
            return np.array([]).reshape(len(time_vector[mask]), 0)

        # Build matrix column by column
        matrix_parts = []
        for func in active_functions:
            func_matrix = func.get_design_ts(time_vector[mask])
            if func_matrix.size > 0:
                matrix_parts.append(func_matrix)

        if matrix_parts:
            return np.column_stack(matrix_parts)
        else:
            return np.array([]).reshape(len(time_vector[mask]), 0)

    def alternate_time_vector(self, time_vector: np.ndarray):
        """
        expose _build_matrix
        """
        return self._build_matrix(time_vector)

    def _assign_column_indices(self) -> None:
        """Assign column indices to functions"""
        col_index = 0

        for func in self.functions:
            if func.fit and func.param_count > 0:
                func.column_index = np.arange(col_index, col_index + func.param_count)
                col_index += func.param_count
            else:
                func.column_index = np.array([])

    def _compute_condition_number(self) -> float:
        """Compute condition number of design matrix"""
        if self.matrix.size == 0:
            return 0.0

        try:
            return np.linalg.cond(self.matrix)
        except np.linalg.LinAlgError:
            return np.inf

    def _validate_matrix(self) -> None:
        """Validate design matrix for common issues"""
        matrix = self.matrix
        if matrix.size == 0:
            logger.warning("Empty design matrix")
            return

        # Check for rank deficiency
        rank = np.linalg.matrix_rank(matrix)
        if rank < matrix.shape[1]:
            logger.warning(f"Design matrix is rank deficient: rank={rank}, cols={self.matrix.shape[1]}")

        # Check condition number
        if np.log10(self.condition_number) > self.config.validation.max_condition_number:
            logger.warning(f"High condition number: {self.condition_number:.2e}")

        # Check for NaN or infinite values
        if np.any(np.isnan(matrix)):
            logger.error("Design matrix contains NaN values")

        if np.any(np.isinf(matrix)):
            logger.error("Design matrix contains infinite values")

    def get_submatrix(self, function_types: List[type]) -> np.ndarray:
        """Get submatrix for specific function types"""
        columns = []
        i = 0
        for func in self.functions:
            if func.fit and type(func) in function_types:
                columns.extend(range(i, i+func.param_count))
            else:
                i += func.param_count

        if columns:
            return self.matrix[:, columns]
        else:
            return np.array([]).reshape(self.matrix.shape[0], 0)