from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from etm.core.etm_config import  ETMConfig
from etm.etm_functions.etm_function import EtmFunction

# ============================================================================
# Enhanced Design Matrix
# ============================================================================

# design_matrix.py
class DesignMatrix:
    """Enhanced design matrix with validation and optimization"""

    def __init__(self, time_vector: np.ndarray, functions: List[EtmFunction],
                 config: ETMConfig):
        self.time_vector = time_vector
        self.functions = functions
        self.config = config

        # Build matrix
        self.matrix: np.ndarray = self._build_matrix(self.time_vector)
        self.condition_number = self._compute_condition_number()

        # Assign column indices to functions
        self._assign_column_indices()

        # Validate matrix
        self._validate_matrix()

    def _build_matrix(self, time_vector: np.ndarray) -> np.ndarray:
        """Build the complete design matrix"""
        active_functions = [f for f in self.functions if f.fit]

        if not active_functions:
            return np.array([]).reshape(len(time_vector), 0)

        # Build matrix column by column
        matrix_parts = []
        for func in active_functions:
            func_matrix = func.get_design_ts(time_vector)
            if func_matrix.size > 0:
                matrix_parts.append(func_matrix)

        if matrix_parts:
            return np.column_stack(matrix_parts)
        else:
            return np.array([]).reshape(len(time_vector), 0)

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
        if self.matrix.size == 0:
            logger.warning("Empty design matrix")
            return

        # Check for rank deficiency
        rank = np.linalg.matrix_rank(self.matrix)
        if rank < self.matrix.shape[1]:
            logger.warning(f"Design matrix is rank deficient: rank={rank}, cols={self.matrix.shape[1]}")

        # Check condition number
        if self.condition_number > self.config.validation.max_condition_number:
            logger.warning(f"High condition number: {self.condition_number:.2e}")

        # Check for NaN or infinite values
        if np.any(np.isnan(self.matrix)):
            logger.error("Design matrix contains NaN values")

        if np.any(np.isinf(self.matrix)):
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