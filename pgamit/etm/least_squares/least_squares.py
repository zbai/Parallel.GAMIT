"""
Project: Parallel.GAMIT
Date: 9/15/25 8:56 AM
Author: Demian D. Gomez
"""

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import List, Union

from scipy.stats import chi2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from pgamit.etm.least_squares.design_matrix import DesignMatrix
from etm.core.etm_config import ETMConfig
from etm.data.solution_data import SolutionData

class NoiseModels(IntEnum):
    """Enum for noise models"""
    WHITE_ONLY = auto()
    WHITE_FLICKER = auto()
    WHITE_FLICKER_RW = auto()

@dataclass
class AdjustmentResults:
    """Results from least squares adjustment"""
    parameters: np.ndarray = field(default_factory=lambda: np.array([]))
    parameter_sigmas: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    variance_factor: float = 0
    wrms: float = 0
    weights: np.ndarray = field(default_factory=lambda: np.array([]))
    covariance_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    outlier_flags: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = False
    iterations: int = 0

class WeightBuilder(ABC):
    """Abstract base class for different weighting strategies"""
    def __init__(self, **kwargs):
        self.matrix = np.array([])
        pass

    @abstractmethod
    def update_weights(self, new_weights):
        pass

    @abstractmethod
    def weight_observations(self, observations) -> np.ndarray:
        pass

    @abstractmethod
    def weight_design(self, design_matrix) -> np.ndarray:
        pass


class AdjustmentStrategy(ABC):
    """Abstract base class for different adjustment algorithms"""
    def __init__(self, config: ETMConfig):
        self.so = 1
        self.dof = 0
        self.x1 = 0
        self.x2 = 0
        self.config = config

    @abstractmethod
    def adjust(self, design_matrix: DesignMatrix, observations: np.ndarray,
               weights: WeightBuilder) -> AdjustmentResults:
        """Perform the least squares adjustment"""
        pass


class WhiteNoise(WeightBuilder):
    """Implementation of WeightBuilder for different noise processes"""
    def __init__(self, observation_count: int = 0):
        super().__init__()
        self.matrix = np.diag(np.ones((observation_count,)))
        self._w = np.ones((observation_count,))

    def update_weights(self, new_weights):
        # check for weights <<< than eps, otherwise matrix may become singular
        new_weights[new_weights < np.finfo(float).eps] = np.finfo(float).eps
        if len(new_weights.shape) > 1:
            self.matrix = new_weights
            self._w = np.sqrt(np.diag(new_weights))
        else:
            self.matrix = np.diag(new_weights)
            self._w = np.sqrt(new_weights)

    def weight_observations(self, observations) -> np.ndarray:
        """
        function to multiply the observations and weights
        """
        return self._w * observations

    def weight_design(self, design_matrix):
        """
        function to multiply the design matrix and weights
        """
        return self._w[:, np.newaxis] * design_matrix


class RobustLeastSquares(AdjustmentStrategy):
    """Robust least squares with iterative reweighting"""

    def adjust(self, design_matrix: DesignMatrix, observations: np.ndarray,
               weights: WeightBuilder) -> AdjustmentResults:
        """
        Robust least squares adjustment with outlier detection and reweighting
        """
        results = AdjustmentResults()

        wrms = 1
        limit = self.config.least_squares.sigma_filter_limit

        self.dof = (design_matrix.matrix.shape[0] - design_matrix.matrix.shape[1])
        self.x1 = chi2.ppf(1 - 0.05 / 2, self.dof)
        self.x2 = chi2.ppf(0.05 / 2, self.dof)

        s = np.array([])

        for j in range(self.config.least_squares.iterations):
            # save iteration number
            results.iterations = j + 1

            aw = weights.weight_design(design_matrix.matrix)
            lw = weights.weight_observations(observations)

            results.parameters = np.linalg.lstsq(aw, lw, rcond=-1)[0]
            results.residuals = observations - design_matrix.matrix @ results.parameters

            v = results.residuals
            p = weights.matrix
            # unit variance
            self.so = np.sqrt((v @ p @ v) / self.dof)
            # save in results
            results.variance_factor = np.square(self.so)

            x = np.power(self.so, 2) * self.dof

            # obtain the overall uncertainty predicted by lsq
            wrms = wrms * self.so
            results.wrms = wrms

            # calculate the normalized sigmas
            s = np.abs(np.divide(v, wrms))

            if x < self.x2 or x > self.x1:
                # if it falls in here it's because it didn't pass the Chi2 test

                # reweigh by Mike's method of equal weight until 2 sigma
                f = np.ones((v.shape[0],))
                # f[s > LIMIT] = 1. / (np.power(10, LIMIT - s[s > LIMIT]))
                # do not allow sigmas > 100 m, which is basically not putting
                # the observation in. Otherwise, due to a model problem
                # (missing jump, etc) you end up with very unstable inversions
                # f[f > 500] = 500
                sw = np.power(10, limit - s[s > limit])
                sw[sw < np.finfo(float).eps] = np.finfo(float).eps
                f[s > limit] = sw

                weights.update_weights(np.square(np.divide(f, wrms)))
            else:
                results.converged = True
                break  # cst_pass = True
        # compute statistics
        try:
            if not results.converged:
                logger.info(f'RobustLeastSquares did not converge! final wrms {wrms/1000:.1f} mm')

            results.covariance_matrix = np.linalg.inv(design_matrix.matrix.transpose() @
                                                      weights.matrix @
                                                      design_matrix.matrix) * self.so
        except np.linalg.LinAlgError as e:
            logger.info('np.linalg.inv failed to obtain covariance matrix: %s' % str(e))
            results.covariance_matrix = np.ones((design_matrix.matrix.shape[0],
                                                 design_matrix.matrix.shape[0]))
        # extract the parameter sigmas
        results.parameter_sigmas =  np.sqrt(np.diag(results.covariance_matrix))
        # mark observations with sigma <= LIMIT
        results.outlier_flags = s <= limit

        return results


class ETMFit:
    def __init__(self, config: ETMConfig):
        self.config = config
        self.outlier_flags: bool = False
        self.results: List[AdjustmentResults] = [AdjustmentResults(), AdjustmentResults(), AdjustmentResults()]
        self.design_matrix: Union[None, DesignMatrix] = None

    def run_fit(self, solution_data: SolutionData, design_matrix: DesignMatrix,
                noise_model: NoiseModels = NoiseModels.WHITE_ONLY):

        self.design_matrix = design_matrix
        # transform solutions to NEU
        neu = solution_data.transform_to_local()
        for i in range(3):
            # select the noise model
            if noise_model == NoiseModels.WHITE_ONLY:
                white_noise = WhiteNoise(solution_data.solutions)
            else:
                raise Exception('Noise model not implemented')

            # @todo: apply other least squares strategies
            lsq = RobustLeastSquares(self.config)

            self.results[i] = lsq.adjust(design_matrix, neu[i], white_noise)

            # populate results onto etm_function
            for func in design_matrix.functions:
                if func.fit and func.param_count > 0:
                    func.p.params[i] =  self.results[i].parameters[func.column_index]
                    func.p.sigmas[i] = self.results[i].parameter_sigmas[func.column_index]
                    func.p.covar[i] = self.results[i].covariance_matrix[func.column_index][:, func.column_index]

        # single outlier flag
        self.outlier_flags = np.all((self.results[0].outlier_flags,
                                     self.results[1].outlier_flags,
                                     self.results[2].outlier_flags), axis=0)

    def _compute_covariances(self, results: 'AdjustmentResults') -> None:
        """Compute parameter covariances and correlations"""
        # Implementation for covariance computation
        pass

    def _detect_outliers(self, observations: np.ndarray,
                         results: 'AdjustmentResults') -> None:
        """Detect and flag statistical outliers"""
        # Implementation for outlier detection
        pass