"""
Project: Parallel.GAMIT
Date: 9/15/25 8:56 AM
Author: Demian D. Gomez
"""

from abc import ABC, abstractmethod
from typing import List, Union

from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy.linalg import cho_factor, cho_solve
from time import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from pgamit.etm.least_squares.design_matrix import DesignMatrix
from pgamit.etm.core.etm_config import EtmConfig
from pgamit.etm.core.type_declarations import JumpType, AdjustmentModels, NoiseModels, FitStatus
from pgamit.etm.core.data_classes import AdjustmentResults
from pgamit.etm.data.solution_data import SolutionData
from pgamit.etm.least_squares.ls_collocation import covariance_1d, gaussian_func, cauchy_func
from pgamit.etm.etm_functions.stochastic_signal import StochasticSignal


def optimized_least_squares_bak(config, a, cov, csz, observations, css, dof, results):
    """
    Compute all quantities that depend on w = inv(cov) efficiently
    """
    try:
        # Single Cholesky decomposition
        chol_fac, lower = cho_factor(cov)

        # Solve for parameters (from previous question)
        identity = np.eye(cov.shape[0])
        w = cho_solve((chol_fac, lower), identity)

        limit = config.least_squares.sigma_filter_limit

        x1 = chi2.ppf(0.05 / 2, dof)
        x2 = chi2.ppf(1 - 0.05 / 2, dof)

        for j in range(config.least_squares.iterations):
            # save iteration number
            iterations = j + 1

            aTwA = a.T @ w @ a
            aTwobs = a.T @ w @ observations

            try:
                chol_fac2, lower2 = cho_factor(aTwA)
                parameters = cho_solve((chol_fac2, lower2), aTwobs)
            except np.linalg.LinAlgError:
                parameters = np.linalg.solve(aTwA, aTwobs)

            # residuals
            v = observations - a @ parameters

            # Solve for stochastic signal
            # remove stochastic signal to leave only the white noise
            v -= csz @ cho_solve((chol_fac, lower), v)

            # unit sigma
            var = (v @ w @ v) / dof

            logger.debug(f'LSC variance factor: {var:.3f}' )
            # update wrms
            results.wrms = results.wrms * np.sqrt(var)

            s = np.abs(np.divide(v, results.wrms))

            x = var * dof

            if not x1 <= x <= x2:
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

                # downweight outlier observations and scale weight
                w = w.T @ np.diag(f) * 1/var
            else:
                break

        temp_v = cho_solve((chol_fac, lower), observations - a @ parameters)
        stochastic_signal = css @ temp_v
        obs_sigmas = np.sqrt(1 / np.diag(w))

        return parameters, np.linalg.solve(aTwA, np.eye(parameters.shape[0])), stochastic_signal, np.sqrt(var), obs_sigmas, v

    except np.linalg.LinAlgError:
        # Fallback to standard solve
        logger.warning("Warning: Cholesky failed, using standard solve")

        # Parameters
        X = np.linalg.solve(cov, a)
        aTwA = a.T @ X
        aTwobs = a.T @ np.linalg.solve(cov, observations)
        parameters = np.linalg.solve(aTwA, aTwobs)

        # residuals
        v = observations - a @ parameters
        # Other quantities
        temp_v = np.linalg.solve(cov, v)
        stochastic_signal = css @ temp_v
        so = np.sqrt((v @ temp_v) / dof)

        # Diagonal computation
        identity = np.eye(cov.shape[0])
        w_matrix = np.linalg.solve(cov, identity)
        obs_sigmas = np.sqrt(1 / np.diag(w_matrix))

        return parameters, np.linalg.solve(aTwA, np.eye(parameters.shape[0])), stochastic_signal, so, obs_sigmas, v


def optimized_least_squares(config, a, cov, csz, observations, css, dof, results):
    """
    Compute all quantities that depend on w = inv(cov) efficiently
    """
    # find where the covariance drops below eps
    idx = 1000 #np.where(cov[0, :] <= (np.finfo(float).eps / 1000))[0].min()

    # remainder
    r = np.mod(a.shape[0], idx)
    # idx parts to split the matrix
    parts = int(np.floor(a.shape[0] / idx))
    # actual chunks
    chunks = parts * [idx] + [r] if r > 0 else []

    n = []
    c = []
    s = 0
    for e in np.cumsum(chunks).tolist():
        actual_end = e - 1
        block_size = actual_end - s + 1

        # Solve with correct eye matrix size
        tn = np.linalg.solve(cov[s:actual_end + 1, s:actual_end + 1], np.eye(block_size))
        n.append(a[s:actual_end + 1, :].T @ tn @ a[s:actual_end + 1, :])
        c.append(a[s:actual_end + 1, :].T @ tn @ observations[s:actual_end + 1])

        # Next chunk starts after current chunk ends
        s = e  # or s = actual_end + 1

    # stack n and c
    n = np.sum(n, axis=0)
    c = np.sum(c, axis=0)

    parameters = np.linalg.solve(n, c)

    # residuals
    v = observations - a @ parameters

    # Solve for stochastic signal and variance factor
    chol_fac, lower = cho_factor(cov)
    temp_v = cho_solve((chol_fac, lower), v)
    stochastic_signal = css @ temp_v
    so = 1#np.sqrt((v @ w @ v) / dof)

    obs_sigmas = np.sqrt(1 / np.diag(cov))

    return parameters, np.linalg.solve(n, np.eye(parameters.shape[0])), stochastic_signal, so, obs_sigmas, v



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
    def __init__(self, config: EtmConfig):
        self.so = 1
        self.dof = 0
        self.x1 = 0
        self.x2 = 0
        self.config = config

    @abstractmethod
    def adjust(self, design_matrix: DesignMatrix, observations: np.ndarray,
               weights: WeightBuilder, **kwargs) -> AdjustmentResults:
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
               weights: WeightBuilder, **kwargs) -> AdjustmentResults:
        """
        Robust least squares adjustment with outlier detection and reweighting
        """
        results = AdjustmentResults()
        # get design matrix once to avoid multiple computations
        a = design_matrix.matrix

        wrms = 1
        limit = self.config.least_squares.sigma_filter_limit

        self.dof = (a.shape[0] - a.shape[1])
        self.x1 = chi2.ppf(0.05 / 2, self.dof)
        self.x2 = chi2.ppf(1 - 0.05 / 2, self.dof)

        s = np.array([])

        for j in range(self.config.least_squares.iterations):
            # save iteration number
            results.iterations = j + 1

            aw = weights.weight_design(a)
            lw = weights.weight_observations(observations)

            results.parameters = np.linalg.lstsq(aw, lw, rcond=-1)[0]
            results.residuals = observations - a @ results.parameters

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

            logger.debug(f'RobustLeastSquares so: {self.so:.4f} it: {j:2d} pass: {self.x1 <= x <= self.x2}')

            if not self.x1 <= x <= self.x2:
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
                break

        # iterations done, compute statistics
        try:
            if not results.converged:
                logger.info(f'RobustLeastSquares did not converge! final wrms {wrms/1000:.1f} mm')

            results.covariance_matrix = np.linalg.solve(a.T @ weights.matrix @ a, np.eye(a.shape[1])) * self.so
        except np.linalg.LinAlgError as e:
            logger.info('np.linalg.inv failed to obtain covariance matrix: %s' % str(e))
            results.covariance_matrix = np.ones((design_matrix.matrix.shape[0],
                                                 design_matrix.matrix.shape[0]))
        # extract the parameter sigmas
        results.parameter_sigmas =  np.sqrt(np.diag(results.covariance_matrix))
        # mark observations with sigma <= LIMIT
        results.outlier_flags = s <= limit
        results.obs_sigmas = np.sqrt(1 / np.diag(weights.matrix))
        # declare the origin of the fit
        results.origin = 'RobustLeastSquares'
        # log the results for inspection by user
        logger.debug(f'{" ":<28}: ' +
                     ' '.join([funct.short_name() for funct in design_matrix.functions if funct.fit]))
        logger.debug(f'{"RobustLeastSquares [mm]":>28}: ' +
                     ' '.join([f'{p*1000.:10.3e}' for p in results.parameters.tolist()]))

        return results


class LeastSquaresCollocation(AdjustmentStrategy):

    def adjust(self, design_matrix: DesignMatrix, observations: np.ndarray,
               weights: WeightBuilder,
               time_vector_mjd: np.ndarray = None,
               time_vector_cont_mjd: np.ndarray = None) -> AdjustmentResults:

        tt = time()
        a = design_matrix.matrix
        limit = self.config.least_squares.sigma_filter_limit

        # first need to run the RobustAdjustment to get residuals
        lsq = RobustLeastSquares(self.config)
        white_noise = WhiteNoise(observations.size)

        results = lsq.adjust(design_matrix, observations, white_noise)
        wrms = results.wrms
        ###################################################################################################
        # now use residuals to run lsc
        self.dof = (a.shape[0] - a.shape[1])

        # fit the covariance function (using Kevin Wang's code)
        cov, csz, css, _, para, empirical_cov = self._estimate_covariance_matrices(time_vector_mjd,
                                                                                   time_vector_cont_mjd,
                                                                                   results.residuals,
                                                                                   results.obs_sigmas,
                                                                                   results.outlier_flags)
        # find the parameters
        results = self._least_squares_non_it(a, cov, observations, css, csz)
        # save covariance parameters
        results.covariance_function_params = para
        results.empirical_covariance = np.array(empirical_cov)
        results.wrms = wrms * self.so
        results.variance_factor = self.so ** 2
        #results = self._least_squares_chunks(a, observations, results, time_vector_mjd, time_vector_cont_mjd)

        #results.parameters = np.linalg.inv(a.T @ w @ a) @ (a.T @ w @ observations)
        logger.debug(f'LeastSquaresCollocation least squares in {time() - tt:.3f} s '
                     f'sigma_o: {self.so:.3f} '
                     f'wrms: {results.wrms * 1000.} mm')

        s = np.abs(np.divide(results.residuals, results.wrms))

        results.parameter_sigmas = np.sqrt(np.diag(results.covariance_matrix))

        logger.debug(f'LeastSquaresCollocation [mm]: ' +
                     ' '.join([f'{p*1000.:10.3e}' for p in results.parameters.tolist()]))

        # mark observations with sigma <= LIMIT
        results.outlier_flags = s <= limit
        # declare the origin of the fit
        results.origin = 'LeastSquaresCollocation'

        logger.debug(f'LeastSquaresCollocation in {time() - tt:.3f} s')

        return results

    def _least_squares_chunks(self, a, observations, robust_results, time_vector_mjd, time_vector_cont_mjd):
        """
        Compute all quantities that depend on w = inv(cov) efficiently
        """
        # find where the covariance drops below eps
        idx = 500  # np.where(cov[0, :] <= (np.finfo(float).eps / 1000))[0].min()

        results = AdjustmentResults()

        # remainder
        r = np.mod(a.shape[0], idx)
        # idx parts to split the matrix
        parts = int(np.floor(a.shape[0] / idx))
        # actual chunks
        chunks = parts * [idx] + [r] if r > 0 else []

        n = []
        c = []
        s = 0
        for e in np.cumsum(chunks).tolist():
            actual_end = e - 1
            block_size = actual_end - s + 1

            cov, _, _, _ = self._estimate_covariance_matrices(time_vector_mjd[s:actual_end + 1],
                                                              time_vector_cont_mjd,
                                                              robust_results.residuals[s:actual_end + 1],
                                                              robust_results.obs_sigmas[s:actual_end + 1],
                                                              robust_results.oulier_flags[s:actual_end + 1])

            # Solve with correct eye matrix size
            tn = np.linalg.solve(cov, np.eye(block_size))
            n.append(a[s:actual_end + 1, :].T @ tn @ a[s:actual_end + 1, :])
            c.append(a[s:actual_end + 1, :].T @ tn @ observations[s:actual_end + 1])

            # Next chunk starts after current chunk ends
            s = e  # or s = actual_end + 1

        # stack n and c
        n = np.sum(n, axis=0)
        c = np.sum(c, axis=0)

        results.parameters = np.linalg.solve(n, c)

        # residuals
        #v = observations - a @ results.parameters
        results.residuals = observations - a @ results.parameters

        # Solve for stochastic signal and variance factor
        #w = np.linalg.solve(cov, np.eye(cov.shape[0]))
        results.stochastic_signal = np.zeros(time_vector_cont_mjd.shape)

        #results.residuals = v - csz @ (w @ v)
        # wrms of the database
        # self.so = np.sqrt((v @ w @ v) / self.dof)
        results.wrms = np.sqrt((results.residuals.T @ results.residuals) / self.dof) * self.so

        results.covariance_matrix = np.linalg.solve(n, np.eye(results.parameters.shape[0])) * self.so
        # results.obs_sigmas = np.sqrt(1 / np.diag(cov))

        return results

    def _least_squares_non_it(self, a, cov, observations, css, csz):
        """
        Compute all quantities that depend on w = inv(cov) efficiently
        """
        # find where the covariance drops below eps
        idx = 1000  # np.where(cov[0, :] <= (np.finfo(float).eps / 1000))[0].min()

        results = AdjustmentResults()

        # remainder
        r = np.mod(a.shape[0], idx)
        # idx parts to split the matrix
        parts = int(np.floor(a.shape[0] / idx))
        # actual chunks
        chunks = parts * [idx]
        chunks += [r] if r > 0 else []

        n = []
        c = []
        s = 0
        for e in np.cumsum(chunks).tolist():
            actual_end = e - 1
            block_size = actual_end - s + 1

            # Solve with correct eye matrix size
            p = np.linalg.solve(cov[s:actual_end + 1, s:actual_end + 1], np.eye(block_size))
            n.append(a[s:actual_end + 1, :].T @ p @ a[s:actual_end + 1, :])
            c.append(a[s:actual_end + 1, :].T @ p @ observations[s:actual_end + 1])

            # Next chunk starts after current chunk ends
            s = e  # or s = actual_end + 1

        # stack n and c
        n = np.sum(n, axis=0)
        c = np.sum(c, axis=0)

        results.parameters = np.linalg.solve(n, c)

        # residuals
        v = observations - a @ results.parameters

        # Solve for stochastic signal and variance factor
        w = np.linalg.solve(cov, np.eye(cov.shape[0]))
        results.stochastic_signal = css @ w @ v

        results.residuals = v - csz @ (w @ v)
        # wrms of the database
        self.so = np.sqrt((results.residuals @ w @ results.residuals) / self.dof)
        # results.wrms = np.sqrt((results.residuals.T @ results.residuals) / self.dof) * self.so

        results.covariance_matrix = np.linalg.solve(n, np.eye(results.parameters.shape[0])) * (self.so ** 2)
        results.obs_sigmas = np.sqrt(1 / np.diag(cov))

        return results

    def _estimate_covariance_matrices(self, time_vector_mjd, time_vector_cont_mjd, residuals, obs_sigmas, outliers):
        # obtain covariance function
        c_obs_s, para, bins_mjd, emp_cov, rms_para = self.get_covariance(time_vector_mjd[outliers], residuals[outliers])

        logger.debug(f'LeastSquaresCollocation covariance parameters: {para}')

        # to get the full stochastic signal, find the lag between the point to be predicted and observations
        lag = np.abs(time_vector_mjd - time_vector_cont_mjd[:, np.newaxis])
        css = gaussian_func(lag, *para)
        # noise matrix
        # crr = np.eye(observations.shape[0]) * (results.wrms ** 2)
        crr = np.diag(obs_sigmas ** 2)
        idx = np.where(~np.isin(time_vector_cont_mjd, time_vector_mjd))[0]
        # find which elements to remove from css (gaps) to obtain the covariance of observations ONLY
        csz = np.delete(css, idx, axis=0)
        cov = csz + crr

        return cov, csz, css, crr, para, [bins_mjd, emp_cov]

    @staticmethod
    def get_covariance(time_vector_mjd, residuals, empirical_func=gaussian_func):
        """ Estimate the covariance of the signals.
            The given residuals are used to calculate the covariance of the signals.

        Parameters: xy_interp          : 2D np.ndarray in the size of (num_obs, 2), xy coordinates of the observations.
                    xy_residuals    : 2D np.ndarray in the size of (num_residuals, 2), xy coordinates of the residuals.
                    residuals       : 1D np.ndarray in the size of (num_residuals,), residuals used to calculate the covariance of the signals.
                    bin             : integer, number interval in the lag
                    empir_func      : function, empirical function used to fit the covariance

        Returns:    C_obs_s         : 2D np.ndarray in the size of (num_obs, num_obs), covariance matrix of the signals
                    para            : 1D np.ndarray, parameters of the empirical function
                    lag             : 1D np.ndarray in the size of (bin + 1,), lag in the covariance
                    cov             : 2D np.ndarray in the size of (bin + 1,), covariance values
                    trend           : 1D np.ndarray, trend in the observations.
                    trend_std       : 1D np.ndarray, standard deviation of the trend
                    rms_para        : float, root mean square difference between covariance and the fitting results.
        """
        # now begin the collocation process using the residuals
        bins_mjd, _, cov = covariance_1d(time_vector_mjd, residuals)

        init_scale = np.floor(np.log10(np.abs(bins_mjd).max()))

        if empirical_func.__name__ == 'gaussian_func':
            init_values = [cov[0], 10 ** (init_scale - 1), 10 ** (init_scale - 1)]
        elif empirical_func.__name__ == 'exponential_func':
            init_values = [cov[0], 10 ** (init_scale)]
        else:
            init_values = [cov[0], 10 ** init_scale]

        para, pcov, opt = curve_fit(empirical_func, bins_mjd, cov, p0=init_values, full_output=True)[0:3]

        rms_para = np.sqrt((opt['fvec'] ** 2).mean())

        c_obs_s = empirical_func(bins_mjd, *para)
        c_obs_s[c_obs_s < np.finfo(float).eps] = np.finfo(float).eps

        return c_obs_s, para, bins_mjd, cov, rms_para


class EtmFit:
    def __init__(self, config: EtmConfig, design_matrix: DesignMatrix, solution_data_hash: int = None):

        self.config = config
        self.outlier_flags: np.ndarray = np.array([])
        self.results: List[AdjustmentResults] = [AdjustmentResults(), AdjustmentResults(), AdjustmentResults()]
        self.design_matrix = design_matrix
        self.covar = np.array([])
        # validate design matrix now to catch any changes in jumps (before loading from the db)
        self._validate_function_design_matrix()
        # store solution hash to compare hash value in database save process
        self.hash = solution_data_hash

    def run_fit(self, solution_data: SolutionData, design_matrix: DesignMatrix = None,
                noise_model: NoiseModels = NoiseModels.WHITE_ONLY) -> float:
        """wrapper to run adjustment. Returns runtime of the operation"""
        run_time = time()
        # shorthand notation
        adjustment_strategy = self.config.modeling.adjustment_strategy

        # allow replacing the design matrix
        if design_matrix:
            self.design_matrix = design_matrix

        # transform solutions to NEU
        neu = solution_data.transform_to_local()

        # create instances of the adjustment strategy to use
        if adjustment_strategy == AdjustmentModels.ROBUST_LEAST_SQUARES:
            lsq = RobustLeastSquares(self.config)
        elif adjustment_strategy == AdjustmentModels.LSQ_COLLOCATION:
            lsq = LeastSquaresCollocation(self.config)
        else:
            raise Exception('Adjustment strategy not implemented')

        while True:
            for i in range(3):
                # select the noise model
                if noise_model == NoiseModels.WHITE_ONLY:
                    white_noise = WhiteNoise(neu[i].shape[0])
                else:
                    raise Exception('Noise model not implemented')

                if adjustment_strategy == AdjustmentModels.ROBUST_LEAST_SQUARES:
                    self.results[i] = lsq.adjust(design_matrix, neu[i], white_noise)

                elif adjustment_strategy == AdjustmentModels.LSQ_COLLOCATION:
                    # get mask for least squares collocation
                    mask = self.config.modeling.get_observation_mask(solution_data.time_vector)

                    self.results[i] = lsq.adjust(design_matrix, neu[i],white_noise,
                                                 solution_data.time_vector_mjd[mask],
                                                 solution_data.time_vector_cont_mjd)

            # insert stochastic signal if it was estimates
            for f in design_matrix.functions:
                if f.p.object == 'stochastic':
                    f.p.params = [rs.stochastic_signal for rs in self.results]

            # populate results onto etm_function
            for func in design_matrix.functions:
                if func.fit and func.param_count > 0:
                    func.load_parameters(self.results)

            # single outlier flag
            self.outlier_flags = np.all((self.results[0].outlier_flags,
                                         self.results[1].outlier_flags,
                                         self.results[2].outlier_flags), axis=0)

            self.process_covariance()

            # validate results and redo fit if needed
            validation = design_matrix.validate_parameters()
            if not validation:
                # no issue found, break
                break

            for f, message in validation:
                logger.info(message)
                if message.startswith('Unrealistic amplitude'):
                    f.configure_behavior({'jump_type': JumpType.POSTSEISMIC_ONLY})

        self.config.modeling.status = FitStatus.POSTFIT
        return time() - run_time

    def _validate_function_design_matrix(self):

        validation = self.design_matrix.validate_design()

        for f, message in validation:
            logger.info(message)
            if message.startswith('Condition number too large'):
                min_index = np.argmin(f.p.relaxation)
                rlx = np.delete(f.p.relaxation, min_index)
                f.configure_behavior({'relaxation': rlx})

    def get_var_factor_db_fields(self):
        """method to return the var_factor object for storing solution in database"""
        return {
            'NetworkCode': self.config.network_code,
            'StationCode': self.config.station_code,
            'object': 'var_factor',
            'soln': self.config.solution.soln,
            'stack': self.config.solution.stack_name,
            'params': [result.parameters for result in self.results],
            'sigmas': [[result.variance_factor for result in self.results],
                       [result.wrms for result in self.results]],
            'hash': self.hash
        }

    def get_time_continuous_model(self, time_vector_cont: np.ndarray) -> List[np.ndarray]:
        # spit the auto position by default, in case there is no model
        model = [np.array([0]),
                 np.array([0]),
                 np.array([0])]

        if self.config.modeling.status == FitStatus.POSTFIT:
            for i in range(3):
                model[i] = self.design_matrix.alternate_time_vector(time_vector_cont) @ self.results[i].parameters

        return model

    def load_from_json(self):
        """
        load ETM solution from json file
        """

    def _compute_covariances(self, results: 'AdjustmentResults') -> None:
        """Compute parameter covariances and correlations"""
        # Implementation for covariance computation
        pass

    def _detect_outliers(self, observations: np.ndarray,
                         results: 'AdjustmentResults') -> None:
        """Detect and flag statistical outliers"""
        # Implementation for outlier detection
        pass

    def process_covariance(self):

        cov = np.zeros((3, 1))

        self.covar = np.diag([r.wrms ** 2 for r in self.results])

        # save the covariance between N-E, E-U, N-U
        f = self.outlier_flags

        # load the covariances using the correlations
        for k, i, j in ((0, 0, 1), (1, 1, 2), (2, 0, 2)):
            cov[k] = np.corrcoef(self.results[i].residuals[f],
                                 self.results[j].residuals[f])[0, 1] * self.results[i].wrms * self.results[j].wrms

        # build a variance-covariance matrix
        self.covar[0, 1] = cov[0]
        self.covar[1, 0] = cov[0]
        self.covar[2, 1] = cov[1]
        self.covar[1, 2] = cov[1]
        self.covar[0, 2] = cov[2]
        self.covar[2, 0] = cov[2]

        if not self._isPD(self.covar):
            self.covar = self._nearestPD(self.covar)

    def _nearestPD(self, a):
        """Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """

        b = (a + a.T) / 2
        _, s, v = np.linalg.svd(b)

        h = np.dot(v.T, np.dot(np.diag(s), v))

        a2 = (b + h) / 2

        a3 = (a2 + a2.T) / 2

        if self._isPD(a3):
            return a3

        spacing = np.spacing(np.linalg.norm(a))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        i = np.eye(a.shape[0])
        k = 1

        while not self._isPD(a3):
            mineig = np.min(np.real(np.linalg.eigvals(a3)))
            a3 += i * (-mineig * k ** 2 + spacing)
            k += 1

        return a3

    @staticmethod
    def _isPD(b):
        """Returns true when input is positive-definite, via Cholesky"""
        try:
            _ = np.linalg.cholesky(b)
            return True
        except np.linalg.LinAlgError:
            return False