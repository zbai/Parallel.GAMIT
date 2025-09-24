"""
Project: Parallel.GAMIT
Date: 9/14/25 11:29 AM
Author: Demian D. Gomez
"""

from typing import Tuple, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from pgamit.etm.core.etm_config import EtmConfig
from pgamit.etm.data.solution_data import SolutionData

class TimeSeriesProcessor:
    """Handle coordinate transformations and time series operations"""

    def __init__(self, config: EtmConfig):
        self.config = config

    def transform_to_local(self, solution_data: SolutionData) -> np.ndarray:
        """Transform ECEF coordinates to local NEU frame"""
        from pgamit.Utils import ct2lg

        # Compute coordinate differences from reference
        ecef_diff = np.array([
            solution_data.x - solution_data.auto_x[0],
            solution_data.y - solution_data.auto_y[0],
            solution_data.z - solution_data.auto_z[0]
        ])

        # Transform to local frame
        neu = ct2lg(ecef_diff[0], ecef_diff[1], ecef_diff[2],
                    solution_data.lat[0], solution_data.lon[0])

        return np.array(neu)

    def transform_to_ecef(self, neu_coords: np.ndarray,
                          solution_data: SolutionData) -> np.ndarray:
        """Transform local NEU coordinates back to ECEF"""
        from pgamit.Utils import lg2ct

        ecef_diff = lg2ct(neu_coords[0], neu_coords[1], neu_coords[2],
                          solution_data.lat[0], solution_data.lon[0])

        # Add reference coordinates back
        ecef = np.array([
            ecef_diff[0] + solution_data.auto_x[0],
            ecef_diff[1] + solution_data.auto_y[0],
            ecef_diff[2] + solution_data.auto_z[0]
        ])

        return ecef

    def apply_models(self, observations: np.ndarray, time_vector: np.ndarray,
                     models: List) -> np.ndarray:
        """Apply external models (velocity, postseismic) to observations"""
        corrected_observations = observations.copy()

        for model in models:
            if hasattr(model, 'eval'):
                model_values = model.eval(time_vector)
                corrected_observations -= model_values
                logger.info(f"Applied {model.__class__.__name__} model")

        return corrected_observations

    def compute_data_gaps(self, solution_data: SolutionData,
                          gap_threshold: float = 60.0) -> np.ndarray:
        """Identify data gaps larger than threshold (in days)"""
        if len(solution_data.time_vector_mjd) < 2:
            return np.array([])

        # Find gaps between consecutive observations
        time_diffs = np.diff(solution_data.time_vector_mjd)
        gap_indices = np.where(time_diffs > gap_threshold)[0]

        if len(gap_indices) == 0:
            return np.array([])

        # Create array of MJDs in gaps
        gaps = []
        for idx in gap_indices:
            start_mjd = solution_data.time_vector_mjd[idx]
            end_mjd = solution_data.time_vector_mjd[idx + 1]
            gap_mjds = np.arange(start_mjd + 1, end_mjd)
            gaps.extend(gap_mjds)

        return np.array(gaps)

    @staticmethod
    def filter_by_distance(coordinates: np.ndarray,
                           reference: np.ndarray, max_distance: float) -> np.ndarray:
        """Filter coordinates by distance from reference point"""
        distances = np.sqrt(np.sum(np.square(coordinates - reference), axis=1))
        return distances <= max_distance


class CoordinateTransforms:
    """Coordinate system transformation utilities"""

    @staticmethod
    def ecef_to_neu(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    lat: float, lon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform ECEF coordinates to local NEU frame"""
        from pgamit.Utils import ct2lg
        return ct2lg(x, y, z, lat, lon)

    @staticmethod
    def neu_to_ecef(n: np.ndarray, e: np.ndarray, u: np.ndarray,
                    lat: float, lon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform local NEU coordinates to ECEF frame"""
        from pgamit.Utils import lg2ct
        return lg2ct(n, e, u, lat, lon)

    @staticmethod
    def rotation_matrix_neu_to_ecef(lat: float, lon: float) -> np.ndarray:
        """Get rotation matrix from NEU to ECEF frame"""
        from pgamit.Utils import rotlg2ct
        return rotlg2ct(lat, lon)[:, :, 0]

    @staticmethod
    def transform_covariance_neu_to_ecef(covar_neu: np.ndarray,
                                         lat: float, lon: float) -> np.ndarray:
        """Transform covariance matrix from NEU to ECEF frame"""
        R = CoordinateTransforms.rotation_matrix_neu_to_ecef(lat, lon)
        return R @ covar_neu @ R.T