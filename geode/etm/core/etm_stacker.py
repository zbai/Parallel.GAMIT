"""
Project: Geodesy Database Engine (GeoDE)
Date: 10/26/25 9:10AM
Author: Demian D. Gomez
"""
import os.path
from functools import total_ordering
from typing import List, Tuple, Dict, Union, Callable
from dataclasses import dataclass, field
import numpy as np
import logging
from abc import ABC, abstractmethod
from enum import Enum
import copy
import time

import matplotlib
import numpy.linalg.linalg

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from scipy.stats import iqr
from scipy.interpolate import griddata
from tqdm import tqdm
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

logging.getLogger('geode.etm.core.etm_stacker').setLevel(logging.DEBUG)

from ...dbConnection import Cnn
from .etm_engine import EtmEngine
from .etm_config import EtmConfig
from ..etm_functions.polynomial import PolynomialFunction
from ..data.solution_data import SolutionDataException
from ..etm_functions.periodic import PeriodicFunction
from ..visualization.plot_fields import plot_velocity_field, mask_ocean_points
from ..etm_functions.jumps import JumpFunction
from .type_declarations import SolutionType, JumpType, FitStatus
from .data_classes import BaseDataClass, Earthquake, AdjustmentResults
from ..least_squares.design_matrix import DesignMatrixException
from ...elasticity.elastic_interpolation import get_qpw, get_radius, spline2dgreen
from ...elasticity.diskload import compute_diskload, load_love_numbers
from ...elasticity.green_func import build_design_matrix
from ...elasticity.rectloadhs import mrectloadhs_dif
from ...Utils import stationID, azimuthal_equidistant, print_yellow, inverse_azimuthal
from ...pyDate import Date
from ...pyOkada import Mask

MISSING_DAYS_TOLERANCE = 3

class EtmStackerException(Exception):
    pass


def fill_region_with_grid(x_points, y_points, radius, apply_buffer=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill the convex hull of scattered points with a square grid.
    Circles centered on these points with the given radius will just touch.

    Parameters:
    -----------
    x_points : array-like
        X coordinates of scattered points defining the region
    y_points : array-like
        Y coordinates of scattered points defining the region
    radius : float
        Radius of circles to be placed on grid points

    Returns:
    --------
    grid_x, grid_y : arrays
        Coordinates of grid points inside the region
    hull_x, hull_y : arrays
        X and Y coordinates of the convex hull boundary
    """

    # Convert to numpy arrays
    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)

    # Stack points for convex hull computation
    points = np.column_stack([x_points, y_points])

    # Compute convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    if apply_buffer:
        poly = Polygon(hull_points)
        offset_distance = radius * 4
        expanded_poly = poly.buffer(offset_distance, join_style='mitre')
        expanded_hull_points = np.array(expanded_poly.exterior.coords[:-1])

        hull_x = expanded_hull_points[:, 0]
        hull_y = expanded_hull_points[:, 1]
        # Create a path from convex hull boundary
        boundary_path = Path(expanded_hull_points)
    else:
        hull_x = hull_points[:, 0]
        hull_y = hull_points[:, 1]
        # Create a path from convex hull boundary
        boundary_path = Path(hull_points)

    # Grid spacing = 2 * radius (so circles just touch)
    spacing = 2 * radius

    # Get bounding box
    x_min, x_max = np.min(hull_x), np.max(hull_x)
    y_min, y_max = np.min(hull_y), np.max(hull_y)

    # Create square grid
    x_grid = np.arange(x_min, x_max + spacing, spacing)
    y_grid = np.arange(y_min, y_max + spacing, spacing)

    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid)

    # Flatten to get all grid points
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # Filter points inside the region
    inside = boundary_path.contains_points(grid_points)

    grid_x = grid_points[inside, 0]
    grid_y = grid_points[inside, 1]

    return grid_x, grid_y, hull_x, hull_y


def visualize_disks(x_points, y_points, hull_x, hull_y, grid_x, grid_y, radius):
    """
    Visualize the scattered points, convex hull, grid points, and circles.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Scattered points, convex hull, and grid points
    ax1.plot(np.append(hull_x, hull_x[0]), np.append(hull_y, hull_y[0]),
             'b-', linewidth=2, label='Convex Hull')
    ax1.plot(grid_x, grid_y, 'ro', markersize=4, label='Grid Points')
    ax1.plot(x_points, y_points, 'bo', markersize=5, label='Scattered Points')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Region with Grid Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Right plot: Convex hull with circles
    ax2.plot(np.append(hull_x, hull_x[0]), np.append(hull_y, hull_y[0]),
             'b-', linewidth=2, label='Convex Hull')

    # Draw circles at each grid point
    for x, y in zip(grid_x, grid_y):
        circle = Circle((x, y), radius, fill=False, edgecolor='red', alpha=0.6)
        ax2.add_patch(circle)

    # Plot centers
    ax2.plot(grid_x, grid_y, 'ko', markersize=2, alpha=0.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title(f'Circles with radius={radius:.3f} (just touching)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # Set same limits for both plots
    x_margin = (np.max(hull_x) - np.min(hull_x)) * 0.1
    y_margin = (np.max(hull_y) - np.min(hull_y)) * 0.1
    ax1.set_xlim(np.min(hull_x) - x_margin, np.max(hull_x) + x_margin)
    ax1.set_ylim(np.min(hull_y) - y_margin, np.max(hull_y) + y_margin)
    ax2.set_xlim(np.min(hull_x) - x_margin, np.max(hull_x) + x_margin)
    ax2.set_ylim(np.min(hull_y) - y_margin, np.max(hull_y) + y_margin)

    plt.tight_layout()
    plt.show()
    return fig


def visualize_vectors(grid_x, grid_y, grid_vector):
    """
    Visualize the scattered points, convex hull, grid points, and circles.
    """
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(111)
    # Left plot: Scattered points, convex hull, and grid points
    ax1.plot(grid_x, grid_y, 'ro', markersize=4, label='Grid Points')
    ax1.quiver(grid_x, grid_y, grid_vector[0, :], grid_vector[1, :])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Region with Grid Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    plt.tight_layout()
    plt.show()
    return fig


class CoseimicJumpFunction(JumpFunction):
    def __init__(self, config, time_vector, date):
        # invoking coseismic only jump because steps and decays are
        # applied on separate steps
        super().__init__(config, time_vector=time_vector,
                         date=date,
                         jump_type=JumpType.COSEISMIC_ONLY,
                         fit=True)


class PostseismicJumpFunction(JumpFunction):
    def __init__(self, config, time_vector, date):
        super().__init__(config, time_vector=time_vector,
                         date=date,
                         jump_type=JumpType.POSTSEISMIC_ONLY,
                         fit=True)


class ConstraintType(Enum):
    INTERSEISMIC = "interseismic"
    COSEISMIC = "coseismic"
    POSTSEISMIC = "postseismic"
    PERIODIC = "periodic"

    @property
    def function(self) -> Callable:
        """Get the function type"""
        function_map = {
            ConstraintType.INTERSEISMIC: PolynomialFunction,
            ConstraintType.COSEISMIC: CoseimicJumpFunction,
            ConstraintType.POSTSEISMIC: PostseismicJumpFunction,
            ConstraintType.PERIODIC: PeriodicFunction
        }
        return function_map.get(self, PolynomialFunction)

    @property
    def description(self) -> str:
        """Get the function type"""
        description_map = {
            ConstraintType.INTERSEISMIC: "Interseismic",
            ConstraintType.COSEISMIC: "",
            ConstraintType.POSTSEISMIC: "",
            ConstraintType.PERIODIC: "Periodic"
        }
        return description_map.get(self, "Unknown")


@dataclass
class EtmStackerField(BaseDataClass):
    base_type: ConstraintType
    grids: 'GridSystem'
    onset_date: Date = None
    event: Earthquake = None
    relaxation: float = None
    enu_field: np.ndarray = None
    enu_sigma: Union[np.ndarray, None] = None
    enu_covar: Union[np.ndarray, None] = None
    constrain_stations: List['Station'] = field(default_factory=list)
    constrained_parameters: np.ndarray = None
    parameters_indices: np.ndarray = None
    convex_hull: np.ndarray = None

    @property
    def description(self) -> str:
        if self.base_type == ConstraintType.COSEISMIC:
            event = f' {self.event.date.yyyyddd()} {self.event.id:.16}'
        elif self.base_type == ConstraintType.POSTSEISMIC:
            event = f' {self.event.date.yyyyddd()} {self.event.id:.16} ({self.relaxation:.3f})'
        else:
            event = ''
        return self.base_type.description + event

    @classmethod
    def create_field(cls, stations: List['Station'],
                     solution: np.ndarray,
                     covariance: np.ndarray,
                     grids: 'GridSystem',
                     event: Earthquake = None,
                     relaxation: np.ndarray = None,
                     coseismic_constraint: 'CoseismicConstraint' = None):

        # total parameters
        tp = solution.shape[1]

        if event is None:
            idx = np.array([stn.get_velocity_column() for stn in stations])
            v = solution[:, idx]
            # create array with indices of stations for covariance
            idx_ = np.concatenate((idx, idx + tp, idx + tp * 2))
            c = covariance[idx_][:, idx_]
            velocity_field, velocity_sigma, velocity_cova = grids.interpolate_field(stations, v, c)

            return EtmStackerField(ConstraintType.INTERSEISMIC, grids,
                                   enu_field=velocity_field,
                                   enu_sigma=velocity_sigma,
                                   constrain_stations=stations,
                                   constrained_parameters=v,
                                   parameters_indices=idx,
                                   enu_covar=velocity_cova)

        else:
            # get values and sigmas for each station
            return_fields, idx = [], []; coseismic = {}
            for stn in stations:
                par, _ = stn.get_constrained_jump(event, solution, covariance)
                if par is not None:
                    idx.append(stn.get_coseismic_column(event))
                    coseismic[stn] = par

            # if there is something to process, add an EtmStackerField to the return list
            if len(idx) > 1:
                v = np.array(list(coseismic.values())).T
                idx = np.array(idx).flatten()
                idx_ = np.concatenate((idx, idx + tp, idx + tp * 2))
                c = covariance[idx_][:, idx_]
                # do the prediction
                coseismic_field, coseismic_sigma, coseismic_covar = grids.predict_coseismic(event,
                    list(coseismic.keys()), v, c, coseismic_constraint
                )
                return_fields.append(
                    EtmStackerField(
                        ConstraintType.COSEISMIC, grids, enu_field=coseismic_field, onset_date=event.date,
                        enu_sigma=coseismic_sigma, constrain_stations=list(coseismic.keys()),
                        event=event, constrained_parameters=v, parameters_indices=idx, enu_covar=coseismic_covar)
                )

            # now do it for the postseismic fields
            for relax in relaxation:
                idx = []; postseismic = {}
                for stn in stations:
                    par, _ = stn.get_constrained_relax(event, relax, solution, covariance)
                    if par is not None:
                        idx.append(stn.get_postseismic_column(event, relax))
                        postseismic[stn] = par

                if len(idx) > 1:
                    v = np.array(list(postseismic.values())).T
                    idx = np.array(idx).flatten()
                    idx_ = np.concatenate((idx, idx + tp, idx + tp * 2))
                    c = covariance[idx_][:, idx_]
                    # do the interpolation
                    postseismic_field, postseismic_sigma, postseismic_covar = grids.interpolate_field(
                        list(postseismic.keys()), v, c, event)

                    return_fields.append(
                        EtmStackerField(
                            ConstraintType.POSTSEISMIC, grids, enu_field=postseismic_field, onset_date=event.date,
                            enu_sigma=postseismic_sigma, constrain_stations=list(postseismic.keys()),
                            event=event, relaxation=relax, constrained_parameters=v,
                            parameters_indices=idx, enu_covar=postseismic_covar)
                    )

            return return_fields
    def get_interpolation_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.base_type == ConstraintType.INTERSEISMIC:
            return self.grids.interpolation_grid[0], self.grids.interpolation_grid[1]
        elif self.base_type == ConstraintType.POSTSEISMIC:
            mask = self.grids.earthquake_masks[self.event.id][1]
        else:
            mask = self.grids.earthquake_masks[self.event.id][0]

        return self.grids.interpolation_grid[0][mask], self.grids.interpolation_grid[1][mask]

    def get_interpolation_grid_geographic(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.base_type == ConstraintType.INTERSEISMIC:
            return self.grids.interpolation_geographic[0], self.grids.interpolation_geographic[1]
        elif self.base_type == ConstraintType.POSTSEISMIC:
            mask = self.grids.earthquake_masks[self.event.id][1]
        else:
            mask = self.grids.earthquake_masks[self.event.id][0]

        return self.grids.interpolation_geographic[0][mask], self.grids.interpolation_geographic[1][mask]

    def get_values_geo(self, lon: float, lat: float) -> Tuple[Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None],
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None]]:
        """get the field value interpolated to input lon lat"""
        # first check if the station needs this field (CO or POST)
        if self.base_type in (ConstraintType.COSEISMIC, ConstraintType.POSTSEISMIC):
            # check if we really need to apply
            mask = self.grids.earthquake_masks[self.event.id][2]
            s_score, p_score = mask.score(lat, lon)
            if s_score == 0 and self.base_type == ConstraintType.COSEISMIC:
                return None, None
            elif p_score == 0 and self.base_type == ConstraintType.POSTSEISMIC:
                return None, None

        x, y = self.grids.transform_coordinate(lon, lat)

        return self.get_values_proj(x, y)

    def get_values_proj(self,
                        x: float,
                        y: float) -> Union[Tuple[None, None],
                        Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """get the field value interpolated to input x y"""
        # unpack the grid values
        xg, yg = self.get_interpolation_grid()
        # unpack the field values
        ve, vn, vu = self.enu_field
        if self.enu_sigma is not None:
            se, sn, su = self.enu_sigma
        # determine the valid locations
        valid = ~np.isnan(ve)

        # find the data within a certain radius of the requested point
        r, _, _ = get_radius(np.column_stack([xg, yg]), np.column_stack([x, y]))
        # invalidate range to points outside the valid area
        r[0, ~valid] = np.inf
        # get 10 closest points (for safety)
        sorted_indices = np.argsort(r, axis=None)
        valid[sorted_indices[10:]] = False

        if np.sum(valid) > 3:
            ve_int = griddata((xg[valid], yg[valid]), ve[valid], (x, y), method='cubic')
            vn_int = griddata((xg[valid], yg[valid]), vn[valid], (x, y), method='cubic')
            vu_int = griddata((xg[valid], yg[valid]), vu[valid], (x, y), method='cubic')
            # now interpolate sigmas
            if self.enu_sigma is not None:
                se_int = griddata((xg[valid], yg[valid]), se[valid], (x, y), method='cubic')
                sn_int = griddata((xg[valid], yg[valid]), sn[valid], (x, y), method='cubic')
                su_int = griddata((xg[valid], yg[valid]), su[valid], (x, y), method='cubic')

            if not np.isnan(ve_int):
                if self.base_type == ConstraintType.INTERSEISMIC:
                    ve_out = np.array([0, ve_int])
                    vn_out = np.array([0, vn_int])
                    vu_out = np.array([0, vu_int])

                    if self.enu_sigma is not None:
                        se_out = np.array([0, se_int])
                        sn_out = np.array([0, sn_int])
                        su_out = np.array([0, su_int])
                    else:
                        se_out = np.array([0, 0])
                        sn_out = np.array([0, 0])
                        su_out = np.array([0, 0])
                else:
                    ve_out = np.array([ve_int])
                    vn_out = np.array([vn_int])
                    vu_out = np.array([vu_int])

                    if self.enu_sigma is not None:
                        se_out = np.array([se_int])
                        sn_out = np.array([sn_int])
                        su_out = np.array([su_int])
                    else:
                        se_out = np.array([0, 0])
                        sn_out = np.array([0, 0])
                        su_out = np.array([0, 0])
            else:
                return None, None
        else:
            return None, None

        return (ve_out, vn_out, vu_out), (se_out, sn_out, su_out)

    def get_etm_function(self, lon: float, lat: float, time_vector: np.ndarray,
                         network_code: str = '', station_code: str = ''):
        """
        given a position in lat lon and a time vector (optionally provide a network and station code)
        get the corresponding EtmFunction
        """
        config = EtmConfig(network_code, station_code)
        config.metadata.lon = np.array([lon])
        config.metadata.lat = np.array([lat])
        config.modeling.relaxation = np.array([self.relaxation])

        # init the AdjustmentResults to pass to the etm_function
        result = [AdjustmentResults(), AdjustmentResults(), AdjustmentResults()]
        values, sigmas = self.get_values_geo(lon, lat)

        if values is not None:
            for i in range(3):
                result[i].parameters = values[i]
                result[i].parameter_sigmas = sigmas[i]
                # dummy covariance matrix
                result[i].covariance_matrix = np.ones((2, 2))

            funct = self.base_type.function(config=config, time_vector=time_vector, date=self.onset_date)

            funct.column_index = np.arange(0, funct.param_count)
            funct.load_parameters(result)

            return funct
        else:
            return None

    def eval(self, lon: float, lat: float, time_vector: np.ndarray):
        # a generic config to pass to the functions
        funct = self.get_etm_function(lon, lat, time_vector)

        if funct is not None:
            values = []
            for i in range(3):
                if self.base_type != ConstraintType.INTERSEISMIC:
                    # must remove interseismic
                    val = funct.eval(i, remove_postseismic=True)
                else:
                    val = funct.eval(i)

                if np.any(np.isnan(val)):
                    values.append(0)
                else:
                    values.append(val)

            return np.array(values)
        return np.zeros((3, time_vector.size))


@dataclass
class EtmStackerConfig(BaseDataClass):
    max_condition_number: float = 15
    earthquake_magnitude_limit: float = 8.0
    post_seismic_back_lim: Union[float, Date] = Date(year=1990, doy=1)
    relaxation: np.ndarray = field(default_factory=lambda: np.array([0.05, 1.0]))
    earthquakes_cherry_picked: list = field(default_factory=list)
    earthquakes_to_remove: list = field(default_factory=list)
    station_weight_scale: float = 1.0
    interseismic_h_sigma: float = 0.0005
    interseismic_v_sigma: float = 0.001
    coseismic_h_sigma: float = 10.
    coseismic_v_sigma: float = 10.
    postseismic_h_sigma: float = 0.001
    postseismic_v_sigma: float = 0.003
    vertical_method: str = 'spline2d'
    vertical_load_radius: float = 50 # in km
    tension: float = 0.10
    grid_spacing: float = 25 # in km


@dataclass
class NormalEquations:
    """Holds normal equation components for a station"""
    station: str
    # stored as ENU
    neq: List[np.ndarray] = field(
        default_factory=lambda: [np.array([]), np.array([]), np.array([])]
    )
    # stored as ENU
    ceq: List[np.ndarray] = field(
        default_factory=lambda: [np.array([]), np.array([]), np.array([])]
    )
    design_matrix: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    # stored as ENU
    observation_vector: List[np.ndarray] = field(
        default_factory=lambda: [np.array([]), np.array([]), np.array([])]
    )
    observation_weights: List[np.ndarray] = field(
        default_factory=lambda: [np.array([]), np.array([]), np.array([])]
    )
    weighted_observations: List[float] = field(
        default_factory=lambda: [0., 0., 0.]
    )
    weight_scale: float = 1.0
    dof: int = 0
    prior_wrms: List[float] = field(
        default_factory=lambda: [0., 0., 0.]
    )

    parameter_count: int = 0
    equation_count: int = 0
    parameter_start_idx: int = 0
    # range of indices in the general normal equations matrix
    parameter_range: List[np.ndarray] = field(
        default_factory=lambda: [np.array([])]
    )


@dataclass
class Station:
    """Pure data about a station - no computation logic"""
    network_code: str
    station_code: str
    lon: float
    lat: float
    first_obs: Date = None
    etm: EtmEngine = None
    normal_equations: NormalEquations = None

    # Geometric properties
    projected_coords: Tuple[float, float] = (0., 0.)
    vertical_response: np.ndarray = field(default_factory=lambda: np.array([]))
    earthquake_responses: Dict = field(default_factory=dict)

    # Parameter tracking
    is_interseismic: bool = False
    etm_constraints: List = None
    posterior_wrms: List = None

    def __str__(self):
        return f'{self.network_code}.{self.station_code}'

    def get_velocity_column(self):
        polynomial = self.etm.design_matrix.get_polynomial()
        return polynomial.column_index[1] + self.normal_equations.parameter_start_idx

    def get_periodic_column(self, frequency: float):
        """by default returns the index of sin"""
        periodic = self.etm.design_matrix.get_periodic()
        if periodic is not None:
            return np.array(periodic.get_periodic_cols(frequency)) + self.normal_equations.parameter_start_idx
        else:
            return None

    def get_coseismic_column(self, jump_or_id: Union[Earthquake, str]):
        """get the column of a given earthquake or earthquake id"""
        if isinstance(jump_or_id, Earthquake):
            event_id = jump_or_id.id
        else:
            event_id = jump_or_id

        jumps = [j for j in self.etm.jump_manager.jumps if j.is_geophysical()
                 and j.earthquake is not None and j.p.jump_type < JumpType.POSTSEISMIC_ONLY and
                 j.earthquake.id == event_id and j.fit]

        if jumps:
            return jumps[0].get_jump_col() + self.normal_equations.parameter_start_idx
        else:
            return None

    def get_postseismic_column(self, jump_or_id: Union[Earthquake, str], relaxation: float):
        """get the column of a given earthquake or earthquake id and relaxation"""
        if isinstance(jump_or_id, Earthquake):
            event_id = jump_or_id.id
        else:
            event_id = jump_or_id

        jumps = [j for j in self.etm.jump_manager.jumps if j.is_geophysical()
                 and j.earthquake is not None and j.p.jump_type != JumpType.COSEISMIC_ONLY and
                 j.earthquake.id == event_id and j.fit]

        if jumps:
            # @ todo: remove the np.array: this function is meant to retrieve one relaxation at the time, so
            # @ todo: no need to return an array
            return np.array(jumps[0].get_relaxation_cols(relaxation)) + self.normal_equations.parameter_start_idx
        else:
            return None

    def get_constrained_velocity(self, solution, covariance):
        """return the velocity and sigma"""
        idx = self.get_velocity_column()
        tp = solution.shape[1]
        return solution[:, idx], np.sqrt(np.diag(covariance)[[idx, idx + tp, idx + 2 * tp]])

    def get_constrained_periodic(self, frequency: float, solution, covariance):
        """return the velocity and sigma"""
        idx = self.get_periodic_columns(frequency)
        tp = solution.shape[1]
        return solution[:, idx], np.sqrt(np.diag(covariance)[[idx, idx + tp, idx + 2 * tp]])

    def get_constrained_jump(self, event: Union[Earthquake, str],
                             solution: np.ndarray,
                             covariance: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:

        """return the jump and sigma or none if not fit"""

        idx = self.get_coseismic_column(event)
        if idx:
            tp = solution.shape[1]
            return solution[:, idx], np.sqrt(np.diag(covariance)[[idx, idx + tp, idx + 2 * tp]])
        else:
            return None, None

    def get_constrained_relax(self, event: Union[Earthquake, str],
                          relaxation: float,
                          solution: np.ndarray,
                          covariance: np.ndarray) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:

        idx = self.get_postseismic_column(event, relaxation)
        tp = solution.shape[1]
        if idx:
            # this only requests one relaxation so solution[:, idx] can be flattened to be only 1d
            return solution[:, idx].flatten(), np.sqrt(np.diag(covariance)[[idx, idx + tp, idx + 2 * tp]]).flatten()
        else:
            return None, None

    def extract_etm_constraints(self,
                                earthquakes: List[Earthquake],
                                relaxations: np.ndarray,
                                solution: np.ndarray,
                                covariance: np.ndarray):

        # for mapping NEU to ENU
        col = [1, 0, 2]

        self.etm_constraints = []

        # create the polynomial constraint
        poly = PolynomialFunction(self.etm.config)
        vel, sig = self.get_constrained_velocity(solution, covariance)
        # MULT sigma to force ETM to honor the stacked parameters
        MULT = 100
        for j in range(3):
            # assign the values
            poly.p.params[j][1] = vel[col[j]]
            poly.p.sigmas[j][1] = sig[col[j]] / MULT

        self.etm_constraints.append(poly)

        # create a constraint for each earthquake
        for event in earthquakes:
            jump = self.etm.jump_manager.get_geophysical_jump(event.id)

            if jump and jump.fit:
                # create the constraint
                jc = JumpFunction(self.etm.config, time_vector=np.array([0]),
                                  date=jump.date, jump_type=jump.p.jump_type, fit=False)

                if jump.p.jump_type < JumpType.POSTSEISMIC_ONLY:
                    par, sig = self.get_constrained_jump(event, solution, covariance)

                    for j in range(3):
                        # assign the values (coseismic, always  0)
                        jc.p.params[j][0] = par[col[j]]
                        jc.p.sigmas[j][0] = sig[col[j]] / MULT

                for relax in relaxations:
                    par, sig = self.get_constrained_relax(event, relax, solution, covariance)
                    idj = jump.get_relaxation_cols(relax, False)

                    for j in range(3):
                        # assign the values
                        jc.p.params[j][idj] = par[col[j]]
                        jc.p.sigmas[j][idj] = sig[col[j]] / MULT

                self.etm_constraints.append(jc)
        return

    def __hash__(self):
        """Hash based on station identity only"""
        return hash((self.network_code, self.station_code))

    def __eq__(self, other):
        """Two stations are equal if they have the same network and station code"""
        if not isinstance(other, Station):
            return False
        return (self.network_code == other.network_code and
                self.station_code == other.station_code)


@dataclass
class ConstraintEquation:
    """Parameters for a single constraint equation"""
    station: Station
    # design matrix of the constraint
    constraint_design: Tuple[np.ndarray, np.ndarray, np.ndarray] = field(
        default_factory=lambda: [np.array([]), np.array([]), np.array([])]
    )
    constraint_sigma: np.ndarray = field(default_factory=lambda: np.array([]))  # Constraint sigma (for weighting)
    is_active: bool = True  # Can be toggled on/off


@dataclass
class GridSystem:
    """Encapsulates all grid-related computations"""
    origin: Tuple[float, float]  # (lon, lat)
    interpolation_grid: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    interpolation_geographic: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    grid_vertical_response: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    points: int = 0
    offset: float = 0.0
    poisson_ratio: float = 0.27

    # to store the earthquake responses for the grid
    earthquake_responses: Dict = field(default_factory=dict)
    earthquake_masks: Dict = field(default_factory=dict)

    @classmethod
    def create_from_stations(cls, stations: List[Station],
                             grid_spacing: float = 20,
                             grid_load_radius: float = 50,
                             method='rectload',
                             tension: float = 0.10) -> 'GridSystem':
        """Factory method to create grid system from stations"""
        lat = np.array([stn.lat for stn in stations])
        lon = np.array([stn.lon for stn in stations])

        tqdm.write(f'Creating interpolation grids, station count {len(lat)}')
        tqdm.write(f'lon {np.array(lon).min()} {np.array(lon).max()} '
                    f'lat {np.array(lat).min()} {np.array(lat).max()}')

        # find an origin for the grid system
        grids_origin = (np.array(lon).mean(), np.array(lat).mean())

        ################################################################################
        # project the stations coordinates
        x, y = azimuthal_equidistant(grids_origin[0], grids_origin[1], lon, lat)

        # compute the median of the station separation
        r, _, _ = get_radius(np.column_stack([x, y]),
                             np.column_stack([x, y]))

        tqdm.write(f'Station distance statistics -> mean: {np.mean(r):.1f} km, median: {np.median(r):.1f} km')
        offset = float(np.median(r))

        stations_projected = np.array([x, y])

        # save the station projected coordinates in the station object for convenience
        for i, stn in enumerate(stations):
            stn.projected_coords = (stations_projected[0, i], stations_projected[1, i])

        ################################################################################
        # create the grid and save it
        grid_x, grid_y, _, _ = fill_region_with_grid(x, y, grid_spacing / 2)

        # find the lat lon of points to apply the land/ocean mask
        glon, glat = inverse_azimuthal(grids_origin[0], grids_origin[1], grid_x, grid_y)
        mask = mask_ocean_points(glon, glat, grid_spacing * 4)
        grid_x = grid_x[mask]
        grid_y = grid_y[mask]

        points = grid_x.size
        # visualize_disks(x,y,hx,hy,grid_x,grid_y,grid_spacing / 2)
        interpolation_grid = np.array([grid_x, grid_y])
        interpolation_geographic = np.array(
            inverse_azimuthal(grids_origin[0], grids_origin[1], grid_x, grid_y))

        if method == 'diskload':
            # disk load as the preferred response
            disk_grid_x, disk_grid_y, _, _ = fill_region_with_grid(
                stations_projected[0], stations_projected[1], grid_load_radius
            )

            grid_vertical_response = cls._compute_disk_responses(
                stations, disk_grid_x, disk_grid_y, grid_load_radius, interpolation_grid
            )
        elif method == 'spline2d':

            tqdm.write(f'Computing spline grid response for interpolation '
                       f'grid (total points {grid_x.shape[0]})')

            grid_vertical_response, stations_vertical_response = cls._compute_spline_responses(
                stations_projected, grid_x, grid_y, tension
            )
            # assign the vertical response to each station
            for i, station in enumerate(stations):
                station.vertical_response = stations_vertical_response[i, :]

        elif method == 'rectload':
            disk_grid_x, disk_grid_y, _, _ = fill_region_with_grid(
                stations_projected[0], stations_projected[1], grid_load_radius
            )
            grid_vertical_response = cls._compute_rectload_responses(
                stations, stations_projected, disk_grid_x, disk_grid_y, grid_load_radius, interpolation_grid
            )
        else:
            raise ValueError(f'Method {method} not implemented')

        instance = GridSystem(grids_origin, interpolation_grid, interpolation_geographic,
                              grid_vertical_response, points, offset)

        tqdm.write(f'Created interpolation grids with size {grid_x.shape}')

        return instance

    @staticmethod
    def _compute_rectload_responses(stations: List[Station], stations_projected: np.ndarray,
                                    disk_grid_x: np.ndarray, disk_grid_y: np.ndarray, disk_grid_spacing: float,
                                    interpolation_grid: np.ndarray):

        tqdm.write(f'Computing rectangular load grid response for interpolation '
                   f'grid (total points {interpolation_grid[0].shape[0]})')

        # average earth crust
        e = 50e6
        poisson_ratio = 0.27
        # get mu and lambda
        mu = e / (2 * (1 + poisson_ratio))
        lmb = e * poisson_ratio / ((1+poisson_ratio) * (1 - 2 * poisson_ratio))

        # Extract coordinates
        x, y = stations_projected

        [_, _, a] = mrectloadhs_dif(np.vstack((x, y, np.zeros_like(x))),
                                    np.vstack((disk_grid_x - disk_grid_spacing, disk_grid_x + disk_grid_spacing)).T,
                                    np.vstack((disk_grid_y - disk_grid_spacing, disk_grid_y + disk_grid_spacing)).T,
                                    lmb , mu)

        for i, station in enumerate(stations):
            station.vertical_response = a[i, :]

        grid_x, grid_y = interpolation_grid
        ########## now compute the response of the grid ##############
        [_, _, a] = mrectloadhs_dif(np.vstack((grid_x, grid_y, np.zeros_like(grid_x))),
                                    np.vstack((disk_grid_x - disk_grid_spacing, disk_grid_x + disk_grid_spacing)).T,
                                    np.vstack((disk_grid_y - disk_grid_spacing, disk_grid_y + disk_grid_spacing)).T,
                                    lmb, mu)

        tqdm.write(f'disk count: {disk_grid_x.shape} '
                   f'interpolation grid count: {interpolation_grid[0].shape} '
                   f'size of response matrix: {a.shape}')

        return a

    @staticmethod
    def _compute_spline_responses(stations_projected: np.ndarray,
                                  grid_x: np.ndarray, grid_y: np.ndarray, tension=0.10):

        length_scale = np.abs(np.max(grid_x.flatten()) - np.min(grid_x.flatten()) +
                              1j * (np.max(grid_y.flatten()) - np.min(grid_y.flatten()))) / 50

        p = np.sqrt(tension / (1 - tension))
        p = p / length_scale

        # Extract coordinates
        x, y = stations_projected

        # Compute all pairwise distances at once using broadcasting
        # Create n×n matrices of coordinate differences
        dx = x[:, np.newaxis] - x[np.newaxis, :]  # or: x[:, None] - x
        dy = y[:, np.newaxis] - y[np.newaxis, :]

        # Compute complex distances and take absolute value
        r = np.abs(dx + 1j * dy)

        # Apply spline2dgreen to entire distance matrix at once
        stations_vertical_response = spline2dgreen(r, p)

        ########## now compute the response of the grid ##############
        # Flatten and compute distances in fewer lines
        grid_x_flat = grid_x.ravel()
        grid_y_flat = grid_y.ravel()

        r = np.abs((grid_x_flat[:, None] - x) + 1j * (grid_y_flat[:, None] - y))
        grid_vertical_response = spline2dgreen(r, p)

        return grid_vertical_response, stations_vertical_response

    @staticmethod
    def _compute_disk_responses(stations, disk_grid_x, disk_grid_y, disk_radius,
                                interpolation_grid, nmax_max=2500):
        # read love numbers now to accelerate the process later
        h_love, l_love, k_love = load_love_numbers()

        grid_vertical_response = []
        tqdm.write(f'Computing disk grid response for interpolation '
                   f'grid (total points {interpolation_grid[0].shape[0]})')

        bar = tqdm(total=interpolation_grid[0].shape[0], ncols=120)
        # compute the responses for the interpolation grid
        for i, (x, y) in enumerate(interpolation_grid.T):
            r, _, _ = get_radius(np.column_stack([disk_grid_x, disk_grid_y]),
                                 np.column_stack([x, y]))
            # add an offset to avoid singularities
            r = r + 8
            response = compute_diskload(alpha=disk_radius, theta_range=r, nmax_max=nmax_max,
                                        h_love=h_love, l_love=l_love, k_love=k_love)

            grid_vertical_response.append(response['U'])
            bar.update()

        bar.close()
        grid_vertical_response = np.array(grid_vertical_response)

        # compute the response for each station
        tqdm.write(f'Computing disk grid response for {len(stations)} stations')
        bar = tqdm(total=len(stations), ncols=120)

        for i, stn in enumerate(stations):
            r, _, _ = get_radius(np.column_stack([disk_grid_x, disk_grid_y]),
                                 np.column_stack([stn.projected_coords[0],
                                                  stn.projected_coords[1]]))
            # add an offset to avoid singularities
            r = r + 8
            response = compute_diskload(alpha=disk_radius, theta_range=r, nmax_max=nmax_max,
                                        h_love=h_love, l_love=l_love, k_love=k_love)

            # response on the site to all disks
            stn.vertical_response = response['U'].flatten()
            bar.update()

        bar.close()
        tqdm.write(f'disk count: {disk_grid_x.shape} '
                   f'interpolation grid count: {interpolation_grid[0].shape} '
                   f'size of response matrix: {grid_vertical_response.shape}')

        return grid_vertical_response

    def transform_coordinate(self, lon: float, lat: float):
        return azimuthal_equidistant(np.array(self.origin[0]), np.array(self.origin[1]),
                                     np.array(lon), np.array(lat))

    def compute_horizontal_interpolant_at_point(self, target_x: float, target_y: float,
                                                source_x: np.ndarray,
                                                source_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sandwell-Wessel interpolation for a single point"""
        # compute response between all stations
        q, p, w = get_qpw(np.column_stack([source_x, source_y]), np.column_stack([source_x, source_y]),
                          self.offset, self.poisson_ratio)
        a = np.block([[q, w], [w, p]])  # type: ignore[arg-type]

        # compute response between current station (0, 0) and all stations
        q, p, w = get_qpw(np.column_stack([source_x, source_y]), np.array([[target_x, target_y]]),
                          self.offset, self.poisson_ratio)

        ke = np.linalg.solve(a, np.concatenate((q.flatten(), w.flatten())))
        kn = np.linalg.solve(a, np.concatenate((w.flatten(), p.flatten())))

        return ke, kn

    @staticmethod
    def compute_vertical_interpolant_at_point(target_response: np.ndarray,
                                              source_responses: List[np.ndarray]) -> np.ndarray:
        """Elastic load interpolation for a single point"""

        # get the SVD decomposition for the design matrix
        u, s, vt = np.linalg.svd(np.array(source_responses), full_matrices=True)
        v = vt.T
        # a number smaller than the number of observations to truncate the SVD decomp
        # k = np.max(np.where(s >= 1)[0])
        # k = int(len(source_responses) / 4)
        k = GridSystem._determine_svd_truncation(s)

        u_k = u[:, :k]
        s_k = s[:k]
        v_k = v[:, :k]

        ku = target_response @ (v_k / s_k) @ u_k.T

        return ku

    def compute_horizontal_grid_interpolant(self, source_x: np.ndarray,
                                            source_y: np.ndarray,
                                            mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate to full grid (for visualization)"""
        # now create the influence matrix for each grid point to use in the horizontal interpolation
        # compute response between all stations participating
        q, p, w = get_qpw(np.column_stack([source_x, source_y]),
                          np.column_stack([source_x, source_y]), self.offset, self.poisson_ratio)
        a = np.block([[q, w], [w, p]])  # type: ignore[arg-type]

        # compute response between current station (0, 0) and all stations
        q, p, w = get_qpw(np.column_stack([source_x, source_y]),
                          np.column_stack([self.interpolation_grid[0][mask],
                                           self.interpolation_grid[1][mask]]), self.offset, self.poisson_ratio)
        # here, we do (A^-1 * B^t)^t so that we can multiply by v and get the answer at the interp points
        ke = np.linalg.solve(a, np.hstack((q, w)).T).T
        kn = np.linalg.solve(a, np.hstack((w, p)).T).T

        return ke, kn

    def compute_vertical_grid_interpolant(self, source_responses: List[np.ndarray],
                                          grid_vertical_response: np.ndarray) -> np.ndarray:
        """Interpolate vertical to full grid"""
        a = np.array(source_responses)
        g = grid_vertical_response

        # get the SVD decomposition for the design matrix
        u, s, vt = np.linalg.svd(a, full_matrices=True)
        v = vt.T

        # k = np.max(np.where(s >= 1)[0])
        # k = int(len(source_responses) / 4)
        k = self._determine_svd_truncation(s)

        u_k = u[:, :k]
        s_k = s[:k]
        v_k = v[:, :k]

        ku = g @ (v_k / s_k) @ u_k.T

        return ku

    @staticmethod
    def _determine_svd_truncation(s):
        """
        determine the truncation to apply so that the answer has no more than 7 times more
        uncertainty than the input. This comes from the fact that
        uncertainty(solution) = Cond * uncertainty(data)
        """
        #r = s[0] / s
        #if np.max(np.where(r <= 2)[0]) == 0:
            # too unstable, return fixed number of values
        return int(len(s) * 4 / 5)

        #return np.max(np.where(r <= 7)[0])

    def interpolate_field(self, stations: List[Station],
                          enu: np.ndarray, covar: np.ndarray,
                          event: Earthquake = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """output enu_cov is en, e-u, n-u"""
        # get the number of parameters for this interpolation

        x = np.array([stn.projected_coords[0] for stn in stations])
        y = np.array([stn.projected_coords[1] for stn in stations])

        if event is not None:
            _, p_mask, _ = self.earthquake_masks[event.id]
            active_points = np.sum(p_mask)
        else:
            p_mask = np.array([True] * self.points)
            active_points = self.points

        grid_vertical_response = self.grid_vertical_response[p_mask, :]
        stations_vertical_response = [stn.vertical_response for stn in stations]
        xg, yg = self.interpolation_grid[0][p_mask], self.interpolation_grid[1][p_mask]

        # compute the grid velocities
        ae, an = self.compute_horizontal_grid_interpolant(x, y, p_mask)

        ah = np.vstack((ae, an))

        # find the ideal tension for this field
        pv, av = self._find_tension(x, y, xg, yg, np.zeros_like(xg), enu[2, :].flatten(), enu[2, :].flatten())
        # av = self.compute_vertical_grid_interpolant(stations_vertical_response, grid_vertical_response)

        # full design matrix for sigma values
        zeros = np.zeros((ae.shape[0], av.shape[1]))
        at = np.vstack((np.hstack((ae, zeros)),
                        np.hstack((an, zeros)),
                        np.hstack((zeros, zeros, av))))

        result = np.concatenate((ah @ enu[0:2, :].flatten(), pv))
        predict_cova = at @ covar @ at.T

        values = np.reshape(result, (3, active_points))
        enu_sigmas = np.reshape(np.sqrt(np.diag(predict_cova)), (3, active_points))
        # extract the ENU covariance
        en_cov = np.diag(predict_cova, k=active_points)[:active_points]  # Cov(E,N)
        eu_cov = np.diag(predict_cova, k=2 * active_points)[:active_points]  # Cov(E,U)
        nu_cov = np.diag(predict_cova, k=active_points)[active_points:]  # Cov(N,U)

        # Reshape to (3, n_points)
        enu_cov = np.reshape(np.concatenate((en_cov, nu_cov, eu_cov)),(3, active_points))

        return values, enu_sigmas, enu_cov

    def predict_coseismic(self, event: Earthquake, stations: List[Station],
                          observations: np.ndarray,
                          covar: np.ndarray,
                          constraint: 'CoseismicConstraint') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """output enu_cov is en, eu, nu"""
        tqdm.write(f'Predicting jumps for {event.id}')

        sites = np.isin(constraint.station_list, [stationID(stn) for stn in stations])

        # grab the sites responses
        ae, an, au = [model[sites, :] for model in constraint.dislocation_model]
        # grab the grid response (already masked in CoseismicConstraint.compute_constraint_coefficients)
        pe, pn, pu = self.earthquake_responses[event.id]

        s_score, _, _ = self.earthquake_masks[event.id]

        # get how many points we actually need
        active_points = np.sum(s_score)

        depth = constraint.dislocation_grid[2][constraint.plane]
        depth_range = np.max(depth) - np.min(depth)

        start_stop = np.mean([np.log10(constraint.search_stop_smoothing),
                              np.log10(constraint.search_start_smoothing)])

        start = np.logspace(np.log10(constraint.search_start_smoothing), start_stop, 10)
        stop = np.logspace(start_stop, np.log10(constraint.search_stop_smoothing), 10)
        # print(start)
        # print(stop)
        # Create meshgrid for plotting
        START, STOP = np.meshgrid(start, stop, indexing='ij')

        c_field = np.full((3, active_points), np.nan)

        c_name = ['(a) East', '(b) North', '(c) Up']
        # assume noise floot of np.sqrt(0.001**2 + 0.001**2 + 0.003**2)
        #nf = np.sqrt(0.001**2 + 0.001**2 + 0.003**2)
        #snr = np.hypot(observations[0, :], observations[1, :], observations[2, :]) / nf
        #snrl = 3.
        # determine the minimum rms
        for i in range(3):
            p_ = self.earthquake_responses[event.id][i]
            a_ = constraint.dislocation_model[i]

            # weights for making all stations count
            #s = np.abs(np.sum(a_, axis=1))
            #w_ = 1 / (s / np.max(s))
            #w_[snr < snrl] = w_[snr < snrl] * 10.**(snr[snr < snrl] - snrl)
            #w_[w_ < np.finfo(float).eps] = np.finfo(float).eps
            #w_ = np.diag(w_)

            c_ = a_.T @ observations[i, :]
            n_ = a_.T @ a_

            x, residual, rms = self._find_smoothing(
                stations, start, stop, depth, depth_range,
                constraint, n_, c_, observations[i], p_, i, a_
            )

            c_field[i, :] = (p_ @ x) + residual

            if rms is not None:
                # only plot if the coefficients were determined
                if i == 0:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                ax = axes[i]
                # Plot as colored mesh
                im = ax.pcolormesh(START, STOP, rms * 1000., shading='auto', cmap='viridis')
                # Find and mark minimum RMS
                min_idx = np.unravel_index(np.argmin(rms), rms.shape)
                min_start = start[min_idx[0]]
                min_stop = stop[min_idx[1]]
                min_rms = rms[min_idx] * 1000.
                ax.plot(min_start, min_stop, 'r*', markersize=15,
                        label=f'{c_name[i]} min RMS={min_rms:.2f}')
                # print to screen
                tqdm.write(f'{c_name[i]} min RMS={min_rms:.2f}')

                # Log scales
                ax.set_xscale('log')
                ax.set_yscale('log')
                # Labels
                ax.set_xlabel('Smoothing start', fontsize=12)
                if i == 0:
                    ax.set_ylabel('Smoothing stop', fontsize=12)
                ax.set_title(f'{c_name[i]} grid search', fontsize=14)
                # Colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('RMS [mm]', fontsize=12)
                # Legend
                ax.legend()
                if i == 2:
                    plt.tight_layout()
                    if not os.path.exists('./production/'):
                        os.makedirs('./production/')
                    plt.savefig(f'./production/{event.id}_search.png')
                    plt.close()

        p = np.linalg.inv(covar)
        a = np.vstack((ae, an, au))

        # compute uncertainties
        a_ = np.vstack((pe, pn, pu))
        # extend covar
        covar_e = np.linalg.inv(a.T @ p @ a + constraint.laplacian.T @ constraint.laplacian)
        predict_cova = a_ @ covar_e @ a_.T
        # reshape the result
        enu_sigma = np.reshape(np.sqrt(np.diag(predict_cova)), (3, active_points))

        # extract the ENU covariance
        en_cov = np.diag(predict_cova, k=active_points)[:active_points]  # Cov(E,N)
        eu_cov = np.diag(predict_cova, k=2 * active_points)[:active_points]  # Cov(E,U)
        nu_cov = np.diag(predict_cova, k=active_points)[active_points:]  # Cov(N,U)

        # Reshape to (3, n_points)
        enu_cov = np.reshape(np.concatenate((en_cov, eu_cov, nu_cov)),(3, active_points))

        return c_field, enu_sigma, enu_cov

    def _find_smoothing(self, stations, start, stop, depth, depth_range,
                        constraint, neq, ceq, observations, p_, comp, a_) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        """method to find the best smoothing value for a fault"""

        # get the masked interpolation grid
        mask = self.earthquake_masks[constraint.event.id][0]
        xg, yg = self.interpolation_grid[0][mask], self.interpolation_grid[1][mask]
        # get site positions
        xs, ys = np.array([list(stn.projected_coords) for stn in stations]).T
        # initialize RMS variable
        rms = None

        if constraint.start_smoothing[comp] is None:
            tqdm.write('Need to determine ideal smoothing coefficients, this can take a while...')

            x = np.zeros((len(start), len(stop), neq.shape[1]))
            rms = np.zeros((len(start), len(stop)))

            for row, s in tqdm(enumerate(start), ncols=120):
                for col, e in enumerate(stop):
                    w = s - (depth - np.min(depth)) * (s - e) / depth_range
                    # create laplacian constraint based on depth
                    l = (constraint.laplacian.T @ np.diag(np.concatenate((w, w))) @ constraint.laplacian)
                    # apply to normal equations
                    n = neq + l
                    # compute resolution matrix
                    r = np.abs(np.diag(np.linalg.solve(n, neq))).reshape((2, -1))
                    w = np.array([])
                    # do strike and downdip slip separately
                    for i in range(2):
                        # normalize (1 = the best resolved patch)
                        rn = r[i] / np.max(r[i])
                        r_range = np.max(rn) - np.min(rn)
                        # worst resolved patch (r - np.min(r) = 0) gets the highest smoothing
                        # best resolved patch (r - np.min(r) = r_range) gets smallest smoothing
                        w = np.concatenate((w, s - (rn - np.min(rn)) * (s - e) / r_range))

                    l = (constraint.laplacian.T @ np.diag(w) @ constraint.laplacian)
                    # create a better laplacian based on resolution of depth-dependent laplacian
                    n = neq + l
                    # solve system
                    x[row, col, :] = np.linalg.solve(n, ceq)
                    # predict values and compute residuals
                    f = p_ @ x[row, col, :]

                    # set to zero any possible stations just outside the mask (should not happen!)
                    f_int = griddata((xg, yg), f, (xs, ys), method='cubic', fill_value=0)

                    rms[row, col] = np.sqrt(np.sum((observations - f_int) ** 2) / len(observations))

            min_idx = np.unravel_index(np.argmin(rms), rms.shape)

            constraint.start_smoothing[comp] = start[min_idx[0]]
            constraint.stop_smoothing[comp] = stop[min_idx[1]]

            x = x[min_idx[0], min_idx[1], :]

            f = p_ @ x
            f_int = griddata((xg, yg), f, (xs, ys), method='cubic', fill_value=0)
            vg = observations - f_int
            vo = observations - a_ @ x
            tqdm.write('Station  inverse interpolation')
            for i, stn in enumerate(stations):
                tqdm.write(f' -- {stationID(stn)} {vo[i]*1000.:7.1f} {vg[i]*1000.:7.1f} mm')

        else:
            s = constraint.start_smoothing[comp]
            e = constraint.stop_smoothing[comp]
            w = s - (depth - np.min(depth)) * (s - e) / depth_range
            # create laplacian constraint
            l = (constraint.laplacian.T @ np.diag(np.concatenate((w, w))) @ constraint.laplacian)
            # apply to normal equations
            n = neq + l
            # compute resolution matrix
            r = np.abs(np.diag(np.linalg.solve(n, neq))).reshape((2, -1))
            w = np.array([])
            # do strike and downdip slip separately
            for i in range(2):
                # normalize (1 = the best resolved patch)
                rn = r[i] / np.max(r[i])
                r_range = np.max(rn) - np.min(rn)
                # worst resolved patch (r - np.min(r) = 0) gets the highest smoothing
                # best resolved patch (r - np.min(r) = r_range) gets smallest smoothing
                w = np.concatenate((w, s - (rn - np.min(rn)) * (s - e) / r_range))

            l = (constraint.laplacian.T @ np.diag(w) @ constraint.laplacian)
            n = neq + l
            # solve system
            x = np.linalg.solve(n, ceq)
            # interpolate the residuals using spline
            f = p_ @ x
            f_int = griddata((xg, yg), f, (xs, ys), method='cubic', fill_value=0)
            vg = observations - f_int

        # interpolate the residuals on the grid using spline with tension and add them
        tqdm.write('Interpolating coseismic residuals')
        residual, _ = self._find_tension(xs, ys, xg, yg, f, vg, observations)

        return x, residual, rms

    def _find_tension(self, xs, ys, xg, yg, f, vg, observations):
        i = 0
        rms_tension = np.zeros((99,))
        tensions = np.linspace(0.01, 0.99, 99)

        for tension in tensions:
            grid_response, stations_response = self._compute_spline_responses(
                np.array([xs, ys]), xg, yg, tension
            )
            # compute the interpolation matrix
            ar = self.compute_vertical_grid_interpolant(stations_response, grid_response)
            # compute residuals of this interpolation
            f_aux = griddata((xg, yg), f + ar @ vg, (xs, ys), method='cubic', fill_value=0)
            rms_tension[i] = np.sqrt(np.sum((observations - f_aux) ** 2) / len(observations))
            i += 1

        # choose the lowest rms
        tqdm.write(f'Lowest rms is {rms_tension[np.argmin(rms_tension)] * 1000.: 6.1f} mm '
                   f'tension={tensions[np.argmin(rms_tension)]}')

        grid_response, stations_response = self._compute_spline_responses(
            np.array([xs, ys]), xg, yg, tensions[np.argmin(rms_tension)]
        )
        # compute the interpolation matrix
        ar = self.compute_vertical_grid_interpolant(stations_response, grid_response)
        return ar @ vg, ar


class BaseConstraint(ABC):
    """Base class for all constraint types"""

    def __init__(self, constraint_type: ConstraintType,
                 h_sigma: float = 0.001, v_sigma: float = 0.003):
        self.constraint_type = constraint_type
        self.event: Earthquake = None
        self.h_sigma = h_sigma
        self.v_sigma = v_sigma
        self.equations: List[ConstraintEquation] = []
        self._is_collected = False

    @abstractmethod
    def select_stations(self, all_stations: List[Station],
                        **kwargs) -> Tuple[List[Station], List[Station]]:
        """
        Returns (constraining_stations, stations_to_constrain)
        Constraining stations have data, stations_to_constrain need constraints
        """
        pass

    def collect_constraints(self,
                            all_stations: List[Station],
                            total_parameters: int,
                            grids: GridSystem, **kwargs):
        """Collects constraints for all applicable stations"""

        constraining, to_constrain = self.select_stations(all_stations, **kwargs)

        if len(constraining) > 1:
            for station in tqdm(to_constrain, ncols=120, desc=f'Collecting {self.short_description()}'):
                constraint_params = ConstraintEquation(
                    station=station,
                    constraint_design=self._build_k_matrix(station, constraining, grids, total_parameters),
                    constraint_sigma=np.array([self.h_sigma, self.h_sigma, self.v_sigma])
                )

                self.equations.append(constraint_params)

            # couple parameters of nearby stations (d < 5 km)
            self.constrain_nearby_stations(all_stations, total_parameters)
        else:
            tqdm.write(f'Constraint {self.short_description()} has a single o no constraining stations. '
                        f'Consider removing it to avoid biases in station velocities')

            if len(to_constrain) > 0:
                tqdm.write(f'Adding zero-tie to avoid parameter excursion')
                # add a zero with a lot of weight
                for station in tqdm(to_constrain, ncols=120, desc=f'Collecting {self.short_description()}'):
                    ke = np.zeros((1, total_parameters * 3))
                    kn = np.zeros((1, total_parameters * 3))
                    ku = np.zeros((1, total_parameters * 3))

                    target_idx, _ = self._get_target_cols(station, constraining)

                    ke[0, target_idx] = 1
                    kn[0, target_idx + total_parameters] = 1
                    ku[0, target_idx + total_parameters * 2] = 1

                    constraint_params = ConstraintEquation(
                        station=station,
                        constraint_design=(ke, kn, ku),
                        constraint_sigma=np.array([1e-6, 1e-6, 1e-6])
                    )

                    self.equations.append(constraint_params)

        self._is_collected = True

    def constrain_nearby_stations(self, all_stations: List[Station], total_parameters: int):
        # couple parameters of nearby stations (d < 5 km)
        x, y = np.array([stn.projected_coords for stn in all_stations]).T
        r, _, _ = get_radius(np.column_stack([x, y]),
                             np.column_stack([x, y]))

        # get pairs with distance < 5 km
        i, j = np.where(np.triu(r < 5, k=1))
        # Print the pairs
        for idx_i, idx_j in zip(i, j):

            target_idx, idx = self._get_target_cols(all_stations[idx_i], [all_stations[idx_j]])

            if target_idx is not None and idx[0] is not None:
                tqdm.write(f"Stations {stationID(all_stations[idx_i])} and "
                           f"{stationID(all_stations[idx_j])} are only {r[idx_i, idx_j]:.3f} km from one another: "
                           f"linking their {self.short_description()} parameters")

                ke = np.zeros((1, total_parameters * 3))
                kn = np.zeros((1, total_parameters * 3))
                ku = np.zeros((1, total_parameters * 3))

                # first site
                ke[0, target_idx] = -1
                kn[0, target_idx + total_parameters] = -1
                ku[0, target_idx + total_parameters * 2] = -1
                # second site
                ke[0, idx] = 1
                kn[0, idx + total_parameters] = 1
                ku[0, idx + total_parameters * 2] = 1

                constraint_params = ConstraintEquation(
                    station=all_stations[idx_i],
                    constraint_design=(ke, kn, ku),
                    constraint_sigma=np.array([1e-4, 1e-4, 1e-4])
                )

                self.equations.append(constraint_params)

    def apply_to_normal_equations(self, total_parameters: int) -> np.ndarray:
        """
        Apply all constraints to the normal equation matrix
        Returns: constraint contribution to NEQ
        """
        if not self._is_collected:
            raise ValueError("Constraints must be collected before applying")

        # Count active equations first
        n_active = sum(1 for eq in self.equations if eq.is_active)
        # Pre-allocate
        k = np.zeros((n_active * 3, total_parameters * 3))

        idx = 0
        for equation in self.equations:
            if equation.is_active:
                # Build K matrix for this constraint
                ke, kn, ku = equation.constraint_design
                se, sn, su = equation.constraint_sigma
                # do not square! will get squared when doing k.T @ k
                k[idx:idx+3, :] = np.vstack((ke * (1/se), kn * (1/sn), ku * (1/su)))
                idx += 3

        # return contribution
        return k.T @ k

    def update_weights(self, new_h_sigma: float = None, new_v_sigma: float = None):
        """Update constraint weights (for refinement iterations)"""
        # @todo: implement a method to modify sigmas for single station

        tqdm.write(f'Updating weight for {self.short_description()} '
                   f'current: {self.h_sigma:.6f} {self.v_sigma:.6f} '
                   f'new: {new_h_sigma:.6f} {new_v_sigma:.6f}')

        if new_h_sigma is not None:
            self.h_sigma = new_h_sigma
        if new_v_sigma is not None:
            self.v_sigma = new_v_sigma

        # Update all parameter sigmas
        for param in self.equations:
            param.constraint_sigma = np.array([self.h_sigma, self.h_sigma, self.v_sigma])

    @abstractmethod
    def _get_target_cols(self, station: Station, constraining: List[Station]):
        """
        abstract method to obtain which columns of the normal equations
        belong to each parameter (so it is constraint dependent)
        """
        pass

    def compute_constraint_coefficients(self, target_station: Station,
                                        constraining_stations: List[Station],
                                        grids: GridSystem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Use Sandwell-Wessel for horizontal, elastic loads for vertical"""

        # Horizontal interpolation (Sandwell-Wessel 2016)
        # Do not use the station projected coordinates which have a projection center
        # at a mean regional location. Use the target station as center
        lat = np.array([stn.lat for stn in constraining_stations])
        lon = np.array([stn.lon for stn in constraining_stations])

        x, y = azimuthal_equidistant(np.array([target_station.lon]),
                                     np.array([target_station.lat]), lon, lat)

        ke, kn = grids.compute_horizontal_interpolant_at_point(0, 0, x, y)

        # Vertical interpolation (elastic loads)
        ku = grids.compute_vertical_interpolant_at_point(
            target_station.vertical_response,
            [stn.vertical_response for stn in constraining_stations]
        )

        return ke, kn, ku

    def _build_k_matrix(self, station: Station,
                        constraining: List[Station],
                        grids: GridSystem,
                        total_parameters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the K matrix for a single constraint"""
        _ke, _kn, _ku = self.compute_constraint_coefficients(
            station, constraining, grids
        )
        """Build K matrix for interseismic constraint"""
        ke = np.zeros((1, total_parameters * 3))
        kn = np.zeros((1, total_parameters * 3))
        ku = np.zeros((1, total_parameters * 3))

        target_idx, idx = self._get_target_cols(station, constraining)

        ke[0, target_idx] = -1
        kn[0, target_idx + total_parameters] = -1
        ku[0, target_idx + total_parameters * 2] = -1
        # at the velocity position of the site, place constraint
        ke[0, np.concatenate((idx, idx + total_parameters))] = _ke
        kn[0, np.concatenate((idx, idx + total_parameters))] = _kn
        # vertical component
        ku[0, idx + total_parameters * 2] = _ku

        return ke, kn, ku

    def short_description(self):
        pass


class InterseismicConstraint(BaseConstraint):
    """Constraints for interseismic velocities"""

    def __init__(self, h_sigma: float = 0.0001, v_sigma: float = 0.0003):
        super().__init__(ConstraintType.INTERSEISMIC, h_sigma, v_sigma)

    def select_stations(self, all_stations: List[Station],
                        **kwargs) -> Tuple[List[Station], List[Station]]:
        """
        Constraining: stations with interseismic component (no early earthquakes)
        To constrain: stations without interseismic component
        """

        constraining = [stn for stn in all_stations if stn.is_interseismic]
        to_constrain = [stn for stn in all_stations if not stn.is_interseismic]

        return constraining, to_constrain

    def _get_target_cols(self, station: Station, constraining: List[Station]):
        # Target station gets -1
        target_idx = station.get_velocity_column()

        # Constraining stations get interpolation weights
        idx = np.array([stn.get_velocity_column() for stn in constraining])

        return target_idx, idx

    def short_description(self):
        return f"InterseismicConstraint()"

    def __str__(self) -> str:
        """String representation for debugging"""
        out_str = [f"eq count: {len(self.equations) * 3}",
                   f"h_sig: {self.h_sigma:.6f}", f"v_sig: {self.v_sigma:.6f}"]

        return '; '.join(out_str)

    def __repr__(self) -> str:
        return f"InterseismicConstraint({str(self)})"


class CoseismicConstraint(BaseConstraint):
    """Constraints for coseismic displacements"""

    def __init__(self, event: Earthquake, stations: List[Station],
                 h_sigma: float = 0.007, v_sigma: float = 0.01,
                 smoothing: float = 1e-9,
                 search_start_smoothing: float = 1e-6,
                 search_stop_smoothing: float = 1e-12):

        super().__init__(ConstraintType.COSEISMIC, h_sigma, v_sigma)

        self.event = event
        self.along_strike = 10. ** (-3.22 + 0.69 * event.magnitude) * 1.2 # from Wells & Coppersmith [km] (inflate 20%)
        self.down_dip = (10. ** (-1.01 + 0.32 * event.magnitude)) * 1.6 # from Wells & Coppersmith [km] (inflate 60%)
        self._smoothing: float = smoothing
        self.search_start_smoothing: float = search_start_smoothing
        self.search_stop_smoothing: float = search_stop_smoothing
        self.start_smoothing: list = [None, None, None]
        self.stop_smoothing: list = [None, None, None]
        self.plane = None # will store which plane to use
        self.dislocation_model = None  # Will store computed dislocation matrices
        self.station_list = None       # Will store the station names as arranged in the dislocation matrices
        # to save the dislocation plane patch array
        self._compute_dislocation_plane_grid(stations)
        self._constraint_coefficients: Dict = {}

    @property
    def smoothing(self) -> float:
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value: float):
        """Setter for smoothing"""
        # trigger recomputation of coefficients
        self._constraint_coefficients = {}
        self.start_smoothing = [None, None, None]
        self.stop_smoothing = [None, None, None]
        self._smoothing = value

    def _compute_dislocation_plane_grid(self, stations):
        sind = lambda x: np.sin(np.deg2rad(x))

        # source dimensions L is horizontal, and W is depth
        L1 = -self.along_strike / 2
        L2 = -L1
        W1 = -self.down_dip / 2
        W2 = -W1

        # total number of possible patches is number of observations (ENU) * 3 (Gómez et al 2023)
        n_patches = int(len(stations) * 2)

        # Each circle occupies a square of side 2r
        # Number of circles: (L/2r) × (W/2r) = N
        # Therefore: L×W / 4r² = N => r = sqrt(L×W / 4*N)
        self.radius = np.sqrt(self.along_strike * self.down_dip / (4 * n_patches))

        tqdm.write(f'Event {self.event.id} has {n_patches} patches with a radius of {self.radius:.1f} km -> '
                   f'AS: {self.along_strike:.1f} km DD: {self.down_dip:.1f} km')

        # using the L1/2 W1/W2 coordinate system, build the points using disk code
        # X is the downdip direction, Y the strike direction
        x = np.array([W1, W1, W2, W2])
        y = np.array([L1, L2, L2, L1])

        grid_dd1, grid_ss1, _, _ = fill_region_with_grid(x - 5, y, self.radius, False)
        grid_dd2, grid_ss2, _, _ = fill_region_with_grid(x, y, self.radius, False)
        grid_dd3, grid_ss3, hx, hy = fill_region_with_grid(x + 5, y, self.radius, False)
        # visualize_disks(x, y, hx, hy, grid_dd3, grid_ss3, self.radius)
        grid_dd = np.concatenate((grid_dd1, grid_dd2, grid_dd3))
        grid_ss = np.concatenate((grid_ss1, grid_ss2, grid_ss3))
        tqdm.write(f'Number on fault plane 1 {grid_dd1.size} plane 2 {grid_dd2.size} plane 3 {grid_dd3.size}')

        ss1 = self._compute_laplacian(grid_dd1, grid_ss1, 2 * self.radius)
        ss2 = self._compute_laplacian(grid_dd2, grid_ss2, 2 * self.radius)
        ss3 = self._compute_laplacian(grid_dd3, grid_ss3, 2 * self.radius)

        s1 = np.hstack((ss1, np.zeros_like(ss2), np.zeros_like(ss3), np.zeros_like(ss1), np.zeros_like(ss2), np.zeros_like(ss3)))
        s2 = np.hstack((np.zeros_like(ss1), ss2, np.zeros_like(ss3), np.zeros_like(ss1), np.zeros_like(ss2), np.zeros_like(ss3)))
        s3 = np.hstack((np.zeros_like(ss1), np.zeros_like(ss2), ss3, np.zeros_like(ss1), np.zeros_like(ss2), np.zeros_like(ss3)))
        d1 = np.hstack((np.zeros_like(ss1), np.zeros_like(ss2), np.zeros_like(ss3), ss1, np.zeros_like(ss2), np.zeros_like(ss3)))
        d2 = np.hstack((np.zeros_like(ss1), np.zeros_like(ss2), np.zeros_like(ss3), np.zeros_like(ss1), ss2, np.zeros_like(ss3)))
        d3 = np.hstack((np.zeros_like(ss1), np.zeros_like(ss2), np.zeros_like(ss3), np.zeros_like(ss1), np.zeros_like(ss2), ss3))

        self.laplacian = np.vstack((s1, s2, s3, d1, d2, d3))

        ddip = [-10, 0, 10]
        grid_dep = []
        for dip in self.event.dip:
            grid_patch_dep = np.array([])
            for i, fault in enumerate([grid_dd1, grid_dd2, grid_dd3]):
                fault_dep = self.event.depth + fault * sind(dip + ddip[i])

                # check that no patches stick out of the ground
                if np.any(np.round(fault_dep) <= 0):
                    # do not let the lowest value be zero depth!
                    fault_dep = fault_dep - (fault_dep.min() - self.radius * sind(dip + ddip[i]) - 1.)

                # concatenate all
                grid_patch_dep = np.concatenate((grid_patch_dep, fault_dep))

            grid_dep.append(grid_patch_dep)

        self.dislocation_grid = (grid_dd, grid_ss, grid_dep)

    def select_stations(self, all_stations: List[Station],
                        **kwargs) -> Tuple[List[Station], List[Station]]:
        """
        All stations with coseismic jump for this event constrain each other
        """
        coseismic_stations = []
        for stn in all_stations:
            jump = stn.etm.jump_manager.get_geophysical_jump(self.event.id)
            if jump and jump.p.jump_type == JumpType.COSEISMIC_JUMP_DECAY:
                # @todo: add logic to only constrain stations with data gaps
                if jump.fit:
                    coseismic_stations.append(stn)
                else:
                    tqdm.write(f'WARNING: station {stationID(stn)} is flagged as affected by {self.event.id} '
                               f'but the etm jump is not activated. This may induce a bias in the model '
                               f'around the area of this station')

        return coseismic_stations, coseismic_stations

    def _get_target_cols(self, station: Station, constraining: List[Station]):

        target_idx = station.get_coseismic_column(self.event.id)
        idx = np.array([stn.get_coseismic_column(self.event.id) for stn in constraining if stn != station])

        return target_idx, idx

    def compute_constraint_coefficients(self, target_station: Station,
                                        constraining_stations: List[Station],
                                        grids: GridSystem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For coseismic, 'weights' are actually dislocation model predictions
        This is computed once for all stations together
        """
        if self.dislocation_model is None:
            # initial call, need to determine which plane to use
            self._determine_dislocation_plane(constraining_stations)
            # get the coseismic mask to avoid computing the response everywhere on the grid
            mask = grids.earthquake_masks[self.event.id][0]

            tqdm.write('Computing earthquake response for the interpolation grid')
            # now that the plane is known, compute the response of the grid
            pe, pn, pu = self.compute_dislocation_model(
                self.plane, grids.interpolation_geographic[0][mask],
                grids.interpolation_geographic[1][mask]
            )
            grids.earthquake_responses[self.event.id] = (pe, pn, pu)

        if stationID(target_station) not in self._constraint_coefficients.keys():
            # Find index of target station
            idx = constraining_stations.index(target_station)
            ae, an, au = self.dislocation_model

            # all other stations
            # a = np.vstack((np.delete(ae, idx, axis=0), np.delete(an, idx, axis=0), np.delete(au, idx, axis=0)))
            # normal equations with smoothing constraint
            # n = a.T @ a + (self.laplacian.T @ self.laplacian) * self.smoothing
            # be = np.linalg.solve(n, np.delete(ae, idx, axis=0).T)
            # bn = np.linalg.solve(n, np.delete(an, idx, axis=0).T)
            # bu = np.linalg.solve(n, np.delete(au, idx, axis=0).T)

            ae_ = np.delete(ae, idx, axis=0)
            an_ = np.delete(an, idx, axis=0)
            au_ = np.delete(au, idx, axis=0)

            l = (self.laplacian.T @ self.laplacian) * self.smoothing

            # find A-dagger -> (A.T @ A + smoothing)^-1 @ A.T (see below)
            be = np.linalg.solve(ae_.T @ ae_ + l, ae_.T)
            bn = np.linalg.solve(an_.T @ an_ + l, an_.T)
            bu = np.linalg.solve(au_.T @ au_ + l, au_.T)

            # A-dagger * L = x -> thus at a point p (prediction needed)
            # Ap * x = Lp -> Ap * (A-dagger * L) = Lp
            # with (Ap * A-dagger) being the weighting coefficients
            # here I am finding the coefficients to weight the "observations" from the
            # other sites by doing Ap * A-dagger
            ke = ae[idx, :] @ be
            kn = an[idx, :] @ bn
            ku = au[idx, :] @ bu

            self._constraint_coefficients[stationID(target_station)] = (ke, kn, ku)
        else:
            ke, kn, ku = self._constraint_coefficients[stationID(target_station)]

        # Return the dislocation prediction for this station
        return ke, kn, ku

    def _determine_dislocation_plane(self, constraining_stations: List[Station]):

        sites_lon = np.array([stn.lon for stn in constraining_stations])
        sites_lat = np.array([stn.lat for stn in constraining_stations])

        # now get the jumps for this event
        jumps = [stn.etm.jump_manager.get_geophysical_jump(self.event.id) for stn in constraining_stations]

        le = np.array([j.p.params[1][0] for j in jumps if j])
        ln = np.array([j.p.params[0][0] for j in jumps if j])
        lu = np.array([j.p.params[2][0] for j in jumps if j])

        # create a filter of jumps with sigma larger than 10 cm to avoid using unstable jumps
        sigmas = [np.array(j.p.sigmas) for j in jumps if j]
        use = [np.sqrt(np.sum(s ** 2, axis=0))[0] < 0.15 for s in sigmas]

        tqdm.write(f'Selected {len([u for u in use if u])} stations to test from a total of {len(use)}')

        sites_nam = [stationID(stn) for i, stn in enumerate(constraining_stations) if use[i]]

        if sites_lat.size >= 2:

            ae_ = [np.array([]), np.array([])]
            an_ = [np.array([]), np.array([])]
            au_ = [np.array([]), np.array([])]
            v = [np.array([]), np.array([])]

            obs = [le, ln, lu]
            l = (self.laplacian.T @ self.laplacian) * self.smoothing

            # test both possible faults
            for i in range(2):
                tqdm.write(f'Determining fault plane for {self.event.id} '
                            f'strike: {self.event.strike[i]} dip: {self.event.dip[i]}')

                ae_[i], an_[i], au_[i] = self.compute_dislocation_model(
                    i, sites_lon, sites_lat
                )

                a = [ae_[i][use, :], an_[i][use, :], au_[i][use, :]]

                for j in range(3):
                    n = a[j].T @ a[j] + l
                    c = a[j].T @ obs[j][use]
                    x = np.linalg.solve(n, c)
                    v[i] = np.concatenate((v[i], obs[j][use] - a[j] @ x))

            tqdm.write('ENU residuals [mm] for each station-plane')
            for i in range(len(sites_nam)):
                v0 = ' '.join([f'{val * 1000.:7.1f}' for val in v[0].reshape((3, len(sites_nam)))[:, i]])
                v1 = ' '.join([f'{val * 1000.:7.1f}' for val in v[1].reshape((3, len(sites_nam)))[:, i]])

                tqdm.write(f' -- {sites_nam[i]} {v0} | {v1}')

            self.plane = np.argmin(iqr(v, axis=1))
            tqdm.write(f'iqr for each fault {iqr(v, axis=1)}: '
                       f'selected strike: {self.event.strike[self.plane]} '
                       f'dip: {self.event.dip[self.plane]}')

            # grab the matrices
            self.dislocation_model = (ae_[self.plane], an_[self.plane], au_[self.plane])
            self.station_list = [stationID(stn) for stn in constraining_stations]

            for i, station in enumerate(constraining_stations):
                station.earthquake_responses[self.event.id] = (
                    ae_[self.plane][i, :], an_[self.plane][i, :], au_[self.plane][i, :]
                )

    def compute_dislocation_model(self, dislocation_plane: int,
                                  points_lon: np.ndarray,
                                  points_lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Okada dislocation model for all stations"""
        # This would call the existing _compute_dislocations method
        # Returns weights/predictions for each station
        sind = lambda x: np.sin(np.deg2rad(x))
        cosd = lambda x: np.cos(np.deg2rad(x))

        grid_dd, grid_ss, grid_dep = self.dislocation_grid

        # patch count
        n = grid_dd.shape[0]

        strike, dip = self.event.strike[dislocation_plane], self.event.dip[dislocation_plane]

        # get the depth component based on which plane was selected
        grid_patch_dep = grid_dep[dislocation_plane]

        # clockwise rotation
        R = np.array([[ cosd(strike), sind(strike)],
                      [-sind(strike), cosd(strike)]])

        # compute the transformed patches coordinates. -R because we want to go from fault system to
        # the geographical system. Compress the dd coordinates to match the dipping of the fault
        t = R @ np.array([grid_dd * cosd(dip), grid_ss])
        tx = t[0, :]
        ty = t[1, :]

        # convert the dd and ss grid to lon lat.
        grid_patch_lon, grid_patch_lat = inverse_azimuthal(self.event.lon, self.event.lat, tx, ty)

        a = build_design_matrix(np.array([90 - grid_patch_lat, grid_patch_lon, grid_patch_dep]).T,
                                np.array([90 - points_lat, points_lon]).T,
                                np.array([np.ones(n) * strike,
                                          np.ones(n) * dip,
                                          np.ones(n) * (self.radius * 1000) ** 2]).T)

        ae, an, au = self._dislocation_parts(a)

        # s = self._compute_coseismic_sw_constraints(lon, lat, stnlon, stnlat, ae, an, ge, gn, n)

        return ae, an, au

    @staticmethod
    def _compute_laplacian(x, y, radius):
        """
        Function laplace creates a Laplacian matrix. x,y are the square fault
        patches coordinates in the fault system. r is the radius of of the patch
        """
        b = np.arange(0, 2 * np.pi, 0.01)
        l = np.zeros((len(x), len(x)))

        for i in range(len(x)):
            px = 1.1 * radius * np.cos(b) + x[i]
            py = 1.1 * radius * np.sin(b) + y[i]

            # Create polygon path and check which points are inside
            polygon = Path(np.column_stack([px, py]))
            points = np.column_stack([x, y])
            in_polygon = polygon.contains_points(points)

            l[i, :] = -in_polygon.astype(int)
            l[i, i] = 4

        return l

    def _compute_coseismic_sw_constraints(self, grids: GridSystem,
                                          epi_lon, epi_lat,
                                          stnlon: np.ndarray, stnlat: np.ndarray,
                                          im_e: np.ndarray, im_n: np.ndarray,
                                          im_ie: np.ndarray, im_in: np.ndarray,
                                          patch_count: int):

        # project epicenter
        ex, ey = azimuthal_equidistant(np.array(grids.origin[0]), np.array(grids.origin[1]),
                                       np.array([epi_lon]), np.array([epi_lat]))

        x, y = azimuthal_equidistant(np.array(grids.origin[0]), np.array(grids.origin[1]),
                                     stnlon, stnlat)

        s_score, _, _ = grids.earthquake_masks[self.event.id]

        # compute the weights to form W (ep X in Gomez et al 2026)
        ke, kn = grids.compute_horizontal_grid_interpolant(x, y, s_score)

        # select the points to be constrained based on distance to epicenter
        epi_dist = np.hypot(grids.interpolation_grid[0] - ex[:, np.newaxis],
                            grids.interpolation_grid[1] - ey[:, np.newaxis])

        sorted_indices = np.argsort(epi_dist, axis=None)

        # determine how many constraints we need
        ce = epi_dist.shape[1] #(2 * patch_count - 3 * stnlon.shape[0]) // 2 + 1

        constraints = np.zeros((0, patch_count * 2))
        im = np.vstack((im_e, im_n))
        # Now iterate in ascending order of epi_dist values
        for i in range(ce):
            idx = sorted_indices[i]
            # build the constraint vector, grab the ke and kn[idx] for the interpolation point which form w
            # constraint is (IM_i - w∙IM)∙s = 0
            w = np.vstack((ke[idx, :], kn[idx, :]))
            im_i = np.vstack((im_ie[idx, :], im_in[idx, :]))

            constraints = np.vstack((constraints, im_i - w @ im))

        return constraints

    @staticmethod
    def _dislocation_parts(a):
        rake = np.array([0, 90])  # Rake angles in degrees

        # Convert to radians
        rake_rad = np.deg2rad(rake)

        # Using array slicing with step sizes
        # Extract every 3rd row starting from different offsets for z, y, x components
        # Extract every 2nd column for the two slip components

        # For z-component (rows 0, 3, 6, ...)
        Gz_col1 = a[0::3, 0::2]  # Shape: (m, n)
        Gz_col2 = a[0::3, 1::2]  # Shape: (m, n)

        # For y-component (rows 1, 4, 7, ...)
        Gy_col1 = a[1::3, 0::2]  # Shape: (m, n)
        Gy_col2 = a[1::3, 1::2]  # Shape: (m, n)

        # For x-component (rows 2, 5, 8, ...)
        Gx_col1 = a[2::3, 0::2]  # Shape: (m, n)
        Gx_col2 = a[2::3, 1::2]  # Shape: (m, n)

        # Compute for rake[0]
        uz_ss = np.cos(rake_rad[0]) * Gz_col1 + np.sin(rake_rad[0]) * Gz_col2 # up
        uy_ss = np.cos(rake_rad[0]) * Gy_col1 + np.sin(rake_rad[0]) * Gy_col2 # east
        ux_ss = np.cos(rake_rad[0]) * Gx_col1 + np.sin(rake_rad[0]) * Gx_col2 # north

        # Compute for rake[1]
        uz_dd = np.cos(rake_rad[1]) * Gz_col1 + np.sin(rake_rad[1]) * Gz_col2 # up
        uy_dd = np.cos(rake_rad[1]) * Gy_col1 + np.sin(rake_rad[1]) * Gy_col2 # east
        ux_dd = np.cos(rake_rad[1]) * Gx_col1 + np.sin(rake_rad[1]) * Gx_col2 # north

        return np.hstack((ux_ss, ux_dd)), np.hstack((uy_ss, uy_dd)), np.hstack((uz_ss, uz_dd))

    def _build_k_matrix(self, station: Station,
                        constraining: List[Station],
                        grids: GridSystem,
                        total_parameters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the K matrix for a single constraint"""
        _ke, _kn, _ku = self.compute_constraint_coefficients(
            station, constraining, grids
        )
        """Build K matrix for interseismic constraint"""
        ke = np.zeros((1, total_parameters * 3))
        kn = np.zeros((1, total_parameters * 3))
        ku = np.zeros((1, total_parameters * 3))

        target_idx, idx = self._get_target_cols(station, constraining)

        ke[0, target_idx] = -1
        kn[0, target_idx + total_parameters] = -1
        ku[0, target_idx + total_parameters * 2] = -1
        # at the velocity position of the site, place constraint
        ke[0, idx] = _ke
        kn[0, idx + total_parameters] = _kn
        # vertical component
        ku[0, idx + total_parameters * 2] = _ku

        return ke, kn, ku

    def short_description(self):
        return f"CoseismicConstraint({self.event.id})"

    def __str__(self) -> str:
        """String representation for debugging"""
        out_str = [f"{self.event.id}", f"smoothing: {self.smoothing:.2e}",
                   f"search: {self.search_start_smoothing:.2e} to {self.search_stop_smoothing:.2e}",
                   f"plane: {self.plane}",
                   f"equation count: {len(self.equations) * 3}",
                   f"h_sigma: {self.h_sigma:.6f}", f"v_sigma: {self.v_sigma:.6f}"]

        return '; '.join(out_str)

    def __repr__(self) -> str:
        return f"CoseismicConstraint({str(self)})"


class PostseismicConstraint(BaseConstraint):
    """Constraints for postseismic relaxation"""

    def __init__(self, event: Earthquake, relaxation: float,
                 h_sigma: float = 0.001, v_sigma: float = 0.003):
        super().__init__(ConstraintType.POSTSEISMIC, h_sigma, v_sigma)
        self.event = event
        self.relaxation = relaxation

    def select_stations(self, all_stations: List[Station],
                        **kwargs) -> Tuple[List[Station], List[Station]]:
        """
        Constraining: stations with data and this relaxation
        To constrain: stations with relaxation but insufficient data
        """

        constraining = []
        to_constrain = []
        for stn in all_stations:
            jump = stn.etm.jump_manager.get_geophysical_jump(self.event.id)

            if (jump and jump.p.jump_type != JumpType.COSEISMIC_ONLY and
                    jump.get_relaxation_cols(self.relaxation)):

                dates = np.array([date.mjd for date in stn.etm.solution_data.coordinates.dates])

                if np.min(dates[dates >= jump.date.mjd] - jump.date.mjd) <= MISSING_DAYS_TOLERANCE:
                    constraining.append(stn)
                else:
                    to_constrain.append(stn)


        return constraining, to_constrain

    def _get_target_cols(self, station: Station, constraining: List[Station]):

        target_idx = station.get_postseismic_column(self.event.id, self.relaxation)
        idx = np.array([stn.get_postseismic_column(self.event.id, self.relaxation)
                        for stn in constraining]).flatten()

        return target_idx, idx

    def short_description(self):
        return f"PostseismicConstraint({self.event.id} {self.relaxation:.3f})"

    def __str__(self) -> str:
        """String representation for debugging"""
        out_str = [f"{self.event.id}", f"relax {self.relaxation:.3f}",
                   f"equation count: {len(self.equations) * 3}",
                   f"h_sigma: {self.h_sigma:.6f}", f"v_sigma: {self.v_sigma:.6f}"]

        return '; '.join(out_str)

    def __repr__(self) -> str:
        return f"PostseismicConstraint({str(self)})"


class ConstraintRegistry:
    """Manages all constraints for the stacking problem"""

    def __init__(self):
        self.constraints: Dict[str, List[Union[
            CoseismicConstraint,
            PostseismicConstraint,
            InterseismicConstraint,
            BaseConstraint
        ]]] = {
            'interseismic': [],
            'coseismic': [],
            'postseismic': []
        }
        self._application_order = ['interseismic', 'coseismic', 'postseismic']

    def add_constraint(self, constraint: BaseConstraint):
        """Add a constraint to the registry"""
        key = constraint.constraint_type.value
        self.constraints[key].append(constraint)

    def collect_all_constraints(self, stations: List[Station],
                              total_parameters: int, grids: GridSystem, **kwargs):
        """Collect all registered constraints"""
        for constraint_type in self._application_order:
            tqdm.write(f"Collecting {constraint_type} constraints")
            for constraint in self.constraints[constraint_type]:
                # reset equations before collecting
                constraint.equations = []
                constraint.collect_constraints(stations, total_parameters, grids, **kwargs)

    def add_all_constraints(self, neq: np.ndarray,
                              total_parameters: int) -> int:
        """Add all constraints to normal equations"""
        total_constraints = 0
        for constraint_type in self._application_order:
            for constraint in self.constraints[constraint_type]:
                tqdm.write(f'Adding {repr(constraint)} to the system')
                neq += constraint.apply_to_normal_equations(total_parameters)
                # count how many constraints per equation
                total_constraints += sum(3 for const in constraint.equations if const.is_active)

        return total_constraints

    def get_constraint_summary(self) -> Dict:
        """Get summary of all constraints for reporting"""
        summary = {}
        for ctype, constraints in self.constraints.items():
            summary[ctype] = {
                'count': len(constraints),
                'total_equations': sum(len(c.equations) for c in constraints),
                'active_equations': sum(
                    sum(p.is_active for p in c.equations)
                    for c in constraints
                )
            }
        return summary


class EtmStacker:
    """Simplified main class focusing on orchestration"""

    def __init__(self, config: EtmStackerConfig = None):

        # Core data
        self.stations: List[Station] = []
        self.normal_equations: List[NormalEquations] = []
        self.earthquakes: List[Earthquake] = []

        # Constraint management
        self.constraint_registry: ConstraintRegistry = ConstraintRegistry()

        # Grid system
        self.grids: GridSystem = GridSystem((0, 0))

        # Configuration
        if config is None:
            self.config = EtmStackerConfig()
        else:
            self.config = config

        # System normal equations
        self.total_parameters: int = 0
        self.total_equations: int = 0
        self.total_constraints: int = 0
        self.variance: float = 0.

        # Results
        self.solved: bool = False
        self.solution: np.ndarray = np.array([])
        self.covariance: np.ndarray = np.array([])

        # interpolated fields
        self.fields: List[EtmStackerField] = []

        # to save the command history applied to the stacker instance
        self.command_history: List[str] = []
        # to store the name of the current pickle (without extension)
        self.filename: str = ''

        self.print_config()

    def print_config(self):
        sr = ','.join(['%.3f' % r for r in self.config.relaxation])
        cp = ','.join(['%s' % r for r in self.config.earthquakes_cherry_picked])
        tqdm.write(f' -- Initialized EtmStacker with max cond number: {self.config.max_condition_number}; '
                   f'relaxations: {sr}')
        tqdm.write(f' -- Earthquake mag limit: {self.config.earthquake_magnitude_limit}; '
                   f'Cherry picked earthquakes: {cp};')
        if isinstance(self.config.post_seismic_back_lim, Date):
            tqdm.write(f' -- Considering events starting from {self.config.post_seismic_back_lim.yyyyddd()}')
        else:
            tqdm.write(f' -- Considering events up to {self.config.post_seismic_back_lim / 365} '
                       f'years back from station start')

        tqdm.write(f" -- Interseismic sigmas: {self.config.interseismic_h_sigma * 1000.} mm/yr "
                   f"{self.config.interseismic_v_sigma * 1000.} mm/yr")
        tqdm.write(f" -- Coseismic sigmas: {self.config.coseismic_h_sigma * 1000.} mm "
                   f"{self.config.coseismic_v_sigma * 1000.} mm")
        tqdm.write(f" -- Postseismic sigmas: {self.config.postseismic_h_sigma * 1000.} mm "
                   f"{self.config.postseismic_v_sigma * 1000.} mm")
        tqdm.write(f" -- Station weight scale factor: {self.config.station_weight_scale}")
        tqdm.write(f" -- Vertical interpolation method: {self.config.vertical_method}")
        if self.config.vertical_method != 'spline2d':
            tqdm.write(f" -- Vertical load radius (for diskload or rectload): {self.config.vertical_load_radius} km")
        else:
            tqdm.write(f" -- Spline2d tension: {self.config.tension}")

        tqdm.write(f" -- ETM stacker model filename: {self.filename}")

    def add_station(self, cnn: Cnn, network_code: str, station_code: str,
                    json_folder: str = None,
                    save_json_folder: str = None):
        """Add a station to the stack"""
        # Build ETM
        etm = self._build_etm(cnn, network_code, station_code, json_folder, save_json_folder)
        if etm is None:
            return

        # Create station
        station = self._create_station(etm)

        # Create normal equations
        neq = self._create_normal_equations(station)

        # Store
        self.stations.append(station)
        self.normal_equations.append(neq)

    def remove_station(self, station_id: str):
        remove_from_index = 0
        for i, station in enumerate(self.stations):
            # remove from the parameter range the station that has been removed
            # this is applied to any stations that come after the removed site
            station.normal_equations.parameter_range -= remove_from_index
            station.normal_equations.parameter_start_idx -= remove_from_index
            if stationID(station) == station_id:
                self.total_parameters -= station.normal_equations.parameter_count
                self.total_equations -= station.normal_equations.equation_count
                self.normal_equations.pop(i)
                self.stations.pop(i)
                remove_from_index = station.normal_equations.parameter_count

        # no need to rebuild the registry or the grids
        # they will contain a couple dead sites but they don't affect the calculations
        self.solved = False

    def _build_etm(self, cnn: Cnn, network_code: str, station_code: str,
                   json_folder: str = None, save_json_folder: str = None):

        etm = None
        loaded_from_json = False

        if json_folder is not None:
            if os.path.isfile(os.path.join(json_folder, f'{network_code}.{station_code}_ppp.json')):
                tqdm.write(f'Loading etm for {network_code}.{station_code} from json file')
                config = EtmConfig(json_file=os.path.join(json_folder, f'{network_code}.{station_code}_ppp.json'))
                # remove any prefit models from the json (should be applied when we did the model in the first place)
                etm = EtmEngine(config)
                loaded_from_json = True
            else:
                tqdm.write(f'Could not find etm json for {network_code}.{station_code}, '
                           f'will try to use the database')

        try:
            if etm is None:
                tqdm.write(f'Estimating etm for {network_code}.{station_code}')
                config = EtmConfig(network_code, station_code, cnn=cnn)
                config.solution.solution_type = SolutionType.PPP

                config = self._apply_config(config, cnn)

                etm = EtmEngine(config, cnn=cnn, silent=True)

            #if etm.solution_data.solutions < 100:
            #    tqdm.write(print_yellow(f'Station {network_code}.{station_code} has less than '
            #                             f'100 solutions, skipping'))
            #    return None

            if etm.solution_data.time_vector[-1] - etm.solution_data.time_vector[0] <= 1.5:
                tqdm.write(print_yellow(f' -- Station {network_code}.{station_code} has less than 1.5 '
                                         f'years of data, skipping'))
                return None


            etm.run_adjustment(cnn=cnn)
        except (DesignMatrixException, numpy.linalg.linalg.LinAlgError):
            tqdm.write(print_yellow(f' -- Unable to fit {network_code}.{station_code} -> system is rank deficient. '
                                    f'Will redo ETM with only 10 years of postseismic events.'))
            # default back to max condition number = 3
            config.validation.max_condition_number = 3
            etm = EtmEngine(config, cnn=cnn, silent=True)
            try:
                etm.run_adjustment(cnn=cnn)
            except Exception:
                tqdm.write(print_yellow(f' -- Unable to fit {network_code}.{station_code}. '
                                        f'Station will not be added.'))
                return None

        except SolutionDataException as e:
            tqdm.write(print_yellow(str(e)))
            return None

        if etm.config.modeling.status == etm.config.modeling.status.UNABLE_TO_FIT:
            tqdm.write(print_yellow(f' -- Unable to fit station {network_code}.{station_code} (rank deficient?)'))
            return None

        if np.any([np.isnan(r.parameters) for r in etm.fit.results]):
            tqdm.write(print_yellow(f' -- Station {network_code}.{station_code} combined with the list of earthquakes '
                                    f'yielded a singular solution, station cannot be used'))
            return None

        # gather any mechanical jumps to remove
        mechanical = etm.jump_manager.get_active_mechanical_jumps()
        if len(mechanical):
            if not self._correct_mechanical_jumps(network_code, station_code, etm, mechanical, cnn):
                return None

        if save_json_folder is not None and not loaded_from_json:
            if not os.path.exists(save_json_folder):
                os.makedirs(save_json_folder)
            # let the etm build the filename of the station
            etm.save_etm(save_json_folder + '/', dump_functions=True, dump_observations=True,
                         dump_raw_results=True, dump_design_matrix=True, dump_model=True)

        return etm

    def _apply_config(self, config: EtmConfig, cnn: Cnn):
        config.validation.max_condition_number = self.config.max_condition_number
        config.modeling.check_jump_collisions = False # turn off jump collision check. Add all jumps.
        config.modeling.earthquake_magnitude_limit = self.config.earthquake_magnitude_limit
        config.modeling.post_seismic_back_lim = self.config.post_seismic_back_lim
        config.modeling.relaxation = self.config.relaxation
        config.modeling.earthquakes_cherry_picked = self.config.earthquakes_cherry_picked
        # @todo: change the parameters for minimum number of day between earthquakes
        config.refresh_config(cnn)
        return config

    @staticmethod
    def _create_station(etm: EtmEngine):

        station = Station(
            etm.config.network_code,
            etm.config.station_code,
            etm.config.metadata.lon[0],
            etm.config.metadata.lat[0],
            etm.solution_data.coordinates.dates[0],
            etm
        )

        # figure out if station can participate on interseismic model
        jump = etm.jump_manager.get_first_geophysical()
        if jump is None or (jump is not None and jump.p.jump_date > station.first_obs):
            station.is_interseismic = True
            tqdm.write(f' -- Station {stationID(station)} is interseismic')

        return station

    def _correct_mechanical_jumps(self, network_code: str, station_code: str,
                                  station_etm: EtmEngine,
                                  mechanical: List[JumpFunction],
                                  cnn: Cnn):

        # need to rerun the model but without letting it be unconstrained
        # create a deep copy of the etm
        config = EtmConfig(network_code, station_code, cnn=cnn)
        config.modeling.relaxation = np.array([np.max(self.config.relaxation)])

        if not os.path.exists('./production'):
            os.makedirs('./production')

        etm = EtmEngine(config, cnn=cnn)
        etm.run_adjustment(try_loading_db=False, force_computation=True, try_save_to_db=False)
        etm.config.plotting_config.filename = f'./production/{network_code}.{station_code}_before_correction'
        etm.plot()

        prefit: List[JumpFunction] = []
        for jump in etm.jump_manager.get_active_mechanical_jumps():
            prefit.append(jump)

        # assign prefit models to remove
        station_etm.config.modeling.status = FitStatus.PREFIT
        station_etm.config.modeling.prefit_models = copy.deepcopy(prefit)
        # deactivate these jumps from the design matrix
        for j in mechanical:
            j.fit = False
        # rerun adjustment without the mechanical jumps
        try:
            station_etm.run_adjustment(try_loading_db=False, force_computation=True, try_save_to_db=False)
            tqdm.write(f' -- Found and corrected {len(mechanical)} mechanical jumps in {network_code}.{station_code}')
            station_etm.config.plotting_config.filename = f'./production/{network_code}.{station_code}_corrected'
            station_etm.plot()
        except (DesignMatrixException, numpy.linalg.linalg.LinAlgError):
            tqdm.write(print_yellow(f' -- Unable to fit {network_code}.{station_code} -> system is rank deficient.'))

            # not working, check why  Traceback (most recent call last):
            #File "/home/demian/miniconda3/envs/pgamit/bin/EtmStacker.py", line 700, in <module>
            #  main()
            #File "/home/demian/miniconda3/envs/pgamit/bin/EtmStacker.py", line 672, in main
            #  etm_stacker.add_station(cnn, stn['NetworkCode'], stn['StationCode'],
            #File "/home/demian/Dropbox/OSU/Projects/Parallel.GAMIT/Parallel.GAMIT/geode/etm/core/etm_stacker.py", line 2226, in add_station
            #  etm = self._build_etm(cnn, network_code, station_code, json_folder, save_json_folder)
            #File "/home/demian/Dropbox/OSU/Projects/Parallel.GAMIT/Parallel.GAMIT/geode/etm/core/etm_stacker.py", line 2333, in _build_etm
            #  etm.save_etm(save_json_folder + '/', dump_functions=True, dump_observations=True,
            #File "/home/demian/Dropbox/OSU/Projects/Parallel.GAMIT/Parallel.GAMIT/geode/etm/core/etm_engine.py", line 302, in save_etm
            #  model_values = self.fit.get_time_continuous_model(self.solution_data.time_vector_cont)
            #File "/home/demian/Dropbox/OSU/Projects/Parallel.GAMIT/Parallel.GAMIT/geode/etm/least_squares/least_squares.py", line 771, in get_time_continuous_model
            #  model[i] = self.design_matrix.alternate_time_vector(time_vector_cont) @ self.results[i].parameters
            #ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 29 is different from 28)

            #station_etm.config.validation.max_condition_number = 3
            #station_etm = EtmEngine(station_etm.config, cnn=cnn, silent=True)
            # deactivate mechanical jumps again
            #mechanical = station_etm.jump_manager.get_active_mechanical_jumps()
            #for j in mechanical:
            #    j.fit = False

            #try:
            #    station_etm.run_adjustment(try_loading_db=False, force_computation=True, try_save_to_db=False)
            #except Exception:
            #    tqdm.write(print_yellow(f' -- Unable to fit {network_code}.{station_code}. '
            #                            f'Station will not be added.'))
            return False

        return True

    def _create_normal_equations(self, station: Station):
        """analyze and save the relevant information"""

        # get the observations without the jumps
        l = station.etm.solution_data.transform_to_local()
        a = station.etm.design_matrix.matrix

        n = []
        c = []

        if station.etm.solution_data.solutions < 100:
            tqdm.write(f' -- Upweighting {stationID(station)} because observations count is < 100')
            weight_scale = self.config.station_weight_scale * 100.
        else:
            weight_scale = self.config.station_weight_scale

        lpl = []
        observation_weights = []
        prior_wrms = []
        # rearrange the NEU to ENU
        for i in [1, 0, 2]:
            p = np.diag(1 / station.etm.fit.results[i].obs_sigmas ** 2) * weight_scale
            n.append(a.T @ p @ a)
            c.append(a.T @ p @ l[i])
            lpl.append(l[i].T @ p @ l[i])
            observation_weights.append(1 / station.etm.fit.results[i].obs_sigmas ** 2)
            prior_wrms.append(station.etm.fit.results[i].wrms)

        neq = NormalEquations(
            station=stationID(station),
            neq=n, ceq=c,
            design_matrix=a,
            observation_vector=[l[1], l[0], l[2]],
            weighted_observations=lpl,
            observation_weights=observation_weights,
            weight_scale=weight_scale, dof=a.shape[0] - a.shape[1],
            parameter_count=a.shape[1],
            equation_count=a.shape[0],
            parameter_start_idx=self.total_parameters,
            parameter_range=np.arange(a.shape[1]) + self.total_parameters,
            prior_wrms=prior_wrms
        )
        # save both vectors to the station
        station.normal_equations = neq

        self.total_parameters += a.shape[1]
        self.total_equations += a.shape[0]

        return neq

    def build_system(self):
        """Build the complete stacking system"""
        # 1. Create grids
        self.grids = GridSystem.create_from_stations(self.stations,
                                                     grid_spacing=self.config.grid_spacing,
                                                     grid_load_radius=self.config.vertical_load_radius,
                                                     method=self.config.vertical_method,
                                                     tension=self.config.tension)

        # 3. Register all constraints
        self._register_constraints()

    def change_station_weight(self, station_id: str, new_weight: float, silent=False):

        found = False
        for neq in self.normal_equations:
            if neq.station == station_id:
                if not silent:
                    tqdm.write(f'Found {station_id} with weight {neq.weight_scale}, '
                               f'updating to {new_weight}')
                for i in range(3):
                    neq.ceq[i] = neq.ceq[i] / neq.weight_scale * new_weight
                    neq.neq[i] = neq.neq[i] / neq.weight_scale * new_weight
                    neq.weighted_observations[i] = neq.weighted_observations[i] / neq.weight_scale * new_weight

                neq.weight_scale = new_weight

                found = True

        if found and not silent:
            tqdm.write('Do not forget to invoke solve again!')

    def solve(self, interpolate_fields=True) -> Tuple[List, List]:
        """
        Solve the stacking system
        """
        # rebuild normal equations before solving
        system_neq, system_ceq = self._build_base_normal_equations()

        # collect constraints. Changes to smoothing and weight will be applied here
        self.constraint_registry.collect_all_constraints(
            self.stations, self.total_parameters, self.grids,
            earthquakes=self.earthquakes
        )

        # Solve: Apply constraints to system and do not modify original NEQs
        self.total_constraints = self.constraint_registry.add_all_constraints(
            system_neq, self.total_parameters
        )

        tqdm.write('Solving system...')

        x = np.linalg.solve(system_neq, system_ceq)
        self.solution = np.reshape(x, (3, self.total_parameters))
        # compute covariance for the entire system
        self.covariance = np.linalg.inv(system_neq)

        # compute the variance of unit weight
        lpl = sum(stn.normal_equations.weighted_observations[0] +
                  stn.normal_equations.weighted_observations[1] +
                  stn.normal_equations.weighted_observations[2] for stn in self.stations)

        dof = self.total_equations * 3 + self.total_constraints - (self.total_parameters * 3)
        c_vpv = self._sum_constraint_weighted_residuals()
        o_vpv = lpl - system_ceq.T @ x

        # see Kyle Snow eq 6.36 and 6.37a (y^T P y − c^T N^−1 c)
        self.variance = (o_vpv + c_vpv) / dof
        # update the covariance
        self.covariance *= self.variance
        # compute variance of unit weight for each stations
        increment = []
        for stn in self.stations:
            # add constraints to each station
            stn.extract_etm_constraints(self.earthquakes, self.config.relaxation,
                                        self.solution, self.covariance)
            # access normal equations
            neq = stn.normal_equations
            stn.posterior_wrms = []
            wrms_increment = []
            for i in range(3):
                # compute residuals for station
                x = self.solution[i, stn.normal_equations.parameter_range]
                v = neq.observation_vector[i] - neq.design_matrix @ x
                wrms_increment.append(np.sqrt(
                    v.T @ np.diag(neq.observation_weights[i]) @ v / stn.normal_equations.dof)
                )
                stn.posterior_wrms.append(
                    neq.prior_wrms[i] * wrms_increment[i]
                )
            wrms_increment = np.array(wrms_increment)

            increment.append([stationID(stn), np.mean(wrms_increment), wrms_increment])

        from operator import itemgetter
        tqdm.write('WRMS increment for each station:')
        for stn, wrmsi, wrms in increment:
            tqdm.write(f'{stn} WRMS increment: (total={wrmsi:.2f}) {wrms[0]:.2f} {wrms[1]:.2f} {wrms[2]:.2f}')

        tqdm.write('First five largest WRMS increments:')
        c = 0
        for stn, wrmsi, wrms in sorted(increment, key=itemgetter(1), reverse=True):
            if c == 6:
                break
            tqdm.write(f'{stn} WRMS increment: (total={wrmsi:.2f}) {wrms[0]:.2f} {wrms[1]:.2f} {wrms[2]:.2f}')
            c += 1

        tqdm.write(f'Equations: {self.total_equations * 3}')
        tqdm.write(f'Constraints: {self.total_constraints}')
        tqdm.write(f'Parameters: {self.total_parameters * 3}')
        tqdm.write(f'Sum of squared residuals (obs): {o_vpv:.3f}')
        tqdm.write(f'Sum of squared residuals (con): {c_vpv:.3f}')
        tqdm.write(f'Model redundancy: {dof}')
        tqdm.write(f'SQRT(var) for the stacked system: {np.sqrt(self.variance):.3f} ')
        tqdm.write(f'1/var for the stacked system: {1/self.variance:.4f} ')

        self.solved = True

        if interpolate_fields:
            self.interpolate_fields_to_grid()

        # Extract results
        return self._extract_results()

    def _sum_constraint_weighted_residuals(self):

        # Count active equations first
        n_active = self.total_constraints
        # Pre-allocate
        k = np.zeros((n_active, self.total_parameters * 3))
        idx = 0

        for constraint_type in self.constraint_registry.constraints.keys():
            for const in self.constraint_registry.constraints[constraint_type]:
                for eq in [e for e in const.equations if e.is_active]:
                    # Build K matrix for this constraint
                    ke, kn, ku = eq.constraint_design
                    se, sn, su = eq.constraint_sigma
                    # do not square! will get squared when doing v.T @ v
                    k[idx:idx + 3, :] = np.vstack((ke * (1 / se), kn * (1 / sn), ku * (1 / su)))
                    idx += 3
        # the - comes from z0 − K ξ̂ but in this case, z0 = 0 (see Snow 6.38)
        v = -k @ self.solution.flatten()

        return v.T @ v

    def constraints_rms(self):
        """
        take the registered constraints and find their rms values
        """
        from operator import itemgetter

        wrms = []
        for constraint_type in self.constraint_registry.constraints.keys():
            for const in self.constraint_registry.constraints[constraint_type]:
                v = np.array([])
                v_eq = []
                for eq in [e for e in const.equations if e.is_active]:
                    # get design and weight matrix
                    ke, kn, ku = eq.constraint_design
                    # do not square! will get squared when doing v.T @ v
                    r = np.vstack((ke, kn, ku)) @ self.solution.flatten()
                    v_eq.append([stationID(eq.station), r, np.sqrt((r.T @ r) / 2)])
                    v = np.concatenate((v, r))

                n = len(const.equations)
                # compute the wrms residuals
                if n > 0:
                    wrms.append([const, n, np.sqrt((v.T @ v) / (n - 1)),
                                 sorted(v_eq, key=itemgetter(2), reverse=True)])

        return sorted(wrms, key=itemgetter(2), reverse=True)

    def _register_constraints(self):
        """Register all constraint types"""
        # Interseismic
        self.constraint_registry.add_constraint(
            InterseismicConstraint(
                self.config.interseismic_h_sigma,
                self.config.interseismic_v_sigma
            )
        )

        # record all earthquakes that might affect the ETMs
        self._record_earthquakes()

        # Coseismic and postseismic for each earthquake
        for event in self.earthquakes:
            # Coseismic
            # @todo: do not add to registry directly. Check that there are constraining stations before adding
            stations = [stn for stn in self.stations if stn.get_coseismic_column(event.id) is not None]
            if len(stations):
                self.constraint_registry.add_constraint(
                    CoseismicConstraint(
                        event, stations,
                        self.config.coseismic_h_sigma,
                        self.config.coseismic_v_sigma
                    )
                )
            else:
                tqdm.write(f'No stations observed coseismic event {event.id}. '
                           f'A coseismic constraint for this event will not be added.')

            # Postseismic for each relaxation
            for relax in self.config.relaxation:
                self.constraint_registry.add_constraint(
                    PostseismicConstraint(
                        event, relax,
                        self.config.postseismic_h_sigma,
                        self.config.postseismic_v_sigma
                    )
                )

    def _record_earthquakes(self):

        for stn in self.stations:
            for jump in [jump for jump in stn.etm.jump_manager.jumps
                         if jump.is_geophysical() and jump.fit]:

                if jump.earthquake is None:
                    tqdm.write(f'Could not identify earthquake ID for station '
                                f'{stationID(stn)} for jump date {jump.date}')
                    continue

                if jump.earthquake not in self.earthquakes:
                    tqdm.write('Recording event ' + repr(jump))
                    self.earthquakes.append(jump.earthquake)
                    self.earthquakes.sort()

                    # open connection to database
                    cnn = Cnn('gnss_data.cfg')
                    lon, lat = self.grids.interpolation_geographic

                    # save a mask for the event
                    mask = Mask(cnn, jump.earthquake.id)
                    s_score, p_score = mask.score(lat, lon)

                    tqdm.write(f'Getting mask for event {jump.earthquake.id}')
                    s_score = s_score > 0
                    p_score = p_score > 0
                    # save the actual object to query it
                    self.grids.earthquake_masks[jump.earthquake.id] = (s_score, p_score, mask)

    def _build_base_normal_equations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the base NEQ from individual stations"""

        tqdm.write('Building station system of normal equations')

        tp = self.total_parameters

        system_neq = np.zeros((tp * 3, tp * 3))
        system_ceq = np.zeros((tp * 3,))

        offset = 0
        for neq in self.normal_equations:
            n_params = neq.parameter_count

            for i in range(3):
                neq_comp = neq.neq[i]
                ceq_comp = neq.ceq[i]

                system_neq[
                    i * tp + offset:i * tp + offset + n_params,
                    i * tp + offset:i * tp + offset + n_params] = neq_comp

                system_ceq[
                    i * tp + offset:i * tp + offset + n_params] = ceq_comp

            offset += n_params

        self.solved = False

        return system_neq, system_ceq

    def add_earthquake(self, event: Earthquake, json_folder: str = None, save_json_folder: str = None):
        """Add event to the list of modeled earthquakes"""

        # check the event is not already in the list
        if event in self.earthquakes:
            tqdm.write(f'Event {event.id} is already in the list of modeled events')
            return

        tqdm.write('Adding event ' + str(event))
        self.earthquakes.append(event)
        self.config.earthquakes_cherry_picked.append(f'{event.id}')
        self.earthquakes.sort()

        # get the mask
        # open connection to database
        cnn = Cnn('gnss_data.cfg')
        lon, lat = self.grids.interpolation_geographic

        # save a mask for the event
        mask = Mask(cnn, event.id)
        s_score, p_score = mask.score(lat, lon)

        tqdm.write(f'Getting mask for event {event.id}')
        s_score = s_score > 0
        p_score = p_score > 0
        # save the actual object to query it
        self.grids.earthquake_masks[event.id] = (s_score, p_score, mask)

        # figure out which stations need to be recomputed
        for i, stn in enumerate(self.stations):
            s_score, p_score = mask.score(stn.lat, stn.lon)
            if p_score > 0 or s_score > 0:
                tqdm.write(f'Recomputing etm for {stationID(stn)}')
                # replace old etm
                stn.etm = self._build_etm(cnn, stn.network_code, stn.station_code, json_folder, save_json_folder)
                # get dimensions of neq
                new_par_count = stn.etm.design_matrix.matrix.shape[1]
                # assume number of unknowns will change, so update the rest of the stations down from current
                remove_from_index = 0
                # will be added when calling _create_normal_equations
                self.total_parameters = 0
                self.total_equations = 0

                for station in self.stations:
                    if stationID(station) == stationID(stn):
                        # add the number of new parameters to remove_from_index
                        remove_from_index = station.normal_equations.parameter_count - new_par_count
                        # create a new normal equations object and replace current
                        self.normal_equations[i] = self._create_normal_equations(stn)
                    else:
                        # remove from the parameter range the station that has been removed
                        # this is applied to any stations that come after the removed site
                        station.normal_equations.parameter_range -= remove_from_index
                        station.normal_equations.parameter_start_idx -= remove_from_index
                        self.total_parameters += station.normal_equations.parameter_count
                        self.total_equations += station.normal_equations.equation_count

        # now add the constraint
        stations = [stn for stn in self.stations if stn.get_coseismic_column(event.id) is not None]
        if len(stations):
            self.constraint_registry.add_constraint(
                CoseismicConstraint(
                    event, stations,
                    self.config.coseismic_h_sigma,
                    self.config.coseismic_v_sigma
                )
            )
        else:
            tqdm.write(f'No stations observed coseismic event {event.id}. '
                       f'A coseismic constraint for this event will not be added.')

        # Postseismic for each relaxation
        for relax in self.config.relaxation:
            self.constraint_registry.add_constraint(
                PostseismicConstraint(
                    event, relax,
                    self.config.postseismic_h_sigma,
                    self.config.postseismic_v_sigma
                )
            )

    def remove_earthquake(self, event: Earthquake, json_folder: str = None, save_json_folder: str = None):
        """Add event to the list of modeled earthquakes"""

        # check the event is not already in the list
        if event in self.earthquakes:
            tqdm.write('Removing event ' + str(event))

            self.earthquakes.pop(self.earthquakes.index(event))
            self.config.earthquakes_cherry_picked.pop(self.config.earthquakes_cherry_picked.index(f'{event.id}'))
            self.earthquakes.sort()

            cnn = Cnn('gnss_data.cfg')

            # remove mask
            _, _, mask = self.grids.earthquake_masks[event.id]
            self.grids.earthquake_masks.pop(event.id)

            # figure out which stations need to be recomputed
            for i, stn in enumerate(self.stations):
                s_score, p_score = mask.score(stn.lat, stn.lon)
                if p_score > 0 or s_score > 0:
                    # replace old etm
                    stn.etm = self._build_etm(cnn, stn.network_code, stn.station_code, json_folder, save_json_folder)
                    # get dimensions of neq
                    new_par_count = stn.etm.design_matrix.matrix.shape[1]
                    # assume number of unknowns will change, so update the rest of the stations down from current
                    remove_from_index = 0
                    # will be added when calling _create_normal_equations
                    self.total_parameters = 0
                    self.total_equations = 0

                    for station in self.stations:
                        if stationID(station) == stationID(stn):
                            # add the number of new parameters to remove_from_index
                            remove_from_index = station.normal_equations.parameter_count - new_par_count
                            # create a new normal equations object and replace current
                            self.normal_equations[i] = self._create_normal_equations(stn)
                        else:
                            # remove from the parameter range the station that has been removed
                            # this is applied to any stations that come after the removed site
                            station.normal_equations.parameter_range -= remove_from_index
                            station.normal_equations.parameter_start_idx -= remove_from_index
                            self.total_parameters += station.normal_equations.parameter_count
                            self.total_equations += station.normal_equations.equation_count

            # now remove the constraints
            for i, constraint in enumerate(self.constraint_registry.constraints['coseismic']):
                if constraint.event.id == event.id:
                    tqdm.write(f'Removed {constraint}')
                    self.constraint_registry.constraints['coseismic'].pop(i)
                    break

            # remove the as many postseismic constraints as we have
            while event.id in [c.event.id for c in self.constraint_registry.constraints['postseismic']]:
                for i, constraint in enumerate(self.constraint_registry.constraints['postseismic']):
                    if constraint.event.id == event.id:
                        tqdm.write(f'Removed {constraint}')
                        self.constraint_registry.constraints['postseismic'].pop(i)
                        break

        else:
            tqdm.write(f'Event {event.id} is not in the list of modeled events')

    def get_constraint_summary(self) -> Dict:
        """Get summary of constraint system"""
        return self.constraint_registry.get_constraint_summary()

    def update_smoothing(self, event_id: str, new_smoothing: float):
        for const in self.constraint_registry.constraints['coseismic']:
            if const.event.id == event_id:
                tqdm.write(f'Found event {event_id} with current smoothing {const.smoothing:.3e}')
                const.smoothing = new_smoothing
                self.solved = False

    def update_smoothing_start_stop(self, event_id: str, new_smoothing_start: float,
                                    new_smoothing_stop: float):
        for const in self.constraint_registry.constraints['coseismic']:
            if const.event.id == event_id:
                tqdm.write(f'Found event {event_id} with current smoothing start {const.search_start_smoothing:.3e} '
                           f'stop {const.search_stop_smoothing:.3e}')
                const.search_start_smoothing = new_smoothing_start
                const.search_stop_smoothing = new_smoothing_stop
                # reset fields
                const.start_smoothing = [None, None, None]
                const.stop_smoothing = [None, None, None]

    def update_weights(self, event_id: str = None, relax: float = None, constraint_type: str = None,
                       h_sigma: float = None, v_sigma: float = None):
        """
        Update weights for specific constraint type or all constraints
        """
        apply_to = []

        if constraint_type and not event_id:
            # only constraint type was given
            apply_to = self.constraint_registry.constraints[constraint_type]
        elif constraint_type and event_id and relax:
            # constraint type, event and relax
            for constraint in self.constraint_registry.constraints[constraint_type]:
                if constraint.constraint_type in (ConstraintType.COSEISMIC, ConstraintType.POSTSEISMIC):
                    if constraint.event.id == event_id and constraint.relaxation == relax:
                        apply_to += [constraint]
        elif constraint_type and event_id and not relax:
            # no relax
            for constraint in self.constraint_registry.constraints[constraint_type]:
                if constraint.constraint_type in (ConstraintType.COSEISMIC, ConstraintType.POSTSEISMIC):
                    if constraint.event.id == event_id:
                        apply_to += [constraint]
        elif event_id and relax and not constraint_type:
            # event and relax but no constraint type (but implicitly is postseismic)
            for constraint in self.constraint_registry.constraints['postseismic']:
                if constraint.event.id == event_id and constraint.relaxation == relax:
                    apply_to += [constraint]
        elif event_id and not constraint_type and not relax:
            # only event id
            for constraint in (self.constraint_registry.constraints['coseismic'] +
                               self.constraint_registry.constraints['postseismic']):
                if constraint.event.id == event_id:
                    apply_to += [constraint]
        else:
            # nothing given, apply to all
            apply_to = self.constraint_registry.constraints.values()

        # do the thing
        for constraint in apply_to:
            constraint.update_weights(h_sigma, v_sigma)

        self.solved = False

    def plot_grid_result(self, sigmas=False):

        input_names = [stationID(stn) for stn in self.stations]
        input_lon = [stn.lon for stn in self.stations]
        input_lat = [stn.lat for stn in self.stations]

        available_fields, station_data, grid_lon, grid_lat, fields, fcovar = [], [], [], [], [], []

        postseismic = []
        for ifield in self.fields:
            if ifield.base_type == ConstraintType.POSTSEISMIC:
                if ifield.event.id in postseismic:
                    # field already in
                    continue

            parameters = np.zeros((3, len(self.stations)))
            station_data.append(parameters)
            available_fields.append(ifield.description)
            lon, lat = ifield.get_interpolation_grid_geographic()
            grid_lon.append(lon)
            grid_lat.append(lat)
            if sigmas:
                # do not plot data from stations in uncertainty mode
                fields.append(ifield.enu_sigma)
                fcovar.append(ifield.enu_covar)
            else:
                if ifield.base_type == ConstraintType.POSTSEISMIC:
                    r_field = 0
                    for f in [ff for ff in self.fields if ff.base_type == ConstraintType.POSTSEISMIC]:
                        if ifield.event == f.event:
                            idx = np.isin(np.array([stationID(stn) for stn in self.stations]),
                                          np.array([stationID(stn) for stn in f.constrain_stations]))
                            # pick the max relaxation and use it a dt
                            r_field += f.enu_field * np.log10(1 + self.config.relaxation.max()/f.relaxation)
                            # assign values to where they belong
                            parameters[:, idx] += (f.constrained_parameters *
                                                   np.log10(1 + self.config.relaxation.max()/f.relaxation))
                    # append postseismic field to keep track of which earthquakes were processed already
                    postseismic.append(ifield.event.id)
                    fields.append(r_field)
                else:
                    idx = np.isin(np.array([stationID(stn) for stn in self.stations]),
                                  np.array([stationID(stn) for stn in ifield.constrain_stations]))
                    # assign values to where they belong
                    parameters[:, idx] = ifield.constrained_parameters
                    fields.append(ifield.enu_field)

        # do the thing
        return plot_velocity_field(grid_lon, grid_lat, fields,
                                   np.array(input_lon), np.array(input_lat), station_data, input_names,
                                   self.plot_constrained_etm, available_fields, plot_sigmas=sigmas,
                                   covar=fcovar)

    def plot_constrained_etm(self, station_index, folder=None):
        cnn = Cnn('gnss_data.cfg')

        stn = self.stations[station_index]

        tqdm.write(f'Estimating constrained etm for {stationID(stn)}')

        config = EtmConfig(stn.network_code, stn.station_code, cnn=cnn)
        config = self._apply_config(config, cnn)
        config.solution.solution_type = SolutionType.PPP
        config.modeling.least_squares_strategy.constraints = stn.etm_constraints
        # add the prefit models that got removed from the ETM when we did the stack
        config.modeling.prefit_models = copy.deepcopy(stn.etm.config.modeling.prefit_models)

        for const in config.modeling.least_squares_strategy.constraints:
            par = ''
            for p in const.p.params:
                par += '[' + ' '.join([f'{a * 1000.:.2f}' for a in p.tolist()]) + '] '

            tqdm.write(f' -- Etm constrain: {const} {par}')

        if folder is None:
            config.plotting_config.interactive = True
        else:
            config.plotting_config.filename = folder

        config.plotting_config.plot_show_outliers = True
        config.plotting_config.plot_residuals_mode = True
        etm = EtmEngine(config, cnn=cnn, silent=True)

        # deactivate mechanical jumps jumps
        mechanical = etm.jump_manager.get_active_mechanical_jumps()
        for jump in mechanical:
            jump.fit = False

        etm.run_adjustment(cnn=cnn, try_save_to_db=False, try_loading_db=False)
        etm.plot()
        print('(EtmStacker) > ', end='', flush=True)

    def _extract_results(self) -> Tuple[List, List]:

        interseismic = []
        earthquakes = []
        for stn in self.stations:
            ap = stn.etm.design_matrix.get_polynomial().p.params
            idx = stn.get_velocity_column()
            interseismic.append({
                'station': stationID(stn),
                'lon': stn.lon,
                'lat': stn.lat,
                'a_priori': [ap[1][1], ap[0][1], ap[2][1]],
                'constrained': self.solution[:, idx].tolist(),
                'is_interseismic': stn.is_interseismic
            })
            for event in self.earthquakes:
                idx = stn.get_coseismic_column(event.id)
                if idx:
                    ap = stn.etm.jump_manager.get_geophysical_jump(event.id).p.params
                    earthquakes.append({
                        'station': stationID(stn),
                        'lon': stn.lon,
                        'lat': stn.lat,
                        'event_id': event.id,
                        'relax': 0.0,
                        'a_priori': [ap[1], ap[0], ap[2]],
                        'constrained': self.solution[:, idx].tolist(),
                    })
                for relax in self.config.relaxation:
                    idx = stn.get_postseismic_column(event.id, relax)
                    if idx:
                        jump = stn.etm.jump_manager.get_geophysical_jump(event.id)
                        ap = jump.p.params
                        col = jump.get_relaxation_cols(relax, False)
                        earthquakes.append({
                            'station': stationID(stn),
                            'lon': stn.lon,
                            'lat': stn.lat,
                            'event_id': event.id,
                            'relax': relax,
                            'a_priori': [ap[1][col], ap[0][col], ap[2][col]],
                            'constrained': self.solution[:, idx].tolist(),
                        })

        return interseismic, earthquakes

    def interpolate_fields_to_grid(self):

        if self.solved:
            # clean any previous runs
            self.fields = []

            self.fields.append(
                EtmStackerField.create_field(
                    self.stations, self.solution, self.covariance, self.grids)
            )

            for event in self.earthquakes:
                # find the constraint for this event
                coseismic_constraint = None
                for const in self.constraint_registry.constraints['coseismic']:
                    if const.event == event:
                        coseismic_constraint = const
                        break

                if coseismic_constraint is None:
                    tqdm.write(f'Could not find coseismic constraint for {event.id}')
                    continue

                fields = EtmStackerField.create_field(
                    self.stations, self.solution, self.covariance, self.grids, event,
                    self.config.relaxation, coseismic_constraint)

                self.fields += fields

        else:
            tqdm.write('System has not been solved! Invoke solve first')

    def get_trajectory_functions_at_point(self, lon: float, lat: float, etm: EtmEngine):
        pass
