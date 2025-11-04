"""
Project: Geodesy Database Engine (GeoDE)
Date: 10/26/25 9:10AM
Author: Demian D. Gomez
"""
from functools import total_ordering
from typing import List, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging
import warnings

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.path import Path
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)

logging.getLogger('geode.etm.core.etm_stacker').setLevel(logging.DEBUG)

from ...dbConnection import Cnn
from .etm_engine import EtmEngine
from .etm_config import EtmConfig
from ..etm_functions.polynomial import PolynomialFunction
from ..etm_functions.jumps import JumpFunction
from .type_declarations import SolutionType, JumpType
from .data_classes import BaseDataClass, Earthquake
from ...elasticity.elastic_interpolation import get_qpw, get_radius, interpolate_at_points
from ...elasticity.diskload import compute_diskload, load_love_numbers
from ...elasticity.green_func import build_design_matrix
from ...Utils import stationID, azimuthal_equidistant, print_yellow, inverse_azimuthal
from ...pyDate import Date
from ...pyOkada import Mask

class EtmStackerException(Exception):
    pass


def fill_region_with_grid(x_points, y_points, radius) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    hull_x = hull_points[:, 0]
    hull_y = hull_points[:, 1]

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

    # Create a path from convex hull boundary
    boundary_path = Path(hull_points)

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


@dataclass
class EtmNormalEq(BaseDataClass):
    network_code: str = ''
    station_code: str = ''
    etm: EtmEngine = field(default_factory=lambda: None)
    lon: np.ndarray = field(default_factory=lambda: np.array([]))
    lat: np.ndarray = field(default_factory=lambda: np.array([]))

    parameter_count: int = 0
    interseismic: bool = False
    velocity_col: int = 0
    first_obs: Date = None
    vertical_response: np.ndarray = field(default_factory=lambda: np.array([]))
    normal_eq_n: Tuple = field(default_factory=lambda: ())
    normal_eq_c: Tuple = field(default_factory=lambda: ())

    # variable to save resulting constrained functions
    c_polynomial: PolynomialFunction = field(default_factory=lambda: PolynomialFunction)
    c_earthquakes: List[JumpFunction] = field(default_factory=lambda: [])

    def __str__(self):
        return f'{self.network_code}.{self.station_code}'

    def __repr__(self):
        return f'EtmNormalEq({str(self)})'

@total_ordering
@dataclass
class EarthquakeEq(BaseDataClass):
    event: Earthquake = None
    relaxation: np.ndarray = field(
        default_factory=lambda: np.array([0.5])
    )
    no_constraints: bool = False
    coseismic_constraints: List = field(default_factory=lambda: [])
    postseismic_constraints: List = field(default_factory=lambda: [])
    # for storing the final interpolated fields. List for each relaxation, ndarray
    grid_amplitudes: List[np.ndarray] = field(
        default_factory=lambda: []
    )
    # to save the earthquake mask
    mask: np.ndarray = field(default_factory=lambda: np.array([]))

    def __eq__(self, other):
        """Compare earthquakes based on event_date"""
        if isinstance(other, EarthquakeEq):
            return self.event.id == other.event.id
        # Allow comparison with Date objects directly
        if isinstance(other, Earthquake):
            return self.event.id == other.id
        return False

    def __lt__(self, other):
        """Less than comparison based on event_date"""
        if isinstance(other, EarthquakeEq):
            return self.event.date < other.event.date
        if isinstance(other, Earthquake):
            return self.event.date < other.date
        return NotImplemented

    def __hash__(self):
        """Make the object hashable based on event_date"""
        return hash(self.event.date)

TIME_POST = 15
MAG_LIM = 7.5

class EtmStacker:
    def __init__(self,
                 max_condition_number: float = 8.0,
                 earthquake_magnitude_limit: float = MAG_LIM,
                 post_seismic_back_lim: float = TIME_POST * 365,
                 relaxation: np.ndarray = np.array([0.05, 1.0]),
                 earthquakes_cherry_picked: list = ()):

        # normal equations
        self.etm_neq: List[EtmNormalEq] = []
        self.total_parameters: int = 0

        # parameter lists
        self.earthquakes: List[EarthquakeEq] = []

        # normal equations for horizontal
        self.neq_h: np.ndarray = np.array([])
        self.ceq_h: np.ndarray = np.array([])
        # normal equations for vertical
        self.neq_v: np.ndarray = np.array([])
        self.ceq_v: np.ndarray = np.array([])

        # var to save results
        self.x_horizontal: np.ndarray = np.array([])
        self.x_vertical: np.ndarray = np.array([])

        # read love numbers now to accelerate the process later
        self.h_love, self.l_love, self.k_love = load_love_numbers()

        self.max_condition_number = max_condition_number
        self.earthquake_magnitude_limit = earthquake_magnitude_limit
        self.post_seismic_back_lim = post_seismic_back_lim
        self.relaxation = relaxation
        self.earthquakes_cherry_picked = earthquakes_cherry_picked

        self.grids_origin = None
        self.stations_projected = None
        self.disk_grid = None
        self.interpolation_grid = None
        self.interpolation_geographic = None
        # in vertical response, each row is an interpolation grid point
        # and each column is a disk grid point
        self.grid_vertical_response = []
        # the final interpolated velocities (on the grid)
        self.grid_velocities = np.array([])

        sr = ','.join(['%.3f' % r for r in relaxation])
        cp = ','.join(['%s' % r for r in earthquakes_cherry_picked])
        logger.info(f'Initialized EtmStacker with max cond number: {max_condition_number}; relaxations: {sr}')
        logger.info(f'Earthquake mag limit: {earthquake_magnitude_limit}; Cherry picked earthquakes: {cp};')
        logger.info(f'Considering events up to {post_seismic_back_lim / 365} years back from station start')

    def add_station(self, cnn: Cnn,
                    network_code: str = '',
                    station_code: str = '',
                    etm: EtmEngine = None):

        if etm is None:
            if network_code != '' and station_code != '':
                logger.info(f'Estimating etm for {network_code}.{station_code}')

                config = EtmConfig(network_code, station_code, cnn=cnn)
                config.solution.solution_type = SolutionType.PPP

                config = self._apply_config(config)

                # @todo: change the parameters for minimum number of day between earthquakes
                config.refresh_config(cnn)

                etm = EtmEngine(config, cnn=cnn, silent=True)

                if etm.solution_data.solutions < 100:
                    logger.info(print_yellow(f'Station {network_code}.{station_code} has less than '
                                             f'100 solutions, skipping'))
                    return

                if etm.solution_data.time_vector[-1] - etm.solution_data.time_vector[0] <= 1.5:
                    logger.info(print_yellow(f'Station {network_code}.{station_code} has less than 1.5 '
                                             f'years of data, skipping'))
                    return

                etm.run_adjustment(cnn=cnn)

                if etm.config.modeling.status == etm.config.modeling.status.UNABLE_TO_FIT:
                    logger.info(print_yellow(f'Unable to fit station {network_code}.{station_code} (rank deficient?)'))
                    return
            else:
                raise EtmStackerException('etm, network_code and station_code cannot be None')
        else:
            logger.info(f'Adding external etm for {network_code}.{station_code}')
            network_code = etm.config.network_code
            station_code = etm.config.station_code

        etm_neq = EtmNormalEq(network_code, station_code, etm)
        self.etm_neq.append(etm_neq)
        self.fill_etm_neq(etm_neq)
        self._process_etm_normal_eq(etm_neq)

    @staticmethod
    def _process_etm_normal_eq(etm_neq: EtmNormalEq):
        """analyze and save the relevant information"""

        a = etm_neq.etm.design_matrix.matrix
        l = etm_neq.etm.solution_data.transform_to_local()
        n = []
        c = []
        for i in range(3):
            if etm_neq.etm.solution_data.solutions < 100:
                logger.info(f'Upweighting {stationID(etm_neq)}')
                p = np.diag(1 / etm_neq.etm.fit.results[i].obs_sigmas ** 2) * 100
            else:
                p = np.diag(1 / etm_neq.etm.fit.results[i].obs_sigmas ** 2)
            n.append(a.T @ p @ a)
            c.append(a.T @ p @ l[i])

        # append the normal equations to list
        etm_neq.normal_eq_n = tuple(n)
        etm_neq.normal_eq_c = tuple(c)

    def fill_etm_neq(self, etm_neq):
        # get the polynomial of the ETM
        poly = etm_neq.etm.design_matrix.get_polynomial()
        first_date = etm_neq.etm.solution_data.coordinates.dates[0]
        etm_neq.first_obs = first_date

        # save the velocity col for easy access
        etm_neq.velocity_col = poly.column_index[1] + self.total_parameters

        # save the earthquakes for this station
        self._save_earthquakes(etm_neq)

        # figure out if station can participate on interseismic model
        jump = etm_neq.etm.jump_manager.get_first_geophysical()
        if jump is None or (jump is not None and jump.p.jump_date > first_date):
            etm_neq.interseismic = True
            logger.debug(f'Station {stationID(etm_neq)} is interseismic')

        # get the design matrix before running adjustment
        a = etm_neq.etm.design_matrix.matrix

        # add to the total count of parameters
        self.total_parameters += a.shape[1]
        # save how many parameters this ETM has
        etm_neq.parameter_count = a.shape[1]
        # create shortcuts
        etm_neq.lon = etm_neq.etm.config.metadata.lon
        etm_neq.lat = etm_neq.etm.config.metadata.lat

    def _save_earthquakes(self, etm_neq, missing_days_tolerance=3):
        # dates in the time series
        dates = np.array([date.mjd for date in etm_neq.etm.solution_data.coordinates.dates])

        # find out if events in ETM need to be added to the list
        for jump in [jump for jump in etm_neq.etm.jump_manager.jumps
                     if jump.is_geophysical() and jump.fit]:

            if jump.earthquake is None:
                logger.info(f'Could not identify earthquake ID for station '
                            f'{stationID(etm_neq)} for jump date {jump.date}')
                continue

            # day gap between event and observations after the event
            md = np.min(dates[dates >= jump.date.mjd] - jump.date.mjd)

            if jump.earthquake not in self.earthquakes:
                logger.debug('Saving event ' + repr(jump))
                event = EarthquakeEq(jump.earthquake, self.relaxation)
                self.earthquakes.append(event)
                self.earthquakes.sort()
            else:
                event = [e for e in self.earthquakes if e.event.id == jump.earthquake.id][0]

            add_postseismic = False
            add_coseismic = False
            if jump.p.jump_type == JumpType.COSEISMIC_JUMP_DECAY:
                # print(f'coseismic {stationID(etm_neq)} {event.event.date}')
                add_coseismic = True
                add_postseismic = True
            elif jump.p.jump_type == JumpType.POSTSEISMIC_ONLY:
                # print(f'postseismic {stationID(etm_neq)} {event.event.date}')
                add_postseismic = True

            if add_coseismic:
                logger.debug(f'Adding station {stationID(etm_neq)} as coseismic for {jump.earthquake.id}')

                event.coseismic_constraints.append({
                    'etm_neq': etm_neq,
                    'coseismic_cols': 0,
                    'missing_days': md,
                    'is_constraint': md <= missing_days_tolerance,
                    'jump_object': jump,
                    'jump_amplitudes': [p[0] for p in jump.p.params]  # mostly for debugging purposes
                })

            if add_postseismic:
                logger.debug(f'Adding station {stationID(etm_neq)} as postseismic for {jump.earthquake.id}')
                # extract each column for the postseismic component
                cols = [None] * len(event.relaxation)

                for i, relax in enumerate(event.relaxation):
                    if relax in jump.p.relaxation:
                        relax_col = jump.get_relaxation_cols(relaxation=relax)
                        if len(relax_col):
                            cols[i] = relax_col[0] + self.total_parameters

                if any(cols):
                    event.postseismic_constraints.append({
                        'etm_neq': etm_neq,
                        'postseismic_cols': cols,
                        'missing_days': md,
                        'is_constraint': md <= missing_days_tolerance,
                        'jump_object': jump
                    })
                else:
                    logger.info(f'Event {jump.date} on station {stationID(etm_neq)} has not shared relaxations '
                                f'with input relaxations {event.relaxation}')

    def build_normal_eq(self, cnn: Cnn):
        tp = self.total_parameters

        self.neq_h = np.zeros((tp * 2, tp * 2))
        self.ceq_h = np.zeros((tp * 2,))

        self.neq_v = np.zeros((tp, tp))
        self.ceq_v = np.zeros((tp,))

        offset = 0
        for etm_neq in self.etm_neq:
            # Get the number of parameters for this station
            n_params = etm_neq.parameter_count

            # Place E component in the first block (rows/cols 0:tp)
            self.neq_h[offset:offset + n_params,
                       offset:offset + n_params] = etm_neq.normal_eq_n[1]

            self.ceq_h[offset:offset + n_params] = etm_neq.normal_eq_c[1]

            # Place N component in the second block (rows/cols tp:2*tp)
            self.neq_h[tp + offset:tp + offset + n_params,
                       tp + offset:tp + offset + n_params] = etm_neq.normal_eq_n[0]

            self.ceq_h[tp + offset:tp + offset + n_params] = etm_neq.normal_eq_c[0]

            # repeat for the vertical component
            self.neq_v[offset:offset + n_params,
                       offset:offset + n_params] = etm_neq.normal_eq_n[2]

            self.ceq_v[offset:offset + n_params] = etm_neq.normal_eq_c[2]

            # Update offset for next station
            offset += n_params

        # create the horizontal and vertical grids
        self.create_interpolation_grids(cnn)
        # call to insert the co-seismic constraints
        self._coseismic_constraints()
        # calculate response of disks
        self.create_disk_grid()
        # call to insert the interseismic constraints
        self._interseismic_constraints()
        # call to insert the co-seismic constraints
        self._postseismic_constraints()

    def create_interpolation_grids(self, cnn: Cnn, grid_spacing: float = 50):
        """grid spacing in km"""

        lat = []
        lon = []
        for etm_neq in self.etm_neq:
            lat.append(etm_neq.lat)
            lon.append(etm_neq.lon)

        logger.info(f'Creating interpolation grids, station count {len(lat)}')
        logger.info(f'lon {np.array(lon).min()} {np.array(lon).max()} '
                    f'lat {np.array(lat).min()} {np.array(lat).max()}')
        self.grids_origin = (np.array(lon).mean(), np.array(lat).mean())
        # project the stations coordinates
        x, y = azimuthal_equidistant(np.array(lon), np.array(lat), self.grids_origin[0], self.grids_origin[1])
        self.stations_projected = (x, y)
        # create the grid and save it
        grid_x, grid_y, hx, hy = fill_region_with_grid(x, y, grid_spacing / 2)
        # visualize_disks(x,y,hx,hy,grid_x,grid_y,grid_spacing / 2)
        self.interpolation_grid = (grid_x, grid_y)
        self.interpolation_geographic = np.array(inverse_azimuthal(grid_x, grid_y,
                                                                   self.grids_origin[0], self.grids_origin[1])).T

        lon, lat = self.interpolation_geographic.T

        # loop through seismic events and apply the s-score mask
        for event in self.earthquakes:
            _, score = Mask(cnn, event.event.id).score(lat, lon)
            logger.info(f'Mask for event {event.event.id} {score.shape}')
            event.mask = score > 0

        logger.info(f'Created interpolation grids with size {grid_x.shape}')

    def _compute_horizontal_grid_interpolant(self, x, y):
        """x, y are the projected coordinates of stations participating in a certain interpolation"""
        # now create the influence matrix for each grid point to use in the horizontal interpolation
        # compute response between all stations participating
        q, p, w = get_qpw(np.column_stack([x, y]), np.column_stack([x, y]), 8, 0.5)
        a = np.block([[q, w], [w, p]])

        # compute response between current station (0, 0) and all stations
        q, p, w = get_qpw(np.column_stack([x, y]), np.column_stack([self.interpolation_grid[0],
                                                                    self.interpolation_grid[1]]), 8, 0.5)
        # here, we do (A^-1 * B^t)^t so that we can multiply by v and get the answer at the interp points
        ke = np.linalg.solve(a, np.hstack((q, w)).T ).T
        kn = np.linalg.solve(a, np.hstack((w, p)).T ).T

        #print('solve')
        #print(ke)
        #print('inv')
        #print((np.linalg.inv(a) @ np.hstack((q, w)).T).T)

        return np.vstack((ke, kn))

    @staticmethod
    def _compute_vertical_grid_interpolant(grid_responses, data_responses) -> np.ndarray:

        a = np.array(data_responses)
        g = np.array(grid_responses)

        # a number smaller than the number of observations to truncate the SVD decomp
        k = int(a.shape[0] * 2 / 3)
        # get the SVD decomposition for the design matrix
        u, s, vt = np.linalg.svd(a, full_matrices=True)
        v = vt.T

        u_k = u[:, :k]
        s_k = s[:k]
        v_k = v[:, :k]

        ku = g @ (v_k / s_k) @ u_k.T

        return ku

    def create_disk_grid(self, disk_rad=50, nmax_max=5000):
        # create the grid of disks
        grid_x, grid_y, _, _ = fill_region_with_grid(
            self.stations_projected[0], self.stations_projected[1], disk_rad
        )
        self.disk_grid = (grid_x, grid_y)

        logger.info(f'Disk grid size {grid_x.shape}')

        # compute the response for each station
        for i, etm_neq in enumerate(self.etm_neq):

            r, _, _ = get_radius(np.column_stack([grid_x, grid_y]),
                                 np.column_stack([self.stations_projected[0][i],
                                                  self.stations_projected[1][i]]))
            # add an offset to avoid singularities
            r = r + 8
            response = compute_diskload(alpha=disk_rad, theta_range=r, nmax_max=nmax_max,
                                        h_love=self.h_love, l_love=self.l_love, k_love=self.k_love)

            # response on the site to all disks
            etm_neq.vertical_response = response['U'].flatten()
            logger.info(f'Computed disk grid response for {stationID(etm_neq)} {etm_neq.vertical_response.shape}')

        # compute the responses for the interpolation grid
        for i, (x, y) in enumerate(zip(self.interpolation_grid[0], self.interpolation_grid[1])):
            r, _, _ = get_radius(np.column_stack([grid_x, grid_y]),
                                 np.column_stack([x, y]))
            # add an offset to avoid singularities
            r = r + 8
            response = compute_diskload(alpha=disk_rad, theta_range=r, nmax_max=nmax_max,
                                        h_love=self.h_love, l_love=self.l_love, k_love=self.k_love)

            self.grid_vertical_response.append(response['U'])
            logger.info(f'Computed disk grid response for interpolation grid element {i}')

        self.grid_vertical_response = np.array(self.grid_vertical_response)

        logger.info(f'disk count: {grid_x.shape} '
                    f'interpolation grid count: {self.interpolation_grid[0].shape} '
                    f'size of response matrix: {self.grid_vertical_response.shape}')

    def _postseismic_constraints(self, h_sigma=0.001, v_sigma=0.003):
        tp = self.total_parameters

        for event in self.earthquakes:
            for i, relax in enumerate(event.relaxation):
                logger.info(f'Applying postseismic constraints to {event.event.id} relaxation {relax}')
                idx = []
                lat = []
                lon = []
                const_neq = []
                needs_const = []
                # the ids of the postseismic columns for stations with 'missing_days' < missing_days_tolerance
                for e in event.postseismic_constraints:
                    if e['is_constraint']:
                        if e['postseismic_cols'][i]:
                            # it is a constrain station, since it has data and has the relaxation
                            idx.append(e['postseismic_cols'][i])
                            lon.append(e['etm_neq'].etm.config.metadata.lon)
                            lat.append(e['etm_neq'].etm.config.metadata.lat)
                            const_neq.append(e['etm_neq'])
                    else:
                        if e['postseismic_cols'][i]:
                            # it has the relaxation but not the data, needs to be constrained
                            needs_const.append(e)

                lat = np.array(lat)
                lon = np.array(lon)
                idx = np.array(idx)

                count_needs_const = len(needs_const)

                if len(idx):
                    logger.info(f'{count_needs_const} need to be constrained with {len(idx)} stations')

                    # matrix for the Ve Vn constraints
                    kn = np.zeros((count_needs_const, 2 * tp))
                    ke = np.zeros((count_needs_const, 2 * tp))
                    ph = np.diag(1 / (np.ones(count_needs_const * 2) * h_sigma ** 2))

                    kv = np.zeros((count_needs_const, tp))
                    pv = np.diag(1 / (np.ones(count_needs_const) * v_sigma ** 2))

                    for j, const in enumerate(needs_const):
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings('error')  # Convert warnings to exceptions
                                # calculate equidistant coordinates
                                x, y = azimuthal_equidistant(const['etm_neq'].lon, const['etm_neq'].lat, lon, lat)

                                _ke, _kn = self._compute_interpolant_constraint(x, y)

                                # at the velocity position of the site, place constraint
                                ke[j, const['postseismic_cols'][i]] = -1
                                kn[j, const['postseismic_cols'][i] + tp] = -1
                                ke[j, np.concatenate((idx, idx + tp))] = _ke
                                kn[j, np.concatenate((idx, idx + tp))] = _kn

                                _ku = self._compute_vertical_constraint(const['etm_neq'], const_neq)

                                kv[j, const['postseismic_cols'][i]] = -1
                                kv[j, idx] = _ku

                        except RuntimeWarning as w:
                            station_id = stationID(const['etm_neq'])
                            print(f'Warning in station {station_id}: {w}')

                    kh = np.vstack((ke, kn))

                    # constraints for the horizontal
                    nk = kh.T @ ph @ kh
                    self.neq_h += nk
                    # constraints for the vertical
                    nu = kv.T @ pv @ kv
                    self.neq_v += nu
                else:
                    event.no_constraints = True
                    logger.info(f'Cannot constrain {event.event.id}: no constraining stations')

    def _interseismic_constraints(self, h_sigma=0.00001, v_sigma=0.00001):
        tp = self.total_parameters

        lon = np.array([e.etm.config.metadata.lon for e in self.etm_neq if e.interseismic])
        lat = np.array([e.etm.config.metadata.lat for e in self.etm_neq if e.interseismic])
        # the ids of the velocity columns for stations with interseismic component
        idx = np.array([e.velocity_col for e in self.etm_neq if e.interseismic])

        interseismic = [e for e in self.etm_neq if e.interseismic]
        needs_interseismic_const = [e for e in self.etm_neq if not e.interseismic]
        count_needs = len(needs_interseismic_const)

        # matrix for the Ve Vn constraints
        kn = np.zeros((count_needs, 2 * tp))
        ke = np.zeros((count_needs, 2 * tp))
        ph = np.diag(1 / (np.ones(count_needs * 2) * h_sigma ** 2))

        kv = np.zeros((count_needs, tp))
        pv = np.diag(1 / (np.ones(count_needs) * v_sigma ** 2))

        for i, etm_neq in enumerate(needs_interseismic_const):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')  # Convert warnings to exceptions
                    # calculate equidistant coordinates
                    x, y = azimuthal_equidistant(etm_neq.lon, etm_neq.lat, lon, lat)

                    _ke, _kn = self._compute_interpolant_constraint(x, y)

                    # at the velocity position of the site, place constraint
                    ke[i, etm_neq.velocity_col] = -1
                    kn[i, etm_neq.velocity_col + tp] = -1
                    ke[i, np.concatenate((idx, idx + tp))] = _ke
                    kn[i, np.concatenate((idx, idx + tp))] = _kn

                    # compute the vertical constraint for this station
                    # using all stations with interseismic
                    _ku = self._compute_vertical_constraint(etm_neq, interseismic)

                    logger.debug(f'Assigning values to columns {idx}')

                    kv[i, etm_neq.velocity_col] = -1
                    kv[i, idx] = _ku

            except RuntimeWarning as w:
                logger.critical(f'Warning in station {stationID(etm_neq)}: {w}')

        kh = np.vstack((ke, kn))

        # constraints for the horizontal
        nk = kh.T @ ph @ kh
        self.neq_h += nk
        # constraints for the vertical
        nu = kv.T @ pv @ kv
        self.neq_v += nu

    @staticmethod
    def _compute_interpolant_constraint(x, y) -> Tuple[np.ndarray, np.ndarray]:

        # compute response between all stations
        q, p, w = get_qpw(np.column_stack([x, y]), np.column_stack([x, y]), 8, 0.5)
        a = np.block([[q, w], [w, p]])

        # compute response between current station (0, 0) and all stations
        q, p, w = get_qpw(np.column_stack([x, y]), np.array([[0, 0]]), 8, 0.5)

        ke = np.linalg.solve(a, np.concatenate((q.flatten(), w.flatten())))
        kn = np.linalg.solve(a, np.concatenate((w.flatten(), p.flatten())))

        return ke, kn

    @staticmethod
    def _compute_vertical_constraint(stn, all_stns) -> np.ndarray:

        a = []
        for etm_neq in all_stns:
            # build the design matrix
            a.append(etm_neq.vertical_response)

        logger.debug(f'Computing vertical constraints for {stationID(stn)}')
        logger.debug(f'Size of vertical constraint matrix is {np.array(a).shape}')
        # a number smaller than the number of observations to truncate the SVD decomp
        k = int(len(a) * 2 / 3)
        # get the SVD decomposition for the design matrix
        u, s, vt = np.linalg.svd(np.array(a), full_matrices=True)
        v = vt.T

        u_k = u[:, :k]
        s_k = s[:k]
        v_k = v[:, :k]

        ku = stn.vertical_response @ (v_k / s_k) @ u_k.T

        return ku

    def _compute_okada_disp_field(self, strike: float, dip: float, rake: float, x: np.ndarray, y: np.ndarray,
                            along_strike_l: float, downdip_l: float, depth: float, avg_disp: float,
                            station_count: int):

        cosd = lambda x: np.cos(np.deg2rad(x))
        sind = lambda x: np.sin(np.deg2rad(x))

        from ...pyOkada import okada

        # source dimensions L is horizontal, and W is depth
        L1 = -along_strike_l / 2
        L2 =  -L1
        W1 = -downdip_l / 2
        W2 = -W1

        total_patches = int(station_count * 3 / 2)
        # nw, nl = self.find_grid_with_aspect_ratio(total_patches)
        nl = 1
        nw = 1

        al = np.linspace(L1, L2, nl + 1)
        aw = np.linspace(W1, W2, nw + 1)
        l, w = np.meshgrid(al, aw)

        pl1 = l[:-1, :-1].flatten(order='F')
        pl2 = l[:-1, 1:].flatten(order='F')
        pw1 = w[:-1, :-1].flatten(order='F')
        pw2 = w[1:, :-1].flatten(order='F')

        nt = np.array([])
        et = np.array([])
        ut = np.array([])

        print(f'created {total_patches} with {nl} by {nw}')

        for k, _ in enumerate(pl1):
            W1 = pw1[k]
            W2 = pw2[k]
            L1 = pl1[k]
            L2 = pl2[k]

            # check depth of fault edge (add 500 meters for security factor)
            #depth2 = depth - ((pw1[k] + pw2[k])/2 * sind(dip) + 500)
            d2 = depth - (W2 * sind(dip) + 500)

            if d2 < 0:
                # fault is sticking out of the ground! reduce depth
                depth = depth - d2

            # clockwise rotation
            R = np.array([[cosd(90 - strike), sind(90 - strike)],
                         [-sind(90 - strike), cosd(90 - strike)]])

            # compute the transformed station coordinates
            T = R @ np.array([x.flatten(), y.flatten()])

            tx = T[0, :] #- (pl1[k] + pl2[k]) / 2
            ty = T[1, :] #- (pw1[k] + pw2[k]) / 2

            ns, es, us = okada(0.5, tx, ty, depth, L1, L2, W1, W2,
                               sind(dip), cosd(dip), 1, 0, 0)

            nd, ed, ud = okada(0.5, tx, ty, depth, L1, L2, W1, W2,
                               sind(dip), cosd(dip), 0, 1, 0)

            n = np.concatenate((ns, nd))
            e = np.concatenate((es, ed))
            u = np.concatenate((us, ud))

            if k == 0:
                nt = n
                et = e
                ut = u
            else:
                nt = np.vstack((nt, n))
                et = np.vstack((et, e))
                ut = np.vstack((ut, u))

        return et.T, nt.T, ut.T

    def _compute_dislocations(self, lon: float, lat: float, strike: float, dip: float,
                              along_strike: float, downdip: float, depth: float,
                              stnlon: np.ndarray, stnlat: np.ndarray):

        sind = lambda x: np.sin(np.deg2rad(x))
        cosd = lambda x: np.cos(np.deg2rad(x))
        
        # source dimensions L is horizontal, and W is depth
        L1 = -along_strike / 2
        L2 = -L1
        W1 = -downdip / 2
        W2 = -W1

        # total number of possible patches is number of observations * 3/2 (ENU)/(strike+dip) * 3 (Gómez et al 2023)
        n_patches = int(len(stnlat) * 3 * 3)

        # Each circle occupies a square of side 2r
        # Number of circles: (L/2r) × (W/2r) = N
        # Therefore: L×W / 4r² = N
        radius = np.sqrt(along_strike * downdip / (4 * n_patches))

        # using the L1/2 W1/W2 coordinate system, build the points using disk code
        # X is the downdip direction, Y the strike direction
        x = np.array([W1, W1, W2, W2])
        y = np.array([L1, L2, L2, L1])

        grid_dd, grid_ss, _, _ = fill_region_with_grid(x, y, radius)

        s = self._compute_laplacian(grid_dd, grid_ss, 2 * radius)

        # calculate the depth of the points given the grid_dd coordinate
        grid_dep = depth + grid_dd * sind(dip)
        
        # check that no patches stick out of the ground
        if np.any(grid_dep < 0):
            grid_dep = grid_dep - grid_dep.min()

        # clockwise rotation
        R = np.array([[cosd(strike), sind(strike)],
                      [-sind(strike), cosd(strike)]])

        # compute the transformed patches coordinates. -R because we want to go from fault system to
        # the geographical system. Compress the dd coordinates to match the dipping of the fault
        t = R @ np.array([grid_dd * cosd(dip), grid_ss])
        tx = t[0, :]
        ty = t[1, :]

        # convert the dd and ss grid to lon lat.
        grid_lon, grid_lat = inverse_azimuthal(tx, ty, lon, lat)

        n = grid_lat.shape[0]
        logger.info('Computing stations coseismic response...')
        a = build_design_matrix(np.array([90 - grid_lat, grid_lon, grid_dep]).T,
                                np.array([90 - stnlat, stnlon]).T,
                                np.array([np.ones(n) * strike,
                                          np.ones(n) * dip,
                                          np.ones(n) * (radius * 1000) ** 2]).T)

        ae, an, au = self._dislocation_parts(a)

        # now get response for interpolation grid
        n = self.interpolation_grid[1].shape[0]
        logger.info('Computing stations coseismic response...')
        a = build_design_matrix(np.array([90 - grid_lat, grid_lon, grid_dep]).T,
                                np.array([90 - self.interpolation_geographic[:, 1],
                                          self.interpolation_geographic[:, 0]]).T,
                                np.array([np.ones(n) * strike,
                                          np.ones(n) * dip,
                                          np.ones(n) * (radius * 1000) ** 2]).T)

        ge, gn, gu = self._dislocation_parts(a)

        s1 = np.hstack((s, np.zeros_like(s)))
        s2 = np.hstack((np.zeros_like(s), s))

        return ae, an, au, ge, gn, gu, np.vstack((s1, s2)), grid_lon, grid_lat

    def _compute_laplacian(self, x, y, radius):
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

    def _dislocation_parts(self, a):
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

    def _coseismic_constraints(self):
        # analyze each event an figure out how many stations of each type we have
        for event in self.earthquakes:
            along_strike_l = 10. ** (-3.22 + 0.69 * event.event.magnitude)  # [km]
            downdip_l = 10. ** (-1.01 + 0.32 * event.event.magnitude)  # [km]

            sites_lon = np.array([s['etm_neq'].lon[0] for s in event.coseismic_constraints])
            sites_lat = np.array([s['etm_neq'].lat[0] for s in event.coseismic_constraints])
            le = np.array([s['jump_amplitudes'][1] for s in event.coseismic_constraints])
            ln = np.array([s['jump_amplitudes'][0] for s in event.coseismic_constraints])
            lu = np.array([s['jump_amplitudes'][2] for s in event.coseismic_constraints])

            if sites_lat.shape[0] >= 2:
                ae, an, au, ge, gn, gu, s, patch_lon, patch_lat, = self._compute_dislocations(
                    event.event.lon, event.event.lat,
                    event.event.strike[0], event.event.dip[0],
                    along_strike_l, downdip_l, event.event.depth,
                    sites_lon, sites_lat
                )
                a = np.vstack((ae, an, au))
                g = np.vstack((ge, gn, gu))
                l = np.hstack((le, ln, lu))
                nk = s.T @ s
                n = a.T @ a + nk/1e8
                c = a.T @ l
                x = np.linalg.solve(n, c)

                visualize_vectors(patch_lon, patch_lat, np.reshape(x, (-1, ae.shape[1] // 2)))

                f = np.reshape(g @ x, (-1, self.interpolation_geographic.shape[0])).T
                o = np.hstack(
                    (np.array([self.interpolation_geographic[:, 0], self.interpolation_geographic[:, 1]]).T, f)
                )
                np.savetxt('maule.txt', o)
                #k = int(a.shape[0] * 2 / 3)
                # get the SVD decomposition for the design matrix
                #u, s, vt = np.linalg.svd(a, full_matrices=True)
                #v = vt.T

                #u_k = u[:, :k]
                #s_k = s[:k]
                #v_k = v[:, :k]

                #x = (v_k / s_k) @ u_k.T @ np.hstack((le, ln, lu))


    def fit_stack_parameters(self, h_sigma=0.00001, v_sigma=0.00001) -> Tuple[List, List]:
        xh = self.x_horizontal = np.linalg.solve(self.neq_h, self.ceq_h)
        xv = self.x_vertical = np.linalg.solve(self.neq_v, self.ceq_v)

        out_velocities = []
        out_postseismic = []
        ve = []
        vn = []
        vu = []
        for etm_neq in self.etm_neq:
            # configure the etm_neq to use the constrained velocity
            poly = PolynomialFunction(etm_neq.etm.config)
            poly.p.params = [np.array([np.nan, xh[etm_neq.velocity_col + self.total_parameters]]),
                             np.array([np.nan, xh[etm_neq.velocity_col]]),
                             np.array([np.nan, xv[etm_neq.velocity_col]])]

            poly.p.sigmas = [np.array([np.nan, h_sigma]),
                             np.array([np.nan, h_sigma]),
                             np.array([np.nan, v_sigma])]

            etm_neq.c_polynomial = poly

            # extract parameters to add to the list
            etm_poly = etm_neq.etm.design_matrix.get_polynomial()

            ve.append(xh[etm_neq.velocity_col])
            vn.append(xh[etm_neq.velocity_col + self.total_parameters])
            vu.append(xv[etm_neq.velocity_col])

            out_velocities.append({
                'station': stationID(etm_neq),
                'lon': etm_neq.lon[0],
                'lat': etm_neq.lat[0],
                'vc': [
                    xh[etm_neq.velocity_col] * 1000,
                    xh[etm_neq.velocity_col + self.total_parameters] * 1000,
                    xv[etm_neq.velocity_col] * 1000
                ],
                'vp': [
                    etm_poly.p.params[1][1] * 1000,
                    etm_poly.p.params[0][1] * 1000,
                    etm_poly.p.params[2][1] * 1000
                ],
                'c_interseismic': 1 if etm_neq.interseismic else 0
            })

        ########################################################################################
        self.grid_velocities = self._interpolate_field(np.ones_like(ve, dtype=bool), np.array(ve), np.array(vn), np.array(vu))
        ########################################################################################

        # now get the constrained earthquake components
        for event in self.earthquakes:

            if event.no_constraints:
                # skip event if no constraints available
                continue
            for i, relax in enumerate(event.relaxation):
                ae = []
                an = []
                au = []
                for stn in event.postseismic_constraints:
                    if stn['postseismic_cols'][i]:
                        etm_neq = stn['etm_neq']
                        jump = stn['jump_object']
                        col = stn['postseismic_cols'][i]
                        jcol = jump.get_relaxation_cols(relaxation=relax, return_col_of_design_matrix=False)[0]
                        out_postseismic.append({
                            'station': stationID(etm_neq),
                            'lon': etm_neq.lon[0],
                            'lat': etm_neq.lat[0],
                            'event_id': event.event.id,
                            'relax': relax,
                            'rc': [
                                xh[col] * 1000,
                                xh[col + self.total_parameters] * 1000,
                                xv[col] * 1000
                            ],
                            'rp': [
                                jump.p.params[1][jcol] * 1000,
                                jump.p.params[0][jcol] * 1000,
                                jump.p.params[2][jcol] * 1000
                            ],
                            'c_relax': 1 if stn['is_constraint'] else 0
                        })

                        jc = JumpFunction(etm_neq.etm.config,
                                          time_vector=np.array([0]),
                                          date=jump.date,
                                          jump_type=jump.p.jump_type,
                                          fit=False)

                        ae.append(xh[col])
                        an.append(xh[col + self.total_parameters])
                        au.append(xv[col])

                        par = [xh[col + self.total_parameters], xh[col], xv[col]]
                        sig = [0.001, 0.001, 0.001]

                        for j in range(3):
                            jc.p.params[j] = np.array([np.nan] * jump.param_count)
                            jc.p.params[j][jcol] = par[j]
                            jc.p.sigmas[j] = np.array([np.nan] * jump.param_count)
                            jc.p.sigmas[j][jcol] = sig[j]

                        etm_neq.c_earthquakes.append(jc)

                ########################################################################################
                selected_stations = np.isin(np.array(self.etm_neq),
                                            [e['etm_neq'] for e in event.postseismic_constraints
                                             if e['postseismic_cols'][i]])

                grid_e, grid_n, grid_u = self._interpolate_field(selected_stations, np.array(ae), np.array(an), np.array(au))
                grid_e[~event.mask] = np.nan
                grid_n[~event.mask] = np.nan
                grid_u[~event.mask] = np.nan
                event.grid_amplitudes.append(np.array([grid_e, grid_n, grid_u]))
                ########################################################################################

        return out_velocities, out_postseismic

    def _interpolate_field(self, selected_stations, ve, vn, vu):
        # compute the grid velocities
        ah = self._compute_horizontal_grid_interpolant(
            self.stations_projected[0][selected_stations],
            self.stations_projected[1][selected_stations]
        )

        av = self._compute_vertical_grid_interpolant(
            self.grid_vertical_response,
            [e.vertical_response for i, e in enumerate(self.etm_neq) if selected_stations[i]]
        )

        result = np.concatenate((ah @ np.concatenate((ve, vn)), av @ vu))

        return np.reshape(result, (3, self.interpolation_grid[0].shape[0]))

    def _apply_config(self, config):
        config.validation.max_condition_number = self.max_condition_number
        config.modeling.check_jump_collisions = False # turn off jump collision check. Add all jumps.
        config.modeling.earthquake_magnitude_limit = self.earthquake_magnitude_limit
        config.modeling.post_seismic_back_lim = self.post_seismic_back_lim
        config.modeling.relaxation = self.relaxation
        config.modeling.earthquakes_cherry_picked = self.earthquakes_cherry_picked

        return config

    def plot_constrained_etm(self, cnn: Cnn,
                             network_code: str = '',
                             station_code: str = '',
                             output_folder: str = './') -> None:

        config = EtmConfig(network_code, station_code, cnn=cnn)
        config.solution.solution_type = SolutionType.PPP

        config = self._apply_config(config)

        config.refresh_config(cnn)

        etm_neq = [_etm for _etm in self.etm_neq if _etm.network_code == network_code
                   and _etm.station_code == station_code]

        if len(etm_neq):
            poly = etm_neq[0].c_polynomial

            config.modeling.least_squares_strategy.constraints.append(poly)
            config.modeling.least_squares_strategy.constraints += etm_neq[0].c_earthquakes
            logger.info(f'Plotting {stationID(etm_neq[0])} -> {str(poly.p.params)}')

            etm = EtmEngine(config, cnn=cnn, silent=True)
            etm.run_adjustment(cnn=cnn, try_save_to_db=False, try_loading_db=False)
            etm.config.plotting_config.plot_show_outliers = True
            etm.config.plotting_config.filename = output_folder
            etm.plot()
        else:
            logger.info(f'Station {network_code}.{station_code} could not be found. '
                        f'Most likely removed due to few data points.')

    def get_constrained_velocities(self):
        pass