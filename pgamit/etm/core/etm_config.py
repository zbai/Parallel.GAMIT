"""
Project: Parallel.GAMIT
Date: 09/12/2025 09:20 AM
Author: Demian D. Gomez
"""

# etm_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import IntEnum, auto
from datetime import datetime
from io import BytesIO
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from pgamit.pyDate import Date


class JumpType(IntEnum):
    """Enum for jump types to replace scattered constants"""
    UNDETERMINED = -1
    MECHANICAL_MANUAL = 1
    MECHANICAL_ANTENNA = 2
    REFERENCE_FRAME = 5
    COSEISMIC_JUMP_DECAY = 10
    COSEISMIC_ONLY = 15
    POSTSEISMIC_ONLY = 20

    @property
    def color(self) -> str:
        """Get the plotting color for this jump type"""
        color_map = {
            JumpType.UNDETERMINED: 'gray',
            JumpType.MECHANICAL_MANUAL: 'darkcyan',
            JumpType.MECHANICAL_ANTENNA: 'blue',
            JumpType.REFERENCE_FRAME: 'green',
            JumpType.COSEISMIC_JUMP_DECAY: 'red',
            JumpType.COSEISMIC_ONLY: 'purple',
            JumpType.POSTSEISMIC_ONLY: 'orange'
        }
        return color_map.get(self, 'black')

    @property
    def description(self) -> str:
        descriptions = {
            JumpType.UNDETERMINED: 'UNDETERMINED',
            JumpType.MECHANICAL_MANUAL: 'MECHANICAL (MANUAL)',
            JumpType.MECHANICAL_ANTENNA: 'MECHANICAL (ANTENNA CHANGE)',
            JumpType.REFERENCE_FRAME: 'REFERENCE FRAME CHANGE',
            JumpType.COSEISMIC_JUMP_DECAY: 'CO+POSTSEISMIC',
            JumpType.COSEISMIC_ONLY: 'COSEISMIC ONLY',
            JumpType.POSTSEISMIC_ONLY: 'POSTSEISMIC ONLY'
        }
        return descriptions.get(self, 'UNKNOWN')


@dataclass
class Earthquake:
    id: str = None
    lat: float = None
    lon: float = None
    date: Date = None
    depth: int = None
    magnitude: float = 0
    distance: float = 0
    location: str = None
    jump_type: JumpType = None

    def build_metadata(self) -> str:
        link = ('<a href="https://earthquake.usgs.gov/earthquakes/eventpage/%s" '
                'target="_blank">%s</a>'
                % (self.id, self.id))
        return f'{link}: M{self.magnitude:.1f} {self.location} -> {self.distance:.0f} km'


class PeriodicStatus(IntEnum):
    """Enum for periodic term status"""
    AUTOMATICALLY_ADDED = auto()
    ADDED_BY_USER = auto()
    UNABLE_TO_FIT = auto()


@dataclass
class JumpParameters:
    jump_type: JumpType = field(default_factory=lambda: JumpType)
    relaxation: np.ndarray = field(default_factory=lambda: np.array([0.5]))
    date: Date = None
    action: str = None


@dataclass
class LeastSquares:
    iterations: int = 10
    sigma_filter_limit: float = 2.5

@dataclass
class ModelingParameters:
    """Configuration for modeling parameters"""
    relaxation: np.ndarray = field(default_factory=lambda: np.array([0.5]))
    poly_terms: int = 2
    reference_epoch: float = 0
    frequencies: np.ndarray = field(
        default_factory=lambda: np.array([1 / 365.25, 1 / (365.25 / 2)])
    )
    periodic_status: PeriodicStatus = PeriodicStatus.AUTOMATICALLY_ADDED
    user_jumps: List[JumpParameters] = field(default_factory=lambda: [])
    earthquake_jumps: List[Earthquake] = field(default_factory=lambda: [])
    sigma_floor_h: float = 0.10
    sigma_floor_v: float = 0.15
    robust_lsq_limit: float = 2.5
    earthquake_min_days: int = 15
    jump_min_days: int = 5
    post_seismic_back_lim: int = 365 * 5 # 5 years of postseismic user_jumps back in time
    # master switches activating certain components
    fit_earthquakes: bool = True
    fit_generic_jumps: bool = True
    fit_metadata_jumps: bool = True

    def get_user_jump(self, date: Union[Date, datetime], jump_type: JumpType) -> Union[JumpParameters, None]:
        """obtain a jump from the database jump config using date and type"""
        for jump_params in self.user_jumps:
            if jump_params.date == date:
                # dates match, check types
                if jump_params.jump_type == jump_type:
                    # types match exactly, so it is the jump being looked for
                    return jump_params
                elif (jump_params.jump_type >= JumpType.COSEISMIC_JUMP_DECAY
                      and jump_type >= JumpType.COSEISMIC_JUMP_DECAY):
                    # a geophysical jump with a change in behavior, return it to the caller
                    return jump_params
        return None

@dataclass
class StationMetadata:
    lat: np.ndarray = field(default_factory=lambda: np.array([0]))
    lon: np.ndarray = field(default_factory=lambda: np.array([0]))
    height: np.ndarray = field(default_factory=lambda: np.array([0]))
    auto_x: np.ndarray = field(default_factory=lambda: np.array([6378137.]))
    auto_y: np.ndarray = field(default_factory=lambda: np.array([0]))
    auto_z: np.ndarray = field(default_factory=lambda: np.array([0]))
    first_obs: Date = Date(year=1980, doy=1)
    last_obs: Date = Date(datetime=datetime.now())
    max_dist: float = 20.0
    station_information: list = field(default_factory=lambda: [])


@dataclass
class SolutionOptions:
    soln: str = 'ppp'
    stack_name: str = 'ppp'


@dataclass
class ValidationRules:
    """Configuration for validation rules"""
    max_jump_amplitude: float = 4.0  # meters
    min_solutions_for_etm: int = 4
    min_data_for_jump: int = 50  # data points
    max_condition_number: float = 1e10


@dataclass
class PlotOutputConfig:
    _allowed_attributes = {
        'filename', 'file_io', 'format', 'save_kwargs',
        'plot_show_outliers', 'plot_residuals_mode', 'plot_time_window',
        'plot_remove_jumps', 'plot_remove_polynomial', 'plot_remove_periodic',
        'missing_solutions'
    }
    """Configuration for plot output"""
    filename: Optional[str] = None
    file_io: Optional[BytesIO] = None
    format: str = 'png'
    save_kwargs: Dict[str, Any] = None

    # Plot configuration
    plot_show_outliers: bool = True
    plot_residuals_mode: bool = False
    plot_time_window: Optional[Tuple[float, float]] = None
    plot_remove_jumps: bool = False
    plot_remove_polynomial: bool = False
    plot_remove_periodic: bool = False

    # Missing data
    missing_solutions: Optional[List] = None

    def __post_init__(self):
        if self.save_kwargs is None:
            self.save_kwargs = {}

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self._allowed_attributes:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'. ")
        super().__setattr__(name, value)

class ETMConfig:
    """Central configuration manager for ETM operations"""

    def __init__(self,
                 network_code: str,
                 station_code: str,
                 custom_config: Optional[Dict[str, Any]] = None,
                 cnn: Optional = None):
        """
        Initialize ETM configuration

        Args:
            custom_config: Dictionary of custom configuration overrides
            cnn: Database connection (if loading from database)
            network_code: Station network code (if loading from database)
            station_code: Station code (if loading from database)
        """
        self.network_code = network_code
        self.station_code = station_code

        self.solution = SolutionOptions()
        self.modeling = ModelingParameters()
        self.plotting_config = PlotOutputConfig()
        self.validation = ValidationRules()
        self.metadata = StationMetadata()
        self.least_squares = LeastSquares()

        # Language support
        self.language = 'eng'
        self._language_dict = {
            'eng': {
                "station": "Station",
                "north": "North",
                "east": "East",
                "up": "Up",
                "table_title": "  Year Day Relx    [mm] Mag D [km]",
                "periodic": "Seasonal Amplitude",
                "velocity": "Velocity",
                "from_model": "from model",
                "acceleration": "Acceleration",
                "position": "Conventional Epoch Pos.",
                "completion": "Completion",
                "other": "other polynomial terms",
                "not_enough": "Not enough solutions to fit an ETM.",
                "table_too_long": "Table too long to print!",
                "frequency": "Frequency",
                "N residuals": "N Residuals",
                "E residuals": "E Residuals",
                "U residuals": "U Residuals",
                "histogram plot": "Histogram",
                "residual plot": "Residual Plot",
                "jumps removed": "Jumps Removed",
                "polynomial removed": "Polynomial Removed",
                "seasonal removed": "Seasonal Removed"
            },
            'spa': {
                "station": "Estación",
                "north": "Norte",
                "east": "Este",
                "up": "Arriba",
                "table_title": "  Año  Día Relx    [mm] Mag D [km]",
                "periodic": "Amplitud Estacional",
                "velocity": "Velocidad",
                "from_model": "de modelo",
                "acceleration": "Aceleración",
                "position": "Posición Época Conv.",
                "completion": "Completitud",
                "other": "otros términos polinómicos",
                "not_enough": "No hay suficientes soluciones para ajustar trayectorias.",
                "table_too_long": "Tabla demasiado larga!",
                "frequency": "Frecuencia",
                "N residuals": "Residuos N",
                "E residuals": "Residuos E",
                "U residuals": "Residuos U",
                "histogram plot": "Histograma",
                "residual plot": "Gráfico de Residuos",
                "jumps removed": "Saltos Removidos",
                "polynomial removed": "Polinomio Removido",
                "seasonal removed": "Estacionales Removidas"
            }
        }

        if cnn:
            # loading parameters from the database
            self._load_from_database(cnn)

        if custom_config:
            self.apply_custom_config(custom_config)

    def get_station_id(self) -> str:
        """Get formatted station identifier"""
        return f"{self.network_code}.{self.station_code}"

    def _load_from_database(self, cnn) -> None:
        """Load station-specific configuration from database"""
        try:
            self._load_station_metadata(cnn)
            self._load_polynomial_config(cnn)
            self._load_periodic_config(cnn)
            self._load_jump_config(cnn)

            logger.info(f"Loaded configuration from database for {self.network_code}.{self.station_code}")

        except Exception as e:
            logger.warning(f"Failed to load database config for {self.network_code}.{self.station_code}: {e}")
            logger.info("Using default configuration")

    def _load_station_metadata(self, cnn):
        """Load station metadata from database"""
        query = '''
                SELECT * FROM stations 
                WHERE "NetworkCode" = '%s' AND "StationCode" = '%s'
            ''' % (self.network_code, self.station_code)

        stn = cnn.query_float(query, as_dict=True)

        if not stn or stn[0]['lat'] is None:
            raise ValueError(f"No valid metadata for station {self.get_station_id()}")

        """Load station reference coordinates and metadata"""
        self.metadata.lat = np.array([float(stn[0]['lat'])])
        self.metadata.lon = np.array([float(stn[0]['lon'])])
        self.metadata.height = np.array([float(stn[0]['height'])])
        self.metadata.auto_x = np.array([float(stn[0]['auto_x'])])
        self.metadata.auto_y = np.array([float(stn[0]['auto_y'])])
        self.metadata.auto_z = np.array([float(stn[0]['auto_z'])])
        self.metadata.first_obs = Date(fyear=stn[0]['DateStart'])
        self.metadata.last_obs = Date(fyear=stn[0]['DateEnd'])
        self.metadata.max_dist = 20.0 if not stn[0]['max_dist'] else stn[0]['max_dist']

        # as part of the metadata, load the station info
        from pgamit.pyStationInfo import StationInfo, pyStationInfoException
        try:
            station_info = StationInfo(cnn, self.network_code, self.station_code, allow_empty=True)

            self.metadata.station_information = station_info.records
        except pyStationInfoException:
            self.metadata.station_information = []

    def _load_polynomial_config(self, cnn) -> None:
        """Load polynomial configuration from etm_params table"""
        query = '''
            SELECT "terms", "Year", "DOY" FROM etm_params 
            WHERE "NetworkCode" = '%s' AND "StationCode" = '%s' 
            AND "object" = 'polynomial' AND "soln" = 'gamit'
            LIMIT 1
        ''' % (self.network_code, self.station_code)

        try:
            result = cnn.query_float(query, as_dict=True)

            if result:
                row = result[0]
                if row.get('terms'):
                    self.modeling.poly_terms = int(row['terms'])

                # Set reference epoch if specified
                if row.get('Year') and row.get('DOY'):
                    from pgamit import pyDate
                    ref_date = pyDate.Date(year=int(row['Year']), doy=int(row['DOY']))
                    self.modeling.reference_epoch = ref_date.fyear

        except Exception as e:
            logger.debug(f"No polynomial config in database: {e}")

    def _load_periodic_config(self, cnn) -> None:
        """Load periodic configuration from etm_params table"""
        query = '''
            SELECT "frequencies" FROM etm_params 
            WHERE "NetworkCode" = '%s' AND "StationCode" = '%s' 
            AND "object" = 'periodic' AND "soln" = 'gamit'
            LIMIT 1
        ''' % (self.network_code, self.station_code)

        try:
            result = cnn.query_float(query, as_dict=True)

            if result and result[0].get('frequencies'):
                # Assuming frequencies are stored as array in database
                freqs = result[0]['frequencies']
                if isinstance(freqs, (list, tuple)):
                    self.modeling.frequencies = np.array(freqs)
                    self.modeling.periodic_status = PeriodicStatus.ADDED_BY_USER

        except Exception as e:
            logger.debug(f"No periodic config in database: {e}")

    def _load_jump_config(self, cnn) -> None:
        """Load jump configuration from etm_params table"""
        from etm.core.s_score import ScoreTable

        # @todo: analyze if "soln" = 'gamit' always or should also allow 'ppp'
        query = '''
            SELECT "Year", "DOY", "action", "jump_type", "relaxation" 
            FROM etm_params 
            WHERE "NetworkCode" = '%s' AND "StationCode" = '%s' 
            AND "object" = 'jump' AND "soln" = 'gamit' ORDER BY ("Year", "DOY")
        ''' % (self.network_code, self.station_code)

        try:
            result = cnn.query_float(query, as_dict=True)

            if result:
                # Store jump configuration for later use
                for jump in result:
                    if jump['jump_type'] == 1:
                        jump_type = JumpType.COSEISMIC_JUMP_DECAY
                    elif jump['jump_type'] == 2:
                        jump_type = JumpType.POSTSEISMIC_ONLY
                    else:
                        jump_type = JumpType.MECHANICAL_MANUAL

                    jump_params = JumpParameters(
                        jump_type=jump_type,
                        relaxation=jump['relaxation'],
                        date=Date(year=int(jump['Year']), doy=int(jump['DOY'])),
                        action=jump['action'])

                    self.modeling.user_jumps.append(jump_params)
            else:
                self.modeling.user_jumps = []

            # now earthquakes
            # no information yet of data dates, load everything that is possible
            score = ScoreTable(cnn, self.metadata.lat[0], self.metadata.lon[0],
                               self.metadata.first_obs - self.modeling.post_seismic_back_lim,
                               self.metadata.last_obs)
            self.modeling.earthquake_jumps = score.table

        except Exception as e:
            logger.debug(f"No jump config in database: {e}")
            self.modeling.user_jumps = []

    def get_label(self, key: str) -> str:
        """Get localized label"""
        return self._language_dict[self.language].get(key, key)

    def apply_custom_config(self, config: Dict[str, Any]) -> None:
        """Apply custom configuration overrides"""
        for section_name, section_config in config.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        if self.modeling.poly_terms < 1:
            issues.append("Polynomial terms must be >= 1")

        if self.modeling.sigma_floor_h <= 0 or self.modeling.sigma_floor_v <= 0:
            issues.append("Sigma floor values must be > 0")

        if self.validation.min_solutions_for_etm < 2:
            issues.append("Minimum solutions for ETM must be >= 2")

        return issues
