"""
Project: Parallel.GAMIT
Date: 09/12/2025 09:20 AM
Author: Demian D. Gomez
"""

# etm_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import IntEnum, auto
import numpy as np
import logging

logger = logging.getLogger(__name__)

class JumpType(IntEnum):
    """Enum for jump types to replace scattered constants"""
    UNDETERMINED = -1
    MECHANICAL_MANUAL = 1
    MECHANICAL_ANTENNA = 2
    REFERENCE_FRAME = 5
    COSEISMIC_JUMP_DECAY = 10
    COSEISMIC_ONLY = 15
    POSTSEISMIC_ONLY = 20


class JumpTypeDict:
    @staticmethod
    def get_description(jump_type: JumpType) -> str:
        descriptions = {
            JumpType.UNDETERMINED: 'UNDETERMINED',
            JumpType.MECHANICAL_MANUAL: 'MECHANICAL (MANUAL)',
            JumpType.MECHANICAL_ANTENNA: 'MECHANICAL (ANTENNA CHANGE)',
            JumpType.REFERENCE_FRAME: 'REFERENCE FRAME CHANGE',
            JumpType.COSEISMIC_JUMP_DECAY: 'CO+POSTSEISMIC',
            JumpType.COSEISMIC_ONLY: 'COSEISMIC ONLY',
            JumpType.POSTSEISMIC_ONLY: 'POSTSEISMIC ONLY'
        }
        return descriptions.get(jump_type, 'UNKNOWN')


class PeriodicStatus(IntEnum):
    """Enum for periodic term status"""
    AUTOMATICALLY_ADDED = auto()
    ADDED_BY_USER = auto()
    UNABLE_TO_FIT = auto()


@dataclass
class ModelingParameters:
    """Configuration for modeling parameters"""
    relaxation: np.ndarray = field(default_factory=lambda: np.array([0.5]))
    poly_terms: int = 2
    reference_epoch: float = 0
    frequencies: np.ndarray = field(
        default_factory=lambda: np.array([1 / 365.25, 1 / (365.25 / 2)])
    )
    sigma_floor_h: float = 0.10
    sigma_floor_v: float = 0.15
    limit: float = 2.5
    eq_min_days: int = 15
    jp_min_days: int = 5


@dataclass
class StationMetadata:
    lat: np.ndarray = np.array([0])
    lon: np.ndarray = np.array([0])
    height: np.ndarray = np.array([0])
    auto_x: np.ndarray = np.array([6378137.])
    auto_y: np.ndarray = np.array([0])
    auto_z: np.ndarray = np.array([0])
    max_dist: float = 20.0

@dataclass
class ProcessingOptions:
    """Configuration for processing behavior"""
    fit_earthquakes: bool = True
    fit_generic_jumps: bool = True
    fit_periodic: bool = True
    plot_remove_jumps: bool = False
    plot_polynomial_removed: bool = False
    ignore_db_params: bool = False
    no_model: bool = False


@dataclass
class ValidationRules:
    """Configuration for validation rules"""
    max_jump_amplitude: float = 4.0  # meters
    min_solutions_for_etm: int = 4
    min_data_for_jump: int = 50  # data points
    max_condition_number: float = 1e10


@dataclass
class ParameterVector:
    # station and solution identification
    NetworkCode: str
    StationCode: str
    # Parameter storage
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    params: np.ndarray = field(default_factory=lambda: np.array([]))
    sigmas: np.ndarray = field(default_factory=lambda: np.array([]))
    covar: np.ndarray = field(default_factory=lambda: np.array([]))

    soln: str = ''
    stack: str = ''
    object: str = ''
    metadata: Optional[str] = None
    hash: int = 0
    t_ref: float = 0

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

        self.modeling = ModelingParameters()
        self.processing = ProcessingOptions()
        self.validation = ValidationRules()
        self.metadata = StationMetadata()

        # Language support
        self.language = 'eng'
        self._language_dict = {
            'eng': {
                "station": "Station",
                "north": "North",
                "east": "East",
                "up": "Up",
                "table_title": "Year Day Relx    [mm] Mag   D [km]",
                "periodic": "Periodic amp",
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
                "polynomial removed": "Polynomial Removed"
            },
            'spa': {
                "station": "Estación",
                "north": "Norte",
                "east": "Este",
                "up": "Arriba",
                "table_title": "Año  Día Relx    [mm] Mag   D [km]",
                "periodic": "Amp. Periódica",
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
                "polynomial removed": "Polinomio Removido"
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
            self._load_polynomial_config(cnn)
            self._load_periodic_config(cnn)
            self._load_jump_config(cnn)
            self._load_station_metadata(cnn)

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
        self.metadata.max_dist = 20.0 if not stn[0]['max_dist'] else stn[0]['max_dist']

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

        except Exception as e:
            logger.debug(f"No periodic config in database: {e}")

    def _load_jump_config(self, cnn) -> None:
        """Load jump configuration from etm_params table"""
        query = '''
            SELECT "Year", "DOY", "action", "jump_type", "relaxation" 
            FROM etm_params 
            WHERE "NetworkCode" = '%s' AND "StationCode" = '%s' 
            AND "object" = 'jump' AND "soln" = 'gamit'
        ''' % (self.network_code, self.station_code)

        try:
            result = cnn.query_float(query, as_dict=True)

            if result:
                # Store jump configuration for later use
                self.jump_config = result
            else:
                self.jump_config = []

        except Exception as e:
            logger.debug(f"No jump config in database: {e}")
            self.jump_config = []

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
