"""
Project: Parallel.GAMIT
Date: 09/12/2025 09:20 AM
Author: Demian D. Gomez
"""
from typing import Dict, List, Optional, Any
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

# app
from pgamit.pyDate import Date
from pgamit.dbConnection import Cnn
from pgamit.etm.core.type_declarations import PeriodicStatus, JumpType
from pgamit.etm.core.data_classes import (SolutionOptions, ModelingParameters,
                                          ValidationRules, StationMetadata, LeastSquares, JumpParameters)
from pgamit.etm.visualization.data_classes import PlotOutputConfig


class EtmConfig:
    """Central configuration manager for ETM operations"""

    def __init__(self,
                 network_code: str,
                 station_code: str,
                 custom_config: Optional[Dict[str, Any]] = None,
                 cnn: Cnn = None,
                 solution_type: SolutionOptions = None):
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
                "table_too_long": "[...]",
                "frequency": "Frequency",
                "N residuals": "N Residuals",
                "E residuals": "E Residuals",
                "U residuals": "U Residuals",
                "histogram plot": "Histogram",
                "residual plot": "Residual Plot",
                "jumps": "Jumps",
                "polynomial": "Polynomial",
                "seasonal": "Seasonal",
                "stochastic": "Stochastic",
                "removed": "Removed"
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
                "table_too_long": "[...]",
                "frequency": "Frecuencia",
                "N residuals": "Residuos N",
                "E residuals": "Residuos E",
                "U residuals": "Residuos U",
                "histogram plot": "Histograma",
                "residual plot": "Gráfico de Residuos",
                "jumps": "Saltos",
                "polynomial": "Polinomio",
                "seasonal": "Estacionales",
                "stochastic": "Estocástico",
                "removed": "Removido(s)"
            }
        }
        if solution_type:
            self.solution = solution_type

        if cnn:
            # loading parameters from the database
            self._load_from_database(cnn)

        if custom_config:
            self.apply_custom_config(custom_config)

    def get_station_id(self) -> str:
        """Get formatted station identifier"""
        return f"{self.network_code}.{self.station_code}"

    def build_filename(self):
        """build a generic file name to use to save files"""
        return self.get_station_id() + "_" + self.solution.stack_name

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
        self.metadata.name = stn[0]['StationName']
        self.metadata.country_code = stn[0]['country_code']
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

    def load_from_json(self, filepath: Optional[str] = None, json_string: Optional[str] = None):
        # load basic fields from json file
        if filepath:
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif json_string:
            data = json.loads(json_string)
        else:
            raise ValueError("Either filepath or json_string must be provided")

        self.metadata = StationMetadata(**data['station_meta'])
        #for date in ('first_obs', 'last_obs'):
        #    data['station_meta'][date] = Date(**data['station_meta'][date])
        #for nump in ('auto_x', 'auto_y', 'auto_z')
        #self.metadata = StationMetadata(**data['station_meta'])
