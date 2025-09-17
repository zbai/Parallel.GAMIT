"""
Project: Parallel.GAMIT
Date: 9/12/25 9:32 AM
Author: Demian D. Gomez
"""

# solution_data.py
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from pgamit import pyDate
from pgamit.Utils import crc32
from etm.core.etm_config import ETMConfig


class SolutionData(ABC):
    """Base class for all solution data types"""

    def __init__(self, config: ETMConfig):
        self.network_code = config.network_code
        self.station_code = config.station_code
        self.config = config

        # Common attributes
        self.solutions: int = 0
        self.completion: float = 0.0
        self.hash: int = 0
        self.soln: str = ""
        self.stack_name: str = ""
        self.project: str = ""

        # Coordinate arrays
        self.x: np.ndarray = np.array([])
        self.y: np.ndarray = np.array([])
        self.z: np.ndarray = np.array([])
        self.time_vector: np.ndarray = np.array([])
        self.time_vector_mjd: np.ndarray = np.array([])
        self.date: List[pyDate.Date] = []

        # Reference coordinates
        self.lat = config.metadata.lat
        self.lon = config.metadata.lon
        self.height = config.metadata.height
        self.auto_x = config.metadata.auto_x
        self.auto_y = config.metadata.auto_y
        self.auto_z = config.metadata.auto_z
        self.max_dist = config.metadata.max_dist

        # Quality control
        self.blunders: List = []
        self.excluded: List = []
        self.gaps: np.ndarray = np.array([])

        # Time series for plotting
        self.time_vector_cont: np.ndarray = np.array([])
        self.time_vector_cont_mjds: np.ndarray = np.array([])
        # no solution and blunders
        self.time_vector_ns: np.ndarray = np.array([])  # no solution epochs
        self.time_vector_blunders: np.ndarray = np.array([])  # blunder epochs

    @abstractmethod
    def load_data(self, cnn, **kwargs) -> None:
        """Load data from database or other source"""
        pass

    @abstractmethod
    def validate_data(self) -> List[str]:
        """Validate loaded data and return list of issues"""
        pass

    def get_station_id(self) -> str:
        """Get formatted station identifier"""
        return f"{self.network_code}.{self.station_code}"

    def compute_hash(self, additional_data: str = "") -> int:
        """Compute hash for the solution data"""
        hash_input = (
            f"{len(self.time_vector)}_{len(self.blunders)}_"
            f"{self.auto_x[0] if len(self.auto_x) > 0 else 0}_"
            f"{self.auto_y[0] if len(self.auto_y) > 0 else 0}_"
            f"{self.auto_z[0] if len(self.auto_z) > 0 else 0}_"
            f"{self.time_vector.min() if len(self.time_vector) > 0 else 0}_"
            f"{self.time_vector.max() if len(self.time_vector) > 0 else 0}_"
            f"{additional_data}"
        )
        return crc32(hash_input)

    def create_continuous_time_vector(self) -> None:
        """Create continuous time vectors for plotting"""
        if len(self.time_vector_mjd) > 0:
            ts = np.arange(np.min(self.time_vector_mjd), np.max(self.time_vector_mjd) + 1, 1)
            self.time_vector_cont_mjds = ts
            self.time_vector_cont = np.array([pyDate.Date(mjd=mjd).fyear for mjd in ts])
            self.gaps = np.setdiff1d(self.time_vector_cont_mjds, self.time_vector_mjd)

    def filter_by_distance(self, coordinates: np.ndarray,
                           reference: np.ndarray) -> bool:
        """Filter coordinates by distance from reference"""
        distances = np.sqrt(np.sum(np.square(coordinates - reference), axis=1))
        return distances <= self.max_dist

    def transform_to_local(self) -> np.ndarray:
        """Transform ECEF coordinates to local NEU frame"""
        from pgamit.Utils import ct2lg

        # Compute coordinate differences from reference
        ecef_diff = np.array([
            self.x - self.auto_x[0],
            self.y - self.auto_y[0],
            self.z - self.auto_z[0]
        ])

        # Transform to local frame
        neu = ct2lg(ecef_diff[0], ecef_diff[1], ecef_diff[2],
                    self.lat[0], self.lon[0])

        return np.array(neu)

    def apply_models(self, models: List) -> np.ndarray:
        """Apply external models (velocity, postseismic) to observations"""
        corrected_observations = self.transform_to_local().copy()

        for model in models:
            if hasattr(model, 'eval'):
                model_values = model.eval(self.time_vector)
                corrected_observations -= model_values
                logger.info(f"Applied {model.__class__.__name__} model")

        return corrected_observations


class PPPSolutionData(SolutionData):
    """PPP-specific solution data implementation"""

    def __init__(self, config: ETMConfig):
        super().__init__(config)
        self.soln = 'ppp'
        self.stack_name = 'ppp'
        self.project = 'from_ppp'
        self.rnx_no_ppp: List = []
        # update config to reflect which solution we are working with
        config.solution.soln = 'ppp'
        config.solution.stack_name = 'ppp'
        self.config = config

    def load_data(self, cnn, **kwargs) -> None:
        """Load PPP solutions from database"""

        self._load_ppp_solutions(cnn)
        self._load_excluded_solutions(cnn)
        self._compute_completion_stats(cnn)

        self.create_continuous_time_vector()
        self.hash = self.compute_hash()

    def _load_ppp_solutions(self, cnn) -> None:
        """Load PPP coordinate solutions"""
        ppp_query = '''
            SELECT "X", "Y", "Z", "Year", "DOY" FROM ppp_soln p1
            WHERE p1."NetworkCode" = '%s' AND p1."StationCode" = '%s' 
            ORDER BY "Year", "DOY"
        ''' % (self.network_code, self.station_code)

        solutions = cnn.query_float(ppp_query)

        if not solutions:
            raise ValueError(f"No PPP solutions for {self.get_station_id()}")

        # Filter by distance from reference coordinates
        coordinates = np.array([(s[0], s[1], s[2]) for s in solutions])
        reference = np.array([self.auto_x[0], self.auto_y[0], self.auto_z[0]])

        valid_mask = self.filter_by_distance(coordinates, reference)
        valid_solutions = [s for i, s in enumerate(solutions) if valid_mask[i]]

        if not valid_solutions:
            min_dist = np.sqrt(np.sum(np.square(coordinates - reference), axis=1)).min()
            raise ValueError(
                f"No viable PPP solutions for {self.get_station_id()} "
                f"(minimum distance: {min_dist:.1f}m)"
            )

        # Convert to arrays
        solution_array = np.array(valid_solutions)
        self.x = solution_array[:, 0]
        self.y = solution_array[:, 1]
        self.z = solution_array[:, 2]

        # Create time vectors
        dates_years_doys = solution_array[:, 3:5].astype(int)
        self.date = [pyDate.Date(year=year, doy=doy)
                     for year, doy in dates_years_doys]
        self.time_vector = np.array([date.fyear for date in self.date])
        self.time_vector_mjd = np.array([date.mjd for date in self.date])

        self.solutions = len(valid_solutions)

    def _load_excluded_solutions(self, cnn) -> None:
        """Load list of excluded solutions"""
        excl_query = '''
            SELECT "Year", "DOY" FROM ppp_soln_excl
            WHERE "NetworkCode" = '%s' AND "StationCode" = '%s'
        ''' % (self.network_code, self.station_code)
        self.excluded = cnn.query_float(excl_query)

    def _compute_completion_stats(self, cnn) -> None:
        """Compute completion percentage and missing solution epochs"""
        rnx_query = '''
            SELECT r."ObservationFYear" FROM rinex_proc as r
            LEFT JOIN ppp_soln as p ON 
                r."NetworkCode" = p."NetworkCode" AND
                r."StationCode" = p."StationCode" AND  
                r."ObservationYear" = p."Year" AND
                r."ObservationDOY" = p."DOY"
            WHERE r."NetworkCode" = '%s' AND r."StationCode" = '%s' AND
                p."NetworkCode" IS NULL
        ''' % (self.network_code, self.station_code)

        missing = cnn.query_float(rnx_query)
        self.time_vector_ns = np.array([float(item[0]) for item in missing])

        total_epochs = len(self.time_vector_ns) + len(self.time_vector)
        if total_epochs > 0:
            self.completion = 100.0 - (len(self.time_vector_ns) / total_epochs * 100.0)
        else:
            self.completion = 0.0

    def validate_data(self) -> List[str]:
        """Validate PPP solution data"""
        issues = []

        if self.solutions < self.config.validation.min_solutions_for_etm:
            issues.append(
                f"Insufficient solutions ({self.solutions} < "
                f"{self.config.validation.min_solutions_for_etm})"
            )

        if len(self.x) != len(self.y) or len(self.y) != len(self.z):
            issues.append("Coordinate arrays have different lengths")

        if len(self.time_vector) != len(self.x):
            issues.append("Time array length doesn't match coordinate arrays")

        # Check for reasonable coordinate ranges
        if len(self.x) > 0:
            coord_range = np.ptp([self.x, self.y, self.z])
            if coord_range > 1000:  # 1km seems unreasonable for a single station
                issues.append(f"Coordinate range suspiciously large: {coord_range:.1f}m")

        return issues


class GAMITSolutionData(SolutionData):
    """GAMIT-specific solution data implementation"""

    def __init__(self, stack_name: str, config: ETMConfig):
        super().__init__(config)
        self.soln = 'gamit'
        self.stack_name = stack_name
        self.rnx_no_ppp: List = []
        # update config to reflect which solution we are working with
        config.solution.soln = 'gamit'
        config.solution.stack_name = stack_name
        self.config = config

    def load_data(self, cnn, polyhedrons: Optional[List] = None, **kwargs) -> None:
        """Load GAMIT solutions from database or polyhedron list"""
        if polyhedrons is None:
            polyhedrons = self._load_polyhedrons_from_db(cnn)

        self._load_project_info(cnn)
        self._process_polyhedrons(polyhedrons)
        self._load_missing_solutions(cnn)

        self.create_continuous_time_vector()
        self.hash = self.compute_hash(str(self._get_coordinate_hash(cnn)))

    def _load_polyhedrons_from_db(self, cnn) -> List:
        """Load polyhedrons from stacks table"""
        query = '''
            SELECT "X", "Y", "Z", "Year", "DOY" FROM stacks
            WHERE "name" = '%s' AND "NetworkCode" = '%s' AND "StationCode" = '%s'
            ORDER BY "Year", "DOY", "NetworkCode", "StationCode"
        ''' % (self.stack_name, self.network_code, self.station_code)
        return cnn.query_float(query)

    def _load_project_info(self, cnn) -> None:
        """Load project information"""
        query = '''
            SELECT "Project" FROM stacks 
            WHERE name = '%s' AND "NetworkCode" = '%s' AND "StationCode" = '%s' 
            LIMIT 1
        ''' % (self.stack_name, self.network_code, self.station_code)
        result = cnn.query_float(query, as_dict=True)
        if result:
            self.project = result[0]['Project']

    def _process_polyhedrons(self, polyhedrons: List) -> None:
        """Process polyhedron data into coordinate arrays"""
        if not polyhedrons:
            raise ValueError(f"No GAMIT polyhedrons available for {self.get_station_id()}")

        poly_array = np.array(polyhedrons, dtype=float)

        # Determine if coordinates are absolute or relative
        if np.sqrt(np.sum(np.square(poly_array[0, 0:3]))) > 6.3e3:
            # Absolute coordinates - filter by distance from reference
            reference = np.array([self.auto_x[0], self.auto_y[0], self.auto_z[0]])
            valid_mask = self.filter_by_distance(poly_array[:, 0:3], reference)
        else:
            # Relative coordinates - filter by magnitude
            distances = np.sqrt(np.sum(np.square(poly_array[:, 0:3]), axis=1))
            valid_mask = distances <= self.max_dist

        if not np.any(valid_mask):
            min_dist = np.sqrt(np.sum(np.square(poly_array[:, 0:3] -
                                                np.array([self.auto_x[0], self.auto_y[0], self.auto_z[0]])),
                                      axis=1)).min()
            raise ValueError(
                f"No viable GAMIT solutions for {self.get_station_id()} "
                f"(minimum distance: {min_dist:.1f}m)"
            )

        # Extract valid solutions
        valid_poly = poly_array[valid_mask]
        self.x = valid_poly[:, 0]
        self.y = valid_poly[:, 1]
        self.z = valid_poly[:, 2]

        # Create time vectors
        dates_years_doys = valid_poly[:, 3:5].astype(int)
        self.date = [pyDate.Date(year=year, doy=doy)
                     for year, doy in dates_years_doys]
        self.time_vector = np.array([date.fyear for date in self.date])
        self.time_vector_mjd = np.array([date.mjd for date in self.date])

        self.solutions = len(valid_poly)

    def _load_missing_solutions(self, cnn) -> None:
        """Load epochs with RINEX files but no solutions"""
        query = '''
            SELECT r.* FROM rinex_proc as r
            LEFT JOIN stacks as p ON
                r."NetworkCode" = p."NetworkCode" AND
                r."StationCode" = p."StationCode" AND
                r."ObservationYear" = p."Year" AND
                r."ObservationDOY" = p."DOY" AND
                p."name" = '%s'
            WHERE r."NetworkCode" = '%s' AND r."StationCode" = '%s' AND
                p."NetworkCode" IS NULL
        ''' % (self.stack_name, self.network_code, self.station_code)

        missing = cnn.query(query)
        self.rnx_no_ppp = missing.dictresult()
        self.time_vector_ns = np.array([float(item['ObservationFYear']) for item in self.rnx_no_ppp])

        total_epochs = len(self.time_vector_ns) + len(self.time_vector)
        if total_epochs > 0:
            self.completion = 100.0 - (len(self.time_vector_ns) / total_epochs * 100.0)
        else:
            self.completion = 0.0

    def _get_coordinate_hash(self, cnn) -> float:
        """Get hash of average coordinates for frame change detection"""
        query = '''
            SELECT avg("X") + avg("Y") + avg("Z") AS hash FROM stacks 
            WHERE name = '%s' AND "NetworkCode" = '%s' AND "StationCode" = '%s'
        ''' % (self.stack_name, self.network_code, self.station_code)

        result = cnn.query_float(query, as_dict=True)
        return result[0]['hash'] if result else 0.0

    def validate_data(self) -> List[str]:
        """Validate GAMIT solution data"""
        issues = []

        if self.solutions < self.config.validation.min_solutions_for_etm:
            issues.append(
                f"Insufficient solutions ({self.solutions} < "
                f"{self.config.validation.min_solutions_for_etm})"
            )

        if len(self.x) != len(self.y) or len(self.y) != len(self.z):
            issues.append("Coordinate arrays have different lengths")

        if not self.project:
            issues.append("No project information available")

        return issues