"""
Project: Geode
Date: 5/9/26 1:28 PM
Author: Demian D. Gomez

Module for managing ETM parameters independently of the ETM engine.
This allows reading and writing etm_params table entries without the computational
overhead of instantiating an ETM object.
"""
from typing import Optional, Dict, Any, TYPE_CHECKING
from copy import deepcopy

from geode import dbConnection
from geode.pyDate import Date
from geode.etm.core.etm_config import EtmConfig
from geode.etm.core.type_declarations import SolutionType, PeriodicStatus

if TYPE_CHECKING:
    from geode.etm.core.etm_engine import EtmEngine


class EtmParamsException(Exception):
    pass


type_dict_user = {0: 'MECHANICAL',
                  1: 'CO+POSTSEISMIC',
                  2: 'POSTSEISMIC'}


class EtmParams:
    """
    Class to manage ETM parameters (etm_params table) independently of the ETM engine.

    This class allows reading and writing ETM configuration parameters without
    instantiating an EtmEngine object, which can be computationally expensive.

    Usage:
        # For writing parameters (fast, no ETM needed):
        config = EtmConfig(network_code='igs', station_code='algo', cnn=cnn)
        etm_params = EtmParams(config, cnn)
        etm_params.push_params(params={'object': 'polynomial', 'terms': 2})

        # For reading parameters (requires ETM instance):
        etm = EtmEngine(config, cnn)
        etm_params = EtmParams.from_etm(etm, cnn)
        params = etm_params.pull_params()
    """

    def __init__(self, config: EtmConfig, cnn: Optional[dbConnection.Cnn] = None):
        """
        Initialize EtmParams with configuration and database connection.

        Args:
            config: EtmConfig object with station and solution information
            cnn: Database connection object (required for push_params and pull_params_from_db,
                 not required for pull_params when using from_etm)
        """
        self.config = config
        self.cnn = cnn
        self._etm: Optional['EtmEngine'] = None

    @classmethod
    def from_etm(cls, etm: 'EtmEngine', cnn: Optional[dbConnection.Cnn] = None) -> 'EtmParams':
        """
        Create an EtmParams instance from an existing EtmEngine.

        This allows pull_params to access the ETM's jump_manager and design_matrix.

        Args:
            etm: An EtmEngine instance
            cnn: Database connection object (required for push_params, optional for pull_params)

        Returns:
            EtmParams instance with ETM reference
        """
        instance = cls(etm.config, cnn)
        instance._etm = etm
        return instance

    @property
    def network_code(self) -> str:
        return self.config.network_code

    @property
    def station_code(self) -> str:
        return self.config.station_code

    @property
    def solution_code(self) -> str:
        return self.config.solution.solution_type.code

    def pull_params(self) -> Dict[str, Any]:
        """
        Obtain the current parameters of the station (in json format).

        Note: This method requires an ETM instance. Use EtmParams.from_etm() to create
        an EtmParams object that can pull parameters.

        Returns:
            Dictionary with 'polynomial', 'periodic', and 'jumps' keys

        Raises:
            EtmParamsException: If no ETM instance is available
        """
        if self._etm is None:
            raise EtmParamsException(
                "Cannot pull parameters without an ETM instance. "
                "Use EtmParams.from_etm(etm, cnn) to create an instance with ETM access."
            )

        date = Date(fyear=self._etm.config.modeling.reference_epoch)

        poly_dict = {
            'terms': self._etm.config.modeling.poly_terms,
            'Year': date.year,
            'DOY': date.doy
        }

        jumps = [{
            'Year': jump.p.jump_date.year,
            'DOY': Date(datetime=jump.p.jump_date).doy,
            'action': jump.user_action,
            'fit': jump.fit,
            'type': jump.p.jump_type.description,
            'relaxation': jump.p.relaxation.tolist(),
            'metadata': jump.p.metadata
        } for jump in self._etm.jump_manager.jumps]

        periodic = {}
        for f in self._etm.config.modeling.frequencies:
            funct = self._etm.design_matrix.get_periodic()

            if f is not None and f in funct.p.frequencies:
                status = self._etm.config.modeling.periodic_status
            else:
                status = PeriodicStatus.UNABLE_TO_FIT

            periodic[1 / f] = status

        return {
            'polynomial': poly_dict, 'periodic': periodic, 'jumps': jumps,
            'copy_params': self._should_copy_to_others()
        }

    def pull_params_from_db(self) -> Dict[str, Any]:
        """
        Pull parameters directly from the database without needing an ETM instance.

        This is useful when you need to read the stored parameters but don't need
        the full ETM computation (e.g., to display current settings).

        Returns:
            Dictionary with 'polynomial', 'periodic', and 'jumps' keys

        Raises:
            EtmParamsException: If no database connection is available
        """
        if self.cnn is None:
            raise EtmParamsException(
                "Database connection required for pull_params_from_db. "
                "Pass cnn to the EtmParams constructor."
            )

        result = {
            'polynomial': None,
            'periodic': {},
            'jumps': [],
            'copy_params': self._should_copy_to_others()
        }

        # Get polynomial parameters
        poly_rows = self.cnn.query_float(
            '''SELECT terms, "Year", "DOY" FROM etm_params
               WHERE "NetworkCode" = '%s' AND "StationCode" = '%s'
               AND object = 'polynomial' AND soln = '%s' LIMIT 1'''
            % (self.network_code, self.station_code, self.solution_code),
            as_dict=True
        )
        if poly_rows:
            row = poly_rows[0]
            result['polynomial'] = {
                'terms': int(row['terms']) if row['terms'] else None,
                'Year': int(row['Year']) if row['Year'] else None,
                'DOY': int(row['DOY']) if row['DOY'] else None
            }

        # Get periodic parameters
        periodic_rows = self.cnn.query_float(
            '''SELECT frequencies FROM etm_params
               WHERE "NetworkCode" = '%s' AND "StationCode" = '%s'
               AND object = 'periodic' AND soln = '%s' LIMIT 1'''
            % (self.network_code, self.station_code, self.solution_code),
            as_dict=True
        )
        if periodic_rows and periodic_rows[0].get('frequencies'):
            freqs = periodic_rows[0]['frequencies']
            for f in freqs:
                result['periodic'][1 / f] = PeriodicStatus.ADDED_BY_USER

        # Get jump parameters
        jump_rows = self.cnn.query_float(
            '''SELECT "Year", "DOY", action, jump_type, relaxation FROM etm_params
               WHERE "NetworkCode" = '%s' AND "StationCode" = '%s'
               AND object = 'jump' AND soln = '%s' ORDER BY "Year", "DOY"'''
            % (self.network_code, self.station_code, self.solution_code),
            as_dict=True
        )
        for row in jump_rows:
            result['jumps'].append({
                'Year': int(row['Year']),
                'DOY': int(row['DOY']),
                'action': row['action'],
                'jump_type': int(row['jump_type']) if row['jump_type'] else 0,
                'relaxation': row['relaxation'] if row['relaxation'] else []
            })

        return result

    def push_params(self,
                    params: Optional[Dict[str, Any]] = None,
                    reset_polynomial: bool = False,
                    reset_periodic: bool = False,
                    reset_jumps: bool = False,
                    copy_params: Optional[bool] = None) -> None:
        """
        Push parameters to the etm_params table.

        This method does not require an ETM instance - it works directly with the
        database using only the configuration.

        Args:
            params: Dictionary containing parameter objects. Supported formats:
                {'object': 'polynomial', 'terms': int, 'Year': int, 'DOY': int}
                    - terms: number of polynomial terms (e.g., 2, 3, 4)
                    - Year/DOY: reference date (both required if either is set)
                {'object': 'periodic', 'frequencies': list[float]}
                    - frequencies: list of periods in days to fit (empty list = none)
                {'object': 'jump', 'Year': int, 'DOY': int, 'action': str,
                 'jump_type': int, 'relaxation': list}
                    - Year/DOY: discontinuity date (mandatory)
                    - action: '+' to add, '-' to remove
                    - jump_type: 0=mechanical, 1=co+postseismic, 2=postseismic only
                    - relaxation: required when jump_type > 0
            reset_polynomial: If True, delete existing polynomial params
            reset_periodic: If True, delete existing periodic params
            reset_jumps: If True, delete existing jump params
            copy_params: If True, copy parameters to all solution types.
                        If False, remove the copy_params flag.
                        If None, don't modify the copy_params setting.

        Raises:
            EtmParamsException: If validation fails or no database connection
        """
        if self.cnn is None:
            raise EtmParamsException(
                "Database connection required for push_params. "
                "Pass cnn to the EtmParams constructor."
            )

        # For polynomial and periodic, trigger reset when we push new params
        # since we need to get rid of old records (only one allowed per station/soln)
        if params:
            if params.get('object') == 'polynomial':
                reset_polynomial = True
            elif params.get('object') == 'periodic':
                reset_periodic = True

        # Validate and prepare params if provided
        if params:
            params = self._validate_and_prepare_params(params)

        # Handle resets (also resets other solutions if copy_params is enabled)
        if reset_polynomial:
            self._delete_params('polynomial')

        if reset_periodic:
            self._delete_params('periodic')

        if reset_jumps:
            self._delete_params('jump')

        # Handle copy_params flag
        if copy_params is not None:
            self._handle_copy_params(copy_params)

        # Insert params if provided
        if params:
            self._insert_params(params)

    def _validate_and_prepare_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters and add station/solution identifiers."""
        params = deepcopy(params)

        if params['object'] == 'polynomial':
            self._validate_polynomial_params(params)
        elif params['object'] == 'periodic':
            self._validate_periodic_params(params)
        elif params['object'] == 'jump':
            self._validate_jump_params(params)

        # Add station and solution identifiers
        params['NetworkCode'] = self.network_code
        params['StationCode'] = self.station_code
        params['soln'] = self.solution_code

        return params

    def _validate_polynomial_params(self, params: Dict[str, Any]) -> None:
        """Validate polynomial parameters."""
        # Check Year/DOY pairing
        has_year = 'Year' in params
        has_doy = 'DOY' in params

        if has_year != has_doy:
            raise EtmParamsException(
                'Both parameters Year and DOY must be specified if either one is set'
            )

        if has_year and has_doy:
            if params['Year'] is not None and not isinstance(params['Year'], int):
                raise EtmParamsException('Parameter Year must be of type int')
            if params['Year'] is not None and not isinstance(params['DOY'], int):
                raise EtmParamsException('Parameter DOY must be of type int')
            # Validate date
            if params['Year'] is not None and params['DOY'] is not None:
                _ = Date(year=params['Year'], doy=params['DOY'])

        # Validate terms
        if not isinstance(params.get('terms'), int) or params['terms'] <= 0:
            raise EtmParamsException('Parameter terms must be of type int > 0')

    def _validate_periodic_params(self, params: Dict[str, Any]) -> None:
        """Validate periodic parameters and convert periods to frequencies."""
        for i, period in enumerate(params.get('frequencies', [])):
            if period <= 0:
                raise EtmParamsException(
                    'Cannot insert negative or zero periods for periodic components'
                )
            # Convert period (days) to frequency (1/days)
            params['frequencies'][i] = 1 / period

    def _validate_jump_params(self, params: Dict[str, Any]) -> None:
        """Validate jump parameters."""
        if params.get('jump_type', 0) < 0:
            raise EtmParamsException('jump_type must be >= 0')

        action = params.get('action')
        jump_type = params.get('jump_type', 0)

        if action == '+' and jump_type > 0 and 'relaxation' not in params:
            raise EtmParamsException(
                'Relaxation parameters needed when jump_type > 0 and action = +'
            )

        if 'relaxation' in params:
            relaxation = params['relaxation']
            if relaxation:
                for r in relaxation:
                    if not isinstance(r, (float, int)):
                        raise EtmParamsException('Relaxation parameters must be float type')
                    if r <= 0:
                        raise EtmParamsException('Relaxation parameters must be > 0')
            elif jump_type > 0:
                raise EtmParamsException('At least one relaxation needed for jump_type > 0')

    def _delete_params(self, object_type: str, soln: Optional[str] = None,
                       respect_copy_params: bool = True) -> None:
        """Delete parameters of a specific type.

        Args:
            object_type: Type of parameter to delete ('polynomial', 'periodic', 'jump')
            soln: Solution code to delete from. If None, uses current solution.
            respect_copy_params: If True and soln is None, also delete from other
                                solutions when copy_params is enabled for this station.
        """
        target_soln = soln or self.solution_code

        self.cnn.delete(
            'etm_params',
            NetworkCode=self.network_code,
            StationCode=self.station_code,
            soln=target_soln,
            object=object_type
        )

        # If no explicit soln was provided and copy_params is enabled,
        # also delete from other solutions
        if soln is None and respect_copy_params and self._should_copy_to_others():
            for other_soln in SolutionType:
                if other_soln.code != self.solution_code:
                    self.cnn.delete(
                        'etm_params',
                        NetworkCode=self.network_code,
                        StationCode=self.station_code,
                        soln=other_soln.code,
                        object=object_type
                    )

    def _copy_params_to_solution(self, target_soln: str) -> None:
        """Copy all parameters from current solution type to target solution type."""
        # First delete existing params for target solution
        for obj_type in ('polynomial', 'periodic', 'jump'):
            self._delete_params(obj_type, soln=target_soln)

        # Get all params for current solution
        source_params = self.cnn.query_float(
            '''SELECT object, terms, frequencies, jump_type, relaxation, "Year", "DOY", action
               FROM etm_params
               WHERE "NetworkCode" = '%s' AND "StationCode" = '%s' AND soln = '%s'
               AND object != 'copy_par' '''
            % (self.network_code, self.station_code, self.solution_code),
            as_dict=True
        )

        # Insert each param with the target solution code
        for row in source_params:
            insert_params = {
                'NetworkCode': self.network_code,
                'StationCode': self.station_code,
                'soln': target_soln,
                'object': row['object']
            }
            # Add non-null fields
            for field in ('terms', 'frequencies', 'jump_type', 'relaxation', 'Year', 'DOY', 'action'):
                if row.get(field) is not None:
                    insert_params[field] = row[field]

            self.cnn.insert('etm_params', **insert_params)

    def _handle_copy_params(self, copy_params: bool) -> None:
        """Handle the copy_params flag - copy or remove parameters for other solutions."""
        if copy_params:
            # Copy parameters to all other solution types
            for soln in SolutionType:
                if soln.code != self.solution_code and soln.code != SolutionType.DRA.code:
                    self._copy_params_to_solution(soln.code)

            # Add the copy_params flag
            self.cnn.insert('etm_params',
                           object='copy_par',
                           NetworkCode=self.network_code,
                           StationCode=self.station_code,
                           soln='all',
                           action='T')
        else:
            # Remove the copy_params flag
            self.cnn.delete('etm_params',
                           object='copy_par',
                           NetworkCode=self.network_code,
                           StationCode=self.station_code,
                           soln='all',
                           action='T')

    def _should_copy_to_others(self) -> bool:
        """Check if the copy_params flag is set for this station."""
        try:
            result = self.cnn.get(
                'etm_params',
                {
                    'object': 'copy_par',
                    'NetworkCode': self.network_code,
                    'StationCode': self.station_code,
                    'soln': 'all'
                },
                ['action']
            )
            return len(result) > 0
        except dbConnection.DatabaseError:
            return False

    def _insert_params(self, params: Dict[str, Any]) -> None:
        """Insert parameters into the database, handling copy_to_others."""
        copy_to_others = self._should_copy_to_others()

        if params['object'] == 'jump':
            # For jumps, first remove any existing jump at this date
            qpar = {k: v for k, v in params.items()
                   if k not in ('action', 'relaxation', 'jump_type')}

            self.cnn.delete('etm_params', **qpar)

            if copy_to_others:
                for soln in SolutionType:
                    if soln.code != self.solution_code:
                        qpar_copy = deepcopy(qpar)
                        qpar_copy['soln'] = soln.code
                        self.cnn.delete('etm_params', **qpar_copy)

        # Insert the new parameters
        self.cnn.insert('etm_params', **params)

        # Copy to other solutions if needed
        if copy_to_others:
            for soln in SolutionType:
                if soln.code != self.solution_code and soln.code != SolutionType.DRA.code:
                    # For polynomial/periodic, delete old params by object type only
                    # (not by specific values like 'terms') before inserting new ones
                    if params['object'] in ('polynomial', 'periodic'):
                        self._delete_params(params['object'], soln=soln.code)

                    params_copy = deepcopy(params)
                    params_copy['soln'] = soln.code
                    self.cnn.insert('etm_params', **params_copy)

    def is_copy_params_enabled(self) -> bool:
        """Check if the copy_params feature is enabled for this station."""
        return self._should_copy_to_others()


# Backward compatibility: standalone functions that delegate to the class
def pull_params(etm: 'EtmEngine') -> Dict[str, Any]:
    """
    Legacy function to obtain current parameters of the station.

    Prefer using EtmParams.from_etm(etm, cnn).pull_params() for new code.
    """
    # Create instance and delegate - note: cnn not needed for pull_params with ETM
    params_manager = EtmParams.from_etm(etm, None)
    params_manager._etm = etm  # Ensure ETM is set
    return params_manager.pull_params()


def push_params(config: EtmConfig, cnn: dbConnection.Cnn,
                params: Optional[Dict[str, Any]] = None,
                reset_polynomial: bool = False,
                reset_periodic: bool = False,
                reset_jumps: bool = False,
                copy_params: Optional[bool] = None) -> None:
    """
    Legacy function to push parameters to the database.

    Prefer using EtmParams(config, cnn).push_params() for new code.
    """
    params_manager = EtmParams(config, cnn)
    params_manager.push_params(
        params=params,
        reset_polynomial=reset_polynomial,
        reset_periodic=reset_periodic,
        reset_jumps=reset_jumps,
        copy_params=copy_params
    )
