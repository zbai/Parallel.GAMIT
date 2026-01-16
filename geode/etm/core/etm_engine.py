from enum import IntEnum
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional, Union, List
import logging
import json
import numpy as np
import copy
import platform
import os

from importlib.metadata import version, PackageNotFoundError

from .type_declarations import PeriodicStatus, JumpType

try:
    VERSION = str(version("geode"))
except PackageNotFoundError:
    # package is not installed
    VERSION = 'NOT_AVAIL'
    pass

logger = logging.getLogger(__name__)

# app
from ...dbConnection import Cnn
from ...Utils import file_write, load_json
from ...pyDate import Date

from ...metadata.station_info import StationInfoRecord
from ..data.solution_data import SolutionData
from ..data.etm_database import load_parameters_db, save_parameters_db
from ..core.etm_config import EtmConfig, SolutionOptions
from ..core.type_declarations import EtmSolutionType, FitStatus, SolutionType
from ..core.data_classes import LeastSquares, AdjustmentResults
from ..least_squares.design_matrix import DesignMatrix
from ..least_squares.least_squares import EtmFit, AdjustmentModels
from ..etm_functions.polynomial import PolynomialFunction
from ..etm_functions.periodic import PeriodicFunction
from ..etm_functions.stochastic_signal import StochasticSignal
from ..core.jump_manager import JumpManager
from ..core.logging_config import setup_etm_logging
from ..visualization.data_prep import PlotDataPreparer
from ..visualization.etm_plotting import EtmPlotter


def enum_dict_factory(field_list):
    """Custom dict factory that handles enums with descriptions"""
    result = {}
    for key, value in field_list:
        # Check if value is an IntEnum with a description property
        if isinstance(value, IntEnum) and hasattr(value, 'description'):
            result[key] = {
                'value': value.value,
                'description': value.description
            }
        # Handle lists that might contain enums
        elif isinstance(value, list):
            result[key] = [
                {'value': v.value, 'description': v.description}
                if isinstance(v, IntEnum) and hasattr(v, 'description')
                else v
                for v in value
            ]
        elif isinstance(value, (LeastSquares, SolutionOptions)):
            for k, v in value:
                result[key] = {}
                # Check if value is an IntEnum with a description property
                if isinstance(v, IntEnum) and hasattr(v, 'description'):
                    result[key][k] = {
                        'value': v.value,
                        'description': v.description
                    }
        else:
            result[key] = value
    return result


class EtmEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and float rounding"""

    def __init__(self, *args, round_digits=6, no_round_fields=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_digits = round_digits
        self.no_round_fields = no_round_fields or []
        self._in_rounding_phase = False

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Just convert to list, don't round here
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, StationInfoRecord):
            return obj.to_json()
        if isinstance(obj, datetime):
            date = Date(datetime=obj)
            return {'year': date.year,
                    'month': date.month,
                    'day': date.day,
                    'mjd': date.mjd,
                    'doy': date.doy,
                    'hour': date.hour,
                    'minute': date.minute,
                    'second': date.second}
        if isinstance(obj, Date):
            if obj.from_stninfo:
                return {'stninfo': str(self)}
            else:
                return {'year': obj.year,
                        'month': obj.month,
                        'day': obj.day,
                        'mjd': obj.mjd,
                        'doy': obj.doy,
                        'hour': obj.hour,
                        'minute': obj.minute,
                        'second': obj.second}
        return super().default(obj)

    def iterencode(self, obj, _one_shot=False):
        """Override iterencode to apply rounding after numpy conversion"""
        if not self._in_rounding_phase:
            # First pass: convert all numpy types to native Python types
            self._in_rounding_phase = True
            temp_json = ''.join(super().iterencode(obj, _one_shot))
            self._in_rounding_phase = False

            # Parse back to Python objects
            intermediate = json.loads(temp_json)

            # Apply rounding with path tracking
            rounded = self._round_floats(intermediate, path=[])

            # Encode the rounded result
            for chunk in super().iterencode(rounded, _one_shot):
                yield chunk
        else:
            # Already in rounding phase, just do normal encoding
            for chunk in super().iterencode(obj, _one_shot):
                yield chunk

    def _round_floats(self, obj, path):
        """Recursively round all floats in nested structures, except excluded fields"""
        # Check if any key in the current path is in no_round_fields
        if any(key in self.no_round_fields for key in path):
            return obj

        if isinstance(obj, float):
            return round(obj, self.round_digits)
        elif isinstance(obj, dict):
            return {key: self._round_floats(value, path=path + [key]) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._round_floats(item, path=path) for item in obj]
        return obj


class EtmEngineException(Exception):
    pass


class EtmEngine:
    """Core mathematical engine for ETM processing, separated from business logic"""

    def __init__(self, config: EtmConfig,
                 cnn: Cnn = None, silent: bool = False, **kwargs):

        setup_etm_logging(level=logging.CRITICAL if silent else logging.INFO)

        self.config = config

        if config.solution.solution_type == SolutionType.DRA:
            # DRA, turn off models
            self.config.modeling.fit_metadata_jumps = False
            self.config.modeling.fit_generic_jumps = False
            self.config.modeling.fit_earthquakes = False
            self.config.modeling.fit_auto_detected_jumps = False
            # force zero auto coordinates in DRA mode
            self.config.metadata.auto_x = np.array([0])
            self.config.metadata.auto_y = np.array([0])
            self.config.metadata.auto_z = np.array([0])

        # determine the type of object within the SolutionData class
        self.solution_data = SolutionData.create_instance(config)

        # load the data from connection or json file
        self.solution_data.load_data(cnn, **kwargs)

        # check how many solutions we have available. If less than
        # 100 (most likely campaign) disable metadata jumps
        logger.info(f'Observation count is {self.solution_data.solutions}')
        if self.solution_data.solutions < 100:
            logger.info('Disabling metadata jumps because solution count < 100')
            self.config.modeling.fit_metadata_jumps = False

        # @todo: evaluate if mask should be applied here or inside solution_data
        mask = self.config.modeling.get_observation_mask(self.solution_data.time_vector)

        # set the basic functions
        polynomial = PolynomialFunction(config, time_vector=self.solution_data.time_vector)
        periodic = PeriodicFunction(config, time_vector=self.solution_data.time_vector)

        self.jump_manager = JumpManager(self.solution_data, config)
        self.jump_manager.build_jump_table(self.solution_data.time_vector[mask],
                                           self.solution_data.transform_to_local())

        if config.solution.solution_type == SolutionType.DRA:
            # no periodic for DRA
            self.design_matrix = DesignMatrix(config,
                                              self.solution_data.time_vector,
                                              [polynomial] + self.jump_manager.get_jump_functions())
        else:
            self.design_matrix = DesignMatrix(config,
                                              self.solution_data.time_vector,
                                              [polynomial, periodic] + self.jump_manager.get_jump_functions())

        # if adjustment strategy is LSQ_COLLOCATION, add the stochastic function to the design matrix
        # (although it is not used during the fit)
        if self.config.modeling.least_squares_strategy.adjustment_model == AdjustmentModels.LSQ_COLLOCATION:
            self.design_matrix.functions.append(
                StochasticSignal(self.config, self.solution_data.time_vector_cont_mjd))

        self.fit = EtmFit(self.config, self.design_matrix, self.solution_data.hash)

    def run_adjustment(self, try_loading_db=True, try_save_to_db=True,
                       cnn: Optional[Cnn] = None, force_computation=False) -> None:
        """Run the iterative least squares adjustment"""
        if self.config.json_file is None:
            if try_loading_db and cnn and not force_computation:
                # get mask for time vector: only needed for stochastic signal in load_parameters_db!
                mask = self.config.modeling.get_observation_mask(self.solution_data.time_vector)

                success = load_parameters_db(self.config, cnn, self.fit, self.solution_data.transform_to_local(),
                                             self.solution_data.time_vector_mjd[mask])
                logger.info(f'Loading parameters from database: {success}')
            else:
                success = False
        else:
            if not force_computation:
                # information in the json, see if required data is present
                success = self.load_from_json(self.config.json_file)
            else:
                success = False

        if not success:
            run_time = self.fit.run_fit(self.solution_data, self.design_matrix)
            if cnn and try_save_to_db and self.config.modeling.status == FitStatus.POSTFIT:
                save_parameters_db(cnn, self.fit)
            logger.info(f'Estimated parameters in {run_time} seconds')

    def load_from_json(self, json_: Union[str, dict] = None) -> bool:
        """
        load ETM solution from json file
        """
        logger.info('Loading etm from json file')
        data = load_json(json_)

        if 'raw_results' in data.keys():
            self.fit.results = []
            for i in range(3):
                self.fit.results.append(AdjustmentResults(**data['raw_results'][i]))

            # need to check if the design matrix that the etm built is compatible with the design matrix
            # in the json file. If the json has a deactivated function, it will be in the design matrix
            # but not in the json file
            for f in [ff for ff in self.design_matrix.functions if ff.p.object == 'jump']:
                found = False
                for func in [ff for ff in data['functions'] if ff['object'] == 'jump']:
                    if Date(**func['jump_date']) == f.p.jump_date:
                        found = True
                # if we did not find the jump in the json, deactivate it because it
                # was deactivated before saving the file
                if not found:
                    f.fit = False
                    # reassign the column indices
                    self.design_matrix._assign_column_indices()

            self.fit.load_results_to_functions()

            return True
        else:
            return False

    def save_etm(self,
                 filename: str = None,
                 dump_functions: bool = True,
                 dump_raw_results: bool = False,
                 dump_observations: bool = False,
                 dump_design_matrix: bool = False,
                 dump_model: bool = False) -> Dict:

        dm = self.design_matrix
        mask = self.config.modeling.get_observation_mask(self.solution_data.time_vector)
        # create a station_meta copy to replace the StationInfoRecord instances
        station_meta = copy.deepcopy(self.config.metadata)
        station_meta.station_information = [item.to_json() for item in station_meta.station_information]

        # dump continuous model if requested
        if dump_model:
            model_values = self.fit.get_time_continuous_model(self.solution_data.time_vector_cont)
        else:
            model_values = None

        functions = [asdict(funct.p, dict_factory=enum_dict_factory) for funct in dm.functions if funct.fit]

        etm_dump = {
            "network_code": self.config.network_code,
            "station_code": self.config.station_code,
            "station_meta": asdict(station_meta),
            "solution_options": asdict(self.config.solution, dict_factory=enum_dict_factory),
            "modeling_params": asdict(self.config.modeling, dict_factory=enum_dict_factory),
            "raw_results": [asdict(self.fit.results[0]),
                            asdict(self.fit.results[1]),
                            asdict(self.fit.results[2])] if dump_raw_results else None,
            "functions": functions if dump_functions else None,
            "observations": asdict(self.solution_data.coordinates) if dump_observations else None,
            "model": {'model_cont': model_values,
                      'time_vector_fyear': self.solution_data.time_vector_cont,
                      'time_vector_mjd': self.solution_data.time_vector_cont_mjd
                      } if dump_model else None,
            "covariance": self.fit.covar.tolist(),
            "design_matrix": dm.matrix if dump_design_matrix else None,
            "data_model_window": mask if dump_design_matrix else None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "localhost": platform.node(),
            "version": VERSION
        }

        if filename:
            if not os.path.basename(filename):
                filename = os.path.join(filename, self.config.build_filename()) + '.json'
            else:
                dirs = os.path.dirname(filename)
                file = os.path.basename(filename).split('.')
                filename = os.path.join(dirs, file[0]) + '.json'

            #import bson

            file_write(filename,
                       json.dumps(etm_dump, indent=4, sort_keys=False, cls=EtmEncoder,
                                  round_digits=6, no_round_fields=['covariance', 'parameter_sigmas',
                                                                   'parameters', 'covariance_matrix',
                                                                   'lat', 'lon', 'params', 'sigmas']))

            #binary_data = bson.encode(etm_dump, cls)
            #with open(filename + '.bson', 'wb') as f:
            #    f.write(binary_data)

        return etm_dump

    def plot(self) -> Union[str, None]:

        plotter = EtmPlotter(self.config)
        plot_data = PlotDataPreparer(self.config).prepare_time_series_data(self.solution_data, self.fit)

        return plotter.plot_time_series(plot_data)

    def plot_hist(self) -> Union[str, None]:

        if self.config.modeling.status == FitStatus.POSTFIT:
            plotter = EtmPlotter(self.config)
            plot_data = PlotDataPreparer(self.config).prepare_time_series_data(self.solution_data, self.fit)

            return plotter.plot_histogram(plot_data)
        else:
            return None

    def get_position(self, dates: Union[Date, List[Date]],
                     etm_solution: EtmSolutionType = EtmSolutionType.MODEL) -> Dict:
        """
        use computed model to obtain a station position
        """

        if isinstance(dates, Date):
            dates = [dates]

        sigma_h = self.config.modeling.sigma_floor_h
        sigma_v = self.config.modeling.sigma_floor_v
        limit = self.config.modeling.least_squares_strategy.sigma_filter_limit

        model = [np.array([]), np.array([]), np.array([])]
        model_ecef = [np.array([]), np.array([]), np.array([])]
        stack_name = self.solution_data.stack_name.upper()

        # find the model value, if fit done
        if self.config.modeling.status == self.config.modeling.status.POSTFIT:
            for i in range(3):
                model[i] = (self.design_matrix.alternate_time_vector(
                    np.array([date.fyear for date in dates])) @ self.fit.results[i].parameters)

            # transform back to ecef
            model_ecef = np.array(self.solution_data.transform_to_ecef(model))

        if etm_solution == EtmSolutionType.OBSERVATION:
            # find the date of the request
            mask = np.isin(self.solution_data.time_vector_mjd, [date.mjd for date in dates])
            i = np.where(mask)[0]

            if i.size:
                # solution found, save it
                position = np.array([self.solution_data.x[i], self.solution_data.y[i], self.solution_data.z[i]])
                sigmas = np.array([result.residuals[i] for result in self.fit.results])

                # check if model is not empty
                if model_ecef[0].size:
                    if np.all(self.fit.outlier_flags[i]):
                        source = stack_name + ' with ETM solution: good'
                    else:
                        # filter coordinate because it was an outlier
                        position = model_ecef
                        sigmas = sigmas * limit
                        source = stack_name + ' with ETM solution: filtered'
                else:
                    source = stack_name + ' solution, but no ETM'
            else:
                if model_ecef[0].size:
                    position = model_ecef
                    sigmas = np.array([[result.wrms] * len(dates) for result in self.fit.results]) * limit
                    source = 'No ' + stack_name + ' solution for requested dates: ETM'
                else:
                    source = 'No ' + stack_name + ' solution for requested dates, no ETM: mean coordinate'
                    position = np.array([[self.solution_data.x.mean()] * len(dates),
                                         [self.solution_data.y.mean()] * len(dates),
                                         [self.solution_data.z.mean()] * len(dates)])

                    sigmas = np.array([[self.solution_data.x.std()] * len(dates),
                                       [self.solution_data.y.std()] * len(dates),
                                       [self.solution_data.z.std()] * len(dates)])
        else:
            if model_ecef.size:
                source = 'ETM solution requested'
                position = model_ecef
                sigmas = np.array([[result.wrms] * len(dates) for result in self.fit.results])
            else:
                raise Exception('Model requested by no ETM was found for ' + self.config.get_station_id())

        if self.config.modeling.status == self.config.modeling.status.POSTFIT:
            # get the velocity of the site
            for funct in self.design_matrix.functions:
                if funct.p.object == 'polynomial':
                    if np.sqrt(np.sum(np.square([p[1] for p in funct.p.params]))) > 0.2:
                        # fast moving station! bump up the sigma floor
                        sigma_h = 99.9
                        sigma_v = 99.9
                        source += '. fast moving station, bumping up sigmas'
                    break

        # apply floor sigmas
        sigmas = np.sqrt(np.square(sigmas) +
                         np.square(np.array(
                             [[sigma_h] * len(dates), [sigma_h] * len(dates), [sigma_v] * len(dates)])))

        return {'position': position.tolist(), 'source': source, 'sigmas': sigmas.tolist()}

    def query_jump(self, date: Date):
        """return any jumps that may have occurred in the date being requested"""
        for jump in self.design_matrix.functions:
            if jump.p.object == 'jump' and Date(datetime=jump.p.jump_date) == date and jump.fit:
                return Date(datetime=jump.p.jump_date)

        return None

    def pull_params(self):
        """
        method to obtain the current parameters of the station (in json format)
        periodic returns a dictionary with the periods as keys and a str following the description
        in periodic_status_dict
        """
        date = Date(fyear=self.config.modeling.reference_epoch)

        poly_dict = {'terms': self.config.modeling.poly_terms,
                     'Year': date.year,
                     'DOY' : date.doy}

        jumps = [{'Year': jump.p.jump_date.year,
                  'DOY': Date(datetime=jump.p.jump_date).doy,
                  'action': jump.user_action,
                  'fit': jump.fit,
                  'type': jump.p.jump_type,
                  'relaxation': jump.p.relaxation.tolist(),
                  'metadata': jump.p.metadata} for jump in self.jump_manager.jumps]

        periodic = {}
        # create a dictionary with keys that equal the 1/f (period) requested or automatically added
        for f in self.config.modeling.frequencies:
            funct = self.design_matrix.get_periodic()

            if f is not None and f in funct.p.frequencies:
                # if in the list it is either automatic or requested by user, so pass
                # the status from the modeling configuration
                status = self.config.modeling.periodic_status
            else:
                # if not present maybe it was not possible to fit
                status = PeriodicStatus.UNABLE_TO_FIT

            periodic[1 / f] = status

        return {'polynomial': poly_dict, 'periodic': periodic, 'jumps': jumps}

    def push_params(self, cnn, params=None, reset_polynomial=False, reset_periodic=False, reset_jumps=False):
        """
        cnn             : database connection object
        params          :
            a dictionary containing the following keys, for each of the corresponding objects
                {'object': 'polynomial', 'terms': int, 'Year': int, 'DOY': int}
            terms: sets the number of terms to use in the polynomial (eg. 2, 3, 4, etc)
            Year: sets the reference year part of the date for the site (if passed, then DOY is needed).
                It can be = None or not passed
            DOY: sets the reference day of year part of the date for the site (if passed,
                then year is needed). It can be = None or not passed
            {'object': 'periodic', 'frequencies': list[float]}
                frequencies: a list of integers of the frequencies to fit. This value must be
                passed as days. If no periodic terms are requested, then pass [].
            {'object': 'jump', 'Year': int, 'DOY': int, 'action': str, 'jump_type': int, 'relaxation': list}
                Year: sets the year part of the discontinuity date. Mandatory.
                DOY: sets the day of year part of the discontinuity date. Mandatory
                action: two possible values, '-' or '+' to add or remove a jump. Mandatory.
                jump_type: jump type, can be 0 for mechanical, 1 for co+postseismic and 2 for postseismic
                only, see type_dict_user
            relaxation: mandatory when type is > 0, list of floats with the relaxation times in years
        reset_polynomial: rest
        reset_periodic  : rest
        reset_jumps     : rest

        Raises:
            EtmEngineException
        """

        # for polynomial and periodic, trigger reset anyway since we need to get rid of the old records
        if params:
            if params['object'] == 'polynomial':
                reset_polynomial = True
                # sanity checks
                if 'Year' in params.keys() and 'DOY' in params.keys():
                    if params['Year'] is not None and type(params['Year']) is not int:
                        raise EtmEngineException('Parameter Year must be of type int')
                    if params['Year'] is not None and type(params['DOY']) is not int:
                        raise EtmEngineException('Parameter DOY must be of type int')
                elif ('Year' in params.keys() and 'DOY' not in params.keys()) or \
                        ('Year' not in params.keys() and 'DOY' in params.keys()):
                    raise EtmEngineException('Both parameters Year and DOY must be specified if either one is set')

                # check that terms is an integer > 0
                if type(params['terms']) is not int or params['terms'] <= 0:
                    raise EtmEngineException('Parameter terms must be of type int > 0')

                # check that the date is valid
                if 'Year' in params.keys() and 'DOY' in params.keys():
                    _ = Date(year=params['Year'], doy=params['DOY'])

            elif params['object'] == 'periodic':
                reset_periodic = True

                # check that frequencies are not negative
                for i, f in enumerate(params['frequencies']):
                    if f <= 0:
                        raise EtmEngineException('Cannot insert negative frequencies for periodic components')
                    else:
                        params['frequencies'][i] = 1/f

            # add the fields for station and network
            params['NetworkCode'] = self.config.network_code
            params['StationCode'] = self.config.station_code
            params['soln'] = self.solution_data.config.solution.solution_type.code

        if reset_polynomial:
            # reset to default
            cnn.delete('etm_params', NetworkCode=self.config.network_code, StationCode=self.config.station_code,
                       soln=self.solution_data.config.solution.solution_type.code, object='polynomial')

        if reset_periodic:
            # reset to default
            cnn.delete('etm_params', NetworkCode=self.config.network_code, StationCode=self.config.station_code,
                       soln=self.solution_data.config.solution.solution_type.code, object='periodic')

        if reset_jumps:
            # reset to default
            cnn.delete('etm_params', NetworkCode=self.config.network_code, StationCode=self.config.station_code,
                       soln=self.solution_data.config.solution.solution_type.code, object='jump')

        if params:
            if params['object'] == 'jump':

                if params['jump_type'] < 0:
                    raise EtmEngineException('jump_type must be >= 0')

                if params['action'] == '+' and params['jump_type'] > 0 and 'relaxation' not in params.keys():
                    raise EtmEngineException('Relaxation parameters needed when jump_type > 0 and action = +')

                if 'relaxation' in params.keys():
                    if len(params['relaxation']):
                        for r in params['relaxation']:
                            if type(r) not in (float, int):
                                raise EtmEngineException('Relaxation parameters must be float type')
                            if r <= 0:
                                raise EtmEngineException('Relaxation parameters must be > 0')
                    elif not len(params['relaxation']) and params['jump_type'] > 0:
                        raise EtmEngineException('At least one relaxation needed for jump_type > 0')

                # query params to find the jump
                qpar = copy.deepcopy(params)
                qpar = {k: v for k, v in qpar.items() if k not in ('action', 'relaxation', 'jump_type')}

                # remove existing parameters
                cnn.delete('etm_params', **qpar)

            # insert the parameters passed to the database
            cnn.insert('etm_params', **params)