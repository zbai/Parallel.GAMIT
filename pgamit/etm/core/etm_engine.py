from enum import IntEnum, auto
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional
import logging
import json
import numpy as np
import copy
import platform
import os

from importlib.metadata import version, PackageNotFoundError

from etm.data.etm_database import save_parameters_db

try:
    VERSION = str(version("pgamit"))
except PackageNotFoundError:
    # package is not installed
    VERSION = 'NOT_AVAIL'
    pass

logger = logging.getLogger(__name__)

# app
from dbConnection import Cnn
from pgamit.Utils import file_write
from pgamit.pyDate import Date
from pgamit.pyStationInfo import StationInfoRecord
from pgamit.etm.data.solution_data import GAMITSolutionData, PPPSolutionData
from pgamit.etm.data.etm_database import load_parameters_db
from pgamit.etm.core.etm_config import EtmConfig
from pgamit.etm.core.type_declarations import EtmSolutionType, SolutionType
from pgamit.etm.least_squares.design_matrix import DesignMatrix
from pgamit.etm.etm_functions.polynomial import PolynomialFunction
from pgamit.etm.etm_functions.periodic import PeriodicFunction
from pgamit.etm.etm_functions.stochastic_signal import StochasticSignal
from pgamit.etm.core.jump_manager import JumpManager
from pgamit.etm.least_squares.least_squares import EtmFit, AdjustmentModels
from pgamit.etm.core.logging_config import setup_etm_logging
from pgamit.etm.visualization.data_prep import PlotDataPreparer
from pgamit.etm.visualization.etm_plotting import EtmPlotter



class EtmEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
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


class EtmEngine:
    """Core mathematical engine for ETM processing, separated from business logic"""

    def __init__(self, cnn: Cnn, config: EtmConfig,
                 solution_type: SolutionType,
                 stack_name: str = None,
                 adjustment_strategy: AdjustmentModels = None):

        setup_etm_logging()

        self.config = config
        # save the adjustment strategy
        if adjustment_strategy is not None:
            self.config.modeling.adjustment_strategy = adjustment_strategy

        if solution_type == SolutionType.GAMIT:
            self.solution_data = GAMITSolutionData(stack_name, config)
        elif solution_type == SolutionType.PPP:
            self.solution_data = PPPSolutionData(config)
        else:
            self.solution_data = PPPSolutionData(config)

        # load the data
        self.solution_data.load_data(cnn)

        # set the basic functions
        polynomial = PolynomialFunction(config, time_vector=self.solution_data.time_vector)
        periodic = PeriodicFunction(config, time_vector=self.solution_data.time_vector)

        self.jump_manager = JumpManager(self.solution_data, config)
        self.jump_manager.build_jump_table(self.solution_data.time_vector)

        self.design_matrix = DesignMatrix(self.solution_data.time_vector,
                                          [polynomial, periodic] + self.jump_manager.get_jump_functions(),
                                          config)

        # if adjustment strategy is LSQ_COLLOCATION, add the stochastic function to the design matrix
        # (although it is not used during the fit)
        if self.config.modeling.adjustment_strategy == AdjustmentModels.LSQ_COLLOCATION:
            self.design_matrix.functions.append(
                StochasticSignal(self.config, self.solution_data.time_vector_cont_mjd))

        self.fit = EtmFit(self.config, self.design_matrix, self.solution_data.hash)

    def run_adjustment(self, try_loading_db=True, try_save_to_db=True,
                       cnn: Optional[Cnn] = None) -> None:
        """Run the iterative least squares adjustment"""
        if try_loading_db and cnn:
            success = load_parameters_db(cnn, self.fit, self.solution_data.transform_to_local(),
                                         self.solution_data.time_vector_mjd)
            logger.info(f'Loading parameters from database: {success}')
        else:
            success = False

        if not success:
            run_time = self.fit.run_fit(self.solution_data, self.design_matrix)
            if cnn and try_save_to_db:
                save_parameters_db(cnn, self.fit)
            logger.info(f'Estimated parameters in {run_time} seconds')

    def save_etm(self,
                 filename: str = None,
                 dump_functions: bool = True,
                 dump_raw_results: bool = False,
                 dump_observations: bool = False,
                 dump_design_matrix: bool = False,
                 dump_model: bool = False) -> Dict:

        dm = self.design_matrix
        # create a station_meta copy to replace the stationinforecord instances
        station_meta = copy.deepcopy(self.config.metadata)
        station_meta.station_information = [item.to_json() for item in station_meta.station_information]

        # dump continuous model if requested
        if dump_model:
            model_values = self.fit.get_time_continuous_model(self.solution_data.time_vector_cont)
        else:
            model_values = None

        etm_dump = {
            "network_code": self.config.network_code,
            "station_code": self.config.station_code,
            "station_meta": asdict(station_meta),
            "solution_options": asdict(self.config.solution),
            "modeling_params": asdict(self.config.modeling),
            "raw_results": [asdict(self.fit.results[0]),
                            asdict(self.fit.results[1]),
                            asdict(self.fit.results[2])] if dump_raw_results else None,
            "functions": [asdict(funct.p) for funct in dm.functions if funct.fit] if dump_functions else None,
            "observations": asdict(self.solution_data.coordinates) if dump_observations else None,
            "model": {'observations_neu': self.solution_data.transform_to_local(),
                      'time_vector_fyear': self.solution_data.time_vector_cont,
                      'time_vector_mjd': self.solution_data.time_vector_cont_mjd,
                      'model_cont': model_values} if dump_model else None,
            "covariance": self.fit.covar.tolist(),
            "design_matrix": dm.matrix if dump_design_matrix else None,
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
                       json.dumps(etm_dump, indent=4, sort_keys=False, cls=EtmEncoder))

            #binary_data = bson.encode(etm_dump, cls)
            #with open(filename + '.bson', 'wb') as f:
            #    f.write(binary_data)

        return etm_dump

    def plot(self) -> None:

        plotter = EtmPlotter(self.config)
        plot_data = PlotDataPreparer(self.config).prepare_time_series_data(self.solution_data, self.fit)

        plotter.plot_time_series(plot_data)

    def plot_hist(self) -> None:

        plotter = EtmPlotter(self.config)
        plot_data = PlotDataPreparer(self.config).prepare_time_series_data(self.solution_data, self.fit)

        plotter.plot_histogram(plot_data)

    def get_position(self, date: Date, etm_solution: EtmSolutionType = EtmSolutionType.MODEL) -> Dict:
        """
        use computed model to obtain a station position
        """
        model = [np.array([]), np.array([]), np.array([])]
        model_ecef = np.array([])

        # find the model value, if fit done
        if self.config.modeling.status == self.config.modeling.FitStatus.POSTFIT:
            for i in range(3):
                model[i] = self.design_matrix.alternate_time_vector(
                    np.array([date.fyear])) @ self.fit.results[i].parameters
            # transform back to ecef
            model_ecef = np.array(self.solution_data.transform_to_ecef(model))

        if etm_solution == EtmSolutionType.OBSERVATION:
            # find the date of the request
            i = np.where(self.solution_data.time_vector_mjd == date.mjd)

            if i:
                # solution found, save it
                position = [self.solution_data.x[i], self.solution_data.y[i], self.solution_data.z[i]]
                if model_ecef.size:
                    if self.fit.outlier_flags[i]:
                        source = self.solution_data.stack_name.upper() + ' with ETM solution: good'
                    else:
                        # filter coordinate because it was an outlier
                        position = model_ecef
                        source = self.solution_data.stack_name.upper() + ' with ETM solution: filtered'
                else:
                    source = self.solution_data.stack_name.upper() + ' solution, but no ETM'
            else:
                if model_ecef.size:
                    position = model_ecef
                    source = 'No ' + self.solution_data.stack_name.upper() + ' solution: ETM'
                else:
                    source = 'No ' + self.solution_data.stack_name.upper() + ' solution, no ETM: mean coordinate'
                    position = [self.solution_data.x.mean(), self.solution_data.y.mean(), self.solution_data.z.mean()]
        else:
            if model_ecef.size:
                source = 'ETM solution requested'
                position = model_ecef
            else:
                raise Exception('Model requested by no ETM was found for ' + self.config.get_station_id())

        return {'position': position.tolist(), 'source': source}