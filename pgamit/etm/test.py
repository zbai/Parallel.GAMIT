import logging
import numpy as np

from pgamit.pyDate import Date
from pgamit.etm.core.type_declarations import CovarianceFunction, SolutionType
from pgamit.etm.core.etm_engine import EtmEngine
from pgamit.etm.core.etm_config import EtmConfig
from pgamit.etm.etm_functions.polynomial import PolynomialFunction
from pgamit.etm.etm_functions.periodic import PeriodicFunction
from pgamit.etm.etm_functions.jumps import JumpFunction, JumpType
from pgamit.dbConnection import Cnn
from pgamit.etm.core.logging_config import setup_etm_logging
from pgamit.etm.least_squares.least_squares import AdjustmentModels

setup_etm_logging(level=logging.DEBUG)

cnn = Cnn('/home/demian/pg_osu/gnss_data.cfg')

#config = EtmConfig(json_file='/home/demian/pg_osu/arg.igm1_igs14.json')
#etm = EtmEngine(config)
#etm.run_adjustment(try_loading_db=False, try_save_to_db=False)
#config.plotting_config.filename = '/home/demian/pg_osu/'
#etm.plot()

config = EtmConfig('arg', 'igm1', cnn=cnn)
config.solution.solution_type = SolutionType.GAMIT
config.solution.stack_name = 'igs14'

config.modeling.least_squares_strategy.adjustment_model = AdjustmentModels.ROBUST_LEAST_SQUARES
config.modeling.least_squares_strategy.covariance_function = CovarianceFunction.ARMA
# config.modeling.relaxation = np.array([1.0])
config.modeling.fit_auto_detected_jumps = False
config.modeling.fit_auto_detected_jumps_method = 'dbscan'
# only fit the data within this interval
#config.modeling.data_model_window = [(1995.0, 2005.9999), (2009.606849, 2025.5)]
poly_model = PolynomialFunction(config)
periodic_model = PeriodicFunction(config)
illapel = JumpFunction(config, time_vector=np.array([0]), date=Date(year=2015, doy=259),
                       jump_type=JumpType.COSEISMIC_JUMP_DECAY, fit=False)

poly_model.p.params = [np.array([np.nan, 0.115]), np.array([np.nan, np.nan]), np.array([np.nan, 0.0])]
poly_model.p.sigmas = [np.array([np.nan, 0.0001]), np.array([np.nan, 0.0001]), np.array([np.nan, 0.00001])]

periodic_model.p.params = [np.array([0.0, 0.0, 0.0, 0.0]),
                           np.array([0.0, 0.0, 0.012, 0.001]),
                           np.array([0.0, 0.0, 0.0, 0.0])]

illapel.p.params = [np.array([0.0, np.nan, np.nan]), np.array([0.0, np.nan, np.nan]), np.array([0.0, np.nan, np.nan])]
illapel.p.sigmas = [np.array([0.00001, np.nan, np.nan]), np.array([0.00001, np.nan, np.nan]), np.array([0.00001, np.nan, np.nan])]

config.modeling.prefit_models = []
config.modeling.least_squares_strategy.constraints.extend([poly_model, illapel])

# options for plotting
config.plotting_config.filename = '/home/demian/pg_osu/'
config.plotting_config.plot_show_outliers = False
config.plotting_config.save_kwargs={'dpi': 150}
config.plotting_config.plot_remove_polynomial = True
config.plotting_config.plot_remove_periodic = False
config.plotting_config.plot_remove_stochastic = True
config.plotting_config.plot_residuals_mode = False
config.plotting_config.plot_remove_jumps = False
#config.plotting_config.plot_time_window = [2015.0, 2019.0]

config.validate_config()

etm = EtmEngine(config, cnn=cnn)
etm.run_adjustment(cnn=cnn, try_save_to_db=False, try_loading_db=False)

etm.save_etm(filename='/home/demian/pg_osu/',
             dump_observations=True,
             dump_design_matrix=True,
             dump_raw_results=True,
             dump_functions=True,
             dump_model=True)
etm.plot()
etm.plot_hist()

print('done')