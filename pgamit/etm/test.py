import logging

from pgamit.etm.core.etm_engine import EtmEngine
from pgamit.etm.core.etm_config import EtmConfig
from pgamit.dbConnection import Cnn
from pgamit.etm.core.logging_config import setup_etm_logging
from pgamit.etm.least_squares.least_squares import AdjustmentModels
from pgamit.etm.core.type_declarations import SolutionType

setup_etm_logging(level=logging.DEBUG)

cnn = Cnn('/home/demian/pg_osu/gnss_data.cfg')

config = EtmConfig('igs', 'clrk', cnn=cnn)

config.modeling.adjustment_strategy = AdjustmentModels.LSQ_COLLOCATION
# only fit the data within this interval
#config.modeling.data_model_window = [(1995.0, 2000.0), (2005.0, 2015.9)]

# options for plotting
config.plotting_config.filename = '/home/demian/pg_osu/'
config.plotting_config.plot_show_outliers = True
config.plotting_config.save_kwargs={'dpi': 150}
config.plotting_config.plot_remove_polynomial = True
config.plotting_config.plot_remove_periodic = False
config.plotting_config.plot_remove_stochastic = False
config.plotting_config.plot_residuals_mode = False
config.plotting_config.plot_remove_jumps = True
#config.plotting_config.plot_time_window = [2015.0, 2019.0]

config.validate_config()

etm = EtmEngine(cnn, config, SolutionType.GAMIT, stack_name='igs20_ant')
etm.run_adjustment(cnn=cnn)

etm.save_etm(filename='/home/demian/pg_osu/',
             dump_observations=True,
             dump_design_matrix=True,
             dump_raw_results=True,
             dump_functions=True,
             dump_model=True)
etm.plot()
etm.plot_hist()

print('done')