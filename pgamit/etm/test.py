from etm.core.etm_engine import *
from etm.core.etm_config import ETMConfig
from pgamit.dbConnection import Cnn


cnn = Cnn('/home/demian/pg_osu/gnss_data.cfg')

config = ETMConfig('arg', 'igm1', cnn=cnn)

# options for plotting
config.plotting_config.filename = '/home/demian/pg_osu/test.png'
config.plotting_config.plot_show_outliers = True
config.plotting_config.save_kwargs={'dpi': 150}
config.plotting_config.plot_remove_polynomial = True
config.plotting_config.plot_remove_periodic = True

config.validate_config()

etm = ETMEngine(cnn, config, SolutionType.GAMIT, stack_name='igs14')
etm.run_adjustment()
etm.plot()

print('done')