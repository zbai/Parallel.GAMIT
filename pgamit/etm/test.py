
from pgamit.etm.solution_data import PPPSolutionData
from pgamit.etm.solution_data import GAMITSolutionData
from pgamit.etm.etm_config import ETMConfig
from pgamit.dbConnection import Cnn
from pgamit.etm.function_factory import FunctionFactory

cnn = Cnn('/home/demian/pg_osu/gnss_data.cfg')


config = ETMConfig('arg', 'igm1', cnn=cnn)

config.validate_config()

soln = PPPSolutionData(config)

soln.load_data(cnn)

soln = GAMITSolutionData('igs14', config)

soln.load_data(cnn)

factory = FunctionFactory()
pol = factory.create_polynomial(soln, config, _time_vector=soln.t)

print('done')