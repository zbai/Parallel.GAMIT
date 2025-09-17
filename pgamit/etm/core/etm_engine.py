
from typing import Optional, List
from enum import IntEnum, auto
import numpy as np
import logging

logger = logging.getLogger(__name__)

# app
from pgamit.dbConnection import Cnn
from pgamit.etm.data.solution_data import GAMITSolutionData, PPPSolutionData
from pgamit.etm.core.etm_config import ETMConfig
from pgamit.etm.least_squares.design_matrix import DesignMatrix
from etm.etm_functions.polynomial import PolynomialFunction
from etm.etm_functions.periodic import PeriodicFunction
from etm.core.jump_manager import JumpManager
from etm.least_squares.least_squares import ETMFit
from etm.core.logging_config import setup_etm_logging
from etm.visualization.plot_tempates import PlotDataPreparer, TimeSeriesTemplate, PlotOutputConfig
from etm.visualization.etm_plotting import ETMPlotter

class SolutionType(IntEnum):
    """Enum for noise models"""
    GAMIT = auto()
    PPP = auto()


class ETMEngine:
    """Core mathematical engine for ETM processing, separated from business logic"""

    def __init__(self, cnn: Cnn, config: ETMConfig,
                 solution_type: SolutionType,
                 stack_name: str = None):

        setup_etm_logging()

        self.config = config
        self.fit: ETMFit = ETMFit(self.config)

        if solution_type == SolutionType.GAMIT:
            self.soln = GAMITSolutionData(stack_name, config)
        elif solution_type == SolutionType.PPP:
            self.soln = PPPSolutionData(config)
        else:
            self.soln = PPPSolutionData(config)

        # load the data
        self.soln.load_data(cnn)

        # set the basic functions
        polynomial = PolynomialFunction(config, time_vector=self.soln.time_vector)
        periodic = PeriodicFunction(config, time_vector=self.soln.time_vector)

        self.jump_manager = JumpManager(self.soln, config)
        self.jump_manager.build_jump_table(self.soln.time_vector)

        self.design_matrix = DesignMatrix(self.soln.time_vector,
                                          [polynomial, periodic] + self.jump_manager.get_jump_functions(),
                                          config)


    def run_adjustment(self) -> None:
        """Run the iterative least squares adjustment"""
        self.fit = ETMFit(self.config)

        self.fit.run_fit(self.soln, self.design_matrix)

    def plot(self):

        plotter = ETMPlotter(self.config)
        plot_data = PlotDataPreparer(self.config).prepare_time_series_data(self.soln, self.fit)

        plotter.plot_time_series(plot_data)


