from abc import ABC, abstractmethod

from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

from ..core.etm_config import EtmConfig
from ..core.type_declarations import JumpType, FitStatus
from ..data.solution_data import SolutionData
from ..least_squares.least_squares import EtmFit
from ..visualization.data_classes import TimeSeriesPlotData, ComponentData, PlotOutputConfig

class PlotTemplate(ABC):
    """Abstract base class for plot templates"""

    def __init__(self, config: EtmConfig):
        self.config = config
        self.colors = self.setup_colors()
        self.styles = self.setup_styles()

        logger.debug(f'PlotTemplate: {str(self.config.plotting_config)}')

    @staticmethod
    def setup_colors() -> Dict[str, str]:
        """Setup color scheme"""
        return {
            'observations': (0, 150 / 255, 235 / 255),
            'model': 'red',
            'confidence': 'lightblue',
            'outliers': 'cyan',
            'observations_not_fit': 'darkgray',
            'mechanical_jump': JumpType.MECHANICAL_MANUAL.color,
            'antenna_jump': JumpType.MECHANICAL_ANTENNA.color,
            'frame_jump': JumpType.REFERENCE_FRAME.color,
            'coseismic_jump': JumpType.COSEISMIC_ONLY.color,
            'coseismic_decay': JumpType.COSEISMIC_JUMP_DECAY.color,
            'postseismic': JumpType.POSTSEISMIC_ONLY.color,
            'auto_jump': JumpType.AUTO_DETECTED.color,
            'missing_solution': 'magenta'
        }

    @staticmethod
    def setup_styles() -> Dict[str, Dict[str, Any]]:
        """Setup plot styles"""
        return {
            'observations': {'marker': 'o', 'markersize': 2, 'linestyle': 'none'},
            'model': {'linewidth': 1.5},
            'confidence': {'alpha': 0.2},
            'jumps': {'linestyle': ':', 'linewidth': 1.75},
            'auto_jumps': {'linestyle': '-.', 'linewidth': 1}
        }

    @abstractmethod
    def create_figure_layout(self, *args, **kwargs) -> Tuple[plt.Figure, Any]:
        """Create figure and axis layout"""
        pass

    def generate_title(self, plot_data: TimeSeriesPlotData, output_config: PlotOutputConfig) -> str:
        """Generate plot title"""
        base_title = (f"{plot_data.station_id} ({plot_data.stack_name.upper()} {plot_data.completion:.2f}% - "
                      f"{self.config.modeling.least_squares_strategy.adjustment_model.description}) "
                      f"lat: {plot_data.latitude:.5f} lon: {plot_data.longitude:.5f}")
        return base_title

# ============================================================================
# Data preparation utilities
# ============================================================================

class PlotDataPreparer:
    """Utility class for preparing plot data from ETM results"""

    def __init__(self, config: EtmConfig):

        self.config = config
        self.output_config = config.plotting_config

    def prepare_time_series_data(self, solution_data: SolutionData,
                                 etm_fit: EtmFit = None) -> TimeSeriesPlotData:
        """Prepare data for time series plotting"""

        # Transform coordinates to local frame
        local_coords = solution_data.transform_to_local(ignore_data_window=True)

        # Prepare component data
        n_data, e_data, u_data = self._prepare_component_data(local_coords, solution_data, etm_fit)
        jump_tables = self._prepare_jump_tables(etm_fit)

        # Create plot data container
        plot_data = TimeSeriesPlotData(
            station_id=solution_data.get_station_id().upper(),
            solution_type=solution_data.soln,
            stack_name=solution_data.stack_name,
            completion=solution_data.completion,
            latitude=solution_data.lat[0],
            longitude=solution_data.lon[0],
            north_data=n_data,
            east_data=e_data,
            up_data=u_data,
            covariance_matrix=etm_fit.covar,
            has_etm_results=self.config.modeling.status == FitStatus.POSTFIT,
            jump_tables=(jump_tables[0], jump_tables[1], jump_tables[2]),
            jumps=jump_tables[3]
        )

        # Add ETM-specific information
        if etm_fit:
            plot_data.parameter_summary = self._generate_parameter_summary(etm_fit, solution_data)
            plot_data.periodic_summary = self._generate_periodic_summary(etm_fit)
            plot_data.wrms_values = (etm_fit.results[0].wrms * 1000.,
                                     etm_fit.results[1].wrms * 1000.,
                                     etm_fit.results[2].wrms * 1000.)

        return plot_data

    @staticmethod
    def _prepare_jump_tables(etm_results: EtmFit) -> Tuple[list, list, list, list]:

        jump_tables = ([], [], [], [])

        for funct in etm_results.design_matrix.functions:
            if funct.p.object == 'jump':
                tables = funct.print_parameters()
                # loop and add the jump twice to account for multiple lines in co+postseismic
                # this is because in time series template we loop using jump_tables
                #@todo: make the loop in time series template use the jumps to call and build tables right there
                for _ in range(len(tables[0])):
                    jump_tables[3].append(funct)
                for i in range(3):
                    jump_tables[i].extend(tables[i])

        return jump_tables

    def _prepare_component_data(self, observations: List, solution_data: SolutionData,
                                etm_results: EtmFit) -> List[ComponentData]:
        """Prepare data for one coordinate component"""
        data = []

        # apply observation mask if any
        mask = etm_results.config.modeling.get_observation_mask(solution_data.time_vector)

        for i in range(3):
            data.append(ComponentData(
                observations=observations[i] * 1000, # convert to mm
                observations_fit=observations[i][mask] * 1000,  # convert to mm
                observations_not_fit=observations[i][~mask] * 1000, # convert to mm
                time_vector=solution_data.time_vector,
                time_vector_fit=solution_data.time_vector[mask],
                time_vector_not_fit=solution_data.time_vector[~mask],
                time_range=(solution_data.time_vector.min(), solution_data.time_vector.max())
            ))

            # Add model values
            if etm_results.config.modeling.status == FitStatus.POSTFIT:
                model_values = (etm_results.design_matrix.alternate_time_vector(solution_data.time_vector_cont)
                                @ etm_results.results[i].parameters) + etm_results.results[i].stochastic_signal
                data[i].model_time_vector = solution_data.time_vector_cont
                data[i].model_values = model_values * 1000  # Convert to mm

                # Add residuals
                data[i].residuals = etm_results.results[i].residuals * 1000.

                for funct in etm_results.design_matrix.functions:
                    if ((funct.p.object == 'periodic' and self.output_config.plot_remove_periodic)
                            or (funct.p.object == 'polynomial' and self.output_config.plot_remove_polynomial)
                            or (funct.p.object == 'jump' and self.output_config.plot_remove_jumps)) and funct.fit:
                        data[i].observations -= funct.eval(i) * 1000.
                        data[i].observations_fit -= funct.eval(i, data[i].time_vector_fit) * 1000.
                        if np.any(~mask):
                            data[i].observations_not_fit -= funct.eval(i, data[i].time_vector_not_fit) * 1000.
                        data[i].model_values -= funct.eval(i, solution_data.time_vector_cont) * 1000.

                    if self.output_config.plot_remove_stochastic:
                        if funct.p.object == 'stochastic':
                            # user requested to remove stochastic signal
                            data[i].observations -= funct.eval(i, solution_data.time_vector_mjd) * 1000.
                            data[i].observations_fit -= funct.eval(i, solution_data.time_vector_mjd[mask]) * 1000.
                            if np.any(~mask):
                                data[i].observations_not_fit -= (
                                        funct.eval(i, solution_data.time_vector_mjd[~mask]) * 1000.)
                            data[i].model_values -= etm_results.results[i].stochastic_signal * 1000.
                            # residuals don't have stochastic signal included, so no need to remove
                    else:
                        # because the residuals already have the stochastic signal removed, add back
                        # if user requests to NOT REMOVE
                        if funct.p.object == 'stochastic':
                            data[i].residuals += funct.eval(i, solution_data.time_vector_mjd[mask]) * 1000.

                # Add confidence bounds
                limit = self.config.modeling.least_squares_strategy.sigma_filter_limit
                sigma = etm_results.results[i].wrms * 1000  # Convert to mm
                data[i].confidence_bounds = (
                    data[i].model_values - limit * sigma,
                    data[i].model_values + limit * sigma
                )

                # Add outlier flags
                data[i].outlier_flags = etm_results.outlier_flags

        return data

    @staticmethod
    def _generate_parameter_summary(etm_results: EtmFit, solution_data: SolutionData) -> str:
        """Generate parameter summary string"""
        # This would integrate with the function objects to generate
        # formatted parameter summaries
        output = ['']

        for funct in etm_results.design_matrix.functions:
            if funct.p.object == 'polynomial' and funct.fit:
                # retrieve conventional epoch position
                ce_position =  solution_data.transform_to_ecef(
                    etm_results.get_time_continuous_model(np.array([funct.p.t_ref])))
                output, _, _ = funct.print_parameters(ce_position=ce_position)

        return output[0]

    @staticmethod
    def _generate_periodic_summary(etm_results: EtmFit) -> str:
        """Generate periodic parameter summary"""
        # This would integrate with periodic function to generate summary
        output = ['']

        for funct in etm_results.design_matrix.functions:
            if funct.p.object == 'periodic':
                output, _, _ = funct.print_parameters()

        return output[0]
