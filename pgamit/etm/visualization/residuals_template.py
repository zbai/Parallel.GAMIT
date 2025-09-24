"""
Project: Parallel.GAMIT
Date: 9/22/25 8:40 AM
Author: Demian D. Gomez
"""
import matplotlib.pyplot as plt
from typing import List, Tuple

# app
from pgamit.etm.visualization.data_prep import PlotTemplate
from pgamit.etm.visualization.data_classes import TimeSeriesPlotData


class ResidualTemplate(PlotTemplate):
    """Template for residual-specific plots"""

    def create_figure_layout(self) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create residual plot layout"""
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16, 10))
        return fig, axes

    def plot_residual_time_series(self, axes: List[plt.Axes], plot_data: TimeSeriesPlotData) -> None:
        """Plot residual time series for each component"""
        components = ['North', 'East', 'Up']

        for i, (ax, component) in enumerate(zip(axes, components)):
            data = plot_data.get_component_data(i)

            if data.residuals is not None:
                # Plot residuals in mm
                if data.outlier_flags is not None:
                    good_mask = data.outlier_flags
                    ax.plot(data.time_vector[good_mask], data.residuals[good_mask],
                            color=self.colors['observations'], **self.styles['observations'])
                else:
                    ax.plot(data.time_vector, data.residuals,
                            color=self.colors['observations'], **self.styles['observations'])

                # Add zero line
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

                # Add ±2.5σ bounds if available
                if plot_data.wrms_values:
                    sigma = plot_data.wrms_values[i]
                    ax.axhline(y=2.5 * sigma, color='blue', linestyle='--', alpha=0.5)
                    ax.axhline(y=-2.5 * sigma, color='blue', linestyle='--', alpha=0.5)

                ax.grid(True)
                ax.set_ylabel(f"{component} residuals [mm]")

                if i == len(axes) - 1:  # Last subplot
                    ax.set_xlabel('Time [years]')