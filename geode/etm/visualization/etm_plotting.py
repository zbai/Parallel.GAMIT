import warnings
# Suppress the specific warning
warnings.filterwarnings('ignore', message='Starting a Matplotlib GUI outside')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import base64
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
import os

# app
from ..core.etm_config import EtmConfig
from ..visualization.time_series_template import TimeSeriesTemplate
from ..visualization.histogram_template import HistogramTemplate
from ..visualization.data_classes import  TimeSeriesPlotData, PlotOutputConfig, HistogramPlotData

logger = logging.getLogger(__name__)


@dataclass
class Templates:
    time_series: TimeSeriesTemplate
    histogram: HistogramTemplate


class EtmPlotter:
    """Main plotting class for ETM visualizations"""

    def __init__(self, config: EtmConfig):
        self.config = config
        self.templates = Templates(
            TimeSeriesTemplate(config),
            HistogramTemplate(config)
        )

        # Interactive plotting state
        self.figure = None
        self.picking_enabled = False
        self.pick_callback_id = None

    def plot_time_series(self, plot_data: TimeSeriesPlotData,
                         output_config: Optional[PlotOutputConfig] = None) -> Optional[str]:
        """
        Create time series plot

        Args:
            plot_data: Data container with all plotting information
            output_config: Output configuration (file, format, etc.)

        Returns:
            Base64 encoded image string if output to BytesIO, None otherwise
        """
        # if no specific output_config passed, replaced with default
        if output_config is None:
            output_config = self.config.plotting_config

        template = self.templates.time_series

        # Prepare plot layout
        layout = self._determine_plot_layout(plot_data, output_config)

        # Create figure and axes
        fig, axes = template.create_figure_layout(layout)
        self.figure = fig

        # Generate title
        title = template.generate_title(plot_data, output_config)
        fig.suptitle(title, fontsize=9, family='monospace')

        main_axes = axes[layout['main_axes_indices']]
        # Add jump parameter tables
        if plot_data.jump_tables:
            template.add_jump_tables(fig, main_axes, plot_data.jump_tables, plot_data.jumps)

        # Plot data for each component
        for i, (ax, component) in enumerate(zip(main_axes, ['North', 'East', 'Up'])):
            self._plot_component_data(ax, plot_data, component, i, template, output_config)

        fig.subplots_adjust(left=0.21)

        # Add component-specific annotations
        # template.add_component_annotations(ax, plot_data, component, i)

        # Add outlier plots if requested
        if layout['show_outliers']:
            outlier_axes = axes[layout['outlier_axes_indices']]
            for i, ax in enumerate(outlier_axes):
                self._plot_outlier_data(ax, plot_data, i, template, output_config)

        # Handle output
        return self._handle_plot_output(fig, output_config)

    def plot_histogram(self, plot_data: TimeSeriesPlotData,
                       output_config: Optional['PlotOutputConfig'] = None) -> Optional[str]:
        """Create histogram/residual analysis plot"""
        # if no specific output_config passed, replaced with default
        if output_config is None:
            output_config = self.config.plotting_config

        template = self.templates.histogram

        # Create figure layout
        fig, axes = template.create_figure_layout()

        # Generate title
        title = template.generate_title(plot_data, output_config)
        fig.suptitle(title, fontsize=9, family='monospace')

        # Plot N-E scatter with error ellipse
        xlim, ylim = template.plot_ne_scatter(fig, axes['ne_scatter'], plot_data)

        # Plot component histograms
        template.plot_component_histograms(axes, plot_data, xlim, ylim)

        axes['ne_scatter'].set_xlim(xlim)
        axes['ne_scatter'].set_ylim(ylim)

        return self._handle_plot_output(fig, output_config, histogram=True)

    def _determine_plot_layout(self, plot_data: TimeSeriesPlotData,
                               output_config: Optional[PlotOutputConfig]) -> Dict[str, Any]:
        """Determine plot layout based on data and configuration"""
        # if no specific output_config passed, replaced with default
        if output_config is None:
            output_config = self.config.plotting_config

        show_outliers = (output_config and output_config.plot_show_outliers and
                         plot_data.has_etm_results)

        if show_outliers:
            fig_size = (16, 10)
            subplot_config = {'nrows': 3, 'ncols': 2, 'sharex': True}
            main_axes_indices = [0, 2, 4] #[(0, 0), (1, 0), (2, 0)]
            outlier_axes_indices = [1, 3, 5] #[(0, 1), (1, 1), (2, 1)]
        else:
            fig_size = (16, 10)
            subplot_config = {'nrows': 3, 'ncols': 1, 'sharex': True}
            main_axes_indices = [0, 1, 2]
            outlier_axes_indices = None

        return {
            'fig_size': fig_size,
            'subplot_config': subplot_config,
            'main_axes_indices': main_axes_indices,
            'outlier_axes_indices': outlier_axes_indices,
            'show_outliers': show_outliers
        }

    def _plot_component_data(self, ax, plot_data: TimeSeriesPlotData,
                             component: str, component_idx: int,
                             template: TimeSeriesTemplate,
                             output_config: PlotOutputConfig) -> None:
        """Plot data for one coordinate component"""
        # if no specific output_config passed, replaced with default
        if output_config is None:
            output_config = self.config.plotting_config

        data = plot_data.get_component_data(component_idx)

        # Plot observations
        if output_config.plot_residuals_mode:
            template.plot_residuals(ax, data)
        else:
            template.plot_observations(ax, data)

            # Plot model if available
            if plot_data.has_etm_results:
                template.plot_model(ax, data)
                template.plot_confidence_bounds(ax, data)

        # Add grid and labels
        template.add_grid_and_labels(ax, component)

        # Apply time window if specified
        if output_config.plot_time_window:
            template.apply_time_window(ax, output_config.plot_time_window)

        # Plot jumps
        if plot_data.jumps:
            template.plot_jumps(ax, plot_data.jumps, data.time_range)

    def _plot_outlier_data(self, ax, plot_data: TimeSeriesPlotData,
                           component_idx: int,
                           template: TimeSeriesTemplate,
                           output_config: PlotOutputConfig) -> None:
        """Plot outlier comparison data"""
        # if no specific output_config passed, replaced with default
        if output_config is None:
            output_config = self.config.plotting_config

        data = plot_data.get_component_data(component_idx)

        # activate grid
        ax.grid(True)

        # Plot all data (including outliers)
        template.plot_all_observations(ax, data)
        template.plot_model(ax, data)
        template.plot_confidence_bounds(ax, data)

        # Plot missing solutions if available
        if output_config.missing_solutions:
            template.plot_missing_solutions(ax, output_config.missing_solutions)

        # Apply time window
        if output_config.plot_time_window:
            template.apply_time_window(ax, output_config.plot_time_window)

    def _handle_plot_output(self, fig,
                            output_config: Optional[PlotOutputConfig],
                            histogram: bool = False) -> Optional[str]:
        """Handle plot output based on configuration"""

        if output_config.filename:
            if not os.path.basename(output_config.filename):
                filename = (os.path.join(output_config.filename,
                                                      self.config.build_filename()) + ('_hist' if histogram else '')
                                          + '.' + output_config.format)
            else:
                dirs = os.path.dirname(output_config.filename)
                # split at the format if provided
                file = os.path.basename(output_config.filename).split('.' + output_config.format)
                filename = (os.path.join(dirs, file[0]) + ('_hist' if histogram else '')
                                          + '.' + output_config.format)
        else:
            filename = output_config.filename

        if not output_config.save_kwargs:
            output_config.save_kwargs = {}

        if output_config.interactive:
            # Interactive mode
            self._setup_interactive_mode(fig)
            plt.show()
            return None

        elif filename:
            # Save to file
            plt.savefig(filename, **output_config.save_kwargs)
            plt.close()
            return None

        elif output_config.file_io:
            # Save to BytesIO
            plt.savefig(output_config.file_io, format=output_config.format,
                        **output_config.save_kwargs)
            output_config.file_io.seek(0)
            plt.close()
            return base64.b64encode(output_config.file_io.getvalue()).decode()

        else:
            # Default interactive
            plt.show()
            plt.close()
            return None

    def _setup_interactive_mode(self, fig) -> None:
        """Setup interactive mode with jump picking capability"""
        self.figure = fig

        # Add jump picking button
        ax_button = plt.axes((0.85, 0.01, 0.08, 0.055))
        button = Button(ax_button, 'Add Jump', color='red', hovercolor='green')
        button.on_clicked(self._toggle_jump_picking)

    def _toggle_jump_picking(self, event) -> None:
        """Toggle jump picking mode"""
        if not self.picking_enabled:
            logger.info('Entering jump picking mode')
            self.picking_enabled = True
            self.pick_callback_id = self.figure.canvas.mpl_connect(
                'button_press_event', self._on_pick_jump
            )
        else:
            logger.info('Disabling jump picking mode')
            self.picking_enabled = False
            if self.pick_callback_id:
                self.figure.canvas.mpl_disconnect(self.pick_callback_id)

    def _on_pick_jump(self, event) -> None:
        """Handle jump picking event"""
        if event.xdata is None:
            return

        from geode import pyDate
        picked_date = pyDate.Date(fyear=event.xdata)
        logger.info(f'Picked potential jump at: {picked_date.yyyyddd()}')

        # TODO: Integrate with jump management system
        # For now, just log the event
        print(f'Jump picked at epoch: {event.xdata:.3f} ({picked_date.yyyyddd()})')
