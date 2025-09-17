from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import transforms
from matplotlib.patches import Ellipse
import numpy as np

from etm.core.etm_config import ETMConfig, JumpType
from etm.data.solution_data import SolutionData
from etm.least_squares.least_squares import ETMFit
from etm.core.etm_config import PlotOutputConfig

@dataclass
class ComponentData:
    """Data for one coordinate component"""
    observations: np.ndarray
    time_vector: np.ndarray
    model_values: Optional[np.ndarray] = None
    model_time_vector: Optional[np.ndarray] = None
    confidence_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    residuals: Optional[np.ndarray] = None
    outlier_flags: Optional[np.ndarray] = None
    time_range: Optional[Tuple[float, float]] = None


@dataclass
class TimeSeriesPlotData:
    """Container for time series plot data"""
    station_id: str
    solution_type: str
    stack_name: str
    completion: float
    latitude: float
    longitude: float

    # Coordinate data
    north_data: ComponentData
    east_data: ComponentData
    up_data: ComponentData

    # ETM results
    has_etm_results: bool = False
    parameter_summary: Optional[str] = None
    periodic_summary: Optional[str] = None
    wrms_values: Optional[Tuple[float, float, float]] = None

    # Jump information
    jumps: Optional[List] = None
    jump_tables: Optional[Tuple[list, list, list]] = None
    auto_jumps: Optional[List] = None
    show_auto_jumps: bool = False

    def get_component_data(self, index: int) -> ComponentData:
        """Get data for component by index (0=N, 1=E, 2=U)"""
        return [self.north_data, self.east_data, self.up_data][index]


@dataclass
class HistogramPlotData:
    """Container for histogram plot data"""
    station_id: str
    solution_type: str
    completion: float
    latitude: float
    longitude: float

    # Residual data (in mm)
    north_residuals: np.ndarray
    east_residuals: np.ndarray
    up_residuals: np.ndarray

    # Covariance information
    covariance_matrix: Optional[np.ndarray] = None
    variance_diagonal: Optional[np.ndarray] = None

    # Filter flags
    outlier_flags: Optional[np.ndarray] = None

class PlotTemplate(ABC):
    """Abstract base class for plot templates"""

    def __init__(self, config: ETMConfig):
        self.config = config
        self.colors = self.setup_colors()
        self.styles = self.setup_styles()

    @staticmethod
    def setup_colors() -> Dict[str, str]:
        """Setup color scheme"""
        return {
            'observations': 'blue',
            'model': 'red',
            'confidence': 'lightblue',
            'outliers': 'cyan',
            'mechanical_jump': JumpType.MECHANICAL_MANUAL.color,
            'antenna_jump': JumpType.MECHANICAL_ANTENNA.color,
            'frame_jump': JumpType.REFERENCE_FRAME.color,
            'coseismic_jump': JumpType.COSEISMIC_ONLY.color,
            'coseismic_decay': JumpType.COSEISMIC_JUMP_DECAY.color,
            'postseismic': JumpType.POSTSEISMIC_ONLY.color,
            'auto_jump': 'orange',
            'missing_solution': 'magenta'
        }

    @staticmethod
    def setup_styles() -> Dict[str, Dict[str, Any]]:
        """Setup plot styles"""
        return {
            'observations': {'marker': 'o', 'markersize': 2, 'linestyle': 'none'},
            'model': {'linewidth': 1.5},
            'confidence': {'alpha': 0.2},
            'jumps': {'linestyle': ':', 'linewidth': 1},
            'auto_jumps': {'linestyle': '-.', 'linewidth': 1}
        }

    @abstractmethod
    def create_figure_layout(self, *args, **kwargs) -> Tuple[plt.Figure, Any]:
        """Create figure and axis layout"""
        pass

    def generate_title(self, plot_data: TimeSeriesPlotData, output_config: PlotOutputConfig) -> str:
        """Generate plot title"""
        base_title = (f"{plot_data.station_id} ({plot_data.stack_name.upper()} {plot_data.completion:.2f}%) "
                      f"lat: {plot_data.latitude:.5f} lon: {plot_data.longitude:.5f}")
        return base_title


class TimeSeriesTemplate(PlotTemplate):
    """Template for time series plots"""

    def create_figure_layout(self, layout: Dict[str, Any]) -> Tuple[plt.Figure, List]:
        """Create time series figure layout"""
        fig, axes = plt.subplots(figsize=layout['fig_size'], **layout['subplot_config'])

        # Handle single vs multiple axis cases
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        elif axes.ndim == 2:
            # Flatten 2D array for easier indexing
            axes = axes.flatten()

        return fig, axes

    def generate_title(self, plot_data: TimeSeriesPlotData, output_config: PlotOutputConfig) -> str:
        """Generate comprehensive time series title"""
        base_title = super().generate_title(plot_data, output_config)

        if plot_data.has_etm_results and plot_data.parameter_summary:
            title_parts = [base_title, plot_data.parameter_summary]

            if plot_data.periodic_summary:
                title_parts.append(plot_data.periodic_summary)

            if plot_data.wrms_values:
                wrms_str = (f"NEU wrms [mm]: {plot_data.wrms_values[0]:.2f} "
                            f"{plot_data.wrms_values[1]:.2f} {plot_data.wrms_values[2]:.2f}")
                title_parts.append(wrms_str)

            # Add processing flags
            flags = []
            if output_config.plot_remove_jumps:
                flags.append(self.config.get_label('jumps removed'))
            if output_config.plot_remove_polynomial:
                flags.append(self.config.get_label('polynomial removed'))
            if output_config.plot_remove_periodic:
                flags.append(self.config.get_label('seasonal removed'))

            return '\n'.join(title_parts) + ' ' + ' '.join(flags)

        elif not plot_data.has_etm_results:
            return base_title + '\n' + self.config.get_label('not_enough')

        return base_title

    def plot_observations(self, ax, data: ComponentData) -> None:
        """Plot coordinate observations"""
        if data.outlier_flags is not None:
            # Plot good observations
            good_mask = data.outlier_flags
            ax.plot(data.time_vector[good_mask], data.observations[good_mask],
                    color=self.colors['observations'], **self.styles['observations'])
        else:
            ax.plot(data.time_vector, data.observations,
                    color=self.colors['observations'], **self.styles['observations'])

    def plot_all_observations(self, ax, data: ComponentData) -> None:
        """Plot all observations including outliers"""
        if data.outlier_flags is not None:
            good_mask = data.outlier_flags
            outlier_mask = ~good_mask

            # Plot outliers first (so they appear behind good points)
            if np.any(outlier_mask):
                ax.plot(data.time_vector[outlier_mask], data.observations[outlier_mask],
                        color=self.colors['outliers'], **self.styles['observations'])

            # Plot good observations
            ax.plot(data.time_vector[good_mask], data.observations[good_mask],
                    color=self.colors['observations'], **self.styles['observations'])
        else:
            ax.plot(data.time_vector, data.observations,
                    color=self.colors['observations'], **self.styles['observations'])

    def plot_model(self, ax, data: ComponentData) -> None:
        """Plot ETM model"""
        if data.model_values is not None:
            ax.plot(data.model_time_vector, data.model_values,
                    color=self.colors['model'], **self.styles['model'])

    def plot_confidence_bounds(self, ax, data: ComponentData) -> None:
        """Plot confidence bounds"""
        if data.confidence_bounds is not None:
            lower_bound, upper_bound = data.confidence_bounds
            ax.plot(data.model_time_vector, lower_bound, 'b', alpha=0.1)
            ax.plot(data.model_time_vector, upper_bound, 'b', alpha=0.1)
            ax.fill_between(data.model_time_vector, lower_bound, upper_bound,
                            color=self.colors['confidence'], **self.styles['confidence'])

    def plot_residuals(self, ax, data: ComponentData) -> None:
        """Plot residuals"""
        if data.residuals is not None:
            if data.outlier_flags is not None:
                good_mask = data.outlier_flags
                ax.plot(data.time_vector[good_mask], data.residuals[good_mask] * 1000,
                        color=self.colors['observations'], **self.styles['observations'])
            else:
                ax.plot(data.time_vector, data.residuals * 1000,
                        color=self.colors['observations'], **self.styles['observations'])

    def plot_jumps(self, ax, jumps: List, time_range: Tuple[float, float]) -> None:
        """Plot jump markers"""
        for jump in jumps:
            if time_range[0] <= jump.date.fyear <= time_range[1]:
                ax.axvline(jump.date.fyear, color=jump.p.jump_type.color if jump.fit else 'gray',
                           **self.styles['jumps'])

    def plot_auto_jumps(self, ax, auto_jumps: List, time_range: Tuple[float, float]) -> None:
        """Plot automatically detected jump markers"""
        for jump in auto_jumps:
            if time_range[0] <= jump.date.fyear <= time_range[1]:
                ax.axvline(jump.date.fyear, color=self.colors['auto_jump'],
                           **self.styles['auto_jumps'])

    def plot_missing_solutions(self, ax, missing_solutions: List) -> None:
        """Plot missing solution markers"""
        for missing_time in missing_solutions:
            ax.axvline(missing_time, color=self.colors['missing_solution'],
                       alpha=0.2, linewidth=1)

    def _get_jump_color(self, jump_type: JumpType) -> str:
        """Get color for jump type"""
        color_map = {
            JumpType.MECHANICAL_MANUAL: 'mechanical_jump',
            JumpType.MECHANICAL_ANTENNA: 'antenna_jump',
            JumpType.REFERENCE_FRAME: 'frame_jump',
            JumpType.COSEISMIC_JUMP_DECAY: 'coseismic_jump',
            JumpType.COSEISMIC_ONLY: 'coseismic_jump',
            JumpType.POSTSEISMIC_ONLY: 'coseismic_decay'
        }
        return self.colors[color_map.get(jump_type, 'mechanical_jump')]

    def add_grid_and_labels(self, ax, component: str) -> None:
        """Add grid and component labels"""
        ax.grid(True)
        ax.set_ylabel(f"{self.config.get_label(component.lower())} [mm]")

    def add_component_annotations(self, ax, plot_data: TimeSeriesPlotData,
                                  component: str, component_idx: int) -> None:
        """Add component-specific annotations (jump tables, etc.)"""
        if plot_data.jump_tables:
            # Position jump table on the left side
            pos = ax.get_position()
            ax.figure.text(0.0025, pos.y0, plot_data.jump_tables[component_idx],
                           fontsize=8, family='monospace')

    def add_jump_tables(self, fig: Figure, axes: List[Axes], jump_tables: Tuple[list, list, list]) -> None:
        """Add jump parameter tables to figure"""
        # This is handled in add_component_annotations for time series
        #t = plt.gca().transData

        for i, table in enumerate(jump_tables):
            initial_pos = axes[i].get_position().y0
            axes_bbox = axes[i].get_window_extent()

            for line, color in reversed(table):
                text = fig.text(0.0025, initial_pos, line, color=color, fontsize=8, family='monospace')
                # fig.canvas.draw()
                ex = text.get_window_extent()
                # t = transforms.offset_copy(text.get_transform(), y=ex.height, units='dots')
                initial_pos += ex.height / axes_bbox.height / 3.75
            # draw title
            fig.text(0.0025, initial_pos, self.config.get_label('table_title'), color='black',
                     fontsize=8, family='monospace')

    def apply_time_window(self, ax, time_window: Tuple[float, float]) -> None:
        """Apply time window to axis"""
        ax.set_xlim(time_window)
        # Auto-scale y-axis for the visible data
        self._autoscale_y_axis(ax)

    def _autoscale_y_axis(self, ax, margin: float = 0.1) -> None:
        """Auto-scale y-axis based on visible data"""
        x_min, x_max = ax.get_xlim()

        y_min, y_max = np.inf, -np.inf
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()

            # Find data within x-range
            mask = (x_data >= x_min) & (x_data <= x_max)
            if np.any(mask):
                y_visible = y_data[mask]
                y_min = min(y_min, np.min(y_visible))
                y_max = max(y_max, np.max(y_visible))

        if y_min != np.inf and y_max != -np.inf:
            y_range = y_max - y_min
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)


class HistogramTemplate(PlotTemplate):
    """Template for histogram/residual analysis plots"""

    def create_figure_layout(self) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
        """Create histogram figure layout"""
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

        axes_dict = {
            'ne_scatter': axes[0, 0],
            'n_hist': axes[0, 1],
            'e_hist': axes[1, 0],
            'u_hist': axes[1, 1]
        }

        return fig, axes_dict

    def generate_title(self, plot_data: HistogramPlotData, output_config: PlotOutputConfig) -> str:
        """Generate histogram plot title"""
        base_title = super().generate_title(plot_data, output_config)

        if plot_data.covariance_matrix is not None:
            covar = plot_data.covariance_matrix
            title_parts = [base_title,
                           f"VAR (N E U): {np.diag(covar)[0]:.3e} {np.diag(covar)[1]:.3e} {np.diag(covar)[2]:.3e}",
                           f"COV (N-E N-U E-U): {covar[0, 1]:.3e} {covar[0, 2]:.3e} {covar[1, 2]:.3e}"]
            return '\n'.join(title_parts)

        return base_title

    def plot_ne_scatter(self, ax, plot_data: HistogramPlotData) -> None:
        """Plot North-East scatter with error ellipse"""
        # Filter out extreme outliers for better visualization
        n_res = plot_data.north_residuals
        e_res = plot_data.east_residuals

        # Use only residuals within reasonable bounds (5cm)
        combined_residuals = np.sqrt(n_res ** 2 + e_res ** 2)
        mask = combined_residuals <= 50  # 50mm

        ax.plot(e_res[mask], n_res[mask], 'ob', markersize=2)

        # Add error ellipse if covariance available
        if plot_data.covariance_matrix is not None:
            self._plot_error_ellipse(ax, plot_data.covariance_matrix,
                                     np.mean(e_res[mask]), np.mean(n_res[mask]))

        ax.grid(True)
        ax.set_xlabel(f"{self.config.get_label('east')} [mm]")
        ax.set_ylabel(f"{self.config.get_label('north')} [mm]")
        ax.set_title(f"{self.config.get_label('residual plot')} "
                     f"{self.config.get_label('north')}-{self.config.get_label('east')}")
        ax.axis('equal')

    def plot_component_histograms(self, axes_dict: Dict[str, plt.Axes],
                                  plot_data: HistogramPlotData) -> None:
        """Plot histograms for each component"""
        components = [
            ('n_hist', plot_data.north_residuals, self.config.get_label('north')),
            ('e_hist', plot_data.east_residuals, self.config.get_label('east')),
            ('u_hist', plot_data.up_residuals, self.config.get_label('up'))
        ]

        for ax_key, residuals, label in components:
            ax = axes_dict[ax_key]

            # Filter extreme outliers
            mask = np.abs(residuals) <= 100  # 100mm
            filtered_residuals = residuals[mask]

            # Create histogram
            n, bins, patches = ax.hist(filtered_residuals, bins=50, alpha=0.75,
                                       color='blue', orientation='horizontal' if 'n_hist' in ax_key else 'vertical')

            ax.grid(True)

            if ax_key == 'n_hist':
                ax.set_xlabel(self.config.get_label('frequency'))
                ax.set_ylabel(f"{label} {self.config.get_label('N residuals')} [mm]")
            else:
                ax.set_ylabel(self.config.get_label('frequency'))
                ax.set_xlabel(f"{label} residuals [mm]")

            ax.set_title(f"{self.config.get_label('histogram plot')} {label}")

    @staticmethod
    def _plot_error_ellipse(ax, covariance: np.ndarray, center_e: float, center_n: float) -> None:
        """Add 2.5-sigma error ellipse to scatter plot"""
        # Extract N-E covariance submatrix (swap indices for E-N order)
        cov_en = covariance[0:2, 0:2]
        cov_en = np.array([[cov_en[1, 1], cov_en[1, 0]],
                           [cov_en[0, 1], cov_en[0, 0]]])  # Swap for E-N order

        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_en)

        # Sort by eigenvalue
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

        # Calculate ellipse parameters
        theta = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
        width = 2 * np.sqrt(eigenvals[0]) * 2.5 * 1000  # Convert to mm, 2.5 sigma
        height = 2 * np.sqrt(eigenvals[1]) * 2.5 * 1000

        # Create ellipse
        ellipse = Ellipse((center_e, center_n), width, height, angle=theta,
                          facecolor='none', edgecolor='red', linewidth=2,
                          label=r'$2.5\sigma$')
        ax.add_patch(ellipse)
        ax.legend()


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
                residuals_mm = data.residuals * 1000

                if data.outlier_flags is not None:
                    good_mask = data.outlier_flags
                    ax.plot(data.time_vector[good_mask], residuals_mm[good_mask],
                            color=self.colors['observations'], **self.styles['observations'])
                else:
                    ax.plot(data.time_vector, residuals_mm,
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


# ============================================================================
# Data preparation utilities
# ============================================================================

class PlotDataPreparer:
    """Utility class for preparing plot data from ETM results"""

    def __init__(self, config: ETMConfig):

        self.config = config
        self.output_config = config.plotting_config

    def prepare_time_series_data(self, solution_data: SolutionData,
                                 etm_fit: ETMFit = None) -> TimeSeriesPlotData:
        """Prepare data for time series plotting"""

        # Transform coordinates to local frame
        local_coords = solution_data.transform_to_local()

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
            has_etm_results=etm_fit.design_matrix is not None,
            jump_tables=(jump_tables[0], jump_tables[1], jump_tables[2]),
            jumps=jump_tables[3]
        )

        # Add ETM-specific information
        if etm_fit:
            plot_data.parameter_summary = self._generate_parameter_summary(etm_fit)
            plot_data.periodic_summary = self._generate_periodic_summary(etm_fit)
            plot_data.wrms_values = (etm_fit.results[0].wrms * 1000.,
                                     etm_fit.results[1].wrms * 1000.,
                                     etm_fit.results[2].wrms * 1000.)

        return plot_data

    @staticmethod
    def _prepare_jump_tables(etm_results: ETMFit) -> Tuple[list, list, list, list]:

        jump_tables = ([], [], [], [])

        for funct in etm_results.design_matrix.functions:
            if funct.p.object == 'jump':
                jump_tables[3].append(funct)
                tables = funct.print_parameters()
                for i in range(3):
                    jump_tables[i].extend(tables[i])

        return jump_tables

    def _prepare_component_data(self, observations: np.ndarray, solution_data: SolutionData,
                                etm_results: ETMFit) -> List[ComponentData]:
        """Prepare data for one coordinate component"""
        data = []

        for i in range(3):
            data.append(ComponentData(
                observations=observations[i] * 1000, # convert to mm
                time_vector=solution_data.time_vector,
                time_range=(solution_data.time_vector.min(), solution_data.time_vector.max())
            ))

            # Add model values
            if etm_results.design_matrix is not None:
                model_values = (etm_results.design_matrix.alternate_time_vector(solution_data.time_vector_cont)
                                @ etm_results.results[i].parameters)
                data[i].model_time_vector = solution_data.time_vector_cont
                data[i].model_values = model_values * 1000  # Convert to mm

                if self.output_config.plot_remove_polynomial:
                    for funct in etm_results.design_matrix.functions:
                        if funct.p.object == 'polynomial':
                            data[i].observations -= funct.eval(i) * 1000.
                            data[i].model_values -= funct.eval(i, solution_data.time_vector_cont) * 1000.

                if self.output_config.plot_remove_periodic:
                    for funct in etm_results.design_matrix.functions:
                        if funct.p.object == 'periodic':
                            data[i].observations -= funct.eval(i) * 1000.
                            data[i].model_values -= funct.eval(i, solution_data.time_vector_cont) * 1000.

                # Add confidence bounds
                sigma = etm_results.results[i].wrms * 1000  # Convert to mm
                data[i].confidence_bounds = (
                    data[i].model_values - self.config.modeling.robust_lsq_limit * sigma,
                    data[i].model_values + self.config.modeling.robust_lsq_limit * sigma
                )

                # Add residuals
                data[i].residuals = etm_results.results[i].residuals

                # Add outlier flags
                data[i].outlier_flags = etm_results.outlier_flags

        return data

    @staticmethod
    def _generate_parameter_summary(etm_results: ETMFit) -> str:
        """Generate parameter summary string"""
        # This would integrate with the function objects to generate
        # formatted parameter summaries
        output = ['']

        for funct in etm_results.design_matrix.functions:
            if funct.p.object == 'polynomial':
                output, _, _ = funct.print_parameters()

        return output[0]

    @staticmethod
    def _generate_periodic_summary(etm_results: ETMFit) -> str:
        """Generate periodic parameter summary"""
        # This would integrate with periodic function to generate summary
        output = ['']

        for funct in etm_results.design_matrix.functions:
            if funct.p.object == 'periodic':
                output, _, _ = funct.print_parameters()

        return output[0]
