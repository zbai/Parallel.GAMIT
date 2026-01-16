"""
Project: Geodesy Database Engine (GeoDE)
Date: 9/22/25 8:39 AM
Author: Demian D. Gomez
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple
from matplotlib.patches import Ellipse

from ..visualization.data_prep import PlotTemplate
from ..visualization.data_classes import TimeSeriesPlotData, PlotOutputConfig


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

    def generate_title(self, plot_data: TimeSeriesPlotData, output_config: PlotOutputConfig) -> str:
        """Generate histogram plot title"""
        base_title = super().generate_title(plot_data, output_config)

        if plot_data.covariance_matrix is not None:
            covar = plot_data.covariance_matrix
            title_parts = [base_title,
                           f"VAR (N E U)      : {np.diag(covar)[0]:10.3e} {np.diag(covar)[1]:10.3e} {np.diag(covar)[2]:10.3e}",
                           f"COV (N-E N-U E-U): {covar[0, 1]:10.3e} {covar[0, 2]:10.3e} {covar[1, 2]:10.3e}"]
            return '\n'.join(title_parts)

        return base_title

    def plot_ne_scatter(self, fig, ax, plot_data: TimeSeriesPlotData) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Plot North-East scatter with error ellipse"""
        # Filter out extreme outliers for better visualization
        n_res = plot_data.north_data.residuals
        e_res = plot_data.east_data.residuals

        # Use only residuals within reasonable bounds (5cm)
        combined_residuals = np.sqrt(n_res ** 2 + e_res ** 2)
        mask = combined_residuals <= np.sqrt(np.sum(np.square(plot_data.wrms_values))) * 3.5 # 50mm

        ax.plot(e_res[mask], n_res[mask], 'o', color=(0, 150 / 255, 235 / 255), markersize=2)

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

        # draw to get actual xlim and ylim
        fig.canvas.draw()

        return ax.get_xlim(), ax.get_ylim()

    def plot_component_histograms(self, axes_dict: Dict[str, plt.Axes],
                                  plot_data: TimeSeriesPlotData,
                                  xlim: Tuple[float, float],
                                  ylim: Tuple[float, float]) -> None:
        """Plot histograms for each component"""
        components = [
            ('n_hist', plot_data.north_data.residuals, self.config.get_label('north'),
             plot_data.north_data.outlier_flags),
            ('e_hist', plot_data.east_data.residuals, self.config.get_label('east'),
             plot_data.east_data.outlier_flags),
            ('u_hist', plot_data.up_data.residuals, self.config.get_label('up'),
             plot_data.up_data.outlier_flags)
        ]

        from scipy.stats import norm

        for ax_key, residuals, label, outliers in components:
            ax = axes_dict[ax_key]

            # Filter extreme outliers
            mask = np.abs(residuals) <= np.sqrt(np.sum(np.square(plot_data.wrms_values))) * 3.5  # 100mm
            filtered_residuals = residuals[mask]

            # Create histogram
            ax.hist(filtered_residuals, bins=100, alpha=0.75, density=True,
                    color=(0, 150 / 255, 235 / 255), orientation='horizontal' if 'n_hist' in ax_key else 'vertical')

            if ax_key in ('e_hist', 'u_hist'):
                xmin, xmax = ax.get_xlim()
            else:
                xmin, xmax = ax.get_ylim()

            mu, std = norm.fit(residuals[outliers])
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            if 'n_hist' in ax_key:
                ax.plot(p, x, 'r', linewidth=2,
                        label=f'Fit: mu={mu:.2f}, std={std:.2f}')
            else:
                ax.plot(x, p, 'r', linewidth=2,
                        label=f'Fit: mu={mu:.2f}, std={std:.2f}')

            ax.grid(True)

            if ax_key == 'n_hist':
                ax.set_xlabel(self.config.get_label('frequency'))
                ax.set_ylabel(f"{label} {self.config.get_label('N residuals')} [mm]")
                ax.set_ylim(ylim)

            else:
                ax.set_ylabel(self.config.get_label('frequency'))
                ax.set_xlabel(f"{label} residuals [mm]")
                if ax_key != 'u_hist':
                    ax.set_xlim(xlim)

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
                          facecolor='none', edgecolor='red', linewidth=2, zorder=3,
                          label=r'$2.5\sigma$')
        ax.add_patch(ellipse)
        ax.legend()