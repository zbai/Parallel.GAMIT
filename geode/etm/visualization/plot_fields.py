"""
Project: Geode
Date: 11/7/25 8:42 AM
Author: Demian D. Gomez
"""
import numpy as np
import warnings

from ...Utils import inverse_azimuthal

# Suppress the specific warning
warnings.filterwarnings('ignore', message='Starting a Matplotlib GUI outside')

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.basemap import Basemap
from typing import Literal, Optional, Tuple
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point


def mask_ocean_points(lon, lat, buffer_distance=0.0):
    """
    Use GeoPandas for land/ocean masking

    Parameters
    ----------
    lon, lat : array-like
        Point coordinates
    buffer_distance : float
        Distance to buffer land polygons (in km)
        Use 0 for exact land boundary
    """
    # Load Natural Earth land polygons
    try:
        # Python 3.9+
        from importlib.resources import files
        data_path = files('geode.elasticity.data').joinpath(
            'ne_50m_land.shp'
        )
        filename = str(data_path)
    except (ImportError, AttributeError):
        # Python 3.7-3.8 fallback
        from importlib.resources import path as resource_path
        import geode.elasticity.data

        with resource_path(geode.elasticity.data,
                           'ne_50m_land.shp') as p:
            filename = str(p)

    world = gpd.read_file(filename)

    # Buffer the land polygons if offset requested
    if buffer_distance > 0:
        # Project to equal-area projection for accurate buffering
        world_proj = world.to_crs('ESRI:54009')  # World Mollweide
        world_proj['geometry'] = world_proj.geometry.buffer(buffer_distance * 1000.)
        world = world_proj.to_crs('EPSG:4326')  # Back to lat/lon

    # Create GeoDataFrame of your points
    points = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lon, lat)],
        crs="EPSG:4326"
    )

    # Spatial join to find points on/near land
    points_on_land = gpd.sjoin(points, world, how='left', predicate='within')
    # Remove duplicates caused by overlapping polygons
    # Keep first occurrence of each point index
    points_on_land = points_on_land[~points_on_land.index.duplicated(keep='first')]
    is_land = ~points_on_land.index_right.isna()

    return is_land.values


def plot_velocity_field(
        lon: list,
        lat: list,
        data: list,
        stations_lon: np.ndarray = None,
        stations_lat: np.ndarray = None,
        stations_data: list = None,
        stations_names: list = None,
        on_station_click: Optional[callable] = None,
        available_fields: list = None,
        title: str = "GeoDE Deformation Visualizer",
        cmap: str = 'RdBu_r',
        figsize: Tuple[float, float] = (18, 6),
        dpi: int = 100,
        output_file: str = None,
        coastline_resolution: str = 'i',
        colorbar_extend: str = 'both',
        plot_sigmas: bool = False,
        covar: list = None
) -> plt.Figure:
    """
    Plot geodetic velocity or displacement field on a map with 3 subplots (East, North, Up).

    Parameters
    ----------
    lon : np.ndarray
        Longitude coordinates (1D array of length n)
    lat : np.ndarray
        Latitude coordinates (1D array of length n)
    data : np.ndarray
        Velocity/displacement data with shape (3, n) where:
        - data[0, :] = East component
        - data[1, :] = North component
        - data[2, :] = Up component
    on_station_click : callable, optional
        Callback function when station is clicked.
        Signature: callback(station_index, station_name, lon, lat)
    title : str, default="Velocity Field"
        Main title for the figure
    scale : float, optional
        Scale factor for vectors. If None, automatically determined.
    scale_label : str, default="mm/yr"
        Label for the scale/colorbar units
    cmap : str, default='RdBu_r'
        Colormap for contours
    figsize : tuple, default=(18, 6)
        Figure size (width, height) in inches
    dpi : int, default=100
        Figure resolution
    grid_resolution : int, default=50
        Number of grid points for interpolation in each direction
    quiver_subsample : int, default=1
        Subsample factor for vectors (1 = all points, 2 = every other point, etc.)
    reference_vector : float, optional
        Reference vector length for quiver key. If None, use max(data magnitude)
    output_file : str, optional
        If provided, save figure to this file
    coastline_resolution : str, default='i'
        Basemap coastline resolution: 'c' (crude), 'l' (low), 'i' (intermediate), 'h' (high), 'f' (full)
    colorbar_extend : str, default='both'
        Colorbar extension: 'neither', 'both', 'min', 'max'

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    """
    plt.ion()
    plot_state = {
        'quivers_grid': [None, None, None],  # Grid quivers for each subplot
        'quivers_stn': [None, None, None],  # Station quivers for each subplot
        'quiver_keys': [None, None, None],  # ← Add this
        'scale_factor': 10.0,  # Initial scale factor
        've_grid': None,
        'vn_grid': None,
        've_stn': None,
        'vn_stn': None,
        'reference_vector': None,
        'plot_sigmas': plot_sigmas
    }

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi,
                             sharex='all', sharey='all')
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Create basemaps for each subplot
    # use index = 0 which is interseismic (most extensive)
    basemaps = create_basemaps(axes, lon[0], lat[0], coastline_resolution)

    x_stn, y_stn = basemaps[0](stations_lon, stations_lat)

    # Extract components (starts with interseismic)
    (ve, vn, xi_grid, yi_grid, ve_grid, vn_grid, vu_grid, vh_grid, ve_stn, vn_stn, cov_grid) = (
        None, None, None, None, None, None, None, None,None, None, None)

    def process_data(index):
        nonlocal ve, vn, xi_grid, yi_grid, ve_grid, vn_grid, vu_grid, vh_grid, ve_stn, vn_stn, cov_grid

        # Convert coordinates to map projection
        x, y = basemaps[0](lon[index], lat[index])

        ve = data[index][0, :] * 1000. # East
        vn = data[index][1, :] * 1000. # North
        vu = data[index][2, :] * 1000. # Up

        ve_stn = stations_data[index][0, :] * 1000. # East
        vn_stn = stations_data[index][1, :] * 1000. # North

        # Create interpolation grid if needed for contours
        xi = np.linspace(x.min(), x.max(), int((x.max() - x.min()) / 20000)) # in m!
        yi = np.linspace(y.min(), y.max(), int((y.max() - y.min()) / 20000)) # in m!
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        valid = ~np.isnan(ve)
        # Interpolate fields to grid
        ve_grid = griddata((x[valid], y[valid]), ve[valid], (xi_grid, yi_grid), method='cubic')
        vn_grid = griddata((x[valid], y[valid]), vn[valid], (xi_grid, yi_grid), method='cubic')
        vu_grid = griddata((x[valid], y[valid]), vu[valid], (xi_grid, yi_grid), method='cubic')
        # mask points outside of land areas
        glon, glat = basemaps[0](xi_grid, yi_grid, inverse=True)
        mask = mask_ocean_points(glon.flatten(), glat.flatten(), 25)
        ve_grid.flat[~mask] = np.nan
        vn_grid.flat[~mask] = np.nan
        vu_grid.flat[~mask] = np.nan

        vh_grid = np.sqrt(ve_grid ** 2 + vn_grid ** 2)

        if plot_sigmas:
            cov_en = covar[index][0, :] * (1000.**2)
            cov_grid = griddata((x[valid], y[valid]), cov_en[valid], (xi_grid, yi_grid), method='cubic')
        else:
            cov_grid = 0

    process_data(0)

    labels = ['East', 'North', 'Up']
    colorbars = []
    #####################################################################
    # ==================== SUBPLOT 1: EAST COMPONENT ====================
    for i in range(3):
        cbar = plot_axes(axes, i, x_stn, y_stn, ve_stn, vn_stn, stations_names, on_station_click, labels[i],
                         xi_grid, yi_grid, ve_grid, vn_grid, vu_grid, cov_grid, cmap, colorbar_extend,
                         connect=True, plot_state=plot_state)
        colorbars.append(cbar)

    plt.tight_layout()

    ############################################################################
    # Make room on left
    fig.subplots_adjust(left=0.20)

    field_names = available_fields
    n_fields = len(field_names)

    # Calculate button positions
    button_width = 0.12
    button_height = 0.022
    button_spacing = 0.01
    total_width = n_fields * (button_height + button_spacing)
    start_y = (1 - total_width) / 2

    buttons = []
    current_field = {'name': field_names[0]}

    for i, field_name in enumerate(field_names):
        y_pos = start_y + i * (button_height + button_spacing)

        # Create button axis - well above plots
        ax_btn = fig.add_axes((0.01, y_pos, button_width, button_height))

        # Determine color
        color = 'lightgreen' if field_name == current_field['name'] else 'lightgray'

        btn = Button(ax_btn, field_name, color=color, hovercolor='yellow')
        btn.label.set_horizontalalignment('left')
        btn.label.set_position((0.05, 0.5))  # (x, y) in axis coordinates
        btn.label.set_fontsize(8)

        def make_callback(fn):
            def callback(ev):
                # Update colors
                for i, (b, name) in enumerate(zip(buttons, field_names)):
                    c = 'lightgreen' if name == fn else 'lightgray'
                    b.color = c
                    b.ax.set_facecolor(c)
                    if name == fn:
                        # get new data
                        for cbar in colorbars:
                            cbar.remove()
                        colorbars.clear()
                        for j in range(3):
                            axes[j].clear()

                        process_data(i)

                        create_basemaps(axes, lon[0], lat[0], coastline_resolution)
                        for j in range(3):
                            cbar = plot_axes(axes, j, x_stn, y_stn, ve_stn, vn_stn, stations_names,
                                             on_station_click, labels[j], xi_grid, yi_grid,
                                             ve_grid, vn_grid, vu_grid, cov_grid, cmap, colorbar_extend,
                                             connect=False, plot_state=plot_state)
                            colorbars.append(cbar)

                current_field['name'] = fn

                fig.canvas.draw_idle()

            return callback

        btn.on_clicked(make_callback(field_name))
        buttons.append(btn)

    fig._buttons = buttons

    ############################################################################
    # Create slider axis
    ax_slider = fig.add_axes((0.25, 0.05, 0.5, 0.03))  # [left, bottom, width, height]

    # Create slider
    # Scale range: 1 to 100 (lower = bigger arrows)
    slider = Slider(
        ax=ax_slider,
        label='Vector Scale',
        valmin=1,
        valmax=100,
        valinit=50,  # Initial value
        valstep=0.5
    )

    def update_scale(val):
        """Update quiver scale when slider changes"""
        scale_factor = val * 10 / 50
        plot_state['scale_factor'] = scale_factor

        reference_vector = plot_state['reference_vector']
        scale = reference_vector * scale_factor

        if plot_state['plot_sigmas']:
            scale = scale / reference_vector

        # Update quivers for each subplot
        for i in range(2):
            ax = axes[i]

            # Remove old quivers
            if plot_state['plot_sigmas']:
                for patch in plot_state['quivers_grid'][i]:
                    patch.remove()

                step = 4
                xi_flat = xi_grid[::step, ::step].flatten()
                yi_flat = yi_grid[::step, ::step].flatten()
                ve_flat = ve_grid[::step, ::step].flatten()
                vn_flat = vn_grid[::step, ::step].flatten()
                cov_flat = cov_grid[::step, ::step].flatten()

                q = []
                for j in range(ve_flat.size):
                    # Build 2x2 covariance matrix for this point
                    cov_2x2 = np.array([
                        [ve_flat[j]**2, cov_flat[j]],
                        [cov_flat[j], vn_flat[j]**2]
                    ])

                    # Eigenvalue decomposition
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)

                    # Sort by eigenvalue
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]
                    width  = 2 * 3 * np.sqrt(eigenvalues[0]) * scale * 10000
                    height = 2 * 3 * np.sqrt(eigenvalues[1]) * scale * 10000
                    # Angle from eigenvector
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    ellipse = Ellipse(xy=(xi_flat[j], yi_flat[j]), width=width, height=height, angle=angle,
                                      edgecolor='black', facecolor='none', linewidth=1)
                    q.append(ax.add_patch(ellipse))

                plot_state['quivers_grid'][i] = q
            else:
                if plot_state['quivers_grid'][i] is not None:
                    plot_state['quivers_grid'][i].remove()
                    plot_state['quiver_keys'][i].remove()

                if plot_state['quivers_stn'][i] is not None:
                    plot_state['quivers_stn'][i].remove()

                # Recreate quivers with new scale (only for East and North)
                # Grid quivers
                q_grid = ax.quiver(xi_grid, yi_grid,
                                   ve_grid, vn_grid,
                                   scale=scale,
                                   color='white', width=0.003, alpha=0.6,
                                   edgecolor='black', linewidth=0.25,
                                   zorder=5)
                plot_state['quivers_grid'][i] = q_grid

                q = add_quiver_key(ax, q_grid, plot_state['reference_vector'], 'mm[/yr]')
                plot_state['quiver_keys'][i] = q

            # Station quivers
            q_stn = ax.quiver(x_stn, y_stn,
                              ve_stn, vn_stn,
                              scale=scale,
                              color='white', width=0.003,
                              edgecolor='black', linewidth=0.25,
                              zorder=6)
            plot_state['quivers_stn'][i] = q_stn

        # print(f"Scale factor: {scale_factor:.1f} (scale={scale:.1f})")
        fig.canvas.draw_idle()

    slider.on_changed(update_scale)
    fig._slider = slider  # Keep reference

    ############################################################################
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    plt.show()
    return fig

def create_basemaps(axes, lon, lat, coastline_resolution):

    # Determine map bounds with padding
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    lon_pad = (lon_max - lon_min) * 0.1
    lat_pad = (lat_max - lat_min) * 0.1

    basemaps = []
    for ax in axes:
        m = Basemap(
            projection='merc',
            llcrnrlon=lon_min - lon_pad,
            llcrnrlat=lat_min - lat_pad,
            urcrnrlon=lon_max + lon_pad,
            urcrnrlat=lat_max + lat_pad,
            resolution=coastline_resolution,
            ax=ax
        )
        m.drawcoastlines(linewidth=0.5)
        m.fillcontinents(color='wheat', lake_color='lightblue', alpha=0.3)
        m.drawcountries(linewidth=0.5)
        m.drawstates(linewidth=0.5)
        m.drawparallels(np.arange(lat_min, lat_max, (lat_max - lat_min) / 4),
                        labels=[1, 0, 0, 0], fontsize=8, linewidth=0.3)
        m.drawmeridians(np.arange(lon_min, lon_max, (lon_max - lon_min) / 4),
                        labels=[0, 0, 0, 1], fontsize=8, linewidth=0.3)
        basemaps.append(m)

    return basemaps

def plot_axes(axes, index, x_stn, y_stn, ve_stn, vn_stn, stations_names, on_station_click,
              label, xi_grid, yi_grid, ve_grid, vn_grid, vu_grid, cov_grid, cmap, colorbar_extend,
              connect, plot_state):

    # Contours
    ax = axes[index]

    # Compute horizontal magnitude
    vh = np.sqrt(ve_grid ** 2 + vn_grid ** 2)

    # Determine reference vector for quiver key
    reference_vector = np.nanmax(vh)
    plot_state['reference_vector'] = reference_vector
    scale = reference_vector * plot_state['scale_factor']

    if index == 0:
        field = ve_grid
    elif index == 1:
        field = vn_grid
    else:
        field = vu_grid

    vmin_plot = np.nanmin(field) #float(np.nanpercentile(field, 0.1))
    vmax_plot = np.nanmax(field) #float(np.nanpercentile(field, 99.9))

    if np.abs(vmax_plot) > np.abs(vmin_plot):
        vmin_plot = -vmax_plot
    else:
        vmax_plot = -vmin_plot

    if vmin_plot == vmax_plot:
        vmin_plot = -1
        vmax_plot =  1

    levels = np.linspace(vmin_plot, vmax_plot, 21)
    # norm = TwoSlopeNorm(vmin=vmin_plot, vcenter=0, vmax=vmax_plot)

    cs_east = ax.contourf(xi_grid, yi_grid, field, levels=levels, cmap=cmap, extend=colorbar_extend)
    cbar = plt.colorbar(cs_east, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
    cbar.set_label(label, fontsize=9)

    # Vectors on top
    if index != 2:
        if plot_state['plot_sigmas']:
            step = 4
            xi_flat = xi_grid[::step, ::step].flatten()
            yi_flat = yi_grid[::step, ::step].flatten()
            ve_flat = ve_grid[::step, ::step].flatten()
            vn_flat = vn_grid[::step, ::step].flatten()
            cov_flat = cov_grid[::step, ::step].flatten()

            # normalize the data so that ellipses from very dissimilar
            # field look about the same size
            # user can deduce magnitude from the contour field
            scale = scale / reference_vector

            q = []
            for j in range(ve_flat.size):
                # Build 2x2 covariance matrix for this point
                cov_2x2 = np.array([
                    [ve_flat[j]**2, cov_flat[j]],
                    [cov_flat[j], vn_flat[j]**2]
                ])

                # Eigenvalue decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)

                # Sort by eigenvalue
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
                width  = 2 * 3 * np.sqrt(eigenvalues[0]) * scale * 10000
                height = 2 * 3 * np.sqrt(eigenvalues[1]) * scale * 10000
                # Angle from eigenvector
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                ellipse = Ellipse(xy=(xi_flat[j], yi_flat[j]), width=width, height=height, angle=angle,
                                  edgecolor='black', facecolor='none', linewidth=1)
                q.append(ax.add_patch(ellipse))

            plot_state['quivers_grid'][index] = q
        else:
            q = ax.quiver(xi_grid, yi_grid, ve_grid, vn_grid, scale=scale,
                          color='white', width=0.003, alpha=0.6, edgecolor='black', linewidth=0.25)
            plot_state['quivers_grid'][index] = q
            q = add_quiver_key(ax, q, reference_vector, 'mm[/yr]')
            plot_state['quiver_keys'][index] = q

    if index == 0:
        add_interactive_labels(ax, x_stn, y_stn, stations_names, on_station_click, connect)

    ax.scatter(x_stn, y_stn)

    if index != 2:
        q = ax.quiver(x_stn, y_stn, ve_stn, vn_stn, scale=scale,
                      color='white', width=0.003, edgecolor='black', linewidth=0.25)
        plot_state['quivers_stn'][index] = q

    ax.set_title(label, fontsize=12, fontweight='bold')

    if index == 2:
        # Add contour lines
        cs_lines = ax.contour(xi_grid, yi_grid, field, levels=10, colors='black',
                              linewidths=0.5, alpha=0.4)
        ax.clabel(cs_lines, inline=True, fontsize=7, fmt='%1.1f')

    return cbar


def add_quiver_key(ax, q, reference_val, label):
    # Function to add quiver key
    q = ax.quiverkey(q, 0.85, 0.95, reference_val,
                     f'{reference_val:.1f} {label}',
                     labelpos='E', coordinates='axes',
                     fontproperties={'size': 8})
    return q

def add_interactive_labels(ax, x, y, station_names, on_click_callback=None, connect=True):
    """
    Add interactive labels that appear on mouse hover

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to add labels to
    x, y : np.ndarray
        Coordinates of stations in axis units
    station_names : list
        List of station names
    """
    # Create scatter plot for detecting hover events
    scatter = ax.scatter(x, y, s=20, c='red', marker='^',
                         alpha=0.6, zorder=100, picker=True)

    # Create annotation (initially invisible)
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.9),
        fontsize=8,
        zorder=101
    )
    annot.set_visible(False)

    def on_hover(event):
        """Show label when hovering over a station"""
        if event.inaxes == ax:
            # Check if mouse is over a point
            cont, ind = scatter.contains(event)
            if cont:
                # Get index of the point
                idx = ind["ind"][0]

                # Update annotation
                annot.xy = (x[idx], y[idx])
                annot.set_text(station_names[idx])
                annot.set_visible(True)
                ax.figure.canvas.draw_idle()
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    ax.figure.canvas.draw_idle()

    def on_click(event):
        """Handle click events on stations"""
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont and on_click_callback is not None:
                idx = ind["ind"][0]
                # Call the callback with station info
                on_click_callback(idx)

    # Connect events
    ax.figure.canvas.mpl_connect("motion_notify_event", on_hover)
    if connect:
        ax.figure.canvas.mpl_connect("button_press_event", on_click)

    return scatter, annot


