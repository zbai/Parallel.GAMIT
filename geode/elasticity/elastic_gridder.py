#!/usr/bin/env python
"""
Elastic Gridder - Wrapper for creating ETM constraint functions from elastic interpolation

This module provides a wrapper class that uses elastic interpolation (Sandwell & Wessel 2016)
to sample velocity or displacement fields from model grids and create ETM constraint functions.

Supports both plain text and GGXF (Gridded Geodetic data eXchange Format) file formats.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import logging
from datetime import datetime
from pathlib import Path

# Local imports for elastic interpolation
from elastic_interpolation import interpolate_at_points

logger = logging.getLogger(__name__)

# Optional GGXF support (NetCDF format)
try:
    from netCDF4 import Dataset
    GGXF_AVAILABLE = True
except ImportError:
    GGXF_AVAILABLE = False
    logger.warning("netCDF4 not available - GGXF format support disabled. "
                  "Install with: pip install netCDF4")


def read_ggxf_grid(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read velocity or displacement grid from GGXF (NetCDF) format
    
    GGXF is the OGC standard for gridded geodetic data exchange based on NetCDF/HDF5.
    
    Parameters
    ----------
    filename : str
        Path to GGXF file (.nc extension)
        
    Returns
    -------
    lon : np.ndarray
        Longitude coordinates (degrees, 1D array)
    lat : np.ndarray
        Latitude coordinates (degrees, 1D array)
    east : np.ndarray
        East component grid (mm or mm/yr, 2D array)
    north : np.ndarray
        North component grid (mm or mm/yr, 2D array)
        
    Notes
    -----
    The GGXF format stores gridded geodetic data with comprehensive metadata.
    This function extracts the essential grid data for interpolation.
    
    Expected GGXF structure:
    - dimensions: latitude, longitude
    - variables: latitude(lat), longitude(lon), east_component(lat,lon), north_component(lat,lon)
    """
    if not GGXF_AVAILABLE:
        raise ImportError("netCDF4 library required for GGXF support. Install with: pip install netCDF4")
    
    logger.info(f'Reading GGXF file: {filename}')
    
    with Dataset(filename, 'r') as nc:
        # Read coordinate variables
        # Standard GGXF uses 'latitude' and 'longitude' dimension names
        if 'latitude' in nc.dimensions and 'longitude' in nc.dimensions:
            lat = nc.variables['latitude'][:]
            lon = nc.variables['longitude'][:]
        elif 'lat' in nc.dimensions and 'lon' in nc.dimensions:
            lat = nc.variables['lat'][:]
            lon = nc.variables['lon'][:]
        else:
            raise ValueError("GGXF file must have 'latitude'/'longitude' or 'lat'/'lon' dimensions")
        
        # Try to find east and north component variables
        # Common naming conventions
        east_names = ['east_component', 'east_velocity', 'east_displacement', 
                     'eastComponent', 'east', 've', 'de']
        north_names = ['north_component', 'north_velocity', 'north_displacement',
                      'northComponent', 'north', 'vn', 'dn']
        
        east = None
        north = None
        
        for name in east_names:
            if name in nc.variables:
                east = nc.variables[name][:]
                logger.info(f'Found east component: {name}')
                break
        
        for name in north_names:
            if name in nc.variables:
                north = nc.variables[name][:]
                logger.info(f'Found north component: {name}')
                break
        
        if east is None or north is None:
            # List available variables to help user
            available = list(nc.variables.keys())
            raise ValueError(f"Could not find east/north components. Available variables: {available}")
        
        # Log metadata if available
        if hasattr(nc, 'title'):
            logger.info(f'Grid title: {nc.title}')
        if hasattr(nc, 'description'):
            logger.info(f'Description: {nc.description}')
        
        logger.info(f'Grid size: {len(lat)} x {len(lon)}')
        logger.info(f'Lat range: [{lat.min():.4f}, {lat.max():.4f}]')
        logger.info(f'Lon range: [{lon.min():.4f}, {lon.max():.4f}]')
        
        return lon, lat, east, north


def write_ggxf_grid(filename: str,
                   lon: np.ndarray, lat: np.ndarray,
                   east: np.ndarray, north: np.ndarray,
                   metadata: Optional[dict] = None) -> None:
    """
    Write velocity or displacement grid to GGXF (NetCDF) format
    
    Creates an OGC-compliant GGXF file for gridded geodetic data.
    
    Parameters
    ----------
    filename : str
        Output GGXF file path (.nc extension recommended)
    lon : np.ndarray
        Longitude coordinates (degrees, 1D array)
    lat : np.ndarray
        Latitude coordinates (degrees, 1D array)
    east : np.ndarray
        East component grid (mm or mm/yr, 2D array shape [nlat, nlon])
    north : np.ndarray
        North component grid (mm or mm/yr, 2D array shape [nlat, nlon])
    metadata : dict, optional
        Additional metadata to include in file:
        - 'title': Grid title
        - 'description': Description of the data
        - 'data_type': 'velocity' or 'displacement'
        - 'units': 'mm/yr' or 'mm'
        - 'source': Data source/reference
        - 'interpolation_method': e.g., 'elastic_coupling'
        - 'poisson_ratio': Poisson's ratio used
        - 'min_distance': Minimum distance parameter
        
    Notes
    -----
    Creates a NetCDF4 file following GGXF conventions for gridded geodetic data.
    The format is self-describing and includes comprehensive metadata.
    
    Examples
    --------
    >>> write_ggxf_grid('velocity.nc', lon, lat, ve, vn, metadata={
    ...     'title': 'Interseismic Velocity Model',
    ...     'data_type': 'velocity',
    ...     'units': 'mm/yr',
    ...     'interpolation_method': 'elastic_coupling',
    ...     'poisson_ratio': 0.5
    ... })
    """
    if not GGXF_AVAILABLE:
        raise ImportError("netCDF4 library required for GGXF support. Install with: pip install netCDF4")
    
    if metadata is None:
        metadata = {}
    
    logger.info(f'Writing GGXF file: {filename}')
    
    # Create NetCDF file
    with Dataset(filename, 'w', format='NETCDF4') as nc:
        # Global attributes (GGXF standard)
        nc.Conventions = 'CF-1.8, GGXF-1.0'
        nc.title = metadata.get('title', 'Gridded Geodetic Data')
        nc.institution = metadata.get('institution', 'Generated by ElasticGridder')
        nc.source = metadata.get('source', 'Elastic interpolation (Sandwell & Wessel 2016)')
        nc.history = f'{datetime.now().isoformat()}: Created by ElasticGridder'
        nc.references = metadata.get('references', 
                                    'Sandwell & Wessel (2016) doi:10.1002/2016GL070340')
        nc.comment = metadata.get('description', 'Gridded geodetic data from elastic interpolation')
        
        # GGXF-specific attributes
        nc.ggxf_version = '1.0'
        nc.data_type = metadata.get('data_type', 'velocity')
        
        # Interpolation metadata
        if 'interpolation_method' in metadata:
            nc.interpolation_method = metadata['interpolation_method']
        if 'poisson_ratio' in metadata:
            nc.poisson_ratio = metadata['poisson_ratio']
        if 'min_distance' in metadata:
            nc.min_distance_km = metadata['min_distance']
        
        # Create dimensions
        nc.createDimension('latitude', len(lat))
        nc.createDimension('longitude', len(lon))
        
        # Create coordinate variables
        lat_var = nc.createVariable('latitude', 'f8', ('latitude',))
        lat_var.long_name = 'latitude'
        lat_var.standard_name = 'latitude'
        lat_var.units = 'degrees_north'
        lat_var.axis = 'Y'
        lat_var[:] = lat
        
        lon_var = nc.createVariable('longitude', 'f8', ('longitude',))
        lon_var.long_name = 'longitude'
        lon_var.standard_name = 'longitude'
        lon_var.units = 'degrees_east'
        lon_var.axis = 'X'
        lon_var[:] = lon
        
        # Create data variables
        units = metadata.get('units', 'mm/yr')
        data_type = metadata.get('data_type', 'velocity')
        
        # East component
        east_var = nc.createVariable('east_component', 'f4', ('latitude', 'longitude'),
                                    fill_value=np.nan, zlib=True, complevel=4)
        east_var.long_name = f'East component of {data_type}'
        east_var.standard_name = f'{data_type}_east'
        east_var.units = units
        east_var.grid_mapping = 'crs'
        east_var.coordinates = 'latitude longitude'
        east_var[:] = east
        
        # North component  
        north_var = nc.createVariable('north_component', 'f4', ('latitude', 'longitude'),
                                     fill_value=np.nan, zlib=True, complevel=4)
        north_var.long_name = f'North component of {data_type}'
        north_var.standard_name = f'{data_type}_north'
        north_var.units = units
        north_var.grid_mapping = 'crs'
        north_var.coordinates = 'latitude longitude'
        north_var[:] = north
        
        # Add CRS variable (WGS84)
        crs = nc.createVariable('crs', 'i4')
        crs.grid_mapping_name = 'latitude_longitude'
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.crs_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        
        logger.info(f'Successfully wrote GGXF file with {len(lat)}x{len(lon)} grid')


def load_grid_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load grid from either plain text or GGXF format (auto-detect)
    
    Parameters
    ----------
    filename : str
        Path to grid file (.txt, .dat, .csv, or .nc)
        
    Returns
    -------
    lon : np.ndarray
        Longitude coordinates (degrees)
    lat : np.ndarray
        Latitude coordinates (degrees)
    east : np.ndarray
        East component (mm or mm/yr)
    north : np.ndarray
        North component (mm or mm/yr)
        
    Notes
    -----
    Automatically detects file format based on extension:
    - .nc → GGXF (NetCDF) format
    - .txt, .dat, .csv, or other → Plain text (lon, lat, east, north)
    """
    path = Path(filename)
    
    if path.suffix.lower() in ['.nc', '.nc4', '.netcdf']:
        # GGXF format
        lon_grid, lat_grid, east_grid, north_grid = read_ggxf_grid(filename)
        # Convert 2D grids to coordinate arrays and flattened data
        lon = lon_grid
        lat = lat_grid
        # For plain text compatibility, we'll need to flatten
        # But for now return as-is
        return lon, lat, east_grid, north_grid
    else:
        # Plain text format: lon lat east north
        logger.info(f'Reading plain text grid from {filename}')
        data = np.loadtxt(filename, dtype=float)
        lon = data[:, 0]
        lat = data[:, 1]
        east = data[:, 2]
        north = data[:, 3]
        return lon, lat, east, north


class ElasticGridder:
    """
    Wrapper class for elastic interpolation of GPS velocities
    
    Creates constraint or prefit model functions for use in EtmEngine by sampling
    model grids using elastic interpolation with coupling between velocity components.
    
    Attributes
    ----------
    config : EtmConfig
        Configuration object containing station metadata
    dr : float
        Minimum distance parameter for elastic interpolation (km)
    nu : float
        Poisson's ratio controlling elastic coupling
    
    Examples
    --------
    >>> from geode.etm.core.etm_config import EtmConfig
    >>> config = EtmConfig('net', 'stn', cnn=cnn)
    >>> gridder = ElasticGridder(config, dr=8.0, nu=0.5)
    >>> 
    >>> # Create velocity constraint from model grid
    >>> poly_func = gridder.create_velocity_constraint(
    ...     'velocity_model.txt',
    ...     model_type='poly'
    ... )
    >>> 
    >>> # Use as constraint in EtmEngine
    >>> config.modeling.least_squares_strategy.constraints = [poly_func]
    """
    
    def __init__(self, config, dr: float = 8.0, nu: float = 0.5):
        """
        Initialize ElasticGridder
        
        Parameters
        ----------
        config : EtmConfig
            Configuration object with station metadata (lat, lon)
        dr : float, optional
            Minimum distance parameter in km (default 8.0)
            Should be roughly the mean spacing of grid points
        nu : float, optional
            Poisson's ratio (default 0.5 for typical elastic)
            -1.0 = decoupled, 0.5 = elastic, 1.0 = incompressible
        """
        self.config = config
        self.dr = dr  # minimum distance in km
        self.nu = nu  # Poisson's ratio
        
        logger.info(f'ElasticGridder initialized with dr={dr} km, nu={nu}')
    
    def create_velocity_constraint(self, 
                                  grid_file: str,
                                  model_type: str = 'poly',
                                  sigma: float = 0.001) -> 'EtmFunction':
        """
        Create a velocity constraint function from a model grid
        
        Reads a velocity model grid (plain text or GGXF format), interpolates to the 
        station location using elastic interpolation, and creates an ETM function that 
        can be used as a constraint in the least squares adjustment.
        
        Parameters
        ----------
        grid_file : str
            Path to model grid file:
            - Plain text: lon, lat, ve, vn (mm/yr) - space/tab separated
            - GGXF: NetCDF file with .nc extension
        model_type : str, optional
            Type of ETM function to create (default 'poly' for PolynomialFunction)
        sigma : float, optional
            Uncertainty for constraint in m/yr (default 0.001 = 1 mm/yr)
            
        Returns
        -------
        constraint : EtmFunction
            Constraint function (PolynomialFunction) with velocity populated
            
        Notes
        -----
        File formats:
        
        Plain text (.txt, .dat, .csv):
            lon(deg)  lat(deg)  ve(mm/yr)  vn(mm/yr)
            
        GGXF (.nc):
            NetCDF file following GGXF standard with:
            - dimensions: latitude, longitude
            - variables: east_component, north_component
            
        The function:
        1. Loads grid (auto-detects format from extension)
        2. Projects grid to local coordinates centered at station
        3. Interpolates velocities using elastic method
        4. Creates PolynomialFunction with velocity term populated
        """
        from geode.etm.etm_functions.polynomial import PolynomialFunction
        from geode.Utils import azimuthal_equidistant
        
        # Load the model grid (auto-detect format)
        logger.info(f'Loading model grid from {grid_file}')
        file_path = Path(grid_file)
        
        if file_path.suffix.lower() in ['.nc', '.nc4', '.netcdf']:
            # GGXF format
            lon_grid, lat_grid, ve_grid, vn_grid = read_ggxf_grid(grid_file)
            # Create meshgrid if 1D coords provided
            if lon_grid.ndim == 1 and lat_grid.ndim == 1:
                lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
                lon_flat = lon_2d.flatten()
                lat_flat = lat_2d.flatten()
                ve_flat = ve_grid.flatten()
                vn_flat = vn_grid.flatten()
            else:
                lon_flat = lon_grid.flatten()
                lat_flat = lat_grid.flatten()
                ve_flat = ve_grid.flatten()
                vn_flat = vn_grid.flatten()
        else:
            # Plain text format
            model_grid = np.loadtxt(grid_file, dtype=float)
            lon_flat = model_grid[:, 0]
            lat_flat = model_grid[:, 1]
            ve_flat = model_grid[:, 2]  # east component (mm/yr)
            vn_flat = model_grid[:, 3]  # north component (mm/yr)
        
        # Get station coordinates
        lat = self.config.metadata.lat
        lon = self.config.metadata.lon
        
        logger.info(f'Station location: {lat:.4f}°N, {lon:.4f}°E')
        
        # Project grid onto station position using azimuthal equidistant
        # Returns x, y in km
        x, y = azimuthal_equidistant(lon, lat, lon_flat, lat_flat)
        
        logger.info(f'Grid has {len(x)} points')
        logger.info(f'Grid velocity range: ve=[{ve_flat.min():.2f}, {ve_flat.max():.2f}], '
                   f'vn=[{vn_flat.min():.2f}, {vn_flat.max():.2f}] mm/yr')
        
        # Interpolate at station location (x=0, y=0) using elastic interpolation
        ve_interp, vn_interp = interpolate_at_points(
            xi=x, yi=y,
            ui=ve_flat, vi=vn_flat,
            xo=np.array([0.0]), yo=np.array([0.0]),
            dr=self.dr, nu=self.nu
        )
        
        logger.info(f'Interpolated velocity at station: ve={ve_interp[0]:.2f}, vn={vn_interp[0]:.2f} mm/yr')
        
        # Create polynomial function
        if model_type == 'poly':
            funct = PolynomialFunction(self.config)
            funct.initialize(time_vector=np.array([0]))
            
            # Set velocity parameters (index 1 in polynomial)
            # NEU order: index 0=north, 1=east, 2=up
            funct.p.params[0] = np.array([np.nan, vn_interp[0] / 1000])  # north, mm/yr -> m/yr
            funct.p.params[1] = np.array([np.nan, ve_interp[0] / 1000])  # east, mm/yr -> m/yr
            funct.p.params[2] = np.array([np.nan, np.nan])              # up, no constraint
            
            # Set uncertainties (same units as params)
            funct.p.sigmas[0] = np.array([sigma])  # north uncertainty (m/yr)
            funct.p.sigmas[1] = np.array([sigma])  # east uncertainty (m/yr)
            funct.p.sigmas[2] = np.array([sigma])  # up uncertainty (m/yr)
            
            logger.info(f'Created PolynomialFunction constraint with velocity constraint')
        else:
            raise ValueError(f'Unsupported model_type: {model_type}')
            
        return funct
    
    def create_jump_constraint(self,
                             grid_file: str,
                             jump_date: 'Date',
                             jump_type: 'JumpType' = None,
                             relaxation: Optional[float] = None,
                             sigma: float = 0.001) -> 'EtmFunction':
        """
        Create a jump constraint function from a coseismic displacement grid
        
        Parameters
        ----------
        grid_file : str
            Path to model grid file with columns: lon, lat, de, dn (mm)
        jump_date : Date
            Date of the jump (earthquake)
        jump_type : JumpType, optional
            Type of jump (COSEISMIC_ONLY or POSTSEISMIC_ONLY)
        relaxation : float, optional
            Relaxation time constant in years (only for POSTSEISMIC_ONLY)
        sigma : float, optional
            Uncertainty for constraint in m (default 0.001 = 1 mm)
            
        Returns
        -------
        constraint : JumpFunction
            Jump constraint function with amplitude populated
        """
        from geode.etm.etm_functions.jumps import JumpFunction
        from geode.etm.core.type_declarations import JumpType
        from geode.Utils import azimuthal_equidistant
        
        # Load the model grid (displacement in mm)
        logger.info(f'Loading displacement grid from {grid_file}')
        model_grid = np.loadtxt(grid_file, dtype=float)
        
        # Get station coordinates
        lat = self.config.metadata.lat
        lon = self.config.metadata.lon
        
        # Project grid to local coordinates
        x, y = azimuthal_equidistant(lon, lat, model_grid[:, 0], model_grid[:, 1])
        
        # Extract displacement components (mm)
        de_grid = model_grid[:, 2]  # east component
        dn_grid = model_grid[:, 3]  # north component
        
        logger.info(f'Grid displacement range: de=[{de_grid.min():.2f}, {de_grid.max():.2f}], '
                   f'dn=[{dn_grid.min():.2f}, {dn_grid.max():.2f}] mm')
        
        # Interpolate at station location
        de_interp, dn_interp = interpolate_at_points(
            xi=x, yi=y,
            ui=de_grid, vi=dn_grid,
            xo=np.array([0.0]), yo=np.array([0.0]),
            dr=self.dr, nu=self.nu
        )
        
        logger.info(f'Interpolated displacement: de={de_interp[0]:.2f}, dn={dn_interp[0]:.2f} mm')
        
        # Determine jump type
        if jump_type is None:
            if relaxation is not None:
                jump_type = JumpType.POSTSEISMIC_ONLY
            else:
                jump_type = JumpType.COSEISMIC_ONLY
        
        # Create jump function
        funct = JumpFunction(
            self.config,
            time_vector=np.array([0]),
            date=jump_date,
            jump_type=jump_type,
            fit=True
        )
        
        # Set jump parameters (displacement in meters)
        # NEU order: index 0=north, 1=east, 2=up
        funct.p.params[0] = np.array([dn_interp[0] / 1000])  # north, mm -> m
        funct.p.params[1] = np.array([de_interp[0] / 1000])  # east, mm -> m
        funct.p.params[2] = np.array([np.nan])               # up, no constraint
        
        # Set uncertainties
        funct.p.sigmas[0] = np.array([sigma])  # north
        funct.p.sigmas[1] = np.array([sigma])  # east
        funct.p.sigmas[2] = np.array([sigma])  # up
        
        # Set relaxation if provided
        if relaxation is not None:
            funct.p.relaxation = np.array([relaxation])
            logger.info(f'Created JumpFunction with relaxation={relaxation} years')
        else:
            logger.info(f'Created JumpFunction (coseismic only)')
        
        return funct
    
    def sample_grid_at_locations(self,
                                grid_file: str,
                                lons: np.ndarray,
                                lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample velocity grid at multiple locations using elastic interpolation
        
        Useful for generating velocity fields for visualization or further analysis.
        
        Parameters
        ----------
        grid_file : str
            Path to model grid file
        lons : np.ndarray
            Longitudes of sampling locations (degrees)
        lats : np.ndarray
            Latitudes of sampling locations (degrees)
            
        Returns
        -------
        ve_samples : np.ndarray
            East velocities at sample locations (mm/yr)
        vn_samples : np.ndarray
            North velocities at sample locations (mm/yr)
        """
        from geode.Utils import azimuthal_equidistant
        
        # Load grid
        model_grid = np.loadtxt(grid_file, dtype=float)
        
        # Use center of sampling region as projection center
        lon_center = np.mean(lons)
        lat_center = np.mean(lats)
        
        # Project grid to local coordinates
        x_grid, y_grid = azimuthal_equidistant(
            lon_center, lat_center, 
            model_grid[:, 0], model_grid[:, 1]
        )
        
        # Project sample locations
        x_sample, y_sample = azimuthal_equidistant(
            lon_center, lat_center,
            lons, lats
        )
        
        # Interpolate
        ve_samples, vn_samples = interpolate_at_points(
            xi=x_grid, yi=y_grid,
            ui=model_grid[:, 2], vi=model_grid[:, 3],
            xo=x_sample, yo=y_sample,
            dr=self.dr, nu=self.nu
        )
        
        return ve_samples, vn_samples
    
    def create_ggxf_from_samples(self,
                                grid_file: str,
                                output_ggxf: str,
                                lons: np.ndarray,
                                lats: np.ndarray,
                                metadata: Optional[dict] = None) -> None:
        """
        Sample velocity grid and save results as GGXF format
        
        This method combines sampling with GGXF export, taking the output from
        sample_grid_at_locations and creating a standards-compliant GGXF file.
        
        Parameters
        ----------
        grid_file : str
            Path to input model grid file (plain text or GGXF)
        output_ggxf : str
            Output GGXF file path (.nc extension)
        lons : np.ndarray
            Longitudes of grid points to create (degrees, 1D array)
        lats : np.ndarray
            Latitudes of grid points to create (degrees, 1D array)
        metadata : dict, optional
            Metadata to include in output GGXF file:
            - 'title': Grid title
            - 'description': Description
            - 'data_type': 'velocity' or 'displacement'  
            - 'units': 'mm/yr' or 'mm'
            - 'source': Original data source
            
        Examples
        --------
        >>> # Create a 1° x 1° grid covering a region
        >>> lons = np.arange(-125, -115, 0.1)  # 0.1° spacing
        >>> lats = np.arange(32, 38, 0.1)
        >>> 
        >>> gridder.create_ggxf_from_samples(
        ...     'input_velocity.txt',
        ...     'output_velocity.nc',
        ...     lons, lats,
        ...     metadata={
        ...         'title': 'Southern California Velocity Model',
        ...         'description': 'Interpolated using elastic coupling',
        ...         'data_type': 'velocity',
        ...         'units': 'mm/yr'
        ...     }
        ... )
        """
        from geode.Utils import azimuthal_equidistant
        
        if not GGXF_AVAILABLE:
            raise ImportError("netCDF4 library required for GGXF export. "
                            "Install with: pip install netCDF4")
        
        logger.info(f'Creating GGXF grid from {grid_file}')
        logger.info(f'Output grid: {len(lats)} x {len(lons)} points')
        
        # Load input grid
        file_path = Path(grid_file)
        if file_path.suffix.lower() in ['.nc', '.nc4', '.netcdf']:
            # Input is GGXF
            lon_in, lat_in, ve_in, vn_in = read_ggxf_grid(grid_file)
            if lon_in.ndim == 1 and lat_in.ndim == 1:
                lon_2d, lat_2d = np.meshgrid(lon_in, lat_in)
                lon_flat = lon_2d.flatten()
                lat_flat = lat_2d.flatten()
                ve_flat = ve_in.flatten()
                vn_flat = vn_in.flatten()
            else:
                lon_flat = lon_in.flatten()
                lat_flat = lat_in.flatten()
                ve_flat = ve_in.flatten()
                vn_flat = vn_in.flatten()
        else:
            # Input is plain text
            data = np.loadtxt(grid_file, dtype=float)
            lon_flat = data[:, 0]
            lat_flat = data[:, 1]
            ve_flat = data[:, 2]
            vn_flat = data[:, 3]
        
        # Use center of output grid as projection center
        lon_center = np.mean(lons)
        lat_center = np.mean(lats)
        
        logger.info(f'Projection center: {lat_center:.4f}°N, {lon_center:.4f}°E')
        
        # Project input grid to local coordinates
        x_in, y_in = azimuthal_equidistant(lon_center, lat_center, lon_flat, lat_flat)
        
        # Create output meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lon_out = lon_grid.flatten()
        lat_out = lat_grid.flatten()
        
        # Project output points
        x_out, y_out = azimuthal_equidistant(lon_center, lat_center, lon_out, lat_out)
        
        logger.info('Interpolating velocities using elastic method...')
        
        # Interpolate using elastic method
        ve_out, vn_out = interpolate_at_points(
            xi=x_in, yi=y_in,
            ui=ve_flat, vi=vn_flat,
            xo=x_out, yo=y_out,
            dr=self.dr, nu=self.nu
        )
        
        # Reshape to grid
        ve_grid = ve_out.reshape(len(lats), len(lons))
        vn_grid = vn_out.reshape(len(lats), len(lons))
        
        logger.info(f'Output velocity range: ve=[{ve_grid.min():.2f}, {ve_grid.max():.2f}], '
                   f'vn=[{vn_grid.min():.2f}, {vn_grid.max():.2f}] mm/yr')
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add interpolation parameters to metadata
        metadata['interpolation_method'] = 'elastic_coupling'
        metadata['poisson_ratio'] = self.nu
        metadata['min_distance'] = self.dr
        metadata['institution'] = metadata.get('institution', 'Generated by ElasticGridder')
        
        if 'title' not in metadata:
            metadata['title'] = 'Elastic Interpolation Grid'
        if 'data_type' not in metadata:
            metadata['data_type'] = 'velocity'
        if 'units' not in metadata:
            metadata['units'] = 'mm/yr'
        
        # Write GGXF file
        write_ggxf_grid(output_ggxf, lons, lats, ve_grid, vn_grid, metadata)
        
        logger.info(f'Successfully created GGXF file: {output_ggxf}')


def create_elastic_constraint_from_grid(config: 'EtmConfig',
                                       grid_file: str,
                                       model_type: str = 'poly',
                                       dr: float = 8.0,
                                       nu: float = 0.5,
                                       sigma: float = 0.001) -> 'EtmFunction':
    """
    Convenience function to create an elastic constraint from a grid file
    
    This is a simpler interface for the most common use case.
    
    Parameters
    ----------
    config : EtmConfig
        Station configuration
    grid_file : str
        Path to velocity grid file (lon, lat, ve, vn in mm/yr)
    model_type : str
        Type of constraint function ('poly')
    dr : float
        Minimum distance parameter (km)
    nu : float
        Poisson's ratio
    sigma : float
        Constraint uncertainty (m/yr)
        
    Returns
    -------
    constraint : EtmFunction
        Constraint function for ETM
        
    Examples
    --------
    >>> constraint = create_elastic_constraint_from_grid(
    ...     config, 'interseismic_velocity.txt', nu=0.5
    ... )
    >>> config.modeling.least_squares_strategy.constraints = [constraint]
    """
    gridder = ElasticGridder(config, dr=dr, nu=nu)
    return gridder.create_velocity_constraint(grid_file, model_type, sigma)


if __name__ == '__main__':
    # Example usage
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("ElasticGridder module for ETM constraint creation")
    print("=" * 60)
    print("\nUsage examples:")
    print("\n1. Create velocity constraint:")
    print("   >>> gridder = ElasticGridder(config, dr=8.0, nu=0.5)")
    print("   >>> constraint = gridder.create_velocity_constraint('model.txt')")
    print("\n2. Create jump constraint:")
    print("   >>> jump = gridder.create_jump_constraint('coseismic.txt', jump_date)")
    print("\n3. Use in ETM:")
    print("   >>> config.modeling.least_squares_strategy.constraints = [constraint]")
    print("   >>> etm = EtmEngine(config, cnn=cnn)")
    print("   >>> etm.run_adjustment()")
