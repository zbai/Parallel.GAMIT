"""
Elastic response to a uniform circular load on a spherical earth

This module contains functions for computing the geoelastic response to disk loads
on a spherical earth model.

REFERENCE:
This code is associated with the publication:
    Bevis, M., Melini, D., Spada, G., 2016. On computing the geoelastic 
    response to a disk load, Geophys. J. Int. 205 (3), 1,804-1,812,
    doi:10.1093/gji/ggw115

Original MATLAB code:
v 1.0 DM 03.07.2015  -- original version ported from REAR
v 1.1 MB, reordered LNs in input argument list, to follow the norm
v 1.2 MB, switch the degree index from l to n
v 1.3 DM, replaced the loop over theta with vectorized expressions
v 1.4 DM, Added degree-0 
v 1.5 DM, Added support for multiple nmax values

Python translation: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from importlib.resources import files
except ImportError:
    from importlib.resources import path as resource_path


def pLegendre(lmax, z):
    """
    Evaluate all unnormalized Legendre polynomials up to degree lmax.
    
    This subroutine evaluates all of the unnormalized Legendre polynomials 
    up to degree lmax.
    
    Parameters
    ----------
    lmax : int
        Maximum degree to compute.
    z : float or array-like
        Value(s) within [-1, 1], cos(colatitude) or sin(latitude).
        Can be a scalar or a vector.
    
    Returns
    -------
    p : ndarray
        A 2D array of all unnormalized Legendre polynomials evaluated at 
        z up to lmax. Shape is (lmax+1, nz) where nz is the length of z.
        p[l, :] contains P_l(z) for all z values.
    
    Notes
    -----
    1. The integral of P_l**2 over (-1,1) is 2/(2l+1).
    2. Values are calculated according to the following recursion scheme:
       P_0(z) = 1.0, P_1(z) = z, and 
       P_l(z) = (2l-1) * z * P_{l-1}(z) / l - (l-1) * P_{l-2}(z) / l
    
    Dependencies
    ------------
    None
    
    Original MATLAB code written by Mark Wieczorek June 2004
    Modified by Giorgio Spada, 2007
    Ported to MATLAB by Daniele Melini, 2015
    Ported to Python, 2025
    
    Original code is Copyright (c) 2005, Mark A. Wieczorek
    All rights reserved.
    """
    # Convert z to numpy array
    z = np.atleast_1d(z)
    nz = len(z)
    
    # Check that z is a 1D array (scalar or vector)
    if z.ndim != 1:
        raise ValueError('pLegendre: z must be a scalar or a vector.')
    
    # Convert z to row vector for broadcasting
    z = z.reshape(1, -1)
    
    if lmax < 0:
        raise ValueError('pLegendre: lmax must be greater than or equal to 0.')
    
    if np.any(np.abs(z) > 1):
        raise ValueError('pLegendre: abs(z) must be less than or equal to 1.')
    
    # Initialize arrays
    paux = np.full((lmax + 2, nz), np.nan)
    p = np.full((lmax + 1, nz), np.nan)
    
    # Initial values
    pm2 = 1.0
    paux[0, :] = 1.0
    
    pm1 = z
    paux[1, :] = pm1
    
    # Recursion
    for l in range(2, lmax + 1):
        pl = ((2*l - 1) * z * pm1 - (l - 1) * pm2) / l
        paux[l, :] = pl
        pm2 = pm1
        pm1 = pl
    
    # Copy results
    for j in range(lmax + 1):
        p[j, :] = paux[j, :]
    
    return p


def diskload(alpha, icomp, theta, w, nmin, nmax, h, k, l):
    """
    Elastic response to a uniform circular load on a spherical earth.
    
    This function computes the response to a uniform surface pressure load 
    imposed in a disc of angular radius alpha. The elastic response is found 
    at one or more stations located on the surface of the earth at angular 
    distance(s) theta from the center of the disc load.
    
    The elastic response is computed using user-supplied elastic loading 
    Love numbers (h, k, l) generated using a specific elastic structure model
    for the earth. If three output arguments are requested, this function
    also computes the change in the height of the geoid at each station.
    
    The pressure load imposed within the disk is expressed in terms of the 
    equivalent depth (height, thickness) of liquid water (density=1000 kg/m³).
    
    Parameters
    ----------
    alpha : float
        Disc half-amplitude (degrees).
    icomp : int
        Switch for a compensated (1) / uncompensated (0) disc load.
    theta : float or array-like
        Angular distances of stations from disc center (degrees).
    w : float
        Pressure imposed on the surface within the spherical disk
        expressed as the height or depth of equivalent water load (m).
    nmin : int
        Minimum harmonic degree of the expansion to be used.
    nmax : int or array-like
        Maximum harmonic degree(s) of the expansion to be used
        (may be a scalar or a vector with multiple truncation points).
    h : array-like
        (n+1)-vector containing the loading Love number h for degrees 0:n.
    k : array-like
        (n+1)-vector containing the loading Love number k for degrees 0:n.
    l : array-like
        (n+1)-vector containing the loading Love number l for degrees 0:n.
    
    Returns
    -------
    u : ndarray
        Radial or 'vertical' elastic displacement (mm).
        Shape is [length(nmax), length(theta)].
    v : ndarray
        Tangential or 'horizontal' elastic displacement (mm).
        Shape is [length(nmax), length(theta)].
    g : ndarray, optional
        Geoid change (mm). Only returned if requested.
        Shape is [length(nmax), length(theta)].
    
    Notes
    -----
    Size of output arrays is [length(nmax), length(theta)].
    For example:
        u[i, :] represents U vs theta at the i-th value of nmax
        u[:, j] represents U vs nmax at the j-th value of theta
    
    (1) All elements of nmax must be <= n, the maximum order provided for
        the elastic loading Love numbers.
    (2) Input w can be positive or negative allowing the user to model
        the incremental response to incremental pressure changes.
    (3) It is easy to switch from the state to the rate problem. If input
        w is actually the rate of change of the load (in m/yr w.e.), then 
        outputs u, v and g will become velocities (in mm/yr).
    
    Examples
    --------
    >>> u, v = diskload(alpha, icomp, theta, w, nmin, nmax, h, k, l)
    >>> u, v, g = diskload(alpha, icomp, theta, w, nmin, nmax, h, k, l)
    
    Dependencies
    ------------
    This function calls function pLegendre()
    """
    # Define constants
    ggg = 6.67384e-11          # Newton's constant (SI units)
    radius = 6371              # Radius of the Earth (km)
    radiusm = radius * 1e3     # Radius of the Earth (m)
    grav = 9.8046961           # Surface gravity (m/s²)
    rhow = 1000                # Density of pure water (kg/m³)
    rhoear = 3.0 * grav / (4.0 * ggg * np.pi * radiusm)  # Average Earth density (kg/m³)
    from_m_to_mm = 1000
    
    # Convert inputs to numpy arrays
    h = np.asarray(h)
    k = np.asarray(k)
    l = np.asarray(l)
    theta = np.atleast_1d(theta).flatten()
    nmax = np.atleast_1d(nmax).flatten()
    
    # Check for illegal conditions
    m = len(h)
    if len(k) != m or len(l) != m:
        raise ValueError('Love number vectors do not have same length')
    
    if np.any(nmax > (m - 1)):
        raise ValueError('nmax exceeds the lengths of the Love Number vectors')
    
    # Check for 0-order LNs
    if (l[0] * k[0]) != 0:
        raise ValueError('n=0 Love numbers l_0 and k_0 must be zero')
    
    # Computing the harmonic coefficients of the load
    # Vectors are "offset-indexed", i.e.
    # P_n = leg[n], sigma_n = sigma[n]
    
    leg = pLegendre(int(np.max(nmax)) + 1, np.cos(np.radians(alpha)))
    leg = leg[:, 0]  # Extract single column since alpha is scalar
    
    sigma = np.full(int(np.max(nmax)) + 1, np.nan)
    
    if icomp == 0:  # Uncompensated disc load, eq. (7)
        for n in range(nmin, int(np.max(nmax)) + 1):
            if n == 0:
                sigma[n] = 0.5 * (1 - np.cos(np.radians(alpha)))
            if n > 0:
                sigma[n] = 0.5 * (-leg[n + 1] + leg[n - 1])
    elif icomp == 1:  # Compensated disc load, eq. (8)
        for n in range(nmin, int(np.max(nmax)) + 1):
            if n == 0:
                sigma[n] = 0
            if n > 0:
                sigma[n] = -(leg[n + 1] - leg[n - 1]) / (1 + np.cos(np.radians(alpha)))
    
    # Initialize output arrays
    u = np.zeros((len(nmax), len(theta)))
    v = np.zeros((len(nmax), len(theta)))
    compute_geoid = False
    g = None
    
    # Compute Legendre polynomials at cos(theta)
    x = np.cos(np.radians(theta))
    leg = pLegendre(int(np.max(nmax)) + 1, x)
    
    idx = (np.abs(x) == 1)
    
    # Add the n=0 terms, if required
    if nmin == 0:
        u = u + h[0] * sigma[0]
        if compute_geoid:
            g = g + sigma[0]
    
    # Main loop over harmonic degrees
    for n in range(max(1, nmin), int(np.max(nmax)) + 1):
        # Compute derivative of Legendre polynomial
        dleg = -(n + 1) * (x * leg[n, :] - leg[n + 1, :]) / ((1 - x) * (1 + x)) * np.sqrt(1 - x**2)
        
        dleg[idx] = 0.0
        
        ampl = sigma[n] / (2 * n + 1)
        
        for i in range(len(nmax)):
            if n <= nmax[i]:
                u[i, :] = u[i, :] + h[n] * ampl * leg[n, :]
                v[i, :] = v[i, :] + l[n] * ampl * dleg
    
    # Scale outputs
    u = u * (3 * rhow / rhoear) * w * from_m_to_mm
    v = v * (3 * rhow / rhoear) * w * from_m_to_mm
    
    return u, v


def diskload_with_geoid(alpha, icomp, theta, w, nmin, nmax, h, k, l):
    """
    Elastic response to a uniform circular load on a spherical earth (with geoid).
    
    This is a variant of diskload() that always computes and returns the geoid change.
    See diskload() for full documentation.
    
    Parameters
    ----------
    Same as diskload()
    
    Returns
    -------
    u : ndarray
        Radial or 'vertical' elastic displacement (mm).
    v : ndarray
        Tangential or 'horizontal' elastic displacement (mm).
    g : ndarray
        Geoid change (mm).
    """
    # Define constants
    ggg = 6.67384e-11
    radius = 6371
    radiusm = radius * 1e3
    grav = 9.8046961
    rhow = 1000
    rhoear = 3.0 * grav / (4.0 * ggg * np.pi * radiusm)
    from_m_to_mm = 1000
    
    # Convert inputs to numpy arrays
    h = np.asarray(h)
    k = np.asarray(k)
    l = np.asarray(l)
    theta = np.atleast_1d(theta).flatten()
    nmax = np.atleast_1d(nmax).flatten()
    
    # Check for illegal conditions
    m = len(h)
    if len(k) != m or len(l) != m:
        raise ValueError('Love number vectors do not have same length')
    
    if np.any(nmax > (m - 1)):
        raise ValueError('nmax exceeds the lengths of the Love Number vectors')
    
    # Check for 0-order LNs
    if (l[0] * k[0]) != 0:
        raise ValueError('n=0 Love numbers l_0 and k_0 must be zero')
    
    # Computing the harmonic coefficients of the load
    leg = pLegendre(int(np.max(nmax)) + 1, np.cos(np.radians(alpha)))
    leg_alpha = leg[:, 0]  # Extract single column since alpha is scalar
    
    sigma = np.full(int(np.max(nmax)) + 1, np.nan)
    
    if icomp == 0:  # Uncompensated disc load, eq. (7)
        for n in range(nmin, int(np.max(nmax)) + 1):
            if n == 0:
                sigma[n] = 0.5 * (1 - np.cos(np.radians(alpha)))
            if n > 0:
                sigma[n] = 0.5 * (-leg_alpha[n + 1] + leg_alpha[n - 1])
    elif icomp == 1:  # Compensated disc load, eq. (8)
        for n in range(nmin, int(np.max(nmax)) + 1):
            if n == 0:
                sigma[n] = 0
            if n > 0:
                sigma[n] = -(leg_alpha[n + 1] - leg_alpha[n - 1]) / (1 + np.cos(np.radians(alpha)))
    
    # Initialize output arrays
    u = np.zeros((len(nmax), len(theta)))
    v = np.zeros((len(nmax), len(theta)))
    g = np.zeros((len(nmax), len(theta)))
    
    # Compute Legendre polynomials at cos(theta)
    x = np.cos(np.radians(theta))
    leg = pLegendre(int(np.max(nmax)) + 1, x)
    
    idx = (np.abs(x) == 1)
    
    # Add the n=0 terms, if required
    if nmin == 0:
        u = u + h[0] * sigma[0]
        g = g + sigma[0]
    
    # Main loop over harmonic degrees
    for n in range(max(1, nmin), int(np.max(nmax)) + 1):
        # Compute derivative of Legendre polynomial
        dleg = -(n + 1) * (x * leg[n, :] - leg[n + 1, :]) / ((1 - x) * (1 + x)) * np.sqrt(1 - x**2)
        
        dleg[idx] = 0.0
        
        ampl = sigma[n] / (2 * n + 1)
        
        for i in range(len(nmax)):
            if n <= nmax[i]:
                u[i, :] = u[i, :] + h[n] * ampl * leg[n, :]
                v[i, :] = v[i, :] + l[n] * ampl * dleg
                g[i, :] = g[i, :] + (1 + k[n]) * ampl * leg[n, :]
    
    # Scale outputs
    u = u * (3 * rhow / rhoear) * w * from_m_to_mm
    v = v * (3 * rhow / rhoear) * w * from_m_to_mm
    g = g * (3 * rhow / rhoear) * w * from_m_to_mm
    
    return u, v, g


def load_love_numbers(filename=None):
    """
    Load elastic loading Love numbers from a text file.

    Parameters
    ----------
    filename : str
        Path to the Love numbers file. Expected format is a text file with
        header line followed by rows of: degree h_love l_love k_love

    Returns
    -------
    h_love : ndarray
        Loading Love number h for each degree
    l_love : ndarray
        Loading Love number l for each degree
    k_love : ndarray
        Loading Love number k for each degree
    """
    if filename is None:
        # Load bundled file
        try:
            # Python 3.9+
            from importlib.resources import files
            data_path = files('geode.elasticity.data').joinpath(
                'REF_6371_loading_love_numbers_0_40000.txt'
            )
            filename = str(data_path)
        except (ImportError, AttributeError):
            # Python 3.7-3.8 fallback
            from importlib.resources import path as resource_path
            import geode.elasticity.data

            with resource_path(geode.elasticity.data,
                               'REF_6371_loading_love_numbers_0_40000.txt') as p:
                filename = str(p)

    data = np.loadtxt(filename, skiprows=1)
    h_love = data[:, 1]
    l_love = data[:, 2]
    k_love = data[:, 3]

    return h_love, l_love, k_love

def compute_diskload(alpha=1, theta_range=None, Tw=1.0, nmin=0,
                     nmax_max=40000, imass=0, love_file=None,
                     h_love=None, l_love=None, k_love=None,
                     plot=False):
    """
    Test and demonstrate the diskload function with elastic loading response.

    This function computes the elastic response (vertical displacement U,
    horizontal displacement V, and geoid change G) to a uniform circular
    disk load on a spherical Earth model.

    Parameters
    ----------
    alpha : float, optional
        Disk radius in km (default: 1 km)
    theta_range : array-like, optional
        Range of colatitudes with respect to disk center (km).
        If None, creates a range from 0 to 5*alpha with 100 points.
    Tw : float, optional
        Disk height expressed as equivalent water height in meters (default: 1.0)
    nmin : int, optional
        Minimum harmonic degree (default: 0)
    nmax_max : int, optional
        Maximum harmonic degree to compute (default: 40000)
    imass : int, optional
        Load type selector:
            0 = uncompensated load
            1 = globally compensated load (default: 0)
    love_file : str, optional
        Path to file containing loading Love numbers. If provided, Love numbers
        will be loaded from this file.
    h_love : array-like, optional
        Loading Love number h. Required if love_file is not provided.
    l_love : array-like, optional
        Loading Love number l. Required if love_file is not provided.
    k_love : array-like, optional
        Loading Love number k. Required if love_file is not provided.
    plot : bool, optional
        If True, generate plots of the results (default: True)

    Returns
    -------
    results : dict
        Dictionary containing:
            'theta' : array of colatitudes (degrees)
            'U' : vertical displacement (mm)
            'V' : horizontal displacement (mm)
            'G' : geoid change (mm)
            'alpha' : disk radius used (degrees)
            'Tw' : load height used (m)
            'nmax' : maximum degree used
            'imass' : load type used

    Examples
    --------
    >>> # Using Love numbers from file
    >>> results = compute_diskload(alpha=0.1, Tw=1.0,
    ...                            love_file='data/REF_6371_loading_love_numbers_0_40000.txt')

    >>> # Using provided Love numbers
    >>> results = compute_diskload(alpha=0.1, Tw=1.0,
    ...                            h_love=h, l_love=l, k_love=k)

    Notes
    -----
    The function can operate in two modes:
    1. Uncompensated load (imass=0): Simple disk load without mass compensation
    2. Compensated load (imass=1): Globally compensated disk load

    The elastic response includes:
    - U: Radial (vertical) displacement in mm
    - V: Tangential (horizontal) displacement in mm
    - G: Geoid height change in mm
    """
    # Load Love numbers if file provided
    if love_file is None:
        h_love, l_love, k_love = load_love_numbers()
    elif love_file is not None:
        h_love, l_love, k_love = load_love_numbers(love_file)
    elif h_love is None or l_love is None or k_love is None:
        raise ValueError("Must provide either love_file or all three Love number arrays")

    # Convert to numpy arrays
    h_love = np.asarray(h_love)
    l_love = np.asarray(l_love)
    k_love = np.asarray(k_love)

    # Set up theta range if not provided
    if theta_range is None:
        theta = np.linspace(0, alpha * 5, 100)
    else:
        theta = np.asarray(theta_range)

    # Print load type
    #if imass == 1:
    #    print('Invoking a globally compensated load (icomp=1)')
    #else:
    #    print('Invoking an uncompensated load (icomp=0)')

    # Compute the disc response for the maximum value of nmax
    U, V, G = diskload_with_geoid(np.rad2deg(alpha/6371), imass, np.rad2deg(theta/6371), Tw, nmin,
                                  nmax_max, h_love, k_love, l_love)

    # Create results dictionary
    results = {
        'theta': theta,
        'U': U.flatten(),
        'V': V.flatten(),
        'G': G.flatten(),
        'alpha': alpha,
        'Tw': Tw,
        'nmax': nmax_max,
        'imass': imass
    }

    # Generate plots if requested
    if plot:
        plot_diskload_response(theta, U.flatten(), V.flatten(), G.flatten(),
                               alpha, Tw)

    return results


def plot_diskload_response(theta, U, V, G, alpha, Tw):
    """
    Plot the elastic response vs normalized distance from disk center.

    Parameters
    ----------
    theta : array-like
        Colatitudes (degrees)
    U : array-like
        Vertical displacement (mm)
    V : array-like
        Horizontal displacement (mm)
    G : array-like
        Geoid change (mm)
    alpha : float
        Disk radius (degrees)
    Tw : float
        Load height (m w.e.)
    """
    plt.figure(figsize=(10, 6))

    plt.plot(theta / alpha, U, 'b', linewidth=1.5, label='U')
    plt.plot(theta / alpha, V, 'r', linewidth=1.5, label='V')
    plt.plot(theta / alpha, G, 'g', linewidth=1.5, label='G')

    plt.xlabel(r'$\theta/\alpha$', fontsize=16)
    plt.ylabel('mm', fontsize=16)
    plt.xlim([0, 5])
    plt.ylim([-2.5, 1])
    plt.grid(True)
    plt.legend(loc='best')

    title_str = f'Disk radius α = {alpha:.2f}°  Load = {Tw:.2f} m w.e.'
    plt.title(title_str)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("diskload module loaded successfully")
    print("Available functions:")
    print("  - pLegendre(lmax, z): Compute unnormalized Legendre polynomials")
    print("  - diskload(alpha, icomp, theta, w, nmin, nmax, h, k, l): Compute elastic response (u, v)")
    print("  - diskload_with_geoid(alpha, icomp, theta, w, nmin, nmax, h, k, l): "
          "Compute elastic response with geoid (u, v, g)")
