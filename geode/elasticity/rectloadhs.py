"""
Elastic half-space displacement calculations for rectangular surface loads.

This module implements Love's problem for computing elastic deformation due to 
rectangular loads applied as uniform pressure fields to the surface of a uniform 
elastic halfspace.

Functions:
    mrectloadhs_dif: Displacement influence functions for multiple rectangular loads
    rectloadhs: Love's problem for a single rectangular load
    
References:
    Becker, J., and M. Bevis (2002) Love's Problem, Geophys. J. Int.
    Love, A.E.H. (1929) The stress produced in a semi-infinite solid by 
        pressure on part of the boundary, Phil. Trans. Roy. Soc. London, 
        Ser. A., 228, p. 377.

Version history:
    MATLAB version 1.0 by Michael Bevis, 7 June 2003
    MATLAB version 2.0 added option for single output (F), 15 Nov 2021
    Python translation 2025
"""

import numpy as np
import warnings


def mrectloadhs_dif(spos, xr, yr, lambda_param, mu, return_combined=False):
    """
    Displacement influence functions for multiple rectangular loads on the 
    surface of a homogeneous elastic half-space.
    
    Let u, v, w be the components of displacement in the x, y, z directions at 
    n stations inside or on the halfspace due to surface loading (uniform 
    pressure) within m rectangles (Love's problem). Then u is the u-displacement 
    influence matrix of size [n, m]. That is u[i, j] is the u component of 
    displacement at the i-th station due to unit pressure applied in the j-th 
    rectangle. (And similarly for v and w). It is assumed that all rectangles 
    have their sides parallel to the x and y axes, and that xr[j, :] contains 
    [xmin, xmax] for the j-th rectangle, and yr[j, :] contains [ymin, ymax] 
    for the j-th rectangle.
    
    THIS FUNCTION USES LOVE'S COORDINATE SYSTEM (+ve z down into half-space)
    
    Parameters
    ----------
    spos : ndarray
        Matrix of size [3, n] containing station coordinates [x, y, z].
        z must be >= 0 (on or below surface).
    xr : ndarray
        Matrix of size [m, 2] where each row contains [xmin, xmax] for a rectangle.
    yr : ndarray
        Matrix of size [m, 2] where each row contains [ymin, ymax] for a rectangle.
    lambda_param : float
        Lame parameter lambda.
    mu : float
        Lame parameter mu (shear modulus).
    return_combined : bool, optional
        If True, returns a single influence matrix F of size [3n, m] that 
        combines the influences on u, v, and w to describe the influence on
        the network displacement vector [u1, v1, w1, u2, ..., un, vn, wn].
        If False (default), returns separate u, v, w matrices.
    
    Returns
    -------
    u : ndarray
        If return_combined=False: u-displacement influence matrix of size [n, m].
        If return_combined=True: combined influence matrix F of size [3n, m].
    v : ndarray
        v-displacement influence matrix of size [n, m] (only if return_combined=False).
    w : ndarray
        w-displacement influence matrix of size [n, m] (only if return_combined=False).
    
    Notes
    -----
    The influence functions are computed for unit pressure loads of 1 Pascal, 
    the SI unit for pressure. If you want to use some other 'unit' load, 
    e.g. 1 meter water equivalent load, multiply these influence matrices by 
    Pwe, the pressure associated with 1 meter of water expressed in Pascals.
    
    See also: rectloadhs, lambdamu2enu, enu2lambdamu
    
    Examples
    --------
    >>> # For separate u, v, w matrices:
    >>> u, v, w = mrectloadhs_dif(spos, xr, yr, lambda_param, mu)
    >>> 
    >>> # For combined influence matrix:
    >>> F = mrectloadhs_dif(spos, xr, yr, lambda_param, mu, return_combined=True)
    """
    # Check input arguments
    if spos.shape[0] != 3:
        raise ValueError('spos should be a matrix with three rows')
    
    n = spos.shape[1]
    
    if np.any(spos[2, :] < 0):
        raise ValueError('spos contains stations located above the half space (with z<0)')
    
    if xr.ndim != 2 or xr.shape[1] != 2:
        raise ValueError('xr must be a matrix with two columns '
                        '(and row dimension = number of rectangles)')
    
    if yr.ndim != 2 or yr.shape[1] != 2:
        raise ValueError('yr must be a matrix with two columns '
                        '(and row dimension = number of rectangles)')
    
    m = xr.shape[0]
    
    if yr.shape[0] != m:
        raise ValueError('xr and yr must have same number of rows '
                        '(one for each rectangle)')
    
    for j in range(m):
        if xr[j, 0] >= xr[j, 1] or yr[j, 0] >= yr[j, 1]:
            raise ValueError('the first element of each row of xr (or yr) must be '
                           'smaller than its second element')
    
    if not np.isscalar(lambda_param) or not np.isscalar(mu):
        raise ValueError('input arguments lambda and mu must be scalars')
    
    # Unit pressure
    P = 1.0
    
    # Preallocate space
    u = np.zeros((n, m))
    v = np.zeros((n, m))
    w = np.zeros((n, m))
    
    # Compute displacement for each rectangle
    for j in range(m):
        d = rectloadhs(spos, xr[j, :], yr[j, :], P, lambda_param, mu)
        
        # DEBUG section (preserved from original)
        debug = False
        if debug:
            if np.isnan(d[2, 0]):
                print('debug: nan found by mrectloadhs_dif')
                print(f'xr_ = {xr[j, :]}')
                print(f'yr_ = {yr[j, :]}')
                print(f'spos_ = {spos}')
                print(f'd_ = {d}')
                raise RuntimeError('NaN detected in displacement calculation')
        
        u[:, j] = d[0, :]
        v[:, j] = d[1, :]
        w[:, j] = d[2, :]
    
    # Reorganize output if a single influence matrix is desired
    if return_combined:
        F = np.zeros((3 * n, m))
        for i in range(n):
            F[3*i, :] = u[i, :]
            F[3*i + 1, :] = v[i, :]
            F[3*i + 2, :] = w[i, :]
        return F
    else:
        return u, v, w


def rectloadhs(spos, xr, yr, P, lambda_param, mu):
    """
    Love's problem: elastic deformation due to rectangular load applied as a 
    uniform pressure field to the surface of a uniform elastic halfspace.
    
    Following Love (1929) we adopt a cartesian coordinate system in which the 
    x and y axes are confined to the surface of the halfspace, and the z-axis 
    is positive downward so that points in the elastic medium have z > 0. 
    A uniform pressure P is applied over the rectangular region 
    xmin <= x <= xmax, ymin <= y <= ymax at the surface (z=0) of the halfspace. 
    P is positive if the force is directed downwards into the elastic body 
    (i.e. in the +z direction). The elastic material has Lame parameters 
    lambda and mu. This function uses formulas provided by Becker and Bevis 
    (2002) to compute the displacements at N points (or stations) located on 
    or in the elastic halfspace.
    
    Parameters
    ----------
    spos : ndarray
        Matrix of size [3, N] containing the coordinates of the points or 
        stations at which displacement vectors are to be computed.
        The i-th column of spos contains the position vector [xi, yi, zi]
        for the i-th station. (Note: it is required that zi >= 0).
    xr : ndarray
        Vector [xmin, xmax] describing the x-extent of the rectangle.
    yr : ndarray
        Vector [ymin, ymax] describing the y-extent of the rectangle.
    P : float
        The pressure applied (uniformly) within the rectangle.
    lambda_param : float
        Lame parameter 1.
    mu : float
        Lame parameter 2 (the modulus of rigidity, or shear modulus).
    
    Returns
    -------
    d : ndarray
        Matrix of size [3, N] containing the displacement vectors at each station.
        The i-th column of d is the vector [ui, vi, wi], where u, v and w are 
        the components of displacement in the x, y and z directions.
    
    Notes
    -----
    Use enu2lambdamu for obtaining the Lame parameters from Young's modulus
    (E) and Poisson's ratio (nu).
    
    References
    ----------
    Becker, J., and M. Bevis (2002) Love's Problem, Geophys. J. Int.
    Love, A.E.H. (1929) The stress produced in a semi-infinite solid by 
        pressure on part of the boundary, Phil. Trans. Roy. Soc. London, 
        Ser. A., 228, p. 377.
    
    Version 1.6 by Michael Bevis & Janet Becker, 14 May 2004
    Python translation 2025
    """
    # Allow NaNs to occur during divide-by-zero etc. and then trap the special 
    # cases and fix the answer
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Check input arguments
        if spos.shape[0] != 3:
            raise ValueError('spos should be a matrix with three rows')
        
        N = spos.shape[1]
        
        if np.any(spos[2, :] < 0):
            raise ValueError('spos contains stations located above the half space (with z<0)')
        
        if len(xr) != 2 or len(yr) != 2:
            raise ValueError('xr and yr must both be vectors of length 2')
        
        if xr[0] >= xr[1] or yr[0] >= yr[1]:
            raise ValueError('the first element of xr (or yr) must be smaller than its second element')
        
        if not np.isscalar(lambda_param) or not np.isscalar(mu):
            raise ValueError('input arguments lambda and mu must be scalars')
        
        # Translate coordinate system so origin is at center of rectangular surface load
        # In this new system, the rectangle is located thus: -a <= x <= +a, -b <= y <= +b
        xc = np.mean(xr)
        yc = np.mean(yr)
        xr = xr - xc
        yr = yr - yc
        x = spos[0, :] - xc
        y = spos[1, :] - yc
        z = spos[2, :]
        
        # Rectangle has width 2a in the x direction and 2b in the y direction
        a = (np.max(xr) - np.min(xr)) / 2
        b = (np.max(yr) - np.min(yr)) / 2
        
        # Evaluate Love's elementary terms
        # Added argument to these elementary terms for corner correction
        r10_pb, b10_pb, p10_pb, r20_pb, b20_pb, p20_pb = _love_10_20(x, y, z, a, b, +b)  # y' = +b (& x'=a)
        r10_mb, b10_mb, p10_mb, r20_mb, b20_mb, p20_mb = _love_10_20(x, y, z, a, b, -b)  # y' = -b (& x'=a)
        r01_pa, b01_pa, p01_pa, r02_pa, b02_pa, p02_pa = _love_01_02(x, y, z, a, +a, b)  # x' = +a (& y'=b)
        r01_ma, b01_ma, p01_ma, r02_ma, b02_ma, p02_ma = _love_01_02(x, y, z, a, -a, b)  # x' = -a (& y'=b)
        
        # Compute functions J1 and J2 for cases y' = +b and -b 
        # (where _pb indicates +b and _mb indicates -b)
        J1_pb, J2_pb = _love_J1_J2(r10_pb, p10_pb, b10_pb, r20_pb, p20_pb, b20_pb, a, x, y, z, +b)
        J1_mb, J2_mb = _love_J1_J2(r10_mb, p10_mb, b10_mb, r20_mb, p20_mb, b20_mb, a, x, y, z, -b)
        
        # Compute functions K1 and K2 for cases x' = +a and -a 
        # (where _pa indicates +a and _ma indicates -a)
        K1_pa, K2_pa = _love_K1_K2(r01_pa, p01_pa, b01_pa, r02_pa, p02_pa, b02_pa, b, x, y, z, +a)
        K1_ma, K2_ma = _love_K1_K2(r01_ma, p01_ma, b01_ma, r02_ma, p02_ma, b02_ma, b, x, y, z, -a)
        
        # Compute functions L1 and L2 for cases y' = +b and -b 
        # (where _pb indicates +b and _mb indicates -b)
        L1_pb, L2_pb = _love_L1_L2(r10_pb, p10_pb, b10_pb, r20_pb, p20_pb, b20_pb, a, x, y, z, +b)
        L1_mb, L2_mb = _love_L1_L2(r10_mb, p10_mb, b10_mb, r20_mb, p20_mb, b20_mb, a, x, y, z, -b)
        
        # Compute V and the various derivatives used in the formulas for displacement
        dVdx = -P * (np.log(r10_pb + b - y) - np.log(r10_mb - b - y) - 
                     np.log(r20_pb + b - y) + np.log(r20_mb - b - y))
        
        dVdy = -P * (np.log(r01_pa + a - x) - np.log(r01_ma - a - x) - 
                     np.log(r02_pa + a - x) + np.log(r02_ma - a - x))
        
        dXdx = -P * (J1_pb - J1_mb - J2_pb + J2_mb)
        
        dXdy = -P * (K1_pa - K1_ma - K2_pa + K2_ma)
        
        dVdz = -P * (np.arctan2((a - x) * (+b - y), z * r10_pb) + 
                     np.arctan2((a + x) * (+b - y), z * r20_pb) - 
                     np.arctan2((a - x) * (-b - y), z * r10_mb) - 
                     np.arctan2((a + x) * (-b - y), z * r20_mb))
        
        V = P * (L1_pb - L2_pb - L1_mb + L2_mb)
        
        # Compute displacement vectors
        r4pi = 1.0 / (4.0 * np.pi)
        rlm = 1.0 / (lambda_param + mu)
        
        # When z ~= 0 then u = -r4pi*(rlm*dXdx + (z/mu)*dVdx), but if z=0 then 
        # (z/mu)*dVdx -> 0, but at corners of rectangle dVdz is undetermined 
        # numerically, so (z/mu)*dVdx must be forced to zero
        sterm2 = (z / mu) * dVdx
        sterm2[z == 0] = 0
        u = -r4pi * (rlm * dXdx + sterm2)
        
        # When z ~= 0 v = -r4pi*(rlm*dXdy + (z/mu)*dVdy) but we must perform 
        # the same numerical special case handling as we did for displacement component u
        sterm2 = (z / mu) * dVdy
        sterm2[z == 0] = 0
        v = -r4pi * (rlm * dXdy + sterm2)
        
        # Special cases occur for w
        sjterm2 = z * dVdz
        sjterm2[z == 0] = 0
        w = r4pi * (1.0 / mu) * ((lambda_param + 2*mu) * rlm * V - sjterm2)
        
        d = np.vstack([u, v, w])
        
        return d


def _love_J1_J2(r10, p10, b10, r20, p20, b20, a, x, y, z, yp):
    """
    Compute functions J1 and J2 for cases y' = +b and -b.
    
    Parameters
    ----------
    r10, p10, b10, r20, p20, b20 : ndarray
        Elementary terms from love_10_20.
    a : float
        Half-width of rectangle in x direction.
    x, y, z : ndarray
        Station coordinates (translated).
    yp : float
        y' value (+b or -b).
    
    Returns
    -------
    J1, J2 : ndarray
        Functions J1 and J2.
    
    Notes
    -----
    If yp=+b (case pb) then set r10=r10_pb, r20=r20_pb, p10=p10_pb, etc.
    If yp=-b (case mb) then set r10=r10_mb, r20=r20_mb, p10=p10_mb, etc.
    """
    # J1 calculation
    term1 = (yp - y) * (np.log(z + r10) - 1)
    term1[(z + r10) == 0] = 0
    
    term2 = z * np.log((1 + p10) / (1 - p10))
    term2[z == 0] = 0
    
    arctan = np.arctan(np.abs(a - x) * p10 / (z + b10))
    arctan[z == 0] = np.arctan(p10[z == 0])
    
    J1 = term1 + term2 + 2 * np.abs(a - x) * arctan
    
    # J2 calculation
    term1 = (yp - y) * (np.log(z + r20) - 1)
    term1[(z + r20) == 0] = 0
    
    term2 = z * np.log((1 + p20) / (1 - p20))
    term2[z == 0] = 0
    
    arctan = np.arctan(np.abs(a + x) * p20 / (z + b20))
    arctan[z == 0] = np.arctan(p20[z == 0])
    
    J2 = term1 + term2 + 2 * np.abs(a + x) * arctan
    
    return J1, J2


def _love_K1_K2(r01, p01, b01, r02, p02, b02, b, x, y, z, xp):
    """
    Compute functions K1 and K2 for cases x' = +a and -a.
    
    Parameters
    ----------
    r01, p01, b01, r02, p02, b02 : ndarray
        Elementary terms from love_01_02.
    b : float
        Half-width of rectangle in y direction.
    x, y, z : ndarray
        Station coordinates (translated).
    xp : float
        x' value (+a or -a).
    
    Returns
    -------
    K1, K2 : ndarray
        Functions K1 and K2.
    
    Notes
    -----
    If xp=+a (case pa) then set r01=r01_pa, r02=r02_pa, p01=p01_pa, etc.
    If xp=-a (case ma) then set r01=r01_ma, r02=r02_ma, p01=p01_ma, etc.
    """
    # K1 calculation
    term1 = (+xp - x) * (np.log(z + r01) - 1)
    term1[(z + r01) == 0] = 0
    
    term2 = z * np.log((1 + p01) / (1 - p01))
    term2[z == 0] = 0
    
    arctan = np.arctan(np.abs(b - y) * p01 / (z + b01))
    arctan[z == 0] = np.arctan(p01[z == 0])
    
    K1 = term1 + term2 + 2 * np.abs(b - y) * arctan
    
    # K2 calculation
    term1 = (+xp - x) * (np.log(z + r02) - 1)
    term1[(z + r02) == 0] = 0
    
    term2 = z * np.log((1 + p02) / (1 - p02))
    term2[z == 0] = 0
    
    arctan = np.arctan(np.abs(b + y) * p02 / (z + b02))
    arctan[z == 0] = np.arctan(p02[z == 0])
    
    K2 = term1 + term2 + 2 * np.abs(b + y) * arctan
    
    return K1, K2


def _love_L1_L2(r10, p10, b10, r20, p20, b20, a, x, y, z, yp):
    """
    Compute functions L1 and L2 for cases y' = +b and -b.
    
    Parameters
    ----------
    r10, p10, b10, r20, p20, b20 : ndarray
        Elementary terms from love_10_20.
    a : float
        Half-width of rectangle in x direction.
    x, y, z : ndarray
        Station coordinates (translated).
    yp : float
        y' value (+b or -b).
    
    Returns
    -------
    L1, L2 : ndarray
        Functions L1 and L2.
    
    Notes
    -----
    If yp=+b (case pb) then set r10=r10_pb, r20=r20_pb, p10=p10_pb, etc.
    If yp=-b (case mb) then set r10=r10_mb, r20=r20_mb, p10=p10_mb, etc.
    """
    # L1 calculation
    term1 = (+yp - y) * (np.log(a - x + r10) - 1)
    term1[(a - x + r10) == 0] = 0
    
    term2 = (a - x) * np.log((1 + p10) / (1 - p10))
    term2[(a - x) == 0] = 0
    # Why are there not problems in J's and K's when p10=+-1?
    # The logarithmic singularity occurs when x=a, but apparently it is
    # not enough to just set term2=0 when x=a. Probably should put in the
    # same correction in J and K code. Note that p10=delta y / |delta y|=+-1
    # when x=a and z=0
    term2[p10 == 1] = 0
    term2[p10 == -1] = 0
    
    arctan = np.arctan2(z * p10, b10 + (a - x))
    arctan[(a - x) == 0] = np.arctan(p10[(a - x) == 0])
    
    term3 = 2 * z * arctan
    term3[z == 0] = 0
    
    L1 = term1 + term2 + term3
    
    # L2 calculation
    term1 = (+yp - y) * (np.log(-a - x + r20) - 1)
    term1[(-a - x + r20) == 0] = 0
    
    term2 = (a + x) * np.log((1 + p20) / (1 - p20))
    term2[(a + x) == 0] = 0
    term2[p20 == 1] = 0
    term2[p20 == -1] = 0
    
    arctan = np.arctan2(z * p20, b20 - (a + x))
    arctan[(a + x) == 0] = np.arctan(p20[(a + x) == 0])
    
    # Fix for z=0
    term3 = 2 * z * arctan
    term3[z == 0] = 0
    
    L2 = term1 - term2 + term3
    
    return L1, L2


def _love_10_20(x, y, z, a, b, yp):
    """
    Evaluate Love's terms r_10, beta_10, phi_10, r_20, beta_20, phi_20.
    
    Parameters
    ----------
    x, y, z : ndarray
        Station coordinates (translated).
    a : float
        Half-width of rectangle in x direction.
    b : float
        Half-width of rectangle in y direction.
    yp : float
        y' value (will be set to +b or -b).
    
    Returns
    -------
    r10, b10, p10, r20, b20, p20 : ndarray
        Love's elementary terms.
    
    Notes
    -----
    This is an internal function used by rectloadhs.
    """
    dely = yp - y
    dely2 = dely ** 2
    z2 = z ** 2
    
    r10s = (a - x) ** 2 + dely2 + z2
    r10 = np.sqrt(r10s)
    b10 = np.sqrt(r10s - dely2)
    p10 = dely / (r10 + b10)
    
    r20s = (a + x) ** 2 + dely2 + z2
    r20 = np.sqrt(r20s)
    b20 = np.sqrt(r20s - dely2)
    p20 = dely / (r20 + b20)
    
    # Special case: station(s) at rectangle corner(s)
    isn10 = np.isnan(p10)
    if np.any(isn10):
        p10[isn10 & (y == b)] = 1
        p10[isn10 & (y == -b)] = -1
    
    isn20 = np.isnan(p20)
    if np.any(isn20):
        p20[isn20 & (y == b)] = 1
        p20[isn20 & (y == -b)] = -1
    
    return r10, b10, p10, r20, b20, p20


def _love_01_02(x, y, z, a, xp, b):
    """
    Evaluate Love's terms r_01, beta_01, phi_01, r_02, beta_02, and phi_02.
    
    Parameters
    ----------
    x, y, z : ndarray
        Station coordinates (translated).
    a : float
        Half-width of rectangle in x direction.
    xp : float
        x' value (will be set to +a or -a).
    b : float
        Half-width of rectangle in y direction.
    
    Returns
    -------
    r01, b01, p01, r02, b02, p02 : ndarray
        Love's elementary terms.
    
    Notes
    -----
    This is an internal function used by rectloadhs.
    """
    delx = xp - x
    delx2 = delx ** 2
    z2 = z ** 2
    
    r01s = (b - y) ** 2 + delx2 + z2
    r01 = np.sqrt(r01s)
    b01 = np.sqrt(r01s - delx2)
    p01 = delx / (r01 + b01)
    
    r02s = (b + y) ** 2 + delx2 + z2
    r02 = np.sqrt(r02s)
    b02 = np.sqrt(r02s - delx2)
    p02 = delx / (r02 + b02)
    
    # Special case: station(s) at rectangle corner(s)
    isn01 = np.isnan(p01)
    if np.any(isn01):
        p01[isn01 & (x == -a)] = 1
        p01[isn01 & (x == a)] = -1
    
    isn02 = np.isnan(p02)
    if np.any(isn02):
        p02[isn02 & (x == -a)] = 1
        p02[isn02 & (x == a)] = -1
    
    return r01, b01, p01, r02, b02, p02


if __name__ == '__main__':
    # Example usage
    print("Elastic half-space displacement calculation module")
    print("Use mrectloadhs_dif() for multiple rectangles")
    print("Use rectloadhs() for a single rectangle")
