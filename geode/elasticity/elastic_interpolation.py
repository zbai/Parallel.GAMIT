#!/usr/bin/env python
"""
Elastic interpolation of 2-D vector data using constraints from elasticity

Based on:
Sandwell, D. T., and P. Wessel (2016), Interpolation of 2-D vector data using
constraints from elasticity, Geophys. Res. Lett., 43, 10,703–10,709,
doi:10.1002/2016GL070340.

Translation of MATLAB gpsgridder.m to Python
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def gpsgridder(xo: np.ndarray, yo: np.ndarray, 
               xi: np.ndarray, yi: np.ndarray,
               ui: np.ndarray, vi: np.ndarray,
               dr: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin-plate coupled interpolation of GPS velocity components
    
    Uses Green's functions from elasticity theory to interpolate vector
    velocity data with coupling between components controlled by Poisson's ratio.
    
    Parameters
    ----------
    xo : np.ndarray
        Equidistant lattice for output x nodes (1D array)
    yo : np.ndarray
        Equidistant lattice for output y nodes (1D array)
    xi : np.ndarray
        x-coordinates of data points (1D array, length n)
    yi : np.ndarray
        y-coordinates of data points (1D array, length n)
    ui : np.ndarray
        x-components of GPS velocities at data points (1D array, length n)
    vi : np.ndarray
        y-components of GPS velocities at data points (1D array, length n)
    dr : float
        Minimum distance between points (fudge factor to prevent singularities)
        Typically ~mean GPS spacing. Units same as xi, yi.
    nu : float
        Poisson's ratio controlling elastic coupling
        - nu = -1.0: fully decoupled (no coupling between components)
        - nu = 0.5: typical elastic
        - nu = 1.0: incompressible
        
    Returns
    -------
    U : np.ndarray
        Grid with east-west velocities (2D array, shape [len(yo), len(xo)])
    V : np.ndarray
        Grid with north-south velocities (2D array, shape [len(yo), len(xo)])
        
    Notes
    -----
    The Green's functions are based on equation (8) from Sandwell & Wessel (2016):
        q(r) = (3 - ν)ln(r) + (1 + ν)y²/r²
        p(r) = (3 - ν)ln(r) + (1 + ν)x²/r²  
        w(r) = -(1 + ν)xy/r²
    
    The method solves equation (9) to find force strengths, then uses
    equation (10) to interpolate velocities at any location.
    """
    n = len(xi)
    n1 = n + 1
    n2 = 2 * n
    
    # Remove mean value from each velocity component
    u0 = np.mean(ui)
    ui_centered = ui - u0
    v0 = np.mean(vi)
    vi_centered = vi - v0
    
    # Evaluate the three Green's functions q, p, w at data locations
    # This creates the 2N x 2N system matrix
    q, p, w = get_qpw(np.column_stack([xi, yi]), 
                      np.column_stack([xi, yi]), dr, nu)
    
    # Solve for the weights wt (force strengths at data locations)
    # Build the block matrix from equation (9):
    # [u] = [q  w] [fx]
    # [v]   [w  p] [fy]
    A = np.block([[q, w],
                  [w, p]])
    b = np.concatenate([ui_centered, vi_centered])
    
    # Solve the linear system
    wt = np.linalg.solve(A, b)
    
    logger.info(f'Solved for {n2} force weights using {n} data points')
    
    # Create the 2-D mesh grid based on output xo, yo vectors
    X, Y = np.meshgrid(xo, yo)
    nrows, ncols = X.shape
    
    # Evaluate solution for U, V using equation (10)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    
    xi_yi = np.column_stack([xi, yi])
    
    for row in range(nrows):
        for col in range(ncols):
            # Evaluate Green's functions from data points to this grid point
            q_eval, p_eval, w_eval = get_qpw(xi_yi, 
                                             np.array([[X[row, col], Y[row, col]]]), 
                                             dr, nu)
            # Sum weighted contributions from all data points
            U[row, col] = u0 + np.dot(q_eval.flatten(), wt[:n]) + np.dot(w_eval.flatten(), wt[n:n2])
            V[row, col] = v0 + np.dot(w_eval.flatten(), wt[:n]) + np.dot(p_eval.flatten(), wt[n:n2])
    
    logger.info(f'Interpolated to {nrows} x {ncols} grid')
    
    return U, V


def get_radius(X: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate distances and differences between point sets
    
    Parameters
    ----------
    X : np.ndarray
        Source points with 2-D coordinates in rows (n x 2)
    P : np.ndarray
        Target points with 2-D coordinates in rows (m x 2)
        
    Returns
    -------
    r : np.ndarray
        Radial distances (m x n)
    dx : np.ndarray
        x-differences: X[:,0] - P[:,0] (m x n)
    dy : np.ndarray
        y-differences: X[:,1] - P[:,1] (m x n)
    """
    n = X.shape[0]  # number of points in X
    m = P.shape[0]  # number of points in P

    # Calculate differences using broadcasting
    # Result is (m x n) where each row corresponds to a point in P
    # and each column corresponds to a point in X
    dx = X[:, 0] - P[:, 0][:, np.newaxis]  # (m x n)
    dy = X[:, 1] - P[:, 1][:, np.newaxis]  # (m x n)
    
    # Calculate radial distance
    r = np.hypot(dx, dy)
    
    return r, dx, dy


def get_qpw(X: np.ndarray, P: np.ndarray, dr: float, nu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Green's functions for all (X, P) combinations
    
    Implements equation (8) from Sandwell & Wessel (2016).
    
    Parameters
    ----------
    X : np.ndarray
        Source points with 2-D coordinates in rows (n x 2)
    P : np.ndarray
        Target points with 2-D coordinates in rows (m x 2)
    dr : float
        Minimum distance (fudge factor) to prevent singularities at r=0
    nu : float
        Poisson's ratio
        
    Returns
    -------
    q : np.ndarray
        Green's function q(r) = (3-ν)ln(r) + (1+ν)y²/r² (m x n)
    p : np.ndarray
        Green's function p(r) = (3-ν)ln(r) + (1+ν)x²/r² (m x n)
    w : np.ndarray
        Green's function w(r) = -(1+ν)xy/r² (m x n)
        
    Notes
    -----
    The fudge factor dr is added to r to prevent singularities. This is 
    equivalent to assuming a minimum resolvable distance between points.
    """
    r, dx, dy = get_radius(X, P)
    
    # Add fudge dr term to prevent singularities
    r_fudged = r + dr
    logr = np.log(r_fudged)
    r2 = r_fudged ** 2
    
    # Compute Green's functions from equation (8)
    q = (3 - nu) * logr + (1 + nu) * (dy ** 2) / r2
    p = (3 - nu) * logr + (1 + nu) * (dx ** 2) / r2
    w = -(1 + nu) * dx * dy / r2
    
    return q, p, w


def interpolate_at_points(xi: np.ndarray, yi: np.ndarray,
                         ui: np.ndarray, vi: np.ndarray,
                         xo: np.ndarray, yo: np.ndarray,
                         dr: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate velocities at arbitrary output points (not necessarily on a grid)
    
    This is a more general version that interpolates at specific points
    rather than creating a full grid.
    
    Parameters
    ----------
    xi, yi : np.ndarray
        Input data point coordinates (1D arrays, length n)
    ui, vi : np.ndarray
        Input velocity components (1D arrays, length n)
    xo, yo : np.ndarray
        Output point coordinates (1D arrays, length m)
    dr : float
        Minimum distance parameter
    nu : float
        Poisson's ratio
        
    Returns
    -------
    uo : np.ndarray
        Interpolated x-velocities at output points (1D array, length m)
    vo : np.ndarray
        Interpolated y-velocities at output points (1D array, length m)
    """
    n = len(xi)
    n2 = 2 * n
    m = len(xo)
    
    # Remove mean
    u0 = np.mean(ui)
    v0 = np.mean(vi)
    ui_centered = ui - u0
    vi_centered = vi - v0
    
    # Build and solve system
    q, p, w = get_qpw(np.column_stack([xi, yi]), 
                      np.column_stack([xi, yi]), dr, nu)
    
    A = np.block([[q, w],
                  [w, p]])
    b = np.concatenate([ui_centered, vi_centered])
    wt = np.linalg.solve(A, b)
    
    # Evaluate at output points
    uo = np.zeros(m)
    vo = np.zeros(m)
    
    xi_yi = np.column_stack([xi, yi])
    xo_yo = np.column_stack([xo, yo])
    
    q_eval, p_eval, w_eval = get_qpw(xi_yi, xo_yo, dr, nu)
    
    for i in range(m):
        uo[i] = u0 + np.dot(q_eval[i, :], wt[:n]) + np.dot(w_eval[i, :], wt[n:n2])
        vo[i] = v0 + np.dot(w_eval[i, :], wt[:n]) + np.dot(p_eval[i, :], wt[n:n2])
    
    return uo, vo


def compute_strain_rate(xi: np.ndarray, yi: np.ndarray,
                       ui: np.ndarray, vi: np.ndarray,
                       xo: np.ndarray, yo: np.ndarray,
                       dr: float, nu: float,
                       delta: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute strain rate tensor components from velocity interpolation
    
    Parameters
    ----------
    xi, yi : np.ndarray
        Input data point coordinates
    ui, vi : np.ndarray
        Input velocity components
    xo, yo : np.ndarray
        Output point coordinates (1D arrays for grid)
    dr : float
        Minimum distance parameter
    nu : float
        Poisson's ratio
    delta : float
        Distance for numerical derivatives (in same units as coordinates)
        
    Returns
    -------
    exx : np.ndarray
        Normal strain rate ∂u/∂x
    exy : np.ndarray
        Shear strain rate (∂u/∂y + ∂v/∂x)/2
    eyy : np.ndarray
        Normal strain rate ∂v/∂y
        
    Notes
    -----
    Strain rate components are computed using finite differences.
    The second invariant can be computed as: sqrt(exx² + eyy² + 2*exy²)
    """
    # Create grid
    X, Y = np.meshgrid(xo, yo)
    nrows, ncols = X.shape
    
    exx = np.zeros_like(X)
    eyy = np.zeros_like(X)
    exy = np.zeros_like(X)
    
    # Compute velocities at slightly offset points for derivatives
    for row in range(nrows):
        for col in range(ncols):
            x = X[row, col]
            y = Y[row, col]
            
            # Points for finite differences
            xp = np.array([x + delta, x - delta, x, x])
            yp = np.array([y, y, y + delta, y - delta])
            
            up, vp = interpolate_at_points(xi, yi, ui, vi, xp, yp, dr, nu)
            
            # Central differences
            exx[row, col] = (up[0] - up[1]) / (2 * delta)  # ∂u/∂x
            eyy[row, col] = (vp[2] - vp[3]) / (2 * delta)  # ∂v/∂y
            
            # Shear strain (engineering definition)
            dudy = (up[2] - up[3]) / (2 * delta)  # ∂u/∂y
            dvdx = (vp[0] - vp[1]) / (2 * delta)  # ∂v/∂x
            exy[row, col] = 0.5 * (dudy + dvdx)
    
    return exx, exy, eyy


if __name__ == '__main__':
    # Simple test case
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data: simple shear velocity field
    xi = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20])
    yi = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20])
    ui = yi * 0.1  # Linear shear in x
    vi = np.zeros_like(xi)
    
    # Create output grid
    xo = np.linspace(-5, 25, 31)
    yo = np.linspace(-5, 25, 31)
    
    # Interpolate with elastic coupling
    U, V = gpsgridder(xo, yo, xi, yi, ui, vi, dr=2.0, nu=0.5)
    
    print(f"Input velocity range: u=[{ui.min():.2f}, {ui.max():.2f}], v=[{vi.min():.2f}, {vi.max():.2f}]")
    print(f"Output velocity range: U=[{U.min():.2f}, {U.max():.2f}], V=[{V.min():.2f}, {V.max():.2f}]")
    print("Test completed successfully!")
