
import numpy as np
from typing import Tuple

def gaussian_func(data ,a ,b ,c):
    """ Generate gaussian function given parameters

    Parameters: data    : 1D np.ndarray in the size of (num_data,)
                a       : float
                b       : float
                c       : float

    Returns:    1D np.ndarray in the size of (num_data,)
    """
    return a * (1 - (data / b )**2) * np.exp(-(data  / c )**2)

def cauchy_func(data ,a ,b):
    """ Generate cauchy function given parameters

    Parameters: data    : 1D np.ndarray in the size of (num_data,)
                a       : float
                b       : float

    Returns:    1D np.ndarray in the size of (num_data,)
    """
    return a / (1 + (data / b )**2)


def covariance_1d(time_vector_mjd: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    time_vector_mjd
    """

    n = int(np.floor(time_vector_mjd.size / 2))

    # Normalize time indices
    tx = (time_vector_mjd - np.min(time_vector_mjd) + 1).astype(int) - 1  # Convert to 0-based indexing

    # Outer product with upper triangular
    dd = np.triu(np.outer(data, data))

    # Create time matrix
    time_range = np.max(time_vector_mjd) - np.min(time_vector_mjd) + 1
    tc = np.zeros((time_range, time_range))
    tc[np.ix_(tx, tx)] = dd

    # Calculate covariance along diagonals
    max_offset = min(n, tc.shape[0] - 1)
    cov = np.zeros(max_offset)

    for k in range(max_offset):
        diag_vals = np.diagonal(tc, offset=k)
        diag_vals = diag_vals[diag_vals != 0]  # Remove zeros
        if len(diag_vals) > 0:
            cov[k] = np.mean(diag_vals)

    # Create bins
    bins_mjd = np.arange(max_offset)
    bins_fyr = np.arange(max_offset) / 365.25

    return bins_mjd, bins_fyr, cov

