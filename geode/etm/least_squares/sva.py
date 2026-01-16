import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def sva(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 int, int, np.ndarray, np.ndarray, 
                                                 float, np.ndarray]:
    """
    Singular value analysis of the LS problem Ax = b
    
    INPUT:
    A : ndarray
        The design matrix in which the (i,j) element is the j-th candidate 
        basis function evaluated for the i-th data point.
    b : ndarray
        The values of the response variable (column vector)
    
    OUTPUT:
    x : ndarray
        The solution vector given kp effective degrees of freedom
        (i.e., the preferred solution)
    bh : ndarray
        The model values for the data vector b
    r : ndarray
        The data residuals (observed - calculated)
    kp : int
        The preferred effective number of degrees of freedom
    nsv : int
        The number of non-negligible singular values for A
    sv : ndarray
        The non-negligible singular values of A
    rmsr : ndarray
        The rms residuals associated with the candidate solutions
    rmsrp : float
        The rms residual associated with the preferred fit
    covx : ndarray
        The covariance matrix associated with solution x
    
    REFERENCES:
    Lawson and Hanson (1974) 'Solving Least Squares Problems'
      especially sections 18.4 and 25.6, and
    Forsythe, Malcolm and Moler (1977) 'Computer Methods for 
      Mathematical Computations'
    
    Mike Bevis       version 2.0       February 1995
    Translated to Python 2025
    """
    
    # Input validation
    m, n = A.shape
    
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    
    br, bc = b.shape
    
    if bc != 1:
        raise ValueError('b must be a column vector')
    
    if br != m:
        raise ValueError('A and b must have same number of rows')
    
    if m <= n:
        raise ValueError('A must have more rows than columns')
    
    # Unscaled covariance matrix for solution vector (scale later)
    covx = np.linalg.inv(A.T @ A)
    
    # Display header
    dash = '----------------------'
    print()
    print(f'{dash} sva {dash}')
    print(f' # observations, m= {m}')
    print(f' # basis functions, n= {n}')
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T
    Sdiag = S
    
    # Determine tolerance and count significant singular values
    tol = n * Sdiag[0] * np.finfo(float).eps  # see FMM (1977) p.197 on tolerance
    nsv = np.sum(Sdiag > tol)
    
    print(f' # singular values significantly > 0, nsv= {nsv}')
    
    sv = Sdiag[:nsv]  # vector of effective non-zero SVs
    
    ############ COMPUTE THE RESOLUTION SPECTRUM ############
    g = U.T @ b
    g2 = g ** 2
    ssr = np.zeros(nsv)
    
    for k in range(nsv):
        ssr[k] = np.sum(g2[(k+1):m])
    
    rmsr = np.sqrt(ssr / m)
    rmsb = np.sqrt(np.sum(b ** 2) / m)
    
    ############ GRAPH THE SV AND RESOLUTION SPECTRUM ############
    i = np.arange(1, nsv + 1)
    
    fig = plt.figure(figsize=(12, 8))
    
    # Plot 1: Singular Value Spectrum
    plt.subplot(2, 2, 1)
    plt.semilogy(i, sv, '*-')
    plt.ylabel('k-th singular value')
    plt.xlabel('k')
    plt.grid(True)
    plt.title('Singular Value Spectrum')
    
    # Plot 2: Model Resolution
    plt.subplot(2, 2, 2)
    plt.plot(i, rmsr, '*-')
    plt.ylabel('r.m.s. residual')
    plt.xlabel('k  (effective n.d.o.f.)')
    plt.grid(True)
    plt.title('Model Resolution')
    
    ############ TABULATE THE RESOLUTION SPECTRUM ############
    print()
    print('Table showing rms_dev between data and model as a')
    print('function of e.n.d.o.f (k) assigned to the model:')
    print()
    print('      k  rms_dev        k  rms_dev         k  rms_dev')
    
    # Print table in 3 columns
    for start_idx in range(0, nsv, 3):
        line = ''
        for col in range(3):
            idx = start_idx + col
            if idx < nsv:
                line += f'    {idx+1:3d}  {rmsr[idx]:8.5f}   '
        print(line)
    
    ############ HAVE USER CHOOSE THE PREFERRED SOLN ############
    while True:
        try:
            kp = int(input(f'Enter preferred e.n.d.o.f., kp (1-{nsv}): '))
            if 1 <= kp <= nsv:
                break
            else:
                print(f'Please enter a value between 1 and {nsv}')
        except ValueError:
            print('Please enter a valid integer')
    
    ############ EVALUATE THE CANDIDATE SOLN ################
    rmsrp = rmsr[kp-1]  # Python uses 0-based indexing
    print()
    print(f' rms residual = {rmsrp}')
    print(f' cf rms observed values = {rmsb[0, 0]}')
    
    p = np.zeros((n, 1))
    p[:kp] = g[:kp] / sv[:kp].reshape(-1, 1)
    x = V @ p
    bh = A @ x
    r = b - bh
    
    # Scale factor for the solution covariance matrix
    sf = ssr[kp-1] / (m - kp)
    covx = sf * covx
    
    ############ GRAPH THE RESIDUALS ################
    plt.subplot(2, 2, 3)
    plt.plot(r, '+')
    plt.ylabel('observed - predicted')
    plt.xlabel('datum number')
    plt.grid(True)
    plt.title(f'Residuals   (kp={kp})')
    
    plt.tight_layout()
    plt.show()
    
    print(f'{dash}-----{dash}')
    
    return x, bh, r, kp, nsv, sv, rmsr, rmsrp, covx


# Example usage:
if __name__ == '__main__':
    # Example: fit a polynomial to noisy data
    np.random.seed(42)
    
    # Generate synthetic data
    x_data = np.linspace(0, 10, 50)
    y_true = 2 * x_data + 3 * x_data**2 - 0.5 * x_data**3
    y_data = y_true + np.random.normal(0, 50, len(x_data))
    
    # Create design matrix for polynomial fit (degree 5)
    degree = 5
    A = np.column_stack([x_data**i for i in range(degree + 1)])
    b = y_data.reshape(-1, 1)
    
    # Perform SVA
    x, bh, r, kp, nsv, sv, rmsr, rmsrp, covx = sva(A, b)
    
    print("\nSolution vector:")
    print(x.flatten())
    print(f"\nPreferred degrees of freedom: {kp}")
    print(f"RMS residual: {rmsrp}")
