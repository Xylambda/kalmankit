""" A simple implementation of Kalman filter algorithm using NumPy. """
import numpy as np


def kalman_filter(Z, xk, Pk, A, H, Q, R):
    """Kalman filter.
    
    Applies the Kalman filter algorithm over a given time series. See 2nd and 
    3rd references to understand notation.
    
    Parameters
    ----------
    Z : numpy.array
        The observable data (input variable).
    xk : float
        Initial mean estimate.
    Pk : float
        Initial covariance estimate.
    A : numpy.array
        A matrix that relates the state at the previous time step k-1 to the 
        state at the current step k.
    H : numpy.array
        A matrix that relates the state to the measurement z_k.
    Q : float
        Process noise covariance.
    R : float
        Measurement noise covariance.
        
    Returns
    -------
    states : numpy.array
        A posteriori state estimates for each time step.
    errors : numpy.array
        A posteriori estimate error covariances for each time step.
        
    References
    ----------
    .. [1] Matlab - Understanding Kalman Filters:
       https://www.youtube.com/playlist?list=PLn8PRpmsu08pzi6EMiYnR-076Mh-q3tWr

    .. [2] Bilgin's Blog - Kalman filter for dummies.
       http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies
       
    .. [3] Greg Welch, Gary Bishop - An Introduction to the Kalman Filter:
       https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    
    """
    states = np.zeros_like(Z)
    errors = np.zeros_like(Z)
    
    for k, (zk, Hk) in enumerate(zip(Z, H)):
        # prediction step
        xk = A * xk
        Pk = A * Pk * A.T + Q
        
        # update step
        Kk = (Pk * Hk.T) / (Hk * Pk * Hk.T + R) # kalman gain
        xk = xk + Kk * (zk - Hk * xk)
        Pk = (1 - Kk * Hk) * Pk # 1 is the identity matrix
        
        # a posteriori estimates
        states[k] = xk
        errors[k] = Pk
        
    return states, errors


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
