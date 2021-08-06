"""
By setting the transition and observation matrices to the identity, we can use 
the Kalman Filter to estimate an exponentially weighted moving average, where 
the window is decided by the Kalman Gain K.

See
[1] Quantopian - Kalman Filters:
https://github.com/quantopian/research_public/blob/master/notebooks/lectures/Kalman_Filters/notebook.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
from kalmanfilter import KalmanFilter


# -----------------------------------------------------------------------------
# function to generate observations
def generate_func(start, end, step, beta, var):
    """
    Generate a noisy sine wave function
    
    Parameters
    ----------
    start : int
        Initial X value.
    end : int
        Final X value.
    step : float or int
        Space between values.
    beta : float or int
        Slope of the sine wave.
    var : float or int
        Noise variance.
        
    Returns
    -------
    out : numpy.array
        Noisy sine wave.
    """
    _space = np.arange(start=start, stop=end, step=step)
    _sin = np.sin(_space)

    out = np.array(
        [beta*x + var*np.random.randn() for x in range(len(_space))]
    )
    out += _sin
    return out


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # generate noisy observations
    Z = generate_func(start=-10, end=10, step=0.1, beta=0.02, var=0.3)

    # kalman settings
    A = np.expand_dims(np.ones((len(Z),1)), axis=1) # transition matrix
    xk = np.array([[1]]) # initial mean estimate

    B = np.expand_dims(np.zeros((len(Z),1)), axis=1) # control-input matrix
    U = np.zeros((len(Z), 1)) # control-input vector

    Pk = np.array([[1]]) # initial covariance estimate
    Q = np.ones((len(Z))) * 0.005 # process noise covariance

    H = A.copy() # observation matrix
    R = np.ones((len(Z))) * 0.01 # measurement noise covariance

    # -------------------------------------------------------------------------
    # run Kalman filter
    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
    states, errors = kf.run_filter(Z=Z, U=U)

    # plot
    plt.figure(figsize=(15,7))
    plt.plot(Z)
    plt.plot(states)
    plt.show()
