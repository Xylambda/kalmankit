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
from kalmankit import KalmanFilter


# -----------------------------------------------------------------------------
# function to generate observations
def generate_func(start, end, step, beta, var):
    """Generate a noisy sine wave function.

    This is actually a non-linear function but it will serve its purpose since
    we only want to show how to set up the filter.
    
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
        [beta * x + var * np.random.randn() for x in range(len(_space))]
    )
    out += _sin
    return out


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # generate noisy observations
    Z = generate_func(start=-10, end=10, step=0.1, beta=0.02, var=0.3)

    # kalman settings
    A = np.expand_dims(np.ones((len(Z),1)), axis=1)  # transition matrix
    xk = np.array([[1]])  # initial mean estimate

    B = np.expand_dims(np.zeros((len(Z),1)), axis=1)  # control-input matrix
    U = np.zeros((len(Z), 1))  # control-input vector

    Pk = np.array([[1]])  # initial covariance estimate
    Q = np.ones((len(Z))) * 0.005  # process noise covariance

    H = A.copy()  # observation matrix
    R = np.ones((len(Z))) * 0.01  # measurement noise covariance

    # -------------------------------------------------------------------------
    # run Kalman filter
    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
    states, errors = kf.filter(Z=Z, U=U)
    kalman_gain = np.stack([val.item() for val in kf.kalman_gains])

    # as array
    states = np.stack([val.item() for val in states])
    errors = np.stack([val.item() for val in errors])

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    ax[0].plot(Z, label="Observations")
    ax[0].plot(states, label="Estimated Estates")
    ax[0].set_title('Estimated means over signal')
    ax[0].legend()

    ax[1].plot(errors)
    ax[1].set_title('Covariances')

    ax[2].plot(kalman_gain)
    ax[2].set_title('Kalman Gain')

    plt.show()
