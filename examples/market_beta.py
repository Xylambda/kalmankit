"""
By setting the observation matrix H to be one of the stocks (stock_x) and the 
observed variable Z (stock_y) to be the other, the Kalman Filter will compute 
the beta between both stocks, adapting dynamically to changing conditions.

Notice how we added ones to the observation matrix to account for the intercept
or alpha.

See:
[1] Quantdare - The Kalman Filter:
https://quantdare.com/the-kalman-filter/

[2] Quantopian - Kalman Filters:
https://github.com/quantopian/research_public/blob/master/notebooks/lectures/Kalman_Filters/notebook.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
from kalmanfilter import KalmanFilter


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # generate stock returns using a t distribution
    stock_x = 0.01 * np.random.standard_t(df=30, size=2000)
    stock_y = 0.01 * np.random.standard_t(df=30, size=2000)

    # kalman settings
    Z = stock_y.copy()
    U = np.zeros((len(Z), 2, 1)) # control-input vector

    A = np.array([np.eye(2)] * len(Z))
    B = np.zeros((len(Z), 2, 2))
    H = np.expand_dims(np.vstack([[stock_x], [np.ones(len(stock_x))]]).T, axis=1)

    xk = np.array([0, 0])
    Pk = np.ones((2, 2))

    Q = np.array([0.01 * np.eye(2)] * len(Z)) # process noise / transition covariance
    R = np.ones((len(Z))) * 0.01 # measurement noise / observation covariance

    # -------------------------------------------------------------------------
    # run Kalman filter
    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
    states, errors = kf.filter(Z=Z, U=U)
    kalman_gain = np.stack([gain.T[0] for gain in kf.kalman_gains])

    # as array
    states = np.stack([val[:, 0] for val in states])
    errors = np.stack([val[:, 0] for val in errors])

    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 4))

    ax[0][0].plot(states[:, 0])
    ax[0][0].set_title('Estimated means (beta)')

    ax[1][0].plot(states[:, 1])
    ax[1][0].set_title('Estimated means (alpha / intercept)')

    ax[0][1].plot(errors[:, 0])
    ax[0][1].set_title('Estimated covariances (beta)')

    ax[1][1].plot(errors[:, 1])
    ax[1][1].set_title('Estimated covariances (alpha / intercept)')

    plt.tight_layout()
    plt.show()
