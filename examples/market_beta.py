"""
By setting the observation matrix H to be one of the stocks (stock_x) and the 
observed variable Z (stock_y) to be the other, the Kalman Filter will compute 
the beta between both stocks, adapting dynamically to changing conditions.

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
    A = np.expand_dims(np.ones((len(Z),1)), axis=1) # transition matrix
    xk = np.array([[0]]) # initial mean estimate

    B = np.expand_dims(np.zeros((len(Z),1)), axis=1) # control-input matrix
    U = np.zeros((len(Z), 1)) # control-input vector

    Pk = np.array([[1]]) # initial covariance estimate
    Q = np.ones((len(Z))) * 0.005 # measurement noise covariance

    H = np.expand_dims(stock_x.reshape(-1,1), axis=1) # observation matrix
    R = np.ones((len(Z))) * 0.01 # measurement noise covariance

    # -------------------------------------------------------------------------
    # run Kalman filter
    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
    states, errors = kf.run_filter(Z=Z, U=U)

    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    ax[0].plot(states)
    ax[0].set_title('Beta estimated means')

    ax[1].plot(errors)
    ax[1].set_title('Beta estimated covariances')

    plt.show()
