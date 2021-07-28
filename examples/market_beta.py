"""
Setting the observation matrix H to be one of the stocks (stock_x) and the 
observed variable Z (stock_y) to be the other, the Kalman Filter will compute 
the beta between both stocks, adapting dynamically to changing conditions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kalmanfilter import KalmanFilter


# -----------------------------------------------------------------------------
# generate stock returns using a t distribution
stock_x = 0.01 * np.random.standard_t(df=30, size=2000)
stock_y = 0.01 * np.random.standard_t(df=30, size=2000)

# kalman settings
Z = stock_y.copy()
A = np.expand_dims(np.ones((len(Z),1)), axis=1)
xk = np.array([[0]])

B = np.expand_dims(np.zeros((len(Z),1)), axis=1)
U = np.zeros((len(Z), 1))

Pk = np.array([[1]])
Q = np.ones((len(Z))) * 0.005

H = np.expand_dims(stock_x.reshape(-1,1), axis=1)
R = np.ones((len(Z))) * 0.01

# -----------------------------------------------------------------------------
# run Kalman filter
kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
states, errors = kf.run_filter(Z=Z, U=U)

# -----------------------------------------------------------------------------
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

ax[0].plot(states)
ax[0].set_title('Estimated means')

ax[1].plot(errors)
ax[1].set_title('Estimated covariances')

plt.show()