"""
In this example we are going to see how to use the Extended Kalman Filter. The
problem configuration is taken from the book "Bayesian Filtering and Smoothing"
by Simo Särkkä: Example 5.1 of Bayesian Filtering and Smoothing.

The Unofficial associated code for the book was als used:
https://github.com/EEA-sensors/Bayesian-Filtering-and-Smoothing

To generate the observations we have to use the equations that define the 
system:

xk = f(xk-1, uk-1) + qk
zk = h(xk) + rk

Then, we set the parameters and the jacobian matrices.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv
from kalmankit import ExtendedKalmanFilter

# constants
DT = 0.01  # delta t
G = 9.81
np.random.seed(1)


def f(xk, uk=None):
    arr = np.array([xk[0] + DT * xk[1], xk[1] - G * DT * np.sin(xk[0])])
    return arr


def jacobian_A(xk, uk=None):
    """
    Jacobian of f with respect to x.
    """
    arr = np.array([[1, DT], [-G * np.cos(xk[0]) * DT, 1]])
    return arr
    
    
def h(xk):
    return np.sin(xk[0])


def jacobian_H(xk):
    """
    Jacobian of h with respect to x.
    """
    jac = np.array([[np.cos(xk[0]), 0.]])
    return jac


def generate_observations(f, h, qk, rk, size=100):
    # -------------------------------------------------------------------------
    # initial mean estimate
    xk = np.array([1.5, 0.])
    
    # define noises
    cholesky_qk = np.linalg.cholesky(qk)

    # -------------------------------------------------------------------------
    # generate observations
    Z = np.empty((size, 2))
    X = np.empty((size, 2))
    for k in range(0, size):
        
        noise = erfcinv(2 * np.random.rand(2))
        
        xk_ = f(xk, None) + cholesky_qk @ noise
        zk = h(xk_) + np.sqrt(rk)
        Z[k] = np.sin(zk) + np.sqrt(rk) + noise
        X[k] = xk_

        xk = xk_
    
    time = np.arange(DT, (size + 1) * DT, DT)

    return Z, X, time


def main():
    # -------------------------------------------------------------------------
    xk = np.array([1.5, 0.])
    Pk = np.array(
        [
            [0.1, 0.], 
            [0., 0.1]
        ]
    )

    qk  = 0.01 * np.array(
        [
            [DT ** 3 / 3, DT ** 2 / 2], 
            [DT ** 2 / 2, DT]
        ]
    )
    rk = 0.1

    Z, true_X, time = generate_observations(f, h, qk=qk, rk=0.01, size=500)

    Q = np.stack([qk] * len(Z))
    R = np.ones(len(Z)) * rk

    # -------------------------------------------------------------------------
    ekf = ExtendedKalmanFilter(
        xk=xk,
        Pk=Pk,
        Q=Q,
        R=R,
        f=f,
        h=h,
        jacobian_A=jacobian_A,
        jacobian_H=jacobian_H,
    )

    states, errors = ekf.filter(Z[:, 0], None)

    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.scatter(time, Z[:, 0], alpha=0.5, label="Observations")
    ax.plot(time, true_X[:, 0], color="red", label="True State")
    ax.plot(
        time,
        states[:, 0],
        color="orange",
        label="EKF Estimate",
        linestyle="--"
    )
    ax.grid(True, alpha=0.5)
    ax.set_title("EKF Pendulum", fontsize=25)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
