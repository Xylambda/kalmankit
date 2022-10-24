"""
In this example we are going to see how to use the Extended Kalman Filter. The
problem configuration is random and does nothing particularly useful.

To generate the observations we have to use the equations that define the 
system:

xk = f(xk-1, uk-1) + qk
zk = h(xk) + rk

Then, we set the parameters and the jacobian matrices.
"""
import numpy as np
import matplotlib.pyplot as plt
from kalmankit import ExtendedKalmanFilter


def generate_observations(f, h, size=100):
    """
    Generate observations using nonlinear discrete differentiable functions
    f and h.
    """
    X = np.random.rand(size)
    U = np.random.rand(size)
    Q = np.random.normal(loc=0, scale=0.005, size=size)
    R = np.random.normal(loc=0, scale=0.01, size=size)

    Z = np.empty_like(X)
    for k, (xk, uk, qk, rk) in enumerate(zip(X, U, Q, R)):
        xk_ = f(xk, uk) + qk
        zk = h(xk_) + rk
        Z[k] = zk

    return Z


def main():
    # -------------------------------------------------------------------------
    # define functions and jacobian matrices
    def f(xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        State-transition non-linear function.
        """
        return np.sin(xk) + uk

    def h(xk: np.ndarray) -> np.ndarray:
        """
        Observation non-linear function.
        """
        return np.tan(xk)

    def jacobian_A(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Jacobian of f with respect to x.
        """
        jac = np.cos(x)
        return jac

    def jacobian_H(x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h with respect to x.
        """
        jac = np.arccos(x) ** 2
        return jac

    # -------------------------------------------------------------------------
    # kalman settings
    xk = np.array([[1]])  # initial mean estimate
    Pk = np.array([[1]])  # initial covariance estimate

    Z = generate_observations(f=f, h=h, size=100) # observations
    U = np.zeros((len(Z), 1))  # control-input vector

    Q = np.ones((len(Z))) * 0.005  # process noise covariance
    R = np.ones((len(Z))) * 0.01  # measurement noise covariance

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

    states, errors = ekf.filter(Z=Z, U=U)
    kalman_gain = np.stack([val.item() for val in ekf.kalman_gains])

    # plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
    ax[0].plot(Z, label="Observations")
    ax[0].plot(states, label="Estimated estates")
    ax[0].set_title('Estimated means over signal')
    ax[0].legend()

    ax[1].plot(errors.flatten())
    ax[1].set_title('Covariances')

    ax[2].plot(kalman_gain)
    ax[2].set_title('Kalman Gain')

    plt.show()


if __name__ == "__main__":
    main()
