"""
Extended Kalman filtering and smoothing for a nonlinear pendulum.

The hidden state is angle and angular velocity. We observe noisy values of
sin(angle), then smooth the filtered estimates with the Extended RTS smoother.
"""

import matplotlib.pyplot as plt
import numpy as np

from kalmankit import ExtendedKalmanFilter

DT = 0.02
G = 9.81
N_STEPS = 300
OBSERVATION_STD = 0.12


def f(xk, uk=None):
    return np.array(
        [
            xk[0] + DT * xk[1],
            xk[1] - G * DT * np.sin(xk[0]),
        ]
    )


def jacobian_A(xk, uk=None):
    return np.array(
        [
            [1.0, DT],
            [-G * DT * np.cos(xk[0]), 1.0],
        ]
    )


def h(xk):
    return np.sin(xk[0])


def jacobian_H(xk):
    return np.array([[np.cos(xk[0]), 0.0]])


def generate_observations(qk, rk, size=N_STEPS, seed=1):
    rng = np.random.default_rng(seed)

    states = np.empty((size, 2))
    observations = np.empty(size)
    xk = np.array([1.4, 0.0])

    for k in range(size):
        if k > 0:
            xk = f(xk) + rng.multivariate_normal(np.zeros(2), qk)

        states[k] = xk
        observations[k] = h(xk) + rng.normal(scale=np.sqrt(rk))

    time = np.arange(size) * DT

    return observations, states, time


def main():
    qk = 0.01 * np.array(
        [
            [DT**3 / 3, DT**2 / 2],
            [DT**2 / 2, DT],
        ]
    )
    rk = OBSERVATION_STD**2

    Z, true_states, time = generate_observations(qk=qk, rk=rk)

    ekf = ExtendedKalmanFilter(
        xk=np.array([1.2, 0.0]),
        Pk=np.eye(2) * 0.2,
        Q=np.stack([qk] * len(Z)),
        R=np.ones(len(Z)) * rk,
        f=f,
        h=h,
        jacobian_A=jacobian_A,
        jacobian_H=jacobian_H,
    )

    filtered_states, _ = ekf.filter(Z=Z)
    smoothed_states, _ = ekf.smooth(Z=Z)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=True)

    ax[0].scatter(time, Z, s=12, alpha=0.35, label="Noisy sin(angle)")
    ax[0].plot(time, np.sin(true_states[:, 0]), color="black", label="True sin(angle)")
    ax[0].plot(time, np.sin(filtered_states[:, 0]), label="Filtered sin(angle)")
    ax[0].plot(time, np.sin(smoothed_states[:, 0]), label="Smoothed sin(angle)")
    ax[0].set_ylabel("Observation space")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(time, true_states[:, 0], color="black", label="True angle")
    ax[1].plot(time, filtered_states[:, 0], label="Filtered angle")
    ax[1].plot(time, smoothed_states[:, 0], label="Smoothed angle")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Angle")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.suptitle("Extended RTS smoothing: nonlinear pendulum")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
