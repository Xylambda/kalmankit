"""
Standard Kalman filtering and RTS smoothing for a constant-velocity body.

The hidden state is position and velocity. We observe noisy positions and use
the smoother to refine earlier estimates after the full observation sequence is
available.
"""

import matplotlib.pyplot as plt
import numpy as np

from kalmankit import KalmanFilter

DT = 0.1
N_STEPS = 200
PROCESS_ACCELERATION_STD = 0.35
OBSERVATION_STD = 2.0


def transition_covariance(dt, acceleration_std):
    """Discrete white-noise acceleration covariance."""
    variance = acceleration_std**2
    return variance * np.array(
        [
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2],
        ]
    )


def generate_observations(size=N_STEPS, seed=1):
    rng = np.random.default_rng(seed)

    A = np.array([[1.0, DT], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = transition_covariance(DT, PROCESS_ACCELERATION_STD)

    states = np.empty((size, 2))
    observations = np.empty(size)
    xk = np.array([0.0, 1.2])

    for k in range(size):
        if k > 0:
            xk = A @ xk + rng.multivariate_normal(np.zeros(2), Q)

        states[k] = xk
        observations[k] = H @ xk + rng.normal(scale=OBSERVATION_STD)

    time = np.arange(size) * DT

    return observations, states, time


def main():
    Z, true_states, time = generate_observations()

    transition_matrix = np.array([[1.0, DT], [0.0, 1.0]])
    observation_matrix = np.array([[1.0, 0.0]])
    process_covariance = transition_covariance(DT, PROCESS_ACCELERATION_STD)

    A = np.concatenate(
        [
            np.eye(2)[None],
            np.tile(transition_matrix, (len(Z) - 1, 1, 1)),
        ]
    )
    H = np.tile(observation_matrix, (len(Z), 1, 1))
    Q = np.concatenate(
        [
            np.zeros((1, 2, 2)),
            np.tile(process_covariance, (len(Z) - 1, 1, 1)),
        ]
    )
    R = np.ones(len(Z)) * OBSERVATION_STD**2

    kf = KalmanFilter(
        A=A,
        xk=np.array([0.0, 0.0]),
        B=None,
        Pk=np.eye(2) * 10.0,
        H=H,
        Q=Q,
        R=R,
    )

    filtered_states, _ = kf.filter(Z=Z)
    smoothed_states, _ = kf.smooth(Z=Z)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=True)

    ax[0].scatter(time, Z, s=12, alpha=0.35, label="Noisy observations")
    ax[0].plot(time, true_states[:, 0], color="black", label="True position")
    ax[0].plot(time, filtered_states[:, 0], label="Filtered position")
    ax[0].plot(time, smoothed_states[:, 0], label="Smoothed position")
    ax[0].set_ylabel("Position")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(time, true_states[:, 1], color="black", label="True velocity")
    ax[1].plot(time, filtered_states[:, 1], label="Filtered velocity")
    ax[1].plot(time, smoothed_states[:, 1], label="Smoothed velocity")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Velocity")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.suptitle("Standard RTS smoothing: constant-velocity tracking")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
