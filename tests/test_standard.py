import numpy as np
from kalmankit import KalmanFilter
from pykalman import KalmanFilter as PyKalmanFilter

data = np.array([0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45])


def test_predict_covariance_matches_formula():
    Ak = np.array([[1.0, 0.2], [-0.1, 0.9]])
    xk = np.array([0.4, -0.2])
    Pk = np.array([[0.7, 0.1], [0.1, 0.5]])
    Qk = np.array([[0.03, 0.01], [0.01, 0.04]])

    kf = KalmanFilter(
        A=np.array([Ak]),
        xk=xk,
        B=None,
        Pk=Pk,
        H=np.array([np.eye(2)]),
        Q=np.array([Qk]),
        R=np.array([np.eye(2)]),
    )

    _, obtained = kf.predict(
        Ak=Ak,
        xk=xk,
        Bk=np.full_like(Ak, np.nan),
        uk=np.full(2, np.nan),
        Pk=Pk,
        Qk=Qk,
    )
    expected = Ak @ Pk @ Ak.T + Qk

    np.testing.assert_allclose(obtained, expected)


def test_filter():
    expected_states = np.array(
        [
            0.35454545,
            0.42380952,
            0.44193548,
            0.40487805,
            0.3745098,
            0.36557377,
            0.36197183,
            0.37654321,
            0.38021978,
            0.38712871,
        ]
    )

    expected_covariances = np.array(
        [
            0.09090909,
            0.04761905,
            0.03225806,
            0.02439024,
            0.01960784,
            0.01639344,
            0.01408451,
            0.01234568,
            0.01098901,
            0.00990099,
        ]
    )

    # set parameters
    Z = data
    A = np.expand_dims(np.ones((len(Z), 1)), axis=1)
    xk = np.array([[0]])

    B = np.expand_dims(np.zeros((len(Z), 1)), axis=1)
    U = np.zeros((len(Z), 1))

    Pk = np.array([[1]])
    H = np.expand_dims(np.ones((len(Z), 1)), axis=1)
    Q = np.zeros((len(Z)))
    R = np.ones((len(Z))) * 0.1

    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
    states, covariances = kf.filter(Z=Z, U=U)

    np.testing.assert_allclose(states.flatten(), expected_states, rtol=1e-06, atol=0)

    np.testing.assert_allclose(
        covariances.flatten(), expected_covariances, rtol=1e-06, atol=0
    )


def test_smooth():
    Z = data

    # ----- kalman settings -----
    A = np.expand_dims(np.ones((len(Z), 1)), axis=1)  # transition matrix
    xk = np.array([[1]])  # initial mean estimate

    Pk = np.array([[1]])  # initial covariance estimate
    Q = np.ones((len(Z))) * 0.005  # process noise covariance

    H = A.copy()  # observation matrix
    R = np.ones((len(Z))) * 0.01  # measurement noise covariance
    kf = KalmanFilter(A=A, xk=xk, B=None, Pk=Pk, H=H, Q=Q, R=R)

    expected_means = np.array(
        [
            0.42001,
            0.43213,
            0.41032,
            0.35367,
            0.32886,
            0.34347,
            0.36981,
            0.41107,
            0.41786,
            0.42857,
        ]
    )
    expected_cov = np.array(
        [
            0.00498,
            0.00374,
            0.00344,
            0.00336,
            0.00334,
            0.00334,
            0.00336,
            0.00344,
            0.00375,
            0.005,
        ]
    )

    obtained_means, obtained_cov = kf.smooth(Z, U=None)

    np.testing.assert_almost_equal(
        expected_means, np.stack(obtained_means).flatten(), decimal=5
    )

    np.testing.assert_almost_equal(
        expected_cov, np.stack(obtained_cov).flatten(), decimal=5
    )


def test_smooth_against_pykalman():
    Z = data

    # ----- kalman settings -----
    A = np.expand_dims(np.ones((len(Z), 1)), axis=1)  # transition matrix
    xk = np.array([[1]])  # initial mean estimate

    Pk = np.array([[1]])  # initial covariance estimate
    Q = np.ones((len(Z))) * 0.005  # process noise covariance

    H = A.copy()  # observation matrix
    R = np.ones((len(Z))) * 0.01  # measurement noise covariance

    _Q = np.array([[Q[0]]])
    _R = np.array([[R[0]]])
    pkf = PyKalmanFilter(
        transition_matrices=A[0],
        observation_matrices=H,
        transition_covariance=_Q,
        observation_covariance=_R,
        initial_state_mean=xk,
        initial_state_covariance=Pk,
    )
    kf = KalmanFilter(A=A, xk=xk, B=None, Pk=Pk, H=H, Q=Q, R=R)

    expected_means, expected_cov = pkf.smooth(Z)
    obtained_means, obtained_cov = kf.smooth(Z, U=None)

    np.testing.assert_almost_equal(
        expected_means.flatten(), np.stack(obtained_means).flatten(), decimal=5
    )

    np.testing.assert_almost_equal(
        expected_cov.flatten(), np.stack(obtained_cov).flatten(), decimal=5
    )


def test_filter_and_smooth_against_pykalman_multidimensional():
    Z = np.array(
        [
            [1.2, -0.3],
            [0.7, 0.1],
            [1.1, 0.2],
            [0.9, 0.4],
            [1.3, 0.6],
        ]
    )
    transition_matrices = np.array(
        [
            [[1.0, 0.2], [0.0, 0.9]],
            [[0.8, -0.1], [0.1, 1.1]],
            [[1.05, 0.05], [-0.05, 0.95]],
            [[0.9, 0.3], [0.0, 1.0]],
        ]
    )
    transition_covariance = np.array([[0.03, 0.01], [0.01, 0.04]])
    transition_offsets = np.array(
        [
            [0.1, -0.05],
            [0.0, 0.02],
            [-0.03, 0.04],
            [0.05, 0.01],
        ]
    )
    H = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.1], [0.2, 0.9]],
            [[0.9, -0.2], [0.1, 1.0]],
            [[1.1, 0.0], [0.0, 0.8]],
            [[1.0, 0.05], [-0.1, 1.0]],
        ]
    )
    observation_covariance = np.array([[0.2, 0.0], [0.0, 0.15]])
    xk = np.array([0.4, -0.2])
    Pk = np.array([[0.7, 0.1], [0.1, 0.5]])

    # kalmankit indexes A/Q/B/U by observation time, so index 0 is the
    # transition into the first observation and the pykalman transitions occupy
    # indexes 1..n-1.
    A = np.concatenate([np.eye(2)[None], transition_matrices])
    Q = np.concatenate(
        [np.zeros((1, 2, 2)), np.tile(transition_covariance, (len(Z) - 1, 1, 1))]
    )
    R = np.tile(observation_covariance, (len(Z), 1, 1))
    B = np.tile(np.eye(2), (len(Z), 1, 1))
    U = np.vstack([np.zeros(2), transition_offsets])

    pkf = PyKalmanFilter(
        transition_matrices=transition_matrices,
        observation_matrices=H,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        transition_offsets=transition_offsets,
        initial_state_mean=xk,
        initial_state_covariance=Pk,
    )
    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)

    expected_states, expected_covariances = pkf.filter(Z)
    obtained_states, obtained_covariances = kf.filter(Z=Z, U=U)

    np.testing.assert_allclose(obtained_states, expected_states, rtol=1e-10)
    np.testing.assert_allclose(obtained_covariances, expected_covariances, rtol=1e-10)

    expected_states, expected_covariances = pkf.smooth(Z)
    obtained_states, obtained_covariances = kf.smooth(Z=Z, U=U)

    np.testing.assert_allclose(obtained_states, expected_states, rtol=1e-10)
    np.testing.assert_allclose(obtained_covariances, expected_covariances, rtol=1e-10)
