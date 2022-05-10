import numpy as np
from kalmankit import KalmanFilter
from pykalman import KalmanFilter as PyKalmanFilter


data = np.array([0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45])

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
            0.38712871
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
            0.00990099
        ]
    )

    # set parameters
    Z = data
    A = np.expand_dims(np.ones((len(Z),1)), axis=1)
    xk = np.array([[0]])

    B = np.expand_dims(np.zeros((len(Z),1)), axis=1)
    U = np.zeros((len(Z), 1))

    Pk = np.array([[1]])
    H = np.expand_dims(np.ones((len(Z),1)), axis=1)
    Q = np.zeros((len(Z)))
    R = np.ones((len(Z))) * 0.1

    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
    states, covariances = kf.filter(Z=Z, U=U)

    states = np.stack([val.item() for val in states])
    covariances = np.stack([val.item() for val in covariances])

    np.testing.assert_allclose(
        states, 
        expected_states, 
        rtol=1e-06, 
        atol=0
    )

    np.testing.assert_allclose(
        covariances, 
        expected_covariances, 
        rtol=1e-06, 
        atol=0
    )


def test_filter_against_pykalman():
    Z = data

    # ----- kalman settings -----
    A = np.expand_dims(np.ones((len(Z),1)), axis=1)  # transition matrix
    xk = np.array([[1]])  # initial mean estimate

    Pk = np.array([[1]])  # initial covariance estimate
    Q = np.ones((len(Z))) * 0.005  # process noise covariance

    H = A.copy()  # observation matrix
    R = np.ones((len(Z))) * 0.01  # measurement noise covariance

    pkf = PyKalmanFilter(
        transition_matrices=A[0],
        observation_matrices=H,
        transition_covariance=np.array([[Q[0]]]),
        observation_covariance=np.array([[R[0]]]),
        initial_state_mean=xk,
        initial_state_covariance=Pk,
    )
    kf = KalmanFilter(A=A, xk=xk, B=None, Pk=Pk, H=H, Q=Q, R=R)

    expected_means, expected_cov = pkf.filter(Z)
    obtained_means, obtained_cov = kf.filter(Z, U=None)

    np.testing.assert_almost_equal(
        expected_means.flatten(),
        np.stack(obtained_means).flatten(),
        decimal=5
    )

    np.testing.assert_almost_equal(
        expected_cov.flatten(),
        np.stack(obtained_cov).flatten(),
        decimal=5
    )


def test_smooth():
    Z = data

    # ----- kalman settings -----
    A = np.expand_dims(np.ones((len(Z),1)), axis=1)  # transition matrix
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
            0.42857
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
            0.005
        ]
    )

    obtained_means, obtained_cov = kf.smooth(Z, U=None)

    np.testing.assert_almost_equal(
        expected_means,
        np.stack(obtained_means).flatten(),
        decimal=5
    )

    np.testing.assert_almost_equal(
        expected_cov,
        np.stack(obtained_cov).flatten(),
        decimal=5
    )


def test_smooth_against_pykalman():
    Z = data

    # ----- kalman settings -----
    A = np.expand_dims(np.ones((len(Z),1)), axis=1)  # transition matrix
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
        expected_means.flatten(),
        np.stack(obtained_means).flatten(),
        decimal=5
    )

    np.testing.assert_almost_equal(
        expected_cov.flatten(),
        np.stack(obtained_cov).flatten(),
        decimal=5
    )
