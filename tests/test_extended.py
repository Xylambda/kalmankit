import numpy as np
from kalmankit import ExtendedKalmanFilter
from pykalman import KalmanFilter as PyKalmanFilter

data = np.array([0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45])


def test_predict_covariance_matches_formula():
    xk = np.array([0.4, -0.2])
    Pk = np.array([[0.7, 0.1], [0.1, 0.5]])
    Qk = np.array([[0.03, 0.01], [0.01, 0.04]])
    Ak = np.array([[1.0, 0.2], [-0.1, 0.9]])

    def f(x, u=None):
        return Ak @ x

    def h(x):
        return x

    def jacobian_A(x, u=None):
        return Ak

    def jacobian_H(x):
        return np.eye(2)

    ekf = ExtendedKalmanFilter(
        xk=xk,
        Pk=Pk,
        Q=np.array([Qk]),
        R=np.array([np.eye(2)]),
        f=f,
        h=h,
        jacobian_A=jacobian_A,
        jacobian_H=jacobian_H,
    )

    _, obtained = ekf.predict(xk=xk, uk=None, Pk=Pk, Qk=Qk)
    expected = Ak @ Pk @ Ak.T + Qk

    np.testing.assert_allclose(obtained, expected)


def test_extended_equal_to_standard():
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
    U = np.zeros((len(Z), 1))

    Pk = np.array([[1]])
    xk = np.array([[0]])

    Q = np.zeros((len(Z)))
    R = np.ones((len(Z))) * 0.1

    # define nonlinear functions
    def f(xk, uk):
        return xk + uk

    def h(xk):
        return xk

    # define jacobian matrices
    def jacobian_A(x, u=None):
        return np.array([[1]])

    def jacobian_H(x):
        return np.array([[1]])

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
    states, covariances = ekf.filter(Z=Z, U=U)

    np.testing.assert_allclose(states.flatten(), expected_states, rtol=1e-06, atol=0)

    np.testing.assert_allclose(
        covariances.flatten(), expected_covariances, rtol=1e-06, atol=0
    )


def test_smooth_against_pykalman_multidimensional():
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
    observation_matrix = np.array([[1.0, 0.2], [0.1, 0.9]])
    observation_covariance = np.array([[0.2, 0.0], [0.0, 0.15]])
    xk = np.array([0.4, -0.2])
    Pk = np.array([[0.7, 0.1], [0.1, 0.5]])

    initial_transition = np.concatenate([np.eye(2).ravel(), np.zeros(2)])
    transition_controls = np.column_stack(
        [transition_matrices.reshape(len(Z) - 1, -1), transition_offsets]
    )
    U = np.vstack([initial_transition, transition_controls])
    Q = np.concatenate(
        [np.zeros((1, 2, 2)), np.tile(transition_covariance, (len(Z) - 1, 1, 1))]
    )
    R = np.tile(observation_covariance, (len(Z), 1, 1))

    def _transition_matrix(uk):
        return uk[:4].reshape(2, 2)

    def _transition_offset(uk):
        return uk[4:]

    def f(x, uk):
        return _transition_matrix(uk) @ x + _transition_offset(uk)

    def h(x):
        return observation_matrix @ x

    def jacobian_A(x, u=None):
        return _transition_matrix(u)

    def jacobian_H(x):
        return observation_matrix

    pkf = PyKalmanFilter(
        transition_matrices=transition_matrices,
        observation_matrices=observation_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        transition_offsets=transition_offsets,
        initial_state_mean=xk,
        initial_state_covariance=Pk,
    )
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

    expected_states, expected_covariances = pkf.smooth(Z)
    obtained_states, obtained_covariances = ekf.smooth(Z=Z, U=U)

    np.testing.assert_allclose(obtained_states, expected_states, rtol=1e-10)
    np.testing.assert_allclose(obtained_covariances, expected_covariances, rtol=1e-10)
