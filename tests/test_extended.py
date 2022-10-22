import numpy as np 
from kalmankit import ExtendedKalmanFilter


data = np.array([0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45])


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

    np.testing.assert_allclose(
        states.flatten(), 
        expected_states, 
        rtol=1e-06, 
        atol=0
    )

    np.testing.assert_allclose(
        covariances.flatten(), 
        expected_covariances, 
        rtol=1e-06, 
        atol=0
    )