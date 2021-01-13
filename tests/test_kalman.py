import numpy as np 
from kalmanfilter import KalmanFilter


data = np.array([0.39, 0.50, 0.48, 0.29, 0.25, 0.32, 0.34, 0.48, 0.41, 0.45])

def test_kalman():
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
    A = np.array([[1]])
    xk = np.array([[0]])

    B = np.array([[0]])
    U = np.zeros((len(Z), 1))

    Pk = np.array([[1]])
    H = np.array([[1]])
    Q = 0
    R = 0.1

    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
    states, covariances = kf.run_filter(Z, U)

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