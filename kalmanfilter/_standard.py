""" Vanilla implementation of Kalman Filter """
import numpy as np


class KalmanFilter:
    """Applies the Kalman filter algorithm.

    The standard Kalman Filter uses a form of feedback-control loop of two
    stages to model dynamic linear systems. 
    
    For each time step :math:`k`

    1. Predict step

    .. math::

        \hat{x}_{k}^{-} = A \hat{x}_{k-1}^{-} + B u_{k-1}

    .. math::
        P_{k}^{-} = AP_{k-1}A^{T} + Q

    2. Update step

    .. math::

        K_k = P_{k}^{-} H^{T} (H P_{k}^{-} H^{T} + R)^{-1}

    .. math::

        \hat{x_{k}} = \hat{x}_{k}^{-} + K_k (z_k - H \hat{x}_{k}^{-})
        
    .. math::
        P_k = (I - K_k H) P_{k}^{-}

    See 2nd and 3rd references to understand notation.
    
    Parameters
    ----------
    A : numpy.array
        Transition matrix. A matrix that relates the state at the previous time 
        step k-1 to the state at the current step k.
    xk : numpy.array
        Initial (k=0) mean estimate.
    B : numpy.array
        Control-input matrix.
    Pk : numpy.array
        Initial (k=0) covariance estimate.
    H : numpy.array
        Observation matrix. A matrix that relates the state to the measurement 
        zk.
    Q : numpy.array or float.
        Process noise covariance.
    R : numpy.array or float.
        Measurement noise covariance.

    Attributes
    ----------
    state_size : int
        Dimensionality of the state (n).
    I : numpy.array
        Identity matrix (n x n).
        
    Returns
    -------
    states : numpy.array
        A posteriori state estimates for each time step.
    errors : numpy.array
        A posteriori estimate error covariances for each time step.
        
    References
    ----------
    .. [1] Matlab - Understanding Kalman Filters:
       https://www.youtube.com/playlist?list=PLn8PRpmsu08pzi6EMiYnR-076Mh-q3tWr
    
    .. [2] Bilgin's Blog - Kalman filter for dummies.
       http://bilgin.esme.org/BitsAndBytes/KalmanFilterforDummies
       
    .. [3] Greg Welch, Gary Bishop - An Introduction to the Kalman Filter:
       https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
       
    .. [4] Tucker McClure - How Kalman Filters Work, Part 1.
       http://www.anuncommonlab.com/articles/how-kalman-filters-work/

    .. [5] Matthew B. Rhudy, Roger A. Salguero and Keaton Holappa - A Kalman 
       Filtering Tutorial for Undergraduate students.
       https://aircconline.com/ijcses/V8N1/8117ijcses01.pdf
    
    """
    def __init__(self, A, xk, B, Pk, H, Q, R):
        self.A = A
        self.xk = xk
        self.B = B
        self.Pk = Pk
        self.H = H
        self.Q = Q
        self.R = R   

        # attributes
        self.state_size = self.xk.shape[0] # usually called 'n'
        self.I = np.identity(self.state_size)

    def predict(self, uk):
        """Predicts states and covariances.
        
        Predict step of the Kalman filter. Computes the prior values of state 
        and covariance using the previous timestep (if any).

        Parameters
        ----------
        uk : numpy.array
            Control-input vector at time k.
        """        
        # project state ahead
        self.xk = np.matmul(self.A, self.xk) + np.matmul(self.B, uk)
        
        # project error covariance ahead
        self.Pk = np.matmul(self.A, np.matmul(self.Pk, self.A.T) + self.Q)
    
    def update(self, zk):
        """Updates states and covariances.
        
        Update step of the Kalman filter. That is, the filter combines the 
        predictions with the observed variable Z at time k.
        
        Parameters
        ----------
        zk : numpy.array
            Observed variable at time k.
        """
        # innovation (pre-fit residual) covariance
        Sk = np.matmul(self.H, np.matmul(self.Pk, self.H.T)) + self.R
        
        # optimal kalman gain
        Kk = np.matmul(self.Pk, np.matmul(self.H.T, np.linalg.inv(Sk)))
        
        # update estimate via zk
        self.xk = self.xk + np.matmul(Kk, zk - np.matmul(self.H, self.xk))
        
        # update error covariance
        self.Pk = np.matmul(self.I - np.matmul(Kk, self.H), self.Pk)
    
    def run_filter(self, Z, U):
        """Runs filter over Z.
        
        Applies the filtering process over Z and returns all errors and 
        covariances. That is: given Z, this functions apply the predict and
        update feedback loop for each zk, where k is a timestamp.
        
        Parameters
        ----------
        Z : numpy.array
            Observed variable
        U : numpy.array
            Control-input vector.
        
        Returns
        -------
        states : numpy.array
            A posteriori state estimates for each time step k.
        errors : numpy.array
            A posteriori estimate error covariances for each time step k. 
        """
        states = np.zeros_like(Z)
        errors = np.zeros_like(Z)
        
        for k, (zk, uk) in enumerate(zip(Z, U)):
            self.predict(uk)
            self.update(zk)
            
            states[k] = self.xk
            errors[k] = self.Pk
            
        return states, errors
