""" Vanilla implementation of Kalman Filter """
import numpy as np


class KalmanFilter:
    """Kalman filter.
    
    Applies the Kalman filter algorithm. See 2nd and 3rd references to 
    understand notation.
    
    Parameters
    ----------
    A : numpy.array
        Transition matrix. A matrix that relates the state at the previous time 
        step k-1 to the state at the current step k.
    xk : numpy.array
        Initial (k=0) mean estimate.
    B : numpy.array
        Control-input matrix.
    u : numpy.array
        Control-input vector.
    Pk : numpy.array
        Initial (k=0) covariance estimate.
    H : numpy.array
        Observation matrix. A matrix that relates the state to the measurement 
        zk.
    Q : numpy.array
        Process noise covariance.
    R : numpy.array
        Measurement noise covariance.
        
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
       
    .. [4] Tucker McClure - How Kalman Filters Work, Part 1
       http://www.anuncommonlab.com/articles/how-kalman-filters-work/
    
    """
    def __init__(self, A, xk, B, u, Pk, H, Q, R):
        self.A = A
        self.xk = xk
        self.B = B
        self.u = u
        self.Pk = Pk
        self.H = H
        self.Q = Q
        self.R = R        
    
    def predict(self):
        """Predicts states and covariances.
        
        Predict step of the Kalman filter. Computes the prior values of state 
        and covariance using the previous timestep (if any).
        """        
        # project state ahead
        self.xk = np.matmul(self.A, self.xk) + np.matmul(self.B, self.u)
        
        # project error covariance ahead
        self.Pk = np.matmul(self.A, np.matmul(self.Pk, self.A.T) + self.Q)
    
    def update(self, zk):
        """Updates states and covariances.
        
        Update step of the Kalman filter by combining the predictions with the
        observed variable Z at time k.
        
        Parameters
        ----------
        zk : numpy.array
            Observed variable at time k.
            
        Returns
        -------
        xk : numpy.array
            Updated estimate of the mean at time k.
        Pk : numpy.array
            Updated estimate of the covariance at time k.
        """
        # innovation (pre-fit residual) covariance
        Sk = np.matmul(self.H, np.matmul(self.Pk, self.H.T)) + self.R
        
        # optimal kalman gain
        Kk = np.matmul(self.Pk, np.matmul(self.H.T, np.linalg.inv(Sk)))
        
        # update estimate via zk
        self.xk = self.xk + np.matmul(Kk, zk - np.matmul(self.H, self.xk))
        
        # update error covariance
        _aux = np.matmul(Kk, self.H)
        I = np.identity(_aux.shape[0])
        self.Pk = np.matmul(I - _aux, self.Pk)
        
        return self.xk, self.Pk
    
    def run_filter(self, Z):
        """Runs filter over Z.
        
        Applies the filtering process over Z and returns all errors and 
        covariances. That is: given Z, this functions apply the predict and
        update feedback loop for each zk, where k is a timestamp.
        
        Parameters
        ----------
        Z : numpy.array
            Observed variable
            
        Returns
        -------
        states : numpy.array
            A posteriori state estimates for each time step k.
        errors : numpy.array
            A posteriori estimate error covariances for each time step k. 
        """
        states = np.zeros_like(Z)
        errors = np.zeros_like(Z)
        
        for k, zk in enumerate(Z):
            self.predict()
            xk, Pk = self.update(zk)
            
            states[k] = xk
            errors[k] = Pk
            
        return states, errors