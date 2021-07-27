""" Vanilla implementation of the standard Kalman filter algorithm"""
import numpy as np


class KalmanFilter:
    """Applies the Kalman filter algorithm.

    The standard Kalman Filter uses a form of feedback-control loop of two
    stages to model dynamic linear systems of the form:

    .. math::

        x_{k} = A x_{k} + B u_{k} + q_{k}

    .. math::
        
        z_{k} = H x_{k} + r_{k}

    with :math:`q_{k} \sim \mathcal{N}(0, Q)` and :math:`r_{k} \sim \mathcal{N}(0, R)`
    
    For each time step :math:`k`

    1. Predict step

    .. math::

        \hat{x}_{k}^{-} = A_{k} \hat{x}_{k-1}^{-} + B_{k} u_{k-1}

    .. math::
        P_{k}^{-} = A_{k}P_{k-1}A_{k}^{T} + Q_{k}

    2. Update step

    .. math::

        K_k = P_{k}^{-} H_{k}^{T} (H_{k} P_{k}^{-} H_{k}^{T} + R_{k})^{-1}

    .. math::

        \hat{x}_{k} = \hat{x}_{k}^{-} + K_k (z_k - H_{k} \hat{x}_{k}^{-})
        
    .. math::
        P_k = (I - K_k H) P_{k}^{-}

    See 2nd and 3rd references to understand notation.
    
    Parameters
    ----------
    A : numpy.ndarray
        Transition matrix. A matrix that relates the state at the previous time 
        step k-1 to the state at the current step k.
    xk : numpy.ndarray
        Initial (k=0) mean estimate.
    B : numpy.ndarray
        Control-input matrix.
    Pk : numpy.ndarray
        Initial (k=0) covariance estimate.
    H : numpy.ndarray
        Observation matrix. A matrix that relates the state to the measurement 
        zk.
    Q : numpy.ndarray
        Process noise covariance.
    R : numpy.ndarray or float.
        Measurement noise covariance.

    Attributes
    ----------
    state_size : int
        Dimensionality of the state (n).
    I : numpy.ndarray
        Identity matrix (n x n). This attribute is not accessible by the user.
        
    Returns
    -------
    states : numpy.ndarray
        A posteriori state estimates for each time step.
    errors : numpy.ndarray
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
        self.__I = np.identity(self.state_size)

    def predict(self, Ak, xk, Bk, uk, Pk, Qk):
        """
        Parameters
        ----------
        Ak : numpy.ndarray
            Transition matrix at time :math:`k`.
        xk : numpy.ndarray
            Mean estimate at time :math:`k`.
        Bk : numpy.ndarray
            Control-input matrix at time :math:`k`.
        uk : numpy.ndarray
            Control-input vector at time :math:`k`.
        Pk : numpy.ndarray
            Covariance estimate at time :math:`k`.
        Qk : numpy.ndarray
            Process noise covariance at time :math:`k`.
        
        Return
        ------
        xk_prior : numpy.ndarray
            Prior value of state mean.
        Pk_prior : numpy.ndarray
            Prior value of state covariance.
        """
        # project state ahead
        xk_prior = Ak @ xk + Bk @ uk
        
        # project error covariance ahead
        Pk_prior = Ak @ ((Pk @ Ak.T) + Qk)
        
        return xk_prior, Pk_prior
    
    def update(self, Hk, xk, Pk, zk, Rk):
        """
        Parameters
        ----------
        Hk : numpy.ndarray
            Observation matrix at time :math:`k`.
        xk : numpy.ndarray
            Prior mean state estimate at time :math:`k`.
        Pk : numpy.ndarray
            Prior covariance state estimate at time :math:`k`.
        zk : numpy.ndarray
            Observation at time :math:`k`.
        Rk : numpy.ndarray
            Measurement noise covariance at time :math:`k`.
            
        Returns
        -------
        xk_posterior : numpy.ndarray
            A posteriori estimate error mean at time :math:`k`.
        Pk_posterior : numpy.ndarray
            A posteriori estimate error covariance at time :math:`k`.
        """
        # innovation (pre-fit residual) covariance
        Sk = Hk @ (Pk @ Hk.T) + Rk
        
        # optimal kalman gain
        Kk = Pk @ (Hk.T @ np.linalg.inv(Sk))
        
        # update estimate via zk
        xk_posterior = xk + Kk @ (zk - Hk @ xk)
        
        # update error covariance
        Pk_posterior = (self.__I - Kk @ Hk) @ Pk
        
        return xk_posterior, Pk_posterior
    
    def run_filter(self, Z, U):
        """Run filter over Z and U.
        
        Applies the filtering process over Z and U and returns all errors and 
        covariances. That is: given Z and U, this function apply the predict 
        and update feedback loop for each zk, where k is a timestamp.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Observed variable
        U : numpy.ndarray
            Control-input vector.
            
        Returns
        -------
        states : numpy.ndarray
            A posteriori state estimates for each time step k.
        errors : numpy.ndarray
            A posteriori estimate error covariances for each time step k. 
        """
        states = np.zeros_like(Z)
        errors = np.zeros_like(Z)
        
        # get initial conditions
        xk = self.xk
        Pk = self.Pk
        
        iterable = zip(self.A, self.H, self.B, U, Z, self.Q, self.R)

        for k, (Ak, Hk, Bk, uk, zk, Qk, Rk) in enumerate(iterable):
            # predict step
            xk_prior, Pk_prior = self.predict(
                Ak=Ak,
                xk=xk,
                Bk=Bk,
                uk=uk,
                Pk=Pk,
                Qk=Qk
            )
            
            # update step, correct prior estimates
            xk_posterior, Pk_posterior = self.update(
                Hk=Hk,
                xk=xk_prior,
                Pk=Pk_prior,
                zk=zk,
                Rk=Rk
            )

            states[k] = xk_posterior
            errors[k] = Pk_posterior

            # update estimates for the next iteration
            xk = xk_posterior.copy()
            Pk = Pk_posterior.copy()
            
        return states, errors