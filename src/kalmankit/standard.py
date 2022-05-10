""" Vanilla implementation of the standard Kalman filter algorithm"""
import numpy as np
from typing import Tuple, List
from kalmankit.utils import check_none_and_broadcast, is_nan

__all__ = ["KalmanFilter"]


class KalmanFilter:
    r"""Standard Kalman filter algorithm.

    The standard Kalman Filter uses a form of feedback-control loop of two
    stages to model dynamic linear systems of the form:

    .. math::

        x_{k} = A x_{k} + B u_{k} + q_{k} \\
        z_{k} = H x_{k} + r_{k}

    with :math:`q_{k} \sim \mathcal{N}(0, Q)` and
    :math:`r_{k} \sim \mathcal{N}(0, R)`

    For each time step :math:`k`

    1. Predict step

    .. math::

        \hat{x}_{k}^{-} = A_{k} \hat{x}_{k-1}^{-} + B_{k} u_{k-1} \\
        P_{k}^{-} = A_{k}P_{k-1}A_{k}^{T} + Q_{k}

    2. Update step

    .. math::

        K_k = P_{k}^{-} H_{k}^{T} (H_{k} P_{k}^{-} H_{k}^{T} + R_{k})^{-1} \\
        \hat{x}_{k} = \hat{x}_{k}^{-} + K_k (z_k - H_{k} \hat{x}_{k}^{-}) \\
        P_k = (I - K_k H) P_{k}^{-}

    See 2nd and 3rd references to understand notation.

    Parameters
    ----------
    A : numpy.ndarray
        Transition matrix. A matrix that relates the state at the previous time
        step :math:`k-1` to the state at the current step :math:`k`.
    xk : numpy.ndarray
        Initial (:math:`k=0`) mean estimate.
    B : numpy.ndarray
        Control-input matrix.
    Pk : numpy.ndarray
        Initial (:math:`k=0`) covariance estimate.
    H : numpy.ndarray
        Observation matrix. A matrix that relates the prior state :math:`xk` to
        the measurement :math:`z_{k}`.
    Q : numpy.ndarray
        Process noise covariance (transition covariance).
    R : numpy.ndarray or float.
        Measurement noise covariance (observation covariance).

    Attributes
    ----------
    state_size : int
        Dimensionality of the state :math:`(n)`.
    I : numpy.ndarray
        Identity matrix :math:`(n x n)`. This attribute is not accessible by
        the user.
    kalman_gains : list of numpy.ndarray
        The list of computed Kalman gains.

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

    Warning
    -------
    Please, take your time to ensure the matrices shapes are correct or the
    filter will not work properly.
    """

    def __init__(
        self,
        A: np.ndarray,
        xk: np.ndarray,
        B: np.ndarray,
        Pk: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        state_size: int = None
    ):
        self.A = A
        self.xk = xk
        self.B = check_none_and_broadcast(B, A)
        self.Pk = Pk
        self.H = H
        self.Q = Q
        self.R = R

        # attributes
        if state_size:
            self.state_size = state_size
        else:
            self.state_size = self.xk.shape[0]
        self.__I = np.identity(self.state_size)
        self.kalman_gains: List[np.ndarray] = []

    def predict(
        self,
        Ak: np.ndarray,
        xk: np.ndarray,
        Bk: np.ndarray,
        uk: np.ndarray,
        Pk: np.ndarray,
        Qk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts states and covariances.

        Predict step of the Kalman filter. Computes the prior values of state
        and covariance using the previous timestep (if any).

        .. math::

            \hat{x}_{k}^{-} = A_{k} \hat{x}_{k-1}^{-} + B_{k} u_{k-1} \\
            P_{k}^{-} = A_{k}P_{k-1}A_{k}^{T} + Q_{k}

        Parameters
        ----------
        Ak : numpy.ndarray
            Transition matrix at time :math:`k`.
        xk : numpy.ndarray
            Mean estimate at time :math:`k`.
        Bk : numpy.ndarray
            Control-input matrix at time :math:`k`. Can be set to None to
            ignore the control part of the filter.
        uk : numpy.ndarray
            Control-input vector at time :math:`k`. Can be set to None to
            ignore the control part of the filter.
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
        if is_nan(Bk) or is_nan(uk):
            xk_prior = Ak @ xk
        else:
            xk_prior = Ak @ xk + Bk @ uk

        # project error covariance ahead
        Pk_prior = Ak @ ((Pk @ Ak.T) + Qk)

        return xk_prior, Pk_prior

    def update(
        self,
        Hk: np.ndarray,
        xk: np.ndarray,
        Pk: np.ndarray,
        zk: np.ndarray,
        Rk: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Updates states and covariances.

        Update step of the Kalman filter. That is, the filter combines the
        predictions with the observed variable :math:`Z` at time :math:`k`.

        .. math::

            K_k = P_{k}^{-} H_{k}^{T} (H_{k} P_{k}^{-} H_{k}^{T}+R_{k})^{-1} \\
            \hat{x}_{k} = \hat{x}_{k}^{-}+K_k (z_k - H_{k} \hat{x}_{k}^{-}) \\
            P_k = (I - K_k H) P_{k}^{-}

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
        Kk = Pk @ (Hk.T @ np.linalg.pinv(Sk))
        self.kalman_gains.append(Kk)

        # update estimate via zk
        xk_posterior = xk + Kk @ (zk - Hk @ xk)

        # update error covariance
        Pk_posterior = (self.__I - Kk @ Hk) @ Pk

        return xk_posterior, Pk_posterior

    def filter(
        self, Z: np.ndarray, U: np.ndarray = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run filter over Z and U.

        Applies the filtering process over :math:`Z` and :math:`U` and returns
        all errors and covariances. That is: given :math:`Z` and :math:`U`,
        this function applies the predict and update feedback loop for each
        :math:`zk`, where :math:`k` is a timestamp.

        Parameters
        ----------
        Z : numpy.ndarray
            Observed variable
        U : numpy.ndarray, optional, default: None
            Control-input vector.

        Returns
        -------
        states : list of numpy.ndarray
            A posteriori state estimates for each time step :math:`k`.
        errors : list of numpy.ndarray
            A posteriori estimate error covariances for each time step
            :math:`k`.
        """
        # mean and covariance array allocation
        states = np.zeros((len(Z), self.state_size))
        errors = np.zeros((len(Z), self.state_size, self.state_size))

        # get initial conditions
        xk = self.xk
        Pk = self.Pk

        # allow U to be None without the filter failing
        U = check_none_and_broadcast(U, Z)

        # feedback-control loop
        _iterable = zip(self.A, self.H, self.B, U, Z, self.Q, self.R)
        for k, (Ak, Hk, Bk, uk, zk, Qk, Rk) in enumerate(_iterable):
            # predict step, get prior estimates
            xk_prior, Pk_prior = self.predict(
                Ak=Ak, xk=xk, Bk=Bk, uk=uk, Pk=Pk, Qk=Qk
            )

            # update step, correct prior estimates
            xk_posterior, Pk_posterior = self.update(
                Hk=Hk, xk=xk_prior, Pk=Pk_prior, zk=zk, Rk=Rk
            )

            states[k] = xk_posterior
            errors[k] = Pk_posterior

            # update estimates for the next iteration
            xk = xk_posterior
            Pk = Pk_posterior

        return states, errors

    def smooth(
        self, Z: np.ndarray, U: np.ndarray = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Rauch-Tung-Strieble (RTS) smoother.

        Parameters
        ----------
        Z : numpy.ndarray
            Observed variable
        U : numpy.ndarray, optional, default: None
            Control-input vector.

        Returns
        -------
        xk_smooth : list of numpy.ndarray
            Smoothed means for each time step :math:`k`.
        Pk_smooth : list of numpy.ndarray
            Smoothed error covariances for each time step :math:`k`.
        """
        # filtering process to get posteriors
        x_est, P_est = self.filter(Z=Z, U=U)

        # mean and covariance array allocation
        xk_smooth = np.zeros((len(Z), self.state_size))
        Pk_smooth = np.zeros((len(Z), self.state_size, self.state_size))

        # smooth initialization
        xk_smooth[-1] = x_est[-1]
        Pk_smooth[-1] = P_est[-1]

        n_obs = len(Z)
        for k in range(n_obs - 2, -1, -1):
            # select appropiate parameters for each time step
            Ak = self.A[k]
            Qk = self.Q[k]

            # predicted mean and covariance
            xk_ahead = Ak @ x_est[k]
            Pk_ahead = Ak @ (P_est[k] @ Ak.T) + Qk

            # smooth (like butter) process
            Kk = P_est[k] @ (Ak.T @ np.linalg.pinv(Pk_ahead))
            xk_smooth[k] = x_est[k] + Kk @ (xk_smooth[k + 1] - xk_ahead)
            Pk_smooth[k] = P_est[k] + Kk @ ((Pk_smooth[k + 1] - Pk_ahead.T) @ Kk)

        return xk_smooth, Pk_smooth
