""" Implementation of the extended Kalman filter algorithm"""
import numpy as np
from typing import Tuple, List, Callable


class ExtendedKalmanFilter:
    r"""Extended Kalman filter algorithm.

    The Extended Kalman Filter uses a form of feedback-control loop of two
    stages to model dynamic non-linear systems of the form:

    .. math::

        x_{k} = f(x_{k-1}, u_{k-1}) + q_{k}

    .. math::

        z_{k} = h(x_{k}) + r_{k}

    with :math:`q_{k} \sim \mathcal{N}(0, Q)`, :math:`r_{k} \sim \mathcal{N}(0, R)`
    and :math:`f` and :math:`h` being nonlinear functions.

    For each time step :math:`k`

    1. Predict step

    .. math::

        \hat{x}_{k}^{-} = f(\hat{x}_{k-1}^{-}, u_{k-1})

    .. math::
        P_{k}^{-} = A_{k}P_{k-1}A_{k}^{T} + Q_{k}

    with :math:`A_{k} = \frac{\partial f}{x}` evaluated at :math:`\hat{x}_{k-1}^{-}`.

    2. Update step

    .. math::

        K_k = P_{k}^{-} H_{k}^{T} (H_{k} P_{k}^{-} H_{k}^{T} + R_{k})^{-1}

    .. math::

        \hat{x}_{k} = \hat{x}_{k}^{-} + K_k (z_k - h(\hat{x}_{k}^{-}))

    .. math::
        P_k = (I - K_k H) P_{k}^{-}

    with :math:`H_{k} = \frac{\partial h}{x}` evaluated at :math:`\hat{x}_{k}^{-}`.

    The Extended Kalman Filter can deal with non-linear systems but it is not
    optimal, unless the system is linear and the noises are drawn from a normal
    distribution, which will cause the Extended Kalman Filter to behave as a
    standard Kalman Filter.

    Parameters
    ----------
    xk : numpy.ndarray
        Initial (:math:`k=0`) mean estimate.
    Pk : numpy.ndarray
        Initial (:math:`k=0`) covariance estimate.
    Q : numpy.ndarray
        Process noise covariance (transition covariance).
    R : numpy.ndarray or float.
        Measurement noise covariance (observation covariance).
    f : function
        Non-linear state transition function. It is a nonlinear function that
        relates the state at the previous time step :math:`k-1` to state at the
        current step :math:`k`. It must receive two arguments: :math:`xk` and
        :math:`uk`, which are the state and the control-input.
    h : function
        Non-linear observation function. It is a nonlinear function that
        relates the prior state :math:`xk` to the measurement :math:`zk`. It
        must receive one argument: :math:`xk`.
    jacobian_A : function
        The function that computes the jacobian of :math:`f` with respect to
        :math:`x`. It must receive two arguments: :math:`x` and :math:`u`.
    jacobian_H : function
        The function that computes the jacobian of :math:`h` with respect to
        :math:`x`. It must receive one argument: :math:`x`.

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

    .. [2] Greg Welch, Gary Bishop - An Introduction to the Kalman Filter:
       https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

    .. [3] Tucker McClure - How Kalman Filters Work, Part 1.
       http://www.anuncommonlab.com/articles/how-kalman-filters-work/

    .. [4] Matthew B. Rhudy, Roger A. Salguero and Keaton Holappa - A Kalman
       Filtering Tutorial for Undergraduate students.
       https://aircconline.com/ijcses/V8N1/8117ijcses01.pdf

    Warning
    -------
    Please, take your time to ensure the matrices shapes are correct or the
    filter will not work properly.
    """

    def __init__(
        self,
        xk: np.ndarray,
        Pk: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        f: Callable,
        h: Callable,
        jacobian_A: Callable,
        jacobian_H: Callable,
    ):
        self.f = f
        self.jacobian_A = jacobian_A
        self.xk = xk
        self.Pk = Pk
        self.h = h
        self.jacobian_H = jacobian_H
        self.Q = Q
        self.R = R

        # attributes
        self.state_size = self.xk.shape[0]  # usually called 'n'
        self.__I = np.identity(self.state_size)
        self.kalman_gains = []

    def predict(
        self, xk: np.ndarray, uk: np.ndarray, Pk: np.ndarray, Qk: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts states and covariances.

        Predict step of the Kalman filter. Computes the prior values of state
        and covariance using the previous timestep (if any).

        Parameters
        ----------
        xk : numpy.ndarray
            Mean estimate at time :math:`k`.
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
        # jacobian of f with respect to x evaluated at xk
        Ak = self.jacobian_A(xk, uk)

        # project state ahead
        xk_prior = self.f(xk, uk)

        # project error covariance ahead
        Pk_prior = Ak @ ((Pk @ Ak.T) + Qk)

        return xk_prior, Pk_prior

    def update(
        self, xk: np.ndarray, Pk: np.ndarray, zk: np.ndarray, Rk: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Updates states and covariances.

        Update step of the Kalman filter. That is, the filter combines the
        predictions with the observed variable :math:`Z` at time :math:`k`.

        Parameters
        ----------
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
        # jacobian of h with respect to x evaluated at xk
        Hk = self.jacobian_H(xk)

        # innovation (pre-fit residual) covariance
        Sk = Hk @ (Pk @ Hk.T) + Rk

        # optimal kalman gain
        Kk = Pk @ (Hk.T @ np.linalg.inv(Sk))
        self.kalman_gains.append(Kk)

        # update estimate via zk
        xk_posterior = xk + Kk @ (zk - self.h(xk))

        # update error covariance
        Pk_posterior = (self.__I - Kk @ Hk) @ Pk

        return xk_posterior, Pk_posterior

    def filter(
        self, Z: np.ndarray, U: np.ndarray
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
        U : numpy.ndarray
            Control-input vector.

        Returns
        -------
        states : list of numpy.ndarray
            A posteriori state estimates for each time step :math:`k`.
        errors : list of numpy.ndarray
            A posteriori estimate error covariances for each time step
            :math:`k`.
        """
        states = []  # mean
        errors = []  # covariance

        # get initial conditions
        xk = self.xk
        Pk = self.Pk

        # feedback-control loop
        _iterable = zip(U, Z, self.Q, self.R)
        for k, (uk, zk, Qk, Rk) in enumerate(_iterable):
            # predict step, get prior estimates
            xk_prior, Pk_prior = self.predict(xk=xk, uk=uk, Pk=Pk, Qk=Qk)

            # update step, correct prior estimates
            xk_posterior, Pk_posterior = self.update(
                xk=xk_prior, Pk=Pk_prior, zk=zk, Rk=Rk
            )

            states.append(xk_posterior)
            errors.append(Pk_posterior)

            # update estimates for the next iteration
            xk = xk_posterior
            Pk = Pk_posterior

        return states, errors
