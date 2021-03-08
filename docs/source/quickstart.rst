===========
Quick Start
===========

Usage
#####
The usage is pretty simple given that parameters are correct:

.. code-block:: python
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from kalmanfilter import KalmanFilter


    def generate_func(start, end, step, beta, var):
        """
        Generate a noisy sine wave function
        
        Parameters
        ----------
        start : int
            Initial X value.
        end : int
            Final X value.
        step : float or int
            Space between values.
        beta : float or int
            Slope of the sine wave.
        var : float or int
            Noise variance.
            
        Returns
        -------
        Z : numpy.array
            Noisy sine wave.
        """
        _space = np.arange(start=-10, stop=10, step=0.1)
        _sin = np.sin(_space)
    
        return np.array(
        [beta*x + var*np.random.randn() for x in range(len(_space))]
        ) + _sin

    # set the parameters
    Z = generate_func(start=-10, end=10, step=0.1, beta=0.02, var=0.3)
    A = np.array([[1]])
    xk = np.array([[1]])

    B = np.array([[0]])
    U = np.zeros((len(Z), 1))

    Pk = np.array([[1]])
    Q = 0.005

    H = np.array([[1]])
    R = 0.01

    # apply kalman filter
    kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
    states, errors = kf.run_filter(Z, U)