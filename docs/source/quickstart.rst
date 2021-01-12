===========
Quick Start
===========

Usage
#####
The usage is pretty simple given that parameters are correct:
.. code-block:: python
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from pathlib import Path
    from kalmanfilter import KalmanFilter


    DATA_PATH = Path(os.path.abspath('')).parent / "tests/data"

    # read data
    ibex = pd.read_pickle(DATA_PATH / "ibex35.pkl")

    # set the parameters
    Z = ibex['Close'].values
    A = np.array([[1]])
    x = np.array([[1]])

    B = np.array([[0]])
    u = np.array([[0]])

    Pk = np.array([[1]])
    Q = 0.005

    H = np.array([[1]])
    R = 0.01

    # apply kalman filter
    kf = KalmanFilter(A=A, xk=x, B=B, u=u, Pk=Pk, H=H, Q=Q, R=R)
    states, errors = kf.run_filter(Z)