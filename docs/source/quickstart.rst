===========
Quick Start
===========

For this example we are going to create a noisy sine wave function that will
generate noisy observations:

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from kalmanfilter import KalmanFilter
>>> 
>>> 
>>> def generate_func(start, end, step, beta, var):
...     """
...     Generate a noisy sine wave function
...     
...     Parameters
...     ----------
...     start : int
...         Initial X value.
...     end : int
...         Final X value.
...     step : float or int
...         Space between values.
...     beta : float or int
...         Slope of the sine wave.
...     var : float or int
...         Noise variance.
...         
...     Returns
...     -------
...     out : numpy.array
...         Noisy sine wave.
...     """
...     _space = np.arange(start=start, stop=end, step=step)
...     _sin = np.sin(_space)
... 
...     out = np.array(
...         [beta*x + var*np.random.randn() for x in range(len(_space))]
...     )
...     out += _sin
...     return out

Now we can define our filter and its parameters. Check the documentation and 
its references to understand each one of the parameters and make sure you set
them correctly or the filter will not work.

>>> Z = generate_func(start=-10, end=10, step=0.1, beta=0.02, var=0.3)
>>> A = np.expand_dims(np.ones((len(Z),1)), axis=1) # transition matrix
>>> xk = np.array([[1]]) # initial mean estimate
>>> B = np.expand_dims(np.zeros((len(Z),1)), axis=1) # control-input matrix
>>> U = np.zeros((len(Z), 1)) # control-input vector
>>> Pk = np.array([[1]]) # initial covariance estimate
>>> Q = np.ones((len(Z))) * 0.005 # measurement noise covariance
>>> H = A.copy() # observation matrix
>>> R = np.ones((len(Z))) * 0.01 # measurement noise covariance

After that, we can call `run_filter` to filter the signal and library will take 
care of the feedback-control loop.

>>> kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)
>>> states, errors = kf.run_filter(Z=Z, U=U)
>>> plt.figure(figsize=(15,7))
>>> plt.plot(Z)
>>> plt.plot(states)
>>> plt.show()