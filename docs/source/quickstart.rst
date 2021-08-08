===========
Quick Start
===========

Welcome to `kalmanfilter` docs. This library implements a multidimensional
Kalman Filter using NumPy.

The current version of the library only supports the standard Kalman Filter. To
use it, one has to import the library and configure correctly the filter 
parameters. Make sure the parameters configuration is correct or the filter 
will not work.

Here, we are going to define a Kalman Filter to act as a weighted moving 
average.

For this example we will create a noisy sine wave function to generate noisy 
observations:

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

>>> Z = generate_func(start=-10, end=10, step=0.1, beta=0.02, var=0.3)

Now we can define our filter and its parameters. The :math:`A` matrix 
represents the transition matrix; it is the state-transition model we use to
relate the state at time :math:`k-1` with the state at time :math:`k`. Because
we want to model a martingale, the next state should be set to be the current 
state plus some noise.

>>> A = np.expand_dims(np.ones((len(Z),1)), axis=1) # transition matrix

Notice how we expanded the dimensions of the matrix to get a 1x1 matrix in each
time step

The :math:`B` matrix represents the control-input model. We will not use any
of the control settings of the filter, so we are going to set them as zero.

>>> B = np.expand_dims(np.zeros((len(Z),1)), axis=1) # control-input matrix
>>> U = np.zeros((len(Z), 1)) # control-input vector

The last model we have to define is the observation model, which maps the true 
state space into the observed space. Because we want a 1 to 1 relationship, the
model will be equal to the transition matrix.

>>> H = A.copy() # observation matrix

We also have to set the initial mean and covariance estimates so the loop can
start somewhere.

>>> xk = np.array([[1]]) # initial mean estimate
>>> Pk = np.array([[1]]) # initial covariance estimate

It is not crucial to find optimal values for the initial estimates since the 
Kalman filter will quickly converge, but it helps to the process.

Finally, we need to define the noise covariances. The :math:`Q` parameter is 
the process noise covariance. Remember that :math:`q_{k} \sim \mathcal{N}(0, Q)`.
Similarly, the :math:`R` is the measurement noise covariance, being that noise
:math:`r_{k} \sim \mathcal{N}(0, R)`.

>>> Q = np.ones((len(Z))) * 0.005 # process noise covariance
>>> R = np.ones((len(Z))) * 0.01 # measurement noise covariance

Although the Kalman Filter can handle time-varying noise, in practice we use 
the covariance mean value since is almost impossible to have a value at each
time step.

Once everything is defined, we just have to instantiate the filter and pass all
the parameters

>>> kf = KalmanFilter(A=A, xk=xk, B=B, Pk=Pk, H=H, Q=Q, R=R)

Then, we can call `filter` to filter the signal and the object will take 
care of the feedback-control loop.

>>> states, errors = kf.filter(Z=Z, U=U)

You can use Matplotlib to visualize the results, though you may need to adapt 
the shapes.

>>> plt.figure(figsize=(15,7))
>>> plt.plot(Z)
>>> plt.plot(states)
>>> plt.show()