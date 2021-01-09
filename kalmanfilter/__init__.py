""" General implementation of Kalman filter algorithm using NumPy. """

# relative subpackages import
from . import _standard


# filters
from ._standard import KalmanFilter

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
