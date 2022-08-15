""" Multidimensional implementation of the Kalman Filter algorithm. """

from . import utils
from ._version import get_versions
from .extended import ExtendedKalmanFilter
from .standard import KalmanFilter

__all__ = ["KalmanFilter", "ExtendedKalmanFilter", "utils"]

__version__ = get_versions()["version"]
del get_versions
