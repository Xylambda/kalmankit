""" Multidimensional implementation of the Kalman Filter algorithm. """

from .standard import KalmanFilter
from .extended import ExtendedKalmanFilter
from ._version import get_versions

__all__ = ["KalmanFilter", "ExtendedKalmanFilter"]

__version__ = get_versions()["version"]
del get_versions
