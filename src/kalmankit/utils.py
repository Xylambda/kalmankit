"""
Convenience and utility functions for the library
"""
import numpy as np

__all__ = ["check_none_and_broadcast"]


def check_none_and_broadcast(
    arr: np.ndarray, broad_to: np.ndarray
) -> np.ndarray:
    """Helper function.

    Check wether "arr" is None and generate a new array with shape equal to
    "broad_to" shape of np.nan values.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to check if is None.
    broad_to : numpy.ndarry
        Array whose shape will be mimicked if "arr" is None.

    Returns
    -------
    arr : np.ndarray
        Array with np.nan values or original array if is not None.
    """
    if arr is None:
        arr = np.empty_like(broad_to)
        arr[:] = np.nan

    return arr


def is_nan(arr: np.ndarray) -> bool:
    """Check if all elements of the given array are nan.

    Parameters
    ----------
    arr : numpy.ndarry
        NumPy array to check.

    Return
    ------
    bool
        Whether all elements are nan (True) or not (False).
    """
    isnan = np.isnan(arr).flatten().sum()
    return isnan == arr.size
