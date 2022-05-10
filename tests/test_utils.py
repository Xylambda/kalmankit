import numpy as np
from kalmankit.utils import check_none_and_broadcast, is_nan


def test_is_nan():
    # TODO: rewrite using Fixtures
    array_1 = np.array([1])
    array_2 = np.array([np.nan])
    array_3 = np.array([np.nan, 2])
    array_4 = np.zeros((2,2)) # multidimensional check

    msg = f"Incorrect evaluation of nan in array: {array_1}"
    assert not is_nan(array_1), msg

    msg = f"Incorrect evaluation of nan in array: {array_2}"
    assert is_nan(array_2), msg

    msg = f"Incorrect evaluation of nan in array: {array_3}"
    assert not is_nan(array_3), msg

    msg = f"Incorrect evaluation of nan in array: {array_4}"
    assert not is_nan(array_4), msg