import numpy as np
from kalmankit.utils import check_none_and_broadcast, is_nan_all


def test_is_nan_all():
    # TODO: rewrite using Fixtures
    array_1 = np.array([1])
    array_2 = np.array([np.nan])
    array_3 = np.array([np.nan, 2])
    array_4 = np.zeros((2,2)) # multidimensional check

    msg = f"Incorrect evaluation of nan in array: {array_1}"
    assert not is_nan_all(array_1), msg

    msg = f"Incorrect evaluation of nan in array: {array_2}"
    assert is_nan_all(array_2), msg

    msg = f"Incorrect evaluation of nan in array: {array_3}"
    assert not is_nan_all(array_3), msg

    msg = f"Incorrect evaluation of nan in array: {array_4}"
    assert not is_nan_all(array_4), msg


def test_check_none_and_broadcast():
    to_broad = np.zeros((2,2))

    obtained = check_none_and_broadcast(None, to_broad)
    _expected = to_broad.copy()
    expected = _expected[:] = np.nan

    np.testing.assert_almost_equal(obtained, expected)