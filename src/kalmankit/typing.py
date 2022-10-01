"""
Collection of type hints for the library.
"""
from typing import Optional, Tuple, Union

from numpy import bool_, ndarray

OptionalArray = Optional[ndarray]
OptionalArrayOrFloat = Optional[Union[ndarray, float]]
ArrayOrFloat = Union[ndarray, float]
ReturnArrayTuple = Tuple[ndarray, ndarray]
Boolean = Union[bool, bool_]
