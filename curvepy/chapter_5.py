import sys as s
import numpy as np
import sympy as sy
import scipy.special as scs
from functools import reduce, partial
from typing import Tuple, Callable, Union, List, Any


def intermediate_bezier_points(m: np.ndarray, r: int, i: int, t: float = 0.5,
                               interval: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Method using 5.7 an intermediate point of the bezier curve

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    i: int:
        which intermediate points should be calculated

    t: float:
        value for which Bezier curve are calculated

    r: int:
        optional Parameter to calculate only a partial curve

    interval: Tuple[float,float]:
        Interval of t used for affine transformation

    Returns
    -------
    np.ndarray:
            intermediate point

    Notes
    -----
    Equation used for computing:
    math:: b_i^r(t) = \\sum_{j=0}^r b_{i+j} \\cdot B_i^r(t)
    """
    _, n = m.shape
    t = (t - interval[0]) / (interval[1] - interval[0])
    return np.sum([m[:, i + j] * bernstein_polynomial(n - 1, j, t) for j in range(r)], axis=0)


