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

def subdiv(m: np.ndarray, t: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method using subdivision to calculate right and left polygon with given t using de Casteljau

    Parameters
    ----------
    m: np.ndarray:
        array containing coefficients

    t: float:
        value for which point is calculated

    Returns
    -------
    np.ndarray:
        right polygon
    np.ndarray:
        left polygon
    """
    return de_caes(m, t, True), de_caes(m[:, ::-1], 1.0 - t, True)


def intersect(m: np.ndarray, tol: float = s.float_info.epsilon) -> np.ndarray:
    """
    Method checks if curve intersects with x-axis

    Parameters
    ----------
    m: np.ndarray:
        points of curve

    tol: float:
        tolerance for check_flat

    Returns
    -------
    np.ndarray:
        Points where curve and x-axis intersect
    """
    box = min_max_box(m)
    res = np.array([])

    if box[2] * box[3] > 0:
        # Both y values are positive, ergo curve lies above x_axis
        return np.array([])

    if check_flat(m, tol):
        # poly is flat enough, so we can perform intersect of straight lines
        # since we are assuming poly is a straight line we define a line through first and las point of poly
        # additionally we create a line which demonstrates the x axis
        # having these two lines we can check them for intersection
        p = intersect_lines(m[:, 0], m[:, -1], np.array([0, 0]), np.array([1, 0]))
        if p is not None:
            res = np.append(res, p.reshape((2, 1)), axis=1)
    else:
        # if poly not flat enough we subdivide and check the resulting polygons for intersection
        p1, p2 = subdiv(m, 0.5)
        res = np.append(res, intersect(p1, tol).reshape((2, 1)), axis=1)
        res = np.append(res, intersect(p2, tol).reshape((2, 1)), axis=1)

    return res
