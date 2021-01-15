import numpy as np
import scipy.special as scs
import sympy as sy
import sys as s
from src.utilities import csv_read
from typing import Tuple, Callable, Union
from functools import reduce


def horn_bez(m: np.ndarray, t: float = 0.5) -> np.ndarray:
    """
    Method using horner like scheme to calculate point on curve with given t

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    t: float:
        value for which point is calculated

    Returns
    -------
    np.ndarray:
        point calculated with given t
    """
    n = m.shape[1] - 1  # need degree of curve (n points means degree = n-1)
    res = m[:, 0] * (1 - t)
    for i in range(1, n):
        res = (res + t**i * scs.binom(n, i) * m[:, i]) * (1 - t)

    res += t**n * m[:, n]

    return res


def bezier_to_power(m: np.ndarray) -> Callable:
    """
    Method calculating monomial representation of given bezier form using 5.27

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    Returns
    -------
    np.ndarray:
        array for monomial representation
    """
    _, n = m.shape
    diff = differences(m)
    t = sy.symbols('t')
    res = 0
    for i in range(n):
        res += scs.binom(n-1, i) * diff[:, i] * t**i

    return sy.lambdify(t, res)


def differences(m: np.ndarray, i: int = 0) -> np.ndarray:
    """
    Method using equation 5.23 to calculate all forward differences for a given point i

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    i: int:
        point i for which forward differences are calculated

    Returns
    -------
    np.ndarray:
         array holds all forward differences for given point i
    """
    _, n = m.shape
    diff = [np.sum([scs.binom(r, j)*(-1)**(r - j)*m[:, i + j] for j in range(0, r+1)], axis=0) for r in range(0, n)]
    return np.array(diff).T


def horner(m: np.ndarray, t: float = 0.5) -> Tuple:
    """
    Method using horner scheme to calculate point with given t

    Parameters
    ----------
    m: np.ndarray:
        array containing coefficients

    t: float:
        value for which point is calculated

    Returns
    -------
    tuple:
        point calculated with given t
    """
    return reduce(lambda x, y: t*x + y, m[0, ::-1]), reduce(lambda x, y: t*x + y, m[1, ::-1])


def de_caes_in_place(m: np.ndarray, t: float = 0.5) -> np.ndarray:
    """
    Method computing de Casteljau in place

    Parameters
    ----------
    m: np.ndarray:
        array containing coefficients

    t: float:
        value for which point is calculated

    Returns
    -------
    np.ndarray:
        array containing calculated points with given t
    """
    _, n = m.shape
    for r in range(n):
        m[:, :(n - r - 1)] = (1 - t) * m[:, :(n - r - 1)] + t * m[:, 1:(n - r)]
    return m


def subdiv(m: np.ndarray, t: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method using subdivison to calculate right and left polygon with given t using de Casteljau

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
    return de_caes_in_place(m.copy(), t), de_caes_in_place(m.copy()[:, ::-1], 1.0-t)


def distance_to_line(p1: np.ndarray, p2: np.ndarray, p_to_check: np.ndarray) -> float:
    """
    Method calculating distance of point to line through p1 and p2

    Parameters
    ----------
    p1: np.ndarray:
        beginning point of line

    p2: np.ndarray:
        end point of line

    p_to_check: np.ndarray:
        point for which distance is calculated

    Returns
    -------
    float:
        distance from point to line
    """
    numerator = abs((p2[0] - p1[0]) * (p1[1] - p_to_check[1]) - (p1[0] - p_to_check[0]) * (p2[1] - p1[1]))
    denominator = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
    return numerator/denominator


def check_flat(m: np.ndarray, tol: float = s.float_info.epsilon) -> bool:
    """
    Method checking if all points between the first and the last point
     are less than tol away from line through beginning and end point of bezier curve

    Parameters
    ----------
    m: np.ndarray:
        points of curve

    tol: float:
        tolerance for distance check

    Returns
    -------
    bool:
        True if all point are less than tol away from line otherwise false
    """
    return all(distance_to_line(m[:, 0], m[:, len(m[0])-1], m[:, i]) <= tol for i in range(1, len(m[0])-1))


def min_max_box(m: np.ndarray) -> list:
    """
    Method creating the minmaxbox of a given curve

    Parameters
    ----------
    m: np.ndarray:
        points of curve

    Returns
    -------
    list:
        contains the points that describe the minmaxbox
    """
    return [m[0, :].min(), m[0, :].max(), m[1, :].min(), m[1, :].max()]


def intersect_lines(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Union[np.ndarray, None]:
    """
    Method checking if line through p1, p2 intersects with line through p3, p4


    Parameters
    ----------
    p1: np.ndarray:
        first point of first line

    p2: np.ndarray:
        second point of first line

    p3: np.ndarray:
        first point of second line

    p4: np.ndarray:
        second point of second line

    Returns
    -------
    bool:
        True if all point are less than tol away from line otherwise false
    """
    # First we vertical stack the points in an array
    vertical_stack = np.vstack([p1, p2, p3, p4])
    # Then we transform them to homogeneous coordinates, to perform a little trick
    homogeneous = np.hstack((vertical_stack, np.ones((4, 1))))
    # having our points in this form we can get the lines through the cross product
    line_1, line_2 = np.cross(homogeneous[0], homogeneous[1]), np.cross(homogeneous[2], homogeneous[3])
    # when we calculate the cross product of the lines we get intersect point
    x, y, z = np.cross(line_1, line_2)
    if z == 0:
        return None
    # we dividing with z to turn back to 2D space
    return np.array([x/z, y/z])


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
            np.append(res, p)
    else:
        # if poly not flat enough we subdivide and check the resulting polygons for intersection
        p1, p2 = subdiv(m, 0.5)
        np.append(res, intersect(p1, tol))
        np.append(res, intersect(p2, tol))

    return res


def init() -> None:
    test = csv_read("test.csv")
    print(test)
    #print(min_max_box(test))
    #print(np.ndarray([]).size)
    #print(check_flat(test))
    #print(horn_bez(test))
    #print(differences(test))
    #print(horner(test, 2))


if __name__ == "__main__":
    init()

