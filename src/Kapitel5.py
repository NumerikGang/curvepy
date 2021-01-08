import numpy as np
import scipy.special as scs
from src.utilities import csv_read
from typing import Tuple
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


def bezier_to_power(m: np.ndarray) -> np.ndarray:
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
    return np.ndarray([])


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


def check_flat(m: np.ndarray, tol: float = 1) -> bool:
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


def init() -> None:
    test = csv_read("test.csv")
    print(test)
    print(test[:, ::-1])
    #print(check_flat(test))
    print(horn_bez(test))
    #print(differences(test))
    print(horner(test, 2))


if __name__ == "__main__":
    init()

