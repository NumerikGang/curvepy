"""
This module provides a variety of options to calculate points on a curve defined by Bezier points:

- You can run one step of de Castelljau

- You can run r rounds of de Castelljau for the case you already have precomputed intermediate points or you want to
compute intermediate points by yourself

- You can run n rounds of de Castelljau which returns you the point on the curve

- You can run de Castelljau blossomed which means that in every step of the computation a different parameter value is
taken

- You can run de Castelljau in parallel for multiple parameter values so that you can compute multiple points on the
curve with just one function call

- You can approximate the defined curve by using the subdivision routine

"""
import concurrent.futures
from multiprocessing import cpu_count
from typing import Iterator, List, Tuple

import numpy as np


def de_caes_one_step(m: np.ndarray, t: float = 0.5, interval: Tuple[int, int] = (0, 1),
                     make_copy: bool = True) -> np.ndarray:
    """
    Method computing only one step of de Casteljau

    Parameters
    ----------
     m: np.ndarray:
        array containing coefficients

    t: float:
        value for which point is calculated

    make_copy: bool:
        optional parameter if computation should not be in place

    interval: Tuple[int, int]:
        if interval != (0,1) we need to transform t

    Returns
    -------
    np.ndarray:
        array containing calculated points with given t
    """
    if make_copy:
        m = m.copy()
    l, r = interval
    t2 = (t - l) / (r - l) if interval != (0, 1) else t
    t1 = 1 - t2

    m[:, :-1] = t1 * m[:, :-1] + t2 * m[:, 1:]

    return m[:, :-1] if m.shape != (2, 1) else m


def de_caes_n_steps(m: np.ndarray, t: float = 0.5, r: int = 1, interval: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Method computing r rounds of de Casteljau. So it is possible to start with already precalculated intermediate
    points or to compute intermediate points by yourself.

    Parameters
    ----------
    m: np.ndarray:
        Bezier points

    t: float:
        value for which point is calculated

    r: int:
        amount of rounds to be executed

    interval: Tuple[int, int]:
        if interval != (0,1) we need to transform t

    Returns
    -------
    np.ndarray:
        calculated points with given t
    """

    for _ in range(r):
        m = de_caes_one_step(m, t, interval, make_copy=False)
    return m


def de_caes(m: np.ndarray, t: float = 0.5, make_copy: bool = True, interval: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Method computing n Iterations of de Castelljau. N is defined by the amount of given points.

    Parameters
    ----------
    m: np.ndarray:
        Bezier points

    t: float:
        value for which point is calculated

    make_copy: bool:
        optional parameter if computation should not be in place

    interval: Tuple[int, int]:
        if interval != (0,1) we need to transform t

    Returns
    -------
    np.ndarray:
        calculated points with given t
    """

    _, n = m.shape
    return de_caes_n_steps(m.copy(), t, n, interval) if make_copy else de_caes_n_steps(m, t, n, interval)


def de_caes_blossom(m: np.ndarray, ts: List[float], make_copy: bool = True,
                    interval: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Method computing blossomed de Casteljau which means that in every step a new parameter t is taken from
    the given List and only one iteration of de Castelljau is performed per t.

    Parameters
    ----------
    m: np.ndarray:
        Bezier points

    ts: List[float]:
        values used in calculation

    make_copy: bool:
        optional parameter if computation should not be in place

    interval: Tuple[int, int]:
        if interval != (0,1) we need to transform t

    Returns
    -------
    np.ndarray:
        calculated point with respect to given values
    """

    if m.shape[1] < 2:
        raise Exception("At least two points are needed")

    if len(ts) >= m.shape[1]:
        raise Exception("Too many values to use!")

    if not ts:
        raise Exception("At least one element is needed!")

    c = m.copy() if make_copy else m
    for t in ts:
        c = de_caes_one_step(c, t, interval)

    return c


def parallel_decaes_unblossomed(m: np.ndarray, ts, interval: Tuple[int, int] = (0, 1)) -> Iterator[np.ndarray]:
    """
    Method makes use of parallel execution to compute points on the curve represented by the bezier points. In contrast
    to the blossom method for every t in the given list a complete n iteration de Castelljau ist performed and not just
    one step. There are always 2 * cpu.count threads used to compute the points

    Parameters
    ----------
    m: np.ndarray:
        Bezier points

    ts: List[float]:
        values for which points on the curve should be computed

    interval: Tuple[int, int]:
        if interval != (0,1) we need to transform t

    Returns
    -------
    Iterator[np.ndarray]:
        Every entry is the result of one de Castelljau run wih respect to a given t from the List ts
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        return executor.map(lambda t: de_caes(m, t, make_copy=True, interval=interval), ts)


def subdivision(m: np.ndarray, t: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method uses subdivision to approximate curve defined by the given Bezier points at one value t. However the method
    subdivides only once, which means that it runs n iterations of de Castelljau for given t and splits the result in
    left and right.

    Parameters
    ----------
    m: np.ndarray:
        Bezier points

    t: float:
        value at which is subdivided

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        left and right part
    """
    left, right = np.zeros(m.shape), np.zeros(m.shape)
    current = m
    for i in range(m.shape[1]):
        left[::, i] = current.copy()[::, 0]
        right[::, -i - 1] = current.copy()[::, -1]
        current = de_caes_one_step(current, t, make_copy=True)

    return left, right
