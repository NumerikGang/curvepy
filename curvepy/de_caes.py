import concurrent.futures
from typing import List, Tuple
import numpy as np
from multiprocessing import cpu_count


def de_caes_one_step(m: np.ndarray, t: float = 0.5, interval: Tuple[int, int] = (0, 1),
                     make_copy: bool = True) -> np.ndarray:
    """
    Method computing one round of de Casteljau

    Parameters
    ----------
     m: np.ndarray:
        array containing coefficients

    t: float:
        value for which point is calculated

    interval: Tuple[int, int]:
        if interval != (0,1) we need to transform t

    Returns
    -------
    np.ndarray:
        array containing calculated points with given t
        :param make_copy:
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
    Method computing r round of de Casteljau

    Parameters
    ----------
    m: np.ndarray:
        array containing coefficients

    t: float:
        value for which point is calculated

    r: int:
        how many rounds of de Casteljau algorithm should be performed

    interval: Tuple[int, int]:
        if interval != (0,1) we need to transform t

    Returns
    -------
    np.ndarray:
        array containing calculated points with given t
    """

    for _ in range(r):
        m = de_caes_one_step(m, t, interval, make_copy=False)
    return m


def de_caes(m: np.ndarray, t: float = 0.5, make_copy: bool = True, interval: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Method computing de Casteljau

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

    _, n = m.shape
    return de_caes_n_steps(m.copy(), t, n, interval) if make_copy else de_caes_n_steps(m, t, n, interval)


def de_caes_blossom(m: np.ndarray, ts: List[float], make_copy: bool = True,
                    interval: Tuple[int, int] = (0, 1)) -> np.ndarray:
    """
    Method computing de Casteljau with different values of t in each step

    Parameters
    ----------
    m: np.ndarray:
        array containing coefficients

    ts: List[float]:
        List containing all ts that are used in calculation

    make_copy: bool:
        optional parameter if computation should not be in place

    interval: Tuple[int, int]:
        if interval != (0,1) we need to transform t

    Returns
    -------
    np.ndarray:
        array containing calculated points with given t
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


def parallel_decaes_unblossomed(m: np.ndarray, ts, interval: Tuple[int, int] = (0, 1)):
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
        return executor.map(lambda t: de_caes(m, t, make_copy=True, interval=interval), ts)


def subdivision(m: np.ndarray, t: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    left, right = np.zeros(m.shape), np.zeros(m.shape)
    current = m
    for i in range(m.shape[1]):
        left[::, i] = current.copy()[::, 0]
        right[::, -i - 1] = current.copy()[::, -1]
        current = de_caes_one_step(current, t, make_copy=True)

    return left, right
