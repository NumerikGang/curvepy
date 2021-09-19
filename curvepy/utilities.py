from typing import Any, List, Callable
import numpy as np
import functools


def straight_line_point(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray:
    """
    Method to calculate a single point on a straight line through a and b.

    Parameters
    ----------
    a: np.ndArray
        first point on straight line to calculate new point
    b: np.ndArray
        second point on straight line to calculate new point
    t: float
        for the weight of a and b

    Returns
    -------
    np.ndArray:
        new point on straight line through a and b
    """
    return (1 - t) * a + t * b


def create_straight_line_function(a: np.ndarray, b: np.ndarray) -> Callable[[float], np.ndarray]:
    """
    Method to get the function of a straight line through a and b.

    Parameters
    ----------
    a: np.ndArray
        first point on straight line
    b: np.ndArray
        second point on straight line

    Returns
    -------
    StraightLineFunction:
        function for the straight line through a and b
    """
    return functools.partial(straight_line_point, a, b)


def collinear_check(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """
    Calculates the cross product of (b-a) and (c-a) to see if all 3 points are collinear.

    Parameters
    ----------
    a: np.ndArray
        first point
    b: np.ndArray
        second point
    c: np.ndArray
        third point

    Returns
    -------
    bool:
        True if points are collinear else False
    """
    return np.count_nonzero(np.cross(b - a, c - a)) == 0


def ratio(left: np.ndarray, col_point: np.ndarray, right: np.ndarray) -> float:
    """
    Method to calculate the ratio of the three collinear points from the parameters.
    Throws an exception if the points are not collinear.
    Throws an exception if the points don't have the same dimension.

    Parameters
    ----------
    left: np.ndArray
        left point that defines the straight line
    col_point: np.ndArray
        collinear point to left and right, could be the most left or most right point or between left and right
    right: np.ndArray
        right point that defines the straight line

    Returns
    -------
    np.ndArray:
        the ratio of the three collinear points from the parameters
    """
    if not collinear_check(left, col_point, right):
        raise Exception("The points are not collinear!")

    if left.shape != col_point.shape != right.shape:
        raise Exception("The points don't have the same dimension!")

    for l, r, c in zip(left, right, col_point):
        if l != r and r - c != 0:
            return (c - l) / (r - c)
        elif r - c == 0:
            return np.NINF
    return 0


def flatten_list_of_lists(xss: List[List[Any]]) -> List[Any]:
    return sum(xss, [])


def csv_read(file_path: str) -> np.ndarray:
    try:
        with open(file_path, 'r') as csv_file:
            xs, ys, zs = [], [], []
            for line in csv_file:
                try:
                    x, y, z = line.split(',')
                    zs.append(float(z))
                except ValueError:
                    try:
                        x, y = line.split(',')
                    except ValueError:
                        print('Expected two or three values per line')
                        return np.array([])
                xs.append(float(x))
                ys.append(float(y))
        return np.array([xs, ys], dtype=float) if not zs else np.array([xs, ys, zs], dtype=float)
    except FileNotFoundError:
        print(f'File: {file_path} does not exist.')
        return np.array([])
