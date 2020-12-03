import sys
import numpy as np
import sympy as sy
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
import math
from typing import Tuple, Callable
import functools
import matplotlib.pyplot as plt
from sympy.plotting import plot3d_parametric_line


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
    new point on straight line through a and b
    """
    return (1 - t) * a + t * b


def straight_line_function(a: np.ndarray, b: np.ndarray) -> Callable:
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
    function for the straight line through a and b
    """
    return functools.partial(straight_line_point, a, b)


def collinear_check(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """
    Calculates the cross product of d1 and d2 to see if all 3 points are collinear.

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
    d1 = b - a
    d2 = c - a
    if np.count_nonzero(np.cross(d1, d2)) == 0: return True
    return False


def ratio(left: np.ndarray, col_point: np.ndarray, right: np.ndarray) -> float:
    """
    Method to calculate the ratio of the three collinear points from the parameters.
    Throws an exception if the points are not collinear.

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
    the ratio of the three collinear points from the parameters
    """
    if not collinear_check(left, col_point, right): sys.exit("The points are not collinear!")

    for i in range(len(left)):
        if left[i] == right[i]:
            continue
        if right[i] - col_point[i] == 0:
            return np.NINF
        return (col_point[i] - left[i]) / (right[i] - col_point[i])
    return 0


class Polygon:

    def __init__(self, points: np.ndarray) -> None:
        self._piece_funcs = self.create_polygon(points)

    def create_polygon(self, points: np.ndarray) -> list:
        fs = []
        for a, b in zip(points[0:len(points)-1], points[1:len(points)]):
            fs.append(straight_line_function(a, b))
        return fs

    def eval(self, x: float) -> np.ndarray:
        t, index = math.modf(x)
        if int(index) >= len(self._piece_funcs): sys.exit("Not defined!")
        return self._piece_funcs[int(index)](t)

def ratio_test() -> None:
    left = np.array([0, 0, 0])
    right = np.array([1, 1, 1])
    col_point = np.array([0.00000000000001, 0.00000000000001, 0.00000000000001])
    test = ratio(left, col_point, right)
    print(test)


def straight_line_point_test() -> None:
    t = 0
    fig = plt.figure()
    ax = Axes3D(fig)
    while t <= 1:
        test = straight_line_point(np.array([0, 0, 0]), np.array([1, 1, 1]), t)
        ax.scatter(test[0], test[1], test[2])
        t += 0.1
    plt.show()


def init() -> None:
    # straight_line_point_test()
    # ratio_test()
    test_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    test_PG = Polygon(test_points)
    print(test_PG.eval(1.5))


if __name__ == "__main__":
    init()

#####################################################################################################
"""
interp1d(x, y) for piecewise linear interpolation
"""
