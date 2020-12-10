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
    np.ndArray:
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
    Callable:
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
    return np.count_nonzero(np.cross(d1, d2)) == 0


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
    np.ndArray:
        the ratio of the three collinear points from the parameters
    """
    if not collinear_check(left, col_point, right):
        raise Exception("The points are not collinear!")

    for i in range(len(left)):
        if left[i] == right[i]:
            continue
        if right[i] - col_point[i] == 0:
            return np.NINF
        return (col_point[i] - left[i]) / (right[i] - col_point[i])
    return 0


# split in 2d and 3d Polygon similar to 2d and 3d bezier?
class Polygon:
    """
    Class for creating a 2D or 3D Polygon.

    Parameters
    ----------

    Attributes
    ----------
    _points: np.ndArray
        array containing copy of points that create the polygon
    _dim: int
        dimension of the polygon
    _piece_funcs: list
        list containing all between each points, _points[i] and _points[i+1]

    """

    def __init__(self, points: np.ndarray) -> None:
        self._points = points.copy()
        self._dim = points.shape[1]
        self._piece_funcs = self.create_polygon()

    def create_polygon(self) -> list:
        """
        Creates the polygon by creating an array with all straight_line_functions needed.

        Parameters
        ----------

        Returns
        -------
        fs: list
            the array with all straight_line_functions
        """
        fs = []
        for a, b in zip(self._points[0:len(self._points) - 1], self._points[1:len(self._points)]):
            fs.append(straight_line_function(a, b))
        return fs

    def evaluate(self, x: float) -> np.ndarray:
        """
        Menelaos' Theorem b[index, t] = evaluate(x), where x = t,index, x \in [0, len(self._piece_funcs)]

        Parameters
        ----------
        x: float

        Returns
        -------
        np.ndArray:
            evaluated point: np.ndArray
        """
        t, index = math.modf(x)
        if int(index) > len(self._piece_funcs) or int(index) < 0:
            raise Exception("Not defined!")
        if int(index) == len(self._piece_funcs):
            return self._piece_funcs[len(self._piece_funcs) - 1](1)
        return self._piece_funcs[int(index)](t)

    def plot_polygon(self, xs: np.ndarray) -> None:
        """
        Plots the polygon using matplotlib, either in 2D or 3D. Two Plots are given first one with a given number of points which will be
        highlighted on the function, and second is the function as a whole.

        Parameters
        ----------
        xs: np.ndArray
            the points that may be plotted

        Returns
        -------
        None
        """
        ep = np.array([self.evaluate(x) for x in xs])
        np.append(ep, self._points)
        ravel_points = self._points.ravel()  # the corner points to plot the function
        if self._dim == 2:
            tmp = ep.ravel()
            xs, ys = tmp[0::2], tmp[1::2]
            func_x, func_y = ravel_points[0::2], ravel_points[1::2]
            plt.plot(func_x, func_y)
            plt.plot(xs, ys, 'o', markersize=3)
        if self._dim == 3:
            tmp = ep.ravel()
            xs, ys, zs = tmp[0::3], tmp[1::3], tmp[2::3]
            func_x, func_y, func_z = ravel_points[0::3], ravel_points[1::3], ravel_points[2::3]
            ax = plt.axes(projection='3d')
            ax.plot(func_x, func_y, func_z)
            ax.plot(xs, ys, zs, 'o', markersize=3)
        plt.show()


def ratio_test() -> None:
    left = np.array([0, 0, 0])
    right = np.array([1, 1, 1])
    col_point = np.array([1.2, 1.3, 1.4])
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
    test_points = np.array([[0, 0, 0], [1, 1, 1], [3, 4, 4], [5, -2, -2]])

    test_points2d = np.array([[0, 0], [1, 1], [3, 4], [5, -2]])
    print(len(test_points2d))

    test_PG = Polygon(test_points)
    print(test_PG.evaluate(1.5))
    test_PG.plot_polygon(np.linspace(0, 3, 20))
    # test_PG.plot_polygon(np.array([0.5]))


if __name__ == "__main__":
    init()

#####################################################################################################
