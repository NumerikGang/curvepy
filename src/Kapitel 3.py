import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Callable
import functools


def straight_line_point(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray:
    """
    method to calculate a single point on a straight line through a and b

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


def straight_line_function(a: np.ndarray, b: np.ndarray) -> functools.partial:
    """
    method to get the function of a straight line through a and b

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


def ratio(left: np.ndarray, col_point: np.ndarray, right: np.ndarray) -> float:
    """
    method to calculate the ratio of the three collinear points from the parameters

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
    for i in range(len(left)):
        if left[i] == right[i]:
            continue
        return (col_point[i] - left[i]) / (right[i] - col_point[i])
    return 0


def ratio_test() -> None:
    left = np.array([0, 0, 0])
    right = np.array([1, 1, 1])
    col_point = np.array([.66, .66, .66])
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
    straight_line_point_test()
    ratio_test()
    # print(ratio((0, 0), (1, 1), (10, 10)))


if __name__ == "__main__":
    init()

#####################################################################################################
