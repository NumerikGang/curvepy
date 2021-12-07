from typing import Any, List, Callable, Tuple, Union, Iterable
from numbers import Number
import numpy as np
import functools
import sys
import operator


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
    return np.allclose(np.cross(b - a, c - a), np.zeros(a.shape))


def ratio(left_point: np.ndarray, col_point: np.ndarray, right_point: np.ndarray) -> float:
    """

    (b-a)/(c-b)

    Method to calculate the ratio of the three collinear points from the parameters.
    Throws an exception if the points are not collinear.

    Parameters
    ----------
    left_point: np.ndArray
        left point that defines the straight line
    col_point: np.ndArray
        collinear point to left and right, could be the most left or most right point or between left and right
    right_point: np.ndArray
        right point that defines the straight line

    Returns
    -------
    np.ndArray:
        the ratio of the three collinear points from the parameters
    """
    if not collinear_check(left_point, col_point, right_point):
        raise ValueError("The points are not collinear!")

    for left, right, col in zip(left_point, right_point, col_point):
        if right - col == 0:
            return np.NaN
        elif left != right:
            return (col - left) / (right - col)
    return 0


"""
a + x(b + cx) -> a + x(b + x(c)) -> a + bx + cx^2

a + x(b + x(c+x*(d)))
"""


def horner(m: np.ndarray, t: float = 0.5) -> Tuple[Union[float, Any], ...]:
    """
    TODO show which problem this is
    TODO besserer Name sowie auch BezierCurveHorner mit horner-bez
    TODO First coeff == Highest Degree
    Method using horner's method to calculate point with given t

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
    return tuple(functools.reduce(lambda x, y: t * x + y, m[i, ::-1]) for i in range(m.shape[0]))


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

    Notes
    -----
    Given p1 and p2 we can check the distance p3 has to the line going through p1 and p2 as follows:
    math:: distance(p1,p2,p3) = \\frac{|(x_1-x_1)(y_1-y_3) - (x_1-x_3)(y_2-y_1)|}{//sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}}
    more information on "https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line"
    """

    if any((p.shape[0] < 2 for p in [p1, p2, p_to_check])):
        raise Exception("At least 2 dimensions are needed")

    if p1.shape != p2.shape != p_to_check.shape:
        raise Exception("points need to be of the same dimension!")

    numerator = abs((p2[0] - p1[0]) * (p1[1] - p_to_check[1]) - (p1[0] - p_to_check[0]) * (p2[1] - p1[1]))
    denominator = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return numerator / denominator


def check_flat(m: np.ndarray, tol: float = sys.float_info.epsilon) -> bool:
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
    return all(distance_to_line(m[:, 0], m[:, len(m[0]) - 1], m[:, i]) <= tol for i in range(1, len(m[0]) - 1))


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

    if p1.shape != p2.shape != p3.shape != p4.shape:
        raise Exception("Points need to be of the same dimension!")

    # First we vertical stack the points in an array
    vertical_stack = np.vstack([p1, p2, p3, p4])
    # Then we transform them to homogeneous coordinates, to perform a little trick
    homogeneous = np.hstack((vertical_stack, np.ones((4, 1))))
    # having our points in this form we can get the lines through the cross product
    line_1, line_2 = np.cross(homogeneous[0], homogeneous[1]), np.cross(homogeneous[2], homogeneous[3])
    # when we calculate the cross product of the lines we get intersect_with_x_axis point
    x, y, z = np.cross(line_1, line_2)
    if z == 0:
        return None
    # we divide with z to turn back to 2D space
    return np.array([x / z, y / z])


def flatten_list_of_lists(xss: List[List[Any]]) -> List[Any]:
    return sum(xss, [])


def prod(xs: Iterable[Number]):
    return functools.reduce(operator.mul, xs, 1)
