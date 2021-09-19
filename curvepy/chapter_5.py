import sys as s
import numpy as np
import sympy as sy
import scipy.special as scs
from functools import reduce, partial
from typing import Tuple, Callable, Union, List, Any


def bernstein_polynomial_rec(n: int, i: int, t: float = 1) -> float:
    """
    Method using 5.8 to calculate a point with given bezier points

    Parameters
    ----------
    n: int:
        degree of the Bernstein Polynomials

    i: int:
        starting point for calculation

    t: float:
        value for which Bezier curve are calculated

    Returns
    -------
    float:
        value of Bernstein Polynomial B_i^n(t)

    Notes
    -----
    Equation used for computing:
    Base Case: B_0^0(t) = 1
    math:: i \\notin \\{0, \\dots, n\\} \\rightarrow B_i^n(t) = 0
    math:: B_i^n(t) = (1-t) \\cdot B_i^{n-1}(t) + t \\cdot B_{i-1}^{n-1}(t)
    """
    if i == n == 0:
        return 1
    if not 0 <= i <= n:
        return 0
    return (1-t)*bernstein_polynomial_rec(n-1, i, t) + t*bernstein_polynomial_rec(n-1, i-1, t)


def bernstein_polynomial(n: int, i: int, t: float = 1) -> float:
    """
    Method using 5.1 to calculate a point with given bezier points

    Parameters
    ----------
    n: int:
        degree of the Bernstein Polynomials

    i: int:
        starting point for calculation

    t: float:
        value for which Bezier curve are calculated

    Returns
    -------
    float:
        value of Bernstein Polynomial B_i^n(t)

    Notes
    -----
    Equation used for computing:
    math:: B_i^n(t) = \\binom{n}{i} t^{i} (1-t)^{n-i}
    """
    return scs.binom(n, i) * (t**i) * ((1-t)**(n-i))


def partial_bernstein_polynomial(n: int, i: int) -> Callable[[float], float]:
    """
    Method using 5.1 to calculate a point with given bezier points

    Parameters
    ----------
    n: int:
        degree of the Bernstein Polynomials

    i: int:
        starting point for calculation

    Returns
    -------
    Callable[[float], float]:
        partial application of 5.1

    Notes
    -----
    Equation used for computing:
    math:: B_i^n(t) = \\binom{n}{i} t^{i} (1-t)^{n-i}
    """
    return partial(bernstein_polynomial, n, i)


def bezier_curve_with_bernstein(m: np.ndarray, t: float = 0.5, r: int = 0,
                                interval: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Method using 5.8 to calculate a point with given betier points

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    t: float:
        value for which Bezier curve are calculated

    r: int:
        optional Parameter to calculate only a partial curve if we already have some degree of the bezier points

    interval: Tuple[float,float]:
        Interval of t used for affine transformation

    Returns
    -------
    np.ndarray:
            point on the curve

    Notes
    -----
    Equation used for computing:
    math:: b_i^r(t) = \\sum_{j=0}^r b_{i+j} \\cdot B_i^r(t)
    """
    _, n = m.shape
    t = (t-interval[0])/(interval[1]-interval[0])
    return np.sum([m[:, i] * bernstein_polynomial(n-r-1, i, t) for i in range(n-r)], axis=0)


def intermediate_bezier_points(m: np.ndarray, r: int, i: int, t: float = 1,
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
    t = (t-interval[0])/(interval[1]-interval[0])
    return np.sum([m[:, i+j]*bernstein_polynomial(n-1, j, t) for j in range(r)], axis=0)


def barycentric_combination_bezier(m: np.ndarray, c: np.ndarray, alpha: float = 0, beta: float = 1, t: float = 1,
                                   r: int = 0, interval: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Method using 5.13 to calculate the barycentric combination of two given bezier curves

    Parameters
    ----------
    m: np.ndarray:
        first array containing the Bezier Points

    c: np.ndarray:
        second array containing the Bezier Points

    alpha: float:
        weight for the first Bezier curve

    beta: float:
        weight for the first Bezier curve

    t: float:
        value for which Bezier curves are calculated

    r: int:
        optional Parameter to calculate only a partial curve if we already have some degree of the bezier points

    interval: Tuple[float,float]:
        Interval of t used for affine transformation

    Returns
    -------
    np.ndarray:
            point of the weighted combination

    Notes
    -----
    The Parameter alpha and beta must hold the following condition: alpha + beta = 1
    Equation used for computing:
    math:: \\sum_{j=0}^r (\\alpha \\cdot b_j + \\beta \\cdot c_j)B_j^n(t) =
    \\alpha \\cdot \\sum_{j=0}^r b_j \\cdot B_j^n(t) + \\beta \\cdot \\sum_{j=0}^r c_j \\cdot B_j^n(t)
    """

    if alpha + beta != 1:
        raise Exception("Alpha and Beta must add up to 1!")

    return alpha * bezier_curve_with_bernstein(m, t, r, interval)+beta * bezier_curve_with_bernstein(c, t, r, interval)


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


def bezier_to_power(m: np.ndarray) -> Callable[[float], np.ndarray]:
    """
    Method calculating monomial representation of given bezier form using 5.27

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    Returns
    -------
    Callable:
        bezier function in polynomial form

    Notes
    -----
    Equation 5.27 used for computing polynomial form:
    math:: b^n(t) = \\sum_{j=0}^n \\binom{n}{j} \\Delta^j b_0 t^j

    Initially the method would only compute the polynomial coefficients in an array, and parsing this array with a given
    t to the horner method we would get a point back. Instead the method uses sympy to calculate a function depending on
    t. After initial computation, f(t) calculates the value for a given t. Having a function it is simple to map it on
    an array containing multiple values for t.
    As a result we do not need to call the horner method for each t individually.
    """
    _, n = m.shape
    diff = all_forward_differences(m)
    t = sy.symbols('t')
    res = 0
    for i in range(n):
        res += scs.binom(n-1, i) * diff[:, i] * t**i

    return sy.lambdify(t, res)


def single_forward_difference(m: np.ndarray, i: int = 0, r: int = 0) -> np.ndarray:
    """
    Method using equation 5.23 to calculate forward difference of degree r for specific point i

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    i: int:
        point i for which forward all_forward_differences are calculated

    r: int:
        degree of forward difference

    Returns
    -------
    np.ndarray:
            forward difference of degree r for point i

    Notes
    -----
    Equation used for computing all_forward_differences:
    math:: \\Delta^r b_i = \\sum_{j=0}^r \\binom{r}{j} (-1)^{r-j} b_{i+j}
    """
    _, n = m.shape
    return np.sum([scs.binom(r, j)*(-1)**(r - j)*m[:, i + j] for j in range(0, r+1)], axis=0)


def all_forward_differences(m: np.ndarray, i: int = 0) -> np.ndarray:
    """
    Method using equation 5.23 to calculate all forward all_forward_differences for a given point i.
    First entry is first difference, second entry is second difference and so on.

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    i: int:
        point i for which forward all_forward_differences are calculated

    Returns
    -------
    np.ndarray:
         array holds all forward all_forward_differences for given point i

    Notes
    -----
    Equation used for computing all_forward_differences:
    math:: \\Delta^r b_i = \\sum_{j=0}^r \\binom{r}{j} (-1)^{r-j} b_{i+j}
    """
    _, n = m.shape
    diff = [single_forward_difference(m, i, r) for r in range(0, n)]
    return np.array(diff).T


def derivative_bezier_curve(m: np.ndarray, t: float = 1, r: int = 1) -> np.ndarray:
    """
    Method using equation 5.24 to calculate rth derivative of bezier curve at value t

    Parameters
    ----------
    m: np.ndarray:
        array containing the Bezier Points

    t: float:
        value for which Bezier curves are calculated

    r: int:
        rth derivative

    Returns
    -------
    np.ndarray:
         point of the rth derivative at value t

    Notes
    -----
    Equation used for computing:
    math:: \\frac{d^r}{dt^r} b^n(t) = \\frac{n!}{(n-r)!} \\cdot \\sum_{j=0}^{n-r} \\Delta^r b_j \\cdot B_j^{n-r}(t)
    """
    _, n = m.shape
    factor = scs.factorial(n)/scs.factorial(n-r)
    tmp = [factor * single_forward_difference(m, j, r) * bernstein_polynomial(n-r, j, t) for j in range(n-r)]
    return np.sum(tmp, axis=0)


def horner(m: np.ndarray, t: float = 0.5) -> Tuple[Union[float, Any], ...]:
    """
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
    return tuple(reduce(lambda x, y: t*x+y, m[i, ::-1]) for i in [0, 1])


def de_caes_one_step(m: np.ndarray, t: float = 0.5, interval: Tuple[int, int] = (0, 1)) -> np.ndarray:
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
    """
    if m.shape[1] < 2:
        raise Exception("At least two points are needed")

    t1 = (interval[1] - t)/(interval[1] - interval[0]) if interval != (0, 1) else (1-t)
    t2 = (t - interval[0])/(interval[1] - interval[0]) if interval != (0, 1) else t

    m[:, :-1] = t1 * m[:, :-1] + t2 * m[:, 1:]
    return m


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

    if r >= m.shape[1]:
        raise Exception("Can't perform r rounds!")

    if m.shape[1] < 2:
        raise Exception("At least two points are needed")

    for i in range(r):
        m = de_caes_one_step(m, t, interval)
        if i != r - 1:
            m = m[:, :-1]
    return m


def de_caes(m: np.ndarray, t: float = 0.5, make_copy: bool = False, interval: Tuple[int, int] = (0, 1)) -> np.ndarray:
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

    if m.shape[1] < 2:
        raise Exception("At least two points are needed")

    if m.shape[1] < 2:
        raise Exception("At least two points are needed")

    _, n = m.shape
    return de_caes_n_steps(m.copy(), t, n, interval) if make_copy else de_caes_n_steps(m, t, n, interval)


def de_caes_blossom(m: np.ndarray, ts: List[float], make_copy: bool = False,
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
    for i, t in enumerate(ts):
        c = de_caes_one_step(c, t, interval)
        if i != len(ts):
            c = c[:, :-1]
    return c


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
    return de_caes(m, t, True), de_caes(m[:, ::-1], 1.0-t, True)


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


def min_max_box(m: np.ndarray) -> List[float]:
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

    if p1.shape != p2.shape != p3.shape != p4.shape:
        raise Exception("Points need to be of the same dimension!")

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
    # we divide with z to turn back to 2D space
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
            res = np.append(res, p.reshape((2, 1)), axis=1)
    else:
        # if poly not flat enough we subdivide and check the resulting polygons for intersection
        p1, p2 = subdiv(m, 0.5)
        res = np.append(res, intersect(p1, tol).reshape((2, 1)), axis=1)
        res = np.append(res, intersect(p2, tol).reshape((2, 1)), axis=1)

    return res


def init() -> None:
    x = [0, 0, 8, 4]
    y = [0, 2, 2, 0]

    x_1 = [0]
    y_1 = [1]
    test = np.array([x, y], dtype=float)
    print(test.shape)
    #print(bernstein_polynomial(4, 2, 0.5))
    #print(bezier_curve_with_bernstein(test))
    _, n = test.shape
    print(de_caes_blossom(test, [0.5, 0.5, 0.25]))
    #print(de_caes(test, 0.5))
    #print(bernstein_polynomial_rec.__doc__)
    #print(min_max_box(test))
    #print(np.ndarray([]).size)
    #print(check_flat(test))
    #print(horn_bez(test))
    #print(all_forward_differences(test))


if __name__ == "__main__":
    init()
