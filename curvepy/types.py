from enum import Enum
import numpy as np
import sys
import scipy.special as scs

from abc import ABC, abstractmethod
from typing import Any, Dict, Deque, List, NamedTuple, Tuple, Union, Callable
from functools import cached_property, partial
from collections.abc import Sequence
from curvepy.utilities import create_straight_line_function

MinMaxBox = np.ndarray

class CurveTypes(Enum):
    bezier_curve = 0
    bezier_curve_threaded = 1
    bezier_curve_blossoms = 2


Point2D = Tuple[float, float]
Edge2D = Tuple[Point2D, Point2D]


class TriangleNode(NamedTuple):
    ccw: Point2D
    cw: Point2D
    pt: Point2D
    ccc: Point2D


class Circle:
    def __init__(self, center: Point2D, radius: float):
        self._center = np.array(center)
        self.radius = radius

    @property
    def center(self) -> Point2D:
        return tuple(self._center)

    def __contains__(self, pt: Point2D) -> bool:
        return np.linalg.norm(np.array(pt) - self._center) <= self.radius

    def __str__(self) -> str:
        return f"(CENTER: {self.center}, RADIUS: {self.radius})"

    def __repr__(self) -> str:
        return f"<CIRCLE: {str(self)}>"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Circle) and self.center == other.center and self.radius == other.radius

    def __hash__(self) -> int:
        return hash((*self.center, self.radius))


class Polygon(Sequence):
    """
    Class for creating a 2D or 3D Polygon.

    Attributes
    ----------
    _points: np.ndArray
        array containing copy of points that create the polygon
    _dim: int
        dimension of the polygon
    _piece_funcs: list
        list containing all between each points, _points[i] and _points[i+1]

    """

    def __init__(self, points: List[np.ndarray], make_copy: bool = True) -> None:
        if len(points) < 2:
            raise ValueError("Unsupported Dimension")
        if any([points[0].shape != x.shape for x in points]):
            raise Exception("The points don't have the same dimension!")
        self._dim = points[0].shape[1]
        if self._dim not in [2, 3]:
            raise Exception("The points don't have dimension of 2 or 3!")
        self._points = points.copy() if make_copy else points
        self._piece_funcs = self.create_polygon()

    def create_polygon(self) -> List[Callable[[float], np.ndarray]]:
        """
        Creates the polygon by creating an array with all straight_line_functions needed.

        Returns
        -------
        np.ndarray:
            the array with all straight_line_functions
        """
        return [create_straight_line_function(a, b) for a, b in zip(self._points[:-1], self._points[1:])]

    def __getitem__(self, item: Any) -> Callable[[float], np.ndarray]:
        """
        Throws ValueError when casting to int.
        Throws IndexError when out of bounds.
        And probably something else.
        """
        return self._piece_funcs[int(item)]

    def __len__(self) -> int:
        return len(self._points)

    def blossom(self, ts: List[float]) -> np.ndarray:
        """
        Recursive calculation of a blossom with parameters ts and the polygon.

        Parameters
        ----------
        ts: list
            b[t_1, t_2, ..., t_n]

        Returns
        -------
        np.ndArray:
            Calculated value for the blossom.
        """
        if len(ts) > len(self._piece_funcs):
            raise Exception("The polygon is not long enough for all the ts!")
        if len(ts) == 1:
            return self[0](ts[0])
        return Polygon([self[i](ts[0]) for i in range(len(ts))]).blossom(ts[1:])


class AbstractTriangle(ABC):

    @abstractmethod
    def __init__(self):
        # In order to not have unresolved references to _points
        self._points = None

    @cached_property
    def lines(self) -> Union[List[Edge2D], List[Tuple[np.ndarray]]]:
        a, b, c = self.points
        return [(a, b), (b, c), (a, c)]

    @cached_property
    def points(self) -> Union[Tuple[Point2D, Point2D, Point2D], List[np.ndarray]]:
        # If it was mutable caching would break
        return self._points

    @cached_property
    def area(self) -> float:
        a, b, c = self.points
        return self.calc_area(np.array(a), np.array(b), np.array(c))

    @staticmethod
    def calc_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculates the "calc_area" of a TupleTriangle defined by the parameters. All three points have to be on a plane
        parallel to an axis-plane!

        Parameters
        ----------
        a: np.ndarray
            First point of the TupleTriangle.
        b: np.ndarray
            Second point of the TupleTriangle.
        c: np.ndarray
            Third point of the TupleTriangle.

        Returns
        -------
        float:
            "Area" of the TupleTriangle.
        """
        return np.linalg.det(np.array([a, b, c])) / 2

    @cached_property
    def circumcircle(self) -> Circle:
        """
        see: https://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        :return:

        """
        a, b, c = self.points

        tmp_pts_1 = np.array([np.array(x - np.array(a)) for x in [b, c]])
        tmp_pts_2 = np.sum(tmp_pts_1 ** 2, axis=1)
        tmp_pts_3 = np.array([np.linalg.det([x, tmp_pts_2]) / (2 * np.linalg.det(tmp_pts_1)) for x in tmp_pts_1.T])
        center = a[0] - tmp_pts_3[1], a[1] + tmp_pts_3[0]
        radius = np.linalg.norm(np.array(a) - np.array(center))

        return Circle(center=center, radius=radius)

    def __str__(self) -> str:
        return str(self.points)

    def __repr__(self) -> str:
        return f"<TRIANLGE: {str(self)}>"


class TupleTriangle(AbstractTriangle):
    def __init__(self, a: Point2D, b: Point2D, c: Point2D):
        self._points: Tuple[Point2D, Point2D, Point2D] = (a, b, c)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, TupleTriangle) and sorted(self.points) == sorted(other.points)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.points)))


class PolygonTriangle(Polygon, AbstractTriangle):

    def __init__(self, points: List[np.ndarray], make_copy: bool = True) -> None:
        points.append(points[0])
        # https://stackoverflow.com/a/26927718
        Polygon.__init__(self, points, make_copy)

    def bary_plane_point(self, bary_coords: np.ndarray) -> np.ndarray:
        """
        Given the barycentric coordinates and three points this method will calculate a new point as a barycentric
        combination of the TupleTriangle points.

        Parameters
        ----------
        bary_coords: np.ndarray
            The barycentric coordinates corresponding to a, b, c. Have to sum up to 1.

        Returns
        -------
        np.ndarray:
            Barycentric combination of a, b, c with given coordinates.
        """
        if abs(1 - np.sum(bary_coords)) < sys.float_info.epsilon:
            raise Exception("The barycentric coordinates don't sum up to 1!")
        return np.sum(bary_coords.reshape((3, 1)) * self._points, axis=0)

    def squash_parallel_to_axis_plane(self, p: np.ndarray) -> List[np.ndarray]:
        """
        This method projects p and the points of the TupleTriangle on a plane, for example the y-plane with distance 1
        for all points of the TupleTriangle to the plane, so that cramer's rule can easily be applied to them
        in order to calculate the calc_area of the TupleTriangle corresponding to every 3 out of the 4 points.
        But this method does not overwrite the self._points.

        Parameters
        ----------
        p: np.ndarray
            Additional point that should be on the same plane as the TupleTriangle.

        Returns
        -------
        List[np.ndarray]:
            Copy of p and the TupleTriangle points now mapped on to a plane.
        """
        p_copy, a, b, c = [x.copy() for x in [p, *self._points]]
        for i, _ in enumerate(self._points):
            if a[i] != b[i] != c[i] and a[i - 1] != b[i - 1] != c[i - 1]:
                p_copy[i - 2], a[i - 2], b[i - 2], c[i - 2] = 1, 1, 1, 1
                break

        return [p_copy, a, b, c]

    def check_points_for_area_calc(self, p: np.ndarray) -> List[np.ndarray]:
        """
        This method checks if the point p and the points of the TupleTriangle have the right dimension and will make
        them so that cramer's rule can be applied to them.

        Parameters
        ----------
        p: np.ndarray
            Additional point that has to be on the same plane as the TupleTriangle.

        Returns
        -------
        List[np.ndarrays]:
            The TupleTriangle points and p so that cramer's rule can be used.
        """
        if self._dim == 3:
            return self.squash_parallel_to_axis_plane(p)
        return [np.hstack((x.copy(), [1])) for x in [p, *self._points]]

    # TODO: If 3D: Check if the 3D-Point lies on the 2D-Hyperplane defined by the bary coordinates
    # TODO: If not, throw an exception
    def get_bary_coords(self, p: np.ndarray) -> np.ndarray:
        """
        Calculates the barycentric coordinates of p with respect to the points defining the TupleTriangle.

        Parameters
        ----------
        p: np.ndarray
            Point of which we want the barycentric coordinates.

        Returns
        -------
        np.ndarray:
            Barycentric coordinates of p with respect to a, b, c.
        """
        p_copy, a, b, c = self.check_points_for_area_calc(p)

        abc_area = self.area
        if abc_area == 0:
            raise Exception("The calc_area of the TupleTriangle defined by a, b, c has to be greater than 0!")

        return np.array([self.calc_area(p_copy, b, c) / abc_area, self.calc_area(a, p_copy, c) / abc_area,
                         self.calc_area(a, b, p_copy) / abc_area])


class Triangle:
    def __call__(self, a, b, c):
        if isinstance(a, tuple):
            return TupleTriangle(a, b, c)
        elif isinstance(a, np.ndarray):
            return PolygonTriangle([a, b, c])
        else:
            raise TypeError("Unexpected Datatype")


VoronoiRegions2D = Dict[Point2D, Deque[TriangleNode]]


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
    return (1 - t) * bernstein_polynomial_rec(n - 1, i, t) + t * bernstein_polynomial_rec(n - 1, i - 1, t)


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
    return scs.binom(n, i) * (t ** i) * ((1 - t) ** (n - i))


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


def intermediate_bezier_points(m: np.ndarray, r: int, i: int, t: float = 0.5,
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
    t = (t - interval[0]) / (interval[1] - interval[0])
    return np.sum([m[:, i + j] * bernstein_polynomial(n - 1, j, t) for j in range(r)], axis=0)
