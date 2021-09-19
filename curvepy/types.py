from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Deque, List, NamedTuple, Tuple, Callable, Union
from functools import cached_property


class CurveTypes(Enum):
    bezier_curve = 0
    bezier_curve_threaded = 1
    bezier_curve_blossoms = 2


Point2D = Tuple[float, float]
Edge2D = Tuple[Point2D, Point2D]
StraightLineFunction = Callable[[float], np.ndarray]


class TriangleTuple(NamedTuple):
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
        # TODO: Make flake8 exception
        return self._points

    @cached_property
    def area(self) -> float:
        a, b, c = self.points
        return self.calc_area(np.array(a), np.array(b), np.array(c))

    @staticmethod
    def calc_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculates the "calc_area" of a Triangle defined by the parameters. All three points have to be on a plane
        parallel to an axis-plane!

        Parameters
        ----------
        a: np.ndarray
            First point of the Triangle.
        b: np.ndarray
            Second point of the Triangle.
        c: np.ndarray
            Third point of the Triangle.

        Returns
        -------
        float:
            "Area" of the Triangle.
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


class Triangle(AbstractTriangle):
    def __init__(self, a: Point2D, b: Point2D, c: Point2D):
        self._points: Tuple[Point2D, Point2D, Point2D] = (a, b, c)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Triangle) and sorted(self.points) == sorted(other.points)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.points)))

VoronoiRegions2D = Dict[Point2D, Deque[TriangleTuple]]
