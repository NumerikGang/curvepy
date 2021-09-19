from enum import Enum
import numpy as np
from typing import Any, Dict, Deque, List, NamedTuple, Tuple
from functools import cached_property


class CurveTypes(Enum):
    bezier_curve = 0
    bezier_curve_threaded = 1
    bezier_curve_blossoms = 2


Point2D = Tuple[float, float]
Edge2D = Tuple[Point2D, Point2D]


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
        return hash(tuple([*self.center, self.radius]))


class Triangle:
    def __init__(self, a: Point2D, b: Point2D, c: Point2D):
        self._points: Tuple[Point2D, Point2D, Point2D] = (a, b, c)

    @cached_property
    def lines(self) -> List[Edge2D]:
        a, b, c = self.points
        return [(a, b), (b, c), (a, c)]

    @cached_property
    def points(self) -> Tuple[Point2D, Point2D, Point2D]:
        # If it was mutable caching would break
        return self._points

    @cached_property
    def area(self) -> float:
        a, b, c = self.points
        return self.calc_area(*a, *b, *c)

    @cached_property
    def circumcircle(self) -> Circle:
        """
        :return:

        """

        a, b, c = self.points

        tmp_pts_1 = np.array([np.array(x - np.array(a)) for x in [b, c]])
        tmp_pts_2 = np.sum(tmp_pts_1 ** 2, axis=1)
        tmp_pts_3 = np.array([np.linalg.det([x, tmp_pts_2]) / (2 * np.linalg.det(tmp_pts_1)) for x in tmp_pts_1.T])
        center = a[0] - tmp_pts_3[1], a[1] + tmp_pts_3[0]
        radius = np.linalg.norm(np.array(a) - np.array(center))

        return Circle(center=center, radius=radius)

    @staticmethod
    def calc_area(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
        """
        See: https://www.mathopenref.com/coordtrianglearea.html
        """
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    def __str__(self) -> str:
        return str(self.points)

    def __repr__(self) -> str:
        return f"<TRIANLGE: {str(self)}>"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Triangle) and sorted(self.points) == sorted(other.points)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.points)))


VoronoiRegions2D = Dict[Point2D, Deque[Triangle]]
