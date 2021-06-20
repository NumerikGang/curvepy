import numpy as np
import random as rd
from functools import cached_property
from typing import List, Tuple, Any
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from curvepy.reference_implementation import Delaunay2D

Point2D = Tuple[float, float]
Edge2D = Tuple[Point2D, Point2D]


class Circle:
    def __init__(self, center: Point2D, radius: float):
        self.center = np.array(center)
        self.radius = radius

    def __contains__(self, pt: Point2D) -> bool:
        return np.linalg.norm(np.array(pt) - self.center) <= self.radius

    def __str__(self) -> str:
        return f"(CENTER: {self.center}, RADIUS: {self.radius})"

    def __repr__(self) -> str:
        return f"<CIRCLE: {str(self)}>"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Circle) and self.center == other.center and self.radius == other.radius


class Triangle:
    def __init__(self, a: Point2D, b: Point2D, c: Point2D):
        self._points: List[Point2D] = sorted([a, b, c])

    @property
    def points(self) -> List[Point2D]:
        return self._points

    @cached_property
    def area(self) -> float:
        a, b, c = self.points
        return self.calc_area(*a, *b, *c)

    @cached_property
    def circumcircle(self) -> Circle:
        """
        :return:

        See: https://de.wikipedia.org/wiki/Umkreis#Koordinaten
        See: https://de.wikipedia.org/wiki/Umkreis#Radius
        """
        A, B, C = self.points
        [x1, y1], [x2, y2], [x3, y3] = A, B, C
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        xu = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
        yu = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

        lines = [[A, B], [B, C], [A, C]]
        c, a, b = [np.linalg.norm(np.array(x) - np.array(y)) for x, y in lines]

        R = (a * b * c) / (4 * self.area)
        return Circle(center=(xu, yu), radius=R)

    @cached_property
    def edges(self) -> List[Edge2D]:
        # (<) not only removes same elements but also duplicates in different order
        return [(x, y) for x in self.points for y in self.points if x < y]

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
        return isinstance(other, Triangle) and self.points == other.points


class DelaunayTriangulation2D:
    def __init__(self, center: Point2D = (0, 0), radius: float = 1000):
        self.supertriangles: List[Triangle] = self._create_supertriangles(center, radius)
        self._triangles: List[Triangle] = [*self.supertriangles]

    @property
    def triangles(self):
        # We have to remove everything containing vertices of the supertriangle
        all_points_in_supertriangle = sum([x.points for x in self.supertriangles], [])
        remove_if = lambda t: any(pt in t.points for pt in all_points_in_supertriangle)
        return [t for t in self._triangles if not remove_if(t)]

    def _create_supertriangles(self, center: Point2D, radius):
        # Since we have to start with a valid triangulation, we split our allowed range into 2 triangles like that:
        # x────────────────────────────────┐
        # xx                               │
        # │ xx                             │
        # │   xxx              T2          │
        # │     xxxx                       │
        # │        xxx                     │
        # │          xxx                   │
        # │            xxxx                │
        # │               xxxx             │
        # │                  xxx           │
        # │       T1           xxxx        │
        # │                       xxx      │
        # │                         xxx    │
        # │                           xxx  │
        # │                             xxx│
        # └────────────────────────────────┴
        # Those 2 are called the "supertriangles" in most literature

        # np.array for easier calculation
        center = np.array(center)
        base_rectangle = [center + radius * np.array([i, j]) for i, j in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        lower_left, lower_right, upper_right, upper_left = [tuple(x) for x in base_rectangle]
        lower_triangle = Triangle(lower_left, lower_right, upper_left)
        upper_triangle = Triangle(upper_right, upper_left, lower_right)
        return [lower_triangle, upper_triangle]

    def add_point(self, p: Point2D):
        bad_triangles = [tri for tri in self._triangles if p in tri.circumcircle]

        # An edge is part of the boundary iff it doesn't is not part of another bad triangle
        boundaries = self._find_edges_only_used_by_a_single_triangle(bad_triangles)

        # remove all bad ones
        for tri in bad_triangles:
            self._triangles.remove(tri)
        # Replace the hole with the boundaries and our new point
        for edge in boundaries:
            self._triangles.append(Triangle(p, *edge))

    @staticmethod
    def _find_edges_only_used_by_a_single_triangle(triangles, no_duplicates=True):
        ret = []
        for triangle in triangles:
            others = list(triangles)
            others.remove(triangle)
            for edge in triangle.edges:
                if all(edge not in other.edges for other in others):
                    ret.append(edge)

        # set() works since all edges are ordered, we don't care about the order and tuples are hashable
        if no_duplicates:
            ret = list(set(ret))

        return ret


if __name__ == '__main__':
    n = 50
    min, max = -100, 100
    pts = [(rd.uniform(min, max), rd.uniform(min, max)) for _ in range(n)]
    d = DelaunayTriangulation2D(radius=max + 5)  # Buffer for rounding errors
    for p in pts:
        d.add_point(p)

    plt.rcParams["figure.figsize"] = (10,30)
    figure, axis = plt.subplots(3)

    axis[0].set_title("meins")
    for tri in d.triangles:
        points = np.ravel(tri.points)
        axis[0].triplot(points[0::2], points[1::2])

    axis[1].set_title("scipy")
    points = np.array([list(x) for x in pts])
    scipy_tri = Delaunay(pts)
    axis[1].triplot(points[:, 0], points[:, 1], scipy_tri.simplices)
    axis[1].plot(points[:, 0], points[:, 1], 'o')

    axis[2].set_title("reference implementation")
    d2 = Delaunay2D(radius=max+5)
    for p in pts:
        d2.addPoint(p)
    coord, tris = d2.exportDT()
    my_tris = [(coord[a], coord[b], coord[c]) for a,b,c in tris]
    for tri in my_tris:
        points = np.ravel(tri)
        axis[2].triplot(points[0::2], points[1::2])

    plt.show()