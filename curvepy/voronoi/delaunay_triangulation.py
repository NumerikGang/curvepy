import numpy as np
import random as rd
from functools import cached_property
from typing import List, Tuple, Any, NamedTuple
import matplotlib.pyplot as plt

from curvepy.dev.reference_implementation import Delaunay2D

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

    def __hash__(self) -> int:
        return hash(tuple([*self.center, self.radius]))


class Triangle:
    def __init__(self, a: Point2D, b: Point2D, c: Point2D):
        self._points: Tuple[Point2D, Point2D, Point2D] = (a, b, c)

    @property
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

class DelaunayTriangulation2D:

    class _BoundaryNode(NamedTuple):
        new_triangle: Triangle
        connecting_edge: Edge2D
        opposite_triangle: Triangle

    def __init__(self, center: Point2D = (0, 0), radius: float = 500):
        t1, t2 = self._create_supertriangles(center, radius)
        self.supertriangles: List[Triangle] = [t1, t2]
        self._neighbours = {
            t1: [t2, None, None],
            t2: [t1, None, None]
        }

    @property
    def triangles(self) -> List[Triangle]:
        # We have to remove everything containing vertices of the supertriangle
        all_points_in_supertriangle = sum([list(x.points) for x in self.supertriangles], [])
        remove_if = lambda t: any(pt in t.points for pt in all_points_in_supertriangle)
        return [t for t in self._neighbours.keys() if not remove_if(t)]

    @staticmethod
    def _create_supertriangles(center: Point2D, radius: float) -> List[Triangle]:
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
        bad_triangles = [bad for bad in self._neighbours.keys() if p in bad.circumcircle]

        boundary = self.do_boundary_walk(p, bad_triangles)

        for bad in bad_triangles:
            del self._neighbours[bad]

        # Add new Triangle Entries
        n = len(boundary)
        for i, b in enumerate(boundary):
            triangle_before, triangle_after = boundary[(i - 1) % n][0], boundary[(i + 1) % n][0]
            # other way around to ensure CCW
            self._neighbours[b.new_triangle] = [b.opposite_triangle, triangle_after, triangle_before]

        # Add new triangles to the opposite side
        for b in boundary:
            if not b.opposite_triangle:
                continue

            for i, neigh in enumerate(self._neighbours[b.opposite_triangle]):
                if neigh and set(b.connecting_edge).issubset(set(neigh.points)):
                    self._neighbours[b.opposite_triangle][i] = b.new_triangle

    def do_boundary_walk(self, p, bad_triangles):
        boundary = []
        current_triangle, i = bad_triangles[0], 0

        while True:
            opposite_triangle = self._neighbours[current_triangle][i]
            if opposite_triangle in bad_triangles:
                i = (self._neighbours[opposite_triangle].index(current_triangle) + 1) % 3
                current_triangle = opposite_triangle
                continue

            # remember, CCW
            edge = (current_triangle.points[(i + 1) % 3], current_triangle.points[(i - 1) % 3])
            self._BoundaryNode(Triangle(p, *edge), edge, opposite_triangle)
            boundary.append(self._BoundaryNode(Triangle(p, *edge), edge, opposite_triangle))
            i = (i + 1) % 3
            if boundary[0].connecting_edge[0] == boundary[-1].connecting_edge[1]:
                return boundary


if __name__ == '__main__':
    n = 50
    min, max = -100, 100
    pts = [(rd.uniform(min, max), rd.uniform(min, max)) for _ in range(n)]
    d = DelaunayTriangulation2D(radius=max + 5)  # Buffer for rounding errors
    for p in pts:
        d.add_point(p)

    plt.rcParams["figure.figsize"] = (5, 10)
    figure, axis = plt.subplots(2)

    axis[0].set_title("meins")
    trianglerinos = d.triangles
    for tri in trianglerinos:
        points = np.ravel(tri.points)
        axis[0].triplot(points[0::2], points[1::2])

    axis[1].set_title("reference implementation")
    d2 = Delaunay2D(radius=max + 5)
    for p in pts:
        d2.addPoint(p)
    coord, tris = d2.exportDT()
    my_tris = [(coord[a], coord[b], coord[c]) for a, b, c in tris]
    for tri in my_tris:
        points = np.ravel(tri)
        axis[1].triplot(points[0::2], points[1::2])
    plt.show()