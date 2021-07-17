import numpy as np
import random as rd
from functools import cached_property
from typing import List, Tuple, Any, NamedTuple
from dataclasses import dataclass
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
from curvepy.dev.reference_implementation import Delaunay2D
from collections import namedtuple, deque

Point2D = Tuple[float, float]
Edge2D = Tuple[Point2D, Point2D]


class Circle:
    def __init__(self, center: Point2D, radius: float):
        self._center = np.array(center)
        self.radius = radius

    @property
    def center(self):
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

TriangleTuple = namedtuple('TriangleTuple', 'ccw cw pt ccc')

class DelaunayTriangulation2D:
    class _BoundaryNode(NamedTuple):
        new_triangle: Triangle
        connecting_edge: Edge2D
        opposite_triangle: Triangle

    @dataclass
    class _Plotbox:
        min_x: float = float('Inf')
        min_y: float = float('Inf')
        max_x: float = -float('Inf')
        max_y: float = -float('Inf')



    def __init__(self, center: Point2D = (0, 0), radius: float = 500):
        t1, t2 = self._create_supertriangles(center, radius)
        self.supertriangles: List[Triangle] = [t1, t2]
        self._neighbours = {
            t1: [t2, None, None],
            t2: [t1, None, None]
        }
        self._plotbox = self._Plotbox()

    @property
    def triangles(self) -> List[Triangle]:
        # We have to remove everything containing vertices of the supertriangle
        all_points_in_supertriangle = sum([list(x.points) for x in self.supertriangles], [])
        remove_if = lambda t: any(pt in t.points for pt in all_points_in_supertriangle)
        return [t for t in self._neighbours.keys() if not remove_if(t)]

    # TODO: Exclude supertriangle because its inconsistent with triangles property
    @property
    def points(self) -> List[Point2D]:
        ret = set()
        for t in self._neighbours:
            for p in t.points:
                ret.add(p)
        return list(ret)

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

    def change_plotbox(self, p: Point2D):
        if p[0] < self._plotbox.min_x:
            self._plotbox.min_x = p[0]
        if p[0] > self._plotbox.max_x:
            self._plotbox.max_x = p[0]
        if p[1] < self._plotbox.min_y:
            self._plotbox.min_y = p[1]
        if p[1] > self._plotbox.max_y:
            self._plotbox.max_y = p[1]

    def add_point(self, p: Point2D):

        self.change_plotbox(p)

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
            if b.opposite_triangle is None:
                continue

            for i, neigh in enumerate(self._neighbours[b.opposite_triangle]):
                if neigh is not None and set(b.connecting_edge).issubset(set(neigh.points)):
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

    def voronoi(self):
        triangles_containing = {p: [] for p in self.points}

        # Add all triangles to their vertices
        for t in self._neighbours:
            a, b, c = t.points
            triangles_containing[a].append(TriangleTuple(ccw=b, cw=c, pt=a, ccc=t.circumcircle.center))
            triangles_containing[b].append(TriangleTuple(ccw=c, cw=a, pt=b, ccc=t.circumcircle.center))
            triangles_containing[c].append(TriangleTuple(ccw=a, cw=b, pt=c, ccc=t.circumcircle.center))

        regions = {}
        supertriangle_points = sum([[*t.points] for t in self.supertriangles], [])
        for p in triangles_containing:
            # if part of supertriangle, we don't care
            if p in supertriangle_points:
                continue

            regions[p] = self.do_triangle_walk(p, triangles_containing)

        # TODO: Comment me
        delta_x = (self._plotbox.max_x - self._plotbox.min_x) * 0.05
        delta_y = (self._plotbox.max_y - self._plotbox.min_y) * 0.05

        return regions, (delta_x, delta_y)

    def do_triangle_walk(self, p, triangles_containing):
        tris = triangles_containing[p]

        # Starting at "random" point
        ccw = tris[0]
        cw = ccw

        regions = deque([cw])

        # walk ccw until we have no tris left or we have circle
        while True:
            ccw = self._find_neighbour(ccw, tris, False)
            if ccw is None:
                break
            if ccw == regions[-1]:
                return regions
            regions.appendleft(ccw)

        # walk cw to get the remaining tris
        while True:
            cw = self._find_neighbour(cw, tris, True)
            if cw is None:
                return regions
            regions.append(cw)

    def _find_neighbour(self, tri, others, cw):
        if tri is None:
            return None

        is_ccw_neighbour = lambda tri, other: tri.cw == other.ccw
        is_cw_neighbour = lambda tri, other: tri.ccw == other.cw
        is_neighbour = is_cw_neighbour if cw else is_ccw_neighbour

        for t in others:
            if is_neighbour(tri, t):
                return t
        return None

if __name__ == '__main__':
    numSeeds = 24
    diameter = 100
    seeds = np.array([np.array(
        [rd.uniform(-diameter / 2, diameter / 2),
         rd.uniform(-diameter / 2, diameter / 2)]
    )
        for _ in range(numSeeds)
    ])
    center = np.mean(seeds, axis=0)
    dt = Delaunay2D(center, 50 * diameter)
    for s in seeds:
        dt.addPoint(s)

    d = DelaunayTriangulation2D(tuple(center), 50 * diameter)
    for s in seeds:
        d.add_point(tuple(s))

    plt.rcParams["figure.figsize"] = (5, 10)
    fig, axis = plt.subplots(2)

    axis[0].axis([-diameter / 2 - 1, diameter / 2 + 1, -diameter / 2 - 1, diameter / 2 + 1])
    axis[0].set_title("meins")
    regions, (dx, dy) = d.voronoi()
    for p in regions:
        polygon = [t.ccc for t in regions[p]]  # Build polygon for each region
        axis[0].fill(*zip(*polygon), alpha=0.2)  # Plot filled polygon

    # Plot voronoi diagram edges (in red)
    for p in regions:
        polygon = [t.ccc for t in regions[p]]  # Build polygon for each region
        axis[0].plot(*zip(*polygon), color="red")  # Plot polygon edges in red

    for tri in d.triangles:
        x, y, z = tri.points
        points = [*x, *y, *z]
        axis[0].triplot(points[0::2], points[1::2], linestyle='dashed', color="blue")

    # ---

    axis[1].axis([-diameter / 2 - 1, diameter / 2 + 1, -diameter / 2 - 1, diameter / 2 + 1])
    vc, vr = dt.exportVoronoiRegions()
    cx, cy = zip(*seeds)
    dt_tris = dt.exportTriangles()
    axis[1].triplot(Triangulation(cx, cy, dt_tris), linestyle='dashed', color='blue')
    dt_tris = dt.exportTriangles()
    for r in vr:
        polygon = [vc[i] for i in vr[r]]  # Build polygon for each region
        axis[1].plot(*zip(*polygon), color="red")  # Plot polygon edges in red
        axis[1].fill(*zip(*polygon), alpha=0.2)
    plt.show()
