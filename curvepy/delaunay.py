import numpy as np
from functools import cached_property
from typing import List, NamedTuple
import matplotlib.pyplot as plt
from collections import deque

from curvepy.types import Triangle, Edge2D, Point2D, TriangleTuple


class DelaunayTriangulation2D:
    class _BoundaryNode(NamedTuple):
        new_triangle: Triangle
        connecting_edge: Edge2D
        opposite_triangle: Triangle

    def __init__(self, center: Point2D = (0, 0), radius: float = 500):
        t1, t2 = self._create_supertriangles(center, 50 * radius)
        self.radius = radius
        self.supertriangles: List[Triangle] = [t1, t2]
        self._neighbours = {
            t1: [t2, None, None],
            t2: [t1, None, None]
        }

    @classmethod
    def from_points(cls, seeds):
        center = np.mean(seeds, axis=0)
        d = cls(tuple(center))
        for s in seeds:
            d.add_point(s)
        return d

    @cached_property
    def _points_of_supertriangles(self):
        return sum([list(x.points) for x in self.supertriangles], [])

    @property
    def triangles(self) -> List[Triangle]:
        remove_if = lambda t: any(pt in t.points for pt in self._points_of_supertriangles)
        return [t for t in self._neighbours.keys() if not remove_if(t)]

    @property
    def lines(self) -> List[Edge2D]:
        ret = set()
        for lines in [t.lines for t in self.triangles]:
            for (a, b) in lines:
                ret.add((a, b) if a < b else (b, a))
        return list(ret)

    @property
    def points(self) -> List[Point2D]:
        return self._get_points()

    def _get_points(self, exclude_supertriangle=True):
        ret = set([p for t in self._neighbours for p in t.points])
        return list(ret.difference(set(self._points_of_supertriangles))) if exclude_supertriangle else list(ret)

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
        triangles_containing = {p: [] for p in self._get_points(exclude_supertriangle=False)}

        # Add all triangles to their vertices
        for t in self._neighbours:
            a, b, c = t.points
            triangles_containing[a].append(TriangleTuple(ccw=b, cw=c, pt=a, ccc=t.circumcircle.center))
            triangles_containing[b].append(TriangleTuple(ccw=c, cw=a, pt=b, ccc=t.circumcircle.center))
            triangles_containing[c].append(TriangleTuple(ccw=a, cw=b, pt=c, ccc=t.circumcircle.center))

        regions = {p: self.do_triangle_walk(p, triangles_containing) for p in triangles_containing
                   if p not in self._points_of_supertriangles}

        return regions

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
        is_ccw_neighbour = lambda tri, other: tri.cw == other.ccw
        is_cw_neighbour = lambda tri, other: tri.ccw == other.cw
        is_neighbour = is_cw_neighbour if cw else is_ccw_neighbour

        for t in others:
            if is_neighbour(tri, t):
                return t
        return None

    def plot(self, linestyle='dashed', color='blue'):
        fig, axis = plt.subplots()
        axis.axis([-self.radius / 2 - 1, self.radius / 2 + 1, -self.radius / 2 - 1, self.radius / 2 + 1])
        for (a, b) in self.lines:
            axis.plot([a[0], b[0]], [a[1], b[1]], linestyle=linestyle, color=color)
        return fig, axis
