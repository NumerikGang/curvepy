import numpy as np
import random as rd
from typing import List, NamedTuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import namedtuple, deque

from curvepy.voronoi.types import *
from curvepy.voronoi import Voronoi

class DelaunayTriangulation2D:
    class _BoundaryNode(NamedTuple):
        new_triangle: Triangle
        connecting_edge: Edge2D
        opposite_triangle: Triangle

    _TriangleTuple = namedtuple('TriangleTuple', 'ccw cw pt ccc')

    def __init__(self, center: Point2D = (0, 0), radius: float = 9999):
        t1, t2 = self._create_supertriangles(center, radius)
        self.supertriangles: List[Triangle] = [t1, t2]
        self._neighbours = {
            t1: [t2, None, None],
            t2: [t1, None, None]
        }
        self._plotbox = self._Plotbox()


    @classmethod
    def from_pointlist(cls, points: List[Point2D]):
        pts = np.array([np.array(p) for p in points])
        center = tuple(np.mean(pts, axis=0))
        radius = abs(max(pts[0, :].max() - pts[0, :].min(), pts[1, :].max() - pts[1, :].min()))
        ret = cls(center=center, radius=radius)
        for p in points:
            ret.add_point(p)
        return ret


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

        # To have "infinite" spaces
        radius *= 5

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
            boundary.append(self._BoundaryNode(new_triangle=Triangle(p, *edge),
                                               connecting_edge=edge,
                                               opposite_triangle=opposite_triangle))
            i = (i + 1) % 3
            if boundary[0].connecting_edge[0] == boundary[-1].connecting_edge[1]:
                return boundary


    def voronoi_regions(self):
        triangles_containing = {p: [] for p in self.points}

        # Add all triangles to their vertices
        for t in self._neighbours:
            a, b, c = t.points
            triangles_containing[a].append(self._TriangleTuple(ccw=b, cw=c, pt=a, ccc=t.circumcircle.center))
            triangles_containing[b].append(self._TriangleTuple(ccw=c, cw=a, pt=b, ccc=t.circumcircle.center))
            triangles_containing[c].append(self._TriangleTuple(ccw=a, cw=b, pt=c, ccc=t.circumcircle.center))

        regions = {}
        supertriangle_points = sum([[*t.points] for t in self.supertriangles], [])
        for p in triangles_containing:
            # if part of supertriangle, we don't care
            if p in supertriangle_points:
                continue

            regions[p] = self.do_triangle_walk(p, triangles_containing)

        # delta_x = (self._plotbox.max_x - self._plotbox.min_x) * 0.05
        #delta_y = (self._plotbox.max_y - self._plotbox.min_y) * 0.05

        return regions#, (delta_x, delta_y)

    @property
    def voronoi(self):
        return Voronoi(self)


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
    plt.show()
