"""This file has an implementation of a Delaunay Triangulation Algorithm.

This Algorithm is also used for generating voronoi regions as they are the dual graph of a delaunay triangulation.

Here is an excerpt from the English wikipedia entry:
In mathematics and computational geometry, a Delaunay triangulation (also known as a Delone triangulation) for a
given set P of discrete points in a general position is a triangulation DT(P) such that no point in P is inside
the circumcircle of any triangle in DT(P).

Our algorithm is solely based on that definition.

Here is a Sketch of the algorithm (and it's correctness proof):

1. [The Initialisation]
When starting a new DelaunayTriangulation, we don't want to handle some edge cases for the first few points.
We always want a complete triangulation. Thus, we define 2 "supertriangles" in which we require all points to lie in.
Since we know the accepted radius and center, we can create those 2 triangles as following:
x────────────────────────────────┐
xx                               │
│ xx                             │
│   xxx              T2          │
│     xxxx                       │
│        xxx                     │
│          xxx                   │
│            xxxx                │
│               xxxx             │
│                  xxx           │
│       T1           xxxx        │
│                       xxx      │
│                         xxx    │
│                           xxx  │
│                             xxx│
└────────────────────────────────┴
Note that a triangle is defined by a complete graph of 3 points (i.e. they are all 3 connected with each other).
Since we only get points, our algorithm creates all used triangles. Each triangle is defined counter clock wise (CCW),
which makes traversing way easier.

Also, we initialize our "neighbour structure". A neighbouring triangle is defined as sharing exactly 1 edge.
(Obviously it can't be 2 edges. 0 edges are not neighbouring and 3 edges are the exact same triangle.)
The neighbour structure is a dict (great choice since tuples are hashable and O(1)) with the triangles (3-tuples) as
the keys and a 3-list of it's neighbours as it's values.
(The length is always 3, mutability just makes everything much more smooth as we continually add points).
Very important: Never forget that everything in here is counter clock wise (CCW)!!!! This makes traversing more fun.

2. [Finding the bad triangles]
When adding a point, we first check the following: When we add this point, which Triangles become invalid?
A triangle is invalid or bad iff there is another point in its circumcircle. But we don't have to check every point!
Since all points but the new one are unchanged, a triangle becomes bad iff the new point is in its circumcircle.

Before going to step 3, let us think a bit further. A triangulation is defined such that EVERY point is contained in the
triangulation and that every space inbetween lines is triangular. So we have to connect the point __somehow__.
We also know by definition that we can't have any bad triangles in the Delaunay triangulation. We ALSO know that every
set of points has a uniquely defined delaunay triangulation!

This does not leave many options (which is great).
So every correct iterative algorithm (i.e. adding one point at atime) has to remove every new bad triangle.
Every correct algorithm also has to only have triangular spaces.
W.l.o.g. let this be our delaunay triangulation (excluding supertriangles) after removing every bad triangle.
          X
        -------
       --     -----
     --           ----
   ---               ----
 ---                    ------
 X                           --X
 -                             -
 -                             -
 -                             -
 -              X              -
 -                             -
 -                             -
 X---                          -
    --                       --X
     ----                  ---
        ----              --
            X------------X
One can trivially see that there is only one way to connect all points in such a way that we only have triangles.
Every remaining point of a bad triangle has to be connected to the new point! Otherwise there will be a space with more
than 3 edges.

So we know that after adding a new point there is exactly one way to alter the edges such that the delaunay
circumcircle condition is met. Inductively (by adding all points one by one) we can see that there is only one way we
can create the edges such that the delaunay circumcircle condition is met. Additionally, we know that there has to be
at least one way because a unique Delaunay Triangulation always exist.
Since we have a lower bound of 1 and an upper bound of 1 we can see that the algorithm is correct iff it
removes all bad triangles and connects the new point to each point of a removed triangle.

A naive implementation should be trivial by now. Let's continue with the optimized algorithm.

3. [The Boundary-Walk]
The boundary walk is needed as an optimization. Let me first explain why it is needed.

Let's say that after step 2 we have n distinct points which are part of a bad triangle. Now create new triangles, we
need to know which 2 still have a connecting edge (i.e. are still neighbours). Naively, this alone would take O(n^2),
which is too slow, especially since we set all the triangles ourselves.
So instead, we just remember and update the neighbours every time we change the triangle structure. The Boundary-Walk
uses the neighbour structure (read-only) in order to find all bad points which are still neighbours.

Before we actually go into traversing, we also have to realize a few things:

- Quoting the Wikipedia article:
"Delaunay triangulations maximize the minimum angle of all the angles of the triangles in the triangulation;
they tend to avoid sliver triangles."
(Definition sliver triangle: "A triangle with one or two extremely acute angles, hence a long/thin shape, which has
undesirable properties during some interpolation or rasterization processes.")
Let's inductively assume that the algorithm is correct after (n-1) steps. Let's now add the n-th point.
We can know see that all vertices of the subgraph of bad triangles have to be connected. If this would not be the case,
i.e. we have a degenerated sliver triangle far away, it would have had another point


"""
from __future__ import annotations  # Needed until Py3.10, see PEP 563

from collections import deque
from functools import cached_property
from typing import Deque, Dict, List, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from curvepy.types import (Edge2D, Point2D, TriangleNode, TupleTriangle,
                           VoronoiRegions2D)


class DelaunayTriangulation2D:
    class _BoundaryNode(NamedTuple):
        new_triangle: TupleTriangle
        connecting_edge: Edge2D
        opposite_triangle: TupleTriangle

    def __init__(self, center: Point2D = (0, 0), radius: float = 500):
        t1, t2 = self._create_supertriangles(center, 50 * radius)
        self.radius = radius
        self.supertriangles: List[TupleTriangle] = [t1, t2]
        self._neighbours = {
            t1: [t2, None, None],
            t2: [t1, None, None]
        }

    @classmethod
    def from_points(cls, seeds: np.ndarray) -> DelaunayTriangulation2D:
        center = np.mean(seeds, axis=0)
        d = cls(tuple(center))
        for s in seeds:
            d.add_point(s)
        return d

    @cached_property
    def _points_of_supertriangles(self) -> List[Point2D]:
        return sum([list(x.points) for x in self.supertriangles], [])

    @property
    def triangles(self) -> List[TupleTriangle]:
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

    def _get_points(self, exclude_supertriangle: bool = True) -> List[Point2D]:
        ret = {p for t in self._neighbours for p in t.points}
        return list(ret.difference(set(self._points_of_supertriangles))) if exclude_supertriangle else list(ret)

    @staticmethod
    def _create_supertriangles(center: Point2D, radius: float) -> List[TupleTriangle]:
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
        lower_triangle = TupleTriangle(lower_left, lower_right, upper_left)
        upper_triangle = TupleTriangle(upper_right, upper_left, lower_right)
        return [lower_triangle, upper_triangle]

    def add_point(self, p: Point2D):

        bad_triangles = [bad for bad in self._neighbours.keys() if p in bad.circumcircle]

        boundary = self.do_boundary_walk(p, bad_triangles)

        for bad in bad_triangles:
            del self._neighbours[bad]

        # Add new TupleTriangle Entries
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

    def do_boundary_walk(self, p: Point2D, bad_triangles: List[TupleTriangle]) -> List[
        DelaunayTriangulation2D._BoundaryNode]:
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
            self._BoundaryNode(TupleTriangle(p, *edge), edge, opposite_triangle)
            boundary.append(self._BoundaryNode(TupleTriangle(p, *edge), edge, opposite_triangle))
            i = (i + 1) % 3
            if boundary[0].connecting_edge[0] == boundary[-1].connecting_edge[1]:
                return boundary

    def voronoi(self) -> VoronoiRegions2D:
        triangles_containing = {p: [] for p in self._get_points(exclude_supertriangle=False)}

        # Add all triangles to their vertices
        for t in self._neighbours:
            a, b, c = t.points
            triangles_containing[a].append(TriangleNode(ccw=b, cw=c, pt=a, ccc=t.circumcircle.center))
            triangles_containing[b].append(TriangleNode(ccw=c, cw=a, pt=b, ccc=t.circumcircle.center))
            triangles_containing[c].append(TriangleNode(ccw=a, cw=b, pt=c, ccc=t.circumcircle.center))

        regions = {p: self.do_triangle_walk(p, triangles_containing) for p in triangles_containing
                   if p not in self._points_of_supertriangles}

        return regions

    def do_triangle_walk(self, p: Point2D, triangles_containing: Dict[Point2D, List[TriangleNode]]) -> Deque[
        TriangleNode]:
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

    @staticmethod
    def _find_neighbour(tri: TriangleNode, others: List[TriangleNode], go_cw: bool) -> Optional[TriangleNode]:
        is_ccw_neighbour = lambda tri, other: tri.cw == other.ccw
        is_cw_neighbour = lambda tri, other: tri.ccw == other.cw
        is_neighbour = is_cw_neighbour if go_cw else is_ccw_neighbour

        for t in others:
            if is_neighbour(tri, t):
                return t
        return None

    def plot(self, linestyle: str = 'dashed', color: str = 'blue'):
        fig, axis = plt.subplots()
        axis.axis([-self.radius / 2 - 1, self.radius / 2 + 1, -self.radius / 2 - 1, self.radius / 2 + 1])
        for (a, b) in self.lines:
            axis.plot([a[0], b[0]], [a[1], b[1]], linestyle=linestyle, color=color)
        return fig, axis
