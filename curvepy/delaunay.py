"""This file has an implementation of a 2D Delaunay Triangulation Algorithm.

It uses the incremental algorithm first developed by Green and Sibson in their 1976 paper
"Computing Dirichlet tessellations in the plane"
which was later generalized to n-dimensions independently by both Bowyer in 1981 with his paper
"Computing Dirichlet tessellations"
and Watson in his 1981 paper
"Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes".

We changed the algorithm slightly in 2 ways:

1. It uses a simpler, naive way of finding all bad neighbours after adding a new point.

The algorithm's time complexity is O(n) per addition, thus O(n^2) in total.

This can further be reduced by finding the tile containing the new point via a graph walk starting at the middle.
It is claimed by Bowyer that this reduces the addition time complexity to O(n^(1/d)), where d describes the dimension,
although no rigorous analysis is known (according to Fortune's 1995 paper "VORONOI DIAGRAMS AND DELAUNAY
TRIANGULATIONS")

2. In order to compute the Delaunay Triangulation Green and Sibson required an initial "Window", in which all points
are contained. They just required it to be convex, although it is common to make it rectangular or even square.

We choose to use a so called "Supertriangle", which is a big square with a diagonal line dividing it into 2 triangles.

This makes things a lot easier as we start with an valid triangulation before adding the first point, which means that
we have no special cases for the first addition.

The algorithm works as follows:

After adding a point, we detect all triangles which are now invalid (i.e. their circumcircle contains the new point).
Those triangles are always continguous, which makes it easier to traverse them (See Rebay's 1991 paper "Efficient
Unstructured Mesh Generation by Means of Delaunay Triangulation and Bowyer-Watson Algorithm")

We also know of the existance of a Delaunay Triangulation for each set of points. Thus, the only possible solution has
to be to remove all invalid (bad) triangles and connect those triangles to the new point. Green Sibson proved that this
always results in a valid Delaunay Triangulation.

This algorithm is inspired by pyDelaunay by Benny Cheung.

For more information, read the accompanying paper about curvepy.
"""
from __future__ import annotations  # Needed until Py3.10, see PEP 563

from collections import deque
from functools import cached_property
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from curvepy.types import (Edge2D, Point2D, TriangleNode, TupleTriangle,
                           VoronoiRegions2D)


class DelaunayTriangulation2D:
    """The class which implements the 2D delaunay algorithm proposed by Green and Sibson.

    See the top of the file for a more detailed description.
    """
    class _BoundaryNode(NamedTuple):
        """Helper class.
        Those Nodes aren't actually saved, they are just used to make the traversal more readable and manageable.
        """
        new_triangle: TupleTriangle
        connecting_edge: Edge2D
        opposite_triangle: TupleTriangle

    def __init__(self, center: Point2D = (0, 0), radius: float = 500):
        """Constructor.

        Creates the supertriangle (i.e. Window) and our mesh neighbour datastructure.

        Parameters
        ----------
        center: Point2D
            The center of the Window in which all future points have to coincide.
        radius: float
            The radius (i.e. half diameter) of the square Window.
        """
        t1, t2 = self._create_supertriangles(center, 50 * radius)
        self.radius = radius
        self.supertriangles: List[TupleTriangle] = [t1, t2]
        self._neighbours = {
            t1: [t2, None, None],
            t2: [t1, None, None]
        }

    @classmethod
    def from_points(cls, seeds: np.ndarray) -> DelaunayTriangulation2D:
        """Helper method to run the incremental algorithm on a fixed set of points.

        Determines the needed window size itself.

        Parameters
        ----------
        seeds: np.ndarray
            The 2D points which will be the vertices of the Delaunay Triangulation.

        Returns
        -------
        DelaunayTriangulation2D
            The already computed triangulation of the seeds given.
        """
        center = np.mean(seeds, axis=0)
        d = cls(tuple(center))
        for s in seeds:
            d.add_point(s)
        return d

    @cached_property
    def _points_of_supertriangles(self) -> List[Point2D]:
        """The points which define the supertriangle as a cached property.

        Those are needed to be subtracted from the result since they are not formally part of the triangulation.

        Returns
        -------
        List[Point2D]
            The points which the supertriangle is made of.
        """
        return sum([list(x.points) for x in self.supertriangles], [])

    @property
    def triangles(self) -> List[TupleTriangle]:
        """All triangles of the Delaunay Triangulation, i.e. every triangle except the supertriangles.

        Returns
        -------
        List[TupleTriangle]
            The list of all triangles.
        """
        remove_if = lambda t: any(pt in t.points for pt in self._points_of_supertriangles)
        return [t for t in self._neighbours.keys() if not remove_if(t)]

    @property
    def lines(self) -> List[Edge2D]:
        """All lines of the Delaunay Triangulation, except the lines of the supertriangle.

        Returns
        -------
        List[Edge2D]
            All lines of the Delaunay Triangulation.
        """
        ret = set()
        for lines in [t.lines for t in self.triangles]:
            for (a, b) in lines:
                ret.add((a, b) if a < b else (b, a))
        return list(ret)

    @property
    def points(self) -> List[Point2D]:
        """All points of the Delaunay Triangulation, except the points of the supertriangle.

        Returns
        -------
        List[Point2D]
            All points of the Delaunay Triangulation.
        """
        return self._get_points()

    def _get_points(self, exclude_supertriangle: bool = True) -> List[Point2D]:
        """Helper method to get all points, used internally only.

        For external usage, use the points property.

        Parameters
        ----------
        exclude_supertriangle: bool
            Whether the supertriangle points should be excluded first.

        Returns
        -------
        List[Point2D]
            All points of the Delaunay Triangulation, with or without the supertriangle.
        """
        ret = {p for t in self._neighbours for p in t.points}
        return list(ret.difference(set(self._points_of_supertriangles))) if exclude_supertriangle else list(ret)

    @staticmethod
    def _create_supertriangles(center: Point2D, radius: float) -> List[TupleTriangle]:
        """This method creates the Supertriangle (Window) in which all points need to coincide.

        It looks a little bit like this:
        ```
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
        ```

        Parameters
        ----------
        center: Point2D
            The center of the square.
        radius: float
            half the length of the square.

        Returns
        -------
        List[TupleTriangle]
            The Supertriangle.
        """
        center = np.array(center)
        base_rectangle = [center + radius * np.array([i, j]) for i, j in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        lower_left, lower_right, upper_right, upper_left = [tuple(x) for x in base_rectangle]
        lower_triangle = TupleTriangle(lower_left, lower_right, upper_left)
        upper_triangle = TupleTriangle(upper_right, upper_left, lower_right)
        return [lower_triangle, upper_triangle]

    def add_point(self, p: Point2D):
        """The main method of adding a new point.

        See the description at the top of this file, the accompanying literature or Green and Sibson's paper
        for more information.

        Firstly, we find all newly invalid triangles. We know by induction (and basic reasoning) that our Delaunay
        triangulation was valid before adding the new point. (the empty triangulation is valid; this is the base case).
        Thus, we only need to find the triangles which circumcircle contains the new point. We remove them.

        Secondly, we find the boundary. By boundary, we mean the edge path (which is a circle) that connected the bad
        triangles to a valid triangle. Rebay 1991 showed that all bad triangles are connected to each other and build
        a cavity. For a more detailed description, read the `do_boundary_walk` method docstring or our paper.

        Thirdly, we walk along the boundary and add our new triangles (which are composed of the boundary edge and both
        edge vertices connected to our new point) to our internal mesh datastructure.

        Lastly, we add our new triangles to the opposite triangle (the one which was valid) of each boundary node.

        Parameters
        ----------
        p: Point2D
            The point to be added
        """
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
        """The boundary walk which finds walks along the cavity of the removed (bad) triangles.

        This works as follows: We start at any triangle. We choose any edge of it.
        Then we rotate along the edges counter clock wise (CCW) and go to the triangle that is connected to our
        triangle though this edge.
        One can see that though this motion we go in a more or less specific direction.
        Eventually (in reality it's pretty fast, Green Sibson shows that though the Euler-Poincare formula) we reach
        an edge that is connected to a valid triangle (i.e. a triangle which circumcircle contains only it's 3 points).

        We call this edge-path of valid-to-invalid triangles our boundary.

        Now, by moving counter clock wise, we stay along the boundary. We record not only every boundary edge, but also
        to which boundary edges it is connected to. This allows for linear time insertion of the new triangles since
        we do not have to search the neighbour (which, combinatorically, would result in O(n^2) where n is the number of
        boundary edges).

        Note: the neighbours do not have to be saved explicitly. Since it is a list, and we add it one by one while
        walking along the neighbourship, it is implicitly ordered.

        We stop once we meet our first boundary triangle again (we ran a circle).

        See Rebay 1991 or our accompanying paper for why this is always contiguous.

        Parameters
        ----------
        p: Point2D
            The Point that was added last.
        bad_triangles: List[TupleTriangle]
            The list of bad triangles, which became invalid after adding p.

        Returns
        -------
        List[DelaunayTriangulation2D._BoundaryNode]:
            The list of new triangles in a helper structure for easier O(1) addition to our internal triangle mesh.
        """
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
        """Returns the dual graph of the Delaunay Triangulation.

        The voronoi regions are sometimes also called the Dirichlet Tessellations.

        We first built a neighboring dict of the points and their triangles they are part of.
        Since their circumcircle centers (ccc) are the vertices of a voronoi tile, and the triangle points are the
        points that define a voronoi region, we know that each tile is only defined by the triangles that contain the
        tile point.

        This means that every tile is defined by the triangles which create a graphtheoretical circle around each
        point (or only a partial circle if said point is at the edge. This also results in the tile being infinitely
        large). That is meant by the duality.

        Then we do the triangle walk, see `do_triangle_walk` or our paper for more information.

        This should run in O(n), since the number of vertices defining the border of any voronoi tile are not defined
        by the number of points in the voronoi region. This follows from the optimality criterion of Delaunay
        Triangulations, see Fortune 1995 Theorem 3.1 for a proof.


        Returns
        -------
        VoronoiRegions2D
            The Voronoi regions.
        """
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
        """Helper method to extract the voronoi regions from the Delaunay Triangulation.

        We know, since the voronoi regions are defined the underlying Delaunay Triangulation (to be more precise: each
        triangle point defines a voronoi regions), that our new voronoi region is only defined by the triangles that
        contain said point. (Otherwise there would be another point nearer to the tile; that would contradict
        the whole definition of a voronoi region).

        One can visualize that though a circle. We just extract all triangles that contain a single point, p.
        This point p has to be in the middle, i.e. it has to be the only point not being at the border of our extracted
        subgraph.

        If our voronoi tile is finite, (i.e. the point is not at the border of our Delaunay Triangulation) this is easy
        since it has to be completely enclosed by other voronoi regions.
        Then we can just start at a random point and walk a circle along the neighbours.
        When it is enclosed, we do exactly that.

        When it is not enclosed (i.e. the point IS at the border of our Delaunay Triangulation),
        there is no circle (since the outgoing edges are infinitely long).
        So first, we pick any vertex of the voronoi tile.
        Then, we go first counter clock wise (ccw) until there is nothing left anymore.
        Then, we go clock wise (cw) to pick up the rest.
        One can easily see that this is the most efficient way and that we do not look at any Triangle twice.

        Parameters
        ----------
        p: Point2D
            A point that defines it
        triangles_containing: Dict[Point2D, List[TriangleNode]]
            A dictionary of each point and the triangles which contain this point as a vertex.

        Returns
        -------
        Deque[TriangleNode]
            The triangles which circumcircles define the voronoi regions of p.
            which is the point defining the voronoi tile since
            it's the dual graph).
        """
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
        """Helper method to find the neighbour of a triangle.

        More precisely, we walk along the "triangle circle" defined by all triangles that contain a certain point.
        Geometrically, said point defines the graphtheoretical center of those triangles.

        To be explicit, the triangles themselves are the vertices of the graph and edges are defined between 2
        triangles if and only if they are connected. This is where the whole notion of duality comes from.

        So, while walking along that "triangle circle", we want to find the next triangle neighbour to walk to.
        If `go_cw` is true, then go clock wise, otherwise go counter-clock wise.

        Parameters
        ----------
        tri: TriangleNode
            The current triangle which neighbour (remember, dual graph) we are trying to find.
        others: List[TriangleNode]
            All Triangles containing a specific point.
        go_cw: bool
            Whether we want to have the clock wise (cw) or counter clock wise (ccw) neighbour of said triangle.
        Returns
        -------
        Optional[TriangleNode]
            The neighbouring triangle of specified direction if that triangle contains our wanted point p, None
            otherwise.
        """
        is_ccw_neighbour = lambda tri, other: tri.cw == other.ccw
        is_cw_neighbour = lambda tri, other: tri.ccw == other.cw
        is_neighbour = is_cw_neighbour if go_cw else is_ccw_neighbour

        for t in others:
            if is_neighbour(tri, t):
                return t
        return None

    def plot(self, linestyle: str = 'dashed', color: str = 'blue', with_circumcircle: bool = False) -> Tuple[plt.Figure, plt.Axes]:
        """A helper method to plot the Delaunay Triangulation of our current internal state.

        Parameters
        ----------
        linestyle: str
            The linestyle argument of pyplot's plot function
        color: str
            The colour argument of pyplot's plot function
        with_circumcircle: bool
            Whether the triangle circumcircles should be plotted as well.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The subplot.
        """
        fig, axis = plt.subplots()
        axis.axis([-self.radius / 2 - 1, self.radius / 2 + 1, -self.radius / 2 - 1, self.radius / 2 + 1])
        for (a, b) in self.lines:
            axis.plot([a[0], b[0]], [a[1], b[1]], linestyle=linestyle, color=color)
        if not with_circumcircle:
            return fig, axis
        for tri in self.triangles:
            center, radius = tri.circumcircle.center, tri.circumcircle.radius
            circ = plt.Circle(center, radius, fill=False, color=color)
            axis.add_patch(circ)
        return fig, axis
