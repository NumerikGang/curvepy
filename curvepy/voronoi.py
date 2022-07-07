"""This file has an implementation of Voronoi regions.

Those regions are not directly computed. We use the property that Voronoi regions are just the dual graph of
Delaunay triangulations. See `delaunay.py` or the accompanying paper for a more detailed explanation.

"""
from __future__ import annotations  # Needed until Py3.10, see PEP 563

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from curvepy.delaunay import DelaunayTriangulation2D
from curvepy.types import Point2D, VoronoiRegions2D


class Voronoi:
    """A wrapper around Delaunay triangulations to get Voronoi regions.

    Attributes
    ----------

    d: DelaunayTriangulation2D
        The Delaunay Triangulation used for generating the Voronoi Regions as it's dual graph.
    """

    def __init__(self, d: Optional[DelaunayTriangulation2D] = None):
        self.d = DelaunayTriangulation2D() if d is None else d

    @classmethod
    def from_points(cls, seeds: List[Point2D]) -> Voronoi:
        """Class method to create the Voronoi Regions directly from an numpy array of points.

        Parameters
        ----------
        seeds: np.ndarray
            The points from which a voronoi region should be created.

        Returns
        -------
        Voronoi
            A voronoi region of the given points.
        """
        center = np.mean(np.array(seeds), axis=0)
        d = DelaunayTriangulation2D(tuple(center))
        for s in seeds:
            d.add_point(s)
        return cls(d)

    @property
    def points(self) -> List[Point2D]:
        """Property to get the points of the voronoi class.

        Returns
        -------
        List[Point2D]
            The list of all points defining the voronoi tiles.
        """
        return self.d.points

    @property
    def regions(self) -> VoronoiRegions2D:
        """Property to get the regions of the voronoi class.

        Returns
        -------
        VoronoiRegions2D
            The internal datastructure representing the regions.
        """
        return self.d.voronoi()

    def plot(self, with_delaunay: bool = True, color: bool = True, with_pts: bool = True,
             with_circumcircle: bool = False):
        """Returns pyplot of the voronoi regions, optionally with it's underlying delaunay triangulation.

        Parameters
        ----------
        with_delaunay: bool
            Whether the underlying delaunay triangulation (it's dual graph) should be plotted as well.
        color: bool
            Whether to use color or only black and white.
        with_pts: bool
            Whether the generating points should be plotted as well.
        """
        fig, axis = self.d.plot(color="blue" if color else "black",
                                with_circumcircle=with_circumcircle) if with_delaunay else plt.subplots()
        axis.axis([-self.d.radius / 2 - 1, self.d.radius / 2 + 1, -self.d.radius / 2 - 1, self.d.radius / 2 + 1])
        regions = self.d.voronoi()
        for p in regions:
            polygon = [t.ccc for t in regions[p]]  # Build polygon for each region
            if color:
                axis.fill(*zip(*polygon), alpha=0.2)  # Plot filled polygon
            axis.plot(*zip(*polygon), color="red" if color else "black")
        if not with_pts:
            return fig, axis
        pts = self.d.points
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        axis.scatter(xs, ys, c="blue" if color else "black")

        return fig, axis


if __name__ == "__main__":
    import random as rd
    import matplotlib.pyplot as plt

    random_pt = lambda: rd.random() * 20 - 10
    """
    xs = [(7.121774796988888, -8.699824650605311), (-9.464407819976927, -5.393171652924897),
          (-4.829293057199426, -3.514450505274964), (-0.6033847917949586, 5.538101790932117),
          (1.4496718809949183, 2.5818253142658065), (-8.527285991670192, 2.1147228683351464),
          (-8.547003898367258, -9.18174990032848), (-0.04513200753440216, -5.277461150080267)]
    """
    xs = [
        (9.4542805979508, 17.8542640288422),
        (7.212532538692, 12.5555867978669),
        (9.4135215423279, 7.4199457893831),
        (10.6770522666374, 11.1290198510658),
        (11.2884381009808, 15.4494797470919),
        (14.6306806620575, 18.9955175862831),
        (15.527379885761, 15.0826482464859),
        (15.7719342194984, 11.1290198510658),
        (15.0790302739093, 6.8493190106627),
        (16, 4),
        (21.9265516185543, 5.3004748969929),
        (19.154935836198, 7.460704845006),
        (20.3369484492617, 15.0011301352401),
        (20.948334283605, 19.2808309756433),
        (26.369288681449, 17.6097096951049),
        (24.942721734648, 12.6778639647355),
        (31.3418934674413, 14.308226189651),
        (31.3826525230642, 10.354597794231),
        (26.8991564045466, 6.5232465656796)
    ]
    new_point = (19.154935836198, 11.7404056854091)
    D = DelaunayTriangulation2D(radius=50)
    for x in xs:
        D.add_point(x)
    V = Voronoi.from_points(xs)
    # FIRST PLOT: The n-1 point triangulation
    fig, ax = D.plot(color="black", with_circumcircle=False, linestyle="solid")
    fig.show()
    # SECOND PLOT: Delaunay Triangulation + the new point in another colour
    fig, ax = D.plot(color="black", with_circumcircle=False, linestyle="solid")
    ax.scatter([new_point[0]], [new_point[1]], c="red")
    fig.show()
    # THIRD PLOT: Circumcircles
    fig, ax = D.plot(color="black", with_circumcircle=True, linestyle="solid")
    ax.scatter([new_point[0]], [new_point[1]], c="red")
    fig.show()
    # FOURTH PLOT: Invalid Circumcircles
    fig, ax = D.plot(color="black", with_circumcircle=False, linestyle="solid")
    ax.scatter([new_point[0]], [new_point[1]], c="red")
    for tri in D.triangles:
        if new_point in tri.circumcircle:
            center, radius = tri.circumcircle.center, tri.circumcircle.radius
            circ = plt.Circle(center, radius, fill=False, color="red")
            ax.add_patch(circ)
    fig.show()
    # FIFTH PLOT: Hole (just photoshop)
    # SIXTH PLOT: RECONNECTED HOLE
    D.add_point(new_point)
    fig, ax = D.plot(color="black", with_circumcircle=False, linestyle="solid")
    fig.show()
    # SIXTH PLOT: RECONNECTED HOLE WITH CIRCUMCIRCLES
    fig, ax = D.plot(color="black", with_circumcircle=True, linestyle="solid")
    fig.show()
    # SEVENTH PLOT: JUST THE NEW TRIANGLES
    fig, ax = D.plot(color="black", with_circumcircle=False, linestyle="solid")
    for tri in D.triangles:
        if new_point in tri._points:
            center, radius = tri.circumcircle.center, tri.circumcircle.radius
            circ = plt.Circle(center, radius, fill=False, color="blue")
            ax.add_patch(circ)
    fig.show()

    plt.show()
