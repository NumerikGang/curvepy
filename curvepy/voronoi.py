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

    def plot(self, with_delaunay: bool = True, colour: bool = True, with_pts: bool = True):
        """Returns pyplot of the voronoi regions, optionally with it's underlying delaunay triangulation.

        Parameters
        ----------
        with_delaunay: bool
            Whether the underlying delaunay triangulation (it's dual graph) should be plotted as well.
        """
        fig, axis = self.d.plot(color="blue" if colour else "black") if with_delaunay else plt.subplots()
        axis.axis([-self.d.radius / 2 - 1, self.d.radius / 2 + 1, -self.d.radius / 2 - 1, self.d.radius / 2 + 1])
        regions = self.d.voronoi()
        for p in regions:
            polygon = [t.ccc for t in regions[p]]  # Build polygon for each region
            if colour:
                axis.fill(*zip(*polygon), alpha=0.2)  # Plot filled polygon
            axis.plot(*zip(*polygon), color="red" if colour else "black")
        if not with_pts:
            return fig, axis
        pts = self.d.points
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        axis.scatter(xs, ys, c="blue" if colour else "black")

        return fig, axis

if __name__ == "__main__":
    import random as rd
    import matplotlib.pyplot as plt
    random_pt = lambda : rd.random() * 20 - 10
    xs = [(7.121774796988888, -8.699824650605311), (-9.464407819976927, -5.393171652924897),
          (-4.829293057199426, -3.514450505274964), (-0.6033847917949586, 5.538101790932117),
          (1.4496718809949183, 2.5818253142658065), (-8.527285991670192, 2.1147228683351464),
          (-8.547003898367258, -9.18174990032848), (-0.04513200753440216, -5.277461150080267)]
    D = DelaunayTriangulation2D(radius=50)
    for x in xs:
        D.add_point(x)
    V = Voronoi.from_points(xs)
    #fig, ax = D.plot()
    fig, ax = V.plot(with_delaunay=True, colour=False, with_pts=True)
    fig.show()
    plt.show()
