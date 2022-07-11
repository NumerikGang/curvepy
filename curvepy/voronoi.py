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
