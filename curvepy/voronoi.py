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

    def plot(self, with_delaunay: bool = True):
        """Returns pyplot of the voronoi regions, optionally with it's underlying delaunay triangulation.

        Parameters
        ----------
        with_delaunay: bool
            Whether the underlying delaunay triangulation (it's dual graph) should be plotted as well.
        """
        fig, axis = self.d.plot() if with_delaunay else plt.subplots()
        axis.axis([-self.d.radius / 2 - 1, self.d.radius / 2 + 1, -self.d.radius / 2 - 1, self.d.radius / 2 + 1])
        regions = self.d.voronoi()
        for p in regions:
            polygon = [t.ccc for t in regions[p]]  # Build polygon for each region
            axis.fill(*zip(*polygon), alpha=0.2)  # Plot filled polygon
            axis.plot(*zip(*polygon), color="red")
        return fig, axis

if __name__ == "__main__":
    import random as rd
    import matplotlib.pyplot as plt
    xs = [(-17.020293131128206, 2.9405194178956577), (12.554031229156756, -35.343633947158104),
     (-39.87508424531031, 8.678881100115788), (-31.082623818162414, 34.38790194724238),
     (41.56227694642294, -24.028563842267292), (28.30505259544431, -35.61405485629737),
     (39.89179737670557, -44.069891415458514), (17.15295381699346, -36.67285783599547),
     (23.094432723501242, 6.09226540806489), (-19.91157811695191, -43.20907922114489),
     (31.737876274398744, 48.997404375890056), (4.35991244249864, -11.047448129326519),
     (-46.28525924152926, -40.81554528873802), (-13.417005436539839, -37.76887967720456),
     (2.0212436764692328, -38.37269865281123), (-38.317753337671476, -12.772751223506773),
     (45.36913868205475, -48.285831262475135), (-6.877188950429172, 24.242731309603272),
     (4.23663358436356, -23.273901095685336), (29.061699175478893, -47.96567260813753),
     (21.735689743589717, 6.518162852933685), (29.06152695400715, 36.74718025625427),
     (12.827213728063505, -1.2110607367538861), (32.68125762162184, 29.23663029833253)]
    D = DelaunayTriangulation2D(radius=100)
    for x in xs:
        D.add_point(x)
    V = Voronoi.from_points(xs)
    fig, ax = D.plot()
    fig, ax = V.plot(True)
    fig.show()
