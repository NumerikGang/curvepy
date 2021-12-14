from __future__ import annotations  # Needed until Py3.10, see PEP 563
from typing import List, Optional
from curvepy.delaunay import DelaunayTriangulation2D
from curvepy.types import Point2D, VoronoiRegions2D
import numpy as np
import matplotlib.pyplot as plt


class Voronoi:
    def __init__(self, d: Optional[DelaunayTriangulation2D] = None):
        self.d = DelaunayTriangulation2D() if d is None else d

    @classmethod
    def from_points(cls, seeds: np.ndarray) -> Voronoi:
        center = np.mean(seeds, axis=0)
        d = DelaunayTriangulation2D(tuple(center))
        for s in seeds:
            d.add_point(s)
        return cls(d)

    @property
    def points(self) -> List[Point2D]:
        return self.d.points

    @property
    def regions(self) -> VoronoiRegions2D:
        return self.d.voronoi()

    def plot(self, with_delauny: bool = True):
        fig, axis = self.d.plot() if with_delauny else plt.subplots()
        axis.axis([-self.d.radius / 2 - 1, self.d.radius / 2 + 1, -self.d.radius / 2 - 1, self.d.radius / 2 + 1])
        regions = self.d.voronoi()
        for p in regions:
            polygon = [t.ccc for t in regions[p]]  # Build polygon for each region
            axis.fill(*zip(*polygon), alpha=0.2)  # Plot filled polygon
            axis.plot(*zip(*polygon), color="red")
        return fig, axis
