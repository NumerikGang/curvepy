from typing import Dict, Deque, List

from curvepy.voronoi.types import Point2D, Triangle
from curvepy.voronoi.delaunay_triangulation import DelaunayTriangulation2D


class Voronoi:
    def __init__(self, delaunay):
        self.delaunay = delaunay
        self.regions = self.delaunay.voronoi_regions()

    @classmethod
    def create_from_points(cls, pts: List[Point2D]):
        d = DelaunayTriangulation2D.from_pointlist(pts)
        cls(delaunay=d)
        return cls

    @classmethod
    def create_from_delaunay(cls, d: DelaunayTriangulation2D):
        cls(delaunay=d)
        return cls

    def add_point(self, p):
        self.delaunay.add_point(p)
        self.regions = self.delaunay.voronoi_regions()

    def plot(self, with_delaunay=False):
        ...