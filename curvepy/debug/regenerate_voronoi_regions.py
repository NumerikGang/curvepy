# flake8: noqa
from collections import deque
from curvepy.delaunay import DelaunayTriangulation2D, TriangleNode
from curvepy.tests.test_voronoi import *

if __name__ == '__main__':
    for seed, mean in [*zip([SEEDS[0]] + SEEDS[2:5] + SEEDS[7:], [MEANS[0]] + MEANS[2:5] + MEANS[7:])]:
        d = DelaunayTriangulation2D(mean, DIAMETER)
        for s in seed:
            d.add_point(s)
        print(",",d.voronoi())