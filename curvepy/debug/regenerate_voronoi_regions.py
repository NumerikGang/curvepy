from collections import deque
from curvepy.delaunay import DelaunayTriangulation2D, TriangleTuple
from curvepy.tests.test_voronoi_regions import *

if __name__ == '__main__':
    for seed, mean in [*zip(SEEDS, MEANS)]:
        d = DelaunayTriangulation2D(mean, DIAMETER)
        for s in seed:
            d.add_point(s)
        print(d.voronoi())