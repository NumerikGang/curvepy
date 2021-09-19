from curvepy.tests.test_voronoi_regions import *

from curvepy.voronoi import Voronoi
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for seed, mean in [*zip(SEEDS, MEANS)]:
        d = DelaunayTriangulation2D(mean, DIAMETER)
        for s in seed:
            d.add_point(s)
        v = Voronoi(d)
        _, axis = v.plot()
        plt.show()