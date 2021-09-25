# flake8: noqa
from curvepy.tests.test_voronoi_regions import *
import numpy as np
from curvepy.voronoi import Voronoi
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from curvepy.debug.Reference_Implementation import Delaunay2D

def plot_it(seeds, means):
    for seed, mean in [*zip(seeds, means)]:
        # our
        d = DelaunayTriangulation2D(mean, DIAMETER)
        for s in seed:
            d.add_point(s)
        v = Voronoi(d)

        # their
        dt = Delaunay2D(np.array(mean),100*50)
        for s in seed:
            dt.addPoint(np.array(s))

        plt.rcParams["figure.figsize"] = (5, 10)
        fig, axis = plt.subplots(2)

        axis[0].axis([-DIAMETER / 2 - 1, DIAMETER / 2 + 1, -DIAMETER / 2 - 1, DIAMETER / 2 + 1])
        axis[0].set_title("meins")
        regions = d.voronoi()
        for p in regions:
            polygon = [t.ccc for t in regions[p]]  # Build polygon for each region
            axis[0].fill(*zip(*polygon), alpha=0.2)  # Plot filled polygon

        # Plot voronoi diagram edges (in red)
        for p in regions:
            polygon = [t.ccc for t in regions[p]]  # Build polygon for each region
            axis[0].plot(*zip(*polygon), color="red")  # Plot polygon edges in red

        for tri in d.triangles:
            x, y, z = tri.points
            points = [*x, *y, *z]
            axis[0].triplot(points[0::2], points[1::2], linestyle='dashed', color="blue")

        # ---

        axis[1].axis([-DIAMETER / 2 - 1, DIAMETER / 2 + 1, -DIAMETER / 2 - 1, DIAMETER / 2 + 1])
        vc, vr = dt.exportVoronoiRegions()
        cx, cy = zip(*seed)
        dt_tris = dt.exportTriangles()
        axis[1].triplot(Triangulation(cx, cy, dt_tris), linestyle='dashed', color='blue')
        dt_tris = dt.exportTriangles()
        for r in vr:
            polygon = [vc[i] for i in vr[r]]  # Build polygon for each region
            axis[1].plot(*zip(*polygon), color="red")  # Plot polygon edges in red
            axis[1].fill(*zip(*polygon), alpha=0.2)
        plt.show()


if __name__ == '__main__':
    plot_it(SEEDS, MEANS)
    #plot_it([SEEDS[1], SEEDS[5], SEEDS[6]], [MEANS[1], MEANS[2], MEANS[3]])