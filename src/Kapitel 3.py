import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Callable

def straight_line(a: tuple, b: tuple, scale: float=0) -> Tuple[list, list, list]:
    t = -abs(scale)
    xs = []
    ys = []
    zs = []
    while t <= abs(scale):
        xs.append((1-t)*a[0]+t*b[0])
        ys.append((1-t)*a[1]+t*b[1])
        if len(a) == 3 and len(b) == 3:
            zs.append((1-t)*a[2]+t*b[2])
        t += 0.01
    return xs, ys, zs

def straight_line_test() -> None:
    xs, ys, zs = straight_line((0, 0, 0), (100, 100, 42), -2)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs, ys, zs)
    plt.show()

def init() -> None:
    straight_line_test()

if __name__ == "__main__":
    init()
