import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Callable


def straight_line(a: np.ndarray, b: np.ndarray, t: float = 0.5, phi: Callable[[np.ndarray], np.ndarray] = lambda x: x) \
        -> np.ndarray:
    return (1 - t) * phi(a) + t * phi(b)


def ratio(left: np.ndarray, col_point: np.ndarray, right: np.ndarray) -> float:
    d1 = (col_point - left).reshape(-1, 1)
    d2 = (right - col_point).reshape(-1, 1)
    print(d1)
    print(d2)
    return (d2 / d1)[0][0]


def ratio_test() -> None:
    p1 = np.array([0, 0, 0])
    p2 = np.array([1, 1, 1])
    p3 = np.array([.4, .4, .4])
    test = ratio(p1, p3, p2)
    print(test)
    print(type(test))


def straight_line_test() -> None:
    t = 0
    fig = plt.figure()
    ax = Axes3D(fig)
    while t <= 1:
        test = straight_line(np.array([0, 0, 0]), np.array([1, 1, 1]), t)
        ax.scatter(test[0], test[1], test[2])
        t += 0.1
    plt.show()


def init() -> None:
    # straight_line_test()
    ratio_test()
    # print(ratio((0, 0), (1, 1), (10, 10)))


if __name__ == "__main__":
    init()

#####################################################################################################
"""
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

"""

""" ????
def ratio(a: tuple, b: tuple, c: tuple) -> float:
    return abs(distance.euclidean(a, b)/distance.euclidean(b, c))
"""
