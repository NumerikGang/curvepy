import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Callable
import functools


def straight_line_point(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (1 - t) * a + t * b


def straight_line_function(a, b) -> functools.partial:
    return functools.partial(straight_line_point, a, b)


def ratio(left: np.ndarray, col_point: np.ndarray, right: np.ndarray) -> float:
    for i in range(len(left)):
        if left[i] == right[i]:
            continue
        return (col_point[i] - left[i]) / (right[i] - col_point[i])
    return 0


def ratio_test() -> None:
    left = np.array([0, 0, 0])
    right = np.array([1, 1, 1])
    col_point = np.array([.66, .66, .66])
    test = ratio(left, col_point, right)
    print(test)


def straight_line_test() -> None:
    t = 0
    fig = plt.figure()
    ax = Axes3D(fig)
    while t <= 1:
        test = straight_line_point(np.array([0, 0, 0]), np.array([1, 1, 1]), t)
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
