from curvepy.utilities import *
from curvepy.tests.data.data_bezier_curve import INTERSECT_X_AXIS
from curvepy.bezier_curve import BezierCurveApproximation
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
import scipy.special as scp

INTERVAL = (-20, 20)
TESTCASES = 50
INT_BEZ = (5, 25)
HURENSOHN = 8


def gen_polygon_tests_2D():
    print("[")

    print("]")


def lmao():
    bss = []
    for _ in range(10):
        n = random.randint(*INT_BEZ)
        bs = [(random.random() * 20 - 5, random.random() * 20 - 5) for _ in range(n)]
        xs = [p[0] for p in bs]
        ys = [p[1] for p in bs]
        bss.append([xs, ys])
        """
        bc = BezierCurveApproximation(np.array(bs))
        print(f"{bs},")
        plt.plot(*bc.curve)
        plt.show()
        """
    print([(lol, None) for lol in bss])
    fig, ax = plt.subplots(2, 5)
    for i in range(len(bss)):
        bc = BezierCurveApproximation(np.array(bss[i]))
        ax[int(i >= len(bss) // 2), i % (len(bss) // 2)].plot(*bc.curve)
        ax[int(i >= len(bss) // 2), i % (len(bss) // 2)].plot(np.linspace(-20, 20, 10), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    plt.show()


def _test_this_shit():
    ret = []
    for bs, _ in INTERSECT_X_AXIS:
        ret.append([*BezierCurveApproximation.intersect_with_x_axis(np.array(bs))])

    print(ret)


def llllll():
    fig, ax = plt.subplots(2, 5)
    bss = INTERSECT_X_AXIS
    for i in range(len(INTERSECT_X_AXIS)):
        bc = BezierCurveApproximation(np.array(bss[i][0]))
        ax[int(i >= len(bss) // 2), i % (len(bss) // 2)].plot(*bc.curve)
        ax[int(i >= len(bss) // 2), i % (len(bss) // 2)].plot(np.linspace(-20, 20, 10), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    plt.show()


def pppppppp_deine_mutter(n=8):
    ret = []
    for _ in range(TESTCASES):
        xss = [random.random() * 40 - 20 for _ in range(n)]
        ret.append(xss)
    print(ret)


def poly_gen():
    for _ in range(TESTCASES):
        n = random.randint(1, 50)
        xs = [random.random() * 40 - 20 for _ in range(n)]
        ys = [random.random() * 40 - 20 for _ in range(n)]
        t = random.random() * 40 - 20
        pts = [xs, ys]
        print(f',{(pts, t, against(np.array(pts), t))}')
        assert horner(np.array(pts), t) == against(np.array(pts), t)


def against(m, t):
    return (np.polynomial.polynomial.Polynomial(m[0, ::])(t),
        np.polynomial.polynomial.Polynomial(m[1, ::])(t))


if __name__ == '__main__':
    # _test_this_shit()
    # llllll()
    poly_gen()
