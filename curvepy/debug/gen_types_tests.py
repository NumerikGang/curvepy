from curvepy.types import *
import random
import numpy as np
import sys
import scipy.special as scp

INTERVAL = (-20, 20)
TESTCASES = 50


def gewichtung_n_punkte(alpha, a, beta, b, gamma, c):
    return list(alpha * a + beta * b + gamma * c)


def gen_polygon_tests_2D():
    print("[")
    for _ in range(TESTCASES):
        pts = [random.randint(*INTERVAL) + random.random() for _ in range(3 * 3)]
        #pts = [pts[:len(pts) // 2], pts[len(pts) // 2:]]
        pts = [pts[:len(pts) // 3], pts[len(pts) // 3: 2 * len(pts) // 3], pts[2 * len(pts) // 3:]]

        weights = [random.random() for _ in range(2)]
        weights.append(1 - sum(weights))

        tmp = [np.array([x, y, z]) for x, y, z in zip(pts[0], pts[1], pts[2])]

        print(f",{(pts, weights, gewichtung_n_punkte(tmp[0], weights[0], tmp[1], weights[1], tmp[2], weights[2]))}")

    print("]")

def lmao():
    print("[")
    for _ in range(TESTCASES):
        t = random.random()
        i, n = sorted([random.randint(0,40), random.randint(0,40)])
        while n == i:
            i, n = sorted([random.randint(0, 40), random.randint(0, 40)])
        it = bernstein_polynomial(n,i,t)
        rec = bernstein_polynomial_rec(n,i,t)
        assert abs(it-rec) < 0.01
        print(f"{(n,i,t,it)},")
    print("]")

if __name__ == '__main__':
    lmao()
