import numpy as np
import random
import matplotlib.pyplot as plt
import bezier

import curvepy.types
from curvepy.bezier_curve import *

TEST_CASES = 50
INTERVAL = (-20, 20)

# bsp = [[0.,0.,8.,4.],[0.,2.,2.,0.]]
bsp = [[0, 0, 8, 4], [0, 2, 2, 0]]


def gen_not_even_boxes_intersect(l, r):
    with open('bla3.txt', 'w') as fp:
        print("[", file=fp)
        for _ in range(TEST_CASES):
            xss = [(random.random() * (r - l)) - ((r - l) / 2) for _ in range(4)]
            xs1 = sorted(xss[:2])
            xs1[0] -= 5
            ys1 = sorted(xss[2:])
            ys1[0] -= 5
            # move to right
            delta = ys1[1] - ys1[0] + random.random() * 8
            ys2 = [y + delta for y in ys1]
            print(f",({xs1},{ys1},{xs1},{ys2}", file=fp)

            vals = []
            for (xs, ys) in [(xs1, ys1), (xs1, ys2)]:
                vals.append([[
                    random.random() * (xs[1] - xs[0]) - (xs[1] - xs[0]) / 2 + xs[0] for _ in range(8)
                ], [
                    random.random() * (ys[1] - ys[0]) - (ys[1] - ys[0]) / 2 + ys[0] for _ in range(8)
                ]])
            print(f",{vals[0]}, {vals[1]})", file=fp)
        print("]", file=fp)


def gen_bezier_intersections():
    with open('bla2.txt', 'w') as fp:
        print("[", file=fp)
        number_of_intersections = 0
        while number_of_intersections < 20:
            pts1 = [random.randint(*INTERVAL) + random.random() for _ in range(2 * 8)]
            pts1 = [pts1[:len(pts1) // 2], pts1[len(pts1) // 2:]]
            pts1 = np.array(pts1)
            c1 = bezier.Curve.from_nodes(pts1)
            pts2 = [random.randint(*INTERVAL) + random.random() for _ in range(2 * 8)]
            pts2 = [pts2[:len(pts2) // 2], pts2[len(pts2) // 2:]]
            pts2 = np.array(pts2)
            c2 = bezier.Curve.from_nodes(pts2)
            res = c1.intersect(c2)

            if not res:
                continue

            print(f",(")
            print(f")")

            number_of_intersections += 1
        print("]", file=fp)


def gen_intervals(n):
    print("[")
    for _ in range(n):
        int2 = sorted([random.randint(*INTERVAL), random.randint(*INTERVAL)])
        if int2[0] == int2[1]:
            continue
        print(f",{int2}")
    print("]")


def gen_test_cases(n):
    # fp = open('bla.txt', 'w')
    print(f"[")
    for _ in range(20):
        print(",(", end="")
        for _ in range(2):
            pts = [random.randint(*INTERVAL) + random.random() for _ in range(2 * n)]
            pts = [pts[:len(pts) // 2], pts[len(pts) // 2:]]

            print(f'{pts},', end=" ")
        alpha = random.random()
        print(f'{alpha})')
    print(f"]")


if __name__ == '__main__':
    gen_not_even_boxes_intersect(-20, 20)
