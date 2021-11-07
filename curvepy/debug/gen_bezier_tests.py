import numpy as np
import random
import matplotlib.pyplot as plt

import curvepy.types
from curvepy.bezier_curve import *

TEST_CASES = 50
INTERVAL = (-20, 20)

# bsp = [[0.,0.,8.,4.],[0.,2.,2.,0.]]
bsp = [[0, 0, 8, 4], [0, 2, 2, 0]]


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

            print(f'{pts},' , end=" ")
        alpha = random.random()
        print(f'{alpha})')
    print(f"]")


if __name__ == '__main__':
    gen_test_cases(8)
