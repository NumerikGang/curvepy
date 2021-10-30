import numpy as np
import random
import matplotlib.pyplot as plt

from curvepy.bezier_curve import *

TEST_CASES = 50
INTERVAL = (-20, 20)


def gen_test_cases(n):
    print(f"[")
    for _ in range(1):
        pts = [random.randint(*INTERVAL) + random.random() for _ in range(2*n)]
        pts = [pts[:len(pts)//2], pts[len(pts)//2:]]
        for _ in range(1):
            t = random.random()
            r = random.randint(1, len(pts)-1)
            int2 = sorted([random.randint(*INTERVAL), random.randint(*INTERVAL)])
            pts = np.array(pts)
            if int2[0] == int2[1]:
                continue
            a = BezierCurveSymPy(pts, use_parallel=True)
            b = BezierCurveDeCaes(pts, use_parallel=True)
            c = BezierCurveBernstein(pts, use_parallel=True)
            d = BezierCurveHorner(pts, use_parallel=True)
            #e = BezierCurveMonomial(pts, use_parallel=True)
            f = BezierCurveApproximation(pts, use_parallel=True)
            fig, axs = plt.subplots(6)
            axs[0].plot(*c.curve)
            axs[1].plot(*d.curve)
            axs[2].plot(*a.curve)
            axs[3].plot(*f.curve)
            #axs[4].plot(*e.curve)
            axs[5].plot(*b.curve)
    print(f"]")

if __name__ == '__main__':
    gen_test_cases(8)
