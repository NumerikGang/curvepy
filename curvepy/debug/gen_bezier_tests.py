import numpy as np
import random
import matplotlib.pyplot as plt

import curvepy.types
from curvepy.bezier_curve import *

TEST_CASES = 50
INTERVAL = (-20, 20)

#bsp = [[0.,0.,8.,4.],[0.,2.,2.,0.]]
bsp = [[0,0,8,4],[0,2,2,0]]

def gen_test_cases(n):
    fp = open('bla.txt', 'w')
    print(f"[", file=fp)
    for _ in range(10):
        pts = [random.randint(*INTERVAL) + random.random() for _ in range(2*n)]
        pts = [pts[:len(pts)//2], pts[len(pts)//2:]]
        for _ in range(1):
            t = random.random()
            r = random.randint(1, len(pts)-1)
            int2 = sorted([random.randint(*INTERVAL), random.randint(*INTERVAL)])

            if int2[0] == int2[1]:
                continue
            print(f',({([1,2],[3,4])},', file=fp, end=' ')
            pts = np.array(pts)
            rounds = random.randint(6, 8+1)
            f = BezierCurveApproximation.from_round_number(pts.copy(), rounds)
            c = f.curve
            print(f'{[list([1,2,3]), list([1,2,3])]}, {rounds})', file=fp)
            # fig, axs = plt.subplots(6)
            #
            # acurve = a.curve
            # bcurve = b.curve
            # ccurve = c.curve
            # dcurve = d.curve
            # ecurve = e.curve
            # fcurve = f.curve

            # axs[0].plot(*acurve)
            # axs[1].plot(*bcurve)
            # axs[2].plot(*ccurve)
            # axs[3].plot(*dcurve)
            # axs[4].plot(*e.curve)
            # axs[5].plot(*fcurve)
    print(f"]", file=fp)
    fp.close()

if __name__ == '__main__':
    gen_test_cases(8)
