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
    print(f"[")
    for _ in range(1):
        pts = [random.randint(*INTERVAL) + random.random() for _ in range(2*n)]
        pts = [pts[:len(pts)//2], pts[len(pts)//2:]]
        for _ in range(1):
            t = random.random()
            r = random.randint(1, len(pts)-1)
            int2 = sorted([random.randint(*INTERVAL), random.randint(*INTERVAL)])
            pts = np.array(bsp)
            if int2[0] == int2[1]:
                continue
            a = BezierCurveSymPy(pts.copy(), cnt_ts=8, use_parallel=True)
            b = BezierCurveDeCaes(pts.copy(), cnt_ts=8, use_parallel=True)
            c = BezierCurveBernstein(pts.copy(), cnt_ts=8, use_parallel=True)
            d = BezierCurveHorner(pts.copy(), cnt_ts=8, use_parallel=True)
            #e = BezierCurveMonomial(pts, use_parallel=True)
            f = BezierCurveApproximation(pts.copy(), cnt_ts=8, use_parallel=True)
            fig, axs = plt.subplots(5)

            # acurve = a.curve
            # bcurve = b.curve
            # ccurve = c.curve
            # dcurve = d.curve
            # # ecurve = e.curve
            # fcurve = f.curve



            ccurve = c.curve
            dcurve = d.curve
            acurve = a.curve
            bcurve = b.curve
            fcurve = f.curve
            #ecurve = e.curve
            bcurve = b.curve
            axs[0].plot(*acurve,  marker='o')
            axs[1].plot(*bcurve, marker='o')
            axs[2].plot(*ccurve,  marker='o')
            axs[3].plot(*dcurve,  marker='o')
            #axs[4].plot(*e.curve)
            axs[4].plot(*fcurve,  marker='o')
    print(f"]")

if __name__ == '__main__':
    gen_test_cases(8)
