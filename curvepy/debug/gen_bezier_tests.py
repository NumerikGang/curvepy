import numpy as np
import random

from curvepy.bezier_curve import BezierCurveDeCaes

TEST_CASES = 50
INTERVAL = (-20, 20)


def gen_test_cases(n):
    print(f"[")
    for _ in range(TEST_CASES):
        pts = [random.randint(*INTERVAL) + random.random() for _ in range(2*n)]
        pts = [pts[:len(pts)//2], pts[len(pts)//2:]]
        for _ in range(3):
            t = random.random()
            r = random.randint(1, len(pts)-1)
            b = BezierCurveDeCaes(np.array(pts))
            res = list(b.derivative_bezier_curve(t, r))
            print(f"{(pts,t,r,res)},")
    print(f"]")


if __name__ == '__main__':
    gen_test_cases(8)
