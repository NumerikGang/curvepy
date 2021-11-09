import random
import numpy as np
from curvepy.utilities import *


def gen_vals(n, interval):
    return [random.randint(*interval) for _ in range(n)]

def main(test_cases, interval):
    print("[")
    for _ in range(test_cases):
        p1 = np.array(gen_vals(2, interval))
        m1 = (random.random() * 10.0) - 5
        p2 = p1*m1
        m2 = (random.random() * 10.0) - 5
        p3 = p1*m2
        r = ratio(p1,p2,p3)
        print(f",({[*p1]}, {[*p2]}, {[*p3]}, {r})")
    print("]")

if __name__ == '__main__':
    main(20, (-20, 20))