import random
import numpy as np
from curvepy.utilities import *


def gen_vals(n, interval):
    return [random.randint(*interval) for _ in range(n)]

def main(test_cases, interval):
    print("[")
    for _ in range(test_cases):
        l = random.randint(2, 100)

    print("]")

if __name__ == '__main__':
    main(20, (-20, 20))