import numpy as np
import random

from curvepy.bezier_curve import BezierCurveSymPy, BezierCurveDeCaes

TEST_CASES = 50
dif = lambda s: [x - s[i] for i, x in enumerate(s[1:])]
difn = lambda s, n: difn(dif(s), n - 1) if n else s


def all_forward_dif(s, i):
    n = len(s)
    j = 0
    ret = []
    while True:
        r = difn(s[i:], j)
        ret.append(r)
        if len(r) == 1:
            break
        j += 1
    return ret


def gen_test_cases(min, max, val_min, val_max):
    print("[")
    for _ in range(TEST_CASES):
        x = [random.randint(val_min, val_max) for _ in range(random.randint(min, max) + 1)]
        print("(",x, ",")
        print(f"({difn(x, len(x) - 1)[0]},{difn(x, len(x) - 1)[0]})),")
    print("]")

def gen_test_cases_for_all_forward_for_one_value(min, max, val_min, val_max, fp):
    print("[", file=fp)
    for _ in range(TEST_CASES):
        x = [random.randint(val_min, val_max) for _ in range(random.randint(min, max) + 1)]
        for i in range(len(x)):
            if random.random() < 0.5:
                continue
            print(f"(({x}, {i}), ", file=fp)
            print(f"{all_forward_dif(x, i)}),", file=fp)
    print("]", file=fp)


gen_n = lambda n : [random.randint(-20,20) for _ in range(n+1)]


if __name__ == '__main__':
    # gen_test_cases(15, 50, -20, 20)
    with open("bla.txt", "a") as fp:
        gen_test_cases_for_all_forward_for_one_value(15, 50, -20, 20, fp)

    # xs = gen_n(16)
    # ret = all_forward_dif(xs, 0)
    # for r in ret:
    #     print(r)