# flake8: noqa
import numpy as np
import timeit


def calc_area(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    """
    See: https://www.mathopenref.com/coordtrianglearea.html
    """
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

def old(A,B,C):
    """
    :return:

    See: https://de.wikipedia.org/wiki/Umkreis#Koordinaten
    See: https://de.wikipedia.org/wiki/Umkreis#Radius
    """
    area = calc_area(*A,*B,*C)

    [x1, y1], [x2, y2], [x3, y3] = A, B, C
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    xu = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
    yu = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

    lines = [[A, B], [B, C], [A, C]]
    c, a, b = [np.linalg.norm(np.array(x) - np.array(y)) for x, y in lines]

    R = (a * b * c) / (4 * area)
    return xu,yu,R

def new(a,b,c):
    tmp_pts_1 = np.array([np.array(x - np.array(a)) for x in [b, c]])
    tmp_pts_2 = np.sum(tmp_pts_1 ** 2, axis=1)
    tmp_pts_3 = np.array([np.linalg.det([x, tmp_pts_2]) / (2 * np.linalg.det(tmp_pts_1)) for x in tmp_pts_1.T])
    center = a[0] - tmp_pts_3[1], a[1] + tmp_pts_3[0]
    radius = np.linalg.norm(np.array(a) - np.array(center))
    return center, radius

def timereps(reps, func):
    from time import time
    start = time()
    for i in range(0, reps):
        func((4,3),(1,2),(5,6))
    end = time()
    return (end - start) / reps

if __name__ == '__main__':
    print(timereps(1000, old))
    print(timereps(1000, new))