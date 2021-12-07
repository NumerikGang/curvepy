from typing import Iterable
from numbers import Number
from curvepy.bezier_curve import *

Testcases = Iterable
Points = Iterable
Point = Tuple[Number, ...]
BezierPoints = np.ndarray


def arrayize(xss: Testcases[Points[Point]]):
    return [[np.array(x) for x in xs] for xs in xss]
