import numpy as np
from typing import Iterable, Tuple
from numbers import Number

Testcase = Iterable
Points = Iterable
Point = Tuple[Number, ...]

def arrayize(xss: Testcase[Points[Point]]):
    return [[np.array(x) for x in xs] for xs in xss]
