from numbers import Number
from typing import Iterable

from curvepy.bezier_curve import *

Testcases = Iterable
Points = Iterable
Point = Tuple[Number, ...]
BezierPoints = np.ndarray


def arrayize(xss: Testcases[Points[Point]]) -> Testcases[List[np.ndarray]]:
    """Helper function to make testcaselists filled with numpy arrays instead of other iterables.

    Parameters
    ----------
    xss: Testcases[Points[Point]]
        The testcases to convert

    Returns
    -------
    Testcases[List[np.ndarray]]
        The testcases in numpy-form
    """
    return [[np.array(x) for x in xs] for xs in xss]
