import numpy as np

from functools import partial
from scipy.special import comb
from abc import ABC, abstractmethod
from typing import Optional

# Ways to compute bezier curve
# Blossoming / No Blossoming
# Keeping Intermediates / Not keeping
# Keeping Derivative / Not keeping
# [0,1] / [a,b]
# Using DeCasteljau to approxmate (optimized, see 5.4) / calculate all points
# DeCasteljau / Direct Bernstein / Timos Sympy
# Optimized polar form
# Matrix form (numerically instable)

# ---
# Different Line intersection algorithms
# minmax box
# recursive min max box

# ---
# Implementations book

def bernstein(n: int, k: int, t: float, exact=True) -> float:
    return comb(n, k, exact=exact) * t ** k * (1 - t) ** (n - k)


def get_bernstein_func(n: int, k: int, exact: bool = True) -> partial:
    return partial(bernstein, n=n, k=k, exact=exact)


# ---

class AbstractBezierCurve(ABC):
    _DEFAULT_INTERVAL = (0, 1)

    def __init__(self, bs, n, k: Optional[int] = None, interval_closed=(0, 1), steps=1000):
        self.bs = bs
        self.n = n
        self.k = n if k is None else k
        self.interval_closed = interval_closed
        self.steps = steps

        self.computed_interval_values = None

    def __str__(self):
        return f'b_{self.k}^{self.n}(t)'

    def __repr__(self):
        return f'<id {id(self)}, {self.__str__()}>'

    def __call__(self, t):
        self._calc_value(self._affine_parameter_transformation_helper(t))

    def __eq__(self, other):
        # Needed for cache lookup
        # Why we prefer isinstance:
        # https://stackoverflow.com/a/1549854/9958281
        if isinstance(other, AbstractBezierCurve):
            return self.n == other.n and self.k == other.k and self.bs

    @abstractmethod
    def _calc_value(self, t):
        ...

    def get_interval_values(self):
        if self.computed_interval_values is None:
            self._compute_interval_values()
        return self.computed_interval_values

    @abstractmethod
    def _compute_interval_values(self):
        ...

    # Begin helper methods
    def _uses_default_interval(self):
        return self.interval_closed == self._DEFAULT_INTERVAL

    def _affine_parameter_transformation_helper(self, t):
        a, b = self._DEFAULT_INTERVAL
        return t if self._uses_default_interval() else (t - a) / (b - a)


class BezierCurveBernsteinBasis(AbstractBezierCurve):
    def _calc_value(self, t):
        return sum([self.bs[self.k + i] * bernstein(self.n, i, t) for i in range(self.n + 1)])

    def _compute_interval_values(self):
        a, b = self.interval_closed
        steps_arr = np.linspace(a, b, self.steps)
        return [self._calc_value(t) for t in steps_arr]


class BezierCurveDeCasteljau(AbstractBezierCurve):
    _BEZIER_CURVE_CACHE = []

    def __new__(cls, bs, n, k: Optional[int] = None, interval_closed=(0, 1), steps=1000):
        ...

    def _calc_value(self, t):
        ...

    def _compute_interval_values(self):
        ...

    # Begin helper methods
    @classmethod
    def _get_cache(cls, n, k):
        ...
