import numpy as np
from typing import Iterable, Tuple
from numbers import Number
from dataclasses import dataclass
from curvepy.bezier_curve import *

Testcases = Iterable
Points = Iterable
Point = Tuple[Number, ...]
BezierPoints=np.ndarray

def arrayize(xss: Testcases[Points[Point]]):
    return [[np.array(x) for x in xs] for xs in xss]


@dataclass
class AllBezierCurves:
    bezier_curve_sympy: BezierCurveSymPy
    bezier_curve_de_caes: BezierCurveDeCaes
    bezier_curve_bernstein: BezierCurveBernstein
    bezier_curve_horner: BezierCurveHorner
    bezier_curve_monomial: BezierCurveMonomial
    bezier_curve_approximation: BezierCurveApproximation

    @classmethod
    def from_bezier_points(cls, m: np.ndarray, cnt_ts: int = 1000, use_parallel: bool = False,
                           interval: Tuple[int, int] = (0, 1)):
        params = {
            "m": m, "cnt_ts": cnt_ts, "use_parallel": use_parallel, "interval": interval
        }
        return cls(
            bezier_curve_sympy=BezierCurveSymPy(**params),
            bezier_curve_de_caes=BezierCurveDeCaes(**params),
            bezier_curve_bernstein=BezierCurveBernstein(**params),
            bezier_curve_horner=BezierCurveHorner(**params),
            bezier_curve_monomial=BezierCurveMonomial(**params),
            bezier_curve_approximation=BezierCurveApproximation(**params)
        )

    @staticmethod
    def from_testcases(xss: Testcases[BezierPoints], **kwargs):
        return [AllBezierCurves.from_bezier_points(xs, **kwargs) for xs in xss]