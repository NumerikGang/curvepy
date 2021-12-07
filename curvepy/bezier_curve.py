from __future__ import annotations  # Needed until Py3.10, see PEP 563
import numpy as np
import sympy as sy
import math
import scipy.special as scs
import matplotlib.pyplot as plt
import itertools as itt
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Union
from functools import partial, cached_property
import concurrent.futures
from multiprocessing import cpu_count
import sys
from curvepy.de_caes import de_caes, subdivision
from curvepy.utilities import prod, check_flat, intersect_lines
from curvepy.types import bernstein_polynomial, MinMaxBox


class AbstractBezierCurve(ABC):
    """
    Abstract class for creating a Bezier Curve.
    The function init_func has to be overwritten.

    Parameters
    ----------
    m: np.ndarray
        Array containing control points
    cnt_ts: int
        number of ts to create

    Attributes
    -------
    _bezier_points: np.ndarray
        array containing original bezier points
    _dimension: int
        dimension of the bezier points, therefore of the curve as well
    _cnt_ts: int
        numbers of equidistant ts to calculate
    func: Callable
        function computing the Bezier Curve for a single point
    """

    def __init__(self, m: np.ndarray, cnt_ts: int = 1000, use_parallel: bool = False,
                 interval: Tuple[int, int] = (0, 1)) -> None:
        self._bezier_points = m * 1.0
        self._dimension = self._bezier_points.shape[0]
        self._cnt_ts = cnt_ts
        self.interval = interval
        self.func = self.init_func()
        self._use_parallel = use_parallel

    @abstractmethod
    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        # TODO: Dies ist eine absolut dreiste Luege!
        Method returns the function to calculate all values at once.

        Returns
        -------
        Callable:
            function representing the Bezier Curve
        """
        ...

    def parallel_execution(self, ts: np.ndarray):
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
            return executor.map(self.func, ts)

    def serial_execution(self, ts: np.ndarray):
        return np.frompyfunc(self.func, 1, 1)(ts)

    # TODO FIx typing to nparray (and everywhere else)
    @cached_property
    def curve(self) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        TODO: Immer Unitinterval (dokumentieren)
        Method returning coordinates of all calculated points

        Returns
        -------
        Union[tuple[list, list], tuple[list, list, list]]:
            first list for x coords, second for y coords and third for z if existing
        """

        ts = np.linspace(0, 1, self._cnt_ts)
        curve = self.parallel_execution(ts) if self._use_parallel else self.serial_execution(ts)
        tmp = np.ravel([*curve])
        if self._dimension == 2:
            return tmp[0::2], tmp[1::2]
        return tmp[0::3], tmp[1::3], tmp[2::3]

    @cached_property
    def min_max_box(self) -> MinMaxBox:
        """
        Method creates minmax box for the corresponding curve
        """
        return MinMaxBox.from_bezier_points(self._bezier_points)

    @staticmethod
    def collision_check(b1: AbstractBezierCurve, b2: AbstractBezierCurve, tol: float = 0.01):
        """
        bezInt(B1, B2):
            Does bbox(B1) intersect_with_x_axis bbox(B2)?
                No: Return false.
                Yes: Continue.
            Is area(bbox(B1)) + area(bbox(B2)) < threshold?
                Yes: Return true.
                No: Continue.
            Split B1 into B1a and B1b at t = 0.5
            Split B2 into B2a and B2b at t = 0.5
            Return bezInt(B1a, B2a) || bezInt(B1a, B2b) || bezInt(B1b, B2a) || bezInt(B1b, B2b).
        """
        if not b1.min_max_box & b2.min_max_box:
            return False

        if b1.min_max_box.area + b2.min_max_box.area < tol:
            return True

        b1s = subdivision(b1._bezier_points, 0.5)
        b2s = subdivision(b2._bezier_points, 0.5)

        return any(
            AbstractBezierCurve.collision_check(BezierCurveDeCaes(left), BezierCurveDeCaes(right), tol)
            for left, right in itt.product(b1s, b2s)
        )

    @staticmethod
    def intersect_with_x_axis(m: np.ndarray, tol: float = sys.float_info.epsilon) -> Tuple[List[float], List[float]]:
        """
        Method checks if curve intersects with x-axis

        Parameters
        ----------
        m: np.ndarray:
            bezier points

        tol: float:
            tolerance for check_flat

        Returns
        -------
        np.ndarray:
            Points where curve and x-axis intersect_with_x_axis
        """
        box = MinMaxBox.from_bezier_points(m)
        res = [], []

        if box[2] * box[3] > 0:
            # Both y values are positive, ergo curve lies above x_axis
            return [], []

        if check_flat(m, tol):
            # poly is flat enough, so we can perform intersect_with_x_axis of straight lines
            # since we are assuming poly is a straight line we define a line through first and las point of poly
            # additionally we create a line which demonstrates the x axis
            # having these two lines we can check them for intersection
            p = intersect_lines(m[:, 0], m[:, -1], np.array([0, 0]), np.array([1, 0]))
            if p is not None:
                res[0].append(p[0])
                res[1].append(p[1])
        else:
            # if poly not flat enough we subdivide and check the resulting polygons for intersection
            p1, p2 = subdivision(m)
            rec1 = AbstractBezierCurve.intersect_with_x_axis(p1, tol)
            rec2 = AbstractBezierCurve.intersect_with_x_axis(p2, tol)
            res[0].extend(rec1[0] + rec2[0])
            res[1].extend(rec1[1] + rec2[1])

        return res

    def plot(self) -> None:
        """
        Method plotting the curve by adding it to the current pyplot figure
        """
        if self._dimension == 2:
            plt.plot(*self.curve, 'o')
        else:
            ax = plt.axes(projection='3d')
            ax.scatter3D(*self.curve)

    def single_forward_difference(self, i: int = 0, r: int = 0) -> np.ndarray:
        """
        Method using equation 5.23 to calculate forward difference of degree r for specific point i

        Parameters
        ----------
        i: int:
            point i for which forward all_forward_differences are calculated

        r: int:
            degree of forward difference

        Returns
        -------
        np.ndarray:
                forward difference of degree r for point i

        Notes
        -----
        Equation used for computing all_forward_differences:
        math:: \\Delta^r b_i = \\sum_{j=0}^r \\binom{r}{j} (-1)^{r-j} b_{i+j}
        """
        return np.sum([scs.binom(r, j) * (-1) ** (r - j) * self._bezier_points[:, i + j] for j in range(0, r + 1)],
                      axis=0)

    def all_forward_differences_for_one_value(self, i: int = 0) -> np.ndarray:
        """
        TODO rewrite this docstring
        Method using equation 5.23 to calculate all forward all_forward_differences for a given point i.
        First entry is first difference, second entry is second difference and so on.

        Parameters
        ----------
        i: int:
            point i for which forward all_forward_differences are calculated

        Returns
        -------
        np.ndarray:
             array holds all forward all_forward_differences for given point i

        Notes
        -----
        Equation used for computing all_forward_differences:
        math:: \\Delta^r b_i = \\sum_{j=0}^r \\binom{r}{j} (-1)^{r-j} b_{i+j}
        """
        deg = self._bezier_points.shape[1] - 1
        diff = [self.single_forward_difference(i, r) for r in range(0, (deg - i) + 1)]
        return np.array(diff).T

    def derivative_bezier_curve(self, t: float = 0.5, r: int = 1) -> np.ndarray:
        """
        Method using equation 5.24 to calculate rth derivative of bezier curve at value t

        Parameters
        ----------
        t: float:
            value for which Bezier curves are calculated

        r: int:
            rth derivative

        Returns
        -------
        np.ndarray:
             point of the rth derivative at value t

        Notes
        -----
        Equation used for computing:
        math:: \\frac{d^r}{dt^r} b^n(t) = \\frac{n!}{(n-r)!} \\cdot \\sum_{j=0}^{n-r} \\Delta^r b_j \\cdot B_j^{n-r}(t)
        """
        deg = self._bezier_points.shape[1] - 1
        tmp = [self.single_forward_difference(j, r) * bernstein_polynomial(deg - r, j, t) for j in range((deg - r) + 1)]
        return prod(range(deg - r + 1, deg + 1)) * np.sum(tmp, axis=0)

    @staticmethod
    def barycentric_combination_bezier(m: AbstractBezierCurve, c: AbstractBezierCurve, alpha: float = 0,
                                       beta: float = 1) -> AbstractBezierCurve:
        """
        TODO Docstrings
        Method using 5.13 to calculate the barycentric combination of two given bezier curves

        Parameters
        ----------
        m: np.ndarray:
            first array containing the Bezier Points

        c: np.ndarray:
            second array containing the Bezier Points

        alpha: float:
            weight for the first Bezier curve

        beta: float:
            weight for the first Bezier curve

        Returns
        -------
        np.ndarray:
                point of the weighted combination

        Notes
        -----
        The Parameter alpha and beta must hold the following condition: alpha + beta = 1
        Equation used for computing:
        math:: \\sum_{j=0}^r (\\alpha \\cdot b_j + \\beta \\cdot c_j)B_j^n(t) =
        \\alpha \\cdot \\sum_{j=0}^r b_j \\cdot B_j^n(t) + \\beta \\cdot \\sum_{j=0}^r c_j \\cdot B_j^n(t)
        """

        if alpha + beta != 1:
            raise ValueError("Alpha and Beta must add up to 1!")

        return m * alpha + c * beta

    def __str__(self):
        """
        Returns string represenation as the mathematical bezier curve

        Returns
        -------
        String: represenation as the mathematical bezier curve
        """
        return f"b^{self._bezier_points.shape[1] - 1}(t)"

    def __repr__(self):
        """
        Returns internal represenation based on __str__

        Returns
        -------
        String: internal represenation based on __str__
        """
        return f"<id {id(self)}, {self.__str__()}>"

    def __call__(self, u):
        # 3.9
        a, b = self.interval
        return self.func((u - a) / (b - a))

    def __mul__(self, other: Union[float, int]):
        try:
            del self.curve  # this is fine and as it should be to reset cached_properties, linters are stupid
        except AttributeError:
            # This means that the curve was never called before
            ...
        self._bezier_points *= other
        return self

    __rmul__ = __mul__

    # TODO write in docstrings that anything other than unit intervals are forbidden
    def __add__(self, other: AbstractBezierCurve):
        if not isinstance(other, AbstractBezierCurve):
            raise TypeError("Argument must be an instance of AbstractBezierCurve")
        try:
            del self.curve  # this is fine
        except AttributeError:
            # This means that the curve was never called before
            ...

        self._bezier_points += other._bezier_points
        self._cnt_ts = max(self._cnt_ts, other._cnt_ts)
        return self


class BezierCurveSymPy(AbstractBezierCurve):
    """
    Class for creating a 2-dimensional Bezier Curve by using the De Casteljau Algorithm

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """

    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        Method returns the function to calculate all values at once.

        Returns
        -------
        Callable:
            function representing the Bezier Curve
        """
        m = self._bezier_points
        _, n = m.shape
        m = sy.Matrix(m)
        t = sy.symbols('t')
        for r in range(n):
            m[:, :(n - r - 1)] = (1 - t) * m[:, :(n - r - 1)] + t * m[:, 1:(n - r)]
        return sy.lambdify(t, m[:, 0])


class BezierCurveDeCaes(AbstractBezierCurve):
    """
    Class for creating a 2-dimensional Bezier Curve by using the De Casteljau Algorithm

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """

    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        Method returns the function to calculate all values at once.

        Returns
        -------
        Callable:
            function representing the Bezier Curve
        """
        return partial(de_caes, self._bezier_points)


class BezierCurveBernstein(AbstractBezierCurve):

    def init_func(self) -> Callable[[float], np.ndarray]:
        return self.bezier_curve_with_bernstein

    def bezier_curve_with_bernstein(self, t: float = 0.5, r: int = 0,
                                    interval: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Method using 5.8 to calculate a point with given bezier points

        Parameters
        ----------
        t: float:
            value for which Bezier curve are calculated

        r: int:
            optional Parameter to calculate only a partial curve if we already have some degree of the bezier points

        interval: Tuple[float,float]:
            Interval of t used for affine transformation

        Returns
        -------
        np.ndarray:
                point on the curve

        Notes
        -----
        Equation used for computing:
        math:: b_i^r(t) = \\sum_{j=0}^r b_{i+j} \\cdot B_i^r(t)
        """
        m = self._bezier_points
        _, n = m.shape
        t = (t - interval[0]) / (interval[1] - interval[0])
        return np.sum([m[:, i] * bernstein_polynomial(n - r - 1, i, t) for i in range(n - r)], axis=0)


class BezierCurveHorner(AbstractBezierCurve):

    def init_func(self) -> Callable[[float], np.ndarray]:
        return self.horn_bez

    def horn_bez(self, t: float = 0.5) -> np.ndarray:
        """
        Method using horner like scheme to calculate point on curve with given t

        Parameters
        ----------
        t: float:
            value for which point is calculated

        Returns
        -------
        np.ndarray:
            point calculated with given t
        """
        m = self._bezier_points
        n = m.shape[1] - 1  # need degree of curve (n points means degree = n-1)
        res = m[:, 0] * (1 - t)
        for i in range(1, n):
            res = (res + t ** i * scs.binom(n, i) * m[:, i]) * (1 - t)

        res += t ** n * m[:, n]

        return res


class BezierCurveMonomial(AbstractBezierCurve):

    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        TODO Docstring
        Method calculating monomial representation of given bezier form using 5.27

        Parameters
        ----------
        m: np.ndarray:
            array containing the Bezier Points

        Returns
        -------
        Callable:
            bezier function in polynomial form

        Notes
        -----
        Equation 5.27 used for computing polynomial form:
        math:: b^n(t) = \\sum_{j=0}^n \\binom{n}{j} \\Delta^j b_0 t^j

        Initially the method would only compute the polynomial coefficients in an array, and parsing this array with
        a given t to the horner method we would get a point back. Instead the method uses sympy to calculate a function
        depending on t. After initial computation, f(t) calculates the value for a given t. Having a function it is
        simple to map it on an array containing multiple values for t.
        As a result we do not need to call the horner method for each t individually.
        """
        m = self._bezier_points
        _, n = m.shape
        diff = self.all_forward_differences_for_one_value()
        t = sy.symbols('t')
        res = 0
        for i in range(n):
            res += scs.binom(n - 1, i) * diff[:, i] * t ** i

        return sy.lambdify(t, res)


class BezierCurveApproximation(AbstractBezierCurve):
    def init_func(self) -> Callable[[float], np.ndarray]:
        # dummy, just used for __call__ and __getitem__
        return partial(de_caes, self._bezier_points)

    serial_execution = None
    parallel_execution = None

    @classmethod
    def from_round_number(cls, m: np.ndarray, approx_rounds: int = 5,
                          interval: Tuple[int, int] = (0, 1)) -> BezierCurveApproximation:
        return cls(m, cls.approx_rounds_to_cnt_ts(approx_rounds, m.shape[1]), False, interval)

    @staticmethod
    def cnt_ts_to_approx_rounds(cnt_ts, cnt_bezier_points):
        return math.ceil(math.log(cnt_ts / cnt_bezier_points, 2))

    @staticmethod
    def approx_rounds_to_cnt_ts(approx_rounds, cnt_bezier_points):
        return 2 ** approx_rounds * cnt_bezier_points

    @cached_property
    def curve(self) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[float]]]:
        approx_rounds = self.cnt_ts_to_approx_rounds(self._cnt_ts, self._bezier_points.shape[1])
        current = [self._bezier_points]
        for _ in range(approx_rounds):
            queue = []
            for c in current:
                queue.extend(subdivision(c))
            current = queue

        ret = np.hstack(current)
        ret = np.ravel(ret)
        n = len(ret)
        if self._dimension == 2:
            assert (n / 2).is_integer()  # TODO debug
            return ret[:n // 2], ret[n // 2:]
        assert self._dimension == 3
        assert (n / 3).is_integer()  # todo debug
        return ret[:n // 3], ret[n // 3:2 * n // 3], ret[2 * n // 3:]
