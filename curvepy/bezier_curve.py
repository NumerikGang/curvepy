"""
This module puts the methods defined in curvepy.de_caes to use by creating different classes of Bezier curves. All these
classes use different methods to calculate points on the curve defined by their Bezier points:

- AbstractBezierCurve: the base class provides most of the functionality. Many operations supported by the Bezier curve
classes do not need the actual points on the curve instead they run with the bezier points. Therefore the subclasses
just provide the actual method to compute points on the curve.

- BezierCurveSymPy: at the core it uses de Castelljau but instead of substituting the value every step is done symbolic
so at after the computation a function is returned representing the curve. If a point is calculated it just evaluates
the function.

- BezierCurveDeCaes: this classes uses the standard de Castelljau algorithm to compute points on the curve.

- BezierCurveBernstein: instead of the de Castelljau algorithm this curve uses the Bernstein polynomials to calculate
points on the curve.

- BezierCurveHorner: in this case a Horner like scheme is used to compute the points

- BezierCurveMonomial: this curve makes use of the monomial form of Bezier curves. Therefore the Bezier points
are used to get the coefficients and since this is executed symbolic at the end a function is created.
Similar to BezierCurveSymPy a point is calculated by substituting the variable with the given value.

- BezierCurveApproximation: since it is a approximation it does not support all methods from AbstractBezierCurve.
Subdivision is used multiple times to compute an approximation of the curve.
This type does not support parallel execution.
"""

from __future__ import annotations  # Needed until Py3.10, see PEP 563

import concurrent.futures
import itertools as itt
import math
import sys
from abc import ABC, abstractmethod
from functools import cached_property, partial
from multiprocessing import cpu_count
from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as scs
import sympy as sy

from curvepy.de_caes import de_caes, subdivision
from curvepy.types import MinMaxBox, bernstein_polynomial
from curvepy.utilities import check_flat, intersect_lines, prod


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
    use_parallel: bool
        flag for parallel execution
    interval: Tuple[int, int]
        interval represents the range in which parameter values are accepted

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
    interval: Tuple[int, int]
        interval represents the range in which parameter values are accepted
    _use_parallel: bool
        if computation should be executed parallel
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
        Method is used to define how points on the curve are calculated. Every subclass defines this differently.

        Returns
        -------
        Callable[[float], np.ndarray]:
            function representing the Bezier Curve
        """
        ...

    def parallel_execution(self, ts: np.ndarray):
        """
        Parallel computation of points lying on the curve

        Parameters
        ----------
        ts: np.ndarray:
            value for which point is calculated

        Returns
        -------
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() * 2) as executor:
            return executor.map(self.func, ts)

    def serial_execution(self, ts: np.ndarray):
        """
        serial computation of points lying on the curve

        Parameters
        ----------
        ts: np.ndarray:
            value for which point is calculated

        Returns
        -------
        """
        return np.frompyfunc(self.func, 1, 1)(ts)

    @cached_property
    def curve(self) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Cached Property returning coordinates of all calculated points.
        The computation always uses the unit interval (0,1), since we can perform an affine transformation
        on any interval to revert back to (0,1).

        Returns
        -------
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        Cached Property creates minmax box for the corresponding curve
        """
        return MinMaxBox.from_bezier_points(self._bezier_points)

    @staticmethod
    def collision_check(b1: AbstractBezierCurve, b2: AbstractBezierCurve, tol: float = 0.01) -> bool:
        """
        Using MinMaxBox intersection to check if two curves intersect. If tolerance is not satisfying, subdivision is
        used to get a better result. This is done recursively until tolerance is achieved.

        Parameters
        ----------
        b1: AbstractBezierCurve:
            first curve

        b2: AbstractBezierCurve:
            second curve

        tol: float:
            tolerance for which the method should accept

        Returns
        -------
        bool:
            True if intersect, false otherwise
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
        Staticmethod checks if curve intersects with x-axis. Similar to collision check subdivision is used to achieve a
        better tolerance.

        Parameters
        ----------
        m: np.ndarray:
            Bezier points

        tol: float:
            tolerance for check_flat

        Returns
        -------
        np.ndarray:
            Points where curve and x-axis intersect
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
        \\[\\Delta^r b_i = \\sum_{j=0}^r \\binom{r}{j} (-1)^{r-j} b_{i+j}\\]
        """
        return np.sum([scs.binom(r, j) * (-1) ** (r - j) * self._bezier_points[:, i + j] for j in range(0, r + 1)],
                      axis=0)

    def all_forward_differences_for_one_value(self, i: int = 0) -> np.ndarray:
        """
        Method using equation 5.23 to calculate all forward differences for a given point i.
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
        \\[\\Delta^r b_i = \\sum_{j=0}^r \\binom{r}{j} (-1)^{r-j} b_{i+j}\\]
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
        \\[\\frac{d^r}{dt^r} b^n(t) = \\frac{n!}{(n-r)!} \\cdot \\sum_{j=0}^{n-r} \\Delta^r b_j \\cdot B_j^{n-r}(t)\\]
        """
        deg = self._bezier_points.shape[1] - 1
        tmp = [self.single_forward_difference(j, r) * bernstein_polynomial(deg - r, j, t) for j in range((deg - r) + 1)]
        return prod(range(deg - r + 1, deg + 1)) * np.sum(tmp, axis=0)

    @staticmethod
    def barycentric_combination_bezier(m: AbstractBezierCurve, c: AbstractBezierCurve, alpha: float = 0,
                                       beta: float = 1) -> AbstractBezierCurve:
        """
        Staticmethod using 5.13 to calculate the barycentric combination of two given bezier curves.
        The given scalars must add up to one.

        Parameters
        ----------
        m: AbstractBezierCurve:
           first curve

        c: AbstractBezierCurve:
           second curve

        alpha: float:
            weight for the first Bezier curve

        beta: float:
            weight for the second Bezier curve

        Returns
        -------
        np.ndarray:
                point of the weighted combination

        Notes
        -----
        The Parameter alpha and beta must hold the following condition: alpha + beta = 1
        Equation used for computing:
        \\[\\sum_{j=0}^r (\\alpha \\cdot b_j + \\beta \\cdot c_j)B_j^n(t) =
        \\alpha \\cdot \\sum_{j=0}^r b_j \\cdot B_j^n(t) + \\beta \\cdot \\sum_{j=0}^r c_j \\cdot B_j^n(t)\\]
        """

        if alpha + beta != 1:
            raise ValueError("Alpha and Beta must add up to 1!")

        return m * alpha + c * beta

    def __str__(self):
        """
        Returns string representation as the mathematical bezier curve

        Returns
        -------
        String:
            representation as the mathematical bezier curve
        """
        return f"b^{self._bezier_points.shape[1] - 1}(t)"

    def __repr__(self):
        """
        Returns internal representation based on __str__

        Returns
        -------
        String: internal representation based on __str__
        """
        return f"<id {id(self)}, {self.__str__()}>"

    def __call__(self, u: float) -> np.ndarray:
        """
        Since the default computation is on done on the unit interval. Call let you use values from a custom interval
        that is defined with the initialisation of the class. This possible since we can use an affine transformation
        to map u on the interval (0,1). This is done by using equation 3.9.

        Parameters
        ----------
        u: np.ndarray:
            value for which point is calculated

        Returns
        -------
        np.ndarray:
            calculated point

        Notes
        -----
        Equation used for affine transformation:
        \\[t = \\frac{u - a}{b - a}\\]
        """
        a, b = self.interval
        return self.func((u - a) / (b - a))

    def __mul__(self, other: Union[float, int]):
        """
        Multiplies all Bezier points with given scalar.

        Parameters
        ----------
        other: Union[float, int]:
            scalar

        Returns
        -------
        """
        try:
            del self.curve  # this is fine and as it should be to reset cached_properties, linters are stupid
        except AttributeError:
            # This means that the curve was never called before
            ...
        self._bezier_points *= other
        return self

    __rmul__ = __mul__

    def __add__(self, other: AbstractBezierCurve):
        """
        This addition is performed over the unit interval. So the given AbstractBezierCurve must be defined over the
        unit interval (0,1). In this case the two Bezier points are added up and form a new curve.

        Parameters
        ----------
        other: AbstractBezierCurve:
            Bezier points to add

        Returns
        -------
        """
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
    Class for creating a 2-dimensional Bezier Curve by using the De Casteljau Algorithm but with symbolic execution.
    Which means the Algorithm is executed just once to generate a symbolic function depending on the parameter.
    This function gets lambdified and points are calculated by substituting the parameter.

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """

    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        Method returns the function to calculate all values at once in this case it is a symbolic function that is
        lambdified after the abstract de Castelljau iterations.

        Returns
        -------
        Callable[[float], np.ndarray]:
            lambdified symbolic function
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
    Class for creating a 2-dimensional Bezier Curve by using the De Casteljau Algorithm.

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """

    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        Method returns the function to calculate all values at once in this case it is the standard de Castelljau.

        Returns
        -------
        Callable[[float], np.ndarray]:
            de Castelljau routine from curvepy.de_caes
        """
        return partial(de_caes, self._bezier_points)


class BezierCurveBernstein(AbstractBezierCurve):
    """
    Class for creating a 2-dimensional Bezier Curve by using the Bernstein polynomials.

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """

    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        Method returns the function to calculate all values at once in this case it is the formula 5.8.

        Returns
        -------
        Callable[[float], np.ndarray]:
            bezier_curve_with_bernstein method
        """
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
        \\[b_i^r(t) = \\sum_{j=0}^r b_{i+j} \\cdot B_i^r(t)\\]
        """
        m = self._bezier_points
        _, n = m.shape
        t = (t - interval[0]) / (interval[1] - interval[0])
        return np.sum([m[:, i] * bernstein_polynomial(n - r - 1, i, t) for i in range(n - r)], axis=0)


class BezierCurveHorner(AbstractBezierCurve):
    """
    Class for creating a 2-dimensional Bezier Curve by using a horner like scheme to calculate points on the curve.

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """

    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        Method returns the function to calculate all values at once in this case a horner like scheme is used.

        Returns
        -------
        Callable[[float], np.ndarray]:
            horn_bez function
        """
        return self.horn_bez

    def horn_bez(self, t: float = 0.5) -> np.ndarray:
        """
        Method using horner like scheme to calculate point on curve given some t

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
    """
    Class for creating a 2-dimensional Bezier Curve by using the monomial form of Bezier curves

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """

    def init_func(self) -> Callable[[float], np.ndarray]:
        """
        Method calculating monomial representation of given bezier form using 5.27

        Returns
        -------
        Callable[[float], np.ndarray]:
            bezier function in polynomial form

        Notes
        -----
        Equation 5.27 used for computing polynomial form:
        \\[b^n(t) = \\sum_{j=0}^n \\binom{n}{j} \\Delta^j b_0 t^j\\]

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
    """
    Class for creating a 2-dimensional Bezier Curve by using Subdivision.

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """
    def init_func(self) -> Callable[[float], np.ndarray]:
        # dummy, just used for __call__ and __getitem__
        """
        In this case it is only a dummy func, so it can be used in __call__ and __getitem__. The standard Castelljau
        method from curvepy.de_caes is used.

        Returns
        -------
        Callable[[float], np.ndarray]:
            de Castelljau routine from curvepy.de_caes
        """
        return partial(de_caes, self._bezier_points)

    serial_execution = None
    parallel_execution = None

    @classmethod
    def from_round_number(cls, m: np.ndarray, approx_rounds: int = 5,
                          interval: Tuple[int, int] = (0, 1)) -> BezierCurveApproximation:
        """
        Class Method creates a BezierCurveApproximation from given control points.

        Parameters
        ----------
        m: np.ndarray:
            Bezier points

        approx_rounds: int:
            amount of subdivision rounds

        interval: Tuple[int, int]:
            the curve expects parameters in this interval. Default value is (0,1)


        Returns
        -------
        BezierCurveApproximation:
            constructed BezierCurveApproximation from given Bezier points
        """
        return cls(m, cls.approx_rounds_to_cnt_ts(approx_rounds, m.shape[1]), False, interval)

    @staticmethod
    def cnt_ts_to_approx_rounds(cnt_ts, cnt_bezier_points) -> int:
        """
        In the normal case the Bezier classes calculate points based on the amount of given parameter values t. Since
        subdivision works a bit differently as we get a fixed amount of points in each iteration. We want to approximate
        how many rounds subdivision would need to compute the given number of points.

        Parameters
        ----------
        cnt_ts: int:
            amount of points

        cnt_bezier_points: int:
            amount of control points

        Returns
        -------
        int:
            amount of approximation rounds needed to achieve the given number of points
        """
        return math.ceil(math.log(cnt_ts / cnt_bezier_points, 2))

    @staticmethod
    def approx_rounds_to_cnt_ts(approx_rounds, cnt_bezier_points) -> int:
        """
        This staticmethod functions as the inverse of cnt_ts_to_approx_rounds. If the amount of points is needed
        that approx_rounds iterations of subdivision would compute, we can transform the number of iterations
        into the corresponding amount of points.

        Parameters
        ----------
        approx_rounds: int:
            amount of subdivision rounds

        cnt_bezier_points: int:
            amount of control points

        Returns
        -------
        int:
            amount of points resulting from approx_rounds iteration of subdivision
        """
        return 2 ** approx_rounds * cnt_bezier_points

    @cached_property
    def curve(self) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[float]]]:
        """
        Cached Property returning coordinates of all calculated points. In this routine approx_rounds iterations of
        subdivision are performed. The method always subdivides t = 0.5.

        Returns
        -------
            Union[Tuple[float, float], Tuple[float, float, float]]:
            first list for x coords, second for y coords and third for z if existing
        """
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
            assert (n / 2).is_integer()
            return ret[:n // 2], ret[n // 2:]
        assert self._dimension == 3
        assert (n / 3).is_integer()
        return ret[:n // 3], ret[n // 3:2 * n // 3], ret[2 * n // 3:]
