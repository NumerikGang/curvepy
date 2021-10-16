from __future__ import annotations  # Needed until Py3.10, see PEP 563
import numpy as np
import sympy as sy
import scipy.special as scs
import matplotlib.pyplot as plt
import shapely.geometry as sg
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Union
from functools import partial, cached_property
import concurrent.futures
from multiprocessing import cpu_count

from curvepy.de_caes import de_caes, subdivision
from curvepy.utilities import min_max_box
from curvepy.types import bernstein_polynomial, Polygon


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
        self._bezier_points = m
        self._dimension = self._bezier_points.shape[0]
        self._cnt_ts = cnt_ts
        self.interval = interval
        self.func = self.init_func()
        self._use_parallel = use_parallel

    @abstractmethod
    def init_func(self) -> Callable[[float], np.ndarray]:
        """
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

    @cached_property
    def curve(self) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[float]]]:
        """
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

    @staticmethod
    def intersect(t1: np.ndarray, t2: np.ndarray) -> bool:
        """
        Method checking intersection of two given tuples TODO find more describing phrasing together

        Parameters
        ----------
        t1: tuple
            first tuple
        t2: tuple
            second tuple

        Returns
        -------
        bool:
            true if intersect otherwise false
        """
        return t2[0] <= t1[0] <= t2[1] \
               or t2[0] <= t1[1] <= t2[1] \
               or t1[0] <= t2[0] <= t1[1] \
               or t1[0] <= t2[1] <= t1[1]

    @cached_property
    def min_max_box(self) -> List[np.ndarray]:
        """
        Method creates minmax box for the corresponding curve
        """
        return min_max_box(self._bezier_points)

    def collision_check(self, other_curve: BezierCurve) -> bool:
        """
        Method checking collision with given curve.
        Starts with a box check, if this didn't intersect it is checking the actual curves

        Parameters
        ----------
        other_curve: BezierCurve2D
            curve to check

        Returns
        -------
        bool:
            true if curves collide otherwise false
        """
        if not self.box_collision_check(other_curve):
            return False

        return self.curve_collision_check(other_curve)

    def box_collision_check(self, other_curve: BezierCurve) -> bool:
        """
        Method checking box collision with the given curve.

        Parameters
        ----------
        other_curve: BezierCurve2D
            curve to check

        Returns
        -------
        bool:
            true if boxes collide otherwise false
        """
        o_box = other_curve.min_max_box
        box = self.min_max_box
        for t1, t2 in zip(box, o_box):
            if not self.intersect(t1, t2):
                return False

        return True

    def curve_collision_check(self, other_curve: BezierCurve) -> bool:
        """
        Method checking curve collision with the given curve.

        Parameters
        ----------
        other_curve: Union[BezierCurve2D, BezierCurve3D] # TODO: check whether same type as everywhere
            curve to check

        Returns
        -------
        bool:
            true if curves collide otherwise false
        """
        tuple_of_dimensions = self.curve
        f1 = sg.LineString(np.column_stack(tuple_of_dimensions))
        tuple_of_dimensions = other_curve.curve
        f2 = sg.LineString(np.column_stack(tuple_of_dimensions))
        inter = f1.intersection(f2)
        return not inter.geom_type == 'LineString'

    def plot(self) -> None:
        """
        Method plotting the curve by adding it to the current pyplot figure
        """
        if self._dimension == 2:
            plt.plot(*self.curve, 'o')
        else:
            ax = plt.axes(projection='3d')
            ax.scatter3D(*self.curve)

    def show_funcs(self, list_of_curves: list = None) -> None:
        """
        Method plotting multiple Bezier Curves in one figure

        Parameters
        ----------
        list_of_curves:
            curves to plot
        """

        if list_of_curves is None:
            list_of_curves = []

        self.plot()
        # TODO: debate whether the if below should be thrown away
        # this is possible since we would just iterate through an empty list
        # This shouldn't be that much faster since we check if the list is empty anyways
        # but it would reduce noise
        if not list_of_curves:
            plt.show()
            return
        for c in list_of_curves:
            c.plot()
        plt.show()

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

    def all_forward_differences(self, i: int = 0) -> np.ndarray:
        """
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
        _, n = self._bezier_points.shape
        diff = [self.single_forward_difference(i, r) for r in range(0, n)]
        return np.array(diff).T

    def derivative_bezier_curve(self, t: float = 1, r: int = 1) -> np.ndarray:
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
        _, n = self._bezier_points.shape
        factor = scs.factorial(n) / scs.factorial(n - r)
        tmp = [factor * self.single_forward_difference(j, r) * bernstein_polynomial(n - r, j, t) for j in range(n - r)]
        return np.sum(tmp, axis=0)

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

        t: float:
            value for which Bezier curves are calculated

        r: int:
            optional Parameter to calculate only a partial curve if we already have some degree of the bezier points

        interval: Tuple[float,float]:
            Interval of t used for affine transformation

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
        t = (u - a) / (b - a)
        return self.func((1 - t) * a + t * b)

    def __mul__(self, other: Union[float, int]):
        del self.curve  # this is fine and as it should be to reset cached_properties, linters are stupid
        self._bezier_points *= other
        return self

    __rmul__ = __mul__

    # TODO write in docstrings that anything other than unit intervals are forbidden
    def __add__(self, other: AbstractBezierCurve):
        if not isinstance(other, AbstractBezierCurve):
            raise TypeError("")

        del self.curve  # this is fine

        self._bezier_points += other._bezier_points
        self._cnt_ts = max(self._cnt_ts, other._cnt_ts)
        return self


class BezierCurve(AbstractBezierCurve):
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


# TODO Interval
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


class HornerBezierCurve(AbstractBezierCurve):

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


class MonomialBezierCurve(AbstractBezierCurve):

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

        Initially the method would only compute the polynomial coefficients in an array, and parsing this array with a given
        t to the horner method we would get a point back. Instead the method uses sympy to calculate a function depending on
        t. After initial computation, f(t) calculates the value for a given t. Having a function it is simple to map it on
        an array containing multiple values for t.
        As a result we do not need to call the horner method for each t individually.
        """
        m = self._bezier_points
        _, n = m.shape
        diff = self.all_forward_differences()
        t = sy.symbols('t')
        res = 0
        for i in range(n):
            res += scs.binom(n - 1, i) * diff[:, i] * t ** i

        return sy.lambdify(t, res)


class BezierCurveApproximation(AbstractBezierCurve):
    def __init__(self, m: np.ndarray, approx_rounds: int = 5, interval: Tuple[int, int] = (0, 1)) -> None:
        # cnt_ts = approx_rounds * m.shape[1]
        AbstractBezierCurve.__init__(self, m, 2**approx_rounds*m.shape[1], False, interval)
        self._approx_rounds = approx_rounds

    init_func = None

    @cached_property
    def curve(self) -> Union[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[float]]]:
        current = [self._bezier_points]
        for _ in range(self._approx_rounds):
            queue = []
            for c in current:
                queue.extend(subdivision(c))
            current = queue

        ret = np.hstack(current)
        ret = np.ravel(ret)
        n = len(ret)
        if self._dimension == 2:
            return ret[:n/2], ret[n/2:]
        return ret[:n/3], ret[n/3:2*n/3], ret[2*n/3:]
