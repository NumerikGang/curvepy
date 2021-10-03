import numpy as np
import sympy as sy
import scipy.special as scs
import matplotlib.pyplot as plt
import threading as th
import shapely.geometry as sg
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Union
from functools import partial

from curvepy.utilities import csv_read
from curvepy.types import bernstein_polynomial


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
    _curve: list
        list containing points belonging to actual curve
    box: list
        points describing minmax box of curve
    """

    def __init__(self, m: np.ndarray, cnt_ts: int = 1000) -> None:
        self._bezier_points = m
        self._dimension = self._bezier_points.shape[0]
        self._cnt_ts = cnt_ts
        self.func = self.init_func(m)
        self._curve = None
        self.box = []

    @abstractmethod
    def init_func(self, m: np.ndarray) -> Callable:
        """
        Method returns the function to calculate all values at once.

        Parameters
        ----------
        m: np.ndarray:
            array containing the Bezier Points

        Returns
        -------
        Callable:
            function representing the Bezier Curve
        """
        ...

    def _get_or_create_values(self) -> np.ndarray:
        """
        Method returns minmax box of calculated curve

        Returns
        -------
        np.ndarray:
            calculating no. of points defined by variable cnt_ts
        """
        if self._curve is None:
            ts = np.linspace(0, 1, self._cnt_ts)
            self._curve = self.func(ts)
            self.min_max_box()
        return self._curve

    @property
    def curve(self) -> Union[Tuple[list, list], Tuple[list, list, list]]:
        """
        Method returning coordinates of all calculated points

        Returns
        -------
        Union[tuple[list, list], tuple[list, list, list]]:
            first list for x coords, second for y coords and third for z if existing
        """
        tmp = np.ravel([*self._get_or_create_values()])
        if self._dimension == 2:
            return tmp[0::2], tmp[1::2]
        return tmp[0::3], tmp[1::3], tmp[2::3]

    @staticmethod
    def intersect(t1: tuple, t2: tuple) -> bool:
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

    def min_max_box(self) -> None:
        """
        Method creates minmax box for the corresponding curve
        """
        xs = sorted([*self._bezier_points[0, :]])
        ys = sorted([*self._bezier_points[1, :]])
        if self._dimension == 2:
            self.box = [(xs[0], xs[-1]), (ys[0], ys[-1])]
            return
        zs = sorted([*self._bezier_points[2, :]])
        self.box.append((zs[0], zs[-1]))

    def collision_check(self, other_curve) -> bool:
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

    def box_collision_check(self, other_curve) -> bool:
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
        o_box = other_curve.box
        box = self.box
        for t1, t2 in zip(box, o_box):
            if not self.intersect(t1, t2):
                return False

        return True

    def curve_collision_check(self, other_curve) -> bool:
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
        return np.sum([scs.binom(r, j) * (-1) ** (r - j) * self._bezier_points[:, i + j] for j in range(0, r + 1)], axis=0)

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

    def init_func(self, m: np.ndarray) -> Callable:
        """
        Method returns the function to calculate all values at once.

        Parameters
        ----------
        m: np.ndarray:
            array containing the Bezier Points

        Returns
        -------
        Callable:
            function representing the Bezier Curve
        """
        _, n = m.shape
        m = sy.Matrix(m)
        t = sy.symbols('t')
        for r in range(n):
            m[:, :(n - r - 1)] = (1 - t) * m[:, :(n - r - 1)] + t * m[:, 1:(n - r)]
        f = sy.lambdify(t, m[:, 0])
        return np.frompyfunc(f, 1, 1)


class BezierCurveThreaded(AbstractBezierCurve):
    """
    Class for creating a 2-dimensional Bezier Curve by using the threaded De Casteljau Algorithm

    Parameters
    ----------
    see AbstractBezierCurve

    Attributes
    -------
    see AbstractBezierCurve
    """

    def init_func(self, m: np.ndarray) -> Callable:
        """
        Method returns the function to calculate all values at once.

        Parameters
        ----------
        m: np.ndarray:
            array containing the Bezier Points

        Returns
        -------
        Callable:
            function representing the Bezier Curve
        """
        return self.de_casteljau_threading

    def de_casteljau_threading(self, ts: np.ndarray = None, cnt_threads: int = 4) -> List[np.ndarray]:
        """
        Method implementing the threading for the De Casteljau algorithm

        Parameters
        ----------
        cnt_threads: int
            number of threads to use

        ts: np.ndarray:
            array containing all ts used to calculate points

        """
        ts = Tholder(ts)
        threads = []
        curve = []

        for _ in range(cnt_threads):
            threads.append(CasteljauThread(ts, self._bezier_points))

        for t in threads:
            t.start()

        for t in threads:
            t.join()
            tmp = t.res
            curve = curve + tmp

        return curve


class BezierCurveBernstein(AbstractBezierCurve):

    def init_func(self, m: np.ndarray) -> Callable:
        ...

    def bezier_curve_with_bernstein(self, t: float = 0.5, r: int = 0,
                                    interval: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """
        Method using 5.8 to calculate a point with given bezier points

        Parameters
        ----------
        m: np.ndarray:
            array containing the Bezier Points

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



def init() -> None:
    m = csv_read('test.csv')  # reads csv file with bezier points
    b1 = BezierCurve(m)
    print(b1.func(0.5))
    m = csv_read('test2.csv')  # reads csv file with bezier points
    b2 = BezierCurve(m)
    b2.show_funcs([b1])


if __name__ == "__main__":
    init()
