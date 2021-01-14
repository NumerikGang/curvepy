"""
Multiple Bézier curve implementations with different computations.

The _Bézier curve_ is one of the most infamous functions in Computer Aided Geometric Design (CAGD) as well as other topics
of computer graphics.

# Definition

Although there are many ways to define a bezier curve, the most common ones are as follows:

## 1. Bézier curve with _Bernstein polynomial_ basis

Let the \(i\)-th Bernstein polynomial of Degree \(n\) defined on the unit interval \([0,1]\) as
\[
B_i^n(t) := \\binom{n}{i} t^i (1-t)^{n-i}
\]

where the binomial coefficients are given by
\[
\\binom{n}{i} =
\\begin{cases}
\\frac{n!}{i!(n-i)!} & \\text{if } & 0 \\leq i \\leq n \\\\
0                    & \\text{else}
\\end{cases}
\]

Further let \(b_i\), \(0 \\leq i \\leq n\) denote the _Bézier points_.
Then the Bézier curve is defined as
\[
b(t) := \\sum_{i=0}^n b_i B_i^n(t)
\]

## 2. Bézier curve recursion

A Bézier curve can also be defined through the following recursion:

Let \(b_0,\\dots,b_n \in \\mathbb{E}^3, t \in \\mathbb{R} \), and

\[
\\begin{align*}
b_i^0(t) &:= b_i\\\\
b_i^r(t) &:= (1-t) b_i^{r-1}(t) + t b_{i+1}^{r-1}(t)
\\end{align*}
\]

# Properties
The properties of a Bézier curve are explained [here](./tests/property_based_tests/test_bezier_curve.html).
"""
import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import threading as th
import shapely.geometry as sg
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Union


class _Tholder:
    """
    Helper for scheduling n equidistant values in the unit interval [0,1].

    Parameters
    ----------
    n: int
        Numbers of ts to calculate

    Attributes
    -------
    _tArray : np.ndarray
        array in which all ts are stored
    _pointer : int
        pointer pointing on next t to get
    lockMe: th.Lock
        lock for multithreading
    """

    def __init__(self, ts: np.ndarray = None) -> None:
        self._tArray = ts
        self._pointer = 0
        self.lockMe = th.Lock()  # variable used to control access of threads

    def get_next_t(self) -> float:
        """
        Method for threads to get next t

        Returns
        -------
        float:
            t to calculate the next De Casteljau step
        """
        if self._pointer == len(self._tArray):
            return -1
        res = self._tArray[self._pointer]
        self._pointer += 1
        return res


class CasteljauThread(th.Thread):
    """
    Thread class computing the De Casteljau algorithm

    Parameters
    ----------
    ts_holder: _Tholder
        Class which yields all ts
    c: np.ndarray
        Array with control points
    f: Function
         Function to transform t if necessary

    Attributes
    -------
    _ts_holder : _Tholder
        instance of class Tholder so thread can get the ts for calculating the de Casteljau algorithm
    _coords: np.ndarray
        original control points
    res: list
        actual points on curve
    _func: Function
        function for transforming t
    """

    def __init__(self, ts_holder: _Tholder, c: np.ndarray, f: Callable[[float], float] = lambda x: x) -> None:
        th.Thread.__init__(self)
        self._ts_holder = ts_holder
        self._coords = c
        self.res = []
        self._func = f

    def _de_caes(self, t: float, n: int) -> None:
        """
        Method implementing the the De Casteljau algorithm

        Parameters
        ----------
        t: float
            value at which to be evaluated at
        n: int
            number of iterations TODO find more describing phrasing together
        """
        m = self._coords.copy()
        t = self._func(t)
        for r in range(n):
            m[:, :(n - r - 1)] = (1 - t) * m[:, :(n - r - 1)] + t * m[:, 1:(n - r)]
        self.res.append(m[:, 0])

    def run(self) -> None:
        """
        Method calculates points until depot is empty
        """
        _, n = self._coords.shape
        while True:
            self._ts_holder.lockMe.acquire()
            t = self._ts_holder.get_next_t()
            self._ts_holder.lockMe.release()
            if t == -1:
                break
            self._de_caes(t, n)


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
    _func: Callable
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
        self._func = self.init_func(m)
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
            self._curve = self._func(ts)
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
        other_curve: Union[BezierCurve2D, BezierCurve3D]
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
        if not list_of_curves:
            plt.show()
            return
        for c in list_of_curves:
            c.plot()
        plt.show()

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

    def de_casteljau_threading(self, ts: np.ndarray = None, cnt_threads: int = 4) -> None:
        """
        Method implementing the threading for the De Casteljau algorithm

        Parameters
        ----------
        cnt_threads: int
            number of threads to use

        ts: np.ndarray:
            array containing all ts used to calculate points

        """
        ts = _Tholder(ts)
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
