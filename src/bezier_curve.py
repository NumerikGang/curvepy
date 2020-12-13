import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import threading as th
import shapely.geometry as sg
from abc import ABC, abstractmethod
from scipy.special import comb
from typing import Tuple, Callable, Union, Any

from src.utilities import csv_read


class Tholder:
    """
    Class holds Array with equidistant ts in [0,1] of length n

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
    ts_holder: Tholder
        Class which yields all ts
    c: np.ndarray
        Array with control points
    f: Function
         Function to transform t if necessary

    Attributes
    -------
    _ts_holder : Tholder
        instance of class Tholder so thread can get the ts for calculating the de Casteljau algorithm
    _coords: np.ndarray
        original control points
    res: list
        actual points on curve
    _func: Function
        function for transforming t
    """

    def __init__(self, ts_holder: Tholder, c: np.ndarray, f: Callable[[float], float] = lambda x: x) -> None:
        th.Thread.__init__(self)
        self._ts_holder = ts_holder
        self._coords = c
        self.res = []
        self._func = f

    def de_caes(self, t: float, n: int) -> None:
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
            self.de_caes(t, n)


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

    def de_casteljau_threading(self, ts, cnt_threads: int = 4) -> None:
        """
        Method implementing the threading for the De Casteljau algorithm

        Parameters
        ----------
        cnt_threads: int
            number of threads to use
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


def init() -> None:
    m = csv_read('test.csv')  # reads csv file with bezier points
    b1 = BezierCurve(m)
    m = csv_read('test2.csv')  # reads csv file with bezier points
    b2 = BezierCurve(m)
    b2.show_funcs([b1])


if __name__ == "__main__":
    init()
