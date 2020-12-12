import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import threading as th
import shapely.geometry as sg
from typing import Tuple, Callable

from src.utilities import csv_read

lockMe = th.Lock()  # variable used to control access of threads


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
    """

    def __init__(self, n: int = 1) -> None:
        self._tArray = np.linspace(0, 1, n)
        self._pointer = 0

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
            lockMe.acquire()
            t = self._ts_holder.get_next_t()
            lockMe.release()
            if t == -1:
                break
            self.de_caes(t, n)


class BezierCurve2D:
    """
    Class for creating a 2-dimensional Bezier Curve

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
        self._cnt_ts = cnt_ts
        self.func = self.init_func(m)
        self._curve = self.init_curve()
        self.box = []
        self.min_max_box()

    def init_func(self, m: np.ndarray) -> Callable:
        """
        Method returns minmax box of calculated curve

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

    def init_curve(self) -> np.ndarray:
        """
        Method returns minmax box of calculated curve

        Returns
        -------
        np.ndarray:
            calculating no. of points defined by variable cnt_ts
        """
        ts = np.linspace(0, 1, self._cnt_ts)
        return self.func(ts)

    def get_curve(self) -> Tuple[list, list]:
        """
        Method returning x and y coordinates of all calculated points

        Returns
        -------
        lists:
            first list for x coords, second for y coords
        """
        tmp = np.ravel([*self._curve])
        return tmp[0::2], tmp[1::2]

    def de_casteljau_threading(self, cnt_threads: int = 4) -> None:
        """
        Method implementing the threading for the De Casteljau algorithm

        Parameters
        ----------
        cnt_threads: int
            number of threads to use
        """
        ts = Tholder(self._cnt_ts)
        threads = []

        for _ in range(cnt_threads):
            threads.append(CasteljauThread(ts, self._bezier_points))

        for t in threads:
            t.start()

        for t in threads:
            t.join()
            tmp = t.res
            self._curve = self._curve + tmp

        self.min_max_box()

    def intersect(self, t1: tuple, t2: tuple) -> bool:
        """
        Method checking intersection of two given tuples

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
        xs = [*self._bezier_points[0, :]]
        ys = [*self._bezier_points[1, :]]
        xs.sort()
        ys.sort()
        self.box = [(xs[0], xs[-1]), (ys[0], ys[-1])]

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
        other_curve: BezierCurve2D
            curve to check

        Returns
        -------
        bool:
            true if curves collide otherwise false
        """
        xs, ys = self.get_curve()
        f1 = sg.LineString(np.column_stack((xs, ys)))
        xs, ys = other_curve.get_curve()
        f2 = sg.LineString(np.column_stack((xs, ys)))
        inter = f1.intersection(f2)
        return not inter.geom_type == 'LineString'

    def plot(self) -> None:
        """
        Method plotting the curve
        """
        xs, ys = self.get_curve()
        plt.plot(xs, ys, 'o')

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


class BezierCurve3D(BezierCurve2D):
    """
    Class for creating a 3-dimensional Bezier Curve

    Parameters
    ----------
    see BezierCurve2D

    Attributes
    -------
    see BezierCurve2D
    """

    def get_curve(self) -> Tuple[list, list, list]:
        """
        Method return x, y and z coords of all calculated points

        Returns
        -------
        lists:
            first list for x coords, second for y coords and third for z coords
        """

        tmp = np.ravel([*self._curve])
        return tmp[0::3], tmp[1::3], tmp[2::3]

    def min_max_box(self) -> None:
        """
        Method creates minmax box for the corresponding curve
        """
        super().min_max_box()
        zs = [*self._bezier_points[2, :]]
        zs.sort()
        self.box.append((zs[0], zs[-1]))

    def curve_collision_check(self, other_curve) -> bool:
        """
        Method checking curve collision with given curve.

        Parameters
        ----------
        other_curve: BezierCurve3D
            curve to check

        Returns
        -------
        bool:
            true if curves collide otherwise false
        """
        xs, ys, zs = self.get_curve()
        f1 = sg.LineString(np.column_stack((xs, ys, zs)))
        xs, ys, zs = other_curve.get_curve()
        f2 = sg.LineString(np.column_stack((xs, ys, zs)))
        inter = f1.intersection(f2)
        return not inter.geom_type == 'LineString'

    def plot(self) -> None:
        """
        Method plotting the curve
        """
        xs, ys, zs = self.get_curve()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xs, ys, zs)
        plt.show()


def init() -> None:
    m = csv_read('test.csv')  # reads csv file with bezier points
    b1 = BezierCurve2D(m)
    m = csv_read('test2.csv')  # reads csv file with bezier points
    b2 = BezierCurve2D(m)
    b2.show_funcs([b1])


if __name__ == "__main__":
    init()
