import numpy as np
import sympy as sy
import matplotlib.pyplot as plt
import threading as th
import shapely.geometry as sg
import scipy.optimize as sco
from typing import Tuple, Callable

lockMe = th.Lock()  # variable used to control access of threads


class Tholder:
    """
    class holds Array with equidistant ts in [0,1] of length n

    Parameters
    ----------
    n: int
        Numbers of ts to calculate

    Attributes
    -------
    _tArray : np.ndArray
        array in which all ts are stored
    _pointer : int
        pointer pointing on next t to get
    """

    def __init__(self, n: int = 1) -> None:
        self._tArray = np.linspace(0, 1, n)
        self._pointer = 0

    def get_next_t(self) -> float:
        """
        method for threads to get next t

        Parameters
        ----------

        Returns
        -------
        float:
            t to calculate next De Casteljau step
        """
        if self._pointer == len(self._tArray):
            return -1
        res = self._tArray[self._pointer]
        self._pointer += 1
        return res


class CasteljauThread(th.Thread):
    """
    Init belonging to class CasteljauThread

    Parameters
    ----------
    ts_holder: Tholder
        Class which yields all ts
    c: np.ndArray
        Array with control Points
    f: Function
         Function to transform t if necessary

    Attributes
    -------
    _ts_holder : Tholder
        instance of class Tholder so thread can get ts for calculating de Casteljau
    _coords: np.ndArray
        original control points
    _res: list
        actual points on curve
    _func: Function
        function for transforming t
    """

    def __init__(self, ts_holder: Tholder, c: np.ndarray, f: Callable[[float], float] = lambda x: x) -> None:
        th.Thread.__init__(self)
        self._ts_holder = ts_holder
        self._coords = c
        self._res = []
        self._func = f

    def get_res(self) -> list:
        """
        method to get List with calculated points

        Parameters
        ----------

        Returns
        -------
        List containing all points calculated by the thread instance
        """
        return self._res

    def de_caes(self, t: float, n: int) -> None:
        """
        method implementing de Casteljau algorithm

        Parameters
        ----------
        t: float
            t used in the calculation
        n: int
            number of iterations

        Returns
        -------
        None
        """
        m = self._coords.copy()
        t = self._func(t)
        for r in range(n):
            m[:, :(n - r - 1)] = (1 - t) * m[:, :(n - r - 1)] + t * m[:, 1:(n - r)]
        self._res.append(m[:, 0])

    def run(self) -> None:
        """
        run method calculates points until depot is Empty

        Parameters
        ----------

        Returns
        -------
        None
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
    Init belonging to BezierCurve2D

    Parameters
    ----------
    m: np.ndArray
        Array containing control points
    cnt_ts: int
        number of ts to create

    Attributes
    -------
    _bezier_points: np.ndArray
        array containing original bezier points
    _cnt_ts: int
        numbers of equidistant ts to calculate
    _curve: list
        list containing points belonging to actual curve
    _box: list
        points describing minmax box of curve
    """

    def __init__(self, m: np.ndarray, cnt_ts: int = 1000) -> None:
        self._bezier_points = m
        self._cnt_ts = cnt_ts
        self._func = self.init_func(m)
        self._curve = self.init_curve()
        self._box = []
        self.min_max_box()

    def init_func(self, m: np.ndarray) -> Callable:
        _, n = m.shape
        m = sy.Matrix(m)
        t = sy.symbols('t')
        for r in range(n):
            m[:, :(n - r - 1)] = (1 - t) * m[:, :(n - r - 1)] + t * m[:, 1:(n - r)]
        return sy.lambdify(t, m[:, 0])

    def init_curve(self) -> np.ndarray:
        ts = np.linspace(0, 1, self._cnt_ts)
        return np.array([self._func(t) for t in ts])

    def get_box(self) -> list:
        """
        method returns minmax box of calculated curve

        Parameters
        ----------

        Returns
        -------
        list:
            contains points describing the box
        """
        return self._box

    def get_curve(self) -> Tuple[list, list]:
        """
        method return x and y coordinates of all calculated points

        Parameters
        ----------

        Returns
        -------
        lists:
            first list for x coords, second for y coords
        """
        tmp = np.ravel(self._curve)
        return tmp[0::2], tmp[1::2]

    def get_func(self) -> Callable:
        return self._func

    def de_casteljau_threading(self, cnt_threads: int = 4) -> None:
        """
        method implementing the threading for de Casteljau algorithm

        Parameters
        ----------
        cnt_threads: int
            number of threads to use

        Returns
        -------
        None
        """
        ts = Tholder(self._cnt_ts)
        threads = []

        for _ in range(cnt_threads):
            threads.append(CasteljauThread(ts, self._bezier_points))

        for t in threads:
            t.start()

        for t in threads:
            t.join()
            tmp = t.get_res()
            self._curve = self._curve + tmp

        self.min_max_box()

    def intersect(self, t1: tuple, t2: tuple) -> bool:
        """
        Method checking intersection of two given tuples

        Parameters
        ----------
        t1: tuple
            tuple 1
        t2: tuple
            tuple 2

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
        method creates minmax box for the corresponding curve

        Parameters
        ----------

        Returns
        -------
        None
        """
        xs = [*self._bezier_points[0, :]]
        ys = [*self._bezier_points[1, :]]
        xs.sort()
        ys.sort()
        self._box = [(xs[0], xs[-1]), (ys[0], ys[-1])]

    def collision_check(self, other_curve) -> bool:
        """
        method checking collision with given curve.
        first box check, if false return otherwise checking actual curves

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
        method checking box collision with given curve.

        Parameters
        ----------
        other_curve: BezierCurve2D
            curve to check

        Returns
        -------
        bool:
            true if boxes collide otherwise false
        """
        o_box = other_curve.get_box()
        box = self._box
        for t1, t2 in zip(box, o_box):
            if not self.intersect(t1, t2):
                return False

        return True

    def curve_collision_check(self, other_curve) -> bool:
        """
        method checking curve collision with given curve.

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
        method plotting the curve

        Parameters
        ----------

        Returns
        -------
        None
        """
        xs, ys = self.get_curve()
        plt.plot(xs, ys, 'o')

    def show_funcs(self, list_of_curves: list = []) -> None:
        """
        method plotting multiple Beziercurves in one figure

        Parameters
        ----------
        list_of_curves:
            curves to plot

        Returns
        -------
        None
        """
        self.plot()
        if not list_of_curves:
            plt.show()
            return
        for c in list_of_curves:
            c.plot()
        plt.show()


class BezierCurve3D(BezierCurve2D):
    """
    Init belonging to BezierCurve3D

    Parameters
    ----------
    m: np.ndArray
        Array containing control points
    cnt_ts: int
        number of ts to create

    Attributes
    -------
    see BezierCurve2D
    """

    def __init__(self, m: np.ndarray, cnt_ts: int = 1000) -> None:
        super().__init__(m, cnt_ts)

    def get_curve(self) -> Tuple[list, list, list]:
        """
        method return x, y and z coords of all calculated points

        Parameters
        ----------

        Returns
        -------
        lists:
            first list for x coords, second for y coords and third for z coords
        """

        tmp = np.ravel(self._curve)
        return tmp[0::3], tmp[1::3], tmp[2::3]

    def min_max_box(self) -> None:
        """
        method creates minmax box for the corresponding curve

        Parameters
        ----------

        Returns
        -------
        None
        """
        super().min_max_box()
        zs = [*self._bezier_points[2, :]]
        zs.sort()
        self._box.append((zs[0], zs[-1]))

    def collision_check(self, other_curve) -> bool:
        """
        method checking collision with given curve.
        first box check, if false return otherwise checking actual curves

        Parameters
        ----------
        other_curve: BezierCurve3D
            curve to check

        Returns
        -------
        bool:
            true if curves collide otherwise false
        """
        if not self.box_collision_check(other_curve):
            return False

        return self.curve_collision_check(other_curve)

    def curve_collision_check(self, other_curve) -> bool:
        """
        method checking curve collision with given curve.

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
        method plotting th curve

        Parameters
        ----------

        Returns
        -------
        None
        """
        xs, ys, zs = self.get_curve()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xs, ys, zs)
        plt.show()


def csv_read(file_path: str) -> np.ndarray:
    try:
        with open(file_path, 'r') as csv_file:
            xs, ys, zs = [], [], []
            for line in csv_file:
                try:
                    x, y, z = line.split(',')
                    zs.append(float(z))
                except ValueError:
                    try:
                        x, y = line.split(',')
                    except ValueError:
                        print('Expected two or three values per line')
                        return np.array([])
                xs.append(float(x))
                ys.append(float(y))
        return np.array([xs, ys], dtype=float) if not zs else np.array([xs, ys, zs], dtype=float)
    except FileNotFoundError:
        print(f'File: {file_path} does not exist.')
        return np.array([])


def init() -> None:
    m = csv_read('test.csv')  # reads csv file with bezier points
    b1 = BezierCurve2D(m)
    m = csv_read('test2.csv')  # reads csv file with bezier points
    b2 = BezierCurve2D(m)
    b2.show_funcs([b1])
    f = b1.get_func()
    g = b2.get_func()

    def h(x):
        return f(x) - g(x)
    #print([h(t).ravel() for t in np.linspace(0, 1, 100)])

    print(sco.newton(lambda x: f(x) - g(x), np.array([0.5])))


if __name__ == "__main__":
    init()
