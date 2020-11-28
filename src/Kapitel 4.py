import numpy as np
import matplotlib.pyplot as plt
import threading as th
import shapely.geometry as sg
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

    """
    method for threads to get next t 

    Parameters
    ----------
    None

    Returns
    -------
    float:
        t to calculate next De Casteljau step
    """

    def get_next_t(self) -> float:
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

    """
    method to get List with calculated points

    Parameters
    ----------
    None

    Returns
    -------
    List containing all points calculated by the thread instance
    """

    def get_res(self) -> list:
        return self._res

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

    def de_caes(self, t: float, n: int) -> None:
        m = self._coords.copy()
        t = self._func(t)
        for r in range(n):
            m[:, :(n - r - 1)] = (1 - t) * m[:, :(n - r - 1)] + t * m[:, 1:(n - r)]
        self._res.append(m[:, 0])

    """
    run method calculates points until depot is Empty

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    def run(self) -> None:
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
        self._curve = []
        self._box = []

    """
    method returns minmax box of calculated curve

    Parameters
    ----------
    None

    Returns
    -------
    list:
        contains points describing the box
    """

    def get_box(self) -> list:
        return self._box

    """
    method return x and y coordinates of all calculated points

    Parameters
    ----------
    None

    Returns
    -------
    lists:
        one list for x coords, one for y coords
    """

    def get_curve(self) -> Tuple[list, list]:
        xs = [x for x, _ in self._curve]
        ys = [y for _, y in self._curve]
        return xs, ys

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

    def de_casteljau_threading(self, cnt_threads: int = 4) -> None:
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

    def intersect(self, t1: tuple, t2: tuple) -> bool:
        return t2[0] <= t1[0] <= t2[1] \
               or t2[0] <= t1[1] <= t2[1] \
               or t1[0] <= t2[0] <= t1[1] \
               or t1[0] <= t2[1] <= t1[1]

    """
    method creates minmax box for the corresponding curve

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    def min_max_box(self) -> None:
        xs = [*self._bezier_points[0, :]]
        ys = [*self._bezier_points[1, :]]
        xs.sort()
        ys.sort()
        self._box = [(xs[0], xs[-1]), (ys[0], ys[-1])]

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

    def collision_check(self, other_curve) -> bool:
        if not self.box_collision_check(other_curve):
            return False

        return self.curve_collision_check(other_curve)

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

    def box_collision_check(self, other_curve) -> bool:
        o_box = other_curve.get_box()
        box = self._box
        for t1, t2 in zip(box, o_box):
            if not self.intersect(t1, t2):
                return False

        return True

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

    def curve_collision_check(self, other_curve) -> bool:
        xs, ys = self.get_curve()
        f1 = sg.LineString(np.column_stack((xs, ys)))
        xs, ys = other_curve.get_curve()
        f2 = sg.LineString(np.column_stack((xs, ys)))
        inter = f1.intersection(f2)
        return not inter.geom_type == 'LineString'


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

    """
    method return x, y and z coords of all calculated points

    Parameters
    ----------
    None

    Returns
    -------
    lists:
        one list for x coords, one for y coords and one for z coords
    """

    def get_curve(self) -> Tuple[list, list, list]:
        xs = [x for x, _, _ in self._curve]
        ys = [y for _, y, _ in self._curve]
        zs = [z for _, _, z in self._curve]
        return xs, ys, zs

    """
    method creates minmax box for the corresponding curve

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    def min_max_box(self) -> None:
        super().min_max_box()
        zs = [*self._bezier_points[2, :]]
        zs.sort()
        self._box.append((zs[0], zs[-1]))

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

    def collision_check(self, other_curve) -> bool:
        if not self.box_collision_check(other_curve):
            return False

        return self.curve_collision_check(other_curve)

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

    def curve_collision_check(self, other_curve) -> bool:
        xs, ys, zs = self.get_curve()
        f1 = sg.LineString(np.column_stack((xs, ys, zs)))
        xs, ys, zs = other_curve.get_curve()
        f2 = sg.LineString(np.column_stack((xs, ys, zs)))
        inter = f1.intersection(f2)
        return not inter.geom_type == 'LineString'


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


def init(m: np.ndarray) -> None:
    if m.size == 0:
        return
    b1 = BezierCurve2D(m)
    b1.de_casteljau_threading()
    xs, ys = b1.get_curve()
    plt.plot(xs, ys, 'o')
    plt.show()


if __name__ == "__main__":
    m = csv_read('test.csv')  # reads csv file with bezier points
    init(m)
