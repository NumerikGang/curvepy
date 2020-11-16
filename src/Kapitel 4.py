import numpy as np
import matplotlib.pyplot as plt
import threading as th
import numpy.linalg as npl
import shapely.geometry as sg


lockMe = th.Lock()

class t_holder():

    def __init__(self,n = 1):
        self._tArray = np.linspace(0,1,n)
        self._pointer = 0

    def get_next_t(self):
        if self._pointer == len(self._tArray): return -1
        res = self._tArray[self._pointer]
        self._pointer += 1
        return res

class CasteljauThread(th.Thread):

    def __init__(self, ts_holder, c, f = lambda x: x):
        th.Thread.__init__(self)
        self._ts_holder = ts_holder
        self._coords = c
        self._res = []
        self._func = f

    def get_res(self):
        return self._res

    def deCaes(self,t,n):
        m = self._coords.copy()
        t = self._func(t)
        for r in range(n):
            m[:, :(n - r - 1)] = (1 - t) * m[:, :(n - r - 1)] + t * m[:, 1:(n - r)]
        self._res.append(m[:, 0])

    def run(self):
        _, n = self._coords.shape
        while True:
            lockMe.acquire()
            t = self._ts_holder.get_next_t()
            lockMe.release()
            if t == -1: break
            self.deCaes(t,n)


class Bezier_Curve():

    def __init__(self, m, cnt_ts = 1000):
        self._bezier_points = m
        self._cnt_ts = cnt_ts
        self._curve = []
        self._box = []
        self._edge_points = []

    def get_box(self):
        return self._box

    def get_edge_points(self):
        return self._edge_points

    # Bisher nur für 2D
    # TODO: Auf 3D oder besser upgraden
    def get_curve(self):
        xs = [x for x, _ in self._curve]
        ys = [y for _, y in self._curve]
        return xs,ys

    def deCasteljau_threading(self, cnt_threads=4):
        ts = t_holder(self._cnt_ts)
        threads = []

        for _ in range(cnt_threads): threads.append(CasteljauThread(ts, self._bezier_points))

        for t in threads: t.start()

        for t in threads:
            t.join()
            tmp = t.get_res()
            self._curve = self._curve + tmp

        self.min_max_box()

    def create_edge_points(self):
        xs = [x for x in self._bezier_points[0, :]]
        ys = [y for y in self._bezier_points[1, :]]
        xs.sort(); ys.sort()
        down_left = np.array([xs[0], ys[0]])
        down_right = np.array([xs[-1], ys[0]])
        up_left = np.array([xs[0], ys[-1]])
        up_right = np.array([xs[-1], ys[-1]])
        self._edge_points = [down_left, down_right, up_left, up_right]

    def min_max_box(self):
        self.create_edge_points()
        d_vector_x = self._edge_points[1] - self._edge_points[0]
        d_vector_y = self._edge_points[2] - self._edge_points[0]
        self._box = np.array([self._edge_points[0],d_vector_x,d_vector_y])

    def collision_check(self, other_curve):
        o_edge_points = other_curve.get_edge_points()
        box = self._box
        A = np.array([box[1,:],box[2,:]])
        for p in o_edge_points:
            b = p - box[0,:]
            u,v = npl.solve(A,b)
            if 0 <= u <= 1 or 0 <= v <= 1:
                return self.curve_collision_check(other_curve)
        return False

    def curve_collision_check(self,other_curve):
        xs,ys = self.get_curve()
        f1 = sg.LineString(np.column_stack((xs,ys)))
        xs, ys = other_curve.get_curve()
        f2 = sg.LineString(np.column_stack((xs, ys)))
        inter = f1.intersection(f2)
        print(inter.geom_type)
        if inter.geom_type == 'LineString': return False
        return True


def init():
    xs_1 = np.array([0, 4, 8])
    ys_1 = np.array([0, 5, 0])
    xs_2 = np.array([7, 11, 15])
    ys_2 = np.array([2, 6, 2])
    m1 = np.array([xs_1, ys_1], dtype=float)
    m2 = np.array([xs_2, ys_2], dtype=float)
    b1 = Bezier_Curve(m1)
    b1.deCasteljau_threading()
    b2 = Bezier_Curve(m2)
    b2.deCasteljau_threading()
    print(b1.collision_check(b2))
    xs_2, ys_2 = b2.get_curve()
    xs_1, ys_1 = b1.get_curve()
    fig, ax = plt.subplots()
    ax.plot(xs_1,ys_1, 'o')
    ax.plot(xs_2,ys_2, 'o')
    plt.show()
    print('fertig')

#TODO Input Reader (CSV)

init()









######################################################################################################################


#def plot(coords):
#    xs = [x for x, _ in coords]
#    ys = [y for _, y in coords]
#    plt.plot(xs, ys, 'o')
#    plt.show()

#def deCasteljau_threading(m,cnt_threads = 4, n = 1):
#    ts = t_holder(n)
#    threads = []
#    res = []
#
#    for _ in range(cnt_threads): threads.append(CasteljauThread(ts, m))
#
#    for t in threads: t.start()
#
#    for t in threads:
#        t.join()
#        tmp = t.get_res()
#        res = res + tmp
#    return res




#def deCastellp(m, t = 0.5):
#    _, n = m.shape
#    for r in range(n):
#        m[:,:(n - r - 1)] = (1 - t) * m[:,:(n - r - 1)] + t * m[:,1:(n - r)]
#    return m[:,0]


#def deCastell(xs, ys, n=1):
#    ts = np.linspace(0, 1, n)
#    m = np.array([xs,ys], dtype=float)
#    res = []
#    for t in ts:
#        tmp = m.copy()
#        res.append(deCastellp(tmp,t))
#    return res

#t1 = CasteljauThread(ts,m)
#t2 = CasteljauThread(ts,m)
#t3 = CasteljauThread(ts,m)
#t4 = CasteljauThread(ts,m)
#t1.start()
#t2.start()
#t3.start()
#t4.start()
#t1.join()
#t2.join()
#t3.join()
#t4.join()

#res1 = t1.get_res()
#res2 = t2.get_res()
#res3 = t3.get_res()
#res4 = t4.get_res()
#res = res1 + res2 + res3 + res4

