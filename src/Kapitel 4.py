import numpy as np
import matplotlib.pyplot as plt
import threading as th

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



#Bisher nur für 2D
#TODO: Auf 3D oder besser upgraden
def plot(coords):
    xs = [x for x,_ in coords]
    ys = [y for _,y in coords]
    plt.plot(xs,ys,'o')
    plt.show()

def deCasteljau_threading(m,cnt = 4, n = 1):
    ts = t_holder(n)
    threads = []
    res = []

    for _ in range(cnt): threads.append(CasteljauThread(ts, m))

    for t in threads: t.start()

    for t in threads:
        t.join()
        tmp = t.get_res()
        res = res + tmp
    return res


def init(xs,ys):
    m = np.array([xs, ys], dtype=float)
    res = deCasteljau_threading(m, 4, 100)
    print('fertig')
    #print(res)
    plot(res)

#TODO Input Reader (CSV)
xs = np.array([0,4,8,12,16,20])
ys = np.array([0,5,3,0,4,2])
init(xs,ys)









######################################################################################################################

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

