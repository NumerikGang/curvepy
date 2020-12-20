import numpy as np
import scipy.special as scs
from src.utilities import csv_read


def horn_bez(m: np.ndarray, t: int = 0.5) -> np.ndarray:
    n = m.shape[1] - 1  # need degree of curve (n points means degree = n-1)
    n_above_i, t_fac = 1, 1
    res = m[:, 0] * (1 - t)
    for i in range(1, n):
        t_fac = t_fac * t
        n_above_i *= (n - i + 1) // i  # needs to be int
        res = (res + t_fac * n_above_i * m[:, i]) * (1 - t)

    res += t_fac * t * m[:, n]

    return res


def bezier_to_power(m: np.ndarray):
    _, n = m.shape
    diff = differences(m)
    return 0


def differences(m: np.ndarray, i: int = 0) -> np.ndarray:
    _, n = m.shape
    diff = [np.sum([scs.binom(r, j)*(-1)**(r - j)*m[:, i + j] for j in range(0, r+1)], axis=0) for r in range(0, n)]
    return np.array(diff).T


test = csv_read("test.csv")
print(differences(test))
