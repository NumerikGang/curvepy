from typing import Iterable, Union, Tuple
import numpy as np


class DeCasteljau:

    def __init__(self, m: np.ndarray, make_copy=True, interval: Tuple[int, int] = (0, 1)):
        self.m = m.copy() if make_copy else m
        self.interval = interval

    def __call__(self, ts: Iterable[float]) -> np.ndarray:  # get value
        """
        Method computing r round of de Casteljau

        Parameters
        ----------
        t: float:
            value for which point is calculated

        Returns
        -------
        np.ndarray:
            array containing calculated points with given t
        """
        m = self.m

        for i, t in enumerate(ts):
            m = self.step(t)

        return m

    def __getitem__(self, t: float = 0.5):
        return DeCasteljau(self.step(t)[:, :-1])

    def step(self, t: float = 0.5) -> np.ndarray:
        """
        Method computing one round of de Casteljau

        Parameters
        ----------
        t: float:
            value for which point is calculated

        Returns
        -------
        np.ndarray:
            array containing calculated points with given t
        """
        m = self.m
        l, r = self.interval
        if m.shape[1] < 2:
            raise Exception("At least two points are needed")

        t2 = (t - l) / (r - l) if (l, r) != (0, 1) else t
        t1 = 1 - t2

        m[:, :-1] = t1 * m[:, :-1] + t2 * m[:, 1:]

        return m if m.shape == (2, 1) else m[:, :-1]

    def apply_many(self, ts: Iterable[float]):
        return [self[t] for t in ts]
