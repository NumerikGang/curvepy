from functools import partial
from scipy.special import comb


def bernstein(n: int, k: int, t: float, exact=True) -> float:
    return comb(n, k, exact=exact) * t ** k * (1 - t) ** (n - k)


def get_bernstein_func(n: int, k: int, exact: bool = True) -> partial:
    return partial(bernstein, n=n, k=k, exact=exact)