"""Here are all tests for the De Casteljau algorithm.

See each test for more information.
"""
import numpy as np
import pytest

import curvepy.tests.data.data_de_caes as data
from curvepy.de_caes import de_caes, parallel_decaes_unblossomed, subdivision
from typing import List, Tuple


@pytest.mark.parametrize('m, res, t', data.cases_de_caes)
def test_parametrized_for_de_caes(m: List[List[int]], res: List[List[float]], t: float):
    """Checks whether the De Casteljau algorithm works for a single value.

    Parameters
    ----------
    m: List[List[int]]
        The bezier points.
    res: List[List[float]]
        The expected value for a complete execution of the De Casteljau Algorithm.
    t: float
        On which point in the unit interval to evaluate to.
    """
    assert 0 <= t <= 1
    tmp = de_caes(np.array(m, dtype=float), t)
    assert res == [pytest.approx(list(tmp[0])), pytest.approx(list(tmp[1]))]


@pytest.mark.parametrize('m, res, ts', data.cases_parallel)
def test_parametrized_for_de_caes_parallel(m: List[List[int]], res: List[List[float]], ts: List[float]):
    """Checks whether the concurrent mass evaluation of a complete De Casteljau algorithm works properly.

    Parameters
    ----------
    m: List[List[int]]
        The points to build the bezier curve with.
    res: List[List[float]]
        The expected points for each evaluation of the De Casteljau algorithm for some t.
    ts: List[float]
        The corresponding input values to the res output values.
    """
    tmp = list(parallel_decaes_unblossomed(np.array(m, dtype=float), ts))
    tmp = [[pytest.approx(list(t[0]), rel=1e-5), pytest.approx(list(t[1]), rel=1e-5)] for t in tmp]
    assert res == tmp


@pytest.mark.parametrize('m, res', data.cases_sub)
def test_parametrized_for_subdivision(m: List[List[float]], res: Tuple[List[List[float]], List[List[float]]]):
    """Checks whether a single step of subdivision works correctly (tested against precomputed values).

    Parameters
    ----------
    m: List[List[float]]
        The bezier points.
    res: Tuple[List[List[float]], List[List[float]]]
        The 2 new bezier point sets for each side of the subdivision.
    """
    l, r = subdivision(np.array(m, dtype=float), 0.5)
    l, r = [list(l[0]), list(l[1])], [list(r[0][::-1]), list(r[1][::-1])]
    tmp = (l, r)
    assert res == tmp
