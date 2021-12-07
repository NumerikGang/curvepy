import pytest
from curvepy.de_caes import de_caes, parallel_decaes_unblossomed, subdivision
import curvepy.tests.data.data_de_caes as data
import numpy as np


@pytest.mark.parametrize('m, res, t', data.cases_de_caes)
def test_parametrized_for_de_caes(m, res, t):
    tmp = de_caes(np.array(m, dtype=float), t)
    assert res == [pytest.approx(list(tmp[0])), pytest.approx(list(tmp[1]))]


@pytest.mark.parametrize('m, res, ts', data.cases_parallel)
def test_parametrized_for_de_caes_parallel(m, res, ts):
    tmp = list(parallel_decaes_unblossomed(np.array(m, dtype=float), ts))
    tmp = [[pytest.approx(list(t[0]), rel=1e-5), pytest.approx(list(t[1]), rel=1e-5)] for t in tmp]
    assert res == tmp


@pytest.mark.parametrize('m, res', data.cases_sub)
def test_parametrized_for_subdivision(m, res):
    l, r = subdivision(np.array(m, dtype=float), 0.5)
    l, r = [list(l[0]), list(l[1])], [list(r[0][::-1]), list(r[1][::-1])]
    tmp = (l, r)
    assert res == tmp
