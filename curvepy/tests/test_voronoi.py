import pytest

from curvepy.delaunay import DelaunayTriangulation2D
import curvepy.tests.data.data_voronoi as data

# Jetbrains Formatter doesn't comply with PEP8, but those data lists are
# not well formattable anyways.
# Therefore we ignore the following:
# E121 continuation line under-indented for hanging indent
#
# flake8: noqa: E121



@pytest.mark.parametrize('seed, mean, expected', [*zip(data.SEEDS, data.MEANS, data.REGIONS)])
def test_random_uniform_distribution(seed, mean, expected):
    d = DelaunayTriangulation2D(mean, data.DIAMETER)
    for s in seed:
        d.add_point(s)
    assert d.voronoi() == expected
