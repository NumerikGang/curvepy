import pytest

from curvepy.delaunay import DelaunayTriangulation2D
import curvepy.tests.data.data_voronoi as data


@pytest.mark.parametrize('seed, mean, expected', [*zip(data.SEEDS, data.MEANS, data.REGIONS)])
def test_random_uniform_distribution(seed, mean, expected):
    d = DelaunayTriangulation2D(mean, data.DIAMETER)
    for s in seed:
        d.add_point(s)
    assert d.voronoi() == expected
