import pytest

import curvepy.tests.data.data_voronoi as data
from curvepy.delaunay import DelaunayTriangulation2D
from curvepy.types import TriangleNode

from typing import List, Tuple, Dict, Deque


@pytest.mark.parametrize('seed, mean, expected', [*zip(data.SEEDS, data.MEANS, data.REGIONS)])
def test_random_uniform_distribution(seed: List[Tuple[float, float]], mean: Tuple[float, float],
                                     expected: Dict[Tuple[float, float], Deque[TriangleNode]]):
    """Tests the voronoi region generation on uniformly distributed random values.

    Parameters
    ----------
    seed: List[Tuple[float, float]]
        The uniformly distributed random values.
    mean: Tuple[float, float]
        The mean of all random values (needed for centering the delaunay triangulation)
    expected: Dict[Tuple[float, float], Deque[TriangleNode]]
        The expected voronoi regions of each point.
    """
    d = DelaunayTriangulation2D(mean, data.DIAMETER)
    for s in seed:
        d.add_point(s)
    assert d.voronoi() == expected
