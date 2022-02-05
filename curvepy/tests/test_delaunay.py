"""Here are all tests for the Delaunay triangulation algorithm.

See each test for more information.
"""
from typing import List, Set

import pytest

import curvepy.tests.data.data_delaunay as data
from curvepy.delaunay import DelaunayTriangulation2D, Point2D, TupleTriangle


@pytest.mark.parametrize('xs, expected', data.RANDOMLY_UNIFORMLY_DISTRIBUTED)
def test_random_uniform_distribution(xs: List[Point2D], expected: Set[TupleTriangle]):
    """Checks whether the Delaunay triangulation on uniformly distributed random values works correctly.

    Against precomputed correct values.

    Parameters
    ----------
    xs: List[Point2D]
        The points to build the triangulation from.
    expected: Set[TupleTriangle]
        The expected triangulation.
    """
    d = DelaunayTriangulation2D(radius=10)
    for pt in xs:
        d.add_point(pt)
    assert set(d.triangles) == expected
