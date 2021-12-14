from typing import List, Set

import pytest

import curvepy.tests.data.data_delaunay as data
from curvepy.delaunay import DelaunayTriangulation2D, Point2D, TupleTriangle


@pytest.mark.parametrize('xs, expected', data.RANDOMLY_UNIFORMLY_DISTRIBUTED)
def test_random_uniform_distribution(xs: List[Point2D], expected: Set[TupleTriangle]):
    d = DelaunayTriangulation2D(radius=10)
    for pt in xs:
        d.add_point(pt)
    assert set(d.triangles) == expected
