"""

"""

import pytest

from typing import List, Set
from curvepy.delaunay import DelaunayTriangulation2D, TupleTriangle, Point2D
import curvepy.tests.data.data_delaunay as data


@pytest.mark.parametrize('xs, expected', data.RANDOMLY_UNIFORMLY_DISTRIBUTED)
def test_random_uniform_distribution(xs: List[Point2D], expected: Set[TupleTriangle]):
    d = DelaunayTriangulation2D(radius=10)
    for pt in xs:
        d.add_point(pt)
    assert set(d.triangles) == expected
