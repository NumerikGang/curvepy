"""Here are all tests for all our custom types and their functionalities.

See each test for more information.
"""
import pytest

import curvepy.tests.data.data_types as data
from curvepy.bezier_curve import *
from curvepy.types import Polygon, PolygonTriangle, bernstein_polynomial_rec


@pytest.mark.parametrize('pts', data.POLYGON_2D)
def test_create_straight_line_functions_for_polygon_2d(pts: List[List[float]]):
    """Check whether the polygon draws the lines correctly.

    A polygon is basically just a set of lines between 2 points.
    Each line is a linear function of all convex barycentric combinations of 2 points.

    Here we check whether the lines are connected correctly!

    Parameters
    ----------
    pts: List[List[float]]
        The (2d) points from which to create the polygon from.
    """
    pts = [np.array([x, y]) for x, y in zip(pts[0], pts[1])]
    p = Polygon(pts)
    for i in range(len(p) - 1):
        assert all(p[i](0) == pts[i])
        assert all(p[i](1) == pts[i + 1])


@pytest.mark.parametrize('pts', data.POLYGON_3D)
def test_create_straight_line_functions_for_polygon_3d(pts: List[List[float]]):
    """Check whether the polygon draws the lines correctly.

    A polygon is basically just a set of lines between 2 points.
    Each line is a linear function of all convex barycentric combinations of 2 points.

    Here we check whether the lines are connected correctly!

    Parameters
    ----------
    pts: List[List[float]]
        The (3d) points from which to create the polygon from.
    """
    pts = [np.array([x, y, z]) for x, y, z in zip(pts[0], pts[1], pts[2])]
    p = Polygon(pts)
    for i in range(len(p) - 1):
        assert all(p[i](0) == pts[i])
        assert all(p[i](1) == pts[i + 1])


@pytest.mark.parametrize('pts, weights, exp', data.BARY_PLANE_POINT_2D)
def test_bary_plane_point_2d(pts: List[List[float]], weights: List[float], exp: List[float]):
    """Checks whether the barycentric combination of n 2-dimensional points work.

    A barycentric combination is just a weighted affine-combination.

    Parameters
    ----------
    pts: List[List[float]]
        A set of points to create the Polygon from.
    weights: List[float]
        The weight of each point in the barycentric combination.
    exp: List[float]
        The new point created by the barycentric combination of the old ones.
    """
    assert pytest.approx(sum(weights), 1)
    pts = [np.array([x, y]) for x, y in zip(pts[0], pts[1])]
    p = PolygonTriangle(pts)
    weights = np.array(weights)
    assert np.all((p.bary_plane_point(weights) == exp))


@pytest.mark.parametrize('pts, weights, exp', data.BARY_PLANE_POINT_3D)
def test_bary_plane_point_3d(pts: List[List[float]], weights: List[float], exp: List[float]):
    """Checks whether the barycentric combination of n 3-dimensional points work.

    A barycentric combination is just a weighted affine-combination.

    Parameters
    ----------
    pts: List[List[float]]
        A set of points to create the Polygon from.
    weights: List[float]
        The weight of each point in the barycentric combination.
    exp: List[float]
        The new point created by the barycentric combination of the old ones.
    """
    assert pytest.approx(sum(weights), 1)
    pts = [np.array([x, y, z]) for x, y, z in zip(pts[0], pts[1], pts[2])]
    p = PolygonTriangle(pts)
    weights = np.array(weights)
    assert np.all((p.bary_plane_point(weights) == exp))


@pytest.mark.parametrize('pts, weights, exp', data.BARY_PLANE_POINT_2D)
def test_get_bary_coords_2d(pts: List[List[float]], weights: List[float], exp: List[float]):
    """Checks the inverse function of the barycentric combination.

    A barycentric combination is just a weighted affine-combination.
    Thus it's inverse gets the weights used for the affine-combination.

    Let f be the barycentric combination function and xs be the weights.
    We check that f^(-1)(f(xs)) == xs (think: bijectivity).

    Parameters
    ----------
    pts: List[List[float]]
        A set of points to create the Polygon from.
    weights: List[float]
        The weight of each point in the barycentric combination.
    exp: List[float]
        The new point created by the barycentric combination of the old ones (not needed here).
    """
    pts = [np.array([x, y]) for x, y in zip(pts[0], pts[1])]
    p = PolygonTriangle(pts)
    weights = np.array(weights)
    assert np.all(np.isclose(p.get_bary_coords(p.bary_plane_point(weights)), weights))


@pytest.mark.parametrize('pts, weights, exp', data.BARY_PLANE_POINT_3D)
def test_get_bary_coords_3d(pts: List[List[float]], weights: List[float], exp: List[float]):
    """Checks the inverse function of the barycentric combination.

    A barycentric combination is just a weighted affine-combination.
    Thus it's inverse gets the weights used for the affine-combination.

    Let f be the barycentric combination function and xs be the weights.
    We check that f^(-1)(f(xs)) == xs (think: bijectivity).

    Parameters
    ----------
    pts: List[List[float]]
        A set of points to create the Polygon from.
    weights: List[float]
        The weight of each point in the barycentric combination.
    exp: List[float]
        The new point created by the barycentric combination of the old ones (not needed here).
    """
    pts = [np.array([x, y, z]) for x, y, z in zip(pts[0], pts[1], pts[2])]
    p = PolygonTriangle(pts)
    weights = np.array(weights)
    assert np.all(np.isclose(p.get_bary_coords(p.bary_plane_point(weights)), weights))


@pytest.mark.parametrize('n, i, t, exp', data.BERNSTEIN_POLYNOMIAL)
def test_bernstein_polynomial(n: int, i: int, t: float, exp: float):
    """Checks whether both the closed and recursive formula for computing bernstein polynomial work.

    We also check against our own precomputed values.


    Parameters
    ----------
    n: int
        The degree of the bernstein polynomial.
    i: int
        Defines which polynomial of degree n we want.
    t: float
        On which point we evaulate the bernstein polynomial (unit-interval).
    exp: float
        The precomputed value corresponding to t of the i-th of n-th degree.
    """
    assert 0 <= t <= 1
    a = bernstein_polynomial_rec(n, i, t)
    b = bernstein_polynomial(n, i, t)
    assert abs(a - b) < 1e-3
    assert abs(exp - b) < 1e-3
    assert abs(exp - a) < 1e-3
