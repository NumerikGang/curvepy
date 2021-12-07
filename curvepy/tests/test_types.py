import pytest
import curvepy.tests.data.data_types as data
from curvepy.types import Polygon, PolygonTriangle, bernstein_polynomial_rec
from curvepy.bezier_curve import *
from curvepy.tests.utility import arrayize


@pytest.mark.parametrize('pts', data.POLYGON_2D)
def test_create_straight_line_functions_for_Polygon_2D(pts):
    pts = [np.array([x, y]) for x, y in zip(pts[0], pts[1])]
    p = Polygon(pts)
    for i in range(len(p) - 1):
        assert all(p[i](0) == pts[i])
        assert all(p[i](1) == pts[i + 1])


@pytest.mark.parametrize('pts', data.POLYGON_3D)
def test_create_straight_line_functions_for_Polygon_3D(pts):
    pts = [np.array([x, y, z]) for x, y, z in zip(pts[0], pts[1], pts[2])]
    p = Polygon(pts)
    for i in range(len(p) - 1):
        assert all(p[i](0) == pts[i])
        assert all(p[i](1) == pts[i + 1])


@pytest.mark.parametrize('pts, weights, exp', data.BARY_PLANE_POINT_2D)
def test_bary_plane_point_2d(pts, weights, exp):
    pts = [np.array([x, y]) for x, y in zip(pts[0], pts[1])]
    p = PolygonTriangle(pts)
    weights = np.array(weights)
    assert np.all((p.bary_plane_point(weights) == exp))


@pytest.mark.parametrize('pts, weights, exp', data.BARY_PLANE_POINT_3D)
def test_bary_plane_point_3D(pts, weights, exp):
    pts = [np.array([x, y, z]) for x, y, z in zip(pts[0], pts[1], pts[2])]
    p = PolygonTriangle(pts)
    weights = np.array(weights)
    assert np.all((p.bary_plane_point(weights) == exp))


@pytest.mark.parametrize('pts, weights, exp', data.BARY_PLANE_POINT_2D)
def test_get_bary_coords_2d(pts, weights, exp):
    pts = [np.array([x, y]) for x, y in zip(pts[0], pts[1])]
    p = PolygonTriangle(pts)
    weights = np.array(weights)
    assert np.all(np.isclose(p.get_bary_coords(p.bary_plane_point(weights)), weights))


@pytest.mark.parametrize('pts, weights, exp', data.BARY_PLANE_POINT_3D)
def test_get_bary_coords_3d(pts, weights, exp):
    pts = [np.array([x, y, z]) for x, y, z in zip(pts[0], pts[1], pts[2])]
    p = PolygonTriangle(pts)
    weights = np.array(weights)
    assert np.all(np.isclose(p.get_bary_coords(p.bary_plane_point(weights)), weights))


@pytest.mark.parametrize('n, i, t, exp', data.BERNSTEIN_POLYNOMIAL)
def test_bernstein_polynomial(n, i, t, exp):
    a = bernstein_polynomial_rec(n, i, t)
    b = bernstein_polynomial(n, i, t)
    assert abs(a - b) < 1e-3
    assert abs(exp - b) < 1e-3
    assert abs(exp - a) < 1e-3
