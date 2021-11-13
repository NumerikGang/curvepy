import numpy as np
import pytest
import itertools
import curvepy.tests.data.data_bezier_curve as data
from curvepy.bezier_curve import *
from curvepy.utilities import unzip
from curvepy.tests.utility import arrayize


@pytest.mark.parametrize('m,cnt_ts, use_parallel', data.BEZIER_TESTCASES_NORMAL_SIZE)
def test_compare_all_bezier_curves(m, cnt_ts, use_parallel):
    a = BezierCurveSymPy(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    b = BezierCurveDeCaes(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    c = BezierCurveBernstein(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    d = BezierCurveHorner(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    e = BezierCurveMonomial(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    for x, y in itertools.combinations([a, b, c, d, e], 2):
        assert pytest.approx(x.curve, y.curve)


@pytest.mark.parametrize('m,cnt_ts, use_parallel', data.BEZIER_TESTCASES_LARGE_SIZE)
def test_compare_all_large_bezier_curves(m, cnt_ts, use_parallel):
    a = BezierCurveDeCaes(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    b = BezierCurveBernstein(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    c = BezierCurveHorner(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    d = BezierCurveMonomial(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    for x, y in itertools.combinations([a, b, c, d], 2):
        assert pytest.approx(x.curve, y.curve)


@pytest.mark.parametrize('m,expected,approx_rounds', data.BEZIER_CURVE_APPROXIMATION_PRECOMPUTED)
def test_values_for_bezier_curve_approximation(m, expected, approx_rounds):
    c = BezierCurveApproximation.from_round_number(np.array(m), approx_rounds).curve
    assert pytest.approx(list(c[0]), expected[0]), pytest.approx(list(c[1]), expected[1])


@pytest.mark.parametrize('x,y', arrayize([
    ((a, b), (a, b)) for a, b, _, _ in data.FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_all_points_are_equal(x, y):
    assert AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((a, b), (b, c)) for a, b, c, _ in data.FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_at_least_one_equal_point(x, y):
    assert AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((a, b), (c, d)) for a, b, c, d in data.FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_disjunct_intervals(x, y):
    assert not AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((a, d), (b, c)) for a, b, c, d in data.FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_lies_completely_within_another(x, y):
    assert AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((a, c), (b, d)) for a, b, c, d in data.FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_intersects_left_side(x, y):
    assert AbstractBezierCurve.intersect(x, y)


@pytest.mark.parametrize('x,y', arrayize([
    ((b, d), (a, c)) for a, b, c, d in data.FOUR_DISTINCT_SORTED_VALUES
]))
def test_intersect_intersects_right_size(x, y):
    assert AbstractBezierCurve.intersect(x, y)


# TODO: collision_check testen (dies ist curveunabhaengig)

@pytest.mark.parametrize("xs1,ys1,xs2,ys2,m1,m2", data.NOT_EVEN_BOXES_INTERSECT)
def test_not_even_boxes_intersect(xs1, ys1, xs2, ys2, m1, m2):
    # The first 4 parameters are the boxes
    b1 = BezierCurveDeCaes(np.array(m1))
    b2 = BezierCurveDeCaes(np.array(m2))
    assert not b1.box_collision_check(b2)


@pytest.mark.parametrize("xs1,ys1,xs2,ys2,m1,m2,expected", data.CURVE_COLLISION)
def test_curves_collision_checks_manually_verfied(xs1,ys1,xs2,ys2,m1,m2,expected):
    # The first 4 parameters are the boxes
    assert BezierCurveDeCaes(np.array(m1))\
        .curve_collision_check(
        BezierCurveDeCaes(np.array(m2))
    ) == expected


def test_same_curve_intersects_with_itself():
    ...


# TODO: curve_collision_check testen mit verschiedenen BezCurves (hier einfach mit parametrize)


# Computed with https://rosettacode.org/wiki/Forward_difference#Python

@pytest.mark.parametrize('x, res', data.SINGLE_FORWARD_DIFFS)
def test_single_forward_difference(x, res):
    assert np.all(
        np.isclose(np.array(res), BezierCurveDeCaes(np.array([x, x])).single_forward_difference(i=0, r=len(x) - 1))
    )


@pytest.mark.parametrize('pts, t, r, expected', data.DERIVATIVES)
def test_derivative(pts, t, r, expected):
    assert pytest.approx(BezierCurveDeCaes(pts).derivative_bezier_curve(t, r), expected)


@pytest.mark.parametrize('m, n, alpha', data.BARYCENTRIC_COMBINATIONS)
def test_barycentric_combinations(m, n, alpha):
    n = np.array(n)
    m = np.array(m)
    a = BezierCurveDeCaes(m)
    b = BezierCurveDeCaes(n)
    ab = AbstractBezierCurve.barycentric_combination_bezier(a, b, alpha, 1 - alpha)
    curve = [*ab.curve]
    pre_weighted = m * alpha + n * (1 - alpha)
    expected = [*BezierCurveDeCaes(pre_weighted).curve]
    assert np.array_equal(curve, expected)


@pytest.mark.parametrize('interval, xs', zip(data.RANDOM_INTERVALS, data.BEZIER_TESTCASES_NORMAL_SIZE))
def test_check_intervals(interval, xs):
    m, cnt_ts, use_parallel = xs
    a = BezierCurveDeCaes(m, cnt_ts, use_parallel, interval=tuple(interval))
    res_x, res_y = map(np.array, unzip([a(t) for t in np.linspace(*interval, cnt_ts)]))
    a_curve = a.curve
    assert len(a_curve[0]) == len(res_x) and len(a_curve[1]) == len(res_y)
    assert np.allclose(res_x.reshape(a_curve[0].shape), a_curve[0]) \
           and np.allclose(res_y.reshape(a_curve[1].shape), a_curve[1])


@pytest.mark.parametrize('approx_rounds, cnt_bezier_points', itertools.product(range(2, 7), range(5, 206, 20)))
def test_approx_rounds_to_cnt_ts_to_approx_rounds_equals_id(approx_rounds, cnt_bezier_points):
    assert BezierCurveApproximation.cnt_ts_to_approx_rounds(
        BezierCurveApproximation.approx_rounds_to_cnt_ts(approx_rounds, cnt_bezier_points), cnt_bezier_points
    ) == approx_rounds
