import pytest
import itertools
import curvepy.tests.data.data_bezier_curve as data
from curvepy.bezier_curve import *
from curvepy.tests.utility import arrayize

# TODO: Equality (mit Rundung) aller bez_curves (jeweils zwischen 2 damit man am Funktionsnamen erkennen kann
# TODO: welche beiden

# TODO: Equality (ohne Rundung) aller bez_curves auch parallel vs seriell

# TODO: curve auch generell mit ground truth vergleichen (vorgerechnet)  (Approx)


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

def hello_world():
    return

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

# TODO: curve_collision_check testen mit verschiedenen BezCurves (hier einfach mit parametrize)

# TODO: single_forward_difference testen

# Computed with https://rosettacode.org/wiki/Forward_difference#Python

@pytest.mark.parametrize('x, res', data.SINGLE_FORWARD_DIFFS)
def test_single_forward_difference(x, res):
    assert np.all(
        np.isclose(np.array(res), BezierCurveDeCaes(np.array([x, x])).single_forward_difference(i=0, r=len(x) - 1))
    )


@pytest.mark.parametrize('pts, t, r, expected', data.DERIVATIVES)
def test_derivative(pts, t, r, expected):
    assert pytest.approx(BezierCurveDeCaes(pts).derivative_bezier_curve(t,r), expected)

# TODO: barycentric_combination_bezier testen

# TODO: __call__ testen, explizit mit wylden intervallen

# TODO: Kommutativitaet von Skalarmultiplikation testen (ggf via Hypothesis)

# TODO: BezierCurveApprox Sondertests:
#   - Klappt __add__ mit Approx sowie nicht-Approx?
#   - Gehen bei __add__ (und auch generell und so) die cnt_ts kaputt?
#   - (Nicht als Test schreiben) Klappt eigentlich plot? show_funcs mit nicht Approxes?

# TODO: Refactor as
@pytest.mark.parametrize('approx_rounds, cnt_bezier_points', itertools.product(range(2, 7), range(5, 206, 20)))
def test_approx_rounds_to_cnt_ts_to_approx_rounds_equals_id(approx_rounds, cnt_bezier_points):
    assert BezierCurveApproximation.cnt_ts_to_approx_rounds(
        BezierCurveApproximation.approx_rounds_to_cnt_ts(approx_rounds, cnt_bezier_points), cnt_bezier_points
    ) == approx_rounds
