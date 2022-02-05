"""Here are all tests for bezier curves.

See each test for more information.
"""
import itertools
from typing import List
import numpy as np
import pytest

import curvepy.tests.data.data_bezier_curve as data
from curvepy.bezier_curve import *


@pytest.mark.parametrize('m,cnt_ts, use_parallel', data.BEZIER_TESTCASES_NORMAL_SIZE)
def test_compare_all_bezier_curves(m: np.ndarray, cnt_ts: int, use_parallel: bool):
    """Tests whether all bezier curve implementations return (approximately) the same values.

    Parameters
    ----------
    m : np.ndarray
        The array of bezier points.
    cnt_ts : int
        The number of points on which the bezier curve should be evaluated equidistantly.
    use_parallel : bool
        Whether the points should be evaluated sequentially or concurrently.
    """
    a = BezierCurveSymPy(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    b = BezierCurveDeCaes(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    c = BezierCurveBernstein(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    d = BezierCurveHorner(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    e = BezierCurveMonomial(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    for x, y in itertools.combinations([a, b, c, d, e], 2):
        assert pytest.approx(x.curve, y.curve)


@pytest.mark.parametrize('m,cnt_ts, use_parallel', data.BEZIER_TESTCASES_LARGE_SIZE)
def test_compare_all_large_bezier_curves(m: np.ndarray, cnt_ts: int, use_parallel: bool):
    """Tests whether all bezier curve implementation return (approximately) the same values for large test cases.

    Parameters
    ----------
    m : np.ndarray
        The array of bezier points.
    cnt_ts : int
        The number of points on which the bezier curve should be evaluated equidistantly.
    use_parallel : bool
        Whether the points should be evaluated sequentially or concurrently.
    """
    a = BezierCurveDeCaes(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    b = BezierCurveBernstein(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    c = BezierCurveHorner(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    d = BezierCurveMonomial(m, cnt_ts=cnt_ts, use_parallel=use_parallel)
    for x, y in itertools.combinations([a, b, c, d], 2):
        assert pytest.approx(x.curve, y.curve)


@pytest.mark.parametrize('m,expected,approx_rounds', data.BEZIER_CURVE_APPROXIMATION_PRECOMPUTED)
def test_values_for_bezier_curve_approximation(m: List[List[float]], expected: List[List[float]], approx_rounds: int):
    """Tests whether the approximated curve gives approximately the correct result.

    Parameters
    ----------
    m : List[List[float]]
        The array of bezier points
    expected: List[List[float]]
        The precomputed values which should be approximately what the bezier curve looks like.
    approx_rounds: int
        How many rounds of subdivision should be used.
    """
    c = BezierCurveApproximation.from_round_number(np.array(m), approx_rounds).curve
    assert pytest.approx(list(c[0]), expected[0]), pytest.approx(list(c[1]), expected[1])


@pytest.mark.parametrize('box1, box2, intersection', data.PRECOMPUTED_INTERSECTIONS)
def test_check_intersections_of_two_boxes(box1: List[float], box2: List[float], intersection: List[float]):
    """Checks whether the MinMaxBox intersection of two curves works correctly.

    MinMaxBoxes are boxes defined in the form of [x_min, x_max, y_min, y_max, ...].
    Thus, in this method we check the intersection of two axis parallel boxes.

    Parameters
    ----------
    box1: List[float]
        The values to create a MinMaxBox from in the form of [x_min, x_max, y_min, y_max, ...].
    box2: List[float]
        The values to create a MinMaxBox from in the form of [x_min, x_max, y_min, y_max, ...].
    intersection: List[float]
        The expected intersection in the form of [x_min, x_max, y_min, y_max, ...].
    """
    got = (MinMaxBox(box1)) & MinMaxBox(box2)
    expected = np.array(intersection)
    assert all(got == expected)


@pytest.mark.parametrize('box1, box2', data.PRECOMPUTED_NONINTERSECTIONS)
def test_check_non_intersection_of_two_boxes(box1: List[float], box2: List[float]):
    """Checks whether 2 non-intersecting boxes are correctly identified as non-intersecting.

    Parameters
    ----------
    box1: List[float]
        The values to create a MinMaxBox from in the form of [x_min, x_max, y_min, y_max, ...].
    box2: List[float]
        The values to create a MinMaxBox from in the form of [x_min, x_max, y_min, y_max, ...].
    """
    assert MinMaxBox(box1) & MinMaxBox(box2) is None


@pytest.mark.parametrize('box, pts', data.PRECOMPUTED_POINTS_IN_INTERSECTION_BOX)
def test_point_is_in_min_max_box(box: List[float], pts: Tuple[Tuple[Tuple[float, float], bool], ...]):
    """Checks whether the precomputed point, which is within the area defined by the MinMaxBox, is identified correctly.

    Parameters
    ----------
    box : List[float]
        The values to create a MinMaxBox from in the form of [x_min, x_max, y_min, y_max, ...].
    pts: Tuple[Tuple[Tuple[float, float], bool], ...]
        A tuple of test cases. Each test case has a Point (first tuple) and the truth value of whether it is contained.
    """
    for p in pts:
        assert (p[0] in MinMaxBox(box)) == p[1]


@pytest.mark.parametrize("xs1,ys1,xs2,ys2,m1,m2", data.NOT_EVEN_BOXES_INTERSECT)
def test_not_even_boxes_intersect(xs1: List[float], ys1: List[float], xs2: List[float], ys2: List[float],
                                  m1: List[List[float]], m2: List[List[float]]):
    """Checks whether 2 bezier curves which whole MinMaxBoxes don't even intersect get detected as non-intersecting.

    MinMaxBoxes are boxes defined in the form of [x_min, x_max, y_min, y_max, ...].
    Thus, in this method we check the non-intersection of two axis parallel boxes.
    Remember, bezier curves are convex w.r.t. their MinMaxBox.

    Parameters
    ----------
    xs1: List[float]
        The lower left point of the minmax containing all points m1.
    ys1: List[float]
        The lower left point of the minmax containing all points m2.
    xs2: List[float]
        The upper right point of the minmax containing all points m1.
    ys2: List[float]
        The upper right point of the minmax containing all points m2.
    m1: List[List[float]]
        The points for the first bezier curve, enclosed by xs1/xs2.
    m2: List[List[float]]
        The points for the second bezier curve, enclosed by ys1/ys2.
    """
    # The first 4 parameters are the boxes
    b1 = BezierCurveDeCaes(np.array(m1))
    b2 = BezierCurveDeCaes(np.array(m2))
    assert not AbstractBezierCurve.collision_check(b1, b2)


@pytest.mark.parametrize("xs1,ys1,xs2,ys2,m1,m2,expected", data.CURVE_COLLISION)
def test_curves_collision_checks_manually_verfied(xs1: List[float], ys1: List[float], xs2: List[float],
                                                  ys2: List[float], m1: List[List[float]], m2: List[List[float]],
                                                  expected: bool):
    """Checks whether the collision check works correctly on manually verified examples.

        MinMaxBoxes are boxes defined in the form of [x_min, x_max, y_min, y_max, ...].
        Thus, in this method we check the non-intersection of two axis parallel boxes.
        Remember, bezier curves are convex w.r.t. their MinMaxBox.

        Parameters
        ----------
        xs1: List[float]
            The lower left point of the minmax containing all points m1.
        ys1: List[float]
            The lower left point of the minmax containing all points m2.
        xs2: List[float]
            The upper right point of the minmax containing all points m1.
        ys2: List[float]
            The upper right point of the minmax containing all points m2.
        m1: List[List[float]]
            The points for the first bezier curve, enclosed by xs1/xs2.
        m2: List[List[float]]
            The points for the second bezier curve, enclosed by ys1/ys2.
        expected: bool
            Whether they intersect.
        """
    # The first 4 parameters are the boxes
    b1 = BezierCurveDeCaes(np.array(m1))
    b2 = BezierCurveDeCaes(np.array(m2))
    assert AbstractBezierCurve.collision_check(b1, b2, tol=0.0001) == expected


@pytest.mark.parametrize('x, res', data.SINGLE_FORWARD_DIFFS)
def test_single_forward_difference(x: List[float], res: Tuple[int, int]):
    """Checks whether the computation of a single forward difference works approximately correct.

    We compute them with the binomial coefficient based generic formulas for the n-th order forward difference.
    The formula can be found in Farin equation 5.23.

    Those examples were verified against
    https://rosettacode.org/wiki/Forward_difference#Python

    Parameters
    ----------
    x: List[float]
        The values for generating the bezier curve of 2D-points (x[i],x[i]).
    res: Tuple[int, int]
        The expected value for the first forward difference starting at i=0 (i.e. the first point).
    """
    assert np.all(
        np.isclose(np.array(res), BezierCurveDeCaes(np.array([x, x])).single_forward_difference(i=0, r=len(x) - 1))
    )


@pytest.mark.parametrize('pts, t, r, expected', data.DERIVATIVES)
def test_derivative(pts: np.ndarray, t: float, r: int, expected: np.ndarray):
    """Checks whether the r-th bezier curve derivative works approximately for precomputed values.

    Computed via the forward difference operator.

    Parameters
    ----------
    pts: np.ndarray
        2d array of shape (2,n) for creating the BezierCurve.
    t: float
        On which point we want to evaluate the derivative.
    r: int
        Which derivative to evaluate (for example: r=1 => first derivative).
    expected: np.ndarray
        The expected value of the r-th derivative, evaluated at point t.
    """
    assert pytest.approx(BezierCurveDeCaes(pts).derivative_bezier_curve(t, r), expected)


@pytest.mark.parametrize('m, n, alpha', data.BARYCENTRIC_COMBINATIONS)
def test_barycentric_combinations(m: List[List[float]], n: List[List[float]], alpha: float):
    """Checks whether the barycentric combinations of 2 bezier curves are the same as the bezier curve of a
    barycentric combination of 2 bezier point sets.

    Let B be the bezier curve constructor and f be the barycentric combination.
    We basically want to check that B(f(xs)) == f(B(xs)).
    See Farin for proof.

    Barycentric combinations are just affine combinations of n-dimensional points.
    This means that the weights of all used points need to sum up to 1.

    Parameters
    ----------
    m: List[List[float]]
        The points for the first bezier curve.
    n: List[List[float]]
        The points for the second bezier curve.
    alpha: float
        The weight percentage of the first bezier curve.
        0 <= alpha <= 1
        Since we have 2 bezier curves, and it has to be an affine combination we know that the second bezier curve is
        weighted (1 - alpha).
    """
    assert 0 <= alpha <= 1
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
def test_check_intervals(interval: List[int], xs: Tuple[np.ndarray, int, bool]):
    """Checks whether a bezier curve works not defined for the unit interval.

    This is done as follows:
    For the actual computation, it is irrelevant on which interval the curve is defined.
    This is, because the possible value range is defined by the bezier points, not by the accepted range of parameters.
    Basically, we want to check that bezier curves are invariant under affine transformations.

    Thus, we evaluate the whole curve (just depending on the bezier curves thus we completely ignore the interval)
    and check whether it is the same as the curve evaluated on a equidistant affine transformated grid.

    This just means that we affine transform the space, use the "inverse transformation" when calling
    AbstractBezierCurve.__call__ and hope that we get the identity (the non-transformed grid).

    God, what a pain to explain :D

    Parameters
    ----------
    interval: List[int]
        The valid interval to evaluate the bezier curve in (usually the unit interval).
    xs: Tuple[np.ndarray, int, float]
        A 3-tuple of all parameters required to build the bezier curve.
        First parameters is a numpy array of the points which define the bezier curve.
        Second parameter are on how many points to (equidistantly) evaluate the curve.
        Third parameter defines whether the points are evaluated sequentially or concurrently.
    """
    m, cnt_ts, use_parallel = xs
    a = BezierCurveDeCaes(m, cnt_ts, use_parallel, interval=tuple(interval))
    res_x, res_y = map(np.array, zip(*[a(t) for t in np.linspace(*interval, cnt_ts)]))
    a_curve = a.curve
    assert len(a_curve[0]) == len(res_x) and len(a_curve[1]) == len(res_y)
    assert np.allclose(res_x.reshape(a_curve[0].shape), a_curve[0]) \
           and np.allclose(res_y.reshape(a_curve[1].shape), a_curve[1])


@pytest.mark.parametrize('approx_rounds, cnt_bezier_points', itertools.product(range(2, 7), range(5, 206, 20)))
def test_approx_rounds_to_cnt_ts_to_approx_rounds_equals_id(approx_rounds: int, cnt_bezier_points: int):
    """Checks whether the exact and approximate bezier curves evaluate the approximately same number of points.

    Actually, this is just a unidirectional identity (g(f(x)) == x but f(g(x)) != x)
    (To be precise: f(g(x)) == x iff x is an exact power of 2).

    We approximate bezier curves by subdividing bezier curves with the De Casteljau algorithm.

    Parameters
    ----------
    approx_rounds: int
        The number of subdivisions we do.
    cnt_bezier_points: int
        The number of bezier points our (hypothetical) bezier curve has.
    """
    assert BezierCurveApproximation.cnt_ts_to_approx_rounds(
        BezierCurveApproximation.approx_rounds_to_cnt_ts(approx_rounds, cnt_bezier_points), cnt_bezier_points
    ) == approx_rounds


@pytest.mark.parametrize('m, exp', zip(data.INTERSECT_X_AXIS, data.INTERSECT_X_AXIS_EXPECTED))
def test_intersect_with_x_axis(m: Tuple[List[List[float]], bool], exp: List[List[float]]):
    """Checks whether we can correctly detect an intersection with the x-axis.

    Parameters
    ----------
    m: Tuple[List[List[float]], bool]
        A tuple of the bezier points and whether they intersect with the x-axis (unused).
    exp: List[List[float]]
        A precomputed list of intersecting x values and y values (obviously, the y values are just 0...).
    """
    assert BezierCurveApproximation.intersect_with_x_axis(np.array(m[0])) == tuple(exp)
