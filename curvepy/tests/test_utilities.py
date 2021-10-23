from curvepy.utilities import *
import pytest
import numpy as np

# Those are literally random generated pairs
RANDOM_PAIRS = [(1, 3), (-2, -2), (-4, 1), (-3, -8), (2, 4), (-5, -7), (-2, -2), (-8, -6), (4, 8), (5, 1), (-2, -5),
                (3, 1), (-2, 5), (-4, 7), (5, 4), (5, -9), (-4, 7), (4, -7), (9, 9), (-4, 5), (4, -9), (-7, 1),
                (-9, -10), (-10, 2), (4, -6), (9, -10), (-9, -5), (2, -7), (6, -4), (-8, -1), (7, 2), (-6, 1), (2, 9),
                (8, -8), (-2, 0), (8, -4), (-2, -7), (-6, -8), (2, -6), (0, -7), (-9, 9), (5, -8), (-3, 2), (-3, 4),
                (8, -1), (1, -8), (-5, 0), (3, -10), (7, -1), (-1, 5)]


@pytest.mark.parametrize('a,b', RANDOM_PAIRS)
def test_straight_line_function_equivalent_to_value(a, b):
    return straight_line_point(a, b, 0.5) == create_straight_line_function(a, b)(0.5)


COLLINEAR_POINTS_X_AXIS_2D = [
    ((1, 2), (1, 4), (1, -6)),
    ((0, 0), (0, 0), (0, 0)),
    ((0, 0), (0, -10), (0, 20)),
    ((-25, 10), (-25, 10), (-25, 10)),
    ((-1, -1000000000), (-1, 1000000000), (-1, 1337)),
]

COLLINEAR_POINTS_X_AXIS_3D = [
    ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ((-25, 10, 22), (-25, 10, 22), (-25, 10, 22)),
    ((-1, 25, -3), (-1, 50, -4), (-1, 75, -5)),
    ((1, 230, -500), (1, 203, -500), (1, 20, -500))
]

COLLINEAR_POINTS_Y_AXIS_2D = [((a2, a1), (b2, b1), (c2, c1)) for (a1, a2), (b1, b2), (c1, c2) in
                              COLLINEAR_POINTS_X_AXIS_2D]
COLLINEAR_POINTS_Y_AXIS_3D = [((a2, a1, a3), (b2, b1, b3), (c2, c1, c3)) for (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) in
                              COLLINEAR_POINTS_X_AXIS_3D]
COLLINEAR_POINTS_X_AXIS = [*COLLINEAR_POINTS_X_AXIS_2D, *COLLINEAR_POINTS_X_AXIS_3D]
COLLINEAR_POINTS_Y_AXIS = [*COLLINEAR_POINTS_Y_AXIS_2D, *COLLINEAR_POINTS_Y_AXIS_3D]
COLLINEAR_POINTS_Z_AXIS = [((a2, a3, a1), (b2, b3, b1), (c2, c3, c1)) for (a1, a2, a3), (b1, b2, b3), (c1, c2, c3) in
                           COLLINEAR_POINTS_X_AXIS_3D]


def arrayize(xss):
    return [(np.array(a), np.array(b), np.array(c)) for a, b, c in xss]


NOT_COLLINEAR_POINTS = [
    ((-5, 0), (3, 5), (5, -3)),
    ((2, 1), (3, 5), (4, -2)),
    ((-13, 2), (3, -1), (9, -7)),
    ((1e8, 0), (0, 1), (-1e8, 0))
]


@pytest.mark.parametrize('a,b,c', arrayize(COLLINEAR_POINTS_X_AXIS))
def test_collinear_check_same_x_axis(a, b, c):
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(COLLINEAR_POINTS_Y_AXIS))
def test_collinear_check_same_y_axis(a, b, c):
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(COLLINEAR_POINTS_Z_AXIS))
def test_collinear_check_same_z_axis(a, b, c):
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize([
    ((1, 1), (2, 2), (3, 3)),
    ((-2, -8), (-4, -16), (-6, -24)),
    ((3, 0), (4, 1), (2, -1)),
]))
def test_collinear_not_parallel_to_axis(a, b, c):
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(NOT_COLLINEAR_POINTS))
def test_not_collinear(a, b, c):
    assert not collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(NOT_COLLINEAR_POINTS))
def test_ratio_fails_when_not_collinear(a, b, c):
    try:
        ratio(a, b, c)
        raise Exception("This method should fail as they are not collinear")
    except ValueError as e:
        pass


def test_ratio_is_0_when_a_is_b():
    a = np.array([4, 3])
    b = a
    c = np.array([8, 5])
    assert ratio(a, b, c) == 0


def test_ratio_is_inf_when_b_is_c():
    a = np.array([4, 3])
    b = np.array([8, 5])
    c = b
    assert ratio(a, b, c) == np.NINF


def test_ratio_is_inf_when_all_points_are_the_same():
    a = np.array([4, 3])
    b = a
    c = a
    assert ratio(a, b, c) == np.NINF


@pytest.mark.parametrize('a,b,c,expected', [(np.array(a), np.array(b), np.array(c), d) for a, b, c, d in
                                            [((2, 1), (4, 1), (-6, 1), -0.2),
                                             ((0, 0), (-10, 0), (20, 0), -0.3333333333333333),
                                             ((-1000000000, -1), (1000000000, -1), (1337, -1), -2.000002674003575),
                                             ((25, -1, -3), (50, -1, -4), (75, -1, -5), 1.0),
                                             ((230, 1, -500), (203, 1, -500), (20, 1, -500), 0.14754098360655737),
                                             ((25, -3, -1), (50, -4, -1), (75, -5, -1), 1.0),
                                             ((230, -500, 1), (203, -500, 1), (20, -500, 1), 0.14754098360655737)]])
def test_ratio_sane_values(a, b, c, expected):
    assert ratio(a, b, c) == expected
