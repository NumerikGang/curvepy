"""Here are all tests for all our utility functions.

See each test for more information.
"""
import pytest

import curvepy.tests.data.data_utilities as data
from curvepy.tests.utility import arrayize
from curvepy.utilities import *

from typing import TypeVar, List, Tuple
T = TypeVar('T')


@pytest.mark.parametrize('a,b', data.RANDOM_PAIRS)
def test_straight_line_function_equivalent_to_value(a: int, b: int):
    """Checks whether the creation and evaluation of the straight line function is associative.

    Parameters
    ----------
    a: int
        First point defining the line
    b: int
        Second point defining the line
    """
    a, b = np.array(a), np.array(b)
    assert straight_line_point(a, b, 0.5) == create_straight_line_function(a, b)(0.5)


@pytest.mark.parametrize('a,b,c', arrayize(data.COLLINEAR_POINTS_X_AXIS))
def test_collinear_check_same_x_axis(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Checks whether 3 points with the same x value are detected as collinear.

    Parameters
    ----------
    a: np.ndarray
        First point.
    b: np.ndarray
        Second point.
    c: np.ndarray
        Third point.
    """
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.COLLINEAR_POINTS_Y_AXIS))
def test_collinear_check_same_y_axis(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Checks whether 3 points with the same y value are detected as collinear.

    Parameters
    ----------
    a: np.ndarray
        First point.
    b: np.ndarray
        Second point.
    c: np.ndarray
        Third point.
    """
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.COLLINEAR_POINTS_Z_AXIS))
def test_collinear_check_same_z_axis(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Checks whether 3 points with the same z value are detected as collinear.

    Parameters
    ----------
    a: np.ndarray
        First point.
    b: np.ndarray
        Second point.
    c: np.ndarray
        Third point.
    """
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.COLLINEAR_POINTS_NOT_PARALLEL_TO_AXIS))
def test_collinear_not_parallel_to_axis(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Checks whether 3 collinear points are correctly detected as collinear. (not parallel to any axis)

    Parameters
    ----------
    a: np.ndarray
        First point.
    b: np.ndarray
        Second point.
    c: np.ndarray
        Third point.
    """
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.NOT_COLLINEAR_POINTS))
def test_not_collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Checks whether 3 non-collinear points are correctly detected as non-collinear.

    Parameters
    ----------
    a: np.ndarray
        First point.
    b: np.ndarray
        Second point.
    c: np.ndarray
        Third point.
    """
    assert not collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.NOT_COLLINEAR_POINTS))
def test_ratio_fails_when_not_collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Checks whether ratio fails for non-collinear points.

    The ratio defines how much the third point is defined by the first or second point.
    Since the span of the barycentric combinations of 2 points is a line non-collinear points do not make sense.

    Parameters
    ----------
    a: np.ndarray
        First point.
    b: np.ndarray
        Second point.
    c: np.ndarray
        Third point.
    """
    try:
        ratio(a, b, c)
        raise Exception("This method should fail as they are not collinear")
    except ValueError:
        pass


def test_ratio_is_0_when_a_is_b():
    """Checks whether the ratio is 0 when ratio(a,a,c)."""
    a = np.array([4, 3])
    b = a
    c = np.array([8, 5])
    assert ratio(a, b, c) == 0


def test_ratio_is_nan_when_b_is_c():
    """Checks whether the ratio is nan when ratio(a,b,b). (dividing by zero)"""
    a = np.array([4, 3])
    b = np.array([8, 5])
    c = b
    assert ratio(a, b, c) is np.NaN


def test_ratio_is_nan_when_all_points_are_the_same():
    """Checks whether the ratio is nan when ratio(a,a,a). (dividing by zero)"""
    a = np.array([4, 3])
    b = a
    c = a
    assert ratio(a, b, c) is np.NaN


@pytest.mark.parametrize('a,b,c,r', [(np.array(a), np.array(b), np.array(c), r) for a, b, c, r in data.GOOD_VALUES])
def test_ratio_good_values(a: np.ndarray, b: np.ndarray, c: np.ndarray, r: float):
    """Checks whether the ratio function works intended by comparing against manually verified values.

    Parameters
    ----------
    a: np.ndarray
        First point.
    b: np.ndarray
        Second point.
    c: np.ndarray
        Third point.
    r: float
        Expected value.
    """
    assert pytest.approx(ratio(a, b, c), r)


@pytest.mark.parametrize('xs,expected', data.FLATTEN_LIST_OF_LISTS_TESTS)
def test_flatten_list_of_lists(xs: List[List[T]], expected: List[T]):
    """Checks whether lists are properly flattened one dimension by (ab-)using the list addition monoid.

    Parameters
    ----------
    xs: List[List[T]]
        Some nested list
    expected: List[T]
        One dimension less nested list.
    """
    assert flatten_list_of_lists(xs) == expected


@pytest.mark.parametrize('xs', data.PROD)
def test_prod(xs: List[float]):
    """Checks whether our variadic argument product works properly.

    We compare it against the trivial implementation.

    Parameters
    ----------
    xs: List[float]
        Values to multiply.
    """
    exp = 1
    for x in xs:
        exp *= x
    assert exp == prod(xs)


@pytest.mark.parametrize("m, t, expected", data.HORNER)
def test_horner(m: List[List[float]], t: float, expected: Tuple[float, float]):
    """Checks whether the horner-schema-based full De Casteljau Algorithm works properly against precomputed values.

    Parameters
    ----------
    m: List[List[float]]
        Bezier Points
    t: float
        Where to evaluate it.
    expected: Tuple[float, float]
        The expected result of the horner-based De Casteljau Algorithm
    """
    assert horner(np.array(m), t) == expected
