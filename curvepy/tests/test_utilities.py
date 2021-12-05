from curvepy.tests.utility import arrayize
from curvepy.utilities import *
import curvepy.tests.data.data_utilities as data
import pytest
import numpy as np


@pytest.mark.parametrize('a,b', data.RANDOM_PAIRS)
def test_straight_line_function_equivalent_to_value(a, b):
    assert straight_line_point(a, b, 0.5) == create_straight_line_function(a, b)(0.5)


@pytest.mark.parametrize('a,b,c', arrayize(data.COLLINEAR_POINTS_X_AXIS))
def test_collinear_check_same_x_axis(a, b, c):
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.COLLINEAR_POINTS_Y_AXIS))
def test_collinear_check_same_y_axis(a, b, c):
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.COLLINEAR_POINTS_Z_AXIS))
def test_collinear_check_same_z_axis(a, b, c):
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.COLLINEAR_POINTS_NOT_PARALLEL_TO_AXIS))
def test_collinear_not_parallel_to_axis(a, b, c):
    assert collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.NOT_COLLINEAR_POINTS))
def test_not_collinear(a, b, c):
    assert not collinear_check(a, b, c)


@pytest.mark.parametrize('a,b,c', arrayize(data.NOT_COLLINEAR_POINTS))
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


def test_ratio_is_nan_when_b_is_c():
    a = np.array([4, 3])
    b = np.array([8, 5])
    c = b
    assert ratio(a, b, c) is np.NaN


def test_ratio_is_nan_when_all_points_are_the_same():
    a = np.array([4, 3])
    b = a
    c = a
    assert ratio(a, b, c) is np.NaN


@pytest.mark.parametrize('a,b,c,r', [(np.array(a), np.array(b), np.array(c), r) for a, b, c, r in data.GOOD_VALUES])
def test_ratio_good_values(a, b, c, r):
    assert pytest.approx(ratio(a, b, c), r)


# TODO distance_to_line tests

# TODO check_flat tests

# TODO TEST intersect_with_x_axis lines

# TODO flatten_list_of_lists
FLATTEN_LIST_OF_LISTS_TESTS = [
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ],
    [
        [[]],
        []
    ],
    [
        [],
        []
    ],
    [
        [[[1], 2, 3], [4, [5, 6]]],
        [[1], 2, 3, 4, [5, 6]]
    ]
]


@pytest.mark.parametrize('input,expected', FLATTEN_LIST_OF_LISTS_TESTS)
def test_flatten_list_of_lists(input, expected):
    assert flatten_list_of_lists(input) == expected

# TODO test intersect_with_x_axis

# TODO test prod
