"""This is an example on how tests should be written.

All test files need to have the test_ prefix (Or the _test.py suffix, but we don't do that here).

Tests are executed by typing `python -m pytest <OPTIONAL_LIST_OF_FILES>`, optionally with `-v` for more verbose output.
Another useful parameter is `-s`, which will allow prints to come through. `-h` for further help.
pytest just needs to be imported if it is actually referenced.
"""

import pytest


def test_simple():
    """This is a simple test.

    All functions which are tests need the test_ prefix in order to be run.
    Tests fail if an exception is raised.

    If we just wanted to run this test, we would do it with:
    `python -m pytest <FILEPATH>::test_simple`
    """
    xs = [1, 2, 3]
    ys = sorted([2, 3, 1])
    assert xs == ys


@pytest.mark.skip("This will always fail by purpose.")
def test_simple_fail():
    """This is a simple test which is supposed to fail

    If we don't want tests to be run, we can just use the pytest.mark.skip decorator like above.
    """
    assert 1 + 1 == 3


try:
    import some_library_that_may_not_exist
except ImportError:
    some_library_that_may_not_exist = None


@pytest.mark.skipif(some_library_that_may_not_exist is None, reason="This library may be optional")
def test_simple_fail_conditionally():
    """This is another example of an failing test.

    We can also exclude tests conditionally with the pytest.mark.skipif decorator.
    This is for example useful if someone decides to install it headless and we have some QT checks.
    """
    assert 1 + 1 + 1 == 5


def mult(a, b):
    """This is a simple multiplication function.

    This function won't be executed by pytest because it is missing the test_ prefix.
    This is helpful because it enables us to write test specific functions to minimize code duplication.

    Parameters
    ----------
    a The first Parameter
    b The second Parameter

    Returns
    -------
    The multiplication of a * b
    """
    return a * b


@pytest.mark.parametrize('x, y, res', [(1, 1, 1), (2, 3, 6), (10000, 0, 0)])
def test_parametrized_for_mult(x, y, res):
    """This is an example for parametrized tests, here used on the `mult` function

    Here, the pytest.mark.parametrize decorator is used.
    The first parameter is a list of symbolic variable names, as used in the function.
    The last parameter is a list of n-1 tuples, where the i-th index is a value for the i-th symbolic variable.

    This means, the function gets called as
    test_parametrized_for_mult(1,1,1) (it is expected that mult(1,1) == 1)
    test_parametrized_for_mult(2,3,6) (it is expected that mult(2,3) == 6)
    test_parametrized_for_mult(1,1,1) (it is expected that mult(10000,0) == 0)

    Parameters
    ----------
    x The first parameter of the multiplication
    y The second parameter of the multiplication
    res The expected result
    """
    assert res == mult(x, y)
