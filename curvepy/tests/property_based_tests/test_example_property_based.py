"""The reader is expected to read all unit test examples

Hypothesis explains the concept of property-based testing pretty well.

It works by generating arbitrary data matching your specification and checking that your guarantee still
holds in that case. If it finds an example where it doesn’t, it takes that example and cuts it down to size,
simplifying it until it finds a much smaller example that still causes the problem. It then saves that example
for later, so that once it has found a problem with your code it will not forget it in the future.

For further information when testing, run pytest with `--hypothesis-show-statistics`

More examples may follow.
"""

from hypothesis import given
import hypothesis.strategies as st


@given(st.integers(), st.integers())
def test_ints_are_commutative(x, y):
    """This is an simple example for hypothesis.

    With the @given decorator, you tell the function that you expect integers.
    Therefore it will call the function with a lot of integers and check whether it holds all the time.
    The function itself gets called with pytest as well.

    Parameters
    ----------
    x The first integer, chosen by hypothesis
    y The second integer, chosen by hypothesis
    """
    assert x + y == y + x


@given(st.lists(st.integers()))
def test_reversing_twice_gives_same_list(xs):
    """This is an example for using lists as a strategy.

    This will generate lists of arbitrary length (usually between 0 and 100 elements) whose elements are integers

    Parameters
    ----------
    xs A list of Numbers, chosen by hypothesis
    """
    ys = list(xs)
    ys.reverse()
    ys.reverse()
    assert xs == ys
