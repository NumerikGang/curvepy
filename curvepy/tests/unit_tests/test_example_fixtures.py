"""This is an advanced example how to write code with fixtures.

Fixtures are alternatives to traditional setup/teardown methods, known from classical software testing.
Although due to pytests unittest compatibility they are supported as well, they are not the pythonic way to test.
The idea of fixture is to utilize pythons functional generator concept.
I'll elaborate below:
"""

import pytest


@pytest.fixture
def simple_fixture():
    """This is an example for the simplest fixture

    This way, every time "simple_fixture" is used as an test parameter this function will be called.
    To emphasize: This function will be called __every time__ a test_function with this fixture gets called.
    This would be the pytest equivalent of using a setup function.
    """
    # Doing heavy work...
    return 'heh'


def test_use_simple_fixture(simple_fixture):
    """This is an example usage for the simple_fixture.

    Because this function has simple_fixture as its parameter, it should roughly be called like
    ```
    def calL_this_test():
        tmp = simple_fixture()
        try:
            test_use_simple_fixture(simple_fixture)
            return True
        except:
            return False
    ```

    Parameters
    ----------
    simple_fixture The simple_fixture fixture defined above.
    """
    assert simple_fixture + 'e' == 'hehe'


@pytest.fixture
def teardown_fixture():
    """This is an example for a fixture, which can clean up after itself

    This works as follows: When the function is yielding the value, the test gets called.
    Afterwards this function will execute the rest.
    The best example for its usage would be some kind of external resource, like an database which needs to be closed
    or a temporary text file/pipe which needs to be deleted.
    This would be the pytest equivalent of using a setup as well as teardown function.

    Again, as we are using the simple pytest.fixture this function will get called for __every value__

    Use pytest with the `-s` parameter to see what the function is doing
    """
    print("teardown_fixture: Before yielding the value")
    yield 5
    print("teardown_fixture: After returning from the test")


def test_use_teardown_fixture(teardown_fixture):
    """This is an example function for the teardown_fixture.

    Use pytest with `-s` in order to see the effect.

    Parameters
    ----------
    teardown_fixture the fixture named above
    """
    assert teardown_fixture == 2 + 3


@pytest.fixture(scope="session")
def fixture_which_will_be_only_called_once():
    """This is an example for a fixture with specific scope.

    Fixtures can be defined with different scopes. From the Docs:

    Fixtures are created when first requested by a test, and are destroyed based on their scope:
        - `function`: the default scope, the fixture is destroyed at the end of the test.
        - `class`: the fixture is destroyed during teardown of the last test in the class.
        - `module`: the fixture is destroyed during teardown of the last test in the module.
        - `package`: the fixture is destroyed during teardown of the last test in the package.
        - `session`: the fixture is destroyed at the end of the test session.

    Therefore, this fixture with the `session` scope will only be called once per test run.

    Again, in order to see this you have to use pytest with the `-s` parameter.

    References
    ----------
    https://docs.pytest.org/en/stable/fixture.html
    """
    print('fixture_which_will_be_only_called_once: YOPO (you only print once)')
    return None


@pytest.mark.parametrize('x', [*range(10)])
def test_for_scope_fixture(x, fixture_which_will_be_only_called_once):
    """This is an example test for a fixture with a session scope.

    Parameters
    ----------
    x The parameterized value
    fixture_which_will_be_only_called_once the fixture with the same name
    """
    assert x is not fixture_which_will_be_only_called_once
