"""This module contains all pytest-tests.

Curvepy has a pretty good test coverage. Since this is a pretty mathematical package, we do not have great ways to do
traditional unit tests where one just checks whether the current state is fine.

Instead, we have a lot of proven correct precomputed values which we compare against. For readability’s sake, we
have bundled them into the data submodule.

Each test-file corresponds to one python file with the same name.
"""