import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
import functools
import matplotlib.pyplot as plt
import sys


def straight_line_point(a: np.ndarray, b: np.ndarray, t: float = 0.5) -> np.ndarray:
    """
    Method to calculate a single point on a straight line through a and b.

    Parameters
    ----------
    a: np.ndArray
        first point on straight line to calculate new point
    b: np.ndArray
        second point on straight line to calculate new point
    t: float
        for the weight of a and b

    Returns
    -------
    np.ndArray:
        new point on straight line through a and b
    """
    return (1 - t) * a + t * b


def straight_line_function(a: np.ndarray, b: np.ndarray) -> Callable:
    """
    Method to get the function of a straight line through a and b.

    Parameters
    ----------
    a: np.ndArray
        first point on straight line
    b: np.ndArray
        second point on straight line

    Returns
    -------
    Callable:
        function for the straight line through a and b
    """
    return functools.partial(straight_line_point, a, b)


def collinear_check(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """
    Calculates the cross product of (b-a) and (c-a) to see if all 3 points are collinear.

    Parameters
    ----------
    a: np.ndArray
        first point
    b: np.ndArray
        second point
    c: np.ndArray
        third point

    Returns
    -------
    bool:
        True if points are collinear else False
    """
    return np.count_nonzero(np.cross(b - a, c - a)) == 0


def ratio(left: np.ndarray, col_point: np.ndarray, right: np.ndarray) -> float:
    """
    Method to calculate the ratio of the three collinear points from the parameters.
    Throws an exception if the points are not collinear.

    Parameters
    ----------
    left: np.ndArray
        left point that defines the straight line
    col_point: np.ndArray
        collinear point to left and right, could be the most left or most right point or between left and right
    right: np.ndArray
        right point that defines the straight line

    Returns
    -------
    np.ndArray:
        the ratio of the three collinear points from the parameters
    """
    if not collinear_check(left, col_point, right):
        raise Exception("The points are not collinear!")
    # TODO: check if all points have the same dimension
    # TODO: zip it
    for i in range(len(left)):
        if left[i] == right[i]:
            continue
        if right[i] - col_point[i] == 0:
            return np.NINF
        return (col_point[i] - left[i]) / (right[i] - col_point[i])
    return 0


class Polygon:
    """
    Class for creating a 2D or 3D Polygon.

    Attributes
    ----------
    _points: np.ndArray
        array containing copy of points that create the polygon
    _dim: int
        dimension of the polygon
    _piece_funcs: list
        list containing all between each points, _points[i] and _points[i+1]

    """

    # TODO: Dim Check
    def __init__(self, points: np.ndarray, make_copy=True) -> None:
        self._points = points.copy() if make_copy else points
        self._dim = points.shape[1]
        self._piece_funcs = self.create_polygon()

    def create_polygon(self) -> np.ndarray:
        """
        Creates the polygon by creating an array with all straight_line_functions needed.

        Returns
        -------
        np.ndarray:
            the array with all straight_line_functions
        """
        # TODO: zip it
        return np.array([straight_line_function(self._points[i],
                                                self._points[i + 1]) for i in range(len(self._points) - 1)])

    # TODO: Implement __getitem__
    # TODO (for real chads): Let it inherit from collections.abc.Sequence
    # https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes
    # TODO: Afterwards, remove evaluate
    def evaluate(self, index: int, t: float) -> np.ndarray:
        """
        Evaluates the polygon on the index function with the given t.

        Parameters
        ----------
        index: int
            Which piece of the function to use for given t.
        t: float
            For the weights (a-t) and t to calculate the point on the polygon piece.

        Returns
        -------
        np.ndArray:
            evaluated point
        """
        if int(index) > len(self._piece_funcs) or int(index) < 0:
            raise Exception("Not defined!")
        if int(index) == len(self._piece_funcs):
            return self._piece_funcs[len(self._piece_funcs) - 1](1)
        return self._piece_funcs[int(index)](t)

    # TODO: Use real typing (typing.List[Type_Of])
    def blossom(self, ts: list) -> np.ndarray:
        """
        Recursive calculation of a blossom with parameters ts and the polygon.

        Parameters
        ----------
        ts: list
            b[t_1, t_2, ..., t_n]

        Returns
        -------
        np.ndArray:
            Calculated value for the blossom.
        """
        if len(ts) > len(self._piece_funcs):
            raise Exception("The polygon is not long enough for all the ts!")
        if len(ts) == 1:
            return self.evaluate(0, ts[0])
        return Polygon(np.array([self._piece_funcs[i](ts[0]) for i in range(len(ts))])).blossom(ts[1:])


# TODO: "merge" with other triangle class
class Triangle(Polygon):

    def __init__(self, points: np.ndarray, make_copy=True) -> None:
        points_copy = points.copy() if make_copy else points
        points_copy = np.append(points_copy, [points_copy[0]], axis=0)
        super().__init__(points_copy, make_copy=False)

    def bary_plane_point(self, bary_coords: np.ndarray) -> np.ndarray:
        """
        Given the barycentric coordinates and three points this method will calculate a new point as a barycentric
        combination of the Triangle points.

        Parameters
        ----------
        bary_coords: np.ndarray
            The barycentric coordinates corresponding to a, b, c. Have to sum up to 1.

        Returns
        -------
        np.ndarray:
            Barycentric combination of a, b, c with given coordinates.
        """
        if abs(1 - np.sum(bary_coords)) < sys.float_info.epsilon:
            raise Exception("The barycentric coordinates don't sum up to 1!")
        return np.sum(bary_coords.reshape((3, 1)) * self._points, axis=0)

    @staticmethod
    def area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculates the "calc_area" of a Triangle defined by the parameters. All three points have to be on a plane
        parallel to an axis-plane!

        Parameters
        ----------
        a: np.ndarray
            First point of the Triangle.
        b: np.ndarray
            Second point of the Triangle.
        c: np.ndarray
            Third point of the Triangle.

        Returns
        -------
        float:
            "Area" of the Triangle.
        """
        return np.linalg.det(np.array([a, b, c])) / 2

    # TODO: Write shorter
    def squash_parallel_to_axis_plane(self, p: np.ndarray):
        """
        This method projects p and the points of the Triangle on a plane, for example the y-plane with distance 1 for
        all points of the Triangle to the plane, so that cramer's rule can easily be applied to them
        in order to calculate the calc_area of the Triangle corresponding to every 3 out of the 4 points.
        But this method does not overwrite the self._points.

        Parameters
        ----------
        p: np.ndarray
            Additional point that should be on the same plane as the Triangle.

        Returns
        -------
        np.ndarray:
            Copy of p and the Triangle points now mapped on to a plane.
        """
        p_copy, a, b, c = [x.copy() for x in [p, *self._points]]
        for i in range(len(self._points)):
            if self._points[0][i] != self._points[1][i] != self._points[2][i] \
                    and self._points[0][i - 1] != self._points[1][i - 1] != self._points[2][i - 1]:
                p_copy[i - 2], a[i - 2], b[i - 2], c[i - 2] = 1, 1, 1, 1
                break
        return p_copy, a, b, c

    # TODO: Remove exception (checked at constructor)
    # TODO: Type hinting
    # TODO: Optional make_copy parameter
    def check_points_for_area_calc(self, p):
        """
        This method checks if the point p and the points of the Triangle have the right dimension and will make them so
        that cramer's rule can be applied to them.

        Parameters
        ----------
        p: np.ndarray
            Additional point that has to be on the same plane as the Triangle.

        Returns
        -------
        np.ndarrays:
            The Triangle points and p so that cramer's rule can be used.
        """
        if self._dim == 3:
            return self.squash_parallel_to_axis_plane(p)
        return (np.append(x.copy(), [1]) for x in [p, *self._points])

    # TODO: If 3D: Check if the 3D-Point lies on the 2D-Hyperplane defined by the bary coordinates
    # TODO: If not, throw an exception
    def get_bary_coords(self, p: np.ndarray) -> np.ndarray:
        """
        Calculates the barycentric coordinates of p with respect to the points defining the Triangle.

        Parameters
        ----------
        p: np.ndarray
            Point of which we want the barycentric coordinates.

        Returns
        -------
        np.ndarray:
            Barycentric coordinates of p with respect to a, b, c.
        """
        p_copy, a, b, c = self.check_points_for_area_calc(p)

        abc_area = self.area(a, b, c)
        if abc_area == 0:
            raise Exception("The calc_area of the Triangle defined by a, b, c has to be greater than 0!")

        return np.array([self.area(p_copy, b, c) / abc_area, self.area(a, p_copy, c) / abc_area,
                         self.area(a, b, p_copy) / abc_area])


# TODO: Replace with "real" pytests
def ratio_test() -> None:
    left = np.array([0, 0, 0])
    right = np.array([1, 1, 1])
    col_point = np.array([1.2, 1.3, 1.4])
    test = ratio(left, col_point, right)
    print(test)


def straight_line_point_test() -> None:
    t = 0
    fig = plt.figure()
    ax = Axes3D(fig)
    while t <= 1:
        test = straight_line_point(np.array([0, 0, 0]), np.array([1, 1, 1]), t)
        ax.scatter(test[0], test[1], test[2])
        t += 0.1
    plt.show()


def blossom_testing() -> None:
    b_test_points = np.array([[0, 0], [1, 1], [2, 1], [3, 0]])
    b_test_poly = Polygon(b_test_points)
    print(b_test_poly.blossom([0.3, 0.6, 0.5]))


def init() -> None:
    ...
    # coords = np.array([2 / 3, 1 / 6, 1 / 6])
    # a, b, c = np.array([2, 1]), np.array([4, 3]), np.array([5, 1])

    # straight_line_point_test()
    # ratio_test()
    # test_points = np.array([[0, 0, 0], [1, 1, 1], [3, 4, 4], [5, -2, -2]])

    # test_points2d = np.array([[0, 0], [1, 1], [3, 4], [5, -2]])
    # print(len(test_points2d))

    # test_PG = Polygon(test_points)
    # print(test_PG.evaluate(1.5))

    # Blossom testing
    # blossom_testing()

    # Triangle test
    # t = Triangle(np.array([a, b, c]))

    # barycentric coords test
    # print(t.bary_plane_point(coords))
    # print(np.linalg.det(np.array([a, b, c])))
    # print(t.get_bary_coords(t.bary_plane_point(coords)))


if __name__ == "__main__":
    init()

#####################################################################################################
