import numpy as np
from collections import namedtuple

valid_range = namedtuple("valid_range", "xmin xmax ymin ymax")


class triangle:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.points = sorted([x, y, z])
        self.circumcircle = self.calculate_circumcircle()

    def calculate_circumcircle(self):
        """
        :return:

        See: https://de.wikipedia.org/wiki/Umkreis#Koordinaten
        """
        [x1, y1], [x2, y2], [x3, y3] = self.points
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


class dirchlet_tessellation:
    def __init__(self, valid_range: valid_range):
        self.valid_range = valid_range
        self.triangles = [self.create_supertriangle()]

    def get_all_points(self):
        return set(np.ravel([t.points for t in self.triangles]))

    def add_point(self, pt: np.ndarray):
        if self.is_not_in_range():
            raise AssertionError("point not in predefined range")
        t = self.find_triangle_containing(pt)
        self.add_new_triangles(pt, t)
        for new_t in self.triangles[-3:]:
            self.test_circumcircle_property(new_t)
            ...

    def test_circumcircle_property(self, t):
        ...

    def create_supertriangle(self) -> triangle:
        ...

    def is_not_in_range(self):
        ...

    def add_new_triangles(self, pt, t):
        for a, b in zip(t.points, t.points):
            if a != b:
                self.triangles.append(triangle(a, b, pt))

    def find_triangle_containing(self, pt) -> triangle:
        # for each triangle
        #   if triangle.contains(pt): return triangle
        return None
