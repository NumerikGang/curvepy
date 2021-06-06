import numpy as np
from typing import List, Tuple
import itertools


class Triangle:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        self.points = [np.array(x) for x in sorted([[*A], [*B], [*C]])]
        self.area = self.calc_area(*A, *B, *C)
        self.circumcircle = self.calculate_circumcircle()

    def calculate_circumcircle(self):
        """
        :return:

        See: https://de.wikipedia.org/wiki/Umkreis#Koordinaten
        See: https://de.wikipedia.org/wiki/Umkreis#Radius
        """
        A, B, C = self.points
        [x1, y1], [x2, y2], [x3, y3] = A, B, C
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        xu = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
        yu = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

        lines = [[A, B], [B, C], [A, C]]
        c, a, b = [np.linalg.norm(x - y) for x, y in lines]

        R = (a * b * c) / (4 * self.area)
        return Circle(center=np.array([xu, yu]), radius=R)

    def get_farthest_point_away_and_nearest_line_to(self, pt: np.ndarray):
        points = self.points.copy()
        farthest_p = self.points[np.argmax([np.linalg.norm(x - pt) for x in points])]
        points.remove(farthest_p)
        return farthest_p, points

    def __contains__(self, pt: Tuple[float, float]):
        """
        See: https://www.geeksforgeeks.org/check-whether-a-given-point-lies-inside-a-triangle-or-not/
        """
        [x1, y1], [x2, y2], [x3, y3] = self.points
        x, y = pt
        area1 = self.calc_area(x, y, x2, y2, x3, y3)
        area2 = self.calc_area(x1, y1, x, y, x3, y3)
        area3 = self.calc_area(x1, y1, x2, y2, x, y)
        return self.area == area1 + area2 + area3

    @staticmethod
    def calc_area(x1, y1, x2, y2, x3, y3):
        """
        See: https://www.mathopenref.com/coordtrianglearea.html
        """
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    def __str__(self):
        return str(self.points)

    def __repr__(self):
        return str(self)


class Circle:
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = radius

    def point_in_me(self, pt: np.ndarray) -> bool:
        return np.linalg.norm(pt - self.center) <= self.radius


class Delaunay_triangulation:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.supertriangle = self.create_supertriangle()
        self._triangles: List[Triangle] = [self.supertriangle]
        self.triangle_queue = []

    @property
    def triangles(self):
        # We have to remove everything containing vertices of the supertriangle
        rem_if = lambda t: any(pt in t.points for pt in self.supertriangle.points)
        return [t for t in self._triangles if rem_if(t)]

    def get_all_points(self):
        return set(sum([t.points for t in self._triangles], []))

    def add_point(self, pt: Tuple[float, float]):
        print(f"pt:{pt}")
        if not self.is_in_range(pt):
            raise AssertionError("point not in predefined range")
        t = self.find_triangle_containing(pt)
        self.add_new_triangles(pt, t)
        self._triangles.remove(t)
        # new _triangles
        self.triangle_queue = self._triangles[-3:]
        pts = self.get_all_points()
        while self.triangle_queue:
            t = self.triangle_queue.pop()
            for p in pts:
                if t.check_if_point_is_in_points(p):
                    continue
                if t.circumcircle.point_in_me(p):
                    self.handle_point_in_circumcircle(t, p)
                    break

    def handle_point_in_circumcircle(self, current_t, p_in_circumcircle):
        farthest_pt, nearest_pts = current_t.get_farthest_point_away_and_nearest_line_to(p_in_circumcircle)
        t1 = Triangle(p_in_circumcircle, farthest_pt, nearest_pts[0])
        t2 = Triangle(p_in_circumcircle, farthest_pt, nearest_pts[1])
        self.update_triangle_structures(t1, t2, nearest_pts)

    def update_triangle_structures(self, t1_new, t2_new, removed_line):
        triangles_to_remove = [t for t in self._triangles if
                               removed_line[0] in t.points and removed_line[1] in t.points]
        for x in triangles_to_remove:
            self._triangles.remove(x)
        self._triangles += [t1_new, t2_new]
        self.triangle_queue += [t1_new, t2_new]

    def create_supertriangle(self) -> Triangle:
        # Rectangle CCW
        A = np.array([self.xmin, self.ymax])
        B = np.array([self.xmin, self.ymin])
        C = np.array([self.xmax, self.ymin])
        D = np.array([self.xmax, self.ymax])

        # Those _triangles are chosen "randomly"
        # They would look like
        # lower_left_triangle = [A, B, A_supert]
        # lower_right_triangle = [D, B_supert, C]
        A_supert = np.array([self.xmin - (self.xmax - self.xmin) / 2, self.ymin])
        B_supert = np.array([self.xmax + (self.xmax - self.xmin) / 2, self.ymin])
        C_supert = self.intersect_lines(A_supert, A, B_supert, D)

        return Triangle(A_supert, B_supert, C_supert)

    def intersect_lines(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray):
        """
        Method checking if line through p1, p2 intersects with line through p3, p4


        Parameters
        ----------
        p1: np.ndarray:
            first point of first line

        p2: np.ndarray:
            second point of first line

        p3: np.ndarray:
            first point of second line

        p4: np.ndarray:
            second point of second line

        Returns
        -------
        bool:
            True if all point are less than tol away from line otherwise false
        """
        # First we vertical stack the points in an array
        vertical_stack = np.vstack([p1, p2, p3, p4])
        # Then we transform them to homogeneous coordinates, to perform a little trick
        homogeneous = np.hstack((vertical_stack, np.ones((4, 1))))
        # having our points in this form we can get the lines through the cross product
        line_1, line_2 = np.cross(homogeneous[0], homogeneous[1]), np.cross(homogeneous[2], homogeneous[3])
        # when we calculate the cross product of the lines we get intersect point
        x, y, z = np.cross(line_1, line_2)
        if z == 0:
            return None
        # we divide with z to turn back to 2D space
        return np.array([x / z, y / z])

    def is_in_range(self, pt):
        return self.xmin <= pt[0] <= self.xmax and self.ymin <= pt[1] <= self.ymax

    def add_new_triangles(self, pt, t):
        for a,b in [[0,1],[0,2],[1,2]]:
            self._triangles.append(Triangle(t.points[a], t.points[b], pt))
        # print([*itertools.product(t.points, t.points)])
        # for a, b in itertools.product(t.points, t.points):
        #     if a != b:
        #         self._triangles.append(Triangle(a, b, pt))

    def find_triangle_containing(self, pt) -> Triangle:
        for t in self._triangles:
            if pt in t:
                return t
        raise LookupError(f"No triangle containing ({pt})")


if __name__ == '__main__':
    pts = [np.array(x) for x in ((2, 3), (6, 5), (3, 7), (8, 3), (5, 1), (8, 8), (-3, -2))]
    d = Delaunay_triangulation(-100, 100, -100, 100)
    for p in pts:
        d.add_point(p)

    print(d.triangles)
