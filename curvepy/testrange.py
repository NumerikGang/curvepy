import numpy as np
from typing import List, Tuple
import itertools
import matplotlib.pyplot as plt
import random as rd
from scipy.spatial import Delaunay


class Triangle:
    def __init__(self, A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]):
        self.points = sorted([A, B, C])
        self.area = self.calc_area(*A, *B, *C)
        # print(self)
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
        c, a, b = [np.linalg.norm(np.array(x) - np.array(y)) for x, y in lines]

        R = (a * b * c) / (4 * self.area)
        return Circle(center=np.array([xu, yu]), radius=R)

    def get_farthest_point_away_and_nearest_line_to(self, pt):
        points = self.points.copy()
        farthest_p = self.points[np.argmax([np.linalg.norm(np.array(x) - np.array(pt)) for x in points])]
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
        delta = 0.000000000000550  # das ist neu
        return area1 + area2 + area3 - delta <= self.area <= area1 + area2 + area3 + delta

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

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return False
        return self.points == other.points


class Circle:
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = radius

    def point_in_me(self, pt: np.ndarray) -> bool:
        return np.linalg.norm(np.array(pt) - self.center) <= self.radius

    def __eq__(self, other):
        if not isinstance(other, Circle):
            return False
        return self.center == other.center and self.radius == other.radius


class Delaunay_triangulation:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.supertriangle = self.create_supertriangle()
        self._triangles: List[Triangle] = [*self.supertriangle]
        self.triangle_queue = []

    @property
    def triangles(self):
        # We have to remove everything containing vertices of the supertriangle
        rem_if = lambda t: any(pt in t.points for pt in self.supertriangle[0].points+self.supertriangle[1].points)
        return [t for t in self._triangles if not rem_if(t)]

    def get_all_points(self):
        return set(sum([t.points for t in self._triangles], []))

    def add_point(self, pt: Tuple[float, float]):
        print(f"pt:{pt}")
        if not self.is_in_range(pt):
            raise AssertionError("point not in predefined range")
        t = self.find_triangle_containing(pt)
        print(f"in welchem dreieck ist {pt} = {t}")
        self.add_new_triangles(pt, t)
        self._triangles.remove(t)
        # new _triangles
        self.triangle_queue = self._triangles[-3:]
        pts = self.get_all_points()
        while self.triangle_queue:
            print(f"queue = {self.triangle_queue}")
            t = self.triangle_queue.pop()
            print(f"t = {t}")
            deine_pt_liste = [p for p in pts if (p not in t.points) and (t.circumcircle.point_in_me(np.array(p)))]
            print(f"deine_pt_liste = {deine_pt_liste}")
            if not deine_pt_liste:
                continue
            print(f"es geht was kaputt")
            t_new_1 = self.create_smallest_circumcircle_triangle(pt, deine_pt_liste + t.points)
            t_new_2 = self.get_second_new_triangle(t_new_1, t, pt)
            print(f"t_new_1 = {t_new_1}")
            print(f"t_new_2 = {t_new_2}")
            print(f"bevor: {self._triangles}")
            self.remove_triangles(t, pt, t_new_1)
            print(f"danach: {self._triangles}")
            self._triangles.append(t_new_1)
            self._triangles.append(t_new_2)
            self.triangle_queue.append(t_new_1)
            self.triangle_queue.append(t_new_2)

    def remove_triangles(self, t, pt, t_new_1):
        self._triangles.remove(t)  # WIR ENTFERNEN NUR EIN DREIECK ABER ES GIBT NOCH EIN ZWEITES DREIECK MIT DER KANTE
        point_not_in_org_t = list(set(t_new_1.points).difference(set(t.points)))[0]
        print(f"point_not_in_org_t = {point_not_in_org_t}")
        point_not_in_t_new_1 = list(set(t.points).difference((set(t_new_1.points))))[0]
        last_point = (0, 0)
        # Wir haben 4 punkte und suchen den letzten den wir brauchen für das andere Dreieck um das zu löschen
        for a in t.points + t_new_1.points:
            if a != pt and a != point_not_in_org_t and a != point_not_in_t_new_1:
                last_point = a
                continue
        to_rem = Triangle(point_not_in_org_t, point_not_in_t_new_1, last_point)
        print(f"to_rem = {to_rem}")
        if to_rem in self._triangles:
            self._triangles.remove(to_rem)

    def get_second_new_triangle(self, t_new_1, t, pt):
        point_not_in_org_t = list(set(t_new_1.points).difference(set(t.points)))[0] # kann immer noch leer sein
        point_not_in_t_new_1 = list(set(t.points).difference((set(t_new_1.points))))[0]
        return Triangle(pt, point_not_in_t_new_1, point_not_in_org_t)

    def create_smallest_circumcircle_triangle(self, pt, in_circle_pts):
        xs = [Triangle(pt, a, b) for a in in_circle_pts for b in in_circle_pts if a != b and a != pt and b != pt]
        # xs = sorted(xs, key=lambda t: t.circumcircle.radius)
        return min(xs, key=lambda t: t.circumcircle.radius)

    def flip(self, current_t, p_in_circumcircle):
        # Finde Dreieck, was aus p_in_circumcircle und 2 Punkten aus current_t besteht
        triangle_with_problematic_edge = None
        for t in self._triangles:
            if p_in_circumcircle in t and sum([x in t for x in current_t.points]) == 2:
                triangle_with_problematic_edge = t
                break

        if not triangle_with_problematic_edge:
            raise AssertionError("DEBUG: ich bin doof")

        # Edge herausfinden die blockiert
        edge_to_remove = [*set(triangle_with_problematic_edge.points).intersection(set(current_t.points))]

        # Punkt finden zum Verbinden mit p_in_circumcircle
        point_to_connect_to = set(current_t.points).difference(edge_to_remove).pop()

        return point_to_connect_to, edge_to_remove

    def handle_point_in_circumcircle(self, current_t, p_in_circumcircle):
        point_to_connect_to, edge_to_remove = self.flip(current_t, p_in_circumcircle)
        t1 = Triangle(p_in_circumcircle, point_to_connect_to, edge_to_remove[0])
        t2 = Triangle(p_in_circumcircle, point_to_connect_to, edge_to_remove[1])
        self.update_triangle_structures(t1, t2, edge_to_remove)

    def update_triangle_structures(self, t1_new, t2_new, removed_line):
        triangles_to_remove = [t for t in self._triangles if
                               removed_line[0] in t.points and removed_line[1] in t.points]
        for x in triangles_to_remove:
            self._triangles.remove(x)
        self._triangles += [t1_new, t2_new]
        self.triangle_queue += [t1_new, t2_new]

    def create_supertriangle(self) -> (Triangle, Triangle):
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

        return Triangle(tuple(A), tuple(B), tuple(C)), Triangle(tuple(A), tuple(C), tuple(D))

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

    @staticmethod
    def intersection_is_between(intersection, p1, p2, p3, p4):
        seg1_max_x = max(p1[0], p2[0])
        seg1_min_x = min(p1[0], p2[0])
        seg1_max_y = max(p1[1], p2[1])
        seg1_min_y = min(p1[1], p2[1])
        seg1 = seg1_min_x < intersection[0] < seg1_max_x and seg1_min_y < intersection[1] < seg1_max_y

        seg2_max_x = max(p3[0], p4[0])
        seg2_min_x = min(p3[0], p4[0])
        seg2_max_y = max(p3[1], p4[1])
        seg2_min_y = min(p3[1], p4[1])
        seg2 = seg2_min_x <= intersection[0] <= seg2_max_x and seg2_min_y <= intersection[1] <= seg2_max_y
        return seg1 and seg2

    def is_in_range(self, pt):
        return self.xmin <= pt[0] <= self.xmax and self.ymin <= pt[1] <= self.ymax

    def add_new_triangles(self, pt, t):
        for a, b in [[0, 1], [0, 2], [1, 2]]:
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
    dein_min, dein_max = -5, 5
    n = 20
    pts = np.array([[rd.uniform(dein_min, dein_max), rd.uniform(dein_min, dein_max)] for _ in range(n)])

    tri = Delaunay(pts)
    plt.triplot(pts[:, 0], pts[:, 1], tri.simplices)
    plt.plot(pts[:, 0], pts[:, 1], 'o')
    plt.show()

    d = Delaunay_triangulation(-10, 10, -10, 10)

    for p in pts:
        d.add_point(tuple(p))

    for tri in d.triangles:
        points = np.ravel(tri.points)
        plt.triplot(points[0::2], points[1::2])

    plt.show()

    print(f"anz Dreiecke = {len(d.triangles)}")

    # print(d.triangles)
