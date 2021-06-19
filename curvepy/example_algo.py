import pprint
import random
import numpy as np
from typing import List, Set, Dict, Tuple, Iterable


class h_tuple(np.ndarray):
    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        return len(self) == len(other) and all(x == y for x, y in zip(self, other))

    @staticmethod
    def create(xs: Iterable):
        return np.array(xs, dtype=np.float32).view(h_tuple)


class dirichlet_tessellation:

    def __init__(self):
        self.tiles: Dict[h_tuple, Set[h_tuple]] = {}
        self.valid_triangulation: Set[Tuple[h_tuple, h_tuple]] = set()

    def append_point(self, p: np.ndarray):
        p = h_tuple.create(p)
        print(f"p = {p}")
        if not self.tiles:
            self.tiles[p] = set()
            return

        nearest_p = min(self.tiles.keys(), key=lambda t: np.linalg.norm(t - p))
        print(f"nearest_p = {nearest_p}")
        self.tiles[p] = {nearest_p}
        self.tiles[nearest_p].add(p)

        # ERSTELLE TRIANGULATION ZWISCHEN p UND nearest_p
        self.valid_triangulation.add((p, nearest_p))

        garbage_heap = []

        # To make a more educated guess we solve any collisions __after__ adding the trivial connections
        while True:
            collisions_to_check, garbage_heap = self.calculate_collisions(p, nearest_p, garbage_heap)

            if not collisions_to_check:
                return

            for p, neighbour, collision_edge_p1, collision_edge_p2 in collisions_to_check:
                new_triangles, old_triangles = self.create_triangles(p, neighbour, collision_edge_p1, collision_edge_p2)

                new_is_more_equilateral = self.calculate_triangle_rating(old_triangles) > self.calculate_triangle_rating(new_triangles)

                self.update(new_is_more_equilateral, garbage_heap, p, neighbour, collision_edge_p1, collision_edge_p2)

    def update(self, new_is_more_equilateral, garbage_heap, p1, p2, p3, p4):
        if new_is_more_equilateral:
            self.replace_valid_triangulation((p3, p4), (p1, p2))
            self.update_neighbour(p1, p2, p3, p4)
            garbage_heap.append({p3, p4})
        else:
            garbage_heap.append({p1, p2})

    def calculate_triangle_rating(self, triangles):
        rating = 0
        for t in triangles:
            rating += sum(abs(60 - angle) for angle in self._angles_in_triangle(t))

        return rating

    def create_triangles(self, p1, p2, p3, p4):
        return ((p1, p2, p3), (p1, p2, p4)), ((p1, p2, p4), (p2, p3, p4))

            # Kantenpaare für rate_tri
            # for p, neighbour, collision_edge_p1, collision_edge_p2 in collisions_to_check:
            #     new_triangles = (
            #         # first triangle
            #         ((p, neighbour), (p, collision_edge_p1)),
            #         # ((p, neighbour), (neighbour, collision_edge_p1)), <- HEUTE REINKOPIERT
            #         ((p, collision_edge_p1), (neighbour, collision_edge_p1)),  # maybe jetzt die richtige kante?
            #         ((p, neighbour), (neighbour, collision_edge_p1)),
            #         # second triangle
            #         ((p, neighbour), (p, collision_edge_p2)),
            #         ((p, collision_edge_p2), (neighbour, collision_edge_p2)),
            #         ((p, neighbour), (neighbour, collision_edge_p2))
            #     )
            #
            #     old_triangles = (
            #         # first triangle
            #         ((collision_edge_p1, collision_edge_p2), (collision_edge_p1, p)),
            #         ((collision_edge_p1, p), (collision_edge_p2, p)),
            #         ((collision_edge_p1, collision_edge_p2), (collision_edge_p2, p)),
            #         # second triangle
            #         ((collision_edge_p1, collision_edge_p2), (collision_edge_p1, neighbour)),
            #         ((collision_edge_p1, neighbour), (collision_edge_p2, neighbour)),
            #         ((collision_edge_p1, collision_edge_p2), (collision_edge_p2, neighbour))
            #     )
            #
            #     rate_tri = lambda t: sum(abs(60 - self._angle(*a)) for a in t)
            #     a = rate_tri(old_triangles)
            #     b = rate_tri(new_triangles)
            #     new_is_more_equilateral = a > b
            #     print(f"rate_tri(old_triangles) = {a}")
            #     print(f"rate_tri(new_triangles) = {b}")
            #     if new_is_more_equilateral:
            #         self.replace_valid_triangulation((collision_edge_p1, collision_edge_p2), (p, neighbour))
            #         self.update_neighbour(p, neighbour, collision_edge_p1, collision_edge_p2)
            #         garbage_heap.append({collision_edge_p1, collision_edge_p2})
            #     else:
            #         garbage_heap.append({p, neighbour})
            #     print(f"garbage_heap = {garbage_heap}")

    @staticmethod
    def _angle(edge1, edge2):
        print(f"edge1 = {edge1}, edge2 = {edge2}")
        # print(f"(edge1[1][0] - edge1[0][0]) = {(edge1[1][0] - edge1[0][0])}")
        # print(f"(edge2[1][0] - edge2[0][0]) = {(edge2[1][0] - edge2[0][0])}")
        # print(f"_angle = {abs(180 / np.pi * (np.arctan((edge1[1][1] - edge1[0][1]) / (edge1[1][0] - edge1[0][0])) - np.arctan((edge2[1][1] - edge2[0][1]) / (edge2[1][0] - edge2[0][0]))))}")
        slope1 = dirichlet_tessellation.slope(edge1)
        slope2 = dirichlet_tessellation.slope(edge2)
        print(f"(180 / np.pi) * np.arctan(abs((slope2 - slope1)/(1 + slope1*slope2))) = {(180 / np.pi) * np.arctan(abs((slope2 - slope1)/(1 + slope1*slope2)))}")
        return (180 / np.pi) * np.arctan(abs((slope2 - slope1)/(1 + slope1*slope2)))
        # return abs(180 / np.pi * (np.arctan((edge1[1][1] - edge1[0][1]) / (edge1[1][0] - edge1[0][0]))
        #                           - np.arctan((edge2[1][1] - edge2[0][1]) / (edge2[1][0] - edge2[0][0]))))

    @staticmethod
    def _angles_in_triangle(triangle):
        # law of cosine
        # https://en.wikipedia.org/wiki/Solution_of_triangles

        a = np.linalg.norm(triangle[1] - triangle[2])
        b = np.linalg.norm(triangle[0] - triangle[1])
        c = np.linalg.norm(triangle[2] - triangle[0])

        alpha = (180 / np.pi) * np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        beta = (180 / np.pi) * np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
        gamma = 180 - alpha - beta

        return [alpha, beta, gamma]

    @staticmethod
    def slope(edge):
        if edge[0][1] == edge[1][1]:
            return 0
        return (edge[0][0] - edge[1][0]) / (edge[0][1] - edge[1][1])

    @staticmethod
    def get_all_edge_pairs(p1: h_tuple, p2: h_tuple, p3: h_tuple):
        return [(p1, p2), (p2, p3), (p1, p3)]

    def update_neighbour(self, p_new: h_tuple, p_neighbour: h_tuple, p_rem_1: h_tuple, p_rem_2: h_tuple):
        self.tiles[p_rem_1].remove(p_rem_2)
        self.tiles[p_rem_2].remove(p_rem_1)
        self.tiles[p_new].add(p_neighbour)
        self.tiles[p_neighbour].add(p_new)

    def replace_valid_triangulation(self, old_pair: Tuple[h_tuple, h_tuple],
                                    new_pair: Tuple[h_tuple, h_tuple]):
        # dann kante col_edge_p2 - col_edge_p1 entfernen und kante p - neighbour hinzufügen
        # The list comprehension will always just have a single element matching the condition
        self.valid_triangulation.remove(*[x for x in self.valid_triangulation if set(x) == set(old_pair)])
        self.valid_triangulation.add(new_pair)

    @staticmethod
    def intersect(p1: h_tuple, p2: h_tuple, p3: h_tuple, p4: h_tuple) -> bool:
        # First we vertical stack the points in an array
        vertical_stack = np.vstack([p1, p2, p3, p4])
        # Then we transform them to homogeneous coordinates, to perform a little trick
        homogeneous = np.hstack((vertical_stack, np.ones((4, 1))))
        # having our points in this form we can get the lines through the cross product
        line_1, line_2 = np.cross(homogeneous[0], homogeneous[1]), np.cross(homogeneous[2], homogeneous[3])
        # when we calculate the cross product of the lines we get intersect point
        x, y, z = np.cross(line_1, line_2)
        # print(f"{h_tuple.create([x/z, y/z])} und die Punkte sind: {p1}, {p2}, {p3}, {p4}")
        # print([(h_tuple.create([x/z, y/z]), x) for x in [p1, p2, p3, p4]])
        # we divide with z to turn back to 2D space

        # no intersection
        if z == 0:
            return False
        intersection = h_tuple.create([x / z, y / z])
        # if intersection is at one of the points
        if any([intersection == p for p in [p1, p2, p3, p4]]):
            return False
        # if the intersection is in the convex space of both points we good (aka not on the line segment)
        return dirichlet_tessellation.intersection_is_between(intersection, p1, p2, p3, p4)

    @staticmethod
    def intersection_is_between(intersection: h_tuple, p1: h_tuple, p2: h_tuple, p3, p4):
        # a = np.linalg.norm(intersection - p1) + np.linalg.norm(intersection - p2)
        # b = np.linalg.norm(p2 - p1)
        # c = a == b
        # print(c)
        # return c
        # return abs(intersection - p1) + abs(intersection - p2) == abs(p2 - p1)
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
        # print("hello")
        return seg1 and seg2


if __name__ == '__main__':
    _min, _max = -10, 10
    n = 7
    points = np.array([np.array(x) for x in ((2, 3), (6, 5), (3, 7), (8, 3), (5, 1), (8, 8), (-3, -2))])
    np.save('last_run.npy', points)
    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices)
    plt.plot(points[:, 0], points[:, 1], 'o')
    plt.show()

    d = dirichlet_tessellation()

    for p in points:
        d.append_point(p)

    print(f"finale d.valid_triangulation = {d.valid_triangulation}")
    print(f"len(d.valid_triangulation = {len(d.valid_triangulation)}")

    # xs = [h_tuple.create(x) for x in ((1,2), (3,4), (1,4), (3,6)) ]
    # print(dirichlet_tessellation.intersect(*xs))
    # pts = [np.array(x) for x in ((2, 3), (6, 5), (3, 7), (8, 3), (5, 1), (8, 8), (-3, -2))]
    # d = dirichlet_tessellation()

    # print(h_tuple.create([2,1]) == h_tuple.create([1, 2]))
    # cords A, B, D, E
    # Winkel altes dreieck1 = 108.44, 26.57, 45
    # Winkel altes dreieck2 = 33.69, 112.62, 33.69

    # Winkel neues dreieck1 = 49.4, 70.35, 60.26
    # Winkel neues dreieck2 = 59.04, 78.69, 42.27

    # for p in pts:
    #     d.append_point(p)
    #     print(f"d.valid_triangulation = {d.valid_triangulation}")
    #
    # print(f"finale d.valid_triangulation = {d.valid_triangulation}")
    # print(f"len(d.valid_triangulation = {len(d.valid_triangulation)}")

    # hash = lambda n: hash(())
    # np.ndarray
