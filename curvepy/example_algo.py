import pprint
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
        if not self.tiles:
            self.tiles[p] = set()
            return

        nearest_p = min(self.tiles.keys(), key=lambda t: np.linalg.norm(t - p))
        self.tiles[p] = {nearest_p}
        self.tiles[nearest_p].add(p)

        # ERSTELLE TRIANGULATION ZWISCHEN p UND nearest_p
        self.valid_triangulation.add((p, nearest_p))

        garbage_heap = []

        # To make a more educated guess we solve any collisions __after__ adding the trivial connections
        while True:
            collisions_to_check = []
            # print(f"collisions_to_check: {collisions_to_check}")

            for neighbour in self.tiles[nearest_p]:
                if all(x == y for x, y in zip(p, neighbour)):
                    continue
                # print("b4")
                all_collisions = [x for x in self.valid_triangulation if self.intersect(neighbour, p, *x)
                                  and x not in garbage_heap and {neighbour, p} not in garbage_heap]
                # print("after (no homo)")
                if (not all_collisions) and ({p, neighbour} not in garbage_heap):
                    self.valid_triangulation.add((p, neighbour))
                    # TODO das muss auch unten gemacht werden wenn die neuen Dreiecke cooler sind
                    self.tiles[neighbour].add(p)
                    self.tiles[p].add(neighbour)
                elif len(all_collisions) == 1:
                    # 1 collision could be still fine
                    collisions_to_check.append([p, neighbour, *all_collisions[0]])
                # More than 1 collision is always not the best possible triangulation TODO Is that true? Probably not ;)

            if not collisions_to_check:
                return

            for p, neighbour, collision_edge_p1, collision_edge_p2 in collisions_to_check:
                new_triangles = (
                    # first triangle
                    ((p, neighbour), (p, collision_edge_p1)),
                    ((p, neighbour), (neighbour, collision_edge_p1)),
                    ((p, neighbour), (neighbour, collision_edge_p1)),
                    # second triangle
                    ((p, neighbour), (p, collision_edge_p2)),
                    ((p, collision_edge_p2), (neighbour, collision_edge_p2)),
                    ((p, neighbour), (neighbour, collision_edge_p2))
                )

                old_triangles = (
                    # first triangle
                    ((collision_edge_p1, collision_edge_p2), (collision_edge_p1, p)),
                    ((collision_edge_p1, p), (collision_edge_p2, p)),
                    ((collision_edge_p1, collision_edge_p2), (collision_edge_p2, p)),
                    # second triangle
                    ((collision_edge_p1, collision_edge_p2), (collision_edge_p1, neighbour)),
                    ((collision_edge_p1, neighbour), (collision_edge_p2, neighbour)),
                    ((collision_edge_p1, collision_edge_p2), (collision_edge_p2, neighbour))
                )

                rate_tri = lambda t: sum(abs(60 - self._angle(*a)) for a in t)
                new_is_more_equilateral = rate_tri(old_triangles) > rate_tri(new_triangles)
                if new_is_more_equilateral:
                    self.replace_valid_triangulation((collision_edge_p1, collision_edge_p2), (p, neighbour))
                    self.update_neighbour(p, neighbour, collision_edge_p1, collision_edge_p2)
                    garbage_heap.append({collision_edge_p1, collision_edge_p2})
                else:
                    garbage_heap.append({p, neighbour})

    @staticmethod
    def _angle(edge1, edge2):
        return abs(180 / np.pi * (np.arctan((edge1[1][1] - edge1[0][1]) / (edge1[1][0] - edge1[0][0]))
                                  - np.arctan((edge2[1][1] - edge2[0][1]) / (edge2[1][0] - edge2[0][0]))))

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
        return dirichlet_tessellation.intersection_is_between(intersection, p1, p2)

    @staticmethod
    def intersection_is_between(intersection: h_tuple, p1: h_tuple, p2: h_tuple):
        return abs(intersection - p1) + abs(intersection - p2) == abs(p2 - p1)


if __name__ == '__main__':
    # xs = [h_tuple.create(x) for x in ((1,2), (3,4), (1,4), (3,6)) ]
    # print(dirichlet_tessellation.intersect(*xs))
    pts = [np.array(x) for x in ((2, 3), (6, 5), (3, 7), (8, 3), (5, 1), (8, 8), (-3, -2))]
    d = dirichlet_tessellation()

    # print(h_tuple.create([2,1]) == h_tuple.create([1, 2]))
    # cords A, B, D, E
    # Winkel altes dreieck1 = 108.44, 26.57, 45
    # Winkel altes dreieck2 = 33.69, 112.62, 33.69

    # Winkel neues dreieck1 = 49.4, 70.35, 60.26
    # Winkel neues dreieck2 = 59.04, 78.69, 42.27

    for p in pts:
        d.append_point(p)

    print(d.valid_triangulation)

    # hash = lambda n: hash(())
    # np.ndarray
