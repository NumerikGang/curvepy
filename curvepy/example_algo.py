from dataclasses import dataclass
import numpy as np
from typing import List, Set, Union, Dict, Tuple


@dataclass
class tile:
    center: np.ndarray  # 2D float
    neighbours: list  # of tiles


class dirichlet_tessellation:

    def __init__(self):
        self.tiles: List[tile] = []
        self.valid_triangulation: List[Tuple[np.ndarray, np.ndarray]] = []

    def append_point(self, p: np.ndarray):
        if not self.tiles:
            self.tiles.append(tile(p, []))
            return

        min_idx = np.argmin([np.linalg.norm(p - t.center) for t in self.tiles])
        nearest_tile = self.tiles[min_idx]
        p_tile = tile(center=p, neighbours=[])
        nearest_tile.neighbours.append(p_tile)
        p_tile.neighbours.append(nearest_tile)
        self.tiles.append(p_tile)

        # ERSTELLE TRIANGULATION ZWISCHEN p UND nearest_tile
        self.valid_triangulation.append((p, nearest_tile.center))

        # To make a more educated guess we solve any collisions __after__ adding the trivial connections
        collisions_to_check = []

        for neighbour in nearest_tile.neighbours:
            all_collisions = [x for x in self.valid_triangulation if self._intersect(neighbour.center, p, *x)]
            if len(all_collisions) == 0:
                self.valid_triangulation.append((p, neighbour.center))
                neighbour.neighbours.append(
                    p_tile)  # TODO das muss auch unten gemacht werden wenn die neuen Dreiecke cooler sind
            elif len(all_collisions) == 1:
                # 1 collision could be still fine
                collisions_to_check.append([p, neighbour.center, *all_collisions[0]])
            # More than 1 collision is always not the best possible triangulation TODO is that true? Yesn't

        for p, neighbour, collision_edge_p1, collision_edge_p2 in collisions_to_check:
            new_triangles = (
                # first triangle
                (p, neighbour),
                (p, collision_edge_p1),
                (neighbour, collision_edge_p1),
                # ((p, neighbour), (p, collision_edge_p1)),
                # ((p, neighbour), (neighbour, collision_edge_p1)),
                # ((p, neighbour), (neighbour, collision_edge_p1)),
                # second triangle
                self.get_all_edgepairs(p, neighbour, collision_edge_p2)
            )

            old_triangles = (
                # first triangle
                self.get_all_edgepairs(collision_edge_p1, collision_edge_p2, p),
                # second triangle
                self.get_all_edgepairs(collision_edge_p1, collision_edge_p2, neighbour)
            )

            rate_tri = lambda t: sum(abs(60 - self._angle(*a)) for a in t)
            new_is_more_equilateral = rate_tri(old_triangles) > rate_tri(new_triangles)
            if new_is_more_equilateral:
                # dann kante col_edge_p2 - col_edge_p1 entfernen und kante p - neighbour hinzufügen
                collision_edge_p1
                ...

    @staticmethod
    def _angle(edge1, edge2):
        return abs(180 / np.pi * (np.arctan((edge1[1][1] - edge1[0][1]) / (edge1[1][0] - edge1[0][0]))
                                  - np.arctan((edge2[1][1] - edge2[0][1]) / (edge2[1][0] - edge2[0][0]))))

    @staticmethod
    def get_all_edgepairs(A, B, C):
        return [(A, B), (B, C), (A, C)]

    @staticmethod
    def _rate_triangle(A, B, C):
        ...

    @staticmethod
    def _intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
        # First we vertical stack the points in an array
        vertical_stack = np.vstack([p1, p2, p3, p4])
        # Then we transform them to homogeneous coordinates, to perform a little trick
        homogeneous = np.hstack((vertical_stack, np.ones((4, 1))))
        # having our points in this form we can get the lines through the cross product
        line_1, line_2 = np.cross(homogeneous[0], homogeneous[1]), np.cross(homogeneous[2], homogeneous[3])
        # when we calculate the cross product of the lines we get intersect point
        x, y, z = np.cross(line_1, line_2)
        if z == 0:
            return False
        # we divide with z to turn back to 2D space
        return True
