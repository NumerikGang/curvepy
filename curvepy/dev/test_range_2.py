import numpy as np
from typing import Tuple, List, Dict, Optional

PointIndex = int
Triangle = Tuple[PointIndex, PointIndex, PointIndex]
Neighbours = List[Optional[Triangle]]

Radius = float
Center = Tuple[float, float]
Circle = Tuple[Center, Radius]


class Delaunay:
    def __init__(self, center=(0, 0), radius=1000):
        center = np.array(center)

        # Rechteck erstellen indem man den Radius mal die Basisvektoren nimmt
        self.coords = [center + radius * np.array([i, j]) for i, j in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]

        # Triangles werden wiefolgt gespeichert:
        # Der Key ist ein 3-Tuple, welches CCW aus den Indexen der self.coords bestehen.
        # Sprich das Tuple
        # (2,3,1)
        # bedeutet das Dreieck
        # (self.coords[2], self.coords[3], self.coords[1])
        self.triangles: Dict[Triangle, Neighbours] = {}

        # Hier speichern wir zu jedem Triangle (Ueber indexe definiert) das 2-Tuple (center, radius)
        self.circles: Dict[Triangle, Circle] = {}

        # Hier werden die Indexe der in den coords als Triangles gespeichert
        index_lower_left = 0
        index_lower_right = 1
        index_upper_right = 2
        index_upper_left = 3

        # Wir splitten den durch die self.coords definierten sapce in 2 Triangle auf, indem wir so eine Linie
        # durchziehen:
        # x────────────────────────────────┐
        # xx                               │
        # │ xx                             │
        # │   xxx              T2          │
        # │     xxxx                       │
        # │        xxx                     │
        # │          xxx                   │
        # │            xxxx                │
        # │               xxxx             │
        # │                  xxx           │
        # │       T1           xxxx        │
        # │                       xxx      │
        # │                         xxx    │
        # │                           xxx  │
        # │                             xxx│
        # └────────────────────────────────┴
        # Wir definieren sie Counter Clock Wise (CCW), angefangen beim Punkt gegenueber der Hypothenuse
        first_super_triangle = (index_lower_left, index_lower_right, index_upper_left)
        second_super_triangle = (index_upper_right, index_upper_left, index_lower_right)

        # TODO: Stimmt das, dass wir so die Datenstruktur nutzen?
        # Wir speichern die Triangles mit ihren Nachbarn. Offensichtlich sind T1 und T2 oben benachbart und haben
        # sonst keine Nachbarn, da sie den validen Bereich begrenzen
        self.triangles[first_super_triangle] = [second_super_triangle, None, None]
        self.triangles[second_super_triangle] = [first_super_triangle, None, None]

        self.circles[first_super_triangle] = self.circumcenter(first_super_triangle)
        self.circles[second_super_triangle] = self.circumcenter(second_super_triangle)

    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                     [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri.
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def add_point(self, p):
        """
        TODO: Replace mich
        Add point to the current DT, and refine it using Bowyer-Watson
        """
        p = np.array(p)
        # Wir schieben den Punkt ganz nach hinten (da wir ja alles in indexen speichern und somit darauf angewiesen sind
        # dass die vorherigen gleichbleiben).
        # (Laenge ist ja 1-indexiert, somit entspricht es dem naechsten Punkt)
        idx = len(self.coords)
        self.coords.append(p)
        # also gilt
        # assert self.coords[idx] == self.coords[-1]

        # Wir nehmen an, dass vor diesem Punkt die Delaunay-Triangulation optimal war
        # Somit muessen wir nur diesen Punkt beachten.
        # Per Definition ist ein Triangulationsdreieck nur valide wenn in seinem Umkreis kein weiterer Punkt liegt.
        bad_trinagles = [tri for tri in self.triangles if self.inCircleFast(tri, p)]

        # TODO: Verstehe folgendes
        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        #
        # Bisherige Vermutung:
        #   - Star Shape == Triforce mit dem Punkt im Mitteldreieck
        #   - each edge = Jede Kante des Dreiecks, welches den Punkt im Umkreis beinhaltet
        #   - opposite triangle = Das Dreieck, was nicht schlecht ist und trotzdem die edge teilt.
        #
        # UPDATE: Wir sammeln hierbei alle nicht-problematischen Nachbarn (siehe if in der while)
        boundaries: List[Tuple[PointIndex, PointIndex, Triangle]] = []

        # Wir starten bei einem "random" Triangle mit dessen ersten Edge
        T = bad_trinagles[0]
        edge = 0

        # Jetzt suchen wir das opposite triangle fuer jede edge, angefangen mit edge = 0 weil irgendwo muss man ja
        #
        while True:
            tri_opposite_to: Triangle = self.triangles[T][edge]
            if tri_opposite_to not in bad_trinagles:
                # wir speichern die Kante ab an die Tri_op angrenzt
                # und zwar die indices der Punkte die die Kante aufspannen
                boundaries.append((T[(edge + 1) % 3], T[(edge - 1) % 3], tri_opposite_to))

                edge = (edge + 1) % 3

                # Wenn der naechste Punkt
                if boundaries[0][0] == boundaries[-1][1]:
                    break
            else:
                # CCW edge nach der edge die wir grade betrachten
                # da an dieser noch aktuellen edge das tri_op liegt
                # wir wollen aber die ccw nächste da wir die aktuelle abgearbeitet haben
                same_edge_other_triangle = self.triangles[tri_opposite_to].index(T)
                edge = (same_edge_other_triangle + 1) % 3
                T = tri_opposite_to


