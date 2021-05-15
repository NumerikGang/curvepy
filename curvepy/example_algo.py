from dataclasses import dataclass
import numpy as np
from typing import List, Set

@dataclass
class tile:
    center: np.ndarray # 2D float
    neighbours: list # of tiles


class dirichlet_tessellation:

    def __init__(self):
        self.tiles : List[tile] = []
        self.valid_triangulation: List[Set[np.ndarray, np.ndarray]] = []


    def append_point(self, p: np.ndarray):
        if not self.tiles:
            self.tiles.append(tile(p, []))
            return

        min_idx = np.argmin([np.linalg.norm(p - t.center) for t in self.tiles])
        act_tile = self.tiles[min_idx]

        # ERSTELLE TRIANGULATION ZWISCHEN p UND act_tile

        # ZUERST ALLE KOLLISIONSFREIEN
        for n in act_tile.neighbours:
            if self._intersect(n,p):
                continue
            # FÜGE TRIANGULATION HINZU

        # DANN ALLE MIT KOLLISIONEN
        for n in act_tile.neighbours:
            if self._intersect(n,p):
                if new_is_more_gleichseitig_sprich_abs_minus_60_minimieren(self, (n,p)):
                    # replatziere alte Kante mit neuen kanten
                    # TODO was passiert wenn wir mehr als eine Kollision haben? -> trashen wir die neue Kante dann?

    def _intersect(self, n,p):
        return True

# sum(abs(60-x) for x in [39.80, 71.88, 68.31, 48.63, 68.31, 63.06])
# sum(abs(60-y) for y in [39.04, 116.94, 24.01, 32.84, 108.11, 39.04])

# sum(abs(60-x) for x in [76.95, 85.12,  17.93, 18.19, 140.55, 21.25])
# sum(abs(60-y) for y in [30.90, 36.12, 112.98, 46.05, 106.37, 27.57])

"""

"""