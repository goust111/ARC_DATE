import random

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def _norm(pts):
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return frozenset((r - r0, c - c0) for r, c in pts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior colored shape to the edge-touching notch of the same color and translate it into that notch.

    Input structure:
    - A multicolor one-cell border surrounds the grid.
    - Each border color has one black inward notch touching the outer edge.
    - Inside the grid, one object of that same color matches the notch shape exactly.

    Rule:
    - For each color, find the notch component touching the boundary and the matching interior object.
    - Translate the object so its cells exactly cover the notch cells.
    - Erase the object's original position and keep all existing border cells unchanged.

    Key insight:
    - Color identifies the target border segment, while exact shape identifies the translation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    edge_notches = [c for c in _components(g == BLACK) if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in c)]
    for notch in edge_notches:
        neigh = [int(g[r + dr, c + dc]) for r, c in notch for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1] and g[r + dr, c + dc] != BLACK]
        color = max(set(neigh), key=neigh.count)
        hole = _norm(notch)
        for comp in _components(g == color):
            if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in comp):
                continue
            if _norm(comp) == hole:
                for r, c in comp:
                    out[r, c] = BLACK
                for r, c in notch:
                    out[r, c] = color
                break
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    g[0, :] = COLORS[0]
    g[-1, :] = COLORS[1]
    g[:, 0] = COLORS[2]
    g[:, -1] = COLORS[3]
    notches = {COLORS[0]: [(0, 5), (1, 5)], COLORS[1]: [(17, 12), (16, 12)], COLORS[2]: [(8, 0), (8, 1)], COLORS[3]: [(10, 17), (10, 16)]}
    for color, pts in notches.items():
        for r, c in pts:
            g[r, c] = BLACK
        rr, cc = random.randint(4, 13), random.randint(4, 13)
        for dr, dc in _norm(pts):
            g[rr + dr, cc + dc] = color
    return g
