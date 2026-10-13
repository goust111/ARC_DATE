import random

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each concentric square set by a centered cross whose arm length equals layer count and whose colors follow the inner-to-outer layer order.

    Input structure:
    - Several disjoint groups of concentric square frames appear on black background.
    - Frames in a group share a center but may use different colors.
    - There is one black cell gap between layers.

    Rule:
    - Count the layers in each center-aligned group.
    - Read the colors from inner layer to outer layer.
    - Erase the frames and draw a centered cross where arm step k uses the kth outer color.

    Key insight:
    - Nested squares encode both a length (number of layers) and a color sequence.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    groups = {}
    for color in [c for c in np.unique(g) if c != BLACK]:
        for comp in _components(g == color):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            ctr = ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
            groups.setdefault(ctr, []).append((max(rs) - min(rs), color))
    for (cr, cc), items in groups.items():
        seq = [c for _, c in sorted(items)]
        out[cr, cc] = seq[0]
        for k, color in enumerate(seq[1:], 1):
            out[cr - k, cc] = out[cr + k, cc] = out[cr, cc - k] = out[cr, cc + k] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((19, 19), dtype=int)
    centers = [(5, 5), (13, 13)]
    for cr, cc in centers:
        cols = random.sample(COLORS, random.randint(2, 2))
        for k, color in enumerate(reversed(cols), 1):
            r0, c0, r1, c1 = cr - k * 2, cc - k * 2, cr + k * 2, cc + k * 2
            g[r0, c0 : c1 + 1] = color
            g[r1, c0 : c1 + 1] = color
            g[r0 : r1 + 1, c0] = color
            g[r0 : r1 + 1, c1] = color
    return g
