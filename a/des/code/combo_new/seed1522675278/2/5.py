import random

import numpy as np


BLACK = 0
GREY = 5
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
    Count grey instruction cells inside each object, rotate the object's occupied mask by that count, and erase the markers.

    Input structure:
    - Black background with disconnected objects made of one foreground color plus embedded grey cells.
    - Each object's non-black cells stay within one bounding box.
    - Grey cells are part of the object mask but not part of the final color.

    Rule:
    - 0, 1, 2, 3 grey cells mean 0, 90, 180, 270 degrees clockwise rotation.
    - Rotation is applied to the full occupied mask inside the object bounding box.
    - The rotated occupied cells are repainted with the object's base color only.

    Key insight:
    - Only the number of grey cells matters; their positions are not semantic after rotation.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    for comp in _components(g != BLACK):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub != BLACK
        count = int(np.count_nonzero(sub == GREY))
        color = int(np.bincount(sub[(sub != BLACK) & (sub != GREY)]).argmax())
        rot = np.rot90(mask, (-count) % 4)
        out[r0 : r0 + rot.shape[0], c0 : c0 + rot.shape[1]][rot] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((20, 20), dtype=int)
    shapes = [np.array([[1, 1, 0], [0, 1, 1]]), np.array([[1, 1, 1], [1, 0, 0]]), np.array([[1, 1], [1, 1]])]
    for color, shape, count, pos in zip(random.sample(COLORS, 3), shapes, [0, 1, 2], [(2, 2), (8, 6), (13, 12)]):
        r0, c0 = pos
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c]:
                    g[r0 + r, c0 + c] = color
        cells = [(r0 + r, c0 + c) for r in range(shape.shape[0]) for c in range(shape.shape[1]) if shape[r, c]]
        for r, c in random.sample(cells, count):
            g[r, c] = GREY
    return g
