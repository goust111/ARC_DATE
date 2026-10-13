import random

import numpy as np


BLACK = 0
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SHAPES = {
    "right": {(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)},
    "left": {(1, 0), (1, 1), (1, 2), (0, 0), (2, 0)},
    "up": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "down": {(2, 0), (2, 1), (2, 2), (0, 1), (1, 1)},
}
CW = {"right": "down", "down": "left", "left": "up", "up": "right"}


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


def _kind(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, c0 = min(rs), min(cs)
    rel = {(r - r0, c - c0) for r, c in comp}
    for k, v in SHAPES.items():
        if rel == v:
            return k
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each arrow along its pointing axis to the boundary-near stop, and spawn one clockwise-rotated static copy at every same-colored trigger point on the swept tail line.

    Input structure:
    - Colored 5-cell arrows and same-colored singleton triggers sit on black background.
    - Arrow direction comes from its tip shape.
    - Triggers are isolated single cells of an arrow's color.

    Rule:
    - Move each arrow until its foremost cell is one step before the boundary.
    - Any trigger on the swept tail line is replaced by a clockwise-rotated arrow centered there.
    - Trigger copies do not move again.

    Key insight:
    - The moving arrow and the static spawned copies are driven by the same color but different rules.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    singles = {(r, c): int(g[r, c]) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] != BLACK}
    for comp in _components(g != BLACK):
        if len(comp) != 5:
            continue
        color = int(g[comp[0]])
        kind = _kind(comp)
        r0, c0 = min(r for r, _ in comp), min(c for _, c in comp)
        if kind == "right":
            shift = g.shape[1] - 2 - max(c for _, c in comp)
            tail_line = [(r0 + 1, c) for c in range(c0, c0 + shift + 1)]
        elif kind == "left":
            shift = 1 - min(c for _, c in comp)
            tail_line = [(r0 + 1, c) for c in range(c0 + 2, c0 + shift + 1, -1)]
        elif kind == "up":
            shift = 1 - min(r for r, _ in comp)
            tail_line = [(r, c0 + 1) for r in range(r0 + 2, r0 + shift + 1, -1)]
        else:
            shift = g.shape[0] - 2 - max(r for r, _ in comp)
            tail_line = [(r, c0 + 1) for r in range(r0, r0 + shift + 1)]
        dr, dc = {"right": (0, shift), "left": (0, shift), "up": (shift, 0), "down": (shift, 0)}[kind]
        for rr, cc in SHAPES[kind]:
            out[r0 + rr + dr, c0 + cc + dc] = color
        for tr, tc in tail_line:
            if singles.get((tr, tc)) == color:
                nk = CW[kind]
                for rr, cc in SHAPES[nk]:
                    ar, ac = tr + rr - 1, tc + cc - 1
                    if 0 <= ar < out.shape[0] and 0 <= ac < out.shape[1]:
                        out[ar, ac] = color
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    for color, kind, pos in zip(random.sample(COLORS, 2), random.sample(list(SHAPES), 2), [(3, 3), (10, 10)]):
        r0, c0 = pos
        for r, c in SHAPES[kind]:
            g[r0 + r, c0 + c] = color
        if kind in ("right", "left"):
            g[r0 + 1, c0 + 4] = color
        else:
            g[r0 + 4, c0 + 1] = color
    return g
