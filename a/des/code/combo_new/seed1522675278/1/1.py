import random
from collections import defaultdict

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
            out.append(sorted(pts))
    return out


def _runs(idxs):
    if not idxs:
        return []
    idxs = sorted(idxs)
    runs = [[idxs[0], idxs[0]]]
    for x in idxs[1:]:
        if x == runs[-1][1] + 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    return [(a, b) for a, b in runs]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match interior colored line keys to frame-side lock gaps, then paint outward strips in key colors.

    Input structure:
    - Black background with one grey single-cell rectangular frame.
    - The frame sides contain contiguous missing runs on their borders.
    - Inside the frame, each non-grey object is a monochrome horizontal or vertical line segment.

    Rule:
    - A horizontal key matches a top/bottom gap of the same length; a vertical key matches a left/right gap of the same length.
    - Each matched gap paints a solid rectangle from the gap outward to the grid edge using the key color.
    - The frame and the interior keys remain unchanged.

    Key insight:
    - Only segment length and orientation matter; position does not.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frame_pts = np.argwhere(g == GREY)
    r0, c0 = frame_pts.min(axis=0)
    r1, c1 = frame_pts.max(axis=0)
    holes = []
    for side, fixed, lo, hi in (
        ("top", r0, c0 + 1, c1 - 1),
        ("bottom", r1, c0 + 1, c1 - 1),
    ):
        cols = [c for c in range(lo, hi + 1) if g[fixed, c] != GREY]
        for a, b in _runs(cols):
            holes.append((side, a, b))
    for side, fixed, lo, hi in (
        ("left", c0, r0 + 1, r1 - 1),
        ("right", c1, r0 + 1, r1 - 1),
    ):
        rows = [r for r in range(lo, hi + 1) if g[r, fixed] != GREY]
        for a, b in _runs(rows):
            holes.append((side, a, b))
    buckets = defaultdict(list)
    for side, a, b in holes:
        orient = "h" if side in ("top", "bottom") else "v"
        buckets[(orient, b - a + 1)].append((side, a, b))
    inside = (g != BLACK) & (g != GREY)
    keys = []
    for comp in _components(inside):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        orient = "h" if len(rows) == 1 else "v"
        color = g[comp[0]]
        keys.append((orient, len(comp), color))
    keys.sort(key=lambda x: (x[0], x[1], x[2]))
    for orient, length, color in keys:
        side, a, b = buckets[(orient, length)].pop(0)
        if side == "top":
            out[:r0, a : b + 1] = color
        elif side == "bottom":
            out[r1 + 1 :, a : b + 1] = color
        elif side == "left":
            out[a : b + 1, :c0] = color
        else:
            out[a : b + 1, c1 + 1 :] = color
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(14, 24)
        w = random.randint(14, 24)
        r0 = random.randint(3, 6)
        c0 = random.randint(3, 6)
        r1 = random.randint(h - 7, h - 4)
        c1 = random.randint(w - 7, w - 4)
        g = np.zeros((h, w), dtype=int)
        g[r0, c0 : c1 + 1] = GREY
        g[r1, c0 : c1 + 1] = GREY
        g[r0 : r1 + 1, c0] = GREY
        g[r0 : r1 + 1, c1] = GREY
        specs = []
        used = {"top": set(), "bottom": set(), "left": set(), "right": set()}
        for side in random.sample(["top", "bottom", "left", "right"], random.randint(2, 4)):
            length = random.randint(2, 3)
            if side in ("top", "bottom"):
                choices = [x for x in range(c0 + 1, c1 - length + 1) if all(y not in used[side] for y in range(x, x + length))]
                if not choices:
                    continue
                a = random.choice(choices)
                for y in range(a, a + length):
                    g[r0 if side == "top" else r1, y] = BLACK
                    used[side].add(y)
                specs.append((side, a, a + length - 1))
            else:
                choices = [x for x in range(r0 + 1, r1 - length + 1) if all(y not in used[side] for y in range(x, x + length))]
                if not choices:
                    continue
                a = random.choice(choices)
                for y in range(a, a + length):
                    g[y, c0 if side == "left" else c1] = BLACK
                    used[side].add(y)
                specs.append((side, a, a + length - 1))
        if len(specs) < 2:
            continue
        taken = np.zeros_like(g, dtype=bool)
        ok = True
        for color, (side, a, b) in zip(random.sample(COLORS, len(specs)), specs):
            length = b - a + 1
            if side in ("top", "bottom"):
                choices = []
                for rr in range(r0 + 2, r1 - 1):
                    for cc in range(c0 + 1, c1 - length):
                        if not taken[rr, cc : cc + length].any():
                            choices.append((rr, cc))
                if not choices:
                    ok = False
                    break
                rr, cc = random.choice(choices)
                g[rr, cc : cc + length] = color
                for ccc in range(cc - 1, cc + length + 1):
                    for rrr in range(rr - 1, rr + 2):
                        if 0 <= rrr < h and 0 <= ccc < w:
                            taken[rrr, ccc] = True
            else:
                choices = []
                for rr in range(r0 + 1, r1 - length):
                    for cc in range(c0 + 2, c1 - 1):
                        if not taken[rr : rr + length, cc].any():
                            choices.append((rr, cc))
                if not choices:
                    ok = False
                    break
                rr, cc = random.choice(choices)
                g[rr : rr + length, cc] = color
                for rrr in range(rr - 1, rr + length + 1):
                    for ccc in range(cc - 1, cc + 2):
                        if 0 <= rrr < h and 0 <= ccc < w:
                            taken[rrr, ccc] = True
        if ok:
            return g
