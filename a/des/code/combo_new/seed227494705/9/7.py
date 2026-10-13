import random
from collections import deque

import numpy as np


BG = 0
GREEN = 3


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            yield pts


def _gaps(g):
    h, w = g.shape
    result = []
    for side, cells in [
        ("top", [(0, c) for c in range(1, w - 1)]),
        ("bottom", [(h - 1, c) for c in range(1, w - 1)]),
        ("left", [(r, 0) for r in range(1, h - 1)]),
        ("right", [(r, w - 1) for r in range(1, h - 1)]),
    ]:
        run = []
        for p in cells:
            if int(g[p]) == BG:
                run.append(p)
            else:
                if run:
                    result.append((side, list(run)))
                    run = []
        if run:
            result.append((side, list(run)))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push internal bars out through nearest same-width border gaps.

    Input structure:
    - Black background enclosed by a green one-pixel border with black gaps.
    - Interior contains separated horizontal or vertical colored bars of width one or two.
    - Gap width matches the bar width that should exit through that side.

    Rule:
    - For each colored bar, compute its thickness and find same-width border gaps.
    - Choose the same-width gap with smallest perpendicular distance to the bar.
    - Remove that bar from the grid, leaving the green border and its black gaps unchanged.

    Key insight:
    - Gap width pairs bars to exits, and edge position gives the push direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    gaps = _gaps(g)
    for color in [int(x) for x in np.unique(g) if int(x) not in (BG, GREEN)]:
        for pts in _components(g == color):
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            height = max(rows) - min(rows) + 1
            width = max(cols) - min(cols) + 1
            thickness = min(height, width)
            candidates = []
            for side, run in gaps:
                if len(run) != thickness:
                    continue
                if side in ("top", "bottom"):
                    dist = min(abs(r - run[0][0]) for r in rows)
                else:
                    dist = min(abs(c - run[0][1]) for c in cols)
                candidates.append((dist, side, run))
            if not candidates:
                continue
            for r, c in pts:
                out[r, c] = BG
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 4, 6, 7, 8, 9]
    h = rng.randint(14, 26)
    w = rng.randint(14, 26)
    g = np.zeros((h, w), dtype=int)
    g[0, :] = GREEN
    g[-1, :] = GREEN
    g[:, 0] = GREEN
    g[:, -1] = GREEN
    n = rng.randint(2, 5)
    used = set()
    for color in rng.sample(colors, n):
        orient = rng.choice(["h", "v"])
        thickness = rng.randint(1, 2)
        if orient == "h":
            length = rng.randint(4, w - 6)
            r = rng.randint(2, h - thickness - 2)
            c = rng.randint(2, w - length - 2)
            cells = {(r + rr, c + cc) for rr in range(thickness) for cc in range(length)}
            side = rng.choice(["top", "bottom"])
            gc = min(max(c + length // 2, 1), w - thickness - 1)
            for k in range(thickness):
                g[0 if side == "top" else h - 1, gc + k] = BG
        else:
            length = rng.randint(4, h - 6)
            r = rng.randint(2, h - length - 2)
            c = rng.randint(2, w - thickness - 2)
            cells = {(r + rr, c + cc) for rr in range(length) for cc in range(thickness)}
            side = rng.choice(["left", "right"])
            gr = min(max(r + length // 2, 1), h - thickness - 1)
            for k in range(thickness):
                g[gr + k, 0 if side == "left" else w - 1] = BG
        if cells & used:
            continue
        used |= cells
        for p in cells:
            g[p] = color
    return g
