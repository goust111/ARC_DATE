import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5


def _components(mask: np.ndarray):
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


def _border(r0, c0, r1, c1):
    for c in range(c0, c1 + 1):
        yield r0, c
        yield r1, c
    for r in range(r0 + 1, r1):
        yield r, c0
        yield r, c1


def _rectangles(g):
    rects = []
    for pts in _components(g == FRAME):
        rows = [p[0] for p in pts]
        cols = [p[1] for p in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(pts) != set(_border(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        vals, counts = np.unique(inner[inner != BG], return_counts=True)
        vals = [int(v) for v in vals if int(v) != FRAME]
        if not vals:
            continue
        fill = max(vals, key=lambda v: int(np.count_nonzero(inner == v)))
        markers = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(g[r, c]) not in (BG, FRAME, fill)]
        rects.append({"bbox": (r0, c0, r1, c1), "fill": fill, "level": len(markers)})
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Center-stack filled rectangles in the order encoded by their marker counts.

    Input structure:
    - Black background with several non-overlapping grey rectangular frames.
    - Each frame interior has one fill color and one to four singleton marker pixels of other colors.
    - All output keeps the original grid size.

    Rule:
    - Count the marker pixels inside each rectangle; the count is that rectangle's layer level.
    - Remove all original rectangles and redraw clean filled rectangles centered on the grid.
    - Draw lower marker counts first and higher counts later, so higher levels cover lower ones.

    Key insight:
    - Marker count is not decoration; it is the stacking order for centered rendering.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)
    for rect in sorted(_rectangles(g), key=lambda item: (item["level"], item["bbox"])):
        r0, c0, r1, c1 = rect["bbox"]
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        nr0 = (h - rh) // 2
        nc0 = (w - rw) // 2
        nr1, nc1 = nr0 + rh - 1, nc0 + rw - 1
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = rect["fill"]
        for r, c in _border(nr0, nc0, nr1, nc1):
            out[r, c] = FRAME
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        h = rng.randint(16, 30)
        w = rng.randint(16, 30)
        g = np.zeros((h, w), dtype=int)
        n = rng.randint(2, 4)
        fills = rng.sample(colors, n)
        marker_pool = [c for c in colors if c not in fills]
        occupied = set()
        ok = True
        for level, fill in enumerate(fills, start=1):
            placed = False
            for _try in range(100):
                rh = rng.randint(4, 8)
                rw = rng.randint(4, 8)
                r0 = rng.randint(1, h - rh - 1)
                c0 = rng.randint(1, w - rw - 1)
                r1, c1 = r0 + rh - 1, c0 + rw - 1
                pad = {(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)}
                if pad & occupied:
                    continue
                occupied |= pad
                g[r0 + 1 : r1, c0 + 1 : c1] = fill
                for r, c in _border(r0, c0, r1, c1):
                    g[r, c] = FRAME
                cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
                for (mr, mc), mcol in zip(rng.sample(cells, level), rng.sample(marker_pool, level)):
                    g[mr, mc] = mcol
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return g
    raise RuntimeError("generate_input failed")
