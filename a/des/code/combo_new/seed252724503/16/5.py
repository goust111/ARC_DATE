import random
from collections import deque

import numpy as np


LINE = 5


def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each colored rectangle across its nearest grey guide line and scale it by that gap distance.

    Input structure:
    - The grid has a dominant background color plus full-span grey horizontal and vertical lines.
    - Away from those lines, each non-grey foreground object is a single-color solid rectangle.
    - Every rectangle has a unique nearest guide line at distance one to three cells.

    Rule:
    - Measure the smallest gap D from each rectangle to any full-span grey line.
    - Reflect the rectangle across that nearest line so its closest edge is the same distance D from the line on the opposite side.
    - Replace each source pixel by a D×D block of the same color, clipping at the canvas boundary.
    - Remove the original rectangle and keep the grey guide lines unchanged.

    Key insight:
    - The same geometric distance to the nearest guide line controls both the reflection destination and the integer scale factor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg(g)
    out = np.full_like(g, bg)
    out[g == LINE] = LINE
    line_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == LINE)]
    line_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == LINE)]
    rect_mask = (g != bg) & (g != LINE)
    for cells in _components(rect_mask):
        pts = np.array(cells)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        color = int(g[cells[0]])
        candidates = []
        for lr in line_rows:
            if lr < r0:
                candidates.append((r0 - lr - 1, "down", lr))
            elif lr > r1:
                candidates.append((lr - r1 - 1, "up", lr))
        for lc in line_cols:
            if lc < c0:
                candidates.append((c0 - lc - 1, "right", lc))
            elif lc > c1:
                candidates.append((lc - c1 - 1, "left", lc))
        d, direction, axis = min(candidates, key=lambda x: x[0])
        hh = (r1 - r0 + 1) * d
        ww = (c1 - c0 + 1) * d
        if direction == "left":
            nr0, nc0 = r0, axis + d
        elif direction == "right":
            nr0, nc0 = r0, axis - d - ww + 1
        elif direction == "up":
            nr0, nc0 = axis + d, c0
        else:
            nr0, nc0 = axis - d - hh + 1, c0
        for rr in range(max(0, nr0), min(g.shape[0], nr0 + hh)):
            for cc in range(max(0, nc0), min(g.shape[1], nc0 + ww)):
                out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(18, 28), random.randint(18, 28)
        bg = 0
        grid = np.full((h, w), bg, dtype=int)
        line_rows = sorted(random.sample(range(4, h - 4), random.randint(1, 2)))
        line_cols = sorted(random.sample(range(4, w - 4), random.randint(1, 2)))
        grid[line_rows, :] = LINE
        grid[:, line_cols] = LINE
        occ = grid == LINE
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
        ok = True
        for color in colors:
            placed = False
            for _ in range(100):
                hh, ww = random.randint(1, 3), random.randint(1, 4)
                r0 = random.randint(1, h - hh - 2)
                c0 = random.randint(1, w - ww - 2)
                r1, c1 = r0 + hh - 1, c0 + ww - 1
                if occ[r0 : r1 + 1, c0 : c1 + 1].any():
                    continue
                candidates = []
                for lr in line_rows:
                    if lr < r0:
                        candidates.append((r0 - lr - 1, "down", lr))
                    elif lr > r1:
                        candidates.append((lr - r1 - 1, "up", lr))
                for lc in line_cols:
                    if lc < c0:
                        candidates.append((c0 - lc - 1, "right", lc))
                    elif lc > c1:
                        candidates.append((lc - c1 - 1, "left", lc))
                if not candidates:
                    continue
                candidates.sort(key=lambda x: x[0])
                if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                    continue
                d, direction, axis = candidates[0]
                if not (1 <= d <= 3):
                    continue
                hh2, ww2 = hh * d, ww * d
                if direction == "left":
                    nr0, nc0 = r0, axis + d
                elif direction == "right":
                    nr0, nc0 = r0, axis - d - ww2 + 1
                elif direction == "up":
                    nr0, nc0 = axis + d, c0
                else:
                    nr0, nc0 = axis - d - hh2 + 1, c0
                if nr0 < 0 or nc0 < 0 or nr0 + hh2 > h or nc0 + ww2 > w:
                    continue
                if occ[nr0 : nr0 + hh2, nc0 : nc0 + ww2].any():
                    continue
                grid[r0 : r0 + hh, c0 : c0 + ww] = color
                occ[r0 : r0 + hh, c0 : c0 + ww] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate input")
