import random
from collections import deque

import numpy as np

BLACK = 0
GREY = 5


def _neighbors(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _find_legend(grid: np.ndarray):
    if np.count_nonzero(grid[:, -1]) == 5:
        rows = np.where(grid[:, -1] != BLACK)[0]
        return "col", [int(grid[r, -1]) for r in rows]
    cols = np.where(grid[-1, :] != BLACK)[0]
    return "row", [int(grid[-1, c]) for c in cols]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _count_holes(grid: np.ndarray, cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        local[r - r0, c - c0] = True
    h, w = local.shape
    seen = np.zeros_like(local, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if local[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches = r == 0 or r == h - 1 or c == 0 or c == w - 1
            size = 0
            while q:
                cr, cc = q.popleft()
                size += 1
                for nr, nc in _neighbors(cr, cc, h, w):
                    if not local[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                            touches = True
                        q.append((nr, nc))
            if not touches and size == 1:
                holes += 1
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed grey region by the legend color indexed by its hole count.

    Input structure:
    - Black background with several disconnected grey filled regions.
    - Each grey region may contain 0 to 4 isolated black 1x1 holes.
    - A five-color legend sits either in the rightmost column or the bottom row, with black gaps between legend pixels.

    Rule:
    - Read the legend in top-to-bottom or left-to-right order to map hole counts 0..4 to colors.
    - For every grey connected component outside the legend, count its enclosed black 1x1 holes.
    - Repaint all grey pixels in that component with the legend color for that hole count, keeping holes black and leaving the legend unchanged.

    Key insight:
    - The legend is indexed by the number of visible holes inside each closed region rather than by the region's size or position.
    """
    out = grid.copy()
    legend_kind, legend = _find_legend(grid)
    work = grid.copy()
    if legend_kind == "col":
        work[:, -1] = BLACK
    else:
        work[-1, :] = BLACK
    for cells in _components(work == GREY):
        color = legend[_count_holes(work, cells)]
        for r, c in cells:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors_pool = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(400):
        use_col = rng.choice([True, False])
        h = rng.randint(14, 24)
        w = rng.randint(14, 24)
        grid = np.zeros((h, w), dtype=int)
        region_h = h if use_col else h - 1
        region_w = w - 1 if use_col else w
        n_regions = rng.randint(2, 4)
        placed = []
        ok = True
        for _idx in range(n_regions):
            rh = rng.randint(3, 6)
            rw = rng.randint(3, 6)
            if rh >= region_h - 1 or rw >= region_w - 1:
                ok = False
                break
            for _ in range(80):
                r0 = rng.randint(1, region_h - rh - 1)
                c0 = rng.randint(1, region_w - rw - 1)
                if all(r0 + rh + 1 < pr or pr + ph + 1 < r0 or c0 + rw + 1 < pc or pc + pw + 1 < c0 for pr, pc, ph, pw in placed):
                    break
            else:
                ok = False
                break
            grid[r0:r0 + rh, c0:c0 + rw] = GREY
            interior = [(r, c) for r in range(r0 + 1, r0 + rh - 1) for c in range(c0 + 1, c0 + rw - 1)]
            hole_count = rng.randint(0, min(4, len(interior)))
            rng.shuffle(interior)
            for r, c in interior[:hole_count]:
                grid[r, c] = BLACK
            placed.append((r0, c0, rh, rw))
        if not ok:
            continue
        legend_colors = rng.sample(colors_pool, 5)
        if use_col:
            rows = list(range(1, h, max(1, (h - 2) // 4)))[:5]
            rows = sorted(rows)
            while len(rows) < 5:
                rows.append(rows[-1] + 1)
            for r, color in zip(rows, legend_colors):
                if r >= h:
                    ok = False
                    break
                grid[r, -1] = color
        else:
            cols = list(range(1, w, max(1, (w - 2) // 4)))[:5]
            cols = sorted(cols)
            while len(cols) < 5:
                cols.append(cols[-1] + 1)
            for c, color in zip(cols, legend_colors):
                if c >= w:
                    ok = False
                    break
                grid[-1, c] = color
        if ok:
            return grid
    raise RuntimeError("failed to generate task 13/1")
