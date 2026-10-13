import random
from collections import deque

import numpy as np


BG = 7
COLORS = [1, 2, 3, 4, 5]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Alternately shift ordered objects, discard objects clipped below area four, then bottom-pack occupied rows.

    Input structure:
    - The background is orange.
    - Several non-background 4-connected monochrome objects have distinct topmost rows.
    - Some objects touch the left or right grid edge, so the parity-directed shift can clip them.

    Rule:
    - Sort objects by their topmost bounding-box row.
    - Shift rank 1, 3, 5, ... objects one column left; shift rank 2, 4, 6, ... objects one column right.
    - Pixels shifted outside the grid are discarded, and objects with fewer than four remaining pixels are removed.
    - Delete all-orange rows after shifting and pack the remaining rows against the bottom, padding the top with orange.

    Key insight:
    - Survival depends on both rank parity and edge position because area is measured only after the clipping shift.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _components(g != BG)
    comps.sort(key=lambda comp: _bbox(comp)[0])
    shifted = np.full_like(g, BG)
    for i, comp in enumerate(comps):
        dc = -1 if i % 2 == 0 else 1
        color = int(g[comp[0]])
        cells = [(r, c + dc) for r, c in comp if 0 <= c + dc < w]
        if len(cells) < 4:
            continue
        for r, c in cells:
            shifted[r, c] = color

    rows = [shifted[r].copy() for r in range(h) if np.any(shifted[r] != BG)]
    out = np.full_like(g, BG)
    if rows:
        out[h - len(rows) :, :] = np.asarray(rows, dtype=int)
    return out


SHAPES = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 1]], dtype=bool),
]


def _place(grid, shape, r0, c0, color):
    h, w = shape.shape
    if r0 + h > grid.shape[0] or c0 < 0 or c0 + w > grid.shape[1]:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0], r0 + h + 1)
    cc1 = min(grid.shape[1], c0 + w + 1)
    if np.any(grid[rr0:rr1, cc0:cc1] != BG):
        return False
    grid[r0 : r0 + h, c0 : c0 + w][shape] = color
    return True


def _gen_attempt():
    n = random.randint(6, 8)
    h = random.randint(max(21, n * 3 + 2), 28)
    w = random.randint(14, 20)
    grid = np.full((h, w), BG, dtype=int)
    rows = sorted(random.sample(range(1, h - 3), n))
    if len(set(rows)) < n or any(b - a < 2 for a, b in zip(rows, rows[1:])):
        return None
    for i, r0 in enumerate(rows):
        shape = random.choice(SHAPES).copy()
        color = random.choice(COLORS)
        sw = shape.shape[1]
        if i == 0:
            c0 = 0
        elif i == 1:
            c0 = w - sw
        elif random.random() < 0.35:
            c0 = 0 if i % 2 == 0 else w - sw
        else:
            c0 = random.randint(2, w - sw - 3)
        if not _place(grid, shape, r0, c0, color):
            return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
