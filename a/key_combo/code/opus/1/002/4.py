import random
from collections import deque

import numpy as np


BG = 0
MARKER_COLORS = [2, 1, 3]
YELLOW = 4
GRAY = 5


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


BODY_OFFSETS = {
    "up": [(1, 0), (1, -1), (1, 1), (2, 0), (2, -1), (2, 1), (3, 0)],
    "down": [(-1, 0), (-1, -1), (-1, 1), (-2, 0), (-2, -1), (-2, 1), (-3, 0)],
    "left": [(0, 1), (-1, 1), (1, 1), (0, 2), (-1, 2), (1, 2), (0, 3)],
    "right": [(0, -1), (-1, -1), (1, -1), (0, -2), (-1, -2), (1, -2), (0, -3)],
}


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


def _direction(gray_cells, marker):
    cr = sum(r for r, _ in gray_cells) / len(gray_cells)
    cc = sum(c for _, c in gray_cells) / len(gray_cells)
    dr = marker[0] - cr
    dc = marker[1] - cc
    if abs(dr) >= abs(dc):
        return (1, 0) if dr > 0 else (-1, 0)
    return (0, 1) if dc > 0 else (0, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast each marker-colored ray in the direction encoded by its gray sprite's
    centroid bias; a hit on one yellow mirror marker fills the whole pair gap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    sprite_mask = (g == GRAY) | np.isin(g, MARKER_COLORS)

    hits = []
    for comp in _components(sprite_mask):
        gray_cells = [(r, c) for r, c in comp if g[r, c] == GRAY]
        markers = [(r, c) for r, c in comp if int(g[r, c]) in MARKER_COLORS]
        assert len(gray_cells) == 7 and len(markers) == 1, "expected one seven-pixel directional sprite"
        marker = markers[0]
        color = int(g[marker])
        dr, dc = _direction(gray_cells, marker)
        r, c = marker[0] + dr, marker[1] + dc
        hit = None
        while 0 <= r < h and 0 <= c < w:
            if g[r, c] == YELLOW:
                out[r, c] = color
                hit = (r, c, color)
                break
            if g[r, c] != BG:
                break
            out[r, c] = color
            r += dr
            c += dc
        if hit is not None:
            hits.append(hit)

    for r, c, color in hits:
        pc = w - 1 - c
        lo, hi = sorted((c, pc))
        for cc in range(lo, hi + 1):
            if out[r, cc] in (BG, YELLOW):
                out[r, cc] = color
    return out


def _can_place(grid, cells):
    for r, c in cells:
        if not (1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1):
            return False
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] not in (BG, YELLOW) and (nr, nc) not in cells:
                return False
    return True


def _place_sprite(grid, direction, color):
    dr, dc = DIRS[direction]
    h, w = grid.shape
    for _ in range(150):
        mr = random.randint(4, h - 5)
        mc = random.randint(4, w - 5)
        dist = random.randint(3, 6)
        tr, tc = mr + dr * dist, mc + dc * dist
        pc = w - 1 - tc
        if not (1 <= tr < h - 1 and 1 <= tc < w - 1 and tc != pc):
            continue
        ray = [(mr + dr * i, mc + dc * i) for i in range(1, dist)]
        body = [(mr + rr, mc + cc) for rr, cc in BODY_OFFSETS[direction]]
        cells = [(mr, mc)] + body
        if not all(0 <= r < h and 0 <= c < w and grid[r, c] == BG for r, c in ray + [(tr, tc), (tr, pc)]):
            continue
        if any(grid[tr, cc] not in (BG, YELLOW) for cc in range(min(tc, pc), max(tc, pc) + 1)):
            continue
        if not _can_place(grid, cells):
            continue
        for r, c in body:
            grid[r, c] = GRAY
        grid[mr, mc] = color
        grid[tr, tc] = YELLOW
        grid[tr, pc] = YELLOW
        return True
    return False


def _add_extra_pair(grid):
    h, w = grid.shape
    for _ in range(80):
        r = random.randint(1, h - 2)
        c = random.randint(2, w // 2 - 1)
        pc = w - 1 - c
        if all(grid[r, cc] == BG for cc in range(c, pc + 1)):
            grid[r, c] = YELLOW
            grid[r, pc] = YELLOW
            return


def _gen_attempt():
    size = random.choice([15, 17, 19])
    grid = np.zeros((size, size), dtype=int)
    specs = random.sample(list(DIRS), random.randint(2, 3))
    colors = random.sample(MARKER_COLORS, len(specs))
    for direction, color in zip(specs, colors):
        if not _place_sprite(grid, direction, color):
            return None
    if random.random() < 0.8:
        _add_extra_pair(grid)
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(700):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
