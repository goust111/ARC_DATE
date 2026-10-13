from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FG = 8
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

QUADRANT_FILL = {
    "ul": 6,
    "ur": 2,
    "ll": 1,
    "lr": 3,
}

MIN_H, MAX_H = 5, 7
MIN_W, MAX_W = 6, 8
MIN_AREA, MAX_AREA = 8, 15
MAX_GEN_ATTEMPTS = 1200


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """ARC-style 4-connected components as (N, 2) `(r, c)` arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _top_left(coords: np.ndarray) -> tuple[int, int]:
    idx = np.lexsort((coords[:, 1], coords[:, 0]))[0]
    r, c = coords[idx]
    return int(r), int(c)


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rr = coords[:, 0]
    cc = coords[:, 1]
    return bool(np.any((rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)))


def _enclosed_regions(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected True-regions that do not touch the outer border."""
    regions = []
    for coords in _components(mask):
        if not _touches_border(coords, mask.shape):
            regions.append(coords)
    return regions


def _compose(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((2 * h, 2 * w), dtype=int)
    out[:h, :w] = np.rot90(grid, 2)
    out[:h, w:] = np.flipud(grid)
    out[h:, :w] = np.fliplr(grid)
    out[h:, w:] = grid
    return out


def _quadrant_color(r: int, c: int, h: int, w: int) -> int:
    if r < h and c < w:
        return QUADRANT_FILL["ul"]
    if r < h and c >= w:
        return QUADRANT_FILL["ur"]
    if r >= h and c < w:
        return QUADRANT_FILL["ll"]
    return QUADRANT_FILL["lr"]


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-concatenate the teal shape, then color each emergent enclosed black region by quadrant.

    Input structure:
    - The grid has black background and exactly one 4-connected teal foreground shape.
    - The teal shape is irregular, touches the input border, and has no enclosed black hole on its own.
    - The grid is small enough that doubling both axes still stays within ARC limits.

    Rule:
    - Build a 2H x 2W output: place the input in the lower-right quadrant, its left-right mirror in the lower-left,
      its top-bottom mirror in the upper-right, and its 180-degree rotation in the upper-left.
    - In that doubled grid, find every 4-connected black region and ignore the ones connected to the outer border.
    - For each enclosed black region, take its topmost then leftmost pixel and recolor the whole region by that pixel's quadrant:
      upper-left -> pink(6), upper-right -> red(2), lower-left -> blue(1), lower-right -> green(3).
    - Keep all teal pixels unchanged and keep all non-enclosed background black.

    Key insight:
    - The input shape is hole-free by itself, but the four mirrored copies close each other across the seams and create
      new enclosed background regions whose fill colors are determined by quadrant position in the stitched image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BG, FG}

    fg_components = _components(g == FG)
    assert len(fg_components) == 1

    h, w = g.shape
    out = _compose(g)

    for region in _enclosed_regions(out == BG):
        rr, cc = _top_left(region)
        color = _quadrant_color(rr, cc, h, w)
        out[region[:, 0], region[:, 1]] = color

    return out


def _connect_path(cells: set[tuple[int, int]], a: tuple[int, int], b: tuple[int, int]) -> None:
    r, c = a
    tr, tc = b
    if random.random() < 0.5:
        while r != tr:
            r += 1 if tr > r else -1
            cells.add((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            cells.add((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            cells.add((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            cells.add((r, c))


def _grow_candidate() -> np.ndarray | None:
    h = random.randint(MIN_H, MAX_H)
    w = random.randint(MIN_W, MAX_W)
    target_area = random.randint(MIN_AREA, MAX_AREA)

    cells: set[tuple[int, int]] = {(0, 0)}
    top_c = random.randint(1, min(w - 2, 4))
    left_r = random.randint(1, min(h - 2, 4))
    _connect_path(cells, (0, 0), (0, top_c))
    _connect_path(cells, (0, 0), (left_r, 0))

    pivots = random.choices([1, 2], weights=[3, 2])[0]
    start = random.choice([(0, 0), (0, top_c), (left_r, 0)])
    for _ in range(pivots):
        pivot = (random.randint(1, h - 2), random.randint(1, w - 2))
        _connect_path(cells, start, pivot)
        start = pivot

    if len(cells) > target_area:
        return None

    while len(cells) < target_area:
        weighted: dict[tuple[int, int], int] = {}
        for r, c in tuple(cells):
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in cells:
                    continue
                adj = sum((nr + dr2, nc + dc2) in cells for dr2, dc2 in DIRS4)
                if adj == 0:
                    continue
                weight = 1 + 3 * adj
                weight += max(0, 4 - nr) + max(0, 4 - nc)
                if nr == 0 or nc == 0:
                    weight += 3
                if nr == h - 1 or nc == w - 1:
                    weight += 1
                weighted[(nr, nc)] = max(weighted.get((nr, nc), 0), weight)
        if not weighted:
            return None

        choices = list(weighted.items())
        spots = [pos for pos, _weight in choices]
        weights = [weight for _pos, weight in choices]
        cells.add(random.choices(spots, weights=weights, k=1)[0])

    grid = np.zeros((h, w), dtype=int)
    for r, c in cells:
        grid[r, c] = FG
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if g.ndim != 2 or not (MIN_H <= h <= MAX_H and MIN_W <= w <= MAX_W):
        return False
    if set(map(int, np.unique(g))) - {BG, FG}:
        return False

    components = _components(g == FG)
    if len(components) != 1:
        return False
    fg = components[0]
    if not (MIN_AREA <= len(fg) <= MAX_AREA):
        return False
    if not _touches_border(fg, g.shape):
        return False

    r0, c0, r1, c1 = _bbox(fg)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False

    if _enclosed_regions(g == BG):
        return False

    composed = _compose(g)
    regions = _enclosed_regions(composed == BG)
    if len(regions) < 2:
        return False

    region_colors = {_quadrant_color(*_top_left(region), h, w) for region in regions}
    if len(region_colors) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _grow_candidate()
        if grid is not None and _valid_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirror-emergent enclosed-region input")
