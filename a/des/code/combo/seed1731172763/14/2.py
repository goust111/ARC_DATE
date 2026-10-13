from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 200
LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2))
CELL_SIZE_OPTIONS = (5, 7)
MARGIN_OPTIONS = (1, 2)

SOLID = 0
CROSS = 1
FRAME = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    regions = []
    for comp in _components(grid != WALL):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _markers_in_region(grid: np.ndarray, region: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    markers = [
        (r, c, int(grid[r, c]))
        for r, c in region
        if int(grid[r, c]) not in (BG, WALL)
    ]
    markers.sort()
    return markers


def _cross_pixels(region: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in region]
    cols = [c for _, c in region]
    center_r = (min(rows) + max(rows)) // 2
    center_c = (min(cols) + max(cols)) // 2
    return [(r, c) for r, c in region if r == center_r or c == center_c]


def _frame_pixels(grid: np.ndarray, region: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    pixels = []
    for r, c in region:
        if any(
            0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == WALL
            for nr, nc in _neighbors4(r, c)
        ):
            pixels.append((r, c))
    return pixels


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each enclosed non-wall region, count the colored markers inside it.
    Small counts trigger solid fill, medium counts trigger a centered cross,
    and large counts trigger an inner frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for region in _enclosed_regions(g):
        markers = _markers_in_region(g, region)
        if not markers:
            continue

        fill_color = markers[0][2]
        marker_count = len(markers)

        for r, c in region:
            out[r, c] = BG

        if marker_count <= 2:
            paint = region
        elif marker_count <= 4:
            paint = _cross_pixels(region)
        else:
            paint = _frame_pixels(g, region)

        for r, c in paint:
            out[r, c] = fill_color

    return out


def _branch_from_count(count: int) -> int:
    if count <= 2:
        return SOLID
    if count <= 4:
        return CROSS
    return FRAME


def _sample_layout(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    rows, cols = rng.choice(LAYOUT_OPTIONS)
    cell_heights = [rng.choice(CELL_SIZE_OPTIONS) for _ in range(rows)]
    cell_widths = [rng.choice(CELL_SIZE_OPTIONS) for _ in range(cols)]

    top = rng.choice(MARGIN_OPTIONS)
    bottom = rng.choice(MARGIN_OPTIONS)
    left = rng.choice(MARGIN_OPTIONS)
    right = rng.choice(MARGIN_OPTIONS)

    frame_h = sum(cell_heights) + rows + 1
    frame_w = sum(cell_widths) + cols + 1
    grid = np.full((top + frame_h + bottom, left + frame_w + right), BG, dtype=int)

    r = top
    grid[r, left : left + frame_w] = WALL
    for cell_h in cell_heights:
        r += cell_h + 1
        grid[r, left : left + frame_w] = WALL

    c = left
    grid[top : top + frame_h, c] = WALL
    for cell_w in cell_widths:
        c += cell_w + 1
        grid[top : top + frame_h, c] = WALL

    cells: list[tuple[int, int, int, int]] = []
    r = top + 1
    for cell_h in cell_heights:
        c = left + 1
        for cell_w in cell_widths:
            cells.append((r, c, r + cell_h - 1, c + cell_w - 1))
            c += cell_w + 1
        r += cell_h + 1

    return grid, cells


def _balanced_branches(n_regions: int, rng: random.Random) -> list[int]:
    base = [SOLID, CROSS, FRAME] * (n_regions // 3)
    extras = [SOLID, CROSS, FRAME]
    rng.shuffle(extras)
    base.extend(extras[: n_regions % 3])
    rng.shuffle(base)
    return base


def _marker_count_for_branch(branch: int, rng: random.Random) -> int:
    if branch == SOLID:
        return rng.randint(1, 2)
    if branch == CROSS:
        return rng.randint(3, 4)
    return rng.randint(5, 6)


def _spread_out_sample(
    coords: list[tuple[int, int]], count: int, rng: random.Random
) -> list[tuple[int, int]]:
    for _ in range(80):
        sample = rng.sample(coords, count)
        ok = True
        for i, (r1, c1) in enumerate(sample):
            for r2, c2 in sample[i + 1 :]:
                if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            sample.sort()
            return sample
    sample = rng.sample(coords, count)
    sample.sort()
    return sample


def _fill_cell(
    grid: np.ndarray,
    cell: tuple[int, int, int, int],
    branch: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = cell
    coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    count = _marker_count_for_branch(branch, rng)
    points = _spread_out_sample(coords, count, rng)

    colors = list(rng.sample(MARKER_COLORS, count))
    rng.shuffle(colors)
    for (r, c), color in zip(points, colors):
        grid[r, c] = color


def _valid_input(grid: np.ndarray, expected_regions: int) -> bool:
    regions = _enclosed_regions(grid)
    if len(regions) != expected_regions:
        return False

    branches = set()
    for region in regions:
        markers = _markers_in_region(grid, region)
        if not (1 <= len(markers) <= 6):
            return False
        if len({color for _, _, color in markers}) != len(markers):
            return False

        rows = [r for r, _ in region]
        cols = [c for _, c in region]
        if (max(rows) - min(rows) + 1) % 2 == 0:
            return False
        if (max(cols) - min(cols) + 1) % 2 == 0:
            return False

        branches.add(_branch_from_count(len(markers)))

    return branches == {SOLID, CROSS, FRAME}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid, cells = _sample_layout(rng)
    branches = _balanced_branches(len(cells), rng)
    for cell, branch in zip(cells, branches):
        _fill_cell(grid, cell, branch, rng)

    if not _valid_input(grid, len(cells)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
