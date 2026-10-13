from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
GEN_RETRIES = 500


def _max_true_run(values: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in values:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _bottom_palette(g: np.ndarray) -> list[int]:
    bottom = g[-1]
    mask = (bottom != BG) & (bottom != WALL)
    runs: list[tuple[int, int]] = []
    c = 0
    while c < bottom.size:
        if not bool(mask[c]):
            c += 1
            continue
        start = c
        while c < bottom.size and bool(mask[c]):
            c += 1
        runs.append((start, c))

    candidates = []
    for start, stop in runs:
        if stop - start == 5:
            colors = [int(x) for x in bottom[start:stop]]
            if len(set(colors)) == 5:
                candidates.append(colors)

    if not candidates:
        for start in range(0, bottom.size - 4):
            colors = [int(x) for x in bottom[start : start + 5]]
            if all(c not in (BG, WALL) for c in colors) and len(set(colors)) == 5:
                candidates.append(colors)

    assert len(candidates) == 1, "expected one contiguous five-color bottom palette"
    return candidates[0]


def _wall_lines(area: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = area.shape
    horizontal = [r for r in range(h) if _max_true_run(area[r] == WALL) >= 2]
    vertical = [c for c in range(w) if _max_true_run(area[:, c] == WALL) >= 2]
    assert len(horizontal) >= 2 and len(vertical) >= 2, "expected a rectangular wall network"
    return horizontal, vertical


def _boundary_rows(area: np.ndarray, horizontal: list[int], vertical: list[int]) -> list[int]:
    h = area.shape[0]
    bounds = set(horizontal)
    if vertical and any(int(area[0, c]) == WALL for c in vertical):
        bounds.add(-1)
    if vertical and any(int(area[h - 1, c]) == WALL for c in vertical):
        bounds.add(h)
    return sorted(bounds)


def _boundary_cols(area: np.ndarray, horizontal: list[int], vertical: list[int]) -> list[int]:
    w = area.shape[1]
    bounds = set(vertical)
    if horizontal and any(int(area[r, 0]) == WALL for r in horizontal):
        bounds.add(-1)
    if horizontal and any(int(area[r, w - 1]) == WALL for r in horizontal):
        bounds.add(w)
    return sorted(bounds)


def _horizontal_doors(area: np.ndarray, r: int, c0: int, c1: int) -> int | None:
    h, w = area.shape
    if not (0 <= r < h):
        return None
    count = 0
    for c in range(c0, c1 + 1):
        v = int(area[r, c])
        if v == WALL:
            continue
        if (
            v == BG
            and 0 < c < w - 1
            and int(area[r, c - 1]) == WALL
            and int(area[r, c + 1]) == WALL
        ):
            count += 1
            continue
        return None
    return count


def _vertical_doors(area: np.ndarray, c: int, r0: int, r1: int) -> int | None:
    h, w = area.shape
    if not (0 <= c < w):
        return None
    count = 0
    for r in range(r0, r1 + 1):
        v = int(area[r, c])
        if v == WALL:
            continue
        if (
            v == BG
            and 0 < r < h - 1
            and int(area[r - 1, c]) == WALL
            and int(area[r + 1, c]) == WALL
        ):
            count += 1
            continue
        return None
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room with the bottom legend color indexed by its number of wall doors.

    Input structure:
    - The background is black.
    - Gray one-pixel orthogonal wall segments form a rectangular room network; room interiors are background.
    - A door is a single background-colored missing pixel in a gray wall segment along a room boundary.
    - The bottom row contains one contiguous five-cell color strip; its left-to-right colors encode door counts 0 through 4.

    Rule:
    - Read the bottom strip as a count-to-color palette.
    - Identify the gray horizontal and vertical wall lines and use adjacent boundary lines to enumerate rectangular rooms.
    - Count the one-pixel gray-wall gaps on each room boundary; grid-edge boundaries contribute zero doors.
    - Fill each room's interior background cells with palette[door_count], while preserving gray walls, door gaps, and the bottom strip.

    Key insight:
    - Door gaps split the visual wall but not the room lattice; the number of such gaps is the index into the bottom color legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 4 and g.shape[1] >= 5, "grid is too small"

    palette = _bottom_palette(g)
    area = g[:-1]
    horizontal, vertical = _wall_lines(area)
    row_bounds = _boundary_rows(area, horizontal, vertical)
    col_bounds = _boundary_cols(area, horizontal, vertical)

    out = g.copy()
    area_h, area_w = area.shape
    for bi in range(len(row_bounds) - 1):
        top = int(row_bounds[bi])
        bottom = int(row_bounds[bi + 1])
        r0, r1 = top + 1, bottom - 1
        if r0 > r1:
            continue

        for bj in range(len(col_bounds) - 1):
            left = int(col_bounds[bj])
            right = int(col_bounds[bj + 1])
            c0, c1 = left + 1, right - 1
            if c0 > c1:
                continue

            interior = area[r0 : r1 + 1, c0 : c1 + 1]
            if interior.size == 0 or np.any(interior != BG):
                continue

            counts: list[int] = []
            if top != -1:
                n = _horizontal_doors(area, top, c0, c1)
                if n is None:
                    continue
                counts.append(n)
            if bottom != area_h:
                n = _horizontal_doors(area, bottom, c0, c1)
                if n is None:
                    continue
                counts.append(n)
            if left != -1:
                n = _vertical_doors(area, left, r0, r1)
                if n is None:
                    continue
                counts.append(n)
            if right != area_w:
                n = _vertical_doors(area, right, r0, r1)
                if n is None:
                    continue
                counts.append(n)

            door_count = sum(counts)
            if 0 <= door_count < len(palette):
                out[r0 : r1 + 1, c0 : c1 + 1] = palette[door_count]

    return out


def _wall_coordinates(margin: int, sizes: list[int]) -> list[int]:
    coords = [int(margin)]
    cur = int(margin)
    for size in sizes:
        cur += int(size) + 1
        coords.append(cur)
    return coords


def _render_expected(
    grid: np.ndarray,
    palette: list[int],
    row_walls: list[int],
    col_walls: list[int],
    counts: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    for r in range(counts.shape[0]):
        for c in range(counts.shape[1]):
            rr0, rr1 = row_walls[r] + 1, row_walls[r + 1]
            cc0, cc1 = col_walls[c] + 1, col_walls[c + 1]
            out[rr0:rr1, cc0:cc1] = palette[int(counts[r, c])]
    return out


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.choice([2, 3, 3, 4])
    room_cols = rng.choice([2, 3, 3, 4])
    heights = [rng.randint(3, 6) for _ in range(room_rows)]
    widths = [rng.randint(3, 6) for _ in range(room_cols)]

    top_margin = rng.randint(0, 2)
    left_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 1)

    row_walls = _wall_coordinates(top_margin, heights)
    col_walls = _wall_coordinates(left_margin, widths)
    room_h = row_walls[-1] - row_walls[0] + 1
    room_w = col_walls[-1] - col_walls[0] + 1
    grid_h = top_margin + room_h + bottom_margin + 1
    grid_w = max(left_margin + room_w + right_margin, 5)
    if grid_h > 30 or grid_w > 30:
        return None

    palette = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], 5)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    top, bottom = row_walls[0], row_walls[-1]
    left, right = col_walls[0], col_walls[-1]

    for r in row_walls:
        grid[r, left : right + 1] = WALL
    for c in col_walls:
        grid[top : bottom + 1, c] = WALL

    counts = np.zeros((room_rows, room_cols), dtype=int)
    door_prob = rng.choice([0.30, 0.45, 0.60, 0.75])

    for r in range(1, room_rows):
        wall_r = row_walls[r]
        for c in range(room_cols):
            if rng.random() >= door_prob:
                continue
            lo = col_walls[c] + 2
            hi = col_walls[c + 1] - 2
            if lo > hi:
                continue
            door_c = rng.randint(lo, hi)
            grid[wall_r, door_c] = BG
            counts[r - 1, c] += 1
            counts[r, c] += 1

    for c in range(1, room_cols):
        wall_c = col_walls[c]
        for r in range(room_rows):
            if rng.random() >= door_prob:
                continue
            lo = row_walls[r] + 2
            hi = row_walls[r + 1] - 2
            if lo > hi:
                continue
            door_r = rng.randint(lo, hi)
            grid[door_r, wall_c] = BG
            counts[r, c - 1] += 1
            counts[r, c] += 1

    flat_counts = [int(x) for x in counts.ravel()]
    if max(flat_counts) > 4 or len(set(flat_counts)) < 2 or sum(flat_counts) == 0:
        return None

    start = rng.randint(0, grid_w - 5)
    grid[-1, start : start + 5] = palette

    expected = _render_expected(grid, palette, row_walls, col_walls, counts)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        result = _make_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid input")
