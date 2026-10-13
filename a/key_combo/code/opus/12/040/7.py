from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


BACKGROUND = 0
RED = 2
GREY = 5
MAGENTA = 6
BAR_COLORS = (1, 3, 4, 7, 8, 9)

GRID_HEIGHT_RANGE = (18, 22)
GRID_WIDTH_RANGE = (18, 22)
BAR_COUNT_CHOICES = (2, 3)
BAR_HEIGHT_RANGE = (2, 4)
TOTAL_OBSTACLE_RANGE = (2, 4)
MIN_DIAGONAL_LENGTH = 6
MAX_OVERLAPS = 4
GENERATION_ATTEMPTS = 800


@dataclass(frozen=True)
class Bar:
    color: int
    top: int
    height: int
    col: int

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _away_from_center(col: int, width: int) -> int:
    if 2 * col < width - 1:
        return -1
    if 2 * col > width - 1:
        return 1
    raise AssertionError("bar column cannot lie on the exact vertical centerline")


def _bar_ray_cells(bar: Bar, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    cells: list[tuple[int, int]] = []
    direction = _away_from_center(bar.col, width)
    for row in range(bar.top - 1, -1, -1):
        offset = (bar.top - row) // bar.height
        col = bar.col + direction * offset
        if not (0 <= col < width):
            break
        cells.append((row, col))
    return cells


def _natural_diagonal_path(height: int, width: int, source_col: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for step in range(1, max(height, width) + 1):
        row = height - 1 - step
        col = source_col + step
        if row < 0 or col >= width:
            break
        path.append((row, col))
    return path


def _extract_bars(grid: np.ndarray) -> list[Bar]:
    g = np.asarray(grid, dtype=int)
    colors = _colors_present(g, exclude={BACKGROUND, RED, GREY})
    assert 2 <= len(colors) <= 3, "expected 2 or 3 vertical bars"
    assert MAGENTA not in colors, "magenta is reserved for output overlaps"

    bars: list[Bar] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, "bar color must be present"
        cols = sorted(set(map(int, coords[:, 1])))
        assert len(cols) == 1, "each bar color must occupy one column"
        col = cols[0]
        rows = sorted(map(int, coords[:, 0]))
        assert rows == list(range(rows[0], rows[-1] + 1)), "bar pixels must be contiguous"
        height = rows[-1] - rows[0] + 1
        assert 2 <= height <= 4, "bar height must be between 2 and 4"
        bars.append(Bar(int(color), rows[0], height, col))

    cols = [bar.col for bar in bars]
    assert len(set(cols)) == len(cols), "bars must use distinct columns"
    return sorted(bars, key=lambda bar: (bar.col, bar.top, bar.color))


def _extract_source(grid: np.ndarray, bg: int) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    bottom_cols = np.flatnonzero(g[-1] == RED)
    assert bottom_cols.size == 1, "expected one red source on the bottom row"
    assert int(np.count_nonzero(g == RED)) == 1, "red source must be unique"
    return g.shape[0] - 1, int(bottom_cols[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend top bars into outward stair-rays and recolor their crossings with the obstacle-skipping bottom diagonal ray.

    Input structure:
    - The background is black.
    - There are 2 or 3 distinct single-column vertical bars of non-red, non-grey colors near the upper half of the grid.
    - A unique red source pixel sits on the bottom row, and grey single-pixel obstacles are scattered away from the bars.

    Rule:
    - For each bar, start from its top endpoint and trace upward one row at a time; every full bar-height rows, shift the ray one column away from the grid center.
    - From the red bottom source, trace an upward-right diagonal one row at a time; when the next cell is a grey obstacle, skip painting that row but continue the diagonal on the following row.
    - Preserve the original bars, source, and grey obstacles. Paint empty stair-ray cells with their bar color, empty diagonal cells red, and any empty cell reached by both ray types magenta.

    Key insight:
    - The bar height encodes the stair-ray period, while the diagonal ray keeps its geometric rhythm and only loses cells at obstacle hits, so the magenta cells can only be found by tracking both paths together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _get_bg(g)
    assert bg == BACKGROUND, "expected black background"

    bars = _extract_bars(g)
    source_row, source_col = _extract_source(g, bg)
    assert all(bar.col != source_col for bar in bars), "source cannot share a column with a bar"

    preserved = g != bg
    stair_claims: dict[tuple[int, int], int] = {}
    for bar in bars:
        for row, col in _bar_ray_cells(bar, g.shape):
            if preserved[row, col]:
                continue
            prev = stair_claims.get((row, col))
            assert prev in (None, bar.color), "different stair-rays may not overlap"
            stair_claims[(row, col)] = bar.color

    diagonal_claims: set[tuple[int, int]] = set()
    for row, col in _natural_diagonal_path(height, width, source_col):
        cell = int(g[row, col])
        if cell == GREY:
            continue
        assert cell == bg, "diagonal ray should only traverse background or grey obstacles"
        diagonal_claims.add((row, col))

    out = g.copy()
    for cell, color in stair_claims.items():
        if cell in diagonal_claims:
            out[cell] = MAGENTA
        else:
            out[cell] = color
    for cell in diagonal_claims:
        if cell not in stair_claims:
            out[cell] = RED
    return out


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _sample_bar_columns(width: int, count: int) -> list[int] | None:
    pool = [c for c in range(2, width - 2) if 2 * c != width - 1]
    left_pool = [c for c in pool if 2 * c < width - 1]
    right_pool = [c for c in pool if 2 * c > width - 1]
    if not left_pool or not right_pool:
        return None

    if count == 2:
        side_targets = (1, 1)
    else:
        side_targets = random.choice(((2, 1), (1, 2)))

    if len(left_pool) < side_targets[0] or len(right_pool) < side_targets[1]:
        return None

    cols = random.sample(left_pool, side_targets[0]) + random.sample(right_pool, side_targets[1])
    cols.sort()
    if any(abs(a - b) < 2 for a, b in zip(cols, cols[1:])):
        return None
    return cols


def _sample_bars(height: int, width: int) -> list[Bar] | None:
    count = random.choice(BAR_COUNT_CHOICES)
    cols = _sample_bar_columns(width, count)
    if cols is None:
        return None

    colors = random.sample(BAR_COLORS, count)
    bars: list[Bar] = []
    for color, col in zip(colors, cols):
        bar_height = random.randint(*BAR_HEIGHT_RANGE)
        top_min = max(4, 2 * bar_height)
        top_max = min(height // 2, height - bar_height - 6)
        if top_min > top_max:
            return None
        top = random.randint(top_min, top_max)
        bars.append(Bar(color, top, bar_height, col))

    bar_pixels = {
        (row, bar.col)
        for bar in bars
        for row in range(bar.top, bar.bottom + 1)
    }
    stair_seen: dict[tuple[int, int], int] = {}
    for bar in bars:
        ray = _bar_ray_cells(bar, (height, width))
        if len(ray) < 2:
            return None
        for cell in ray:
            if cell in bar_pixels:
                return None
            prev = stair_seen.get(cell)
            if prev is not None and prev != bar.color:
                return None
            stair_seen[cell] = bar.color
    return bars


def _safe_obstacle_cell(
    cell: tuple[int, int],
    bar_pixels: set[tuple[int, int]],
    source: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> bool:
    if cell in blocked:
        return False
    if any(_chebyshev(cell, other) <= 1 for other in bar_pixels):
        return False
    if _chebyshev(cell, source) <= 1:
        return False
    return True


def _render_from_specs(
    height: int,
    width: int,
    bars: list[Bar],
    source_col: int,
    obstacles: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, set[tuple[int, int]], set[tuple[int, int]], list[tuple[int, int]]]:
    inp = np.zeros((height, width), dtype=int)
    for bar in bars:
        inp[bar.top : bar.bottom + 1, bar.col] = bar.color
    inp[height - 1, source_col] = RED
    for row, col in obstacles:
        inp[row, col] = GREY

    out = inp.copy()
    stair_claims: dict[tuple[int, int], int] = {}
    for bar in bars:
        for cell in _bar_ray_cells(bar, (height, width)):
            if inp[cell] != BACKGROUND:
                continue
            stair_claims[cell] = bar.color

    path = _natural_diagonal_path(height, width, source_col)
    diagonal_claims = {cell for cell in path if cell not in obstacles}

    for cell, color in stair_claims.items():
        out[cell] = MAGENTA if cell in diagonal_claims else color
    for cell in diagonal_claims:
        if cell not in stair_claims:
            out[cell] = RED

    return inp, out, set(stair_claims), diagonal_claims, path


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bars = _sample_bars(height, width)
    if bars is None:
        return None

    bar_pixels = {
        (row, bar.col)
        for bar in bars
        for row in range(bar.top, bar.bottom + 1)
    }
    stair_union = {cell for bar in bars for cell in _bar_ray_cells(bar, (height, width))}

    source_candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]] = []
    for source_col in range(width - 1):
        if source_col in {bar.col for bar in bars}:
            continue
        path = _natural_diagonal_path(height, width, source_col)
        if len(path) < MIN_DIAGONAL_LENGTH:
            continue
        if any(cell in bar_pixels for cell in path):
            continue
        intersections = [cell for cell in path if cell in stair_union]
        if not intersections:
            continue
        source = (height - 1, source_col)
        obstacle_candidates = [
            cell
            for cell in path
            if cell not in stair_union and _safe_obstacle_cell(cell, bar_pixels, source, set())
        ]
        if not obstacle_candidates:
            continue
        source_candidates.append((source_col, path, intersections, obstacle_candidates))

    if not source_candidates:
        return None

    source_col, path, intersections, path_obstacle_candidates = random.choice(source_candidates)
    source = (height - 1, source_col)

    max_path_obstacles = min(2, len(path_obstacle_candidates))
    path_obstacle_count = random.randint(1, max_path_obstacles)
    random.shuffle(path_obstacle_candidates)
    path_obstacles: list[tuple[int, int]] = []
    for cell in path_obstacle_candidates:
        if any(_chebyshev(cell, other) <= 1 for other in path_obstacles):
            continue
        path_obstacles.append(cell)
        if len(path_obstacles) == path_obstacle_count:
            break
    if len(path_obstacles) != path_obstacle_count:
        return None

    total_obstacles = random.randint(max(2, len(path_obstacles)), TOTAL_OBSTACLE_RANGE[1])
    extra_needed = total_obstacles - len(path_obstacles)
    blocked = set(path_obstacles) | bar_pixels | stair_union | set(path)

    extra_pool = [
        (row, col)
        for row in range(height - 1)
        for col in range(width)
        if _safe_obstacle_cell((row, col), bar_pixels, source, blocked)
    ]
    random.shuffle(extra_pool)

    extra_obstacles: list[tuple[int, int]] = []
    for cell in extra_pool:
        if any(_chebyshev(cell, other) <= 1 for other in path_obstacles + extra_obstacles):
            continue
        extra_obstacles.append(cell)
        if len(extra_obstacles) == extra_needed:
            break
    if len(extra_obstacles) != extra_needed:
        return None

    obstacles = set(path_obstacles) | set(extra_obstacles)
    inp, expected, stair_claims, diagonal_claims, full_path = _render_from_specs(
        height, width, bars, source_col, obstacles
    )

    overlap = stair_claims & diagonal_claims
    stair_only = stair_claims - diagonal_claims
    diagonal_only = diagonal_claims - stair_claims
    encountered = [cell for cell in full_path if cell in obstacles]

    if not overlap or not stair_only or not diagonal_only:
        return None
    if len(overlap) > MAX_OVERLAPS:
        return None
    if len(encountered) < 1:
        return None
    if any(cell in overlap for cell in obstacles):
        return None
    if any(not _safe_obstacle_cell(cell, bar_pixels, source, obstacles - {cell}) for cell in obstacles):
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    if int(np.count_nonzero(inp == GREY)) != len(obstacles):
        return None
    if int(np.count_nonzero(expected == MAGENTA)) != len(overlap):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stair-ray / obstacle-deflection input")
