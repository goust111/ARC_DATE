from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GREY = 5


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _marker_columns(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    unique = set(map(int, np.unique(g)))
    assert unique <= {GREY, GREEN}, "input must contain only gray background and green markers"

    coords = np.argwhere(g == GREEN)
    assert coords.size > 0, "expected at least one green marker"
    assert np.all((coords[:, 0] > 0) & (coords[:, 0] < g.shape[0] - 1)), "markers must avoid the top and bottom rows"

    cols = [int(c) for c in coords[:, 1]]
    assert len(set(cols)) == len(cols), "markers must occupy distinct columns"
    return sorted(cols)


def _draw_paths(shape: tuple[int, int], columns: list[int]) -> np.ndarray:
    height, width = map(int, shape)
    out = np.full((height, width), GREY, dtype=int)
    middle = width // 2
    has_unique_middle = (width % 2) == 1

    for col in columns:
        out[:, int(col)] = BLUE
        if has_unique_middle and int(col) == middle:
            out[0, int(col) : width] = BLUE
            out[height - 1, 0 : int(col) + 1] = BLUE
        else:
            out[0, 0 : int(col) + 1] = BLUE
            out[height - 1, int(col) : width] = BLUE
    return out


def _solve_from_columns(shape: tuple[int, int], columns: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    path_grid = _draw_paths(shape, columns)
    out = path_grid.copy()
    height, width = path_grid.shape
    seen = np.zeros((height, width), dtype=bool)
    enclosed_areas: list[int] = []
    boundary_areas: list[int] = []

    for row in range(height):
        for col in range(width):
            if int(path_grid[row, col]) != GREY or seen[row, col]:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            component: list[tuple[int, int]] = []
            touches_boundary = False

            while stack:
                rr, cc = stack.pop()
                component.append((rr, cc))
                if rr in (0, height - 1) or cc in (0, width - 1):
                    touches_boundary = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(path_grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            area = len(component)
            if touches_boundary:
                boundary_areas.append(area)
                continue

            enclosed_areas.append(area)
            for rr, cc in component:
                out[rr, cc] = RED

    return out, enclosed_areas, boundary_areas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every gray region enclosed by interwoven folded blue paths determined by green marker columns.

    Input structure:
    - A gray background contains green single-pixel markers in distinct columns.
    - The input contains no blue or red pixels, and markers do not sit on the top or bottom row.
    - Marker rows are irrelevant; only their columns control the folded paths.

    Rule:
    - For each green marker column, paint that full column blue from top to bottom.
    - Extend the top endpoint horizontally to the left edge and the bottom endpoint horizontally to the right edge, except for the unique middle column of an odd-width grid, which instead extends top-right and bottom-left.
    - After all blue paths are drawn, find every 4-connected gray component that touches no grid edge and recolor it red.
    - Leave blue path pixels and every non-enclosed gray pixel unchanged.

    Key insight:
    - No single folded path encloses anything; red regions emerge only from how several full-height columns interact with the top and bottom border segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    columns = _marker_columns(g)
    out, _enclosed_areas, _boundary_areas = _solve_from_columns(g.shape, columns)
    return out


GEN_HEIGHTS = tuple(range(12, 19))
GEN_EVEN_WIDTHS = (12, 14, 16, 18)
GEN_ODD_WIDTHS = (13, 15, 17, 19)
GEN_MARKER_COUNT_RANGE = (3, 5)
GEN_MIN_VISIBLE_ENCLOSED_AREA = 8
GEN_MAX_ATTEMPTS = 400
GEN_MODES = ["even", "odd_no_center", "odd_center"]
_GEN_MODE_BAG: list[str] = []


def _next_mode() -> str:
    global _GEN_MODE_BAG
    if not _GEN_MODE_BAG:
        _GEN_MODE_BAG = GEN_MODES.copy()
        random.shuffle(_GEN_MODE_BAG)
    return _GEN_MODE_BAG.pop()


def _sample_non_center_columns(width: int, count: int, *, forbid_center: bool) -> list[int] | None:
    middle = width // 2
    available = [col for col in range(1, width - 1) if not (forbid_center and col == middle)]
    if len(available) < count:
        return None

    for _ in range(200):
        cols = sorted(random.sample(available, count))
        if any(right - left >= 3 for left, right in zip(cols, cols[1:])):
            return cols
    return None


def _sample_center_columns(width: int, count: int) -> list[int] | None:
    middle = width // 2
    left_choices = list(range(1, middle - 1))
    right_choices = list(range(middle + 2, width - 1))
    side_choices = [choices for choices in (left_choices, right_choices) if len(choices) >= 2]
    if not side_choices:
        return None

    for _ in range(200):
        pair_pool = random.choice(side_choices)
        pair = sorted(random.sample(pair_pool, 2))
        if pair[1] - pair[0] < 3:
            continue

        cols = {middle, pair[0], pair[1]}
        extra_pool = [col for col in range(1, width - 1) if col not in cols and abs(col - middle) >= 2]
        random.shuffle(extra_pool)
        while len(cols) < count and extra_pool:
            cols.add(extra_pool.pop())
        if len(cols) == count:
            return sorted(cols)
    return None


def _validate_generated_case(grid: np.ndarray, mode: str) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False

    coords = np.argwhere(g == GREEN)
    if not (GEN_MARKER_COUNT_RANGE[0] <= coords.shape[0] <= GEN_MARKER_COUNT_RANGE[1]):
        return False
    if set(map(int, np.unique(g))) != {GREY, GREEN}:
        return False
    if len({int(r) for r, _c in coords}) != 1:
        return False
    if int(coords[0, 0]) != g.shape[0] // 2:
        return False

    cols = sorted(int(c) for _r, c in coords)
    middle = g.shape[1] // 2
    if mode == "even":
        if (g.shape[1] % 2) != 0:
            return False
    elif mode == "odd_no_center":
        if (g.shape[1] % 2) != 1 or middle in cols:
            return False
    else:
        if (g.shape[1] % 2) != 1 or middle not in cols:
            return False
        if any(abs(col - middle) < 2 and col != middle for col in cols):
            return False

    expected, enclosed_areas, boundary_areas = _solve_from_columns(g.shape, cols)
    if not enclosed_areas or max(enclosed_areas) < GEN_MIN_VISIBLE_ENCLOSED_AREA:
        return False
    if not boundary_areas:
        return False
    if not (np.any(expected == BLUE) and np.any(expected == RED) and np.any(expected == GREY)):
        return False
    if not np.array_equal(main(g), expected):
        return False

    for row, col in coords:
        single = np.full_like(g, GREY)
        single[int(row), int(col)] = GREEN
        if np.any(main(single) == RED):
            return False

        reduced = g.copy()
        reduced[int(row), int(col)] = GREY
        if np.array_equal(main(reduced), expected):
            return False

    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    height = random.choice(GEN_HEIGHTS)
    width = random.choice(GEN_EVEN_WIDTHS if mode == "even" else GEN_ODD_WIDTHS)
    marker_count = random.randint(*GEN_MARKER_COUNT_RANGE)

    if mode == "odd_center":
        cols = _sample_center_columns(width, marker_count)
    else:
        cols = _sample_non_center_columns(width, marker_count, forbid_center=(mode == "odd_no_center"))
    if cols is None:
        return None

    grid = np.full((height, width), GREY, dtype=int)
    marker_row = height // 2
    for col in cols:
        grid[marker_row, int(col)] = GREEN

    if not _validate_generated_case(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    mode = _next_mode()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle-9 input for mode {mode}")
