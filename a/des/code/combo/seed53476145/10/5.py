from __future__ import annotations

from collections import Counter
from fractions import Fraction
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _find_separator_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    height, width = map(int, grid.shape)

    row_candidates = []
    for r in range(height):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r, :] == color):
            row_candidates.append((r, color))

    col_candidates = []
    for c in range(width):
        color = int(grid[0, c])
        if color != bg and np.all(grid[:, c] == color):
            col_candidates.append((c, color))

    assert len(row_candidates) == 1, "expected exactly one solid non-background separator row"
    assert len(col_candidates) == 1, "expected exactly one solid non-background separator column"

    cross_row, row_color = row_candidates[0]
    cross_col, col_color = col_candidates[0]
    assert row_color == col_color, "separator row and column must share one color"
    assert 0 < cross_row < height - 1 and 0 < cross_col < width - 1, "cross must leave four nonempty quadrants"
    return int(cross_row), int(cross_col), int(row_color)


def _foreground_colors_row_major(grid: np.ndarray, exclude: set[int]) -> list[int]:
    seen: list[int] = []
    used: set[int] = set()
    for value in map(int, grid.reshape(-1)):
        if value in exclude or value in used:
            continue
        used.add(value)
        seen.append(value)
    return seen


def _group_stats(coords: np.ndarray) -> tuple[int, Fraction]:
    count = int(len(coords))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    return area, Fraction(count, area)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge same-color pixels across quadrants, rank colors by merged bounding-box density,
    and emit the colors as a single top-to-bottom column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    cross_row, cross_col, sep = _find_separator_cross(g, bg)
    assert np.all(g[cross_row, :] == sep)
    assert np.all(g[:, cross_col] == sep)

    colors = _foreground_colors_row_major(g, exclude={bg, sep})
    assert colors, "expected at least one non-background, non-separator color"

    ranked = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) > 0
        area, density = _group_stats(coords)
        ranked.append(
            {
                "color": int(color),
                "count": int(len(coords)),
                "area": area,
                "density": density,
            }
        )

    ranked.sort(key=lambda item: item["density"], reverse=True)
    return np.array([[item["color"]] for item in ranked], dtype=int)


# --------------- generator ---------------

BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

HEIGHT_RANGE = (15, 22)
WIDTH_RANGE = (15, 22)
CROSS_MARGIN = 4
COLOR_COUNT_RANGE = (5, 6)
MAX_TOTAL_PER_COLOR = 5
MAX_LOCAL_BOX_SIDE = 5
LOCAL_PLACEMENT_ATTEMPTS = 60
MAX_GENERATION_ATTEMPTS = 500


def _quadrant_bounds(cross_row: int, cross_col: int, height: int, width: int) -> list[tuple[int, int, int, int]]:
    return [
        (0, cross_row, 0, cross_col),
        (0, cross_row, cross_col + 1, width),
        (cross_row + 1, height, 0, cross_col),
        (cross_row + 1, height, cross_col + 1, width),
    ]


def _cell_quadrant(row: int, col: int, cross_row: int, cross_col: int) -> int:
    if row < cross_row:
        return 0 if col < cross_col else 1
    return 2 if col < cross_col else 3


def _is_clear(row: int, col: int, blocked: set[tuple[int, int]]) -> bool:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (row + dr, col + dc) in blocked:
                return False
    return True


def _sample_quad_counts(n_colors: int) -> list[int]:
    quad_counts = [1, 2, 3, 4]
    while len(quad_counts) < n_colors:
        quad_counts.append(random.randint(1, 4))
    random.shuffle(quad_counts)
    return quad_counts


def _sample_cells_per_quadrant(quadrant_count: int) -> list[int]:
    min_total = max(2, quadrant_count)
    max_total = min(MAX_TOTAL_PER_COLOR, quadrant_count + 2)
    total = random.randint(min_total, max_total)

    counts = [1] * quadrant_count
    for _ in range(total - quadrant_count):
        counts[random.randrange(quadrant_count)] += 1
    random.shuffle(counts)
    return counts


def _box_options(height: int, width: int, need: int) -> list[tuple[int, int]]:
    options = []
    max_h = min(height, MAX_LOCAL_BOX_SIDE)
    max_w = min(width, MAX_LOCAL_BOX_SIDE)
    max_area = max(need, min(height * width, need + 6))
    for box_h in range(1, max_h + 1):
        for box_w in range(1, max_w + 1):
            area = box_h * box_w
            if need <= area <= max_area:
                options.append((box_h, box_w))
    return options


def _place_group(
    quadrants: list[tuple[int, int, int, int]],
    quad_ids: list[int],
    counts_per_quad: list[int],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    placed: list[tuple[int, int]] = []

    for quad_id, count in zip(quad_ids, counts_per_quad):
        r0, r1, c0, c1 = quadrants[quad_id]
        box_choices = _box_options(r1 - r0, c1 - c0, count)
        if not box_choices:
            return None

        success = False
        for _ in range(LOCAL_PLACEMENT_ATTEMPTS):
            box_h, box_w = random.choice(box_choices)
            top = random.randint(r0, r1 - box_h)
            left = random.randint(c0, c1 - box_w)
            box_cells = [(rr, cc) for rr in range(top, top + box_h) for cc in range(left, left + box_w)]
            blocked = occupied | set(placed)
            clear_cells = [(rr, cc) for rr, cc in box_cells if _is_clear(rr, cc, blocked)]
            if len(clear_cells) < count:
                continue
            placed.extend(random.sample(clear_cells, count))
            success = True
            break

        if not success:
            return None

    occupied.update(placed)
    return placed


def _build_grid(
    height: int,
    width: int,
    cross_row: int,
    cross_col: int,
    groups: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[cross_row, :] = SEPARATOR
    grid[:, cross_col] = SEPARATOR
    for group in groups:
        color = int(group["color"])
        for row, col in group["cells"]:
            grid[row, col] = color
    return grid


def _stats_from_cells(
    color: int,
    cells: list[tuple[int, int]],
    cross_row: int,
    cross_col: int,
) -> dict[str, object]:
    coords = np.array(cells, dtype=int)
    area, density = _group_stats(coords)
    quadrants = {
        _cell_quadrant(int(row), int(col), cross_row, cross_col)
        for row, col in cells
    }
    return {
        "color": int(color),
        "cells": list(cells),
        "count": int(len(cells)),
        "area": area,
        "density": density,
        "quadrant_count": int(len(quadrants)),
    }


def _quadrants_nonempty(grid: np.ndarray, cross_row: int, cross_col: int) -> bool:
    for r0, r1, c0, c1 in _quadrant_bounds(cross_row, cross_col, *grid.shape):
        region = grid[r0:r1, c0:c1]
        if not np.any((region != BG) & (region != SEPARATOR)):
            return False
    return True


def _has_duplicate_with_different_density(groups: list[dict[str, object]], key: str) -> bool:
    seen: dict[int, Fraction] = {}
    for group in groups:
        value = int(group[key])
        density = group["density"]
        if value in seen and seen[value] != density:
            return True
        seen[value] = density
    return False


def _expected_output(groups: list[dict[str, object]]) -> np.ndarray:
    ranked = sorted(groups, key=lambda item: item["density"], reverse=True)
    return np.array([[int(group["color"])] for group in ranked], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    cross_row = random.randint(CROSS_MARGIN, height - CROSS_MARGIN - 1)
    cross_col = random.randint(CROSS_MARGIN, width - CROSS_MARGIN - 1)
    quadrants = _quadrant_bounds(cross_row, cross_col, height, width)

    n_colors = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, n_colors)
    quad_counts = _sample_quad_counts(n_colors)

    occupied: set[tuple[int, int]] = set()
    groups = []
    for color, quadrant_count in zip(colors, quad_counts):
        quad_ids = random.sample(range(4), quadrant_count)
        counts_per_quad = _sample_cells_per_quadrant(quadrant_count)
        cells = _place_group(quadrants, quad_ids, counts_per_quad, occupied)
        if cells is None:
            return None
        groups.append(_stats_from_cells(color, cells, cross_row, cross_col))

    densities = [group["density"] for group in groups]
    if len(set(densities)) != len(densities):
        return None
    if not _has_duplicate_with_different_density(groups, "count"):
        return None
    if not _has_duplicate_with_different_density(groups, "quadrant_count"):
        return None

    grid = _build_grid(height, width, cross_row, cross_col, groups)
    if not _quadrants_nonempty(grid, cross_row, cross_col):
        return None

    expected = _expected_output(groups)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
