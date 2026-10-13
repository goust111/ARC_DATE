from __future__ import annotations

import random

import numpy as np


BG_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator bounds.
MIN_GROUPS = 3
MAX_GROUPS = len(PALETTE)
INNER_SIZE_EXTRA = (2, 5)
OUTER_MARGIN_RANGE = (0, 4)
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ordered_groups(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    coords = np.argwhere(grid != bg)
    assert len(coords) > 0, "expected at least one marker"

    groups: dict[int, dict[str, int]] = {}
    color_to_sum: dict[int, int] = {}
    for r, c in coords:
        color = int(grid[r, c])
        diag_sum = int(r + c)
        group = groups.setdefault(diag_sum, {"color": color, "count": 0})
        assert group["color"] == color, "each anti-diagonal must use exactly one color"

        prev_sum = color_to_sum.setdefault(color, diag_sum)
        assert prev_sum == diag_sum, "different anti-diagonals must use different colors"
        group["count"] += 1

    return [groups[diag_sum] | {"sum": diag_sum} for diag_sum in sorted(groups)]


def _render_groups(groups: list[dict[str, int]], bg: int) -> np.ndarray:
    size = len(groups)
    out = np.full((size, size), bg, dtype=int)

    for row, group in enumerate(groups):
        count = int(group["count"])
        color = int(group["color"])
        assert 1 <= count <= row + 1, "group count must fit the square output"
        steps = np.arange(count, dtype=int)
        out[row - steps, steps] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered markers by anti-diagonal (row + col), count each group,
    then redraw the groups as left-anchored diagonal segments in anti-diagonal order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    groups = _ordered_groups(g, bg)
    return _render_groups(groups, bg)


def _triangle_cells(diag_sum: int) -> list[tuple[int, int]]:
    return [(r, diag_sum - r) for r in range(diag_sum + 1)]


def _sample_specs() -> tuple[int, list[int], list[int]]:
    n_groups = random.randint(MIN_GROUPS, MAX_GROUPS)
    inner_size = random.randint(n_groups + INNER_SIZE_EXTRA[0], n_groups + INNER_SIZE_EXTRA[1])
    diag_sums = sorted(random.sample(range(inner_size), n_groups))
    counts = [random.randint(1, index + 1) for index in range(n_groups)]
    return inner_size, diag_sums, counts


def _build_input(inner_size: int, diag_sums: list[int], counts: list[int]) -> np.ndarray:
    height = inner_size + random.randint(*OUTER_MARGIN_RANGE)
    width = inner_size + random.randint(*OUTER_MARGIN_RANGE)
    row_offset = random.randint(0, height - inner_size)
    col_offset = random.randint(0, width - inner_size)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    colors = random.sample(PALETTE, len(diag_sums))

    for diag_sum, count, color in zip(diag_sums, counts, colors):
        picks = random.sample(_triangle_cells(diag_sum), count)
        for r, c in picks:
            grid[row_offset + r, col_offset + c] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    inner_size, diag_sums, counts = _sample_specs()
    grid = _build_input(inner_size, diag_sums, counts)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (len(diag_sums), len(diag_sums)):
        return None
    if _background_color(grid) != BG_COLOR:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
