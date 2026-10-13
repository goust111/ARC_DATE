from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_RETRY_LIMIT = 300
REPEAT_COUNT = 2
LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2), (3, 3))
START_RANGE = (2, 4)
GAP_RANGE = (4, 5)
END_MARGIN_RANGE = (2, 4)

# Visible color order from the prompt:
# blue, red, green, yellow, grey, magenta, orange, azure, maroon.
COLOR_ORDER = (1, 2, 3, 4, 5, 6, 7, 8, 9)
COLOR_PRIORITY = {color: idx for idx, color in enumerate(COLOR_ORDER)}

SHAPES = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),  # plus
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),  # x
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),  # frame
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),  # full
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),  # vertical
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),  # horizontal
    np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),  # corners
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),  # main diagonal
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),  # anti diagonal
)
FULL_SHAPE_INDEX = 3


def _earlier_color(color_a: int, color_b: int) -> int:
    if COLOR_PRIORITY[int(color_a)] <= COLOR_PRIORITY[int(color_b)]:
        return int(color_a)
    return int(color_b)


def _extract_markers(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = grid.shape
    row_markers = [(r, int(grid[r, 0])) for r in range(1, height) if int(grid[r, 0]) != BG]
    col_markers = [(c, int(grid[0, c])) for c in range(1, width) if int(grid[0, c]) != BG]
    return row_markers, col_markers


def _draw_shape(grid: np.ndarray, center_r: int, center_c: int, mask: np.ndarray, color: int) -> None:
    patch = grid[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read top-row and left-column color markers, recover the observed 3x3 shape
    for each visible (row-color, col-color) pair, and stamp that shape into the
    blank intersections of the same color pair using the earlier of the two
    colors in the visible palette order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_markers, col_markers = _extract_markers(g)
    assert row_markers and col_markers, "expected both row and column markers"

    pair_to_shape: dict[tuple[int, int], np.ndarray] = {}
    for r, row_color in row_markers:
        for c, col_color in col_markers:
            block = g[r - 1 : r + 2, c - 1 : c + 2]
            assert block.shape == (3, 3), "intersection patterns must fit inside the grid"
            mask = block != BG
            if not mask.any():
                continue
            key = (int(row_color), int(col_color))
            if key in pair_to_shape:
                assert np.array_equal(pair_to_shape[key], mask), "same color pair must reuse one shape"
            else:
                pair_to_shape[key] = mask.copy()

    out = g.copy()
    for r, row_color in row_markers:
        for c, col_color in col_markers:
            block = g[r - 1 : r + 2, c - 1 : c + 2]
            if np.any(block != BG):
                continue
            mask = pair_to_shape.get((int(row_color), int(col_color)))
            if mask is None:
                continue
            _draw_shape(out, r, c, mask, _earlier_color(int(row_color), int(col_color)))

    return out


def _sample_centers(count: int, rng: random.Random) -> tuple[list[int], int]:
    centers = [rng.randint(*START_RANGE)]
    for _ in range(count - 1):
        centers.append(centers[-1] + rng.randint(*GAP_RANGE))
    extent = centers[-1] + rng.randint(*END_MARGIN_RANGE) + 1
    return centers, extent


def _choose_color_sets(
    row_kind_count: int,
    col_kind_count: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int]]:
    shared_count = 1 if rng.random() < 0.5 else 0
    shared = rng.sample(list(COLOR_ORDER), shared_count)

    remaining = [color for color in COLOR_ORDER if color not in shared]
    row_only = rng.sample(remaining, row_kind_count - shared_count)
    remaining = [color for color in remaining if color not in row_only]
    col_only = rng.sample(remaining, col_kind_count - shared_count)

    row_colors = shared + row_only
    col_colors = shared + col_only
    rng.shuffle(row_colors)
    rng.shuffle(col_colors)
    return row_colors, col_colors, shared


def _marker_sequence(colors: list[int], rng: random.Random) -> list[int]:
    sequence = list(colors) * REPEAT_COUNT
    rng.shuffle(sequence)
    return sequence


def _group_positions(markers: list[tuple[int, int]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for pos, color in markers:
        grouped.setdefault(int(color), []).append(int(pos))
    return grouped


def _assign_shapes(
    row_colors: list[int],
    col_colors: list[int],
    shared_colors: list[int],
    rng: random.Random,
) -> dict[tuple[int, int], np.ndarray]:
    pairs = [(row_color, col_color) for row_color in row_colors for col_color in col_colors]
    rng.shuffle(pairs)

    available = list(range(len(SHAPES)))
    mapping: dict[tuple[int, int], np.ndarray] = {}
    if shared_colors:
        shared_color = int(shared_colors[0])
        mapping[(shared_color, shared_color)] = SHAPES[FULL_SHAPE_INDEX]
        pairs.remove((shared_color, shared_color))
        available.remove(FULL_SHAPE_INDEX)

    rng.shuffle(available)
    for pair, shape_idx in zip(pairs, available, strict=False):
        mapping[pair] = SHAPES[shape_idx]

    assert len(mapping) == len(row_colors) * len(col_colors)
    return mapping


def _expected_output(
    grid: np.ndarray,
    row_markers: list[tuple[int, int]],
    col_markers: list[tuple[int, int]],
    pair_to_shape: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for r, row_color in row_markers:
        for c, col_color in col_markers:
            mask = pair_to_shape[(int(row_color), int(col_color))]
            color = _earlier_color(int(row_color), int(col_color))
            _draw_shape(out, int(r), int(c), mask, color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_kind_count, col_kind_count = rng.choice(LAYOUT_OPTIONS)
    row_colors, col_colors, shared_colors = _choose_color_sets(row_kind_count, col_kind_count, rng)

    row_centers, height = _sample_centers(row_kind_count * REPEAT_COUNT, rng)
    col_centers, width = _sample_centers(col_kind_count * REPEAT_COUNT, rng)

    row_markers = sorted(zip(row_centers, _marker_sequence(row_colors, rng)), key=lambda item: item[0])
    col_markers = sorted(zip(col_centers, _marker_sequence(col_colors, rng)), key=lambda item: item[0])

    rows_by_color = _group_positions(row_markers)
    cols_by_color = _group_positions(col_markers)
    pair_to_shape = _assign_shapes(row_colors, col_colors, shared_colors, rng)

    observed_cells: set[tuple[int, int]] = set()
    hidden_cells: set[tuple[int, int]] = set()
    for row_color in row_colors:
        for col_color in col_colors:
            cells = [
                (r, c)
                for r in rows_by_color[int(row_color)]
                for c in cols_by_color[int(col_color)]
            ]
            assert len(cells) == REPEAT_COUNT * REPEAT_COUNT
            visible_count = rng.randint(1, len(cells) - 1)
            visible = set(rng.sample(cells, visible_count))
            observed_cells |= visible
            hidden_cells |= set(cells) - visible

    if not observed_cells or not hidden_cells:
        return None

    grid = np.zeros((height, width), dtype=int)
    for r, color in row_markers:
        grid[int(r), 0] = int(color)
    for c, color in col_markers:
        grid[0, int(c)] = int(color)

    row_color_by_pos = {int(r): int(color) for r, color in row_markers}
    col_color_by_pos = {int(c): int(color) for c, color in col_markers}

    for r, c in observed_cells:
        row_color = row_color_by_pos[int(r)]
        col_color = col_color_by_pos[int(c)]
        mask = pair_to_shape[(row_color, col_color)]
        color = _earlier_color(row_color, col_color)
        _draw_shape(grid, int(r), int(c), mask, color)

    expected = _expected_output(grid, row_markers, col_markers, pair_to_shape)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
