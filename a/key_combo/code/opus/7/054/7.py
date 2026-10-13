from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _occupied_intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(map(bool, flags)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(flags)))
    return intervals


def _extract_nested_square(grid: np.ndarray, bg: int) -> np.ndarray:
    col_groups = _occupied_intervals(np.any(grid != bg, axis=0))
    assert len(col_groups) == 2, "expected a left nested square and a right legend area"
    assert col_groups[1][0] - col_groups[0][1] >= 2, "regions must be separated by at least two background columns"

    c0, c1 = col_groups[0]
    left = grid[:, c0:c1]
    row_groups = _occupied_intervals(np.any(left != bg, axis=1))
    assert len(row_groups) == 1, "nested square must occupy one contiguous row band"
    r0, r1 = row_groups[0]
    square = left[r0:r1, :]
    h, w = square.shape
    assert h == w and h % 2 == 0 and h >= 2, "nested object must be an even square"
    assert not np.any(square == bg), "nested square should be fully filled by colored rings"
    return square


def _extract_legends(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    col_groups = _occupied_intervals(np.any(grid != bg, axis=0))
    c0, c1 = col_groups[1]
    legends = grid[:, c0:c1]
    assert legends.shape[1] == 2, "legend area must be two columns wide"

    pairs: list[tuple[int, int]] = []
    for r in range(legends.shape[0]):
        row = legends[r]
        occupied = np.flatnonzero(row != bg)
        if occupied.size == 0:
            continue
        assert occupied.tolist() == [0, 1], "each legend must be a horizontal 1x2 domino"
        pairs.append((int(row[0]), int(row[1])))
    assert pairs, "expected at least one legend"
    return pairs


def _layer_blocks(square: np.ndarray) -> list[np.ndarray]:
    side = square.shape[0]
    blocks: list[np.ndarray] = []
    for layer in range(side // 2 - 1, -1, -1):
        start = layer
        end = side - layer - 1
        border = np.concatenate(
            [
                square[start, start : end + 1],
                square[end, start : end + 1],
                square[start + 1 : end, start],
                square[start + 1 : end, end],
            ]
        )
        assert len(set(map(int, border))) == 1, "each nested layer border must be monochrome"
        block = np.array(
            [
                [square[start, start], square[start, end]],
                [square[end, start], square[end, end]],
            ],
            dtype=int,
        )
        blocks.append(block)
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the nested square into a strip of 2x2 layer blocks and recolor each block by its bound legend row.

    Input structure
    - The grid has a dominant background and exactly two foreground column bands separated by at least two background columns.
    - The left band contains one filled even-sized square made of concentric one-cell-thick colored rings.
    - The right band contains one horizontal 1x2 legend domino per ring, stacked vertically with optional blank rows between them.

    Rule
    - For each nested ring, take its four corner pixels as a 2x2 block.
    - Output those blocks from inner layer to outer layer, so the innermost block is leftmost and the outermost block is rightmost.
    - Read each legend row left-to-right as replacement color, source color, and bind legend row i to output block i.
    - Within each bound block, replace only the pixels equal to that legend's source color with its replacement color.

    Key insight
    - The legends are position-bound after the layer order is reversed: they are not global color substitutions, and the first legend applies to the innermost layer block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    square = _extract_nested_square(g, bg)
    legends = _extract_legends(g, bg)
    blocks = _layer_blocks(square)
    assert len(blocks) == len(legends), "legend count must equal the number of nested layers"

    out = np.empty((2, 2 * len(blocks)), dtype=int)
    for i, (block, (new_color, old_color)) in enumerate(zip(blocks, legends)):
        recolored = block.copy()
        recolored[recolored == old_color] = new_color
        out[:, 2 * i : 2 * i + 2] = recolored
    return out


_LAYER_COUNTS = (3, 4)
_BG = 0
_COLORS = tuple(range(1, 10))
_MAX_ATTEMPTS = 300


def _draw_nested_square(layer_colors: list[int]) -> np.ndarray:
    side = 2 * len(layer_colors)
    square = np.empty((side, side), dtype=int)
    for layer, color in enumerate(layer_colors):
        square[layer : side - layer, layer : side - layer] = int(color)
    return square


def _valid_old_orders(layer_colors: list[int]) -> list[tuple[int, ...]]:
    outer = tuple(layer_colors)
    inner_to_outer = outer[::-1]
    valid = []
    for perm in itertools.permutations(outer):
        correct_matches = [perm[i] == inner_to_outer[i] for i in range(len(outer))]
        if not any(correct_matches):
            continue
        if all(correct_matches):
            continue
        if not any(perm[i] == outer[i] and outer[i] != inner_to_outer[i] for i in range(len(outer))):
            continue
        valid.append(tuple(int(v) for v in perm))
    assert valid, "expected at least one non-degenerate legend order"
    return valid


def _expected_output(layer_colors: list[int], new_colors: list[int], old_colors: tuple[int, ...]) -> np.ndarray:
    layer_count = len(layer_colors)
    out = np.empty((2, 2 * layer_count), dtype=int)
    for i, layer_color in enumerate(reversed(layer_colors)):
        value = int(new_colors[i]) if int(old_colors[i]) == int(layer_color) else int(layer_color)
        out[:, 2 * i : 2 * i + 2] = value
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    layer_count = random.choice(_LAYER_COUNTS)
    side = 2 * layer_count
    legend_height = 2 * layer_count - 1

    palette = random.sample(_COLORS, 2 * layer_count)
    layer_colors = [int(v) for v in palette[:layer_count]]
    new_colors = [int(v) for v in palette[layer_count:]]
    old_colors = random.choice(_valid_old_orders(layer_colors))

    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    gap = random.randint(2, 5)
    square_top = random.randint(1, 4)
    legend_top = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)

    height = max(square_top + side, legend_top + legend_height) + bottom_margin
    width = left_margin + side + gap + 2 + right_margin
    assert height <= 30 and width <= 30

    grid = np.full((height, width), _BG, dtype=int)
    square = _draw_nested_square(layer_colors)
    grid[square_top : square_top + side, left_margin : left_margin + side] = square

    legend_col = left_margin + side + gap
    for i, (new_color, old_color) in enumerate(zip(new_colors, old_colors)):
        r = legend_top + 2 * i
        grid[r, legend_col] = int(new_color)
        grid[r, legend_col + 1] = int(old_color)

    expected = _expected_output(layer_colors, new_colors, old_colors)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid, expected = _build_case()
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate valid input")
