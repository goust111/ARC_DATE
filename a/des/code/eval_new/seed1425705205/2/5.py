from __future__ import annotations

from collections import Counter
import random

import numpy as np


ALL_COLORS = list(range(10))
WIDTH_RANGE = (4, 8)
TOWER_HEIGHT_RANGE = (2, 5)
TOP_PADDING_RANGE = (4, 10)
INSTRUCTION_COUNT_RANGE = (1, 7)
BOTTOM_PADDING_RANGE = (0, 3)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    bg = int(values[order[0]])
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]]), "background must be uniquely dominant"
    return bg


def _solid_non_bg_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if int(row[0]) == bg:
            continue
        if bool(np.all(row == row[0])):
            rows.append(int(r))
    return rows


def _parse_with_separator(
    grid: np.ndarray, bg: int, sep_row: int
) -> tuple[list[dict[str, int | list[int]]], list[int]]:
    h, w = grid.shape
    assert 1 < sep_row < h - 1, "separator must split the grid into non-empty upper and lower regions"
    sep_color = int(grid[sep_row, 0])
    assert sep_color != bg and bool(np.all(grid[sep_row] == sep_color)), "separator must be a solid non-bg row"

    top = grid[:sep_row]
    bottom = grid[sep_row + 1 :]
    assert top.shape[0] >= TOWER_HEIGHT_RANGE[0], "upper region must hold the towers"

    towers: list[dict[str, int | list[int]]] = []
    upper_palette: set[int] = set()
    for c in range(w):
        filled = np.flatnonzero(top[:, c] != bg)
        assert TOWER_HEIGHT_RANGE[0] <= len(filled) <= TOWER_HEIGHT_RANGE[1], "each column needs one 2-5 cell tower"
        start = int(filled[0])
        expected = np.arange(start, sep_row)
        assert np.array_equal(filled, expected), "tower pixels must form one contiguous bottom-aligned stack"
        colors = [int(top[r, c]) for r in filled]
        assert len(colors) == len(set(colors)), "tower colors must be distinct within each column"
        towers.append({"col": int(c), "start": start, "colors": colors[:]})
        upper_palette.update(colors)

    assert sep_color not in upper_palette, "separator color must not appear in the towers"

    instruction_rows = [int(r) for r in range(bottom.shape[0]) if bool(np.any(bottom[r] != bg))]
    assert instruction_rows, "expected at least one instruction row"

    marker_color = None
    instruction_cols: list[int] = []
    for r in instruction_rows:
        row = bottom[r]
        non_bg = np.flatnonzero(row != bg)
        assert len(non_bg) == w, "instruction rows must span the same columns as the towers"
        values = [int(v) for v in row]
        marker_positions = [c for c, color in enumerate(values) if color not in upper_palette]
        assert len(marker_positions) == 1, "each instruction row must contain exactly one special marker"
        marker_col = int(marker_positions[0])
        row_marker = int(values[marker_col])
        assert row_marker != sep_color, "marker color must differ from the separator color"
        assert all(color == row_marker or color in upper_palette for color in values), "non-marker cells must be tower colors"
        if marker_color is None:
            marker_color = row_marker
        assert marker_color == row_marker, "all instruction rows must use the same marker color"
        instruction_cols.append(marker_col)

    return towers, instruction_cols


def _parse_grid(grid: np.ndarray) -> tuple[int, int, list[dict[str, int | list[int]]], list[int]]:
    bg = _most_common_color(grid)
    candidates = _solid_non_bg_rows(grid, bg)
    parsed: list[tuple[int, list[dict[str, int | list[int]]], list[int]]] = []
    for sep_row in candidates:
        try:
            towers, instruction_cols = _parse_with_separator(grid, bg, int(sep_row))
        except AssertionError:
            continue
        parsed.append((int(sep_row), towers, instruction_cols))

    assert len(parsed) == 1, "expected a unique valid separator row"
    sep_row, towers, instruction_cols = parsed[0]
    return bg, sep_row, towers, instruction_cols


def _rotate_up(colors: list[int]) -> list[int]:
    return colors[1:] + colors[:1]


def _render(shape: tuple[int, int], bg: int, sep_row: int, towers: list[dict[str, int | list[int]]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for tower in towers:
        start = int(tower["start"])
        col = int(tower["col"])
        colors = list(tower["colors"])
        out[start:sep_row, col] = colors
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the instruction rows as column-indexed cyclic shifts on the stacked color towers above the separator.

    Input structure:
    - The background is the uniquely most common color.
    - A single solid non-background row splits the grid into an upper tower area and a lower instruction area.
    - In the upper area, every column contains one contiguous vertical tower of 2-5 distinct colors, bottom-aligned to the separator.
    - In each non-empty lower row, every column contains one colored pixel, and exactly one of those pixels uses a marker color that never appears in any tower.

    Rule:
    - Identify the separator row, the tower in each column, and the non-empty instruction rows below.
    - Process the instruction rows from bottom to top.
    - For each instruction row, locate the unique marker pixel; its column index selects the tower to modify.
    - Rotate that tower upward by one step: the top color moves to the bottom, and every other color shifts up by one cell.
    - After all instructions are applied, clear the separator and the entire lower area to background, keeping only the transformed upper towers.

    Key insight:
    - The special lower-row color is not payload data; it is a positional pointer that selects which tower column to rotate, while the other lower-row colors are visual distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, sep_row, towers, instruction_cols = _parse_grid(g)

    for col in reversed(instruction_cols):
        colors = list(towers[col]["colors"])
        towers[col]["colors"] = _rotate_up(colors)

    return _render(g.shape, bg, sep_row, towers)


def _valid_background(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return int(counts[bg]) > max((count for color, count in counts.items() if color != bg), default=0)


def _random_operations(width: int, heights: list[int], count: int) -> list[int]:
    for _ in range(40):
        if count > 1 and random.random() < 0.55:
            focus = random.randrange(width)
            repeat_count = random.randint(2, min(4, count))
            ops = [focus] * repeat_count
            ops.extend(random.randrange(width) for _ in range(count - repeat_count))
            random.shuffle(ops)
        else:
            ops = [random.randrange(width) for _ in range(count)]
        if any(ops.count(col) % heights[col] != 0 for col in range(width)):
            return ops
    return []


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    heights = [random.randint(*TOWER_HEIGHT_RANGE) for _ in range(width)]
    if len(set(heights)) == 1 and random.random() < 0.75:
        idx = random.randrange(width)
        alternatives = [h for h in range(TOWER_HEIGHT_RANGE[0], TOWER_HEIGHT_RANGE[1] + 1) if h != heights[idx]]
        heights[idx] = random.choice(alternatives)

    top_padding = random.randint(*TOP_PADDING_RANGE)
    instruction_count = random.randint(*INSTRUCTION_COUNT_RANGE)
    bottom_padding = random.randint(*BOTTOM_PADDING_RANGE)

    sep_row = max(heights) + top_padding
    total_height = sep_row + 1 + instruction_count + bottom_padding
    if total_height > 30:
        return None

    palette = ALL_COLORS[:]
    random.shuffle(palette)
    bg, sep_color, marker_color = palette[:3]
    tower_palette = palette[3:]

    operations = _random_operations(width, heights, instruction_count)
    if not operations:
        return None

    grid = np.full((total_height, width), bg, dtype=int)
    grid[sep_row, :] = sep_color

    used_upper_colors: set[int] = set()
    for c, height in enumerate(heights):
        tower_colors = random.sample(tower_palette, k=height)
        start = sep_row - height
        grid[start:sep_row, c] = tower_colors
        used_upper_colors.update(map(int, tower_colors))

    upper_palette = sorted(used_upper_colors)
    for i, col in enumerate(operations):
        row_colors = [random.choice(upper_palette) for _ in range(width)]
        row_colors[col] = marker_color
        grid[sep_row + 1 + i, :] = row_colors

    if _solid_non_bg_rows(grid, bg) != [sep_row]:
        return None
    if not _valid_background(grid, bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or not bool(np.all(out[sep_row:] == bg)):
        return None

    changed = False
    for c, height in enumerate(heights):
        if operations.count(c) % height != 0:
            changed = True
            break
    if not changed:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
