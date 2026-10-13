from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import Color, colors_present, get_bg


def _count_prefix(row: np.ndarray, color: int) -> int:
    length = 0
    for value in map(int, row):
        if value != int(color):
            break
        length += 1
    return length


def _increment_digits(digits: list[int], base: int) -> list[int]:
    out = list(digits)
    carry = 1
    for idx in range(len(out) - 1, -1, -1):
        if carry == 0:
            break
        new_value = out[idx] + carry
        if new_value >= base:
            out[idx] = 0
            carry = 1
        else:
            out[idx] = new_value
            carry = 0
    return out


def _extract_layout(grid: np.ndarray) -> tuple[int, int, int, int, np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    non_bg = colors_present(g, exclude={bg})
    assert len(non_bg) == 2, "expected exactly two foreground colors"

    first_col_colors = sorted(set(map(int, g[:, 0])))
    assert len(first_col_colors) == 1 and first_col_colors[0] != bg, "left edge bar missing"
    edge_color = int(first_col_colors[0])

    legend_candidates = [color for color in non_bg if int(color) != edge_color]
    assert len(legend_candidates) == 1, "legend color must differ from edge color"
    legend_color = int(legend_candidates[0])

    legend_coords = np.argwhere(g == legend_color)
    assert legend_coords.size > 0, "legend bar missing"
    legend_rows = sorted(set(map(int, legend_coords[:, 0])))
    assert legend_rows == [0], "legend bar must occupy the top row"
    legend_cols = sorted(map(int, legend_coords[:, 1]))
    assert legend_cols == list(range(legend_cols[0], legend_cols[0] + len(legend_cols))), "legend must be contiguous"
    base = len(legend_cols)
    assert base >= 2, "legend length must define a valid base"

    digits: list[int] = []
    for r in range(1, g.shape[0]):
        row = g[r]
        prefix = _count_prefix(row, edge_color)
        assert prefix >= 1, "every lane must include the edge bar"
        digit = prefix - 1
        assert 0 <= digit < base, "marker bar length must be a base-N digit"
        assert not np.any(row[prefix:] == edge_color), "marker bar must be a single prefix run"
        assert not np.any(row == legend_color), "legend color should appear only in the top row"
        digits.append(digit)

    return bg, edge_color, legend_color, legend_cols[0], legend_coords, digits


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Treat the marker lengths on rows 1..H-1 as base-N digits, with the bottom row
    as the least-significant digit, and add one with carry. The left edge bar and
    top legend bar remain fixed.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, edge_color, legend_color, _legend_start, legend_coords, digits = _extract_layout(g)
    base = int(legend_coords.shape[0])
    new_digits = _increment_digits(digits, base)

    out = np.full(g.shape, int(bg), dtype=int)
    out[:, 0] = int(edge_color)
    out[legend_coords[:, 0], legend_coords[:, 1]] = int(legend_color)

    for row_idx, digit in enumerate(new_digits, start=1):
        if digit > 0:
            out[row_idx, 1 : digit + 1] = int(edge_color)
    return out


def _pick_colors() -> tuple[int, int, int]:
    palette = list(Color.ALL_COLORS)
    bg = int(random.choice(palette))
    palette.remove(bg)
    edge_color = int(random.choice(palette))
    palette.remove(edge_color)
    legend_color = int(random.choice(palette))
    return bg, edge_color, legend_color


def _sample_digits(base: int, count: int) -> list[int]:
    mode = random.choice(["free", "no_carry", "single_carry", "cascade"])
    digits = [random.randint(0, base - 1) for _ in range(count)]

    if mode == "no_carry":
        digits[-1] = random.randint(0, base - 2)
    elif mode == "single_carry":
        digits[-1] = base - 1
        if count >= 2:
            digits[-2] = random.randint(0, base - 2)
    elif mode == "cascade":
        span = random.randint(1, count)
        for idx in range(count - span, count):
            digits[idx] = base - 1
        if count - span - 1 >= 0:
            digits[count - span - 1] = random.randint(0, base - 2)

    return digits


def generate_input() -> np.ndarray:
    """
    Build a counter scene:
    - leftmost column: fixed full edge bar
    - top row: separate legend bar whose length is the counter base
    - remaining rows: marker lengths encoding digits from top to bottom
    """
    for _ in range(200):
        base = random.randint(2, 10)
        digit_count = random.randint(2, 10)
        legend_start = random.randint(2, 6)
        height = digit_count + 1

        digits = _sample_digits(base, digit_count)
        edge_cells = height + sum(digits)
        structural_width = legend_start + base + random.randint(1, 4)
        bg_safe_width = ((2 * edge_cells + base) // height) + 2
        width = max(structural_width, bg_safe_width)
        if height > 30 or width > 30:
            continue

        bg, edge_color, legend_color = _pick_colors()
        grid = np.full((height, width), int(bg), dtype=int)
        grid[:, 0] = int(edge_color)
        grid[0, legend_start : legend_start + base] = int(legend_color)

        for row_idx, digit in enumerate(digits, start=1):
            if digit > 0:
                grid[row_idx, 1 : digit + 1] = int(edge_color)

        assert int(get_bg(grid)) == int(bg)
        return grid

    raise RuntimeError("failed to sample a valid counter grid")
