from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


GEN_ATTEMPTS = 400
LATTICE_COLOR = Color.TEAL
BACKGROUND = Color.BLACK
MARKER_COLORS = [color for color in Color.ALL_COLORS if color not in (BACKGROUND, LATTICE_COLOR)]

CHAMBER_ROW_RANGE = (2, 5)
CHAMBER_COL_RANGE = (2, 5)
CELL_HEIGHT_RANGE = (2, 5)
CELL_WIDTH_RANGE = (2, 5)
TOP_MARGIN_RANGE = (2, 5)
SIDE_MARGIN_RANGE = (0, 3)
BOTTOM_MARGIN_RANGE = (0, 2)


def _wall_lines(counts: np.ndarray) -> list[int]:
    lines: list[int] = []
    for idx, value in enumerate(counts):
        value = int(value)
        if value <= 0:
            continue
        left = int(counts[idx - 1]) if idx > 0 else 0
        right = int(counts[idx + 1]) if idx + 1 < len(counts) else 0
        if value > left and value > right:
            lines.append(idx)
    if not lines:
        positive = [int(v) for v in counts if int(v) > 0]
        assert positive, "expected lattice-colored rows/cols"
        high = max(positive)
        lines = [idx for idx, value in enumerate(counts) if int(value) == high]
    assert len(lines) >= 2, "expected at least two wall lines"
    return lines


def _find_lattice_color(grid: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    options = [(int(count), int(color)) for color, count in zip(colors, counts) if int(color) != int(bg)]
    assert options, "expected non-background colors"
    return max(options)[1]


def _build_template(
    shape: tuple[int, int],
    row_lines: list[int],
    col_lines: list[int],
    lattice_color: int,
    bg: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    top, bottom = row_lines[0], row_lines[-1]
    left, right = col_lines[0], col_lines[-1]
    for row in row_lines:
        out[row, left : right + 1] = int(lattice_color)
    for col in col_lines:
        out[top : bottom + 1, col] = int(lattice_color)
    return out


def _chamber_col(col_lines: list[int], c: int) -> int:
    for index in range(len(col_lines) - 1):
        if col_lines[index] < c < col_lines[index + 1]:
            return index
    raise AssertionError("marker column must lie inside a chamber column")


def _solve_from_structure(grid: np.ndarray) -> np.ndarray:
    bg = get_bg(grid)
    lattice_color = _find_lattice_color(grid, bg)
    lattice_mask = grid == int(lattice_color)
    row_lines = _wall_lines(lattice_mask.sum(axis=1))
    col_lines = _wall_lines(lattice_mask.sum(axis=0))

    out = _build_template(grid.shape, row_lines, col_lines, lattice_color, bg)
    markers = np.argwhere((grid != int(bg)) & (grid != int(lattice_color)))
    assert markers.size > 0, "expected at least one marker"

    for r, c in markers:
        color = int(grid[r, c])
        chamber_index = _chamber_col(col_lines, int(c))
        assert int(grid[row_lines[0], c]) == int(bg), "marker column must open through the top wall"

        wall_index = 1
        while wall_index < len(row_lines) - 1 and int(grid[row_lines[wall_index], c]) == int(bg):
            wall_index += 1
        stop_row = wall_index - 1

        r0 = row_lines[stop_row] + 1
        r1 = row_lines[stop_row + 1]
        c0 = col_lines[chamber_index] + 1
        c1 = col_lines[chamber_index + 1]
        out[r0:r1, c0:c1] = color

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing pixels of a periodic teal lattice, then drop each colored
    marker down its column and fill the first chamber whose lower wall is intact.
    """
    grid = np.asarray(input_grid, dtype=int)
    return _solve_from_structure(grid)


def _make_instance() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_ATTEMPTS):
        chamber_rows = random.randint(*CHAMBER_ROW_RANGE)
        chamber_cols = random.randint(*CHAMBER_COL_RANGE)
        cell_h = random.randint(*CELL_HEIGHT_RANGE)
        cell_w = random.randint(*CELL_WIDTH_RANGE)
        top_margin = random.randint(*TOP_MARGIN_RANGE)
        left_margin = random.randint(*SIDE_MARGIN_RANGE)
        right_margin = random.randint(*SIDE_MARGIN_RANGE)
        bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)

        lattice_h = chamber_rows * cell_h + chamber_rows + 1
        lattice_w = chamber_cols * cell_w + chamber_cols + 1
        height = top_margin + lattice_h + bottom_margin
        width = left_margin + lattice_w + right_margin
        if height > 30 or width > 30:
            continue

        row_lines = [top_margin + idx * (cell_h + 1) for idx in range(chamber_rows + 1)]
        col_lines = [left_margin + idx * (cell_w + 1) for idx in range(chamber_cols + 1)]

        template = _build_template((height, width), row_lines, col_lines, LATTICE_COLOR, BACKGROUND)
        grid = template.copy()
        out = template.copy()

        marker_count = random.randint(1, min(chamber_cols, len(MARKER_COLORS)))
        chamber_choices = random.sample(range(chamber_cols), marker_count)
        color_choices = random.sample(MARKER_COLORS, marker_count)

        for chamber_col, color in zip(chamber_choices, color_choices):
            col = random.randint(col_lines[chamber_col] + 1, col_lines[chamber_col + 1] - 1)
            stop_row = random.randint(0, chamber_rows - 1)
            for wall_row in row_lines[: stop_row + 1]:
                grid[wall_row, col] = BACKGROUND

            marker_row = random.randint(0, top_margin - 1)
            grid[marker_row, col] = int(color)

            r0 = row_lines[stop_row] + 1
            r1 = row_lines[stop_row + 1]
            c0 = col_lines[chamber_col] + 1
            c1 = col_lines[chamber_col + 1]
            out[r0:r1, c0:c1] = int(color)

        assert np.array_equal(main(grid), out), "generator must match solver"
        return grid, out

    raise RuntimeError("failed to generate a valid lattice-drop instance")


def generate_input() -> np.ndarray:
    return _make_instance()[0]
