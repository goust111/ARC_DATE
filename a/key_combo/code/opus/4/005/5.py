from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components


_STEP = 4
_DOMINO_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_GEN_RETRY_LIMIT = 300
_GEN_MODE_SEQUENCE = ("mixed_cross", "row_only", "col_only", "mixed_dense")
_GEN_CALL_COUNT = 0


def _line_positions(grid: np.ndarray, scaffold_color: int, axis: int) -> list[int]:
    counts = np.sum(grid == int(scaffold_color), axis=1 - int(axis))
    full_len = int(grid.shape[1 - int(axis)])
    lines = [idx for idx, count in enumerate(counts.tolist()) if int(count) >= full_len - 2]
    assert len(lines) >= 3, "expected a repeated scaffold lattice"
    gaps = [b - a for a, b in zip(lines, lines[1:])]
    assert len(set(gaps)) == 1, "expected evenly spaced scaffold lines"
    assert gaps[0] == _STEP, "expected 3x3 cell spacing"
    return lines


def _center_lookup(lines: list[int]) -> tuple[list[int], dict[int, int]]:
    centers = [(a + b) // 2 for a, b in zip(lines, lines[1:])]
    lookup = {lines[idx]: centers[idx - 1] for idx in range(1, len(lines))}
    return centers, lookup


def _infer_structure(grid: np.ndarray) -> tuple[int, int, list[int], list[int], list[int], list[int], dict[int, int], dict[int, int]]:
    bg = int(Color.BLACK)
    assert np.any(grid == bg), "expected a black background"
    fg_colors = colors_present(grid, exclude={bg})
    assert len(fg_colors) >= 2, "expected scaffold plus domino markers"

    counts = {color: int(np.sum(grid == color)) for color in fg_colors}
    scaffold_count = max(counts.values())
    scaffold_candidates = [color for color, count in counts.items() if count == scaffold_count]
    assert len(scaffold_candidates) == 1, "expected a unique scaffold color"
    scaffold_color = int(scaffold_candidates[0])

    row_lines = _line_positions(grid, scaffold_color, axis=0)
    col_lines = _line_positions(grid, scaffold_color, axis=1)
    row_centers, row_lookup = _center_lookup(row_lines)
    col_centers, col_lookup = _center_lookup(col_lines)
    return bg, scaffold_color, row_lines, col_lines, row_centers, col_centers, row_lookup, col_lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill scaffold-cell centers from boundary domino markers, with row fills overriding column fills at crossings.

    Input structure:
    - A black background contains a regular blue lattice made of complete rows and columns spaced four pixels apart.
    - The lattice divides the grid into repeated 3x3 cells whose center pixels are black.
    - Two to four same-color domino markers replace blue pixels on scaffold boundary lines.

    Rule:
    - Identify the scaffold color and infer the evenly spaced boundary rows and columns of the cell lattice.
    - A horizontal domino on boundary row r selects the cell row immediately above that boundary; a vertical domino on boundary column c selects the cell column immediately to the left.
    - Paint only the selected cell-center pixels: vertical dominoes fill every center in their column first, then horizontal dominoes fill every center in their row.
    - Preserve the scaffold and domino markers themselves, so only the chosen center pixels change.

    Key insight:
    - The dominoes are not drawing freehand bars; their orientation chooses an entire lattice row or column in cell coordinates, and the only conflict rule is row-over-column at shared centers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, scaffold_color, _row_lines, _col_lines, row_centers, col_centers, row_lookup, col_lookup = _infer_structure(g)
    marker_colors = [color for color in colors_present(g, exclude={bg, scaffold_color})]
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=marker_colors,
        return_format="dict",
    )
    assert 2 <= len(components) <= 4, "expected two to four domino markers"

    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}

    for comp in components:
        color = int(comp["color"])
        coords = comp["coords"]
        assert int(coords.shape[0]) == 2, "each marker must be a two-pixel domino"
        r0, c0, r1, c1 = comp["bbox"]

        if r0 == r1 and c1 == c0 + 1:
            assert r0 in row_lookup, "horizontal domino must sit on a selectable boundary row"
            target_row = row_lookup[r0]
            assert target_row not in row_colors, "each scaffold boundary row may carry at most one horizontal domino"
            row_colors[target_row] = color
        elif c0 == c1 and r1 == r0 + 1:
            assert c0 in col_lookup, "vertical domino must sit on a selectable boundary column"
            target_col = col_lookup[c0]
            assert target_col not in col_colors, "each scaffold boundary column may carry at most one vertical domino"
            col_colors[target_col] = color
        else:
            raise AssertionError("domino markers must be strictly horizontal or vertical")

    out = g.copy()
    for col_center, color in col_colors.items():
        for row_center in row_centers:
            out[row_center, col_center] = int(color)
    for row_center, color in row_colors.items():
        for col_center in col_centers:
            out[row_center, col_center] = int(color)
    return out


def _build_lattice(cell_rows: int, cell_cols: int) -> np.ndarray:
    grid = np.zeros((cell_rows * _STEP + 1, cell_cols * _STEP + 1), dtype=int)
    grid[::_STEP, :] = Color.BLUE
    grid[:, ::_STEP] = Color.BLUE
    return grid


def _cell_centers(cell_rows: int, cell_cols: int) -> tuple[list[int], list[int]]:
    row_centers = [_STEP * idx - 2 for idx in range(1, cell_rows + 1)]
    col_centers = [_STEP * idx - 2 for idx in range(1, cell_cols + 1)]
    return row_centers, col_centers


def _choose_mode() -> str:
    global _GEN_CALL_COUNT
    mode = _GEN_MODE_SEQUENCE[_GEN_CALL_COUNT % len(_GEN_MODE_SEQUENCE)]
    _GEN_CALL_COUNT += 1
    return mode


def _sample_shape(mode: str) -> tuple[int, int]:
    if mode == "mixed_dense":
        return random.randint(4, 7), random.randint(4, 7)
    return random.randint(3, 7), random.randint(3, 7)


def _sample_counts(mode: str, cell_rows: int, cell_cols: int) -> tuple[int, int]:
    if mode == "row_only":
        return random.randint(2, min(4, cell_rows)), 0
    if mode == "col_only":
        return 0, random.randint(2, min(4, cell_cols))
    if mode == "mixed_dense":
        return random.randint(2, min(2, cell_rows)), random.randint(2, min(2, cell_cols))
    return random.randint(1, min(2, cell_rows)), random.randint(1, min(2, cell_cols))


def _place_horizontal(grid: np.ndarray, line_idx: int, cell_cols: int, color: int) -> tuple[tuple[int, int], tuple[int, int]]:
    row = _STEP * line_idx
    band = random.randint(0, cell_cols - 1)
    start = band * _STEP + random.choice((1, 2))
    grid[row, start : start + 2] = int(color)
    return (row, start), (row, start + 1)


def _place_vertical(grid: np.ndarray, line_idx: int, cell_rows: int, color: int) -> tuple[tuple[int, int], tuple[int, int]]:
    col = _STEP * line_idx
    band = random.randint(0, cell_rows - 1)
    start = band * _STEP + random.choice((1, 2))
    grid[start : start + 2, col] = int(color)
    return (start, col), (start + 1, col)


def _render_expected(
    input_grid: np.ndarray,
    cell_rows: int,
    cell_cols: int,
    horizontal: dict[int, int],
    vertical: dict[int, int],
) -> np.ndarray:
    out = input_grid.copy()
    row_centers, col_centers = _cell_centers(cell_rows, cell_cols)
    for line_idx, color in vertical.items():
        col_center = _STEP * line_idx - 2
        for row_center in row_centers:
            out[row_center, col_center] = int(color)
    for line_idx, color in horizontal.items():
        row_center = _STEP * line_idx - 2
        for col_center in col_centers:
            out[row_center, col_center] = int(color)
    return out


def _render_columns_only(input_grid: np.ndarray, cell_rows: int, cell_cols: int, vertical: dict[int, int]) -> np.ndarray:
    return _render_expected(input_grid, cell_rows, cell_cols, {}, vertical)


def _validate_case(
    input_grid: np.ndarray,
    cell_rows: int,
    cell_cols: int,
    horizontal: dict[int, int],
    vertical: dict[int, int],
    marker_cells: list[tuple[int, list[tuple[int, int]]]],
) -> bool:
    if max(input_grid.shape) > 30:
        return False

    expected = _render_expected(input_grid, cell_rows, cell_cols, horizontal, vertical)
    actual = main(input_grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, input_grid):
        return False

    row_centers, col_centers = _cell_centers(cell_rows, cell_cols)
    all_center_cells = {(r, c) for r in row_centers for c in col_centers}
    changed_centers = {(int(r), int(c)) for r, c in np.argwhere(actual != input_grid)}
    if not changed_centers.issubset(all_center_cells):
        return False

    for color, coords in marker_cells:
        if any(int(input_grid[r, c]) != int(color) for r, c in coords):
            return False
        pruned = input_grid.copy()
        for r, c in coords:
            pruned[r, c] = Color.BLUE
        try:
            pruned_out = main(pruned)
        except AssertionError:
            pruned_out = None
        if pruned_out is not None and np.array_equal(pruned_out, expected):
            return False

    if horizontal and vertical:
        columns_only = _render_columns_only(input_grid, cell_rows, cell_cols, vertical)
        visible_override = False
        for row_idx, row_color in horizontal.items():
            rr = _STEP * row_idx - 2
            for col_idx, col_color in vertical.items():
                cc = _STEP * col_idx - 2
                if int(expected[rr, cc]) != int(row_color):
                    return False
                if int(columns_only[rr, cc]) != int(col_color):
                    return False
                visible_override = True
        if not visible_override:
            return False

    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    cell_rows, cell_cols = _sample_shape(mode)
    h_count, v_count = _sample_counts(mode, cell_rows, cell_cols)
    if h_count + v_count < 2 or h_count + v_count > 4:
        return None

    grid = _build_lattice(cell_rows, cell_cols)
    colors = random.sample(_DOMINO_COLORS, h_count + v_count)
    horizontal_lines = sorted(random.sample(range(1, cell_rows + 1), h_count))
    vertical_lines = sorted(random.sample(range(1, cell_cols + 1), v_count))

    horizontal: dict[int, int] = {}
    vertical: dict[int, int] = {}
    marker_cells: list[tuple[int, list[tuple[int, int]]]] = []

    for line_idx, color in zip(horizontal_lines, colors[:h_count]):
        coords = list(_place_horizontal(grid, line_idx, cell_cols, color))
        horizontal[line_idx] = int(color)
        marker_cells.append((int(color), coords))
    for line_idx, color in zip(vertical_lines, colors[h_count:]):
        coords = list(_place_vertical(grid, line_idx, cell_rows, color))
        vertical[line_idx] = int(color)
        marker_cells.append((int(color), coords))

    if _validate_case(grid, cell_rows, cell_cols, horizontal, vertical, marker_cells):
        return grid
    return None


def generate_input() -> np.ndarray:
    mode = _choose_mode()
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input for mode={mode}")
