import numpy as np
import math
import random


def _unique_background(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    assert winners.size == 1, "expected a unique dominant background color"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Double the grid, preserving foreground pixels as solid blocks and marking their NW-SE diagonals through background cells.

    Input structure:
    - The grid has a unique dominant background color.
    - All non-background pixels are foreground markers; their colors are preserved.
    - Blue is the generated diagonal marker color.

    Rule:
    - Make an output canvas with twice the input height and width.
    - Each foreground input pixel becomes a solid 2x2 block of the same color.
    - Each background input pixel whose row-column offset matches at least one foreground pixel becomes a 2x2 block with only its main diagonal colored blue.

    Key insight:
    - The foreground pixels select complete NW-SE diagonals in the input coordinate system, and the doubled output visualizes those selected background cells with mini-diagonals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    bg = _unique_background(g)
    marker = 1
    assert marker != bg, "expected blue to be distinct from the background"

    foreground_positions = np.argwhere(g != bg)
    assert foreground_positions.shape[0] > 0, "expected at least one foreground pixel"
    diagonal_offsets = {int(row - col) for row, col in foreground_positions}

    out = np.full((2 * height, 2 * width), bg, dtype=int)
    for row in range(height):
        for col in range(width):
            out_row = 2 * row
            out_col = 2 * col
            color = int(g[row, col])
            if color != bg:
                out[out_row : out_row + 2, out_col : out_col + 2] = color
            elif row - col in diagonal_offsets:
                out[out_row, out_col] = marker
                out[out_row + 1, out_col + 1] = marker

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, foreground, and retry configuration.
GRID_SIZE_MIN, GRID_SIZE_MAX = 2, 6
CELL_SCALE = 2
BACKGROUND_COLOR, DIAGONAL_COLOR = 0, 1
FOREGROUND_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
FOREGROUND_COUNT_MIN, FOREGROUND_COUNT_MAX = 1, 3
FOREGROUND_DENSITY_MIN, FOREGROUND_DENSITY_MAX_DIVISOR = 0.05, 4
SEPARATION_RADIUS, MIN_ANCHOR_DIAGONAL_CELLS = 1, 2
GENERATION_ATTEMPTS = 300


def _gen_cell_offset(cell: int, size: int) -> int:
    row, col = divmod(cell, size)
    return row - col


def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    """Render the generator oracle independently from main()."""
    height, width = map(int, grid.shape)
    foreground_mask = grid != BACKGROUND_COLOR
    foreground_positions = np.argwhere(foreground_mask)
    diagonal_offsets = {int(row - col) for row, col in foreground_positions}

    out = np.repeat(np.repeat(grid, CELL_SCALE, axis=0), CELL_SCALE, axis=1)
    rows, cols = np.indices((height, width))
    selected_background = (~foreground_mask) & np.isin(rows - cols, list(diagonal_offsets))
    out[CELL_SCALE * rows[selected_background], CELL_SCALE * cols[selected_background]] = DIAGONAL_COLOR
    out[CELL_SCALE * rows[selected_background] + 1, CELL_SCALE * cols[selected_background] + 1] = DIAGONAL_COLOR
    return out


def _gen_cells_on_multi_cell_diagonals(all_cells: list[int], size: int) -> list[int]:
    return [cell for cell in all_cells if size - abs(_gen_cell_offset(cell, size)) >= MIN_ANCHOR_DIAGONAL_CELLS]


def _gen_is_separated(cell: int, chosen_cells: list[int], size: int) -> bool:
    row, col = divmod(cell, size)
    for chosen in chosen_cells:
        other_row, other_col = divmod(chosen, size)
        if max(abs(row - other_row), abs(col - other_col)) <= SEPARATION_RADIUS:
            return False
    return True


def _gen_foreground_cells(size: int, foreground_count: int, all_cells: list[int]) -> list[int] | None:
    """Pick separated foreground cells while reserving one same-diagonal background cell."""
    anchor_cell = random.choice(_gen_cells_on_multi_cell_diagonals(all_cells, size))
    anchor_offset = _gen_cell_offset(anchor_cell, size)
    reserved_background = random.choice(
        [
            cell
            for cell in all_cells
            if cell != anchor_cell and _gen_cell_offset(cell, size) == anchor_offset
        ]
    )
    available_cells = [cell for cell in all_cells if cell not in {anchor_cell, reserved_background}]
    foreground_cells = [anchor_cell]
    for _ in range(foreground_count - 1):
        candidates = [cell for cell in available_cells if _gen_is_separated(cell, foreground_cells, size)]
        if not candidates:
            return None
        cell = random.choice(candidates)
        foreground_cells.append(cell)
        available_cells.remove(cell)
    return foreground_cells


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    total_cells = size * size
    min_foreground = max(FOREGROUND_COUNT_MIN, math.ceil(FOREGROUND_DENSITY_MIN * total_cells))
    max_foreground = min(FOREGROUND_COUNT_MAX, total_cells // FOREGROUND_DENSITY_MAX_DIVISOR)
    if min_foreground > max_foreground:
        return None

    foreground_count = random.randint(min_foreground, max_foreground)
    foreground_color = random.choice(FOREGROUND_COLORS)

    all_cells = list(range(total_cells))
    foreground_cells = _gen_foreground_cells(size, foreground_count, all_cells)
    if foreground_cells is None:
        return None

    grid = np.zeros((size, size), dtype=int)
    rows, cols = np.divmod(foreground_cells, size)
    grid[rows, cols] = foreground_color

    if total_cells - foreground_count <= foreground_count:
        return None

    expected = _gen_expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid f0afb749 input")
