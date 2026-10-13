import numpy as np

from arc_utils import colors_present, get_bg
import random


def _divider_grid(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, list[int], list[int]]] = []
    H, W = g.shape
    for color in colors_present(g, exclude={bg}):
        rows = [r for r in range(H) if np.all(g[r, :] == color)]
        cols = [c for c in range(W) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))

    assert len(candidates) == 1
    return candidates[0]


def _segments(n: int, dividers: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = 0
    for divider in dividers:
        assert divider >= start
        if start < divider:
            segments.append((start, divider))
        start = divider + 1
    if start < n:
        segments.append((start, n))

    assert segments
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the unique template mask into every divider-grid cell, recolored by each row's left marker.

    Input structure
    - The background is the dominant color.
    - One non-background color forms full horizontal and vertical divider bars, splitting the grid into equal-size cells.
    - In the first column of cells, each row has a single-color marker that gives that row's output color.
    - Exactly one non-first-column cell contains the single-color template mask to copy.

    Rule
    - Read the row color from the marker in the first cell of each row band.
    - Read the template as the non-background mask in the unique template cell.
    - Clear every cell interior to background, then paint the template mask into every cell of a row using that row's marker color.
    - Leave the divider bars unchanged.

    Key insight
    - The visible colors are not copied as objects; the first column supplies a row palette and the lone extra cell supplies the shape mask.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)
    divider_color, divider_rows, divider_cols = _divider_grid(g, bg)

    row_segments = _segments(H, divider_rows)
    col_segments = _segments(W, divider_cols)
    assert len(set(r1 - r0 for r0, r1 in row_segments)) == 1
    assert len(set(c1 - c0 for c0, c1 in col_segments)) == 1

    row_colors: list[int] = []
    for r0, r1 in row_segments:
        c0, c1 = col_segments[0]
        cell = g[r0:r1, c0:c1]
        marker_colors = colors_present(cell, exclude={bg})
        assert len(marker_colors) == 1
        assert marker_colors[0] != divider_color
        row_colors.append(marker_colors[0])

    template_masks: list[np.ndarray] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments[1:]:
            cell = g[r0:r1, c0:c1]
            cell_colors = colors_present(cell, exclude={bg})
            if not cell_colors:
                continue
            assert len(cell_colors) == 1
            assert cell_colors[0] != divider_color
            template_masks.append(cell != bg)

    assert len(template_masks) == 1
    template_mask = template_masks[0]
    assert np.any(template_mask)

    out = g.copy()
    for color, (r0, r1) in zip(row_colors, row_segments):
        for c0, c1 in col_segments:
            cell = out[r0:r1, c0:c1]
            cell[:, :] = bg
            cell[template_mask] = color

    return out

# --------------- generator (o5x) ---------------

# Grid geometry.
GRID_N_MIN, GRID_N_MAX = 3, 5
CELL_SIZE, DIVIDER_WIDTH = 4, 1
CELL_STRIDE = CELL_SIZE + DIVIDER_WIDTH

# Color palette.
BG_COLOR = 0
COLOR_VALUES = tuple(range(10))
NON_BG_EXTRA_COLORS = 2

# Template and marker shapes.
TEMPLATE_PIXELS_MIN, TEMPLATE_PIXELS_MAX = 6, 8
MARKER_OFFSET, MARKER_SIZE, TEMPLATE_COL_MIN = 1, 2, 1

# Retry guard.
MAX_GENERATION_ATTEMPTS = 300

def _gen_cell_bounds(index: int) -> tuple[int, int]:
    start = index * CELL_STRIDE
    return start, start + CELL_SIZE

def _gen_random_mask() -> np.ndarray:
    mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
    count = random.randint(TEMPLATE_PIXELS_MIN, TEMPLATE_PIXELS_MAX)
    for pos in random.sample(range(CELL_SIZE * CELL_SIZE), count):
        mask[pos // CELL_SIZE, pos % CELL_SIZE] = True
    return mask

def _gen_divider_grid(n: int, divider_color: int) -> np.ndarray:
    size = n * CELL_SIZE + (n - 1) * DIVIDER_WIDTH
    grid = np.full((size, size), BG_COLOR, dtype=int)
    for d in range(CELL_SIZE, size, CELL_STRIDE):
        grid[d, :] = divider_color
        grid[:, d] = divider_color
    return grid

def _gen_output_grid(n, divider_color, mask, row_colors):
    grid = _gen_divider_grid(n, divider_color)
    for row_index, color in enumerate(row_colors):
        r0, r1 = _gen_cell_bounds(row_index)
        for col_index in range(n):
            c0, c1 = _gen_cell_bounds(col_index)
            grid[r0:r1, c0:c1][mask] = color
    return grid

def _gen_input_grid(n, divider_color, mask, row_colors, template_color):
    grid = _gen_divider_grid(n, divider_color)
    marker_end = MARKER_OFFSET + MARKER_SIZE
    for row_index, color in enumerate(row_colors):
        r0, _ = _gen_cell_bounds(row_index)
        c0, _ = _gen_cell_bounds(0)
        grid[r0 + MARKER_OFFSET : r0 + marker_end, c0 + MARKER_OFFSET : c0 + marker_end] = color
    template_row = random.randrange(n)
    template_col = random.randrange(TEMPLATE_COL_MIN, n)
    r0, r1 = _gen_cell_bounds(template_row)
    c0, c1 = _gen_cell_bounds(template_col)
    grid[r0:r1, c0:c1][mask] = template_color
    return grid

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_N_MIN, GRID_N_MAX)
    palette = random.sample([c for c in COLOR_VALUES if c != BG_COLOR], n + NON_BG_EXTRA_COLORS)
    divider_color, row_colors, template_color = palette[0], palette[1 : 1 + n], palette[-1]
    template_mask = _gen_random_mask()

    output = _gen_output_grid(n, divider_color, template_mask, row_colors)
    inp = _gen_input_grid(n, divider_color, template_mask, row_colors, template_color)

    if not np.array_equal(main(inp), output):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
