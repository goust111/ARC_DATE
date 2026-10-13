import numpy as np

from arc_utils import Color, bbox_coords, components_mask
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the cell regions of a lattice so perimeter regions become red and interior regions become green.

    Input structure:
    - The input has exactly two colors: one lattice color and one cell-region color.
    - The lattice color fills the whole outer border and separates rectangular 4-connected cell regions.
    - The cell regions form a complete rectangular array with at least three rows and three columns.

    Rule:
    - Preserve every lattice pixel.
    - For each cell region, recolor it red if its region-row or region-column is on the outer edge of the array.
    - Recolor all strictly interior cell regions green.

    Key insight:
    - The black cells are not independent pixels; they are the holes of a larger lattice, and their array position determines the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    colors = set(map(int, np.unique(g)))
    assert len(colors) == 2, "expected exactly two input colors"

    border_size_with_corners_counted_twice = 2 * height + 2 * width
    lattice_colors = [
        color
        for color in colors
        if int(
            (g[0, :] == color).sum()
            + (g[-1, :] == color).sum()
            + (g[:, 0] == color).sum()
            + (g[:, -1] == color).sum()
        )
        == border_size_with_corners_counted_twice
    ]
    assert len(lattice_colors) == 1, "expected one color to fill the complete outer border"
    lattice_color = lattice_colors[0]

    cell_colors = [color for color in colors if color != lattice_color]
    assert len(cell_colors) == 1, "expected one cell-region color"
    cell_color = cell_colors[0]

    regions = []
    for coords in components_mask(g == cell_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        assert coords.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "cell regions must be rectangular"
        regions.append((r0, c0, coords))

    row_starts = sorted({r0 for r0, _, _ in regions})
    col_starts = sorted({c0 for _, c0, _ in regions})
    assert len(row_starts) >= 3 and len(col_starts) >= 3, "expected an array with an interior"
    positions = {(r0, c0) for r0, c0, _ in regions}
    expected_positions = {(r0, c0) for r0 in row_starts for c0 in col_starts}
    assert positions == expected_positions, "expected every lattice slot to contain one region"

    out = g.copy()
    edge_rows = {row_starts[0], row_starts[-1]}
    edge_cols = {col_starts[0], col_starts[-1]}
    for r0, c0, coords in regions:
        fill_color = Color.RED if r0 in edge_rows or c0 in edge_cols else Color.GREEN
        out[coords[:, 0], coords[:, 1]] = fill_color

    return out

# --------------- generator (o5x) ---------------

# Canvas and lattice geometry.
GRID_DIM_RANGE, ARRAY_SIZE_RANGE = range(19, 31), range(3, 12)
CELL_SIZE_RANGE, GAP_RANGE = range(1, 5), range(1, 3)
BORDER_TOP_LEFT = 1

# Fixed colors and retry policy.
LATTICE_COLOR, CELL_COLOR = 1, 0
MAX_GENERATION_ATTEMPTS = 300

VALID_LAYOUTS_BY_SIZE = {}
for n in ARRAY_SIZE_RANGE:
    combos = []
    for cell_size in CELL_SIZE_RANGE:
        for gap in GAP_RANGE:
            br_options = (BORDER_TOP_LEFT,) if gap == BORDER_TOP_LEFT else (BORDER_TOP_LEFT, gap)
            for border_bottom_right in br_options:
                dim = BORDER_TOP_LEFT + n * cell_size + (n - 1) * gap + border_bottom_right
                if dim in GRID_DIM_RANGE:
                    combos.append((cell_size, gap, border_bottom_right, dim))
    if combos:
        VALID_LAYOUTS_BY_SIZE[n] = combos


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(tuple(VALID_LAYOUTS_BY_SIZE))
    cell_size, gap, _, dim = random.choice(VALID_LAYOUTS_BY_SIZE[n])

    grid = np.full((dim, dim), LATTICE_COLOR, dtype=int)
    expected = grid.copy()
    stride = cell_size + gap

    for i in range(n):
        r0 = BORDER_TOP_LEFT + i * stride
        for j in range(n):
            c0 = BORDER_TOP_LEFT + j * stride
            fill_color = Color.RED if i in (0, n - 1) or j in (0, n - 1) else Color.GREEN
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = CELL_COLOR
            expected[r0 : r0 + cell_size, c0 : c0 + cell_size] = fill_color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 8a371977 input")
