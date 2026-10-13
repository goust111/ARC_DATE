import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the square grid across its uniform foreground main diagonal.

    Input structure
    - The input is a square grid with a dominant background color.
    - The main diagonal is a complete one-color foreground line.
    - The diagonal color appears only on that line; other foreground colors are off the line.

    Rule
    - Keep the diagonal fixed and move every off-diagonal pixel to the cell with swapped row/column coordinates.
    - Equivalently, return the matrix transpose of the input grid.

    Key insight
    - The one-color diagonal is the reflection axis, so color values are preserved while coordinates are transposed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"
    assert height == width, "expected a square grid"

    bg = get_bg(g)
    diag = np.diag(g)
    diag_color = int(diag[0])
    assert np.all(diag == diag_color), "expected a uniform main diagonal"
    assert diag_color != bg, "diagonal must be foreground"
    assert np.count_nonzero(g == diag_color) == height, "diagonal color must occur only on the main diagonal"

    return g.T.copy()

# --------------- generator (o5x) ---------------

# Canvas and colors.
MIN_GRID_SIZE, MAX_GRID_SIZE = 3, 6
BACKGROUND_COLOR, DIAGONAL_COLOR = 0, 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
# Lower-triangle occupancy.
MIN_TOTAL_CELLS, MAX_TOTAL_CELLS = 1, 7
MIN_DENSITY_NUMERATOR, MIN_DENSITY_DENOMINATOR = 3, 10
MAX_DENSITY_DIVISOR = 2
# Segment/color sampling.
MIN_COLOR_GROUPS, MAX_COLOR_GROUPS = 1, 4
SINGLE_CELL_SIZE, TWO_CELL_SIZE = 1, 2
# Retry budget.
GENERATION_RETRIES = 300


def _gen_color_sizes(n):
    """Return sampled (color, size) pairs, or None if density bounds are impossible."""
    lower_cells = n * (n - 1) // 2
    raw_min_cells = MIN_DENSITY_NUMERATOR * lower_cells + MIN_DENSITY_DENOMINATOR - 1
    min_cells = max(MIN_TOTAL_CELLS, raw_min_cells // MIN_DENSITY_DENOMINATOR)
    max_cells = min(MAX_TOTAL_CELLS, lower_cells // MAX_DENSITY_DIVISOR)

    max_colors = min(MAX_COLOR_GROUPS, n - 1)
    valid_counts = [k for k in range(MIN_COLOR_GROUPS, max_colors + 1)
                    if any(min_cells <= total <= max_cells for total in range(k, TWO_CELL_SIZE * k + 1))]
    if not valid_counts:
        return None

    k = random.choice(valid_counts)
    valid_totals = [total for total in range(min_cells, max_cells + 1) if k <= total <= TWO_CELL_SIZE * k]
    total_cells = random.choice(valid_totals)
    colors = random.sample(PALETTE_COLORS, k)
    sizes = [TWO_CELL_SIZE] * (total_cells - k) + [SINGLE_CELL_SIZE] * (TWO_CELL_SIZE * k - total_cells)
    random.shuffle(sizes)
    return list(zip(colors, sizes))


def _gen_segment_candidates(n, size):
    """Enumerate strict-lower-triangle one-cell or two-cell segments."""
    if size == SINGLE_CELL_SIZE:
        return [((r, c),) for r in range(1, n) for c in range(r)]

    candidates = []
    for r in range(2, n):
        for c in range(r - 1):
            candidates.append(((r, c), (r, c + 1)))
    for r in range(1, n - 1):
        for c in range(r):
            candidates.append(((r, c), (r + 1, c)))
    return candidates


def _gen_place_segments(n, color_sizes):
    """Place all sampled segments without overlap using randomized backtracking."""
    items = list(color_sizes)
    random.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    candidates_by_size = {size: _gen_segment_candidates(n, size) for size in (SINGLE_CELL_SIZE, TWO_CELL_SIZE)}
    placed = []
    occupied = set()

    def _gen_backtrack(i):
        if i == len(items):
            return True

        color, size = items[i]
        candidates = list(candidates_by_size[size])
        random.shuffle(candidates)
        for coords in candidates:
            if any(coord in occupied for coord in coords):
                continue
            occupied.update(coords)
            placed.append((color, coords))
            if _gen_backtrack(i + 1):
                return True
            placed.pop()
            for coord in coords:
                occupied.remove(coord)
        return False

    if not _gen_backtrack(0):
        return None
    return placed


def _gen_render_grid(n, placed):
    grid = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    np.fill_diagonal(grid, DIAGONAL_COLOR)
    for color, coords in placed:
        rows, cols = zip(*coords)
        grid[rows, cols] = color
    return grid


def _gen_attempt():
    n = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    color_sizes = _gen_color_sizes(n)
    if color_sizes is None:
        return None

    placed = _gen_place_segments(n, color_sizes)
    if placed is None:
        return None
    return _gen_render_grid(n, placed)


def _gen_passes_oracle(inp):
    out = inp.T.copy()
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def generate_input():
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None and _gen_passes_oracle(inp):
            return inp
    raise RuntimeError("failed to generate a valid 9dfd6313 input")
