import numpy as np

from arc_utils import Color, colors_present


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse two separator-stacked masks into a green mask of their foreground disagreements.

    Input structure
    - A single full-width separator row has a color that appears only in that row.
    - The separator splits the grid into equal-height upper and lower masks.
    - The two masks share exactly one color, the background, and each has one non-background foreground color.
    Rule
    - Remove the separator row and align the upper and lower masks cell by cell.
    - Paint green where exactly one of the two aligned cells is foreground.
    - Paint the shared background color everywhere the two aligned cells agree.
    Key insight
    - The output is the symmetric difference (xor) of the two aligned foreground masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    colors, counts = np.unique(g, return_counts=True)
    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}
    assert Color.GREEN not in count_by_color, "green is reserved for the output marker"

    separator_rows = []
    for row in range(height):
        row_color = int(g[row, 0])
        if np.all(g[row, :] == row_color) and count_by_color[row_color] == width:
            separator_rows.append(row)
    assert len(separator_rows) == 1, "expected exactly one full-width separator row"

    separator_row = separator_rows[0]
    assert 2 * separator_row + 1 == height, "separator must split equal-height masks"

    upper = g[:separator_row, :]
    lower = g[separator_row + 1 :, :]
    assert upper.shape == lower.shape, "upper and lower masks must align"

    upper_colors = set(colors_present(upper))
    lower_colors = set(colors_present(lower))
    shared_colors = upper_colors & lower_colors
    assert len(shared_colors) == 1, "expected one shared background color"
    bg = shared_colors.pop()

    upper_fg = upper_colors - {bg}
    lower_fg = lower_colors - {bg}
    assert len(upper_fg) == 1, "upper mask must have one foreground color"
    assert len(lower_fg) == 1, "lower mask must have one foreground color"

    out = np.full(upper.shape, bg, dtype=int)
    out[(upper != bg) ^ (lower != bg)] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------

# Canvas size bounds.
MASK_HEIGHT_RANGE, MASK_WIDTH_RANGE = (3, 7), (3, 7)

# Density constraints, encoded as exact integer fractions.
OUTPUT_DENSITY_RANGE, FOREGROUND_DENSITY_RANGE = ((3, 10), (7, 10)), ((1, 4), (4, 5))
MIN_OUTPUT_ON_CELLS, MIN_OUTPUT_OFF_CELLS = 1, 1

# Fixed task palette and retry budget.
BG_COLOR, UPPER_FG_COLOR, LOWER_FG_COLOR, OUTPUT_COLOR, SEPARATOR_COLOR = 0, 1, 2, 3, 4
GEN_RETRY_LIMIT = 300


def _count_bounds(cell_count, density_range, min_count=0, max_count=None):
    (min_num, min_den), (max_num, max_den) = density_range
    lower = max(min_count, (min_num * cell_count + min_den - 1) // min_den)
    upper = (max_num * cell_count) // max_den
    return lower, upper if max_count is None else min(max_count, upper)


def _sample_masks(shape):
    cell_count = shape[0] * shape[1]

    min_green, max_green = _count_bounds(cell_count, OUTPUT_DENSITY_RANGE, MIN_OUTPUT_ON_CELLS, cell_count - MIN_OUTPUT_OFF_CELLS)
    if min_green > max_green:
        return None

    green_count = int(np.random.randint(min_green, max_green + 1))
    order = np.random.permutation(cell_count)
    out_mask_flat = np.zeros(cell_count, dtype=bool)
    out_mask_flat[order[:green_count]] = True

    min_fg, max_fg = _count_bounds(cell_count, FOREGROUND_DENSITY_RANGE)
    valid_splits = [
        (upper_only_count, both_count)
        for upper_only_count in range(green_count + 1)
        for both_count in range(cell_count - green_count + 1)
        if min_fg <= upper_only_count + both_count <= max_fg
        and min_fg <= green_count - upper_only_count + both_count <= max_fg
    ]
    if not valid_splits:
        return None

    upper_only_count, both_count = valid_splits[int(np.random.randint(len(valid_splits)))]
    xor_order = np.random.permutation(np.flatnonzero(out_mask_flat))
    same_order = np.random.permutation(np.flatnonzero(~out_mask_flat))

    upper_fg_flat = np.zeros(cell_count, dtype=bool)
    lower_fg_flat = np.zeros(cell_count, dtype=bool)
    upper_fg_flat[xor_order[:upper_only_count]] = True
    lower_fg_flat[xor_order[upper_only_count:]] = True
    upper_fg_flat[same_order[:both_count]] = True
    lower_fg_flat[same_order[:both_count]] = True

    return out_mask_flat.reshape(shape), upper_fg_flat.reshape(shape), lower_fg_flat.reshape(shape)


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*MASK_HEIGHT_RANGE))
    width = int(np.random.randint(*MASK_WIDTH_RANGE))

    sampled = _sample_masks((height, width))
    if sampled is None:
        return None
    out_mask, upper_fg, lower_fg = sampled

    expected = np.full(out_mask.shape, BG_COLOR, dtype=int)
    expected[out_mask] = OUTPUT_COLOR

    grid = np.full((2 * height + 1, width), BG_COLOR, dtype=int)
    grid[:height][upper_fg] = UPPER_FG_COLOR
    grid[height, :] = SEPARATOR_COLOR
    grid[height + 1 :][lower_fg] = LOWER_FG_COLOR

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 99b1bc43 input")
