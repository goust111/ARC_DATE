import numpy as np

from arc_utils import Color, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reduce two same-sized masks separated by a solid divider into the cells empty in both masks.

    Input structure
    - The background is the dominant color.
    - A single full-width separator row has a non-background color that appears only in that row.
    - The separator splits the grid into two equal rectangular masks, each with one foreground color.
    Rule
    - Remove the separator row and align the upper and lower masks cell by cell.
    - Output the same shape as one mask.
    - Paint green where both aligned cells are background; paint background everywhere else.
    Key insight
    - The output is the complement of the union of the two foreground masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = int(get_bg(g))
    colors, counts = np.unique(g, return_counts=True)
    count_by_color = {int(color): int(count) for color, count in zip(colors, counts)}
    assert Color.GREEN not in count_by_color, "green is reserved for the output marker"

    separator_rows = []
    for row in range(height):
        row_color = int(g[row, 0])
        if (
            row_color != bg
            and np.all(g[row, :] == row_color)
            and count_by_color[row_color] == width
        ):
            separator_rows.append(row)
    assert len(separator_rows) == 1, "expected exactly one full-width separator row"

    separator_row = separator_rows[0]
    assert 2 * separator_row + 1 == height, "separator must split equal-height masks"

    upper = g[:separator_row, :]
    lower = g[separator_row + 1 :, :]
    assert upper.shape == lower.shape, "upper and lower masks must align"

    upper_colors = set(map(int, upper.reshape(-1))) - {bg}
    lower_colors = set(map(int, lower.reshape(-1))) - {bg}
    assert len(upper_colors) == 1, "upper mask must have one foreground color"
    assert len(lower_colors) == 1, "lower mask must have one foreground color"

    out = np.full(upper.shape, bg, dtype=int)
    out[(upper == bg) & (lower == bg)] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------

_GEN_RNG = np.random.default_rng()

# Fixed mask geometry and colors.
MASK_HEIGHT, MASK_WIDTH = 4, 4
MASK_SHAPE = (MASK_HEIGHT, MASK_WIDTH)
N_CELLS = MASK_HEIGHT * MASK_WIDTH
BACKGROUND_COLOR, SEPARATOR_COLOR = 0, 4
UPPER_FG_COLOR, LOWER_FG_COLOR = 7, 2
OUTPUT_MARK_COLOR = Color.GREEN

# Output/complement and foreground fill ranges.
GREEN_COUNT_RANGE = (2, 7)  # numpy rng upper bound is exclusive.
UPPER_FILL_RANGE = range(8, 11)
LOWER_FILL_RANGE = range(6, 9)

# Retry budget for rejecting invalid generated candidates.
GENERATION_RETRIES = 300


def _valid_fill_pairs(active_count):
    """Return all upper/lower fill counts compatible with covering active cells."""
    fill_pairs = []
    for upper_count in UPPER_FILL_RANGE:
        for lower_count in LOWER_FILL_RANGE:
            both_count = upper_count + lower_count - active_count
            if 0 <= both_count <= min(upper_count, lower_count):
                fill_pairs.append((upper_count, lower_count))
    return fill_pairs


def _sample_layout(rng):
    """Sample complement cells, active cells, and compatible mask fill counts."""
    green_count = int(rng.integers(*GREEN_COUNT_RANGE))
    green_flats = set(map(int, rng.choice(N_CELLS, size=green_count, replace=False)))
    active_flats = [flat for flat in range(N_CELLS) if flat not in green_flats]
    rng.shuffle(active_flats)
    fill_pairs = _valid_fill_pairs(len(active_flats))
    if not fill_pairs:
        return None
    fill_pair = fill_pairs[int(rng.integers(len(fill_pairs)))]
    return green_flats, active_flats, fill_pair


def _paint_flats(grid, flat_cells, color):
    grid.ravel()[list(flat_cells)] = color


def _build_mask_pair(active_flats, fill_pair):
    """Render the upper/lower masks from a shared shuffled active-cell order."""
    upper_count, lower_count = fill_pair
    both_count = upper_count + lower_count - len(active_flats)
    upper_only_count = upper_count - both_count
    both_flats = active_flats[:both_count]
    upper_only_flats = active_flats[both_count : both_count + upper_only_count]
    lower_only_flats = active_flats[both_count + upper_only_count :]

    upper = np.full(MASK_SHAPE, BACKGROUND_COLOR, dtype=int)
    lower = np.full(MASK_SHAPE, BACKGROUND_COLOR, dtype=int)
    _paint_flats(upper, both_flats, UPPER_FG_COLOR)
    _paint_flats(lower, both_flats, LOWER_FG_COLOR)
    _paint_flats(upper, upper_only_flats, UPPER_FG_COLOR)
    _paint_flats(lower, lower_only_flats, LOWER_FG_COLOR)
    return upper, lower


def _expected_output(green_flats):
    expected = np.full(MASK_SHAPE, BACKGROUND_COLOR, dtype=int)
    _paint_flats(expected, green_flats, OUTPUT_MARK_COLOR)
    return expected


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout(_GEN_RNG)
    if layout is None:
        return None
    green_flats, active_flats, fill_pair = layout

    upper, lower = _build_mask_pair(active_flats, fill_pair)
    inp = np.vstack([upper, np.full((1, MASK_WIDTH), SEPARATOR_COLOR, dtype=int), lower])
    expected = _expected_output(green_flats)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 6430c8c4 input")
