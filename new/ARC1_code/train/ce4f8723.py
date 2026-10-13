import numpy as np

from arc_utils import Color, colors_present
import math
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse two separator-stacked masks into a green mask of their foreground union.

    Input structure
    - A single full-width separator row has a color that appears only in that row.
    - The separator splits the grid into equal-height upper and lower masks.
    - The two masks share exactly one color, the background, and each has one non-background foreground color.
    Rule
    - Remove the separator row and align the upper and lower masks cell by cell.
    - Paint green where either of the two aligned cells is foreground.
    - Paint the shared background color where both aligned cells are background.
    Key insight
    - The output is the union of the two aligned foreground masks.
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
    out[(upper != bg) | (lower != bg)] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------
# Canvas bounds.
MASK_SIZE_RANGE = (3, 6)
# Output/mask densities and required overlap.
OUTPUT_BG_DENSITY_RANGE = (0.02, 0.40)
MASK_FG_DENSITY_RANGE = (0.38, 0.69)
MIN_OUTPUT_BG_CELLS = 1
MIN_OVERLAP_CELLS = 1
# Palette and retry policy.
INPUT_PALETTE = [c for c in Color.ALL_COLORS if c != Color.GREEN]
MAX_GENERATION_ATTEMPTS = 300


def _gen_split_choices(total_cells, width, bg_count, green_count):
    """Return valid overlap/only-count splits, preferring observed fg densities."""
    density_lo = math.ceil(MASK_FG_DENSITY_RANGE[0] * total_cells)
    density_hi = math.floor(MASK_FG_DENSITY_RANGE[1] * total_cells)
    preferred = []
    fallback = []
    for both_count in range(MIN_OVERLAP_CELLS, green_count + 1):
        remaining = green_count - both_count
        input_bg_count = total_cells + bg_count - both_count
        if input_bg_count == width:
            continue
        for upper_only_count in range(remaining + 1):
            lower_only_count = remaining - upper_only_count
            upper_fg_count = both_count + upper_only_count
            lower_fg_count = both_count + lower_only_count
            if upper_fg_count == width or lower_fg_count == width:
                continue
            choice = (both_count, upper_only_count, lower_only_count)
            fallback.append(choice)
            if density_lo <= upper_fg_count <= density_hi and density_lo <= lower_fg_count <= density_hi:
                preferred.append(choice)
    return preferred or fallback


def _sample_output_mask(shape, bg, bg_count):
    """Sample the union mask and return its green-cell indices for splitting."""
    total_cells = shape[0] * shape[1]
    out = np.full(shape, Color.GREEN, dtype=int)
    all_indices = list(range(total_cells))
    random.shuffle(all_indices)
    out.flat[all_indices[:bg_count]] = bg
    return out, all_indices[bg_count:]


def _render_input_masks(shape, bg, upper_fg, lower_fg, green_indices, both_count, upper_only_count):
    """Render aligned upper/lower masks from sampled union-cell partitions."""
    random.shuffle(green_indices)
    both_indices = green_indices[:both_count]
    upper_only_indices = green_indices[both_count : both_count + upper_only_count]

    upper = np.full(shape, bg, dtype=int)
    lower = np.full(shape, bg, dtype=int)
    upper.flat[both_indices] = upper_fg
    lower.flat[both_indices] = lower_fg
    upper.flat[upper_only_indices] = upper_fg
    lower.flat[green_indices[both_count + upper_only_count :]] = lower_fg
    return upper, lower


def _passes_oracle(inp, out):
    """Verify the sampled input still solves to the sampled output."""
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*MASK_SIZE_RANGE)
    width = random.randint(*MASK_SIZE_RANGE)
    shape = (height, width)
    total_cells = height * width

    bg, upper_fg, lower_fg, separator = random.sample(INPUT_PALETTE, 4)
    min_bg = max(MIN_OUTPUT_BG_CELLS, math.ceil(OUTPUT_BG_DENSITY_RANGE[0] * total_cells))
    max_bg = max(min_bg, math.floor(OUTPUT_BG_DENSITY_RANGE[1] * total_cells))
    bg_count = random.randint(min_bg, max_bg)
    green_count = total_cells - bg_count

    split_choices = _gen_split_choices(total_cells, width, bg_count, green_count)
    if not split_choices:
        return None
    both_count, upper_only_count, _ = random.choice(split_choices)

    out, green_indices = _sample_output_mask(shape, bg, bg_count)
    upper, lower = _render_input_masks(shape, bg, upper_fg, lower_fg, green_indices, both_count, upper_only_count)
    inp = np.vstack([upper, np.full((1, width), separator, dtype=int), lower])

    if not _passes_oracle(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ce4f8723 input")
