import numpy as np

from arc_utils import Color, colors_present, components_mask


def main(grid: np.ndarray) -> np.ndarray:
    """Compare the two stacked half-size masks and emit pink where exactly one mask is filled.

    Input structure
    - The input has even height and is vertically partitioned into two equal-size regions.
    - The two regions share one background color, and each region has exactly one foreground color.
    Rule
    - Align the top and bottom regions by position.
    - Output pink at positions where exactly one corresponding input cell is foreground.
    - Output the shared background at positions where both cells agree as background or foreground.
    Key insight
    - The result is the symmetric difference of the two stacked foreground masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert height % 2 == 0

    half = height // 2
    top = g[:half, :]
    bottom = g[half:, :]

    common_colors = set(colors_present(top)) & set(colors_present(bottom))
    assert len(common_colors) == 1
    bg = common_colors.pop()

    top_colors = colors_present(top, exclude={bg})
    bottom_colors = colors_present(bottom, exclude={bg})
    assert len(top_colors) == 1
    assert len(bottom_colors) == 1
    assert top_colors[0] != bottom_colors[0]

    out = np.full((half, width), bg, dtype=int)
    out[(top != bg) ^ (bottom != bg)] = Color.PINK
    return out

# --------------- generator (o5x) ---------------
# Fixed canvas and colors observed in all task examples.
HALF_HEIGHT, WIDTH = 3, 5
TOTAL_CELLS = HALF_HEIGHT * WIDTH
BACKGROUND_COLOR, TOP_COLOR, BOTTOM_COLOR = Color.BLACK, Color.MAROON, Color.YELLOW

# Foreground/overlap ranges and inclusive XOR size bounds.
TOP_COUNT_RANGE, BOTTOM_COUNT_RANGE, OVERLAP_RANGE = range(5, 10), range(8, 11), range(3, 8)
PINK_COUNT_BOUNDS = (5, 9)

# Inclusive connected-component filters and retry cap.
TOP_COMPONENT_BOUNDS, BOTTOM_COMPONENT_BOUNDS = (2, 5), (1, 4)
GENERATION_ATTEMPTS = 300


def _gen_count_choices():
    choices = []
    for top_count in TOP_COUNT_RANGE:
        for bottom_count in BOTTOM_COUNT_RANGE:
            for overlap in OVERLAP_RANGE:
                if overlap > top_count or overlap > bottom_count:
                    continue
                bottom_only_count = bottom_count - overlap
                if bottom_only_count > TOTAL_CELLS - top_count:
                    continue
                pink_count = top_count + bottom_count - 2 * overlap
                if PINK_COUNT_BOUNDS[0] <= pink_count <= PINK_COUNT_BOUNDS[1]:
                    choices.append((top_count, bottom_count, overlap))
    return choices


COUNT_CHOICES = _gen_count_choices()


def _gen_num_components(mask):
    return len(components_mask(mask, connectivity=4))


def _gen_sample_masks(top_count, bottom_count, overlap):
    all_cells = np.arange(TOTAL_CELLS)
    top_cells = np.random.choice(all_cells, size=top_count, replace=False)
    overlap_cells = np.random.choice(top_cells, size=overlap, replace=False)
    free_cells = np.setdiff1d(all_cells, top_cells, assume_unique=False)
    bottom_only_cells = np.random.choice(
        free_cells, size=bottom_count - overlap, replace=False
    )

    top_mask = np.zeros(TOTAL_CELLS, dtype=bool)
    bottom_mask = np.zeros(TOTAL_CELLS, dtype=bool)
    top_mask[top_cells] = True
    bottom_mask[overlap_cells] = True
    bottom_mask[bottom_only_cells] = True
    return top_mask.reshape(HALF_HEIGHT, WIDTH), bottom_mask.reshape(HALF_HEIGHT, WIDTH)


def _gen_render(top_mask, bottom_mask):
    top = np.where(top_mask, TOP_COLOR, BACKGROUND_COLOR)
    bottom = np.where(bottom_mask, BOTTOM_COLOR, BACKGROUND_COLOR)
    expected = np.where(top_mask ^ bottom_mask, Color.PINK, BACKGROUND_COLOR)
    return np.vstack([top, bottom]).astype(int), expected.astype(int)


def _gen_attempt():
    top_count, bottom_count, overlap = COUNT_CHOICES[
        np.random.randint(len(COUNT_CHOICES))
    ]
    top_mask, bottom_mask = _gen_sample_masks(top_count, bottom_count, overlap)
    top_components = _gen_num_components(top_mask)
    bottom_components = _gen_num_components(bottom_mask)
    top_ok = TOP_COMPONENT_BOUNDS[0] <= top_components <= TOP_COMPONENT_BOUNDS[1]
    bottom_ok = BOTTOM_COMPONENT_BOUNDS[0] <= bottom_components <= BOTTOM_COMPONENT_BOUNDS[1]
    if not (top_ok and bottom_ok):
        return None

    inp, expected = _gen_render(top_mask, bottom_mask)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 31d5ba1a input")
