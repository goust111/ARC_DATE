import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste, scale_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the foreground shape to its tight bounding box and magnify that crop by a factor of two.

    Input structure
    A 2D grid with a dominant background color and exactly one non-background color forming a sparse shape.

    Rule
    Find the tight bounding box of all non-background pixels, crop that rectangular region, and replace each cropped pixel with a 2x2 block of the same color.

    Key insight
    The output is the entire foreground bounding-box region enlarged, so background gaps inside the shape are scaled along with the colored pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    colors = set(map(int, np.unique(g)))
    foreground_colors = colors - {bg}
    assert len(foreground_colors) == 1, "expected one foreground color"

    foreground = g != bg
    crop_box = bbox(foreground)
    assert crop_box is not None, "expected at least one foreground pixel"

    return scale_sprite(crop_bbox(g, crop_box), 2)

# --------------- generator (o5x) ---------------

# Canvas, colors, and output scale.
GEN_GRID_SHAPE, GEN_SCALE_FACTOR = (9, 9), 2
GEN_BG_COLOR, GEN_FG_COLOR = 0, 4

# Shape and density bounds.
GEN_BBOX_SIZE_RANGE = range(3, 6)
GEN_FG_COUNT_MIN, GEN_FG_COUNT_MAX = 4, 8
GEN_DENSITY_MIN, GEN_DENSITY_MAX = (2, 5), (7, 10)

# Placement and validation limits.
GEN_MARGIN = 1
GEN_MAX_ATTEMPTS = 300


def _gen_pick_parameters() -> tuple[int, int, int]:
    fg_count = int(np.random.randint(GEN_FG_COUNT_MIN, GEN_FG_COUNT_MAX + 1))
    choices = []
    min_num, min_den = GEN_DENSITY_MIN
    max_num, max_den = GEN_DENSITY_MAX
    for height in GEN_BBOX_SIZE_RANGE:
        for width in GEN_BBOX_SIZE_RANGE:
            area = height * width
            min_count = max(GEN_FG_COUNT_MIN, (min_num * area + min_den - 1) // min_den)
            max_count = min(GEN_FG_COUNT_MAX, (max_num * area) // max_den)
            if min_count <= fg_count <= max_count:
                choices.append((height, width))

    height, width = choices[int(np.random.randint(len(choices)))]
    return height, width, fg_count


def _gen_edge_touching_mask(height: int, width: int, fg_count: int) -> np.ndarray:
    cells = {
        (0, int(np.random.randint(width))),
        (height - 1, int(np.random.randint(width))),
        (int(np.random.randint(height)), 0),
        (int(np.random.randint(height)), width - 1),
    }

    remaining = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in cells
    ]
    need = fg_count - len(cells)
    if need > 0:
        picks = np.random.choice(len(remaining), size=need, replace=False)
        for idx in np.atleast_1d(picks):
            cells.add(remaining[int(idx)])

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_expected_scaled_crop(grid: np.ndarray) -> np.ndarray:
    crop_box = bbox(grid != GEN_BG_COLOR)
    assert crop_box is not None
    return scale_sprite(crop_bbox(grid, crop_box), GEN_SCALE_FACTOR)


def _gen_attempt() -> np.ndarray:
    height, width, fg_count = _gen_pick_parameters()
    mask = _gen_edge_touching_mask(height, width, fg_count)

    grid = np.full(GEN_GRID_SHAPE, GEN_BG_COLOR, dtype=int)
    row_limit = GEN_GRID_SHAPE[0] - GEN_MARGIN - height + 1
    col_limit = GEN_GRID_SHAPE[1] - GEN_MARGIN - width + 1
    r0 = int(np.random.randint(GEN_MARGIN, row_limit))
    c0 = int(np.random.randint(GEN_MARGIN, col_limit))
    patch = np.where(mask, GEN_FG_COLOR, GEN_BG_COLOR)
    paste(grid, patch, r0, c0)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        expected = _gen_expected_scaled_crop(inp)
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            pass
    raise RuntimeError("failed to generate a valid f25fbde4 input")
