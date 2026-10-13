import numpy as np

from arc_utils import Color, bbox_coords, components_mask
import random


def _solid_rectangles_for_background(g: np.ndarray, bg: int):
    fg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    if len(fg_colors) != 1:
        return None

    fg = fg_colors[0]
    rectangles = []
    for coords in components_mask(g == fg, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            return None
        if len(coords) != height * width:
            return None
        if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == fg):
            return None
        rectangles.append((r0, c0, r1, c1))

    if not rectangles:
        return None
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each solid rectangle into nested alternating interior layers.

    Input structure
    - The input has one empty background color and one foreground color.
    - Under the correct background choice, the foreground forms one or more
      4-connected filled axis-aligned rectangles.

    Rule
    - Keep each rectangle's outer border in the original foreground color.
    - Recolor pixels at odd positive distance from the rectangle edge red.
    - Recolor pixels at even positive distance from the rectangle edge green.

    Key insight
    - The new colors depend only on each pixel's rectangular layer depth; the
      rectangles themselves do not move or resize.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0, "expected a nonempty grid"

    candidates = []
    for bg in map(int, np.unique(g)):
        rectangles = _solid_rectangles_for_background(g, bg)
        if rectangles is not None:
            candidates.append(rectangles)
    assert len(candidates) == 1, "expected a unique solid-rectangle foreground"

    rectangles = candidates[0]
    out = g.copy()
    for r0, c0, r1, c1 in rectangles:
        rows = np.arange(r0, r1 + 1)[:, None]
        cols = np.arange(c0, c1 + 1)[None, :]
        depth = np.minimum(
            np.minimum(rows - r0, r1 - rows),
            np.minimum(cols - c0, c1 - cols),
        )

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[(depth > 0) & (depth % 2 == 1)] = Color.RED
        region[(depth > 0) & (depth % 2 == 0)] = Color.GREEN

    return out

# --------------- generator (o5x) ---------------

# Colors used by the input and deterministic transform.
BG_COLOR, FG_COLOR, ODD_DEPTH_COLOR, EVEN_DEPTH_COLOR = 0, 1, 2, 3

# Canvas and rectangle sampling bounds.
GRID_MIN_SIZE, GRID_MAX_SIZE = 11, 17
RECT_COUNT_CHOICES = (2, 3)
RECT_MIN_SIDE, RECT_MAX_SIDE, LARGE_RECT_MIN_SIDE = 3, 11, 5

# Rejection bounds and retry budget.
MIN_COVERAGE, MAX_COVERAGE = 0.30, 0.65
GENERATION_RETRIES = 300

def _gen_rect_fits(rect, placed):
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True

def _gen_valid_positions(height, width, rect_h, rect_w, placed):
    positions = []
    for r0 in range(height - rect_h + 1):
        for c0 in range(width - rect_w + 1):
            rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if _gen_rect_fits(rect, placed):
                positions.append((r0, c0))
    return positions

def _gen_sample_rect_dims(height, width, num_rects):
    max_h = min(height, RECT_MAX_SIDE)
    max_w = min(width, RECT_MAX_SIDE)
    dims = [
        (random.randint(RECT_MIN_SIDE, max_h), random.randint(RECT_MIN_SIDE, max_w))
        for _ in range(num_rects)
    ]
    if all(min(rect_h, rect_w) < LARGE_RECT_MIN_SIDE for rect_h, rect_w in dims):
        idx = random.randrange(num_rects)
        dims[idx] = (
            random.randint(LARGE_RECT_MIN_SIDE, max_h),
            random.randint(LARGE_RECT_MIN_SIDE, max_w),
        )
    return dims

def _gen_expected_output(grid, rects):
    out = grid.copy()
    for r0, c0, r1, c1 in rects:
        rows = np.arange(r0, r1 + 1)[:, None]
        cols = np.arange(c0, c1 + 1)[None, :]
        depth = np.minimum(
            np.minimum(rows - r0, r1 - rows),
            np.minimum(cols - c0, c1 - cols),
        )
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[(depth > 0) & (depth % 2 == 1)] = ODD_DEPTH_COLOR
        region[(depth > 0) & (depth % 2 == 0)] = EVEN_DEPTH_COLOR
    return out

def _gen_attempt():
    height = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    num_rects = random.choice(RECT_COUNT_CHOICES)

    dims = _gen_sample_rect_dims(height, width, num_rects)
    order = sorted(range(num_rects), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    placed = []

    for idx in order:
        rect_h, rect_w = dims[idx]
        positions = _gen_valid_positions(height, width, rect_h, rect_w, placed)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = FG_COLOR
        placed.append(rect)

    coverage = np.count_nonzero(grid == FG_COLOR) / grid.size
    if not MIN_COVERAGE <= coverage <= MAX_COVERAGE:
        return None

    expected = _gen_expected_output(grid, placed)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid

def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
