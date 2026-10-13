from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the single foreground rectangle and swap its two non-background colors.

    Input structure:
    - A dominant background surrounds one fully occupied rectangular foreground region.
    - The foreground rectangle contains exactly two non-background colors and no background holes.

    Rule:
    - Crop to the bounding box of all non-background pixels.
    - In that crop, replace each foreground color with the other foreground color.

    Key insight:
    - The output is the compact foreground rectangle after a two-color inversion, not the original canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    fg_mask = g != bg
    rect_bbox = bbox(fg_mask)
    assert rect_bbox is not None, "expected foreground pixels"

    crop = crop_bbox(g, rect_bbox).copy()
    assert np.all(crop != bg), "foreground bbox must be fully occupied"

    fg_colors = colors_present(crop, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly two foreground colors"
    first, second = fg_colors

    out = crop.copy()
    out[crop == first] = second
    out[crop == second] = first
    return out

# --------------- generator (o5x) ---------------
# Square geometry, canvas bounds, palette, and retries.
OUTER_SIDE_RANGE = (3, 8)
INNER_SIDE_MIN, MIN_CENTER_MARGIN = 1, 1
BACKGROUND_COLOR, PLACEMENT_MARGIN = 0, 1
CANVAS_DIM_BOUNDS, CANVAS_EXTRA_BOUNDS = (10, 16), (2, 10)
FOREGROUND_COLORS, FOREGROUND_COLOR_COUNT = np.arange(1, 10), 2
MAX_GEN_ATTEMPTS = 300

def _centered_square(side: int, inner: int, fill: int, center: int) -> np.ndarray:
    pad = (side - inner) // 2
    grid = np.full((side, side), fill, dtype=int)
    grid[pad : pad + inner, pad : pad + inner] = center
    return grid

def _valid_canvas_sizes(side: int, inner: int) -> list[tuple[int, int]]:
    outer_count = side * side - inner * inner
    inner_count = inner * inner
    min_dim = max(CANVAS_DIM_BOUNDS[0], side + CANVAS_EXTRA_BOUNDS[0])
    max_dim = min(CANVAS_DIM_BOUNDS[1], side + CANVAS_EXTRA_BOUNDS[1])
    return [
        (h, w)
        for h in range(min_dim, max_dim + 1)
        for w in range(min_dim, max_dim + 1)
        if h * w - side * side > max(outer_count, inner_count)
    ]

def _gen_attempt() -> np.ndarray | None:
    side = int(np.random.randint(OUTER_SIDE_RANGE[0], OUTER_SIDE_RANGE[1] + 1))
    inner_options = [
        inner
        for inner in range(INNER_SIDE_MIN, side - MIN_CENTER_MARGIN)
        if (side - inner) % 2 == 0 and inner * inner < side * side - inner * inner
    ]
    inner = int(np.random.choice(inner_options))
    colors = np.random.choice(FOREGROUND_COLORS, size=FOREGROUND_COLOR_COUNT, replace=False)
    majority, minority = map(int, colors)

    expected = _centered_square(side, inner, minority, majority)
    fg = _centered_square(side, inner, majority, minority)
    valid_sizes = _valid_canvas_sizes(side, inner)
    if not valid_sizes:
        return None

    height, width = valid_sizes[int(np.random.randint(len(valid_sizes)))]
    canvas = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    row = int(np.random.randint(PLACEMENT_MARGIN, height - side - PLACEMENT_MARGIN + 1))
    col = int(np.random.randint(PLACEMENT_MARGIN, width - side - PLACEMENT_MARGIN + 1))
    paste(canvas, fg, row, col)

    try:
        actual = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas

def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b94a9452 input")
