import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the red row and column dividers through two marker pixels and fill the marked rectangle interior blue.

    Input structure:
    - The grid has a uniform background and exactly two non-background marker pixels.
    - The two markers occupy different rows and different columns, making opposite corners of a nonempty rectangle.
    - The marker color is preserved as the divider color, while blue is absent from the input and used for the fill.

    Rule:
    - Use the two marker rows and two marker columns as full-width/full-height dividers.
    - Paint those rows and columns with the marker color.
    - Paint the strict interior of the rectangle between the marker rows and columns blue.

    Key insight:
    - The isolated markers are opposite rectangle corners; extending their horizontal and vertical projections partitions the whole grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == Color.BLUE), "blue must not appear in the input"

    bg = get_bg(g)
    marker_colors = colors_present(g, exclude={bg})
    assert len(marker_colors) == 1, "expected exactly one non-background marker color"

    marker = int(marker_colors[0])
    pts = np.argwhere(g == marker)
    assert pts.shape == (2, 2), "expected exactly two marker pixels"

    (r0, c0), (r1, c1) = pts
    rlo, rhi = sorted((int(r0), int(r1)))
    clo, chi = sorted((int(c0), int(c1)))
    assert rlo < rhi and clo < chi, "markers must be diagonal rectangle corners"
    assert rhi - rlo > 1 and chi - clo > 1, "marked rectangle must have a nonempty interior"

    out = np.full_like(g, bg)
    out[[rlo, rhi], :] = marker
    out[:, [clo, chi]] = marker
    out[rlo + 1 : rhi, clo + 1 : chi] = Color.BLUE
    return out

# --------------- generator (o5x) ---------------
# Canvas/marker sampling bounds.
GRID_SIZE_RANGE = (10, 21)
MIN_MARKER_GAP = 3
DIAGONAL_PATTERN_COUNT = 2
# Task colors and retry budget.
BACKGROUND_COLOR, MARKER_COLOR, INTERIOR_COLOR = Color.BLACK, Color.RED, Color.BLUE
GENERATION_RETRIES = 300


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(*GRID_SIZE_RANGE))
    w = int(np.random.randint(*GRID_SIZE_RANGE))

    row_gap = int(np.random.randint(MIN_MARKER_GAP, h))
    col_gap = int(np.random.randint(MIN_MARKER_GAP, w))
    rlo = int(np.random.randint(0, h - row_gap))
    clo = int(np.random.randint(0, w - col_gap))
    rhi = rlo + row_gap
    chi = clo + col_gap

    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    if bool(np.random.randint(0, DIAGONAL_PATTERN_COUNT)):
        grid[rlo, clo] = MARKER_COLOR
        grid[rhi, chi] = MARKER_COLOR
    else:
        grid[rlo, chi] = MARKER_COLOR
        grid[rhi, clo] = MARKER_COLOR

    out = np.full_like(grid, BACKGROUND_COLOR)
    out[[rlo, rhi], :] = MARKER_COLOR
    out[:, [clo, chi]] = MARKER_COLOR
    out[rlo + 1 : rhi, clo + 1 : chi] = INTERIOR_COLOR
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
