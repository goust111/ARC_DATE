import numpy as np

from arc_utils import Color, bbox_coords, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the blue rectangular border spanned by the three marker pixels while preserving the markers.

    Input structure
    - The grid has a uniform background and exactly three same-color foreground marker pixels.
    - The marker pixels lie on the boundary of their shared bounding box.
    - Blue is absent from the input and is the output border color.

    Rule
    - Compute the bounding box of the three marker pixels.
    - Paint the top, bottom, left, and right sides of that box blue.
    - Restore the original marker pixels on top of the blue border.

    Key insight
    - The sparse marker pixels are control points for the missing rectangle; their bounding box is the frame to complete.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == Color.BLUE), "blue must be absent from the input"

    bg = get_bg(g)
    fg = g != bg
    coords = np.argwhere(fg)
    assert coords.shape == (3, 2), "expected exactly three marker pixels"

    marker_color = int(g[coords[0, 0], coords[0, 1]])
    assert marker_color != Color.BLUE, "marker color must be distinct from the border color"
    assert np.all(g[fg] == marker_color), "all marker pixels must have the same color"

    r0, c0, r1, c1 = bbox_coords(coords)
    assert r0 < r1 and c0 < c1, "marker bbox must be a nondegenerate rectangle"
    on_border = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    assert bool(np.all(on_border)), "all markers must lie on the rectangle border"

    out = g.copy()
    out[r0, c0 : c1 + 1] = Color.BLUE
    out[r1, c0 : c1 + 1] = Color.BLUE
    out[r0 : r1 + 1, c0] = Color.BLUE
    out[r0 : r1 + 1, c1] = Color.BLUE
    out[fg] = marker_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and bbox bounds from the task spec.
GRID_HEIGHT_RANGE = (8, 14)
GRID_WIDTH_RANGE = (12, 20)
BBOX_HEIGHT_RANGE = (6, 10)
BBOX_WIDTH_RANGE = (8, 14)
BBOX_MARGIN = 1
MAX_BBOX_ASPECT = 1.75

# Marker color/pattern and retry limit.
MARKER_COLOR = Color.TEAL
CORNER_LABELS = ("tl", "tr", "bl", "br")
GENERATION_RETRIES = 300


def _valid_bbox_sizes(height: int, width: int) -> list[tuple[int, int]]:
    """Return bbox sizes whose margin/aspect constraints match the spec."""
    box_heights = range(
        BBOX_HEIGHT_RANGE[0],
        min(BBOX_HEIGHT_RANGE[1], height - 2 * BBOX_MARGIN) + 1,
    )
    box_widths = range(
        BBOX_WIDTH_RANGE[0],
        min(BBOX_WIDTH_RANGE[1], width - 2 * BBOX_MARGIN) + 1,
    )
    return [
        (box_h, box_w)
        for box_h in box_heights
        for box_w in box_widths
        if max(box_h, box_w) / min(box_h, box_w) <= MAX_BBOX_ASPECT
    ]


def _sample_markers(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Place one corner marker plus two edge-only markers on uncovered sides."""
    corner = random.choice(CORNER_LABELS)
    edge_col = random.randint(c0 + 1, c1 - 1)
    edge_row = random.randint(r0 + 1, r1 - 1)
    corner_row = r0 if corner in ("tl", "tr") else r1
    corner_col = c0 if corner in ("tl", "bl") else c1
    horizontal_row = r1 if corner_row == r0 else r0
    vertical_col = c1 if corner_col == c0 else c0
    return [(corner_row, corner_col), (horizontal_row, edge_col), (edge_row, vertical_col)]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    box_pairs = _valid_bbox_sizes(height, width)
    if not box_pairs:
        return None

    box_h, box_w = random.choice(box_pairs)
    r0 = random.randint(BBOX_MARGIN, height - box_h - BBOX_MARGIN)
    c0 = random.randint(BBOX_MARGIN, width - box_w - BBOX_MARGIN)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1

    markers = _sample_markers(r0, c0, r1, c1)
    grid = np.zeros((height, width), dtype=int)
    rows, cols = zip(*markers)
    grid[rows, cols] = MARKER_COLOR

    expected = grid.copy()
    expected[r0, c0 : c1 + 1] = Color.BLUE
    expected[r1, c0 : c1 + 1] = Color.BLUE
    expected[r0 : r1 + 1, c0] = Color.BLUE
    expected[r0 : r1 + 1, c1] = Color.BLUE
    expected[rows, cols] = MARKER_COLOR

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e7639916 input")
