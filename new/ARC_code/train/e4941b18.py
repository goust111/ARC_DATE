import numpy as np

from arc_utils import Color, bbox, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Move red into teal's top marker position, then move teal to the bottom outside corner of the grey rectangle.

    Input structure
    - The dominant background surrounds one filled rectangular non-background block.
    - There is exactly one red marker and one teal marker, both in the row immediately above the rectangle.
    - Both markers are horizontally aligned over columns inside the rectangle.

    Rule
    - Clear the original red marker.
    - Recolor the original teal marker cell red.
    - Place the teal marker beside the rectangle's bottom row, on the side indicated by the horizontal direction from red to teal.

    Key insight
    - The red-to-teal horizontal order chooses the rectangle side where teal exits at the bottom.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    bg = get_bg(g)

    red_coords = np.argwhere(g == Color.RED)
    teal_coords = np.argwhere(g == Color.TEAL)
    assert len(red_coords) == 1, "expected exactly one red marker"
    assert len(teal_coords) == 1, "expected exactly one teal marker"
    red_r, red_c = map(int, red_coords[0])
    teal_r, teal_c = map(int, teal_coords[0])

    rect_mask = (g != bg) & (g != Color.RED) & (g != Color.TEAL)
    rect_colors = set(map(int, g[rect_mask]))
    assert len(rect_colors) == 1, "expected one rectangle color besides the markers"

    rect_bbox = bbox(rect_mask)
    assert rect_bbox is not None, "expected a filled rectangle"
    r0, c0, r1, c1 = rect_bbox
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    assert int(rect_mask.sum()) == rect_h * rect_w, "rectangle must fill its bbox"
    assert np.all(rect_mask[r0 : r1 + 1, c0 : c1 + 1]), "rectangle bbox must be solid"

    assert red_r == teal_r == r0 - 1, "markers must sit directly above the rectangle"
    assert c0 <= red_c <= c1 and c0 <= teal_c <= c1, "markers must align over rectangle columns"
    assert red_c != teal_c, "red and teal markers must be horizontally separated"

    side = 1 if teal_c > red_c else -1
    exit_r = r1
    exit_c = c1 + 1 if side > 0 else c0 - 1
    assert 0 <= exit_r < H and 0 <= exit_c < W, "teal exit cell must be inside the grid"
    assert g[exit_r, exit_c] == bg, "teal exit cell must be background"

    out = g.copy()
    out[red_r, red_c] = bg
    out[teal_r, teal_c] = Color.RED
    out[exit_r, exit_c] = Color.TEAL
    return out

# --------------- generator (o5x) ---------------

# Canvas/object sampling bounds.
GRID_SIZE_RANGE = (7, 19)
RECT_WIDTH_RANGE = (3, 17)
RECT_HEIGHT_RANGE = (4, 9)
RECT_AREA_RESERVED_CELLS = 3
RECT_AREA_DIVISOR = 2

# Marker/layout placement settings.
MARKER_SEPARATION_RANGE = (2, 5)
RECT_MAX_WIDTH_MARGIN = 2
RECT_MAX_HEIGHT_MARGIN = 3
ADJACENT_CELL_OFFSET = 1
EXIT_SIDES = [-1, 1]
MAX_GENERATION_ATTEMPTS = 300

# Fixed task colors.
BACKGROUND_COLOR = Color.ORANGE
RECTANGLE_COLOR = Color.GREY

def _gen_rect_dims(n: int) -> tuple[int, int]:
    pairs = []
    max_w = min(RECT_WIDTH_RANGE[1], n - RECT_MAX_WIDTH_MARGIN)
    max_h = min(RECT_HEIGHT_RANGE[1], n - RECT_MAX_HEIGHT_MARGIN)
    max_rect_area = (n * n - RECT_AREA_RESERVED_CELLS) // RECT_AREA_DIVISOR
    for h in range(RECT_HEIGHT_RANGE[0], max_h + 1):
        for w in range(RECT_WIDTH_RANGE[0], max_w + 1):
            if w * h <= max_rect_area:
                pairs.append((h, w))
    return random.choice(pairs)

def _gen_marker_columns(c0: int, c1: int, exit_side: int) -> tuple[int, int]:
    width = c1 - c0 + 1
    min_sep, max_sep = MARKER_SEPARATION_RANGE
    sep = random.randint(min_sep, min(max_sep, width - ADJACENT_CELL_OFFSET))
    if exit_side > 0:
        red_offset = random.randint(0, width - sep - 1)
        teal_offset = red_offset + sep
    else:
        teal_offset = random.randint(0, width - sep - 1)
        red_offset = teal_offset + sep
    return c0 + red_offset, c0 + teal_offset

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*GRID_SIZE_RANGE)

    rect_h, rect_w = _gen_rect_dims(n)
    valid_sides = []
    if rect_w <= n - RECT_MAX_WIDTH_MARGIN:
        valid_sides = EXIT_SIDES
    if not valid_sides:
        return None

    exit_side = random.choice(valid_sides)
    if exit_side > 0:
        c0 = random.randint(0, n - rect_w - ADJACENT_CELL_OFFSET)
    else:
        c0 = random.randint(ADJACENT_CELL_OFFSET, n - rect_w)
    c1 = c0 + rect_w - 1
    r0 = n - rect_h
    r1 = n - 1
    marker_r = r0 - ADJACENT_CELL_OFFSET
    exit_c = c1 + ADJACENT_CELL_OFFSET if exit_side > 0 else c0 - ADJACENT_CELL_OFFSET

    red_c, teal_c = _gen_marker_columns(c0, c1, exit_side)

    out = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = RECTANGLE_COLOR
    out[marker_r, teal_c] = Color.RED
    out[r1, exit_c] = Color.TEAL

    inp = out.copy()
    inp[marker_r, teal_c] = Color.TEAL
    inp[marker_r, red_c] = Color.RED
    inp[r1, exit_c] = BACKGROUND_COLOR

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, out):
        return inp
    return None

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e4941b18 input")
