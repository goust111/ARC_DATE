import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by the colored top-row marker whose column passes through it.

    Input structure
    - The grid has a dominant background, grey rectangular 4-connected objects,
      and colored singleton control pixels in the top row.
    - Each grey rectangle has exactly one control-pixel column inside its horizontal span.

    Rule
    - Keep the background and top-row control pixels unchanged.
    - For each 4-connected grey rectangle, find its bounding-box column range.
    - Recolor the whole object with the unique control-pixel color whose column
      lies inside that range.

    Key insight
    - The markers act as vertical labels for grey regions: horizontal-span
      containment, not distance or object size, determines the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    grey = Color.GREY
    control_mask = (g != bg) & (g != grey)
    control_coords = np.argwhere(control_mask)
    assert control_coords.size > 0, "expected top-row control pixels"
    assert np.all(control_coords[:, 0] == 0), "control pixels must be in the top row"

    marker_by_col = {int(c): int(g[0, c]) for _r, c in control_coords}
    assert len(marker_by_col) == len(control_coords), "expected one control pixel per column"

    grey_objects = components_mask(g == grey, connectivity=4)
    assert len(grey_objects) == len(marker_by_col), "expected one grey object per control pixel"

    out = g.copy()
    for obj in grey_objects:
        r0, c0, r1, c1 = bbox_coords(obj)
        assert obj.shape[0] == (r1 - r0 + 1) * (c1 - c0 + 1), "expected solid grey rectangles"
        matches = [color for c, color in marker_by_col.items() if c0 <= c <= c1]
        assert len(matches) == 1, "each grey object must contain exactly one marker column"
        out[obj[:, 0], obj[:, 1]] = matches[0]

    return out

# --------------- generator (o5x) ---------------

# Canvas and colors.
GRID_ROWS = GRID_COLS = 10
GRID_SHAPE = (GRID_ROWS, GRID_COLS)
BACKGROUND_COLOR, GREY_COLOR = 0, 5
MARKER_ROW = 0
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Rectangle sizing and placement.
RECT_COUNT, RECT_MIN_ROW = 3, 2
RECT_HEIGHTS = range(2, 8)
RECT_WIDTHS = range(2, 6)
MAX_TOTAL_RECT_AREA = 48
RECT_TOUCH_OFFSET = 1

# Retry limits.
MAX_GENERATION_ATTEMPTS = 300

SIZE_CHOICES = [(h, w) for h in RECT_HEIGHTS for w in RECT_WIDTHS]
VALID_SIZE_TRIPLES = [
    (a, b, c)
    for a in SIZE_CHOICES
    for b in SIZE_CHOICES
    for c in SIZE_CHOICES
    if a[0] * a[1] + b[0] * b[1] + c[0] * c[1] <= MAX_TOTAL_RECT_AREA
]


def _gen_rect_bounds(rect):
    r0, c0, h, w = rect
    return r0, c0, r0 + h - 1, c0 + w - 1


def _gen_rects_conflict(a, b):
    ar0, ac0, ar1, ac1 = _gen_rect_bounds(a)
    br0, bc0, br1, bc1 = _gen_rect_bounds(b)
    rows_overlap = ar0 <= br1 and br0 <= ar1
    cols_overlap = ac0 <= bc1 and bc0 <= ac1
    if rows_overlap and cols_overlap:
        return True
    if rows_overlap and (ac1 + RECT_TOUCH_OFFSET == bc0 or bc1 + RECT_TOUCH_OFFSET == ac0):
        return True
    if cols_overlap and (ar1 + RECT_TOUCH_OFFSET == br0 or br1 + RECT_TOUCH_OFFSET == ar0):
        return True
    return False


def _gen_marker_columns(rects):
    """Choose one random column per rect that belongs to no other rect span."""
    spans = [(rect[1], rect[1] + rect[3] - 1) for rect in rects]
    marker_cols = []
    for c0, c1 in spans:
        exclusive = [
            c
            for c in range(c0, c1 + 1)
            if sum(s0 <= c <= s1 for s0, s1 in spans) == 1
        ]
        if not exclusive:
            return None
        marker_cols.append(random.choice(exclusive))
    return marker_cols


def _gen_place_rectangles(sizes):
    order = sorted(range(RECT_COUNT), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    rects = [None] * RECT_COUNT

    def backtrack(k):
        if k == len(order):
            return rects[:] if _gen_marker_columns(rects) is not None else None

        idx = order[k]
        h, w = sizes[idx]
        candidates = [
            (r0, c0, h, w)
            for r0 in range(RECT_MIN_ROW, GRID_ROWS + 1 - h)
            for c0 in range(GRID_COLS + 1 - w)
        ]
        random.shuffle(candidates)
        for rect in candidates:
            if all(other is None or not _gen_rects_conflict(rect, other) for other in rects):
                rects[idx] = rect
                placed = backtrack(k + 1)
                if placed is not None:
                    return placed
                rects[idx] = None
        return None

    return backtrack(0)


def _gen_render_pair(rects, colors, marker_cols):
    out = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    for rect, color in zip(rects, colors):
        r0, c0, h, w = rect
        patch = np.full((h, w), color, dtype=int)
        paste(out, patch, r0, c0)
    for c, color in zip(marker_cols, colors):
        out[MARKER_ROW, c] = color

    inp = out.copy()
    for r0, c0, h, w in rects:
        inp[r0 : r0 + h, c0 : c0 + w] = GREY_COLOR
    return inp, out


def _gen_attempt():
    colors = random.sample(MARKER_COLORS, RECT_COUNT)
    sizes = random.choice(VALID_SIZE_TRIPLES)
    rects = _gen_place_rectangles(sizes)
    if rects is None:
        return None
    marker_cols = _gen_marker_columns(rects)
    if marker_cols is None:
        return None

    inp, out = _gen_render_pair(rects, colors, marker_cols)
    try:
        return inp if np.array_equal(main(inp), out) else None
    except Exception:
        return None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ddf7fa4f input")
