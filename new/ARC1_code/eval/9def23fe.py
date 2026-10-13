import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def _paint_bg_cells(out: np.ndarray, rows: slice | int, cols: slice | int, fill: int, bg: int) -> None:
    region = out[rows, cols]
    region[region == int(bg)] = int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a solid rectangle into side stripes, leaving blank the rows or columns marked by control pixels.

    Input structure:
    - The background is the dominant color.
    - The foreground contains one unique largest 4-connected solid rectangle and singleton marker pixels of one other color.
    - Every marker lies in one of the four side strips aligned with the rectangle, not in a diagonal quadrant.

    Rule:
    - Keep the original rectangle and markers unchanged.
    - Above and below the rectangle, fill each rectangle column that has no marker on that side.
    - Left and right of the rectangle, fill each rectangle row that has no marker on that side.

    Key insight:
    - The markers encode the excluded projected rows or columns for their side of the rectangle; the unmarked positions become rectangle-colored extension stripes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert components, "expected foreground components"

    max_area = max(int(comp["area"]) for comp in components)
    rectangles = [comp for comp in components if int(comp["area"]) == max_area]
    assert len(rectangles) == 1, "expected one unique largest rectangle"

    rectangle = rectangles[0]
    fill = int(rectangle["color"])
    r0, c0, r1, c1 = rectangle["bbox"]
    assert int(rectangle["area"]) == bbox_h(rectangle["bbox"]) * bbox_w(rectangle["bbox"]), "expected a solid rectangle"

    markers = [comp for comp in components if comp is not rectangle]
    assert markers, "expected marker pixels"
    assert all(int(marker["area"]) == 1 for marker in markers), "markers must be singleton pixels"
    marker_colors = {int(marker["color"]) for marker in markers}
    assert len(marker_colors) == 1, "expected one marker color"

    marked_top: set[int] = set()
    marked_bottom: set[int] = set()
    marked_left: set[int] = set()
    marked_right: set[int] = set()
    for marker in markers:
        r, c = map(int, marker["coords"][0])
        side_hits = 0
        if r < r0 and c0 <= c <= c1:
            marked_top.add(c)
            side_hits += 1
        if r > r1 and c0 <= c <= c1:
            marked_bottom.add(c)
            side_hits += 1
        if c < c0 and r0 <= r <= r1:
            marked_left.add(r)
            side_hits += 1
        if c > c1 and r0 <= r <= r1:
            marked_right.add(r)
            side_hits += 1
        assert side_hits == 1, "each marker must align with exactly one rectangle side"

    out = g.copy()
    for c in range(c0, c1 + 1):
        if c not in marked_top:
            _paint_bg_cells(out, slice(0, r0), c, fill, bg)
        if c not in marked_bottom:
            _paint_bg_cells(out, slice(r1 + 1, out.shape[0]), c, fill, bg)

    for r in range(r0, r1 + 1):
        if r not in marked_left:
            _paint_bg_cells(out, r, slice(0, c0), fill, bg)
        if r not in marked_right:
            _paint_bg_cells(out, r, slice(c1 + 1, out.shape[1]), fill, bg)

    return out

# --------------- generator (o5x) ---------------
# Canvas and rectangle bounds.
_GEN_GRID_SIZE_BOUNDS_9DEF23FE = (15, 26)
_GEN_RECT_SIDE_BOUNDS_9DEF23FE = (5, 9)
_GEN_RECT_MARGIN_9DEF23FE = 2
# Marker layout and palette.
_GEN_SIDES_9DEF23FE = ("top", "bottom", "left", "right")
_GEN_MARKER_COUNT_9DEF23FE = 8
_GEN_ACTIVE_SIDE_BOUNDS_9DEF23FE = (2, 5)
_GEN_MARKER_MIN_GAP_9DEF23FE = 2
_GEN_MARKER_COLORS_9DEF23FE = (1, 3, 4, 5, 6, 7, 8, 9)
# Render colors and retry budget.
_GEN_BG_COLOR_9DEF23FE, _GEN_FILL_COLOR_9DEF23FE = 0, 2
_GEN_MAX_ATTEMPTS_9DEF23FE = 300


def _gen_side_distributions_9def23fe(total, slots, max_per_side):
    distributions = []

    def rec(remaining, k, prefix):
        if k == 0:
            if remaining == 0: distributions.append(tuple(prefix))
            return
        lo = max(1, remaining - max_per_side * (k - 1))
        hi = min(max_per_side, remaining - (k - 1))
        for value in range(lo, hi + 1):
            prefix.append(value)
            rec(remaining - value, k - 1, prefix)
            prefix.pop()

    rec(total, slots, [])
    return distributions


def _gen_nonadjacent_indices_9def23fe(length, count):
    if count == 0: return []
    base = np.random.choice(np.arange(length - count + 1), size=count, replace=False)
    base.sort()
    return [int(value) + i for i, value in enumerate(base)]


def _gen_choose_layout_9def23fe(h, w, side):
    """Choose rectangle placement plus marked row/column indices per side."""
    r0 = int(np.random.randint(_GEN_RECT_MARGIN_9DEF23FE, h - side - 1))
    c0 = int(np.random.randint(_GEN_RECT_MARGIN_9DEF23FE, w - side - 1))
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    max_per_side = (side + 1) // 2
    side_counts = [k for k in range(*_GEN_ACTIVE_SIDE_BOUNDS_9DEF23FE) if k <= _GEN_MARKER_COUNT_9DEF23FE <= k * max_per_side]
    active_count = int(np.random.choice(side_counts))
    active_sides = list(np.random.choice(_GEN_SIDES_9DEF23FE, size=active_count, replace=False))
    distributions = _gen_side_distributions_9def23fe(_GEN_MARKER_COUNT_9DEF23FE, active_count, max_per_side)
    counts = distributions[int(np.random.randint(len(distributions)))]
    marked = {name: [] for name in _GEN_SIDES_9DEF23FE}
    for name, count in zip(active_sides, counts):
        marked[name] = _gen_nonadjacent_indices_9def23fe(side, count)
    return (r0, c0, r1, c1), marked


def _gen_render_extensions_9def23fe(shape, bbox, marked):
    """Render rectangle plus side extensions, leaving marked rows/columns blank."""
    r0, c0, r1, c1 = bbox
    out = np.zeros(shape, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = _GEN_FILL_COLOR_9DEF23FE
    marked_top = {c0 + idx for idx in marked["top"]}
    marked_bottom = {c0 + idx for idx in marked["bottom"]}
    marked_left = {r0 + idx for idx in marked["left"]}
    marked_right = {r0 + idx for idx in marked["right"]}
    for col in range(c0, c1 + 1):
        if col not in marked_top:
            out[:r0, col] = _GEN_FILL_COLOR_9DEF23FE
        if col not in marked_bottom:
            out[r1 + 1 :, col] = _GEN_FILL_COLOR_9DEF23FE
    for row in range(r0, r1 + 1):
        if row not in marked_left:
            out[row, :c0] = _GEN_FILL_COLOR_9DEF23FE
        if row not in marked_right:
            out[row, c1 + 1 :] = _GEN_FILL_COLOR_9DEF23FE
    return out


def _gen_place_marker_9def23fe(side, idx, r0, c0, r1, c1, h, w):
    if side == "top":
        return int(r0 - np.random.randint(_GEN_MARKER_MIN_GAP_9DEF23FE, r0 + 1)), c0 + idx
    if side == "bottom":
        return int(r1 + np.random.randint(_GEN_MARKER_MIN_GAP_9DEF23FE, h - r1)), c0 + idx
    if side == "left":
        return r0 + idx, int(c0 - np.random.randint(_GEN_MARKER_MIN_GAP_9DEF23FE, c0 + 1))
    return r0 + idx, int(c1 + np.random.randint(_GEN_MARKER_MIN_GAP_9DEF23FE, w - c1))


def _gen_add_markers_9def23fe(out, bbox, marked):
    r0, c0, r1, c1 = bbox
    h, w = out.shape
    marker_color = int(np.random.choice(_GEN_MARKER_COLORS_9DEF23FE))
    for name in _GEN_SIDES_9DEF23FE:
        for idx in marked[name]:
            row, col = _gen_place_marker_9def23fe(name, idx, r0, c0, r1, c1, h, w)
            out[row, col] = marker_color

def _gen_input_from_output_9def23fe(out, bbox):
    """Remove generated extensions while preserving rectangle and markers."""
    r0, c0, r1, c1 = bbox
    inp = out.copy()
    for region in (
        inp[:r0, c0 : c1 + 1],
        inp[r1 + 1 :, c0 : c1 + 1],
        inp[r0 : r1 + 1, :c0],
        inp[r0 : r1 + 1, c1 + 1 :],
    ):
        region[region == _GEN_FILL_COLOR_9DEF23FE] = _GEN_BG_COLOR_9DEF23FE
    return inp


def _gen_attempt():
    h = int(np.random.randint(*_GEN_GRID_SIZE_BOUNDS_9DEF23FE))
    w = int(np.random.randint(*_GEN_GRID_SIZE_BOUNDS_9DEF23FE))
    side = int(np.random.randint(*_GEN_RECT_SIDE_BOUNDS_9DEF23FE))
    bbox, marked = _gen_choose_layout_9def23fe(h, w, side)
    out = _gen_render_extensions_9def23fe((h, w), bbox, marked)
    _gen_add_markers_9def23fe(out, bbox, marked)
    inp = _gen_input_from_output_9def23fe(out, bbox)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(solved, out) else None


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS_9DEF23FE):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid 9def23fe input")
