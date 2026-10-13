import numpy as np

from arc_utils import bbox, colors_present, components_mask, get_bg
import random


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    b = bbox(mask)
    assert b is not None, "expected a nonempty mask"
    r0, c0, r1, c1 = b
    return int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _largest_solid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    best: list[tuple[int, int, int, int, int]] = []
    best_area = 0

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0]:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if not mask[r0 : r1 + 1, c0 : c1 + 1].all():
                        continue
                    if area > best_area:
                        best_area = area
                        best = []
                    best.append((area, r0, c0, r1, c1))

    assert best_area > 0, "expected at least one solid rectangle"
    assert len(best) == 1, "movable body must be the unique largest solid rectangle"
    _area, r0, c0, r1, c1 = best[0]
    return r0, c0, r1, c1


def _stem_shift(stem: np.ndarray, body_box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, c0, r1, c1 = body_box
    rows, cols = np.where(stem)
    assert rows.size > 0, "expected a connector stem outside the body"

    unique_rows = set(map(int, rows))
    unique_cols = set(map(int, cols))
    if len(unique_rows) == 1:
        row = int(rows[0])
        cs = sorted(map(int, cols))
        assert cs == list(range(cs[0], cs[-1] + 1)), "horizontal stem must be contiguous"
        assert r0 <= row <= r1, "horizontal stem must attach to the body side"
        if cs[-1] < c0:
            return 0, -1, c0 - cs[0]
        if c1 < cs[0]:
            return 0, 1, cs[-1] - c1
        raise AssertionError("horizontal stem must lie outside the body")

    if len(unique_cols) == 1:
        col = int(cols[0])
        rs = sorted(map(int, rows))
        assert rs == list(range(rs[0], rs[-1] + 1)), "vertical stem must be contiguous"
        assert c0 <= col <= c1, "vertical stem must attach to the body side"
        if rs[-1] < r0:
            return -1, 0, r0 - rs[0]
        if r1 < rs[0]:
            return 1, 0, rs[-1] - r1
        raise AssertionError("vertical stem must lie outside the body")

    raise AssertionError("connector stem must be a straight one-cell-wide segment")


def _touches_anchor(rows: np.ndarray, cols: np.ndarray, anchor_mask: np.ndarray) -> bool:
    h, w = anchor_mask.shape
    for r, c in zip(map(int, rows), map(int, cols)):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and anchor_mask[nr, nc]:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the rectangular body of the non-rectangular foreground object along its straight stem until it touches the rectangular anchor.

    Input structure
    - A black-background grid with exactly two foreground colors.
    - One foreground color is a solid rectangular anchor object.
    - The other foreground color is one connected object made from a unique largest
      solid rectangular body plus a straight one-cell-wide connector stem.
    Rule
    - Erase the whole non-rectangular object.
    - Translate only its rectangular body along the connector direction by the
      connector length, so the body lands next to the anchor.
    Key insight
    - The connector is not kept; it is a path telling how far and in which
      direction the solid body should slide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"
    masks = {color: g == color for color in colors}

    rectangular = {color: _is_solid_rectangle(mask) for color, mask in masks.items()}
    moving_colors = [color for color in colors if not rectangular[color]]
    anchor_colors = [color for color in colors if rectangular[color]]
    assert len(moving_colors) == 1, "expected one non-rectangular moving object"
    assert len(anchor_colors) == 1, "expected one rectangular anchor object"

    moving_color = moving_colors[0]
    anchor_color = anchor_colors[0]
    moving_mask = masks[moving_color]
    assert len(components_mask(moving_mask, connectivity=4)) == 1, "moving object must be 4-connected"
    body_box = _largest_solid_rectangle(moving_mask)
    r0, c0, r1, c1 = body_box

    body_mask = np.zeros_like(moving_mask)
    body_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    stem = moving_mask & ~body_mask
    dr, dc, distance = _stem_shift(stem, body_box)

    body_rows, body_cols = np.where(body_mask)
    new_rows = body_rows + dr * distance
    new_cols = body_cols + dc * distance
    h, w = g.shape
    assert np.all((0 <= new_rows) & (new_rows < h) & (0 <= new_cols) & (new_cols < w)), "moved body must stay in bounds"
    assert _touches_anchor(new_rows, new_cols, masks[anchor_color]), "moved body must land adjacent to the anchor"

    out = g.copy()
    out[moving_mask] = bg
    assert np.all(out[new_rows, new_cols] == bg), "moved body must land on background after erasing its source object"
    out[new_rows, new_cols] = moving_color
    return out

# --------------- generator (o5x) ---------------

# Canvas, margin, and palette bounds.
_GEN_CANVAS_ROWS = (12, 17)
_GEN_CANVAS_COLS = (14, 19)
_GEN_MARGIN = 1
_GEN_COLORS = range(1, 10)

# Rectangle and stem size bounds.
_GEN_ANCHOR_HEIGHTS = range(2, 9)
_GEN_ANCHOR_WIDTHS = range(5, 8)
_GEN_ANCHOR_AREA = (10, 49)
_GEN_BODY_HEIGHTS = range(3, 7)
_GEN_BODY_WIDTHS = range(3, 9)
_GEN_STEM_LENGTHS = range(3, 6)

# Layout choices and retry budget.
_GEN_HORIZONTAL_SIDES = ("left", "right")
_GEN_BODY_FIRST_SIDES = ("left", "above")
_GEN_SIDES = (*_GEN_HORIZONTAL_SIDES, "above", "below")
_GEN_RETRY_LIMIT = 300

_GEN_ANCHOR_SIZES = [
    (rr, cc)
    for rr in _GEN_ANCHOR_HEIGHTS
    for cc in _GEN_ANCHOR_WIDTHS
    if _GEN_ANCHOR_AREA[0] <= rr * cc <= _GEN_ANCHOR_AREA[1]
]


def _gen_start(axis_size: int, run_length: int) -> int:
    return random.randint(_GEN_MARGIN, axis_size - run_length - _GEN_MARGIN)


def _gen_overlap_start(anchor_lo: int, anchor_hi: int, length: int, limit: int) -> int | None:
    lo = max(_GEN_MARGIN, anchor_lo - length + 1)
    hi = min(limit - length - _GEN_MARGIN, anchor_hi)
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _gen_feasible_specs(rows: int, cols: int, side: str) -> list[tuple[int, int, int, int, int]]:
    horizontal = side in _GEN_HORIZONTAL_SIDES
    axis_limit = cols if horizontal else rows
    specs = []
    for anchor_h, anchor_w in _GEN_ANCHOR_SIZES:
        anchor_axis_len = anchor_w if horizontal else anchor_h
        for body_h in _GEN_BODY_HEIGHTS:
            for body_w in _GEN_BODY_WIDTHS:
                body_axis_len = body_w if horizontal else body_h
                for stem_len in _GEN_STEM_LENGTHS:
                    if body_axis_len + stem_len + anchor_axis_len <= axis_limit - 2 * _GEN_MARGIN:
                        specs.append((anchor_h, anchor_w, body_h, body_w, stem_len))
    return specs


def _gen_pick_side_spec(rows: int, cols: int) -> tuple[str, tuple[int, int, int, int, int]] | None:
    side_specs = {side: _gen_feasible_specs(rows, cols, side) for side in _GEN_SIDES}
    feasible_sides = [side for side, specs in side_specs.items() if specs]
    if not feasible_sides:
        return None

    side = random.choice(feasible_sides)
    return side, random.choice(side_specs[side])


def _gen_layout(rows, cols, side, spec):
    """Return (anchor, output body, input body, stem); stem stop is exclusive."""
    anchor_h, anchor_w, body_h, body_w, stem_len = spec
    horizontal = side in _GEN_HORIZONTAL_SIDES
    axis_size = cols if horizontal else rows
    cross_size = rows if horizontal else cols
    anchor_axis_len = anchor_w if horizontal else anchor_h
    body_axis_len = body_w if horizontal else body_h
    anchor_cross_len = anchor_h if horizontal else anchor_w
    body_cross_len = body_h if horizontal else body_w

    run_start = _gen_start(axis_size, body_axis_len + stem_len + anchor_axis_len)
    if side in _GEN_BODY_FIRST_SIDES:
        input_body_axis = run_start
        body_axis = run_start + stem_len
        anchor_axis = run_start + body_axis_len + stem_len
        stem_start = input_body_axis + body_axis_len
    else:
        anchor_axis = run_start
        body_axis = run_start + anchor_axis_len
        input_body_axis = body_axis + stem_len
        stem_start = body_axis

    anchor_cross = _gen_start(cross_size, anchor_cross_len)
    body_cross = _gen_overlap_start(
        anchor_cross,
        anchor_cross + anchor_cross_len - 1,
        body_cross_len,
        cross_size,
    )
    if body_cross is None:
        return None

    lane0 = max(anchor_cross, body_cross)
    lane1 = min(anchor_cross + anchor_cross_len - 1, body_cross + body_cross_len - 1)
    stem_cross = random.randint(lane0, lane1)
    stem = ("h" if horizontal else "v", stem_cross, stem_start, stem_start + stem_len)

    if horizontal:
        return (anchor_cross, anchor_axis), (body_cross, body_axis), (body_cross, input_body_axis), stem
    return (anchor_axis, anchor_cross), (body_axis, body_cross), (input_body_axis, body_cross), stem


def _gen_render_pair(rows, cols, spec, layout, anchor_color, moving_color):
    anchor_h, anchor_w, body_h, body_w, _ = spec
    (anchor_r, anchor_c), (body_r, body_c), (input_body_r, input_body_c), stem = layout
    stem_axis, stem_cross, stem_start, stem_stop = stem

    out = np.zeros((rows, cols), dtype=int)
    inp = np.zeros((rows, cols), dtype=int)
    if stem_axis == "h":
        inp[stem_cross, stem_start:stem_stop] = moving_color
    else:
        inp[stem_start:stem_stop, stem_cross] = moving_color

    out[anchor_r : anchor_r + anchor_h, anchor_c : anchor_c + anchor_w] = anchor_color
    out[body_r : body_r + body_h, body_c : body_c + body_w] = moving_color
    inp[anchor_r : anchor_r + anchor_h, anchor_c : anchor_c + anchor_w] = anchor_color
    inp[input_body_r : input_body_r + body_h, input_body_c : input_body_c + body_w] = moving_color
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*_GEN_CANVAS_ROWS)
    cols = random.randint(*_GEN_CANVAS_COLS)

    picked = _gen_pick_side_spec(rows, cols)
    if picked is None:
        return None
    side, spec = picked
    anchor_color, moving_color = random.sample(_GEN_COLORS, 2)

    layout = _gen_layout(rows, cols, side, spec)
    if layout is None:
        return None

    inp, out = _gen_render_pair(rows, cols, spec, layout, anchor_color, moving_color)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 98cf29f8 input")
