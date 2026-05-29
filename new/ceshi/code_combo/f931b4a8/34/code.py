# Combo variant of f931b4a8 (index 34)
# combo_spec: D1/0+D10/2
# Name: 数字形状识别+外接矩形编码

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color, bbox, crop_bbox


# Design analysis:
# a_concretization_and_trace:
#   - The lower half keeps the original task: lower-left is a non-black fill tile, lower-right
#     is either an all-black tile or a stencil whose black cells are replaced by coarse colors
#     from lower-left.
#   - The fused change is entirely in the two control quadrants. Each control object is now a
#     readable Arabic digit drawn as a seven-segment-style bitmap, but the bitmap is stretched
#     so that its outer bbox also carries the size cue:
#       * upper-left: digit d in {5,6,7,8}, rendered inside a bbox of height d and width 3;
#       * upper-right: digit d in {5,6,7,8}, rendered inside a bbox of height 5 and width d.
#   - Example: if the upper-left quadrant shows a tall "6" occupying a 6x3 bbox, the original
#     solver would count many foreground cells (for this font usually > 10), while the fused
#     solver decodes the shape as 6. Measuring bbox height also gives 6, so D10 and D1 are
#     attached to the same visible object.
#   - This makes the difference from the seed task explicit: count != size, and the top-half
#     geometry is no longer arbitrary scatter.
# b_complexity_calibration:
#   - A literal two-digit font cannot simultaneously satisfy D1 in an 8x8 control quadrant,
#     because bbox-based size encoding is capped by quadrant size. The fusion therefore narrows
#     the controllable output range to {5,6,7,8}, which still provides repeated support across
#     several examples without making the cue microscopic.
#   - Using a seven-segment family keeps the digits visually obvious while allowing axis-specific
#     stretching. This is more structured than random sparse pixels and avoids ambiguity between
#     bbox extraction and numeral recognition.
# c_hidden_assumption_audit:
#   - Original lines `np.count_nonzero(upper_left != bg)` and `np.count_nonzero(upper_right != bg)`
#     assume multiplicity is the only signal. That assumption is invalid here; the solver must
#     crop the foreground sprite and match its shape.
#   - The lower-half tiling loops assume decoded dimensions are positive and not larger than the
#     quadrant side when coarse fill indices are used. The generator enforces output sizes 5..8
#     and fixes control quadrants to 8x8, so those assumptions remain valid.
#   - Matching by exact bitmap shape introduces a new assumption: each control quadrant must
#     contain exactly one translated control glyph. The generator enforces that by construction
#     and validates every sample by running `main()`.
# d_placement_legality:
#   - The input grid is fixed to 16x16, hence each quadrant is 8x8. A height-coded glyph has
#     shape d x 3 and a width-coded glyph has shape 5 x d, so every control glyph fits with room
#     for random translation. Lower-half tiles stay confined to their original quadrants.
# e_visual_observability:
#   - A human sees literal digits in the top half, not anonymous blobs. Across examples, the same
#     output size would correspond to different foreground counts than the seed task, so counting
#     is contradicted by the visuals.
#   - The bbox cue is also visible: the left digit is vertically stretched to its own value and
#     the right digit is horizontally stretched to its own value.
# f_visual_cleanliness:
#   - Each quadrant has one role and one dominant object family. The top controls are isolated on
#     black backgrounds; the bottom tiles remain regular tiling textures, so there is no overlap
#     between semantic layers.
# g_color_change_inference:
#   - Colors do not identify digits. Recognition is done on a binary foreground mask after cropping.
#     Control colors are sampled from a pool disjoint from the output colors so the top-half colors
#     never leak into the output and cannot be mistaken as semantic labels.
# h_variant_metric_for_post_checks:
#   - Useful metrics are:
#       1. `count_gap = foreground_count - decoded_digit`, which must be non-zero to rule out the
#          original counting solver.
#       2. `bbox_match = decoded_digit == bbox_height_or_width`, which must hold to show D1 is also
#          active on the same control object.
# i_new_solver_assumptions:
#   - Upper-left controls always match one mask from `VERTICAL_NUMBER_MASKS`.
#   - Upper-right controls always match one mask from `HORIZONTAL_NUMBER_MASKS`.
#   - When the stencil is non-empty, the visible output window contains both stencil foreground and
#     stencil holes, so both bottom quadrants influence the answer in those samples.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)
CONTROL_Q = 8
CONTROL_SINGLE_COLOR_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
CONTROL_VALUES = (5, 6, 7, 8)
VERTICAL_GLYPH_WIDTH = 3
HORIZONTAL_GLYPH_HEIGHT = 5
STENCIL_EMPTY_PROB = 0.28
LL_SOLID_PROB = 0.72
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
OUTPUT_DIM_MIN = min(CONTROL_VALUES)
OUTPUT_DIM_MAX = max(CONTROL_VALUES)
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


DIGIT_SEGMENTS = {
    5: ("a", "f", "g", "c", "d"),
    6: ("a", "f", "g", "e", "c", "d"),
    7: ("a", "b", "c"),
    8: ("a", "b", "c", "d", "e", "f", "g"),
}


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    assert b is not None
    return crop_bbox(mask.astype(bool), b)


def _bbox_dims(tile: np.ndarray, bg: int = BLACK) -> tuple[int, int]:
    b = bbox(tile != bg)
    assert b is not None
    r0, c0, r1, c1 = b
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _seven_segment_mask(height: int, width: int, digit: int) -> np.ndarray:
    assert height >= 5 and width >= 3
    mask = np.zeros((height, width), dtype=bool)
    mid = height // 2
    upper_rows = range(1, mid)
    lower_rows = range(mid + 1, height)

    def draw(segment: str) -> None:
        if segment == "a":
            mask[0, :] = True
        elif segment == "b":
            mask[list(upper_rows), width - 1] = True
        elif segment == "c":
            mask[list(lower_rows), width - 1] = True
        elif segment == "d":
            mask[height - 1, :] = True
        elif segment == "e":
            mask[list(lower_rows), 0] = True
        elif segment == "f":
            mask[list(upper_rows), 0] = True
        elif segment == "g":
            mask[mid, :] = True
        else:
            raise AssertionError(f"unknown segment {segment}")

    for segment in DIGIT_SEGMENTS[digit]:
        draw(segment)
    return _trim_mask(mask)


VERTICAL_NUMBER_MASKS = {
    value: _seven_segment_mask(value, VERTICAL_GLYPH_WIDTH, value)
    for value in CONTROL_VALUES
}
HORIZONTAL_NUMBER_MASKS = {
    value: _seven_segment_mask(HORIZONTAL_GLYPH_HEIGHT, value, value)
    for value in CONTROL_VALUES
}

assert all(mask.shape == (value, VERTICAL_GLYPH_WIDTH) for value, mask in VERTICAL_NUMBER_MASKS.items())
assert all(mask.shape == (HORIZONTAL_GLYPH_HEIGHT, value) for value, mask in HORIZONTAL_NUMBER_MASKS.items())
assert all(int(mask.sum()) != value for value, mask in VERTICAL_NUMBER_MASKS.items())
assert all(int(mask.sum()) != value for value, mask in HORIZONTAL_NUMBER_MASKS.items())


def _decode_control_number(tile: np.ndarray, axis: str) -> int:
    fg = tile != BLACK
    b = bbox(fg)
    assert b is not None
    sprite = crop_bbox(fg.astype(bool), b)
    library = VERTICAL_NUMBER_MASKS if axis == "height" else HORIZONTAL_NUMBER_MASKS
    for value, target in library.items():
        if sprite.shape == target.shape and np.array_equal(sprite, target):
            return value
    raise AssertionError("unrecognized control numeral")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the two upper control quadrants as stretched digit glyphs, then build the output from the lower texture pair.

    Input structure:
    - The input is a 16x16 square split into four 8x8 quadrants with black as background.
    - The upper-left quadrant contains one vertically stretched Arabic digit; the upper-right contains one horizontally stretched Arabic digit.
    - The lower-left quadrant is a non-black fill tile, and the lower-right quadrant is either an all-black tile or a stencil with black holes.

    Rule:
    - Crop the foreground sprite in the upper-left quadrant and recognize which digit shape it forms; that digit is the output height.
    - Crop the foreground sprite in the upper-right quadrant and recognize which digit shape it forms; that digit is the output width.
    - The recognized glyphs are also stretched so their foreground bboxes agree with those dimensions: left bbox height = output height, right bbox width = output width.
    - If the lower-right tile is fully black, repeat the lower-left tile across the output.
    - Otherwise repeat the lower-right stencil; whenever a repeated stencil cell is black, replace it with the coarse color from the lower-left tile at the corresponding tile-block position.

    Key insight:
    - The top half is no longer a pure counting puzzle: each control object must be read as a digit shape, even though that same object also exposes a matching bbox-size cue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w == 16 and h % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw == CONTROL_Q

    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = _decode_control_number(upper_left, axis="height")
    out_w = _decode_control_number(upper_right, axis="width")
    bbox_h, _ = _bbox_dims(upper_left, bg)
    _, bbox_w = _bbox_dims(upper_right, bg)
    assert bbox_h == out_h and bbox_w == out_w

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == bg:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "solid":
        tile[:] = palette[0]
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = palette[(c + offset) % k]
    else:
        offset = random.randrange(k)
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [c for c in palette if c not in existing]
    if missing:
        fix_coords = list(required_coords) if required_coords is not None else [
            (r, c) for r in range(q) for c in range(q)
        ]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t:q - t, t:q - t] = False
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = _random_color_array((q,), palette)
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = _random_color_array((q,), palette)
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    return tile


def _control_palette(max_colors: int, pool: list[int]) -> list[int]:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, min(max_colors, len(pool)))
    return _palette(k, pool)


def _make_control(q: int, value: int, axis: str, max_colors: int, pool: list[int]) -> np.ndarray:
    assert q == CONTROL_Q
    palette = _control_palette(max_colors, pool)
    mask = VERTICAL_NUMBER_MASKS[value] if axis == "height" else HORIZONTAL_NUMBER_MASKS[value]
    mh, mw = mask.shape
    assert mh <= q and mw <= q
    r0 = random.randint(0, q - mh)
    c0 = random.randint(0, q - mw)
    tile = np.full((q, q), BLACK, dtype=int)
    for dr, dc in np.argwhere(mask):
        tile[r0 + int(dr), c0 + int(dc)] = random.choice(palette)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    assert q == CONTROL_Q
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    if stencil_empty:
        ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    else:
        ll_k = 1
    ll_palette = _palette(ll_k)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    valid_dims = [
        (h, w)
        for h in CONTROL_VALUES
        for w in CONTROL_VALUES
        if h * w <= OUTPUT_AREA_MAX
    ]
    out_h, out_w = random.choice(valid_dims)

    used_fill = (
        [(r, c) for r in range(out_h) for c in range(out_w)]
        if stencil_empty
        else [(0, 0)]
    )
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, "height", UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, "width", UR_MAX_COLORS, control_pool)

    if not stencil_empty:
        visible_stencil = stencil[:out_h, :out_w]
        if not np.any(visible_stencil != BLACK):
            return None
        if not np.any(visible_stencil == BLACK):
            return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    ul_decoded = _decode_control_number(upper_left, "height")
    ur_decoded = _decode_control_number(upper_right, "width")
    ul_bbox_h, _ = _bbox_dims(upper_left)
    _, ur_bbox_w = _bbox_dims(upper_right)
    if ul_decoded != out_h or ur_decoded != out_w:
        return None
    if ul_bbox_h != out_h or ur_bbox_w != out_w:
        return None
    if np.count_nonzero(upper_left != BLACK) == out_h:
        return None
    if np.count_nonzero(upper_right != BLACK) == out_w:
        return None

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
