# Combo variant of f931b4a8 (index 3)
# combo_spec: D1/0+D10/0
# Name: 外接矩形编码+空间排列编码相位

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the combo D1/0 + D10/0 variant
# ---------------------------------------------------
# a) Concrete behavior and trace:
#    - The lower half keeps the original task: lower-left is an all-non-black fill tile,
#      lower-right is a stencil tile whose black cells are replaced by coarse colors.
#    - The top half now carries two meanings at once.
#      1. D1/0: upper-left foreground bbox height = output height; upper-right bbox width = output width.
#      2. D10/0: the same bbox is intentionally anchored on one side of its unused axis, and that anchor sets the
#         tiling phase. A left-anchored upper-left band means "start from the middle column of the repeated lower
#         pattern"; a right-anchored upper-left band means "start from column 0". Likewise, a top-anchored upper-right
#         band means "start from the middle row"; a bottom-anchored one means "start from row 0".
#    - Small q=4 trace:
#      upper-left active cells might span rows 1..3 and cols 0..1 with 5 total pixels. Old task: height=5.
#      D1-only: height=bbox_h=3. Combo: still height=3, but because the band hugs the left wall, col_phase=2.
#      upper-right might span rows 0..1 and cols 1..3 with 5 total pixels. D1-only: width=3. Combo: width=3 and,
#      because the band hugs the top wall, row_phase=2. The output is therefore a 3x3 crop of the bottom pattern
#      starting from stencil/fill coordinates (2,2), not from (0,0).
#
# b) Complexity calibration:
#    - The new freedom is deliberately only binary on each phase axis: left vs right, top vs bottom. That is enough to
#      make the phase visible without introducing too many cases.
#    - Control shapes are sparse bands whose measured span is easy to read as a bbox, while their unused-axis span is
#      narrow and edge-anchored, so the offset cue is visually clean instead of looking like random scatter.
#    - The generator forces the non-black pixel count to exceed the bbox dimension on both axes, so samples cannot
#      silently collapse back to the original count-based task.
#
# c) Hidden-assumption audit against the original solver:
#    - `np.count_nonzero(upper_left != bg)` and the symmetric width line are invalid now. Foreground multiplicity is no
#      longer the size signal; spatial extent is.
#    - The old tiling loops assumed a zero-phase origin. That is also invalid: both the fine stencil lookup and the
#      coarse fill-block lookup must be phase-shifted together.
#    - The old code implicitly allowed arbitrarily large output sizes up to q*q. This combo instead caps each output
#      dimension at q so the edge-anchor cue stays readable and the shifted coarse lookup remains within the qxq fill
#      tile. The generator enforces that cap.
#
# d) Placement legality:
#    - Each control band stays entirely inside its own quadrant. The measured span is chosen first, then the band is
#      anchored either to the left/right edge or top/bottom edge on the unused axis. No tie cases are generated.
#    - Lower-left remains all non-black. Lower-right may be empty or partially black, exactly as in the source task.
#    - Because `out_h, out_w <= q` and each phase is either 0 or q//2, shifted coarse indices stay in {0,1}, which is
#      safely inside the qxq fill tile.
#
# e) Visual observability:
#    - Across samples, a human can see that the output size follows the visible row/column span rather than raw pixel
#      count, and that edge-hugging the control band moves the bottom pattern's starting point.
#    - The generator rejects any sample whose combo output equals the zero-phase D1-only output, so the phase signal is
#      guaranteed to matter in the final grid.
#
# f) Visual cleanliness:
#    - Each upper quadrant contains one sparse band on black background; the lower quadrants keep the original tidy
#      tile/stencil roles. There is no overlap between object types or extra annotation layer.
#
# g) Post-check metrics and new solver assumptions:
#    - Metric 1: `ul_count - out_h` and `ur_count - out_w` must both be positive, proving the bbox rule differs from
#      count-based sizing.
#    - Metric 2: `main(grid)` must differ from the same grid solved with zero phase, proving the offset cue matters.
#    - New solver assumptions:
#      1. each control quadrant has at least one foreground cell;
#      2. the upper-left band's horizontal anchor is unambiguous;
#      3. the upper-right band's vertical anchor is unambiguous;
#      4. lower-left has no black cells.
#      The generator constructs and validates all four conditions explicitly.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)                   # q in {4, 8}; q=2 is too cramped for bbox+anchor cues
STENCIL_EMPTY_PROB = 0.18
LL_SOLID_PROB = 0.55
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.72
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_BAND_SPAN_MAX = 4
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _bbox_info(tile: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(tile != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1)


def _decode_col_phase(control: np.ndarray, bg: int) -> int:
    q = control.shape[1]
    _r0, c0, _h, w = _bbox_info(control, bg)
    left_gap = c0
    right_gap = q - (c0 + w)
    assert left_gap != right_gap
    return q // 2 if left_gap < right_gap else 0


def _decode_row_phase(control: np.ndarray, bg: int) -> int:
    q = control.shape[0]
    r0, _c0, h, _w = _bbox_info(control, bg)
    top_gap = r0
    bottom_gap = q - (r0 + h)
    assert top_gap != bottom_gap
    return q // 2 if top_gap < bottom_gap else 0


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    row_phase: int,
    col_phase: int,
) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    sr = (rr + row_phase) % q
    sc = (cc + col_phase) % q

    if not np.any(stencil != BLACK):
        return fill_tile[sr, sc].copy()

    tiled = stencil[sr, sc].copy()
    bg_mask = tiled == BLACK
    fill_r = (rr + row_phase) // q
    fill_c = (cc + col_phase) // q
    assert int(fill_r.max(initial=0)) < q
    assert int(fill_c.max(initial=0)) < q
    tiled[bg_mask] = fill_tile[fill_r[bg_mask], fill_c[bg_mask]]
    return tiled


def _zero_phase_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    return _expected_output(fill_tile, stencil, out_h, out_w, 0, 0)


def _legacy_count_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    upper_left: np.ndarray,
    upper_right: np.ndarray,
) -> np.ndarray:
    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    return _expected_output(fill_tile, stencil, out_h, out_w, 0, 0)


def _phased_stencil_view(
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    row_phase: int,
    col_phase: int,
) -> np.ndarray:
    q = stencil.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    return stencil[(rr + row_phase) % q, (cc + col_phase) % q]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the upper control bands by their bounding spans and edge anchors, then phase-shift the lower pattern pair.

    Input structure:
    - The input is an even square split into four equal quadrants with black as background.
    - The upper-left and upper-right quadrants are sparse control bands made of non-black pixels.
    - The lower-left quadrant is a non-black color tile, and the lower-right quadrant is a stencil tile that may have black holes or may be entirely black.

    Rule:
    - Output height is the height of the foreground bounding box in the upper-left quadrant, and output width is the width of the foreground bounding box in the upper-right quadrant.
    - The upper-left band's horizontal anchor also sets the column phase: left-anchored means start from the stencil/fill midpoint column, right-anchored means start from column 0.
    - The upper-right band's vertical anchor sets the row phase in the analogous way: top-anchored means start from the midpoint row, bottom-anchored means start from row 0.
    - Tile the lower-right stencil with those row/column phases; whenever the shifted stencil lands on black, replace that cell with the shifted coarse fill color from the lower-left tile. If the stencil is empty, tile the lower-left tile directly with the same phases.

    Key insight:
    - The same top-half shapes now encode two semantics simultaneously: their bounding spans determine the output size, while their within-quadrant offsets determine where the repeated lower pattern starts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    _ul_r0, _ul_c0, out_h, _ul_w = _bbox_info(upper_left, bg)
    _ur_r0, _ur_c0, _ur_h, out_w = _bbox_info(upper_right, bg)
    row_phase = _decode_row_phase(upper_right, bg)
    col_phase = _decode_col_phase(upper_left, bg)

    assert out_h > 0 and out_w > 0
    assert np.all(fill_tile != bg)
    return _expected_output(fill_tile, stencil, out_h, out_w, row_phase, col_phase)


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


def _make_vertical_control(
    q: int,
    height: int,
    max_colors: int,
    pool: list[int],
    anchor_left: bool,
) -> np.ndarray:
    assert 1 <= height <= q
    pal = _control_palette(max_colors, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    span_w = random.randint(2, min(q - 1, CONTROL_BAND_SPAN_MAX))
    r0 = random.randint(0, q - height)
    c0 = 0 if anchor_left else q - span_w
    rows = list(range(r0, r0 + height))
    cols = list(range(c0, c0 + span_w))

    for r in rows:
        picks = random.randint(1, span_w)
        for c in random.sample(cols, picks):
            tile[r, c] = random.choice(pal)

    tile[r0, c0] = random.choice(pal)
    tile[r0 + height - 1, c0 + span_w - 1] = random.choice(pal)

    if np.count_nonzero(tile != BLACK) <= height:
        extras = [(r, c) for r in rows for c in cols if tile[r, c] == BLACK]
        assert extras
        r, c = random.choice(extras)
        tile[r, c] = random.choice(pal)

    _r, _c, bbox_h, bbox_w = _bbox_info(tile, BLACK)
    assert bbox_h == height and bbox_w == span_w
    assert np.count_nonzero(tile != BLACK) > height
    decoded = _decode_col_phase(tile, BLACK)
    assert decoded == (q // 2 if anchor_left else 0)
    return tile


def _make_horizontal_control(
    q: int,
    width: int,
    max_colors: int,
    pool: list[int],
    anchor_top: bool,
) -> np.ndarray:
    assert 1 <= width <= q
    pal = _control_palette(max_colors, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    span_h = random.randint(2, min(q - 1, CONTROL_BAND_SPAN_MAX))
    r0 = 0 if anchor_top else q - span_h
    c0 = random.randint(0, q - width)
    rows = list(range(r0, r0 + span_h))
    cols = list(range(c0, c0 + width))

    for c in cols:
        picks = random.randint(1, span_h)
        for r in random.sample(rows, picks):
            tile[r, c] = random.choice(pal)

    tile[r0, c0] = random.choice(pal)
    tile[r0 + span_h - 1, c0 + width - 1] = random.choice(pal)

    if np.count_nonzero(tile != BLACK) <= width:
        extras = [(r, c) for r in rows for c in cols if tile[r, c] == BLACK]
        assert extras
        r, c = random.choice(extras)
        tile[r, c] = random.choice(pal)

    _r, _c, bbox_h, bbox_w = _bbox_info(tile, BLACK)
    assert bbox_h == span_h and bbox_w == width
    assert np.count_nonzero(tile != BLACK) > width
    decoded = _decode_row_phase(tile, BLACK)
    assert decoded == (q // 2 if anchor_top else 0)
    return tile


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    row_phase = q // 2 if random.random() < 0.5 else 0
    col_phase = q // 2 if random.random() < 0.5 else 0
    anchor_top = row_phase == q // 2
    anchor_left = col_phase == q // 2

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    if stencil_empty:
        ll_k = max(2, ll_k)
    ll_palette = _palette(ll_k)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    valid_dims = [
        (h, w)
        for h in range(OUTPUT_DIM_MIN, q + 1)
        for w in range(OUTPUT_DIM_MIN, q + 1)
        if h * w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    visible_fill = [
        ((r + row_phase) % q, (c + col_phase) % q)
        for r in range(out_h)
        for c in range(out_w)
    ]
    fill_tile = _make_lower_left(q, ll_palette, visible_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_vertical_control(q, out_h, UL_MAX_COLORS, control_pool, anchor_left)
    upper_right = _make_horizontal_control(q, out_w, UR_MAX_COLORS, control_pool, anchor_top)

    expected = _expected_output(fill_tile, stencil, out_h, out_w, row_phase, col_phase)
    if np.array_equal(expected, _zero_phase_output(fill_tile, stencil, out_h, out_w)):
        return None

    if not stencil_empty:
        stencil_view = _phased_stencil_view(stencil, out_h, out_w, row_phase, col_phase)
        if not np.any(stencil_view != BLACK):
            return None
        if not np.any(stencil_view == BLACK):
            return None

    if np.count_nonzero(upper_left != BLACK) <= out_h:
        return None
    if np.count_nonzero(upper_right != BLACK) <= out_w:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    legacy = _legacy_count_output(fill_tile, stencil, upper_left, upper_right)
    if legacy.shape == expected.shape and np.array_equal(legacy, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
