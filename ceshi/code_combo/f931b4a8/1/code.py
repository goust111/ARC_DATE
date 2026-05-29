# Combo variant of f931b4a8 (index 1)
# combo_spec: D1/0+D2/0
# Name: 外接矩形编码+行列互换控制

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis:
# spec_variant:
#   unchanged_core:
#     - The input is still an even square split into four equal quadrants.
#     - The lower-left quadrant is still a fully non-black fill tile.
#     - The lower-right quadrant is still a stencil tile; black cells are holes
#       that must be replaced by coarse colors from the lower-left tile.
#   fused_change:
#     - D1/0 replaces pixel-count decoding with bounding-box decoding.
#     - D2/0 swaps the roles of the two control quadrants.
#     - Combined effect: upper-left bbox width sets output columns, while
#       upper-right bbox height sets output rows.
#   behavior_trace:
#     - Example with Q=4:
#       upper-left fg at {(1,0),(1,1),(2,1),(2,2)} has bbox 2x3 and count 4.
#       upper-right fg at {(0,2),(1,2),(2,1),(2,2)} has bbox 3x2 and count 4.
#       Original task would decode (rows, cols) = (4, 4) from counts.
#       D1-only would decode (rows, cols) = (2, 2) from (UL bbox height, UR bbox width).
#       This fused task must decode (rows, cols) = (3, 3) from
#       (UR bbox height, UL bbox width).
#   complexity_calibration:
#     - The upper-left control is generated as a horizontal sparse band:
#       measured axis = width, unmeasured axis = a small height span.
#     - The upper-right control is generated as a vertical sparse band:
#       measured axis = height, unmeasured axis = a small width span.
#     - Foreground counts are forced to exceed the measured bbox dimension, so
#       the samples do not silently collapse back to the original count-based task.
#     - The unused axis is kept narrow (2-4 cells) so the bbox cue is visible
#       without turning the control panel into noisy scatter.
#   hidden_assumptions_audit:
#     - `count_nonzero` in the original solver is no longer valid; the fused
#       solver must explicitly extract bbox spans.
#     - Because a bbox span cannot exceed Q, attainable output dimensions are
#       now restricted to 1..Q; the generator therefore only samples 3..Q.
#     - The bottom-half tiling logic assumes positive output dimensions and
#       valid coarse indices. Since out_h,out_w <= Q here, `r // Q` and `c // Q`
#       stay within bounds, so the original replacement logic remains safe.
#     - A non-empty stencil could still be visually useless if the visible
#       output window sees only fg or only bg. The generator rejects such cases
#       so the lower-half interaction stays observable.
#   placement_legality:
#     - Every control bbox is placed wholly inside its quadrant by sampling a
#       valid top-left anchor and a span that fits within Q.
#     - The lower-left tile contains no black cells, so any stencil hole can be
#       filled safely.
#     - The four quadrants never overlap because they occupy disjoint slices of
#       the NxN canvas.
#   observability_cleanliness:
#     - The top half stays black-background and sparse, while the bottom half
#       stays dense and pattern-like, so region roles remain easy to separate.
#     - Left and right controls have deliberately different aspect tendencies
#       (wide vs tall), making the role swap human-visible.
#     - Color identities inside the controls remain decorative only; all control
#       information is carried by visible span, not by color value.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)
STENCIL_EMPTY_PROB = 0.18
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
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


def _bbox_dims(tile: np.ndarray, bg: int) -> tuple[int, int]:
    coords = np.argwhere(tile != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _decode_fused_dims(upper_left: np.ndarray, upper_right: np.ndarray, bg: int) -> tuple[int, int]:
    _, out_w = _bbox_dims(upper_left, bg)
    out_h, _ = _bbox_dims(upper_right, bg)
    return out_h, out_w


def _decode_d1_dims(upper_left: np.ndarray, upper_right: np.ndarray, bg: int) -> tuple[int, int]:
    out_h, _ = _bbox_dims(upper_left, bg)
    _, out_w = _bbox_dims(upper_right, bg)
    return out_h, out_w


def _decode_original_dims(upper_left: np.ndarray, upper_right: np.ndarray, bg: int) -> tuple[int, int]:
    return int(np.count_nonzero(upper_left != bg)), int(np.count_nonzero(upper_right != bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read crossed bounding-box cues from the top half, then render an output from the lower tile pair.

    Input structure:
    - The input is an even square partitioned into four equal quadrants on a black background.
    - The upper-left and upper-right quadrants contain scattered control pixels.
    - The lower-left quadrant is a non-black color tile, and the lower-right quadrant is a stencil tile that may include black holes.

    Rule:
    - Output column count is the width of the foreground bounding box in the upper-left quadrant.
    - Output row count is the height of the foreground bounding box in the upper-right quadrant.
    - If the lower-right stencil is all black, tile the lower-left quadrant directly across the output.
    - Otherwise, tile the lower-right stencil over the output and replace each black stencil cell with the lower-left color selected by the corresponding coarse tile-block coordinate.

    Key insight:
    - The top controls no longer encode size by pixel count, and they do not control their intuitive axis: the left panel contributes width while the right panel contributes height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h, out_w = _decode_fused_dims(upper_left, upper_right, bg)
    assert out_h > 0 and out_w > 0

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


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


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
        hi = max(lo, int(ceil(STENCIL_FG_FRAC[1] * area)))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    return tile


def _control_palette(max_colors: int, pool: list[int]) -> list[int]:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, min(max_colors, len(pool)))
    return _palette(k, pool)


def _alt_span(q: int, avoid: int) -> int:
    hi = min(q, CONTROL_BAND_SPAN_MAX)
    candidates = [span for span in range(2, hi + 1) if span != avoid]
    assert candidates
    return random.choice(candidates)


def _make_width_control(
    q: int,
    width: int,
    avoid_height: int,
    max_colors: int,
    pool: list[int],
) -> np.ndarray:
    assert 1 <= width <= q
    span_h = _alt_span(q, avoid_height)
    pal = _control_palette(max_colors, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    r0 = random.randint(0, q - span_h)
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

    bbox_h, bbox_w = _bbox_dims(tile, BLACK)
    assert bbox_h == span_h and bbox_w == width
    assert np.count_nonzero(tile != BLACK) > width
    return tile


def _make_height_control(
    q: int,
    height: int,
    avoid_width: int,
    max_colors: int,
    pool: list[int],
) -> np.ndarray:
    assert 1 <= height <= q
    span_w = _alt_span(q, avoid_width)
    pal = _control_palette(max_colors, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    r0 = random.randint(0, q - height)
    c0 = random.randint(0, q - span_w)
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

    bbox_h, bbox_w = _bbox_dims(tile, BLACK)
    assert bbox_h == height and bbox_w == span_w
    assert np.count_nonzero(tile != BLACK) > height
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
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
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
        (out_h, out_w)
        for out_h in range(OUTPUT_DIM_MIN, q + 1)
        for out_w in range(OUTPUT_DIM_MIN, q + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    visible_fill = [(r, c) for r in range(out_h) for c in range(out_w)]
    fill_tile = _make_lower_left(q, ll_palette, visible_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_width_control(q, out_w, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_height_control(q, out_h, out_w, UR_MAX_COLORS, control_pool)

    if not stencil_empty:
        visible_stencil = stencil[:out_h, :out_w]
        if not np.any(visible_stencil != BLACK):
            return None
        if not np.any(visible_stencil == BLACK):
            return None

    fused_dims = _decode_fused_dims(upper_left, upper_right, BLACK)
    d1_dims = _decode_d1_dims(upper_left, upper_right, BLACK)
    original_dims = _decode_original_dims(upper_left, upper_right, BLACK)
    ul_bbox_h, ul_bbox_w = _bbox_dims(upper_left, BLACK)
    ur_bbox_h, ur_bbox_w = _bbox_dims(upper_right, BLACK)

    if fused_dims != (out_h, out_w):
        return None
    if d1_dims == fused_dims or original_dims == fused_dims:
        return None
    if ul_bbox_h == out_h or ur_bbox_w == out_w:
        return None
    if original_dims[0] == out_h or original_dims[1] == out_w:
        return None
    if ul_bbox_w != out_w or ur_bbox_h != out_h:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

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
