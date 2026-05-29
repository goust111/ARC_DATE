# Variant of f931b4a8
# Dimension: D10 — 控制象限前景的空间排列 (meta)
# Change: 控制象限的前景像素可以是任意排列（均匀填充、稀疏散布等），仅计数有意义，空间位置无额外语义... → 前景像素排列成数字形状（如3x5点阵数字）

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color, bbox, crop_bbox

# Design analysis for the D10 variant
# source_task: none
# a_concretization_and_trace:
#   control_encoding:
#     - The lower half keeps the original roles: lower-left is the non-black fill tile,
#       lower-right is an optional stencil tile.
#     - The change is only in the two control quadrants. They no longer mean "count all
#       non-black cells"; instead each quadrant contains a readable Arabic numeral drawn
#       as a 3x5 bitmap font. Values 3..9 use one digit, values 10..16 use two digits.
#     - To keep the digits legible, the input side length is fixed to 16 so each control
#       quadrant is 8x8. A 5x7 two-digit glyph fits comfortably inside 8x8.
#   behaviour_trace:
#     - Original: if the upper-left quadrant happened to contain 11 foreground cells in
#       any arrangement, the output height was 11.
#     - Variant: if the upper-left quadrant visually shows the numeral "3", that bitmap
#       also contains 11 foreground cells, but the intended height is 3. Counting would
#       now be wrong; the solver must crop the foreground bounding box and recognize the
#       bitmap pattern as the number 3.
# b_complexity_calibration:
#   - The variant must be visually obvious in 2-4 ARC examples, so the control encoding
#     cannot collapse to tiny 2x2/4x4 blobs. Restricting to 8x8 control quadrants is a
#     necessary adaptation rather than a separate innovation.
#   - The number range stays 3..16, preserving both one-digit and two-digit outputs. This
#     creates repeated support for the "digit reading" rule while keeping the space small.
# c_hidden_assumption_audit:
#   - Old line: np.count_nonzero(upper_left != bg). Hidden assumption: control geometry is
#     irrelevant and only multiplicity matters. This is false in the variant.
#   - Old line: np.count_nonzero(upper_right != bg). Same issue.
#   - Tiling loops below depend only on:
#       1. decoded dimensions being positive;
#       2. dimensions staying <= 16 so coarse fill indices are within the 8x8 fill tile.
#     Those assumptions still hold because the generator only emits numbers 3..16.
# d_placement_legality:
#   - Every control numeral is rendered on a 5-row bitmap whose width is 3 or 7, then
#     translated within the 8x8 quadrant. No overlap with other quadrants is possible
#     because each control region is confined to its own 8x8 block.
#   - Lower-left remains all non-black; lower-right may be empty or partially black; both
#     match the original solver's expectations.
# e_visual_observability:
#   - A human can now see literal numerals in the top half. Across examples, the same
#     digit can correspond to different foreground counts, so the old counting heuristic
#     is contradicted directly by the visuals.
# f_visual_cleanliness:
#   - The control glyph uses at most 3 colors and is isolated in a black quadrant, so the
#     shape remains readable even when colors vary. The bottom half retains the original,
#     already tidy quadrant structure.
# g_color_change_inference:
#   - Colors are not part of numeral identity. Solver converts control quadrants to a
#     binary foreground mask before recognition, so the task never depends on color ids.
# h_variant_metric_for_post_checks:
#   - The key metric is control_mismatch = foreground_pixel_count - decoded_number.
#     Non-zero mismatch shows the new D10 behaviour is active and would disagree with the
#     original counting solver on the same input.
# i_new_solver_assumptions:
#   - Each control quadrant contains exactly one renderable numeral from 3..16.
#   - Translation inside the quadrant is allowed, so the solver must read the cropped
#     foreground bbox instead of relying on an absolute placement.
#   - The generator guarantees these assumptions by construction and validates them by
#     running main() on every candidate instance.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)                     # fixed so each control quadrant is 8x8
CONTROL_Q = 8
CONTROL_SINGLE_COLOR_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
CONTROL_NUMBER_MIN = 3
CONTROL_NUMBER_MAX = 16
STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


DIGIT_FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "011", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
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


def _glyph_mask(ch: str) -> np.ndarray:
    rows = [[cell == "1" for cell in row] for row in DIGIT_FONT[ch]]
    return np.array(rows, dtype=bool)


def _render_number_mask(value: int) -> np.ndarray:
    glyphs = [_glyph_mask(ch) for ch in str(value)]
    rows = []
    for r in range(glyphs[0].shape[0]):
        parts = [glyph[r] for glyph in glyphs]
        rows.append(np.concatenate(parts, axis=0) if len(parts) == 1 else np.concatenate(
            [parts[0], np.array([False], dtype=bool), *parts[1:]],
            axis=0,
        ))
    return np.stack(rows, axis=0)


NUMBER_MASKS = {
    value: _trim_mask(_render_number_mask(value))
    for value in range(CONTROL_NUMBER_MIN, CONTROL_NUMBER_MAX + 1)
}
NUMBER_PIXEL_COUNTS = {value: int(mask.sum()) for value, mask in NUMBER_MASKS.items()}
assert all(NUMBER_PIXEL_COUNTS[value] != value for value in NUMBER_PIXEL_COUNTS)


def _decode_control_number(tile: np.ndarray) -> int:
    fg = tile != BLACK
    b = bbox(fg)
    assert b is not None
    sprite = crop_bbox(fg.astype(bool), b)
    for value, target in NUMBER_MASKS.items():
        if sprite.shape == target.shape and np.array_equal(sprite, target):
            return value
    raise AssertionError("unrecognized control numeral")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the two upper quadrants as bitmap numerals that specify the output size, then build the output from the lower quadrant tiles.

    Input structure:
    - The grid is a 16x16 square split into four 8x8 quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control displays containing Arabic numerals drawn with foreground pixels.
    - The lower-left quadrant is a non-black fill tile; the lower-right quadrant is either a stencil tile or empty.

    Rule:
    - Crop the foreground shape in the upper-left quadrant and identify which numeral it draws; that numeral is the output height.
    - Crop the foreground shape in the upper-right quadrant and identify which numeral it draws; that numeral is the output width.
    - If the lower-right stencil has foreground pixels, repeat it across the output; whenever the repeated stencil lands on black, fill that cell from the lower-left quadrant using the corresponding coarse tile position.
    - If the lower-right stencil is fully black, repeat the lower-left quadrant directly across the entire output.

    Key insight:
    - The lower half still defines a two-level tiled texture, but the top half now encodes dimensions by recognizable digit shapes rather than by the raw number of foreground cells.
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

    out_h = _decode_control_number(upper_left)
    out_w = _decode_control_number(upper_right)
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


def _make_control(q: int, value: int, max_colors: int, pool: list[int]) -> np.ndarray:
    assert q == CONTROL_Q
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, min(max_colors, len(pool)))
    palette = _palette(k, pool)
    mask = _render_number_mask(value)
    mh, mw = mask.shape
    assert mh <= q and mw <= q
    r0 = random.randint(0, q - mh)
    c0 = random.randint(0, q - mw)
    tile = np.full((q, q), BLACK, dtype=int)
    coords = np.argwhere(mask)
    for dr, dc in coords:
        tile[r0 + int(dr), c0 + int(dc)] = random.choice(palette)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    assert q == CONTROL_Q
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
        (h, w)
        for h in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        for w in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        if h * w <= OUTPUT_AREA_MAX and (stencil_empty or ceil(h / q) * ceil(w / q) >= ll_k)
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

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
