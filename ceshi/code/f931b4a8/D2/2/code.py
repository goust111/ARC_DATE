# Variant of f931b4a8
# Dimension: D2 — 控制象限的布局位置 (src)
# Change: 左上象限控制输出高度，右上象限控制输出宽度
#   → 控制信息编码在网格边缘的单行/单列
#
# Design analysis for the D2 variant
# ----------------------------------
# spec:
#   grid:
#     - odd square N x N, with N in {5, 9, 17}
#     - the first row and first column are reserved for control pixels
#     - the top-left corner is always black so the two 1D controls stay separate
#   interior:
#     - removing the top row and left column leaves a 2Q x 2Q body, Q in {2, 4, 8}
#     - the left half is the color-source material, shown as two stacked identical QxQ copies
#     - the right half is the stencil material, also shown as two stacked identical QxQ copies
#     - this keeps the body visually full without inventing a second transformation rule
#   controls:
#     - left column fg count (excluding the corner) = output height
#     - top row fg count (excluding the corner) = output width
#     - control colors never appear in the output, as in the source task
#   output:
#     - uses the same synthesis rule as the source task once QxQ fill/stencil exemplars are read
#     - if stencil is empty, tile fill directly
#     - otherwise tile stencil; every stencil black cell is replaced by fill_tile[r // Q, c // Q]
#
# behavior trace:
#   - Original task: a non-black count in the upper-left quadrant set height, and a non-black count
#     in the upper-right quadrant set width. The lower quadrants provided fill and stencil.
#   - Variant: the same counts move onto the border. For example, with Q=4, 6 non-black cells on
#     the left border mean height=6, and 7 non-black cells on the top border mean width=7. The
#     8x8 interior still contains the fill/stencil data, so only the control-source location changes.
#   - Difference: the solver must no longer interpret any 2D control quadrant. It must isolate the
#     1D border encodings first, then read the interior body separately.
#
# complexity calibration:
#   - A single row/column can encode at most 2Q cells, so output dimensions are limited to [max(3,Q), 2Q].
#   - This is still large enough to show repeated tiling and coarse fill replacement while keeping the
#     new border encoding visually obvious.
#   - Control lines are usually contiguous segments, but sometimes compact or scattered, so the task
#     depends on counting foreground cells rather than memorizing one fixed shape.
#
# implicit-assumption audit:
#   - Source line `assert h == w and h % 2 == 0` relied on quadrant controls. Here the correct invariant
#     is an odd square with an even interior body, so the parity check must change.
#   - Source slices `upper_left` and `upper_right` used to contain only control pixels. That assumption
#     is false in the variant because the controls live on the border, so those reads must be replaced.
#   - Source reads of `fill_tile` and `stencil` as QxQ tiles still work if the generator guarantees a
#     readable QxQ exemplar inside the body. The duplicated stacked layout provides that guarantee.
#   - The coarse lookup `fill_tile[r // Q, c // Q]` still assumes the coarse indices stay in bounds.
#     The generator now enforces output dims <= 2Q, so coarse indices are only 0 or 1 and always valid.
#
# placement legality and visual cleanliness:
#   - Control pixels occupy only the first row/column, so they never overlap the pattern body.
#   - The body is a clean left/right split between source and stencil material, with no overdraw.
#   - Duplicating the QxQ exemplars vertically fills the square body and makes the two functional
#     regions readable to a human from a few examples.
#
# color inference:
#   - No ordering or branching depends on color id.
#   - Control colors are excluded from output colors, so a solver can visually separate “count marks”
#     from “pattern material”.

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read output dimensions from border counts, then render the interior fill/stencil pattern.

    Input structure:
    - The grid is an odd square with a control row on top and a control column on the left.
    - The top-left corner is background.
    - Removing that border leaves an even square body split into four equal blocks; the left side shows the fill tile and the right side shows the stencil tile.

    Rule:
    - Output height is the number of non-background cells in the left border column, excluding the corner.
    - Output width is the number of non-background cells in the top border row, excluding the corner.
    - Read one fill exemplar from the body's upper-left block and one stencil exemplar from the body's upper-right block. If the stencil is empty, tile the fill exemplar directly.
    - Otherwise tile the stencil exemplar, and whenever the stencil has background use the coarse fill color selected by the repeated-tile block position.

    Key insight:
    - The counting signal moved from 2D control quadrants to 1D border encodings, but the interior pattern-composition rule is the same two-level tiling mechanism as before.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1 and h >= 5

    bg = int(Color.BLACK)
    assert int(g[0, 0]) == bg

    out_h = int(np.count_nonzero(g[1:, 0] != bg))
    out_w = int(np.count_nonzero(g[0, 1:] != bg))
    assert out_h > 0 and out_w > 0

    body = g[1:, 1:]
    bh, bw = body.shape
    assert bh == bw and bh % 2 == 0
    qh, qw = bh // 2, bw // 2

    fill_tile = body[:qh, :qw]
    stencil = body[:qh, qw:]
    assert np.all(fill_tile != bg)

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


BLACK = 0
COLORS = tuple(range(1, 10))

BODY_SIZES = (4, 8, 16)                # even interior side-lengths 2Q
GRID_SIZES = tuple(size + 1 for size in BODY_SIZES)
STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_SEGMENT_PROB = 0.85
LEFT_MAX_COLORS = 3
TOP_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


def _ordered_positions(length: int, mode: str) -> list[int]:
    positions = list(range(length))
    if mode == "compact":
        anchor = random.randrange(length)
        random.shuffle(positions)
        return sorted(positions, key=lambda idx: (abs(idx - anchor), random.random()))
    if mode == "ends":
        return positions[::2] + positions[1::2]
    random.shuffle(positions)
    return positions


def _make_lower_left(q: int, palette: list[int],
                     required_coords: list[tuple[int, int]] | None = None) -> np.ndarray:
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
        fix_coords = list(required_coords) if required_coords is not None else \
            [(r, c) for r in range(q) for c in range(q)]
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


def _make_control_line(length: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else \
        random.randint(2, min(max_colors, count, len(pool)))
    pal = _palette(k, pool)
    line = np.full(length, BLACK, dtype=int)

    if random.random() < CONTROL_SEGMENT_PROB:
        start = random.randint(0, length - count)
        line[start:start + count] = _random_color_array((count,), pal)
        return line

    mode = random.choice(["compact", "ends", "scatter"])
    positions = _ordered_positions(length, mode)[:count]
    for idx in positions:
        line[idx] = random.choice(pal)
    return line


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _assemble_body(fill_tile: np.ndarray, stencil: np.ndarray) -> np.ndarray:
    q = fill_tile.shape[0]
    body = np.zeros((2 * q, 2 * q), dtype=int)
    body[:q, :q] = fill_tile
    body[q:, :q] = fill_tile
    body[:q, q:] = stencil
    body[q:, q:] = stencil
    return body


def _gen_attempt() -> np.ndarray | None:
    body_size = random.choice(BODY_SIZES)
    q = body_size // 2
    n = body_size + 1
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else \
            random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, 2 * q)
    valid_dims = [
        (h, w) for h in range(lo, hi + 1) for w in range(lo, hi + 1)
        if h * w <= OUTPUT_AREA_MAX and (stencil_empty or ceil(h / q) * ceil(w / q) >= ll_k)
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    body = _assemble_body(fill_tile, stencil)

    top_ctrl = _make_control_line(body_size, out_w, TOP_MAX_COLORS, control_pool)
    left_ctrl = _make_control_line(body_size, out_h, LEFT_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[0, 1:] = top_ctrl
    grid[1:, 0] = left_ctrl
    grid[1:, 1:] = body

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
