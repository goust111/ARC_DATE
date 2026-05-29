# Combo variant of f931b4a8 (index 32)
# combo_spec: D1/3+D7/0
# Name: 非空行列计数+正方形输出

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The input keeps the original 2x2 quadrant layout. The bottom half keeps the
#    original material roles: lower-left is an all-non-black fill tile and
#    lower-right is either an all-black direct-copy signal or a stencil whose
#    black cells are replaced from the fill tile by coarse coordinates.
#    The fused change is entirely in how the top half determines output size:
#    - upper-left contributes the number of occupied rows, not the number of
#      foreground cells.
#    - upper-right contributes the number of occupied columns, not the number of
#      foreground cells.
#    - these two decoded counts do not become height and width directly;
#      instead the output is forced to be square with side = max(row_count,
#      col_count).
#    Example with q = 4:
#      UL row occupancies = [2, 0, 3, 1]  -> row_count = 3, raw fg count = 6
#      UR col occupancies = [0, 4, 0, 2]  -> col_count = 2, raw fg count = 6
#      Original task would output 6x6 from raw pixel counts.
#      The D1-only variant would output 3x2 from occupied-line counts.
#      This fused combo outputs 3x3 because it first decodes (3, 2) and then
#      applies the square-max rule.
#    The visible difference is therefore two-stage: line occupancy must be read
#    first, and then the smaller decoded axis is overridden by the larger one.
#
# b) Complexity calibration:
#    If UL and UR always had the same occupied-line count, the square rule would
#    be invisible and the task would collapse to the D1 variant. The generator
#    therefore rejects equal line counts and samples both "UL larger" and
#    "UR larger" cases. To keep D1 visible as well, each active control line is
#    usually wider than one cell, so raw foreground totals are strictly larger
#    than occupied-line counts.
#
# c) Hidden-assumption audit of the original main():
#    - The original `count_nonzero` lines no longer encode dimensions; they must
#      be replaced by active-row / active-column counts.
#    - The original solver assumed output height and width were independent.
#      That assumption is broken by the combo, so both decoded counts must feed
#      a single `side = max(...)`.
#    - The bottom-half logic itself still works unchanged, but after the D1
#      change we always have `side <= q`. Therefore black stencil holes only
#      ever read `fill_tile[0, 0]`, and the empty-stencil branch reduces to the
#      top-left `side x side` crop of the fill tile. The generator guarantees
#      that this visible crop is informative.
#
# d) Placement legality:
#    Four quadrants remain disjoint by construction. Control pixels are painted
#    only inside their own qxq quadrant. The fill tile stays entirely non-black.
#    The stencil remains inside its quadrant and may be all black or mixed.
#    Because line counts are at most q, `side <= q` always holds, so the coarse
#    fill lookup indices used by the solver stay in-bounds.
#
# e) Visual observability:
#    A human can inspect the upper-left by rows and the upper-right by columns,
#    then compare those two counts with the square output side. The generator
#    ensures at least one widened active row and one widened active column so
#    simple raw-pixel counting is visibly wrong. It also enforces unequal
#    decoded counts, making the "take the larger one and square it" rule visible
#    from a few examples.
#
# f) Visual cleanliness:
#    The picture remains cleanly partitioned: top = controls, lower-left = fill
#    source, lower-right = stencil/direct-copy signal. No overlays or extra
#    markers are introduced. Control colors are excluded from the output palette
#    so the instructional regions stay visually separate from the produced
#    pattern.
#
# g) Color inference:
#    No rule depends on numeric color ids. Colors still serve only to separate
#    control regions from output-producing regions. The output palette is exactly
#    the union of lower-left colors and lower-right stencil foreground colors.
#
# New solver assumptions introduced by the combo:
#    - UL must contain at least one occupied row and UR at least one occupied
#      column; the generator guarantees both.
#    - The generator restricts grid sizes to N in {8, 16}, so q is 4 or 8.
#      That leaves enough room to show multi-cell rows/columns and to keep the
#      square-max rule legible while preserving `side >= 3`.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)                   # q >= 4 keeps occupied-line controls legible
STENCIL_EMPTY_PROB = 0.16              # chance lower-right is all-black
LL_SOLID_PROB = 0.70                   # chance fill tile is single-color
LL_MULTI_COLOR_RANGE = (2, 3)          # color count when fill tile is multi-color
STENCIL_SINGLE_COLOR_PROB = 0.75       # chance stencil uses one fg color
STENCIL_MULTI_COLOR_RANGE = (2, 4)     # color count when stencil is multi-color
STENCIL_FG_FRAC = (0.30, 0.75)         # fg fraction range for random stencil mode
STENCIL_FRAME_MIN_Q = 4                # min quadrant size to allow frame stencil
CONTROL_SINGLE_COLOR_PROB = 0.85       # chance control quadrant uses one color
CONTROL_RECT_PROB = 0.85               # chance active lines share one common segment
UL_MAX_COLORS = 3                      # max distinct colors in upper-left control
UR_MAX_COLORS = 2                      # max distinct colors in upper-right control
OUTPUT_SIDE_MIN = 3                    # minimum final square side
MINOR_COUNT_MIN = 2                    # smaller occupied-line count for visibility
GEN_MAX_RETRIES = 400                  # retry budget for generate_input


def _count_active_rows(tile: np.ndarray, bg: int) -> int:
    """Return the number of rows containing at least one non-background cell."""
    return int(np.count_nonzero(np.any(tile != bg, axis=1)))


def _count_active_cols(tile: np.ndarray, bg: int) -> int:
    """Return the number of columns containing at least one non-background cell."""
    return int(np.count_nonzero(np.any(tile != bg, axis=0)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode output size from occupied control lines, then square it before rendering the lower pattern.

    Input structure:
    - The grid is an even square divided into four equal quadrants.
    - Black is the background color.
    - The upper-left and upper-right quadrants are control panels.
    - The lower-left quadrant is an all-foreground fill tile, while the lower-right quadrant is either an all-black direct-copy signal or a stencil tile.

    Rule:
    - Count how many rows in the upper-left quadrant contain at least one non-black cell.
    - Count how many columns in the upper-right quadrant contain at least one non-black cell.
    - Let `side` be the larger of those two counts; the output is always a `side x side` square.
    - If the lower-right quadrant is all black, copy the lower-left tile pattern over that square by modular repetition.
    - Otherwise tile the lower-right stencil over the square, and replace each black stencil cell with the lower-left value chosen by coarse block coordinates.

    Key insight:
    - The top half first encodes two line-occupancy counts and then collapses them into one square side by taking the maximum; the bottom half still provides the pattern that is copied or used to fill stencil holes.
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

    row_count = _count_active_rows(upper_left, bg)
    col_count = _count_active_cols(upper_right, bg)
    side = max(row_count, col_count)
    assert 0 < side <= qh

    out = np.empty((side, side), dtype=int)
    if not np.any(stencil != bg):
        for r in range(side):
            for c in range(side):
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(side):
        for c in range(side):
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
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _make_lower_left(
    q: int,
    palette: list[int],
    visible_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile (all non-black) using stripe/solid/diagonal patterns."""
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
        fix_coords = list(visible_coords) if visible_coords is not None else [
            (r, c) for r in range(q) for c in range(q)
        ]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black = holes filled from fill tile)."""
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


def _paint_line(
    tile: np.ndarray,
    axis: int,
    line: int,
    positions: list[int],
    palette: list[int],
) -> None:
    """Paint one active row or column with random control colors."""
    if axis == 0:
        for c in positions:
            tile[line, c] = random.choice(palette)
    else:
        for r in positions:
            tile[r, line] = random.choice(palette)


def _make_line_control(
    q: int,
    line_count: int,
    max_colors: int,
    pool: list[int],
    axis: int,
) -> np.ndarray:
    """Build a control quadrant whose meaning is active rows or active columns."""
    assert 1 <= line_count <= q
    max_k = min(max_colors, len(pool))
    if max_k <= 1 or random.random() < CONTROL_SINGLE_COLOR_PROB:
        k = 1
    else:
        k = random.randint(2, max_k)
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)
    active_lines = random.sample(range(q), line_count)

    if random.random() < CONTROL_RECT_PROB:
        span = random.randint(2, q)
        start = random.randint(0, q - span)
        positions = list(range(start, start + span))
        for line in active_lines:
            _paint_line(tile, axis, line, positions, pal)
    else:
        widened = random.choice(active_lines)
        for line in active_lines:
            width_lo = 2 if line == widened else 1
            width = random.randint(width_lo, q)
            if random.random() < 0.5:
                start = random.randint(0, q - width)
                positions = list(range(start, start + width))
            else:
                positions = random.sample(range(q), width)
            _paint_line(tile, axis, line, positions, pal)

    active_count = _count_active_rows(tile, BLACK) if axis == 0 else _count_active_cols(tile, BLACK)
    assert active_count == line_count
    assert int(np.count_nonzero(tile != BLACK)) > line_count
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, side: int) -> np.ndarray:
    """Reproduce the solver's logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:side, :side]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _visible_stencil_is_mixed(stencil: np.ndarray, side: int) -> bool:
    """Require both holes and foreground to appear in the visible crop."""
    visible = stencil[:side, :side]
    return bool(np.any(visible == BLACK) and np.any(visible != BLACK))


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if len(control_pool) < 1:
        return None

    side_lo = min(max(OUTPUT_SIDE_MIN, 1), q)
    valid_pairs = [
        (row_count, col_count)
        for row_count in range(MINOR_COUNT_MIN, q + 1)
        for col_count in range(MINOR_COUNT_MIN, q + 1)
        if row_count != col_count and max(row_count, col_count) >= side_lo
    ]
    if not valid_pairs:
        return None
    row_count, col_count = random.choice(valid_pairs)
    side = max(row_count, col_count)

    visible_fill = [(r, c) for r in range(side) for c in range(side)]
    fill_tile = _make_lower_left(q, ll_palette, visible_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    if not stencil_empty and not _visible_stencil_is_mixed(stencil, side):
        return None

    upper_left = _make_line_control(q, row_count, UL_MAX_COLORS, control_pool, axis=0)
    upper_right = _make_line_control(q, col_count, UR_MAX_COLORS, control_pool, axis=1)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, side)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    raw_ul = int(np.count_nonzero(upper_left != BLACK))
    raw_ur = int(np.count_nonzero(upper_right != BLACK))
    if raw_ul == row_count or raw_ur == col_count:
        return None
    if actual.shape != (side, side):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
