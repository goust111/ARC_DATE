# Variant of f931b4a8
# Dimension: D1 — 输出尺寸的编码方式 (symbol_instructions)
# Change: 左上象限前景像素计数 = 输出行数，右上象限前景像素计数 = 输出列数... → 前景像素行数/列数编码尺寸

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the D1 variant
# ----------------------------------
# a) Concrete behavior and trace:
#    The input is still a 2x2 quadrant layout. The bottom half behaves exactly
#    like the original task: lower-left is a non-black fill tile and
#    lower-right is either an all-black direct-copy signal or a stencil whose
#    black cells get replaced from the lower-left tile by coarse block lookup.
#    The only intended semantic change is in the control quadrants:
#    - upper-left no longer contributes "how many foreground cells";
#      it contributes "how many rows contain foreground at all".
#    - upper-right analogously contributes "how many columns contain
#      foreground at all".
#    Example with Q=4:
#      UL active rows are {0, 2, 3}; row occupancies are [2, 0, 1, 4].
#      Original solver would produce height 7, while the variant produces 3.
#      UR active cols are {0, 1, 3}; column occupancies are [3, 2, 0, 1].
#      Original solver would produce width 6, while the variant produces 3.
#      The lower quadrants are then applied to a 3x3 output instead of 7x6.
#    This means the changed behavior occurs entirely at the "decode dimensions
#    from top quadrants" step; the bottom tiling logic remains intact.
# b) Complexity calibration:
#    If each active row/column had exactly one foreground cell, the new rule
#    would visually resemble simple counting and many samples would still be
#    solvable by the old heuristic. To keep D1 observable, the generator makes
#    each control region use active line occupancy rather than isolated single
#    pixels, and it guarantees the raw foreground count is strictly larger than
#    the active-row/active-column count.
# c) Hidden-assumption audit of the original main():
#    - `np.count_nonzero(upper_left != bg)` and the analogous width line are
#      the only lines that directly encode the old D1 behavior; they must be
#      replaced by active-row / active-column counts.
#    - The quadrant split, background color assumption, stencil-empty branch,
#      and coarse fill substitution do not depend on whether dimensions came
#      from totals or from occupied lines, so they remain valid.
#    - A subtle consequence of the new rule is that out_h <= Q and out_w <= Q.
#      Therefore `r // Q` and `c // Q` never exceed 0 in valid samples. This
#      does not break the original bottom-half logic, but it means the fill tile
#      is no longer required to support multiple coarse blocks. The generator
#      therefore ensures the visible top-left crop of the lower-left tile is
#      informative, because that is the portion the new outputs actually use.
# d) Placement legality:
#    The quadrant partition is unchanged, so there is no cross-object overlap
#    problem beyond keeping each control pattern inside its own QxQ area.
#    Active-row controls only paint selected rows; active-column controls only
#    paint selected columns; the lower-left tile stays fully non-black; the
#    stencil stays inside its quadrant and may be all black.
# e) Visual observability:
#    The user should be able to infer "count occupied rows / columns" from a
#    few examples. To make that possible, the control shapes are generated as
#    horizontal bands or row-wise segments in UL and vertical bands or
#    column-wise segments in UR, so line occupancy is visible by inspection.
#    We also keep at least one widened row and one widened column in every
#    sample, making the old total-pixel-count reading visibly incorrect.
# f) Visual cleanliness:
#    Each quadrant still has a single role and a clean border given by the
#    2x2 split. No new markers or overlapping symbols are introduced. The
#    bottom half preserves the original repeated-pattern aesthetic.
# g) Color inference:
#    Colors are not part of the new rule. As in the source task, control colors
#    are chosen outside the output palette so they are clearly instructional,
#    not to be copied into the output.
# New solver assumptions introduced by the variant:
#    - UL must contain at least one non-black row and UR at least one non-black
#      column. The generator guarantees out_h >= 1 and out_w >= 1.
#    - Because dimensions are capped by Q, the generator restricts grid sizes
#      to N in {8, 16}. This keeps Q >= 4, so there is enough room to show
#      multi-cell rows/columns and keep the rule visually legible.


def _count_active_rows(tile: np.ndarray, bg: int) -> int:
    """Return the number of rows containing at least one non-background cell."""
    return int(np.count_nonzero(np.any(tile != bg, axis=1)))


def _count_active_cols(tile: np.ndarray, bg: int) -> int:
    """Return the number of columns containing at least one non-background cell."""
    return int(np.count_nonzero(np.any(tile != bg, axis=0)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode output dimensions from occupied control lines, then render the lower pattern.

    Input structure:
    - The grid is an even square divided into four equal quadrants.
    - Black is the background color.
    - The upper-left and upper-right quadrants are control panels.
    - The lower-left quadrant is an all-foreground fill tile, while the lower-right quadrant is either an all-black direct-copy signal or a stencil tile.

    Rule:
    - The output height equals the number of rows in the upper-left quadrant that contain at least one non-black pixel.
    - The output width equals the number of columns in the upper-right quadrant that contain at least one non-black pixel.
    - If the lower-right quadrant is all black, copy the lower-left tile pattern over the output by modular repetition.
    - Otherwise tile the lower-right stencil over the output, and replace each black stencil cell with the lower-left value chosen by coarse block coordinates.

    Key insight:
    - The top half now encodes size by line occupancy rather than by raw pixel totals; the bottom half still provides the texture that gets copied or used to fill stencil holes.
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

    out_h = _count_active_rows(upper_left, bg)
    out_w = _count_active_cols(upper_right, bg)
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


# --------------- generator (o5x) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)                   # Q must be >= 4 so line occupancy is visible
STENCIL_EMPTY_PROB = 0.16              # chance lower-right is all-black
LL_SOLID_PROB = 0.70                   # chance fill tile is single-color
LL_MULTI_COLOR_RANGE = (2, 3)          # color count when fill tile is multi-color
STENCIL_SINGLE_COLOR_PROB = 0.75       # chance stencil uses one fg color
STENCIL_MULTI_COLOR_RANGE = (2, 4)     # color count when stencil is multi-color
STENCIL_FG_FRAC = (0.30, 0.75)         # fg fraction range for random stencil mode
STENCIL_FRAME_MIN_Q = 4                # min quadrant size to allow frame stencil
CONTROL_SINGLE_COLOR_PROB = 0.85       # chance control quadrant uses one color
CONTROL_RECT_PROB = 0.85               # chance control lines share one common segment
UL_MAX_COLORS = 3                      # max distinct colors in upper-left control
UR_MAX_COLORS = 2                      # max distinct colors in upper-right control
OUTPUT_DIM_MIN = 3                     # minimum output height/width
OUTPUT_DIM_MAX = 8                     # line-count encoding caps each dimension at Q <= 8
OUTPUT_AREA_MAX = 256                  # maximum output area
GEN_MAX_RETRIES = 300                  # retry budget for generate_input


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


def _make_lower_left(q: int, palette: list[int],
                     visible_coords: list[tuple[int, int]] | None = None) -> np.ndarray:
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
    elif mode == "diagonal":
        offset = random.randrange(k)
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [c for c in palette if c not in existing]
    if missing:
        fix_coords = list(visible_coords) if visible_coords is not None else \
            [(r, c) for r in range(q) for c in range(q)]
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


def _paint_line(tile: np.ndarray, axis: int, line: int,
                positions: list[int], palette: list[int]) -> None:
    """Paint one active row or column with random control colors."""
    if axis == 0:
        for c in positions:
            tile[line, c] = random.choice(palette)
    else:
        for r in positions:
            tile[r, line] = random.choice(palette)


def _make_line_control(q: int, line_count: int, max_colors: int,
                       pool: list[int], axis: int) -> np.ndarray:
    """
    Build a control quadrant whose meaning is active rows (axis=0) or
    active columns (axis=1).
    """
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


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the solver's tiling logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


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
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else \
            random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = min(max(OUTPUT_DIM_MIN, 1), q)
    hi = min(OUTPUT_DIM_MAX, q)
    valid_dims = [(h, w) for h in range(lo, hi + 1) for w in range(lo, hi + 1)
                  if h * w <= OUTPUT_AREA_MAX]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    visible_fill = [(r, c) for r in range(out_h) for c in range(out_w)]
    fill_tile = _make_lower_left(q, ll_palette, visible_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_line_control(q, out_h, UL_MAX_COLORS, control_pool, axis=0)
    upper_right = _make_line_control(q, out_w, UR_MAX_COLORS, control_pool, axis=1)

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
    if int(np.count_nonzero(upper_left != BLACK)) == out_h:
        return None
    if int(np.count_nonzero(upper_right != BLACK)) == out_w:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
