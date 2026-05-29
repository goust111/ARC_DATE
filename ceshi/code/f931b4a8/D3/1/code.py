# Variant of f931b4a8
# Dimension: D3 — 颜色源（左下象限）的结构 (src)
# Change: 左下象限作为颜色源瓦片，可以是均匀单色或含有多色行列图案；其像素值在铺贴时按瓦片重复坐标粗粒度填充模板的背景位置... → 颜色源按模板内像素坐标逐像素映射

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the D3 variant
# a) Concrete behavior and trace:
#    The input is still an even square split into four equal quadrants. UL and UR
#    only encode counts. LL is still an all-nonblack source tile. LR is still a
#    stencil whose black cells are "holes".
#    The changed part is the lookup used for those holes.
#    Example with q=4:
#      LL =
#        1 2 1 2
#        1 2 1 2
#        3 4 3 4
#        3 4 3 4
#      stencil has holes at local coords (0,1) and (2,3).
#    Original task:
#      inside the first repeated 4x4 block, both holes use LL[0,0];
#      inside the next block to the right, both holes use LL[0,1].
#      So fill color is constant inside each repeated block, but changes from
#      block to block.
#    Variant:
#      every repeat uses LL[0,1] for the first hole and LL[2,3] for the second.
#      So each repeated block has the same internal multicolor hole pattern.
#    The effect is therefore localized to the stencil-background substitution
#    step; counting controls and quadrant layout remain unchanged.
# b) Complexity calibration:
#    To make the change visible, non-empty-stencil cases use multi-color LL tiles
#    and output dimensions at least q, so one full template is visible. LL still
#    uses a small family of simple patterns (solid/rows/cols/diagonal) to avoid
#    visual noise. Stencil modes remain limited and structured.
# c) Hidden assumptions from the original solver:
#    - Counting non-black pixels in UL/UR is still valid.
#    - Splitting the grid into four equal quadrants is still valid.
#    - The original lines using fill_r = r // q and fill_c = c // q depended on
#      the old coarse lookup rule. That assumption is now wrong because holes in
#      different local stencil cells within the same repeated block may need
#      different LL colors.
#    - The new local lookup assumes every LL cell is non-black; the generator
#      preserves that invariant.
# d) Placement legality:
#    Objects never overlap because the four quadrants are fixed partitions of the
#    square grid. Control quadrants are filled independently with exactly the
#    needed counts. LL and LR occupy full qxq quadrants, so there are no packing
#    or boundary issues beyond choosing valid output sizes.
# e) Visual observability:
#    The generator rejects non-empty cases where the variant output accidentally
#    matches the original coarse-fill output. It also requires at least two hole
#    fill colors to be visible, so the D3 change is not hidden inside a uniform
#    source tile.
# f) Visual cleanliness:
#    The task still reads as four clean quadrants. LL patterns are regular, LR
#    patterns are sparse/striped/checkered/frame-like, and output colors come
#    only from LL and LR, so there is little ambiguity about which region
#    contributes what.
# g) Color inference:
#    No ordering or transformation depends on color ids. The human-solvable cue
#    is purely positional: black stencil cells inherit the LL color at the same
#    local template coordinate. Color variety is capped so the structure remains
#    legible from a few examples.
# h) New solver assumptions checked against the generator:
#    - LL must be all non-black: guaranteed by _make_lower_left().
#    - Non-empty LR should expose both fg and bg over the visible output if we
#      want the variant to be learnable: enforced by _variant_is_observable().
#    - Output dimensions stay <= q*q, which keeps the original task's control
#      count bounds and remains compatible with the solver.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper quadrants to size an output assembled from a repeated lower-half template.

    Input structure:
    - The grid is an even square divided into four equal quadrants.
    - Black is the background color.
    - The two upper quadrants are control regions whose non-black counts set the output size.
    - The lower-left quadrant is an all-nonblack source tile, and the lower-right quadrant is a stencil tile that may be empty.

    Rule:
    - Output height is the number of non-black cells in the upper-left quadrant.
    - Output width is the number of non-black cells in the upper-right quadrant.
    - Repeat the lower-right stencil across the output. Wherever the stencil cell is non-black, copy that stencil color.
    - Wherever the repeated stencil cell is black, use the lower-left source color at the same local stencil coordinate inside the quadrant.

    Key insight:
    - The top half still controls only the output dimensions, but the bottom half now defines a single fine-grained repeated tile: the stencil supplies foreground cells and the source tile supplies per-coordinate background colors.
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

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        sr = r % qh
        for c in range(out_w):
            sc = c % qw
            value = int(stencil[sr, sc])
            if value == bg:
                value = int(fill_tile[sr, sc])
                assert value != bg
            out[r, c] = value

    return out


# --------------- generator (o5x) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.08
LL_SOLID_PROB_IF_STENCIL_EMPTY = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300
VARIANT_DELTA_MIN = 2


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    """Return all (r,c) in a qxq grid in a layout-dependent order."""
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows, cols = list(range(q)), list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar, ac = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_lower_left(q: int, palette: list[int]) -> np.ndarray:
    """Build the source tile (all non-black) using regular small-pattern modes."""
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
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for color, (r, c) in zip(missing, coords):
            tile[r, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black cells are holes filled from the LL tile)."""
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
    assert np.any(tile != BLACK)
    assert np.any(tile == BLACK)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else \
        random.randint(2, min(max_colors, count, len(pool)))
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    """Oracle for the new local-coordinate fill rule."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled_stencil = stencil[rr % q, cc % q]
    tiled_fill = fill_tile[rr % q, cc % q]
    return np.where(tiled_stencil != BLACK, tiled_stencil, tiled_fill)


def _coarse_reference_output(fill_tile: np.ndarray, stencil: np.ndarray,
                             out_h: int, out_w: int) -> np.ndarray:
    """Old-task oracle, kept only to reject degenerate samples."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _variant_is_observable(fill_tile: np.ndarray, stencil: np.ndarray,
                           out_h: int, out_w: int) -> bool:
    """Require visible evidence that local-coordinate filling matters."""
    if not np.any(stencil != BLACK):
        return True

    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    hole_mask = stencil[rr % q, cc % q] == BLACK
    fg_mask = ~hole_mask
    if not np.any(hole_mask) or not np.any(fg_mask):
        return False

    local_fill_colors = set(map(int, fill_tile[rr[hole_mask] % q, cc[hole_mask] % q]))
    if len(local_fill_colors) < 2:
        return False

    variant = _expected_output(fill_tile, stencil, out_h, out_w)
    coarse = _coarse_reference_output(fill_tile, stencil, out_h, out_w)
    delta = int(np.count_nonzero(variant != coarse))
    return delta >= VARIANT_DELTA_MIN


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    if stencil_empty:
        ll_k = 1 if random.random() < LL_SOLID_PROB_IF_STENCIL_EMPTY else \
            random.randint(*LL_MULTI_COLOR_RANGE)
    else:
        ll_k = random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else \
            random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (h, w) for h in range(lo, hi + 1) for w in range(lo, hi + 1)
        if h * w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    fill_tile = _make_lower_left(q, ll_palette)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    if not _variant_is_observable(fill_tile, stencil, out_h, out_w):
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
