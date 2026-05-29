# Variant of f931b4a8
# Dimension: D5 — 铺贴方式 (transform)
# Change: 模板以周期铺贴方式填充输出矩形：output[r][c] = stencil[r % H_stencil][c % W_stencil] → 偏移铺贴（砖墙式错位）

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the D5 variant
# ----------------------------------
# variant_spec:
#   concretization:
#     - The input is still an even square split into four equal quadrants.
#     - The upper quadrants still encode only the output height and width via
#       non-black pixel counts; no new markers are introduced there.
#     - The lower-left quadrant remains an all-non-black coarse color tile.
#     - The lower-right quadrant remains a stencil tile whose black cells are
#       holes. The only rule change is how this stencil repeats across the
#       output: tile rows of height Q alternate between normal alignment and a
#       right shift of Q/2 columns, producing a brick-wall offset.
#     - The fill lookup for stencil holes stays on the original coarse output
#       blocks: fill_tile[r // Q, c // Q]. This keeps D5 isolated to stencil
#       placement rather than changing the color-source dimension too.
#   behavior_trace:
#     - Example with Q=4 and shift=2:
#       even tile row: stencil column at output c is c % 4
#       odd tile row:  stencil column at output c is (c - 2) % 4
#     - Original logic on row r=5, c=1 used stencil[1,1].
#       New logic sees r // 4 == 1 (odd tile row), so it uses stencil[1,3].
#       If that sampled stencil value is black, both old and new rules still
#       use fill_tile[1,0]; only the stencil's placement changed.
#   complexity_calibration:
#     - Empty stencils would hide the transform completely, so the generator
#       keeps solver support for that branch but does not emit it.
#     - Horizontally symmetric or half-periodic stencils would also hide the
#       offset, so generated stencils are rejected if rolling them by Q/2
#       columns leaves them unchanged.
#     - Outputs must span at least two stencil tile rows; otherwise every sample
#       would show only the unshifted row and the variant would collapse back to
#       the original visual behavior.
#   implicit_assumption_audit:
#     - The old `stencil[r % q, c % q]` assumed all stencil copies share one
#       global column alignment. That assumption is false now and must be
#       replaced with a parity-dependent column lookup.
#     - The old `fill_tile[r // q, c // q]` depends only on output size bounds
#       (`out_h`, `out_w` <= q*q), not on stencil alignment, so it remains safe.
#     - A new assumption appears: Q must be even so that Q/2 is an integer half
#       tile shift. The generator guarantees this because Q is drawn from
#       {2, 4, 8}, matching the original task family.
#   placement_legality:
#     - Quadrant assembly is unchanged, so no new overlap or boundary risks are
#       introduced inside the input grid.
#     - The offset acts only in the output rendering stage; sampled fill indices
#       remain in-bounds because output dimensions stay <= Q*Q.
#   visual_observability:
#     - Generator outputs always use a non-empty, offset-sensitive stencil.
#     - Stencil shapes are biased toward asymmetric bars, steps, blocks, and
#       scatter so the odd tile rows visibly slide sideways.
#     - Output widths are forced above Q + Q/2 and heights above Q, ensuring the
#       first shifted tile row is actually visible in the output crop.
#   visual_cleanliness:
#     - The bottom quadrants still separate coarse color and fine stencil roles,
#       so the scene remains legible.
#     - Stencil foreground density stays moderate (roughly 30%-75%) to avoid
#       either all-noise or almost-solid blocks.
#   color_reasoning:
#     - No hidden color ordering or color-ID arithmetic is introduced.
#     - Control colors remain disjoint from output colors, so the human-visible
#       clue stays spatial rather than numeric.


def _shifted_stencil_col(c: int, q: int, tile_row: int) -> int:
    """Return the stencil column after applying the brick offset on odd tile rows."""
    shift = (q // 2) if tile_row % 2 == 1 else 0
    return (c - shift) % q


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper quadrants as size counters and render the lower-right stencil with brick-offset tiling.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the background color.
    - The upper-left and upper-right quadrants are control areas whose non-black counts set the output height and width.
    - The lower-left quadrant is an all-foreground color tile, and the lower-right quadrant is either a stencil tile or fully black.

    Rule:
    - Output height equals the number of non-black pixels in the upper-left quadrant.
    - Output width equals the number of non-black pixels in the upper-right quadrant.
    - If the lower-right quadrant is empty, repeat the lower-left tile directly over the output as before.
    - Otherwise, repeat the stencil in tile rows of height Q; odd tile rows are shifted right by half a tile width, and any sampled black stencil cell is replaced by the coarse color from lower-left at [r // Q, c // Q].

    Key insight:
    - The top half still controls only the rectangle size, but the bottom half now combines a coarse fill source with a fine stencil whose copies alternate between aligned and half-shifted rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw and qw % 2 == 0

    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
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
        tile_row = r // qh
        stencil_r = r % qh
        for c in range(out_w):
            stencil_c = _shifted_stencil_col(c, qw, tile_row)
            value = int(stencil[stencil_r, stencil_c])
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

GRID_SIZES = (4, 8, 16)                # allowed input grid side-lengths N
STENCIL_EMPTY_PROB = 0.0               # keep solver support, but always expose the D5 transform
LL_SOLID_PROB = 0.70                   # chance fill tile is single-color
LL_MULTI_COLOR_RANGE = (2, 3)          # color count when fill tile is multi-color
STENCIL_SINGLE_COLOR_PROB = 0.65       # chance stencil uses one fg color
STENCIL_MULTI_COLOR_RANGE = (2, 4)     # color count when stencil is multi-color
STENCIL_FG_FRAC = (0.30, 0.75)         # fg fraction range for stencil mode
CONTROL_SINGLE_COLOR_PROB = 0.85       # chance control quadrant uses one color
CONTROL_RECT_PROB = 0.85               # chance control pixels form an axis-aligned rect
UL_MAX_COLORS = 3                      # max distinct colors in upper-left control
UR_MAX_COLORS = 2                      # max distinct colors in upper-right control
OUTPUT_DIM_MIN = 3                     # minimum output height/width
OUTPUT_DIM_MAX = 16                    # maximum output height/width
OUTPUT_AREA_MAX = 256                  # maximum output area
GEN_MAX_RETRIES = 300                  # retry budget for generate_input


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


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
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
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(q) for c in range(q)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _rebalance_stencil_density(
    tile: np.ndarray,
    palette: list[int],
    lo: int,
    hi: int,
) -> np.ndarray:
    """Adjust a stencil to stay within the target foreground-density band."""
    q = tile.shape[0]
    fg_coords = [(r, c) for r in range(q) for c in range(q) if tile[r, c] != BLACK]
    bg_coords = [(r, c) for r in range(q) for c in range(q) if tile[r, c] == BLACK]
    fg_count = len(fg_coords)

    if fg_count < lo:
        random.shuffle(bg_coords)
        for r, c in bg_coords[: lo - fg_count]:
            tile[r, c] = random.choice(palette)
    elif fg_count > hi:
        random.shuffle(fg_coords)
        for r, c in fg_coords[: fg_count - hi]:
            tile[r, c] = BLACK
    return tile


def _is_offset_sensitive(tile: np.ndarray) -> bool:
    """True iff shifting by half a tile width changes the stencil."""
    shift = tile.shape[1] // 2
    return not np.array_equal(tile, np.roll(tile, shift=shift, axis=1))


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a stencil whose half-width horizontal shift is visually meaningful."""
    area = q * q
    lo = max(2, int(ceil(STENCIL_FG_FRAC[0] * area)))
    hi = min(area - 1, max(lo, int(ceil(STENCIL_FG_FRAC[1] * area))))

    for _ in range(80):
        tile = np.zeros((q, q), dtype=int)
        mode = random.choice(["bars", "steps", "blocks", "scatter"])

        if mode == "bars":
            bar_cols = random.sample(range(q), k=random.randint(1, max(1, q - 1)))
            for c in bar_cols:
                r0 = random.randint(0, q - 1)
                r1 = random.randint(r0, q - 1)
                tile[r0:r1 + 1, c] = _random_color_array((r1 - r0 + 1,), palette)
            arm_r = random.randrange(q)
            arm_c0 = random.randrange(q)
            arm_len = random.randint(1, max(1, q // 2))
            for dc in range(arm_len):
                c = (arm_c0 + dc) % q
                tile[arm_r, c] = random.choice(palette)

        elif mode == "steps":
            start = random.randrange(q)
            width = random.randint(1, max(1, q // 2))
            drift = random.choice([1, 2]) if q > 2 else 1
            for r in range(q):
                c0 = (start + r * drift) % q
                span = width if random.random() < 0.7 else min(q, width + 1)
                for dc in range(span):
                    c = (c0 + dc) % q
                    if span == 1 or (r + dc) % 2 == 0 or random.random() < 0.35:
                        tile[r, c] = random.choice(palette)

        elif mode == "blocks":
            block_count = random.randint(2, 3)
            for _ in range(block_count):
                bh = random.randint(1, max(1, q // 2))
                bw = random.randint(1, max(1, q // 2))
                r0 = random.randint(0, q - bh)
                c0 = random.randint(0, q - bw)
                tile[r0:r0 + bh, c0:c0 + bw] = _random_color_array((bh, bw), palette)

        else:
            fg_count = random.randint(lo, hi)
            coords = [(r, c) for r in range(q) for c in range(q)]
            random.shuffle(coords)
            for r, c in coords[:fg_count]:
                tile[r, c] = random.choice(palette)

        tile = _rebalance_stencil_density(tile, palette, lo, hi)
        if np.any(tile == BLACK) and np.any(tile != BLACK) and _is_offset_sensitive(tile):
            return tile

    raise RuntimeError("failed to build an offset-sensitive stencil")


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, min(max_colors, count, len(pool)))
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


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reproduce the solver's shifted-tiling logic for oracle validation."""
    q = fill_tile.shape[0]
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        rr, cc = np.mgrid[:out_h, :out_w]
        return fill_tile[rr % q, cc % q]

    for r in range(out_h):
        tile_row = r // q
        stencil_r = r % q
        for c in range(out_w):
            stencil_c = _shifted_stencil_col(c, q, tile_row)
            value = int(stencil[stencil_r, stencil_c])
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = value
    return out


def _expected_output_original(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reference the original non-offset stencil tiling to reject degenerate samples."""
    q = fill_tile.shape[0]
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        rr, cc = np.mgrid[:out_h, :out_w]
        return fill_tile[rr % q, cc % q]

    for r in range(out_h):
        stencil_r = r % q
        for c in range(out_w):
            value = int(stencil[stencil_r, c % q])
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = value
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
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

    height_lo = max(OUTPUT_DIM_MIN, q if stencil_empty else q + 1)
    width_lo = max(OUTPUT_DIM_MIN, q if stencil_empty else q + (q // 2))
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (h, w)
        for h in range(height_lo, hi + 1)
        for w in range(width_lo, hi + 1)
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
    baseline = _expected_output_original(fill_tile, stencil, out_h, out_w)
    if np.array_equal(expected, baseline):
        return None
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
