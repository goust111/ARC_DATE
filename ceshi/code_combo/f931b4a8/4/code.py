# Combo variant of f931b4a8 (index 4)
# combo_spec: D3/1+D6/0
# Name: 逐像素颜色源+模板内局部坐标替换

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The grid is still split into four equal quadrants. UL and UR still encode
#    only output height and width by counting non-black cells. LL is still an
#    all-non-black source tile, and LR is still a stencil tile whose black cells
#    are holes.
#    The fused change is that LL is no longer read by coarse repeated-block
#    coordinates. Instead, every hole in LR is filled from LL using the same
#    local coordinate inside the qxq stencil period.
#    Example with q=4:
#      LL =
#        1 2 1 2
#        3 4 3 4
#        1 2 1 2
#        3 4 3 4
#      LR has holes at local positions (0,1) and (1,0).
#    Original task on an 8x8 output:
#      all holes inside the top-left repeated 4x4 block use LL[0,0];
#      all holes inside the top-right repeated 4x4 block use LL[0,1].
#      So each repeated block gets nearly uniform hole filling.
#    Combo variant:
#      every copy of the hole at local (0,1) uses LL[0,1]=2;
#      every copy of the hole at local (1,0) uses LL[1,0]=3.
#      Therefore every repeated 4x4 block shows the same internal LL pattern.
# b) Complexity calibration:
#    If LR were empty, if LL were solid, or if the output fit within one qxq
#    period, the fused change could be visually invisible. The generator
#    therefore uses non-empty LR with both foreground and holes, forces LL to be
#    multi-color, and makes both output dimensions exceed one period whenever
#    possible. LL patterns stay simple (rows/cols/diagonal/checker-with-accents)
#    so the local-coordinate rule is readable rather than noisy.
# c) Hidden assumptions audit for main():
#    - Splitting the input into four equal quadrants is still valid.
#    - Counting non-black cells in UL/UR is still the correct way to recover the
#      output size.
#    - The old solver assumed LR holes were resolved by coarse indices
#      (r // q, c // q); that assumption is invalid after fusion and is removed.
#    - The new solver assumes only that LL covers one full qxq local period and
#      contains no black cells. The generator preserves both facts.
#    - The empty-stencil branch remains legal because direct tiling of LL still
#      matches the fused rule when LR has no foreground at all.
# d) Placement legality:
#    All objects live in fixed quadrants, so they never overlap or go out of
#    bounds. Control quadrants are built with exact foreground counts. LL and LR
#    each occupy one qxq quadrant. Output dimensions are capped by the original
#    task bounds, so the solver's indexing always stays in range.
# e) Visual observability:
#    The generator rejects samples unless the fused output differs from the
#    original coarse-fill output in at least two cells. It also requires the set
#    of LL colors exposed through LR holes to contain at least two colors. This
#    makes the local-coordinate behavior directly visible to a human.
# f) Visual cleanliness:
#    Control colors are chosen disjoint from output colors. LL uses only 2-3
#    colors and LR uses only 1-3 colors with structured masks, so the lower half
#    reads as one source tile plus one stencil rather than a cluttered collage.
# g) Change inferability:
#    The repeated cue is positional, not numeric: the same local hole coordinate
#    always reveals the same LL cell in every repeated block. A solver can infer
#    the rule from repeated within-template color structure without needing to
#    inspect absolute color ids.
# h) Fusion-specific non-degeneration checks:
#    D3/1 and D6/0 point to the same operational change here, so the main risk is
#    accidental collapse back to the original task. The generator explicitly
#    measures the difference between local-fill and coarse-fill outputs and
#    rejects samples where the difference is too small.
# i) New solver assumptions versus generator guarantees:
#    - `fill_tile[sr, sc]` must never be black when LR has a hole there:
#      guaranteed because LL is fully non-black.
#    - LR should contain both holes and foreground for the fused rule to be
#      learnable: guaranteed by _make_stencil() and _variant_is_observable().
#    - The output should span more than one repeated block to separate "local
#      coordinate" from "coarse block coordinate": guaranteed by valid_dims.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the output size from the top half and build the output from a repeated lower-half tile pair.

    Input structure:
    - The input is an even square partitioned into four equal quadrants.
    - Black is the background color.
    - The upper-left and upper-right quadrants are control regions whose non-black counts determine the output height and width.
    - The lower-left quadrant is a fully colored source tile, and the lower-right quadrant is a stencil tile that may be empty.

    Rule:
    - Output height is the number of non-black cells in the upper-left quadrant.
    - Output width is the number of non-black cells in the upper-right quadrant.
    - If the lower-right stencil is empty, tile the lower-left source directly across the output.
    - Otherwise, repeat the stencil over the output; repeated stencil foreground stays unchanged, and every repeated stencil background cell is replaced by the lower-left color at the same local coordinate inside the qxq tile, using LL[r % q][c % q].

    Key insight:
    - The top half still provides only dimensions, while the bottom half now acts as one fine-grained repeated template: LR says which local cells come from the stencil itself, and LL supplies the per-pixel colors for the stencil's holes using within-template coordinates rather than coarse block indices.
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
    if not np.any(stencil != bg):
        for r in range(out_h):
            sr = r % qh
            for c in range(out_w):
                value = int(fill_tile[sr, c % qw])
                assert value != bg
                out[r, c] = value
        return out

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


# --------------- generator (combo) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
LL_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
MIN_VARIANT_DELTA = 2
MIN_HOLE_COLOR_COUNT = 2
GEN_MAX_RETRIES = 400
STENCIL_MAX_RETRIES = 40


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    choices = list(pool)
    random.shuffle(choices)
    return choices[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    """Return all coordinates in a qxq grid in a layout-dependent order."""
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows = list(range(q))
        cols = list(range(q))
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
    preferred_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build a fully colored local tile with readable asymmetry."""
    k = len(palette)
    assert 2 <= k <= 3
    tile = np.empty((q, q), dtype=int)
    mode = random.choice(["rows", "cols", "diagonal", "checker"])

    if mode == "rows":
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
    else:
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + random.randrange(k)) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    coords = list(preferred_coords) if preferred_coords is not None else [(r, c) for r in range(q) for c in range(q)]
    random.shuffle(coords)

    # Add a few accents on coordinates that are likely to be exposed by holes.
    accent_count = min(len(coords), max(1, k))
    for i, (r, c) in enumerate(coords[:accent_count]):
        tile[r, c] = palette[(i + 1) % k]

    existing = set(map(int, np.unique(tile)))
    missing = [color for color in palette if color not in existing]
    if missing:
        fix_coords = coords if coords else [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color

    assert np.all(tile != BLACK)
    assert len(np.unique(tile)) >= 2
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a stencil tile with both foreground and holes."""
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
        hi = min(area - 1, max(lo, int(STENCIL_FG_FRAC[1] * area)))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)

    assert np.any(tile != BLACK)
    assert np.any(tile == BLACK)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly count non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2,
        min(max_colors, count, len(pool)),
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
    else:
        coords = _ordered_coords(q, random.choice(["rect", "compact", "rows"]))[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output_local(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Oracle for the fused local-coordinate rule."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled_stencil = stencil[rr % q, cc % q]
    tiled_fill = fill_tile[rr % q, cc % q]
    return np.where(tiled_stencil != BLACK, tiled_stencil, tiled_fill)


def _expected_output_coarse(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Original-task oracle, kept only to reject degenerate combo samples."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _variant_delta(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> int:
    """Count cells changed by the fused rule relative to the original task."""
    local = _expected_output_local(fill_tile, stencil, out_h, out_w)
    coarse = _expected_output_coarse(fill_tile, stencil, out_h, out_w)
    return int(np.count_nonzero(local != coarse))


def _hole_fill_color_count(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> int:
    """Count distinct LL colors visible through repeated stencil holes."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    hole_mask = stencil[rr % q, cc % q] == BLACK
    colors = fill_tile[rr[hole_mask] % q, cc[hole_mask] % q]
    return len(set(map(int, colors)))


def _variant_is_observable(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> bool:
    """Require direct visible evidence of local-coordinate filling."""
    if not np.any(stencil != BLACK):
        return False

    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    hole_mask = stencil[rr % q, cc % q] == BLACK
    fg_mask = ~hole_mask
    if not np.any(hole_mask) or not np.any(fg_mask):
        return False

    if _hole_fill_color_count(fill_tile, stencil, out_h, out_w) < MIN_HOLE_COLOR_COUNT:
        return False
    return _variant_delta(fill_tile, stencil, out_h, out_w) >= MIN_VARIANT_DELTA


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid observable combo sample."""
    n = random.choice(GRID_SIZES)
    q = n // 2

    ll_palette = _palette(random.randint(*LL_COLOR_RANGE))
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
    stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q + 1)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    for _ in range(STENCIL_MAX_RETRIES):
        stencil = _make_stencil(q, stencil_palette)
        hole_coords = [tuple(map(int, rc)) for rc in np.argwhere(stencil == BLACK)]
        if len(hole_coords) < MIN_HOLE_COLOR_COUNT:
            continue

        fill_tile = _make_lower_left(q, ll_palette, preferred_coords=hole_coords)
        if not _variant_is_observable(fill_tile, stencil, out_h, out_w):
            continue

        upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
        upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

        grid = np.zeros((n, n), dtype=int)
        grid[:q, :q] = upper_left
        grid[:q, q:] = upper_right
        grid[q:, :q] = fill_tile
        grid[q:, q:] = stencil

        expected = _expected_output_local(fill_tile, stencil, out_h, out_w)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo input")
