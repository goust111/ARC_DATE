# Variant of f931b4a8
# Dimension: D6 — 背景替换规则 (transform)
# Change: 模板中的零值像素被替换为颜色源在粗粒度坐标（瓦片重复索引）处的颜色：fill_color = LL[tile_row][... → 背景替换为颜色源在模板内局部坐标处的颜色

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis for the D6 variant
# a) Concrete behavior and trace
#    - The input is still an even square split into four equal quadrants.
#    - UL/UR still encode only output height/width by counting non-black cells.
#    - LL is still an all-non-black color tile and LR is still a stencil tile.
#    - The changed step is only how LR's black cells are resolved when LR is non-empty.
#    - Original rule on a 4x4 output with q=2:
#         LL = [[2, 3],
#               [4, 5]]
#         stencil = [[8, 0],
#                    [0, 8]]
#      At output cell (r=0,c=3), the stencil cell is black.
#      Old fill uses coarse tile index: (r//2,c//2)=(0,1) -> LL[0,1]=3.
#      New fill uses local stencil coordinate: (r%2,c%2)=(0,1) -> LL[0,1]=3.
#      At (r=1,c=2), old fill is LL[0,1]=3, but new fill is LL[1,0]=4.
#      So the difference shows up exactly where output repeats the stencil and a hole
#      exposes LL using local coordinates rather than repeated-tile coordinates.
# b) Complexity calibration
#    - If LR were empty, or LL were solid, or the output fit within one qxq period,
#      the D6 change would be invisible. The generator therefore always uses a
#      non-empty stencil with holes, requires LL to use 2-3 colors, and requires
#      both output dimensions to exceed one stencil period.
#    - LL patterns stay structured (rows/cols/diagonal/checker with small accents)
#      so the result is readable instead of noisy.
# c) Hidden assumptions in main()
#    - `qh == qw` still holds because the whole grid remains an even square.
#    - The empty-stencil branch can still tile LL directly because LL is guaranteed
#      to contain no black cells. The generator keeps that invariant.
#    - In the non-empty branch, the old code assumed `r//qh` and `c//qw` stayed
#      within LL because LL doubled as a coarse lookup table. That assumption is no
#      longer needed. The new branch instead assumes only that LL covers one stencil
#      period, so `r % qh` and `c % qw` are valid local coordinates.
# d) Placement legality
#    - All four quadrants still occupy disjoint fixed regions of the input square.
#    - Control quadrants are built with exactly the requested foreground counts.
#    - LL is qxq and fully non-black. LR is qxq, non-empty, and has at least one hole.
#    - The assembled grid is therefore always valid for the solver's quadrant split.
# e) Visual observability
#    - The generator rejects samples where the new local-fill output equals the old
#      coarse-fill output. It measures `variant_delta = #cells where the two outputs
#      differ` and requires at least two changed output cells.
#    - This forces the sample set to actually express the D6 rule rather than merely
#      remaining compatible with the original solver.
# f) Visual cleanliness
#    - Control quadrants use colors disjoint from LL/LR output colors.
#    - LL uses only 2-3 colors and LR uses 1-3 colors, keeping the pattern readable.
#    - LR foreground remains a simple checker/stripe/frame/random mask, so the viewer
#      can separate "stencil pixels" from "holes showing LL-local colors".
# g) Color-change inferability
#    - Color changes are driven only by visible position inside the repeated stencil
#      cell. The same local hole coordinate repeats the same LL color in every copy,
#      giving a direct structural cue for the new mapping rule.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the upper quadrants as size controls and synthesize the output from the lower tiles.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the background color.
    - The upper-left and upper-right quadrants are control regions whose foreground counts set the output height and width.
    - The lower-left quadrant is a fully colored source tile, while the lower-right quadrant is a stencil tile that may also be empty.

    Rule:
    - Output height equals the number of non-black cells in the upper-left quadrant.
    - Output width equals the number of non-black cells in the upper-right quadrant.
    - If the lower-right quadrant is empty, repeat the lower-left tile directly across the whole output.
    - Otherwise, repeat the stencil over the output; whenever a repeated stencil cell is black, replace it with the lower-left color at the same local coordinate inside the qxq tile, using LL[r % q][c % q].

    Key insight:
    - The top half determines only the output dimensions, and the bottom half defines a repeated texture where the stencil chooses which cells show its own foreground and which cells reveal the lower-left tile's local within-period color pattern.
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
            for c in range(out_w):
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == bg:
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
            out[r, c] = value

    return out


# --------------- generator (D6 variant) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
LL_COLOR_RANGE = (2, 3)                # multi-color LL is required for D6 visibility
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
MIN_VARIANT_DELTA = 2                  # reject samples identical to the old coarse rule
GEN_MAX_RETRIES = 400
STENCIL_MAX_RETRIES = 40


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
    preferred_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build a fully colored local tile with readable but asymmetric variation."""
    k = len(palette)
    assert k >= 2
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

    coords = preferred_coords if preferred_coords is not None else [(r, c) for r in range(q) for c in range(q)]
    coords = list(coords)
    random.shuffle(coords)

    # Add a small number of accents to break symmetries that would hide the D6 change.
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
    """Build a stencil tile (black holes expose LL's local coordinates)."""
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
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
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
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output_local(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Oracle for the D6 local-coordinate fill rule."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] % q, cc[bg_mask] % q]
    return tiled


def _expected_output_coarse(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Original task oracle, kept only to reject degenerate variant samples."""
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
    """Count output cells changed by the D6 rule relative to the original task."""
    local = _expected_output_local(fill_tile, stencil, out_h, out_w)
    coarse = _expected_output_coarse(fill_tile, stencil, out_h, out_w)
    return int(np.count_nonzero(local != coarse))


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid D6 sample; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2

    ll_k = random.randint(*LL_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
    stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q + 1)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (h, w)
        for h in range(lo, hi + 1)
        for w in range(lo, hi + 1)
        if h * w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    for _ in range(STENCIL_MAX_RETRIES):
        stencil = _make_stencil(q, stencil_palette)
        hole_coords = [tuple(map(int, rc)) for rc in np.argwhere(stencil == BLACK)]
        if not hole_coords or len(hole_coords) < ll_k:
            continue

        fill_tile = _make_lower_left(q, ll_palette, preferred_coords=hole_coords)
        delta = _variant_delta(fill_tile, stencil, out_h, out_w)
        if delta < MIN_VARIANT_DELTA:
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
    raise RuntimeError("failed to generate a valid D6 input")
