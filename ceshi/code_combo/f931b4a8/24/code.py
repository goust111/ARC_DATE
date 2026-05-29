# Combo variant of f931b4a8 (index 24)
# combo_spec: D6/0+D13/0
# Name: 两轮铺贴+局部坐标替换

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - The input is still an even square split into four equal quadrants.
#    - UL and UR still encode only output height and width via non-black counts.
#    - LL is still a fully colored qxq source tile and LR is still a qxq stencil.
#    - The fused change is procedural:
#      1. First round: tile LL directly over the whole output area, producing a
#         fine-grained base texture.
#      2. Second round: tile LR over that base. LR foreground overwrites the base.
#         LR background does not use coarse block indices; instead it keeps the
#         first-round color at the same local stencil coordinate.
#    - Small trace with q=2, out size 4x4:
#         LL = [[2, 3],
#               [4, 5]]
#         LR = [[8, 0],
#               [0, 7]]
#      Round 1 base:
#         [[2, 3, 2, 3],
#          [4, 5, 4, 5],
#          [2, 3, 2, 3],
#          [4, 5, 4, 5]]
#      Round 2 overlay repeats LR. At (1,0) LR is a hole, so the combo keeps the
#      first-round local color 4. At (0,0) LR is 8, so 8 overwrites 2.
#      The original task would have filled the hole at (1,0) from coarse index
#      LL[0,0]=2, so the combo differs from the original exactly on repeated hole
#      positions, while also differing from round 1 on stencil-foreground cells.
# b) Complexity calibration
#    - If LR were empty, the second round would disappear. The generator therefore
#      always uses a non-empty stencil with both holes and foreground.
#    - If LL were solid or too symmetric, the local-coordinate replacement would
#      look identical to the old coarse fill. The generator therefore forces LL to
#      use 2-3 colors and rejects cases where the combo output matches the original.
#    - Output dimensions are chosen larger than one stencil period whenever possible
#      so repeated evidence appears across multiple copies instead of one-off cells.
# c) Hidden assumptions review
#    - The quadrant split and square-even input assumption still hold unchanged.
#    - `fill_tile` must contain no black cells because round 1 tiles it directly;
#      the generator guarantees this.
#    - The combo no longer needs the original coarse-lookup bounds
#      `fill_tile[r // q, c // q]` to explain hole colors. Instead it assumes that
#      local coordinates `r % q, c % q` are valid inside LL and LR, which is always
#      true for qxq quadrants.
#    - The explicit two-round implementation avoids silently collapsing the rule
#      into a one-pass description in the code, making the D13 contribution visible.
# d) Placement legality
#    - All quadrants remain fixed and disjoint.
#    - Control quadrants are generated with exactly the requested foreground counts.
#    - LL is always qxq and fully non-black.
#    - LR is always qxq, non-empty, and has at least one hole, so both overlay
#      states are present and every array access stays in bounds.
# e) Visual observability
#    - A human can see the first-round LL texture because stencil holes repeatedly
#      expose it, and can see the second-round overlay because LR foreground uses
#      colors disjoint from LL and stamps a repeated mask on top of that texture.
#    - The generator rejects samples with too few changed cells versus round 1 or
#      versus the original coarse-fill rule, so both fused ingredients remain visible.
# f) Visual cleanliness
#    - Control colors are chosen outside the output palette.
#    - LL patterns are limited to rows, columns, diagonals, or checker-like layouts
#      with small accents, keeping the repeated base structured rather than noisy.
#    - LR foreground masks are simple stripe/checker/frame/random patterns and never
#      cover the whole stencil, so holes and overwrites stay easy to separate.
# g) Change inferability
#    - The same hole coordinate inside each repeated stencil tile reveals the same
#      LL local color, giving a stable positional cue.
#    - The same foreground coordinate inside each repeated stencil tile overwrites
#      that base with the stencil color, making the two-round logic visually legible.
# h) New assumptions introduced by the combo and how the generator satisfies them
#    - The solver assumes the second round matters: generator forces LR to have both
#      foreground and holes, and requires a minimum overwrite count.
#    - The solver assumes the local-coordinate rule matters: generator measures the
#      delta against the original coarse rule and rejects degenerate cases.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper controls to size an output built in two rounds from the lower tiles.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the background color.
    - The upper-left and upper-right quadrants are control regions whose non-black counts set the output height and width.
    - The lower-left quadrant is a fully colored source tile, and the lower-right quadrant is a stencil tile.

    Rule:
    - Output height equals the number of non-black cells in the upper-left quadrant.
    - Output width equals the number of non-black cells in the upper-right quadrant.
    - First tile the lower-left quadrant directly across the whole output to form a base layer.
    - Then tile the lower-right stencil over that base: stencil foreground overwrites the base, while stencil background keeps the base color from the same local coordinate inside the repeated qxq tile.

    Key insight:
    - The top half controls only the output size, while the bottom half defines a two-stage texture: a repeated colored substrate from the lower-left quadrant and a repeated stencil from the lower-right quadrant that selectively reveals or replaces that substrate at matching local coordinates.
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
    assert np.all(fill_tile != bg)

    base = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = int(fill_tile[r % qh, c % qw])
            assert value != bg
            base[r, c] = value

    if not np.any(stencil != bg):
        return base

    out = base.copy()
    for r in range(out_h):
        for c in range(out_w):
            stencil_value = int(stencil[r % qh, c % qw])
            if stencil_value != bg:
                out[r, c] = stencil_value
            else:
                out[r, c] = int(base[r % qh, c % qw])

    return out


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
MIN_OVERLAY_DELTA = 2
GEN_MAX_RETRIES = 500
STENCIL_MAX_RETRIES = 60


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
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
    """Build a readable non-black tile with asymmetric local variation."""
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

    coords = list(preferred_coords) if preferred_coords is not None else [
        (r, c) for r in range(q) for c in range(q)
    ]
    random.shuffle(coords)

    accent_count = min(len(coords), max(2, k))
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
    """Build a non-empty stencil with both holes and foreground."""
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
        mask[t : q - t, t : q - t] = False
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

    assert np.any(tile == BLACK)
    assert np.any(tile != BLACK)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
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
        tile[r0 : r0 + h, c0 : c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _tile_fill_round(fill_tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    return fill_tile[rr % q, cc % q]


def _expected_output_combo(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Two-round oracle: first tile LL, then overlay the stencil."""
    base = _tile_fill_round(fill_tile, out_h, out_w)
    if not np.any(stencil != BLACK):
        return base
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled_stencil = stencil[rr % q, cc % q]
    out = base.copy()
    fg_mask = tiled_stencil != BLACK
    out[fg_mask] = tiled_stencil[fg_mask]
    return out


def _expected_output_original(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Original task oracle, kept to reject combo samples that hide D6."""
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _combo_metrics(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> dict[str, int]:
    combo = _expected_output_combo(fill_tile, stencil, out_h, out_w)
    original = _expected_output_original(fill_tile, stencil, out_h, out_w)
    base = _tile_fill_round(fill_tile, out_h, out_w)
    return {
        "variant_delta": int(np.count_nonzero(combo != original)),
        "overlay_delta": int(np.count_nonzero(combo != base)),
        "hole_local_colors": int(
            len(
                {
                    int(fill_tile[r, c])
                    for r, c in map(tuple, np.argwhere(stencil == BLACK))
                }
            )
        ),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2

    ll_k = random.randint(*LL_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
        *STENCIL_MULTI_COLOR_RANGE
    )
    output_reserved = set(ll_palette)
    stencil_palette = _palette(st_k, [c for c in COLORS if c not in output_reserved])

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool or not stencil_palette:
        return None

    lo = max(OUTPUT_DIM_MIN, q + 1 if q * q > q else q)
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
        if len(hole_coords) < 2:
            continue

        fill_tile = _make_lower_left(q, ll_palette, preferred_coords=hole_coords)
        metrics = _combo_metrics(fill_tile, stencil, out_h, out_w)
        if metrics["variant_delta"] < MIN_VARIANT_DELTA:
            continue
        if metrics["overlay_delta"] < MIN_OVERLAY_DELTA:
            continue
        if metrics["hole_local_colors"] < 2:
            continue

        upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
        upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

        grid = np.zeros((n, n), dtype=int)
        grid[:q, :q] = upper_left
        grid[:q, q:] = upper_right
        grid[q:, :q] = fill_tile
        grid[q:, q:] = stencil

        expected = _expected_output_combo(fill_tile, stencil, out_h, out_w)
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
