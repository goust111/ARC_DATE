# Variant of f931b4a8
# Dimension: D6 — 背景替换规则 (transform)
# Change: 模板中的零值像素被替换为颜色源在粗粒度坐标（瓦片重复索引）处的颜色：fill_color = LL[tile_row][... → 模板前景和背景角色互换

from __future__ import annotations

import random
import sys
from math import ceil
from pathlib import Path

import numpy as np

SOLVER_DIR = str(Path("/home/lixinyu/snap/new/solvers"))
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)
from arc_utils import Color

# Design analysis for the D6 variant
# a) Concrete behavior and trace:
#    - The input still has four equal quadrants. UL and UR only count output height/width.
#    - LL is still a non-black coarse color tile.
#    - LR is now treated as a binary-looking mask with one default foreground color F.
#      Non-zero cells in LR mark where LL coarse colors should appear in the output.
#      Zero cells in LR no longer mean "holes"; they become the visible template color F.
#    - Example with q=2, LL=[[2,3],[4,5]], LR=[[7,0],[0,7]], out_h=3, out_w=4:
#      original logic on the first 2x2 block gives [[7,2],[2,7]];
#      new logic gives [[2,7],[7,2]].
#      The same coarse LL lookup (r//q, c//q) is preserved; only which stencil state
#      receives that color is inverted.
# b) Complexity calibration:
#    - If LR had multiple non-zero colors, "the template's default foreground color"
#      would become ambiguous. To keep the inversion learnable, generated LR tiles use
#      exactly one non-black color plus black.
#    - Generator rejects empty LR in normal samples, forces both zero and non-zero LR
#      cells, and requires the output to span at least two coarse LL cells so the fill
#      mechanism is visible rather than collapsing into a flat recolor.
# c) Implicit-assumption audit against the original solver:
#    - Counting UL/UR non-black pixels is unchanged.
#    - The old branch "if stencil cell is black: fill from LL" is no longer valid.
#      It must be inverted to "if stencil cell is non-black: fill from LL".
#    - Keeping the raw non-zero stencil color at those sites would silently reintroduce
#      the original behavior, so the solver must overwrite non-zero stencil cells.
#    - A new assumption appears: non-empty LR has a single default foreground color.
#      The generator guarantees that invariant.
#    - The old empty-stencil fallback is retained for robustness, but the generator
#      nearly eliminates it because it hides the variant's core effect.
# d) Placement legality:
#    - Quadrants remain fixed, non-overlapping, and aligned on the NxN midpoint.
#    - LL stays fully non-black, so every coarse lookup is legal.
#    - Output dimensions remain <= q*q, hence fill indices r//q and c//q stay in-bounds.
# e) Visual observability:
#    - LR always contains both zeros and non-zeros, so the foreground/background swap is
#      visible in every generated non-empty sample.
#    - LR's foreground color is chosen outside LL's palette, preventing the swapped roles
#      from becoming visually indistinguishable.
# f) Visual cleanliness:
#    - The task still separates controls (top) from content (bottom), so there is no
#      overlap ambiguity. LR patterns are simple checker/stripe/frame/scatter masks.
# g) Color inference:
#    - Only one LR foreground color is used, while LL uses at most three colors.
#      This keeps the output palette small and makes the "mask color vs fill colors"
#      relationship inferable from a few examples.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the two upper control quadrants to size an output assembled from the lower quadrants.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the only background color.
    - The upper-left and upper-right quadrants are count-only controls for output height and width.
    - The lower-left quadrant is a non-black coarse color tile, and the lower-right quadrant is an optional single-color mask tile.

    Rule:
    - Output height equals the number of non-black pixels in the upper-left quadrant.
    - Output width equals the number of non-black pixels in the upper-right quadrant.
    - If the lower-right mask is empty, repeat the lower-left quadrant directly across the output.
    - Otherwise, tile the lower-right mask over the output; mask non-black cells are replaced by the lower-left color at the corresponding coarse tile index, while mask black cells become the mask's default foreground color.

    Key insight:
    - The lower-right quadrant no longer contributes its visible pattern at non-black cells; instead it acts as an inverted selector that decides where the coarse lower-left colors appear and where the mask's single foreground color remains.
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

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % qh, c % qw])
        return out

    fg_colors = np.unique(stencil[stencil != bg])
    assert fg_colors.size == 1
    stencil_fg = int(fg_colors[0])

    for r in range(out_h):
        for c in range(out_w):
            mask_value = int(stencil[r % qh, c % qw])
            if mask_value != bg:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            else:
                value = stencil_fg
            out[r, c] = value

    return out


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
LL_SOLID_PROB = 0.35
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
MIN_DIFF_FRAC = 0.20
GEN_MAX_RETRIES = 400


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


def _make_stencil(q: int, fg_color: int) -> np.ndarray:
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = fg_color
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t:q - t, t:q - t] = False
        tile[mask] = fg_color
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = fg_color
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = fg_color
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = min(area - 1, max(lo, int(STENCIL_FG_FRAC[1] * area)))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = fg_color

    assert np.any(tile == BLACK)
    assert np.any(tile == fg_color)
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
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    fg = int(np.unique(stencil[stencil != BLACK])[0])
    out = np.full((out_h, out_w), fg, dtype=int)
    mask = stencil[rr % q, cc % q] != BLACK
    out[mask] = fill_tile[rr[mask] // q, cc[mask] // q]
    return out


def _legacy_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _variant_metrics(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> dict[str, int]:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled_mask = stencil[rr % q, cc % q] != BLACK
    coarse_cells = {(int(r // q), int(c // q)) for r, c in zip(rr[tiled_mask], cc[tiled_mask])}
    return {
        "mask_true_cells": int(np.count_nonzero(tiled_mask)),
        "mask_false_cells": int(np.count_nonzero(~tiled_mask)),
        "coarse_cells_used": len(coarse_cells),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    if stencil_empty:
        stencil_color = None
        stencil_palette = []
    else:
        remaining = [c for c in COLORS if c not in ll_palette]
        if not remaining:
            return None
        stencil_color = random.choice(remaining)
        stencil_palette = [stencil_color]

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (h, w)
        for h in range(lo, hi + 1)
        for w in range(lo, hi + 1)
        if h * w <= OUTPUT_AREA_MAX
        and (stencil_empty or ceil(h / q) * ceil(w / q) >= max(2, ll_k))
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_color)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    legacy = _legacy_output(fill_tile, stencil, out_h, out_w)
    if np.array_equal(expected, legacy):
        return None
    if float(np.mean(expected != legacy)) < MIN_DIFF_FRAC:
        return None

    metrics = _variant_metrics(fill_tile, stencil, out_h, out_w)
    if metrics["mask_true_cells"] == 0 or metrics["mask_false_cells"] == 0:
        return None
    if metrics["coarse_cells_used"] < 2:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

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
