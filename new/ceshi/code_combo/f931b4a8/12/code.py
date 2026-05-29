# Combo variant of f931b4a8 (index 12)
# combo_spec: D6/2+D11/1
# Name: 前景背景互换+颜色源覆盖前景

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


# Design analysis for combo D6/2 + D11/1
# --------------------------------------
# a) Concrete behavior and trace:
#    - The 4-quadrant layout stays the same: UL counts output height, UR counts output width,
#      LL is a coarse all-nonblack color source, LR is the repeated template tile.
#    - Original task on one fine cell:
#         stencil != 0 -> keep stencil color
#         stencil == 0 -> use LL[r // Q, c // Q]
#    - D6/2 says foreground/background roles swap inside the template:
#         stencil != 0 -> use LL[r // Q, c // Q]
#         stencil == 0 -> keep the template's default foreground color F
#    - D11/1 says composition also swaps: the color source now covers template foreground,
#      while template background keeps the fixed template color.
#    - These two changes do not cancel. They describe the same final mapping, so the fusion
#      stabilizes on the D6/2-style rule rather than returning to the original.
#    - Small trace with Q=2, LL=[[2,3],[4,5]], LR=[[7,0],[0,7]], out_h=3, out_w=4:
#         original first 2x2 block -> [[7,2],[2,7]]
#         fused first 2x2 block    -> [[2,7],[7,2]]
#      The coarse LL lookup is unchanged; only which template state receives that lookup flips.
#
# b) Complexity calibration:
#    - "Template default foreground color" must be visually unambiguous, so generated LR tiles
#      use exactly one non-black color plus black.
#    - Empty LR would collapse the task back to direct LL tiling and hide the fusion, so normal
#      samples force a non-empty LR with both zero and non-zero cells.
#    - LL uses at most three colors and LR uses one distinct foreground color, keeping the rule
#      legible without creating noisy many-color overlays.
#
# c) Implicit-assumption audit against the original solver:
#    - Counting non-black pixels in UL/UR still works unchanged.
#    - `if value == bg: value = fill_tile[...]` is no longer valid. The fill now belongs on
#      non-black LR cells, not black LR cells.
#    - Reusing the raw LR non-black colors would silently reintroduce the old overlay behavior,
#      so solver output must overwrite every LR foreground cell with an LL coarse color.
#    - The fused rule introduces one new assumption: non-empty LR has a single visible
#      foreground color F whose value is reused on LR background cells in the output.
#      The generator explicitly guarantees that.
#
# d) Placement legality:
#    - Quadrants remain fixed and non-overlapping inside an even square grid.
#    - LL is fully non-black, so every coarse lookup is legal.
#    - Output dimensions remain <= Q*Q, so coarse indices r // Q and c // Q stay in-bounds.
#
# e) Visual observability:
#    - Every non-empty sample contains both LR zeros and LR foreground cells, so the swap is
#      visible in one example instead of requiring cross-example comparison.
#    - LR's foreground color is kept outside LL's palette, making it obvious that some LR cells
#      survive as a fixed color while other LR cells are recolored from LL.
#
# f) Visual cleanliness:
#    - The top half remains pure control information and the bottom half remains pure content.
#      LR patterns are simple checker/stripe/frame/scatter masks rather than dense clutter.
#
# g) Change inferability:
#    - The output always mixes one persistent LR foreground color with 1-3 LL colors, and the
#      mixing is structured by the repeated LR mask plus coarse LL blocks. No hidden ordering by
#      color id is used.
#
# h) Generator-side anti-degeneracy checks:
#    - Reject samples where fused output matches the original task output.
#    - Reject samples where too few pixels differ from the original (<20%).
#    - Reject samples where LR lacks either state, or where filled LR foreground cells touch
#      fewer than two coarse LL cells, because those cases weaken the fusion signal.
#
# i) New implicit assumptions audit:
#    - `main()` assumes LL has no black cells; generator preserves the original invariant.
#    - `main()` assumes non-empty LR has exactly one non-black color; generator enforces that.
#    - `main()` uses coarse coordinates only when LR is non-black; generator keeps output sizes
#      within the original bounds so those coordinates are always valid.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper control quadrants to size an output whose repeated lower-right template swaps which cells keep the template color versus receive coarse lower-left colors.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - Black is the only background color.
    - The upper-left and upper-right quadrants are count-only controls for output height and width.
    - The lower-left quadrant is a non-black coarse color tile, and the lower-right quadrant is a repeated single-foreground template tile.

    Rule:
    - Output height equals the number of non-black pixels in the upper-left quadrant.
    - Output width equals the number of non-black pixels in the upper-right quadrant.
    - If the lower-right template is empty, repeat the lower-left tile directly across the output.
    - Otherwise, tile the lower-right template over the output; template non-black cells are overwritten by the lower-left color at the corresponding coarse tile index, while template black cells become the template's single foreground color.

    Key insight:
    - The lower-right quadrant no longer donates its own foreground pattern to the output. Instead it acts as an inverted mask: its foreground marks where coarse lower-left colors appear, and its background marks where one fixed template color remains.
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


def _fusion_metrics(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> dict[str, int]:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    mask_true = stencil[rr % q, cc % q] != BLACK
    coarse_cells = {(int(r // q), int(c // q)) for r, c in zip(rr[mask_true], cc[mask_true])}
    return {
        "mask_true_cells": int(np.count_nonzero(mask_true)),
        "mask_false_cells": int(np.count_nonzero(~mask_true)),
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
        stencil_palette: list[int] = []
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

    metrics = _fusion_metrics(fill_tile, stencil, out_h, out_w)
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
