# Combo variant of f931b4a8 (index 13)
# combo_spec: D6/2+D12/2
# Name: 前景背景互换+颜色角色对换

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

# Design analysis for the D6/2 + D12/2 combo
# a) Concrete behavior and trace:
#    - The four quadrants keep their original roles: UL counts output height, UR counts
#      output width, LL is the coarse color tile, LR is the repeated template.
#    - The combo turns LR into an inverted binary mask with a single visible foreground
#      color F. LR non-black cells no longer stay visible; they are overwritten by LL's
#      coarse color. LR black cells no longer fetch LL colors; they become F instead.
#    - Example with q=2, LL=[[2,3],[4,5]], LR=[[7,0],[0,7]], out_h=3, out_w=4:
#      original first 2x4 output rows are [[7,2,7,3],[2,7,3,7]];
#      combo first 2x4 output rows are    [[2,7,3,7],[7,2,7,3]].
#      The coarse lookup (r//q, c//q) is unchanged; only the roles of LR's two states
#      and the visible color source are swapped. Position-by-position, this is the
#      negative of the original rule.
# b) Complexity calibration:
#    - A single LR foreground color is required. Otherwise "keep the template foreground
#      color on zero cells" becomes ambiguous and humans cannot infer a stable swap.
#    - Generator samples avoid empty LR, require both zero and non-zero LR cells, keep
#      F outside LL's palette, and force the output to touch at least two coarse LL cells.
#      These constraints make the inversion visible instead of collapsing into a flat tile.
# c) Implicit-assumption audit:
#    - Counting UL/UR non-black pixels still works unchanged.
#    - The old solver assumed LR non-black colors survive into the output; that is false
#      now. In the combo, LR non-black is only a selector and must be overwritten by LL.
#    - The old "if LR is black, use LL" branch must be inverted to "if LR is non-black,
#      use LL"; otherwise the code silently reverts to the base task.
#    - A new assumption appears: non-empty LR has exactly one non-black color F. The
#      generator guarantees this before calling main().
#    - LL must remain fully non-black because every selected coarse lookup is still read.
# d) Placement legality:
#    - Quadrants remain fixed and non-overlapping inside an even square.
#    - Output dimensions stay in [3, 16] with area <= 256 and each dimension <= q*q, so
#      coarse indices r//q and c//q always stay within LL's qxq bounds.
#    - LR masks are generated with both states present, so no sample degenerates into an
#      all-F or all-LL output.
# e) Visual observability:
#    - Humans can see the swap because LR islands now show LL colors while LR holes show
#      one uniform template color. That is the exact opposite of the original examples.
#    - LL and LR palettes are kept disjoint, so the swapped roles do not blur together.
# f) Visual cleanliness:
#    - Top controls remain separate from bottom content, preserving the original clean
#      four-quadrant layout. LL uses simple stripe/diagonal motifs and LR uses simple
#      checker/stripe/frame/scatter masks, so the relationship stays legible.
# g) Inference of the color change:
#    - Only one LR foreground color participates in the swap, and LL uses at most three
#      colors. This keeps the palette small enough that a solver can infer "LL colors
#      appear where LR is non-black; LR's foreground color appears where LR is black"
#      from a few training pairs.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Size an output from the upper controls and invert how the lower template and color tile interact.

    Input structure:
    - The input is an even square split into four equal quadrants.
    - Black is the only background color.
    - The upper-left and upper-right quadrants encode output height and width by foreground count.
    - The lower-left quadrant is a fully non-black coarse color tile, and the lower-right quadrant is a repeated single-foreground template or an empty tile.

    Rule:
    - Output height is the number of non-black cells in the upper-left quadrant.
    - Output width is the number of non-black cells in the upper-right quadrant.
    - If the lower-right template is empty, tile the lower-left quadrant directly across the output.
    - Otherwise, repeat the lower-right template over the output; template non-black cells are replaced by the lower-left color at the corresponding coarse tile index, while template black cells become the template's unique foreground color.

    Key insight:
    - The lower-right quadrant still controls a fine repeated structure, but its visible and hidden states have been inverted: the template now says where LL colors appear, and the template's own foreground color fills the complementary positions.
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


def _combo_metrics(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> dict[str, float]:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled_mask = stencil[rr % q, cc % q] != BLACK
    combo = _expected_output(fill_tile, stencil, out_h, out_w)
    legacy = _legacy_output(fill_tile, stencil, out_h, out_w)
    coarse_cells = {(int(r // q), int(c // q)) for r, c in zip(rr[tiled_mask], cc[tiled_mask])}
    return {
        "mask_true_cells": float(np.count_nonzero(tiled_mask)),
        "mask_false_cells": float(np.count_nonzero(~tiled_mask)),
        "coarse_cells_used": float(len(coarse_cells)),
        "diff_fraction": float(np.mean(combo != legacy)),
        "output_palette": float(len(np.unique(combo))),
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

    metrics = _combo_metrics(fill_tile, stencil, out_h, out_w)
    if metrics["mask_true_cells"] == 0 or metrics["mask_false_cells"] == 0:
        return None
    if metrics["coarse_cells_used"] < 2:
        return None
    if metrics["diff_fraction"] < MIN_DIFF_FRAC:
        return None
    if metrics["output_palette"] < 2:
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
    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
