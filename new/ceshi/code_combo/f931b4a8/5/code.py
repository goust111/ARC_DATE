# Combo variant of f931b4a8 (index 5)
# combo_spec: D3/2+D6/1
# Name: 色条图例+行索引循环替换

from __future__ import annotations

from math import ceil
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import Color


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The global layout stays the same: an even square is split into four equal quadrants.
#      Upper-left and upper-right still encode output height/width by non-black counts.
#    - The lower-left quadrant no longer acts like a dense qxq lookup tile. Instead it is mostly black
#      and contains one short 1D legend strip of length 2 or 3, placed at the top-left corner.
#      The strip may be horizontal or vertical; its read order is left-to-right or top-to-bottom.
#    - The lower-right quadrant remains a stencil tile. When tiled over the output, stencil foreground
#      stays unchanged. Each stencil black hole on output row r is filled by legend[r % legend_len].
#    - Small trace: let q=4, legend=[2, 5, 8], and stencil row 0 be [7, 0, 7, 0].
#      Original task hole fill for row 0 depended on coarse block coordinates from LL.
#      D3-only legend variant would assign block colors [2, 5, 8, ...] in coarse row-major order.
#      This combo instead ignores block column and uses the absolute output row:
#      output row 0 holes get 2, row 1 holes get 5, row 2 holes get 8, row 3 holes get 2 again.
#      The visible effect is a stencil laid over horizontal color stripes sourced from the legend.
# b) Complexity calibration
#    - Legend length is restricted to 2-3 colors so the sequence is short enough to infer.
#    - Output height is forced to exceed the legend length, so the row cycle repeats visibly.
#    - Non-empty stencil cases dominate, and the stencil builder guarantees at least one black hole
#      in every stencil row, so the row-based color cycle is actually observable in each output row.
# c) Hidden assumptions replaced
#    - The original solver assumed lower-left was fully non-black and indexable by coarse block row/col.
#      That assumption is invalid here because most of lower-left is black.
#    - `main()` now assumes the non-black cells in lower-left form exactly one contiguous 1D strip whose
#      bounding box is completely non-black. The generator constructs precisely that.
#    - The original empty-stencil fallback tiled lower-left directly per pixel. That is also invalid now.
#      In the combo variant, an empty stencil means the row-cycle itself becomes the whole output.
# d) Placement legality
#    - The legend strip is always entirely inside the lower-left quadrant and anchored at its top-left.
#      No object crosses quadrant boundaries, and the stencil stays wholly inside the lower-right quadrant.
# e) Visual observability
#    - A human can first spot that lower-left is a small legend on black, not a full texture tile.
#      Then, across examples, the stencil holes reveal that output rows follow that legend cyclically.
#      Because the cycle repeats and every row exposes holes, the rule is inferable from a few examples.
# f) Visual cleanliness
#    - There is only one cue in lower-left: a single short strip. The output has only two layers:
#      stencil foreground and row-wise background stripes. This avoids ambiguous multi-object overlap.
# g) Change inferability
#    - Legend colors are reused directly; there is no numeric sorting or hidden color ranking.
#      The only transformation is positional: row index modulo legend length.
# Main() assumptions audited after the change
#    - `_extract_legend` assumes there is at least one non-black source cell, its bbox is 1xn or nx1,
#      and every cell inside that bbox is non-black.
#    - `main()` assumes control counts are positive and that the stencil can be tiled modulo q.
#    - `generate_input()` guarantees these properties and rejects any sample where the combo output
#      collapses to the D3-only block-cycle baseline.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.10
LEGEND_LEN_RANGE = (2, 3)
LEGEND_VERTICAL_PROB = 0.45
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400
STENCIL_BUILD_RETRIES = 40


def _palette(k: int, pool=COLORS) -> list[int]:
    items = list(pool)
    random.shuffle(items)
    return items[:k]


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


def _extract_legend(src: np.ndarray, bg: int) -> list[int]:
    """Read the lower-left legend strip as an ordered color list."""
    coords = np.argwhere(src != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = src[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch != bg)
    assert patch.shape[0] == 1 or patch.shape[1] == 1
    if patch.shape[0] == 1:
        legend = [int(v) for v in patch[0, :]]
    else:
        legend = [int(v) for v in patch[:, 0]]
    assert len(legend) >= 2
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """
    Size an output from the top controls, then overlay a tiled stencil on row-cycled legend colors.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants still act only as height and width controls.
    - The lower-left quadrant is mostly black except for one short legend strip, while the lower-right quadrant is a stencil tile or an empty all-black tile.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - Read the lower-left legend strip in order to obtain a short color sequence.
    - For each output row r, choose the stripe color legend[r % len(legend)]. If the stencil is non-empty, tile it over the output and replace each stencil background pixel on row r with that stripe color. If the stencil is empty, fill the whole row with the stripe color.

    Key insight:
    - The lower-left quadrant no longer stores a full spatial lookup table. It stores only a tiny ordered legend, and the missing stencil pixels are colored by cycling through that legend according to the absolute output row index, producing horizontal stripes underneath the repeated stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    src = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    legend = _extract_legend(src, bg)
    has_stencil = bool(np.any(stencil != bg))
    out = np.empty((out_h, out_w), dtype=int)

    for r in range(out_h):
        stripe_color = int(legend[r % len(legend)])
        if not has_stencil:
            out[r, :] = stripe_color
            continue
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == bg:
                value = stripe_color
            out[r, c] = value

    return out


def _make_source_legend(q: int, legend: list[int]) -> np.ndarray:
    """Build the lower-left source quadrant as a single short legend strip."""
    tile = np.zeros((q, q), dtype=int)
    if q > 2 and random.random() < LEGEND_VERTICAL_PROB:
        tile[: len(legend), 0] = legend
    else:
        tile[0, : len(legend)] = legend
    return tile


def _ensure_row_holes(tile: np.ndarray) -> np.ndarray:
    """Guarantee every stencil row exposes at least one background hole."""
    q = int(tile.shape[0])
    for r in range(q):
        if np.all(tile[r, :] != BLACK):
            c = random.randrange(q)
            tile[r, c] = BLACK
    assert np.all(np.any(tile == BLACK, axis=1))
    assert np.any(tile != BLACK)
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a stencil whose holes are visible on every repeated output row."""
    for _ in range(STENCIL_BUILD_RETRIES):
        tile = np.zeros((q, q), dtype=int)
        mode = random.choice(["checker", "col_stripe", "random", "random"])

        if mode == "checker":
            parity = random.randrange(2)
            mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
            tile[mask] = _random_color_array((int(mask.sum()),), palette)
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

        tile = _ensure_row_holes(tile)
        if np.any(tile != BLACK):
            return tile
    raise AssertionError("failed to build a visible stencil")


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2,
        min(max_colors, count, len(pool)),
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(hh, ww) for hh in range(1, q + 1) for ww in range(1, q + 1) if hh * ww == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        tile[r0 : r0 + hh, c0 : c0 + ww] = _random_color_array((hh, ww), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _expected_output(legend: list[int], stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Oracle for the fused row-cycle rule."""
    q = int(stencil.shape[0])
    has_stencil = bool(np.any(stencil != BLACK))
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        stripe_color = int(legend[r % len(legend)])
        if not has_stencil:
            out[r, :] = stripe_color
            continue
        for c in range(out_w):
            value = int(stencil[r % q, c % q])
            if value == BLACK:
                value = stripe_color
            out[r, c] = value
    return out


def _block_cycle_baseline(legend: list[int], stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """D3-only baseline used to reject combo samples that hide the D6 change."""
    q = int(stencil.shape[0])
    coarse_cols = int(ceil(out_w / q))
    has_stencil = bool(np.any(stencil != BLACK))
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        block_r = r // q
        for c in range(out_w):
            block_c = c // q
            fill_color = int(legend[(block_r * coarse_cols + block_c) % len(legend)])
            if has_stencil:
                value = int(stencil[r % q, c % q])
                if value == BLACK:
                    value = fill_color
            else:
                value = fill_color
            out[r, c] = value
    return out


def _behavior_metrics(legend: list[int], stencil: np.ndarray, out_h: int, out_w: int) -> dict[str, int]:
    combo = _expected_output(legend, stencil, out_h, out_w)
    baseline = _block_cycle_baseline(legend, stencil, out_h, out_w)
    diff_pixels = int(np.count_nonzero(combo != baseline))
    used_legend = len(set(legend) & set(map(int, np.unique(combo))))
    return {
        "diff_pixels_vs_block_cycle": diff_pixels,
        "used_legend_colors": used_legend,
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    legend_hi = min(LEGEND_LEN_RANGE[1], q)
    legend_lo = min(LEGEND_LEN_RANGE[0], legend_hi)
    if legend_lo < 2:
        return None
    legend_len = random.randint(legend_lo, legend_hi)
    legend = _palette(legend_len)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k, [c for c in COLORS if c not in legend])

    output_colors = set(legend) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    min_h = max(OUTPUT_DIM_MIN, legend_len + 2)
    min_w = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(min_h, hi + 1)
        for out_w in range(min_w, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    src = _make_source_legend(q, legend)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = src
    grid[q:, q:] = stencil

    metrics = _behavior_metrics(legend, stencil, out_h, out_w)
    if metrics["used_legend_colors"] != len(legend):
        return None
    if metrics["diff_pixels_vs_block_cycle"] == 0:
        return None

    expected = _expected_output(legend, stencil, out_h, out_w)
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
