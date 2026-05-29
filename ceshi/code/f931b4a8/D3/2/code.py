# Variant of f931b4a8
# Dimension: D3 — 颜色源（左下象限）的结构 (src)
# Change: 左下象限作为颜色源瓦片，可以是均匀单色或含有多色行列图案；其像素值在铺贴时按瓦片重复坐标粗粒度填充模板的背景位置... → 颜色源为一个小色条/图例

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the D3 variant
# ==================================
# a) Concrete behavior and trace
#    - The input is still an even square split into four equal quadrants.
#    - Upper-left and upper-right still encode output height/width by non-black counts.
#    - Lower-right is still the fine stencil that gets tiled over the output.
#    - The changed part is lower-left: it is no longer a dense qxq fill tile.
#      Instead it is mostly black and contains exactly one short 1xn or nx1 non-black
#      legend strip in a corner-sized bbox. The strip length is 2 or 3 cells.
#    - Read the legend colors in strip order. Each repeated qxq stencil block gets one
#      background color, assigned in row-major block order and cycled through the legend.
#    - Example with q=4, legend [2, 5, 8], coarse block layout 2x2:
#      old task: block colors came from fill_tile[0,0], fill_tile[0,1], fill_tile[1,0], fill_tile[1,1]
#      new task: block colors are 2, 5, 8, 2 from the legend cycle.
#      The stencil still decides which pixels keep stencil fg and which pixels receive the
#      block color, so the changed behavior is isolated to the "background fill lookup" step.
# b) Complexity calibration
#    - Legend length is restricted to 2-3 distinct colors so humans can infer a cycle.
#    - Output dimensions are chosen so the number of coarse blocks is at least legend_len+1,
#      which forces the cyclic behavior to appear instead of a one-off lookup.
#    - The stencil always has black holes when non-empty, so the legend genuinely affects output.
# c) Hidden assumptions replaced
#    - Original solver assumed every lower-left cell was non-black and indexable by (block_r, block_c).
#      That fails here because most lower-left cells are black.
#    - The new solver instead assumes lower-left contains one contiguous 1D non-black strip whose
#      bbox is fully non-black. This is what the generator constructs.
#    - Original empty-stencil branch tiled lower-left directly per pixel; that is no longer valid.
#      Here an empty stencil means "show only the coarse block colors from the legend cycle".
# d) Placement legality
#    - The legend strip is placed wholly inside the lower-left quadrant, aligned to its top or left edge,
#      never overlapping any other quadrant.
#    - Stencil and control quadrants keep the original placement rules and stay inside bounds.
# e) Visual observability
#    - Lower-left visibly changes from a full tile into a tiny legend strip on a black field.
#    - Because coarse block count exceeds legend length, repeated blocks reveal a clear cyclic order.
#    - Empty-stencil cases remain compatible with the same rule and can make the block cycle even clearer.
# f) Visual cleanliness
#    - Only one simple legend strip appears in lower-left; there is no multi-layer overlap there.
#    - Output remains a tiled stencil with coherent per-block background colors, not random per-pixel noise.
# g) Color inference
#    - Legend colors are exactly the colors cycled into stencil holes; there is no hidden numeric ordering.
#    - The legend uses a short visible sequence, and repeated output blocks provide repeated evidence for it.
# Main() assumptions audited after the change
#    - `_extract_legend` assumes lower-left has exactly one non-black strip with height 1 or width 1.
#    - `main()` assumes control counts are positive and the coarse block color is constant inside each qxq repeat.
#    - `generate_input()` guarantees these properties by construction and rejects any mismatch against `main()`.


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
    Count the two upper control quadrants, then tile a stencil whose hole colors come from a legend strip.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants still act as height and width controls.
    - The lower-left quadrant is mostly black except for one short non-black legend strip; the lower-right quadrant is a stencil tile or empty.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - Read the lower-left legend strip in order. Each repeated qxq stencil block gets one background color, cycling through that legend in row-major block order.
    - Tile the lower-right stencil across the output; stencil foreground stays unchanged, while stencil background is replaced by the current block's legend color. If the stencil is empty, each coarse qxq block is filled uniformly by the legend cycle.

    Key insight:
    - The top half still sets only the output size, but the lower-left quadrant no longer stores a full lookup table. Instead it provides a short ordered color legend that must be cycled over repeated stencil blocks.
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
    coarse_cols = int(ceil(out_w / qw))
    has_stencil = bool(np.any(stencil != bg))

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        block_r = r // qh
        for c in range(out_w):
            block_c = c // qw
            block_idx = block_r * coarse_cols + block_c
            fill_color = int(legend[block_idx % len(legend)])
            if has_stencil:
                value = int(stencil[r % qh, c % qw])
                if value == bg:
                    value = fill_color
            else:
                value = fill_color
            out[r, c] = value

    return out


# --------------- generator (o5x) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.18
LEGEND_LEN_RANGE = (2, 3)
LEGEND_VERTICAL_PROB = 0.45
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


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    items = list(pool)
    random.shuffle(items)
    return items[:k]


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
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_source_legend(q: int, legend: list[int]) -> np.ndarray:
    """Build the lower-left source quadrant as a small horizontal or vertical legend strip."""
    tile = np.zeros((q, q), dtype=int)
    if q > 2 and random.random() < LEGEND_VERTICAL_PROB:
        tile[: len(legend), 0] = legend
    else:
        tile[0, : len(legend)] = legend
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black = holes filled from the legend cycle)."""
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
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    assert np.any(tile == BLACK)
    assert np.any(tile != BLACK)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, min(max_colors, count, len(pool)))
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
    """Reproduce the solver's legend-cycle tiling logic for oracle validation."""
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


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid variant input; return None on failure."""
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
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k, [c for c in COLORS if c not in legend])

    output_colors = set(legend) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX and ceil(out_h / q) * ceil(out_w / q) >= legend_len + 1
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

    expected = _expected_output(legend, stencil, out_h, out_w)
    if not set(legend).issubset(set(map(int, np.unique(expected)))):
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
