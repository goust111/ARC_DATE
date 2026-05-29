# Combo variant of f931b4a8 (index 18)
# combo_spec: D3/3+D12/0
# Name: 模板重着色+渐变颜色源

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - The 2x2 quadrant split is unchanged: UL controls output height, UR controls
#      output width, LL is the color source, LR is the repeated template.
#    - D3/3 changes LL from a coarse 2D lookup table into a directional gradient:
#      either each row is solid with a visible row sequence, or each column is solid
#      with a visible column sequence. Output colors therefore depend on global
#      output position modulo Q, not on coarse block indices r//Q,c//Q.
#    - D12/0 changes LR foreground from “keep LR ink” into “recolor the current LL
#      source color by a visible pairing rule”. The top rows encode that pairing:
#      UL row 0 lists the source colors, UR row 0 lists the paired replacement
#      colors in matching columns.
#    - Example with Q=4, LL row-sequence [2, 3, 4, 2], legend 2->3, 3->4, 4->2,
#      and a checker stencil:
#      old task on a foreground cell would output the stencil's own ink;
#      D3-only would output stencil ink on fg and [2,3,4,2] on bg rows;
#      fused rule outputs [3,4,2,3] on fg rows and [2,3,4,2] on bg rows.
#      The same spatial gradient is therefore visible twice: once directly in the
#      background and once shifted through the recolor map in the template foreground.
# b) Complexity calibration:
#    - LL is restricted to one unambiguous axis (rows xor cols) and 2-4 source
#      colors, so the progression is structured and easy to notice.
#    - LR is forced to be non-empty and mixed (both fg and bg) so the recolored and
#      unrecolored gradient both appear in every sample.
#    - The legend is short, contiguous, and explicit, so the color mapping is
#      inferable from visible evidence rather than hidden arithmetic on color ids.
# c) Hidden-assumption audit for main():
#    - The original solver assumed LL stored arbitrary 2D coarse colors and used
#      fill_tile[r//Q,c//Q] for background holes. That fails here; both background
#      and foreground must instead start from the directional sequence value at the
#      global row/column position.
#    - The original solver also assumed LR foreground color was semantically real.
#      That no longer holds: LR now contributes only a binary mask when non-empty.
#    - The fused solver adds two new assumptions: LL must be a valid row- or
#      column-gradient, and the top-row legend must cover exactly the set of LL
#      colors with a non-identity permutation. The generator enforces both.
# d) Placement legality:
#    - All objects stay inside their fixed quadrants of an even NxN grid.
#    - Legend cells occupy only the leftmost cells of the top row inside each upper
#      quadrant; remaining control pixels are placed in other cells, so there is no
#      overlap and the foreground counts remain exact.
#    - LL has no black cells. LR is always non-empty but still contains black holes,
#      and all output dimensions are positive and bounded by the original spec.
# e) Visual observability:
#    - A solver can see the LL orientation directly from solid rows vs solid columns.
#    - The output shows two synchronized bands: background uses the visible source
#      sequence, while template foreground uses the legend-shifted companion colors.
#    - Template ink is chosen outside the LL palette, so it is obvious that output
#      colors are not copied from LR.
# f) Visual cleanliness:
#    - Roles remain spatially separated: controls on top, source and stencil below.
#    - LL uses clean stripes only; LR uses simple masks that expose both fg and bg;
#      control filler colors are kept outside the output palette to reduce confusion.
# g) Change inferability:
#    - The only color-changing mechanism is the explicit top-row pairing, and the
#      only spatial mechanism is the visible row/column orientation of LL.
#    - Generator rejection ensures every source color appears under both fg and bg
#      stencil positions, so the gradient and its recolored counterpart have repeated
#      evidence in each training example.


def _extract_gradient(fill_tile: np.ndarray) -> tuple[str, np.ndarray]:
    """Return the LL gradient axis and its 1D color sequence."""
    qh, qw = fill_tile.shape
    assert qh == qw and qh >= 2
    assert np.all(fill_tile != BLACK)

    row_uniform = bool(np.all(fill_tile == fill_tile[:, :1]))
    col_uniform = bool(np.all(fill_tile == fill_tile[:1, :]))
    row_seq = fill_tile[:, 0].astype(int)
    col_seq = fill_tile[0, :].astype(int)

    if row_uniform and not col_uniform and len(set(map(int, row_seq))) >= 2:
        return "rows", row_seq
    if col_uniform and not row_uniform and len(set(map(int, col_seq))) >= 2:
        return "cols", col_seq
    raise AssertionError("lower-left quadrant must be a non-uniform row or column gradient")


def _infer_recolor_map(
    upper_left: np.ndarray,
    upper_right: np.ndarray,
    fill_tile: np.ndarray,
    bg: int,
) -> dict[int, int]:
    """Read the contiguous legend from row 0 of the upper quadrants."""
    q = upper_left.shape[1]
    fill_colors = {int(v) for v in np.unique(fill_tile)}
    mapping: dict[int, int] = {}

    for c in range(q):
        src = int(upper_left[0, c])
        dst = int(upper_right[0, c])
        if src == bg or dst == bg:
            break
        if src in fill_colors and dst in fill_colors and src not in mapping:
            mapping[src] = dst
            if len(mapping) == len(fill_colors):
                break
            continue
        break

    assert len(mapping) >= 2
    assert set(mapping) == fill_colors
    assert set(mapping.values()) == fill_colors
    assert all(src != bg and dst != bg for src, dst in mapping.items())
    assert all(src != dst for src, dst in mapping.items())
    return mapping


def _source_grid(axis: str, sequence: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Expand the LL gradient sequence across the requested output size."""
    rr, cc = np.mgrid[:out_h, :out_w]
    if axis == "rows":
        return sequence[rr % len(sequence)]
    return sequence[cc % len(sequence)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the upper control quadrants to size an output whose template and background both derive from a directional color sequence.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants still control output height and width by foreground count.
    - Their top rows also carry a legend: matching columns pair a lower-left source color with the replacement color used on template foreground.
    - The lower-left quadrant is a non-black row- or column-gradient source, and the lower-right quadrant is a stencil tile whose non-black cells mark template foreground.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - Read the lower-left quadrant as a 1D gradient sequence: either one color per row or one color per column.
    - If the lower-right stencil is empty, repeat the lower-left gradient directly over the output.
    - Otherwise, tile the stencil over the output. At each output cell, first read the current source color from the lower-left sequence using the global output row or column position. Stencil background keeps that source color, while stencil foreground is recolored to the paired color indicated by the upper-row legend.

    Key insight:
    - The lower-left quadrant no longer stores coarse block colors; it encodes a visible directional progression, and the lower-right quadrant contributes only a mask. The output is explained by combining those two cues with the top-row color-pair legend.
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

    axis, sequence = _extract_gradient(fill_tile)
    source = _source_grid(axis, sequence, out_h, out_w)

    if not np.any(stencil != bg):
        return source.astype(int, copy=True)

    recolor_map = _infer_recolor_map(upper_left, upper_right, fill_tile, bg)
    out = source.astype(int, copy=True)
    rr, cc = np.mgrid[:out_h, :out_w]
    fg_mask = stencil[rr % qh, cc % qw] != bg
    for src, dst in recolor_map.items():
        out[fg_mask & (source == int(src))] = int(dst)
    return out


# --------------- generator (o5x) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
GRADIENT_DISTINCT_RANGE = (2, 4)
STENCIL_EMPTY_PROB = 0.0
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(q) for c in range(q)]
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


def _make_gradient_source(q: int, palette: list[int]) -> tuple[np.ndarray, str, np.ndarray]:
    """Build LL as a row-gradient or column-gradient with repeated sequence values."""
    axis = random.choice(["rows", "cols"])
    k = len(palette)
    offset = random.randrange(k)
    sequence = np.array([palette[(offset + i) % k] for i in range(q)], dtype=int)
    tile = np.empty((q, q), dtype=int)

    if axis == "rows":
        for r, color in enumerate(sequence):
            tile[r, :] = int(color)
    else:
        for c, color in enumerate(sequence):
            tile[:, c] = int(color)
    return tile, axis, sequence


def _make_stencil(q: int, color: int, axis: str) -> np.ndarray:
    """Build a template mask whose structure exposes the gradient along the other axis."""
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "random", "random", "col_stripe" if axis == "rows" else "row_stripe"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = color
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t : q - t, t : q - t] = False
        tile[mask] = color
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = color
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = color
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = color

    assert np.any(tile == BLACK)
    assert np.any(tile != BLACK)
    return tile


def _make_control_with_legend(
    q: int,
    count: int,
    legend_colors: list[int],
    filler_pool: list[int],
) -> np.ndarray:
    """Build a control quadrant with exact count while keeping the legend contiguous."""
    k = len(legend_colors)
    assert 2 <= k <= q
    assert k <= count <= q * q
    tile = np.full((q, q), BLACK, dtype=int)
    tile[0, :k] = np.array(legend_colors, dtype=int)

    remaining = count - k
    if remaining == 0:
        return tile

    filler_k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else min(2, len(filler_pool), remaining)
    filler_palette = _palette(filler_k, filler_pool)
    mode = random.choice(["compact", "rows", "random"])
    coords = [rc for rc in _ordered_coords(q, mode) if not (rc[0] == 0 and rc[1] < k)]
    for r, c in coords[:remaining]:
        tile[r, c] = random.choice(filler_palette)
    return tile


def _legend_mapping(fill_palette: list[int]) -> dict[int, int]:
    """Create a visible non-identity permutation on the LL palette."""
    shift = random.randint(1, len(fill_palette) - 1)
    return {
        int(fill_palette[i]): int(fill_palette[(i + shift) % len(fill_palette)])
        for i in range(len(fill_palette))
    }


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    recolor_map: dict[int, int],
) -> np.ndarray:
    axis, sequence = _extract_gradient(fill_tile)
    source = _source_grid(axis, sequence, out_h, out_w)
    if not np.any(stencil != BLACK):
        return source.astype(int, copy=True)

    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    fg_mask = stencil[rr % q, cc % q] != BLACK
    out = source.astype(int, copy=True)
    for src, dst in recolor_map.items():
        out[fg_mask & (source == int(src))] = int(dst)
    return out


def _d3_only_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Oracle for the gradient-only variant: stencil fg keeps its own ink."""
    axis, sequence = _extract_gradient(fill_tile)
    source = _source_grid(axis, sequence, out_h, out_w)
    if not np.any(stencil != BLACK):
        return source.astype(int, copy=True)

    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    out = stencil[rr % q, cc % q].copy()
    bg_mask = out == BLACK
    out[bg_mask] = source[bg_mask]
    return out


def _source_colors_by_role(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> tuple[set[int], set[int]]:
    """Return source colors seen under stencil background vs stencil foreground."""
    axis, sequence = _extract_gradient(fill_tile)
    source = _source_grid(axis, sequence, out_h, out_w)
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    fg_mask = stencil[rr % q, cc % q] != BLACK
    bg_mask = ~fg_mask
    fg_colors = {int(v) for v in np.unique(source[fg_mask])}
    bg_colors = {int(v) for v in np.unique(source[bg_mask])}
    return fg_colors, bg_colors


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    grad_k_lo = min(max(2, GRADIENT_DISTINCT_RANGE[0]), q)
    grad_k_hi = min(GRADIENT_DISTINCT_RANGE[1], q)
    if grad_k_lo > grad_k_hi:
        return None

    ll_k = random.randint(grad_k_lo, grad_k_hi)
    fill_palette = _palette(ll_k)
    recolor_map = _legend_mapping(fill_palette)

    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        stencil_pool = [c for c in COLORS if c not in fill_palette]
        if not stencil_pool:
            return None
        stencil_palette = _palette(1, stencil_pool)

    control_pool = [c for c in COLORS if c not in set(fill_palette) | set(stencil_palette)]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [(h, w) for h in range(lo, hi + 1) for w in range(lo, hi + 1) if h * w <= OUTPUT_AREA_MAX]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    fill_tile, axis, _ = _make_gradient_source(q, fill_palette)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette[0], axis)

    if not stencil_empty:
        fg_source_colors, bg_source_colors = _source_colors_by_role(fill_tile, stencil, out_h, out_w)
        palette_set = set(fill_palette)
        if fg_source_colors != palette_set or bg_source_colors != palette_set:
            return None

    upper_left = _make_control_with_legend(q, out_h, fill_palette, control_pool)
    upper_right = _make_control_with_legend(
        q,
        out_w,
        [recolor_map[color] for color in fill_palette],
        control_pool,
    )

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w, recolor_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not stencil_empty and np.array_equal(actual, _d3_only_output(fill_tile, stencil, out_h, out_w)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
