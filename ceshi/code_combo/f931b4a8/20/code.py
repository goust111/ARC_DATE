# Combo variant of f931b4a8 (index 20)
# combo_spec: D3/2+D14/0
# Name: 颜色源缩小+色条图例

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Keep the original four-quadrant layout and the original top-half controls.
#    - Change the lower-left source in two coupled ways at once:
#      1. it is no longer a full qxq fill table;
#      2. it is a much smaller 1x2 or 2x1 non-black legend strip embedded in black.
#    - The legend strip is first extracted as a tiny rectangle, then enlarged to qxq
#      by nearest-neighbor repetition so it aligns with the lower-right stencil.
#    - Example with q=4, horizontal source strip [2, 7]:
#        source bbox = [[2, 7]]
#        enlarged source =
#          [[2, 2, 7, 7],
#           [2, 2, 7, 7],
#           [2, 2, 7, 7],
#           [2, 2, 7, 7]]
#      If an output pixel lies in coarse block (block_r=1, block_c=3), then:
#        original task would read fill_tile[1, 3] from a dense 4x4 lower-left tile;
#        D3-only would cycle legend colors by block index;
#        this combo instead reads enlarged_source[1, 3] = 7.
#      The stencil foreground still survives unchanged; only black stencil cells are replaced.
# b) Complexity calibration
#    - The generator fixes the input size at 8x8, so q=4 and the scaling ratio is always
#      visible and integral: 1x2 -> 4x4 or 2x1 -> 4x4.
#    - Legend length is fixed at 2. This avoids ambiguous non-integer 3->4 scaling while
#      still making the strip visibly "small" and the enlargement visibly meaningful.
#    - Output dimensions are restricted to 9..16 so the coarse block grid spans at least
#      3x3 blocks. That guarantees both legend colors can affect the output under the
#      coarse lookup rule instead of collapsing to a single color.
# c) Hidden assumptions replaced
#    - Original main() assumed the whole lower-left quadrant was non-black and directly
#      indexable by coarse block coordinates. That is false here because most of that
#      quadrant is black.
#    - D3-only logic assumed the lower-left strip should be read as an ordered sequence
#      and cycled by row-major block index. That is also false here: the strip is a tiny
#      spatial source that must be enlarged before coarse indexing.
#    - Therefore the new solver must:
#      1. isolate the unique non-black strip by bbox,
#      2. verify it is a solid 1D rectangle,
#      3. enlarge it to qxq,
#      4. use enlarged coarse coordinates for stencil-hole replacement.
# d) Placement legality
#    - The lower-left strip is placed wholly inside its quadrant, always flush to a corner,
#      and never overlaps anything else because the quadrant is otherwise black.
#    - The lower-right stencil stays entirely inside its quadrant and either remains empty
#      or contains both black holes and non-black foreground.
#    - Control quadrants still use exactly the counted number of non-black cells.
# e) Visual observability
#    - Humans can see that the lower-left object is a tiny legend strip rather than a full tile.
#    - They can also see that its geometry is smaller than the 4x4 stencil quadrant, forcing
#      an enlargement step before the source can align with stencil block coordinates.
#    - The enlarged source produces wide vertical or horizontal bands in the output background,
#      which is visually distinct from both the dense original lookup and the D3-only cycle.
# f) Visual cleanliness
#    - Only one source object exists in the lower-left quadrant, with black padding around it.
#    - The output remains structured: a repeated stencil layer over a coarse striped fill layer.
#      There is no multi-object overlap inside the source quadrant itself.
# g) Color inference
#    - The legend uses only two colors, keeping the mapping inferable from 2-4 examples.
#    - Stencil colors are sampled from colors outside the legend, so the source colors are
#      easy to track as the ones specifically filling stencil holes or empty-stencil regions.
# Main() assumptions audited after the fusion
#    - `_extract_source_strip` assumes the lower-left quadrant contains exactly one solid
#      non-black 1x2 or 2x1 rectangle and nothing else non-black.
#    - `_expand_source_strip` assumes the target side length is divisible by the strip bbox
#      shape; the generator guarantees q=4 and strip shapes (1,2) or (2,1).
#    - `main()` assumes output control counts are positive and fit within the qxq coarse
#      lookup range used by the original task; the generator keeps output sizes in 9..16.
#    - The generator rejects any sample where the enlarged-strip rule would accidentally
#      collapse to the D3-only legend-cycle output, so the intended combo behavior is active.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIDE = 8
QUADRANT = GRID_SIDE // 2
LEGEND_LEN = 2

STENCIL_EMPTY_PROB = 0.12
LEGEND_VERTICAL_PROB = 0.50
STENCIL_SINGLE_COLOR_PROB = 0.72
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 9
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


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
    """Return all (r, c) in a qxq grid in a layout-dependent order."""
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
        anchor_r, anchor_c = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - anchor_r) + abs(rc[1] - anchor_c), random.random()),
        )
    random.shuffle(coords)
    return coords


def _extract_source_strip(fill_quadrant: np.ndarray, bg: int) -> np.ndarray:
    """Find the unique smaller legend strip embedded in the lower-left quadrant."""
    coords = np.argwhere(fill_quadrant != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    strip = fill_quadrant[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(strip != bg)

    mask = np.zeros(fill_quadrant.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert not np.any((fill_quadrant != bg) & ~mask)
    assert strip.shape in {(1, LEGEND_LEN), (LEGEND_LEN, 1)}
    assert strip.shape[0] < fill_quadrant.shape[0]
    assert strip.shape[1] < fill_quadrant.shape[1]
    return strip


def _expand_source_strip(source_strip: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor expand the small source strip to the stencil's qxq coarse grid."""
    sh, sw = source_strip.shape
    th, tw = target_shape
    assert th % sh == 0 and tw % sw == 0
    scale_r = th // sh
    scale_c = tw // sw
    return np.repeat(np.repeat(source_strip, scale_r, axis=0), scale_c, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Size the output from the top controls, then fill a tiled stencil using an enlarged lower-left legend strip.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants still encode the output height and width by non-black counts.
    - The lower-left quadrant is mostly black and contains one smaller non-black legend strip; the lower-right quadrant is a full-size stencil tile or is empty.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - First extract the small lower-left strip and enlarge it by equal repetition until it matches the stencil quadrant's size.
    - Tile the lower-right stencil across the output. Stencil foreground pixels stay as they are; stencil background pixels are replaced by the enlarged source value at the corresponding coarse repeated-block coordinate.
    - If the stencil is empty, tile the enlarged source itself over the whole output.

    Key insight:
    - The source colors are no longer stored at the stencil's resolution and they are no longer used as a simple cycle. The solver must recognize a tiny legend-like strip, upscale it to the stencil's coarse frame, and only then use that aligned map to fill stencil holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_quadrant = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    source_strip = _extract_source_strip(fill_quadrant, bg)
    coarse_fill = _expand_source_strip(source_strip, (qh, qw))

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(coarse_fill[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == bg:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(coarse_fill[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


def _make_source_legend(q: int, legend: list[int], vertical: bool) -> np.ndarray:
    """Build the lower-left quadrant as one small legend strip placed in a corner."""
    quadrant = np.zeros((q, q), dtype=int)
    if vertical:
        top = random.choice([0, q - len(legend)])
        left = random.choice([0, q - 1])
        quadrant[top : top + len(legend), left] = legend
    else:
        top = random.choice([0, q - 1])
        left = random.choice([0, q - len(legend)])
        quadrant[top, left : left + len(legend)] = legend
    return quadrant


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black cells are filled from the enlarged source)."""
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
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
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


def _expected_output(
    source_strip: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int
) -> np.ndarray:
    """Reproduce the combo solver's enlarged-strip tiling logic for oracle validation."""
    q = int(stencil.shape[0])
    coarse_fill = _expand_source_strip(source_strip, (q, q))
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return coarse_fill[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = coarse_fill[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _legend_cycle_output(legend: list[int], stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reference output for the D3-only legend-cycle rule, used to reject degeneration."""
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
    """Try once to produce a valid combo input; return None on failure."""
    n = GRID_SIDE
    q = QUADRANT
    stencil_empty = random.random() < STENCIL_EMPTY_PROB
    vertical = random.random() < LEGEND_VERTICAL_PROB
    legend = _palette(LEGEND_LEN)

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

    valid_dims = [
        (out_h, out_w)
        for out_h in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        for out_w in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    src = _make_source_legend(q, legend, vertical)
    source_strip = _extract_source_strip(src, BLACK)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = src
    grid[q:, q:] = stencil

    expected = _expected_output(source_strip, stencil, out_h, out_w)
    if not set(legend).issubset(set(map(int, np.unique(expected)))):
        return None
    if np.array_equal(expected, _legend_cycle_output(legend, stencil, out_h, out_w)):
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
