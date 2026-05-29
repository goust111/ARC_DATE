# Combo variant of f931b4a8 (index 31)
# combo_spec: D3/2+D6/0+D14/0
# Name: 小色条+尺寸缩小+局部坐标映射

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - The input remains an even square split into four quadrants. To keep the D14 scaling
#      stable and visually visible, the generator fixes the full grid to 8x8, so each
#      quadrant is q=4.
#    - UL/UR still encode output height/width by foreground counts exactly as in the base task.
#    - LR is still a 4x4 stencil. Its non-black cells remain literal output pixels, while its
#      black cells are the places where the lower-left source matters.
#    - The fusion happens entirely in LL:
#      1. D3/2: LL is no longer a dense 4x4 color tile; it is mostly black and contains one
#         short 1x2 or 2x1 legend strip sitting in a corner.
#      2. D14/0: because that legend is smaller than the 4x4 stencil, the solver must first
#         extract its bbox and enlarge it by integer repetition to a 4x4 aligned color map.
#      3. D6/0: stencil holes are then filled from that enlarged map using local stencil
#         coordinates, i.e. fill = enlarged[r % 4, c % 4], not coarse block coordinates.
#    - Example:
#         LL legend strip = [2, 7] as a 1x2 horizontal strip in a corner
#         enlarged 4x4 map =
#           [[2, 2, 7, 7],
#            [2, 2, 7, 7],
#            [2, 2, 7, 7],
#            [2, 2, 7, 7]]
#         LR stencil =
#           [[8, 0, 0, 8],
#            [0, 8, 8, 0],
#            [8, 0, 0, 8],
#            [0, 8, 8, 0]]
#      In one repeated 4x4 output block, the black cells in the left half become 2 and the
#      black cells in the right half become 7. Under the original task those holes would be
#      filled uniformly per coarse repeated block; under the D3-only legend-cycle variant they
#      would also be uniform inside each repeated block. The combo instead creates within-block
#      color variation driven by local position after upscaling the tiny legend.
# b) Complexity calibration
#    - The legend length is fixed to 2 so the 4x4 stencil can align with a clear visible 2x
#      split. This avoids ambiguous 3-way scaling on q=4.
#    - The legend colors are always distinct and the stencil is always non-empty with holes that
#      touch both enlarged legend regions, so both legend colors visibly affect the output.
#    - Output height/width are restricted to 9..16, guaranteeing several repeated stencil blocks.
#      This makes the "same local pattern repeats in every block" cue visible across the output.
#    - The generator rejects samples that are too similar to either:
#      * D14-only coarse lookup on the enlarged map, or
#      * D3-only blockwise legend cycling.
#      This forces the fused behavior, not a degenerate special case.
# c) Hidden assumptions audited
#    - Base solver assumption: LL is fully non-black and directly indexable as a 4x4 tile.
#      Broken by D3/2 and D14/0. New solver must first isolate the unique non-black strip bbox.
#    - Base solver assumption: when LR is non-empty, a hole uses coarse repeated-block indices
#      (r//4, c//4). Broken by D6/0. New solver must use local indices (r%4, c%4).
#    - New assumption introduced here: the LL non-black region is a single solid 1D strip whose
#      bbox has height 1 or width 1, and whose dimensions divide the stencil size. The generator
#      guarantees this by construction.
#    - New assumption introduced here: the enlarged strip contains no black and covers the full
#      4x4 stencil period after integer repetition. The generator uses only 1x2 or 2x1 strips,
#      so 4 is divisible by each source dimension.
# d) Placement legality
#    - The LL legend strip is placed entirely inside its own 4x4 quadrant and anchored to one of
#      the four corners, so it neither overlaps other quadrants nor exceeds bounds.
#    - The LR stencil is a full 4x4 tile. Control quadrants are constructed with exact counts and
#      remain confined to their quadrants.
# e) Visual observability
#    - Humans can see three separate cues in the input:
#      * LL is tiny and black-surrounded, so it must be decoded rather than tiled directly.
#      * LL has only two colors in a short strip, so its order is salient.
#      * LR contains holes in different local positions, so the output reveals a repeated
#        left/right or top/bottom split inside every stencil period.
#    - Repetition across multiple 4x4 blocks supports the local-coordinate rule: identical local
#      hole positions across different repeated blocks receive the same color.
# f) Visual cleanliness
#    - LL contains only one simple strip and otherwise stays black.
#    - LR uses simple masks with both foreground and holes; control colors are disjoint from
#      output colors, so the four quadrants remain visually separable.
#    - The enlarged legend creates only two broad regions, avoiding noisy per-pixel color chaos.
# g) Color inference
#    - There are only two legend colors, both repeated many times in the output.
#    - Which legend color appears is tied to visible local position within the stencil period,
#      not to hidden numeric ordering or arbitrary color ranking.
# Main() assumption audit after the change
#    - `_extract_source_strip` assumes LL has one contiguous non-black rectangle and that the
#      rectangle is 1xn or nx1 with no black gaps.
#    - `_expand_source_tile` assumes integer scaling from the extracted strip to the full qxq
#      stencil period.
#    - `main()` assumes control counts are positive and that every stencil hole can read a color
#      from the enlarged map via local coordinates.
#    - `generate_input()` guarantees all of the above and validates every sample against `main()`.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIDE = 8
QUADRANT = GRID_SIDE // 2
LEGEND_LEN = 2

STENCIL_SINGLE_COLOR_PROB = 0.70
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.35, 0.70)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 9
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256

MIN_LOCAL_VS_COARSE_DELTA = 8
MIN_LOCAL_VS_CYCLE_DELTA = 8
GEN_MAX_RETRIES = 400
STENCIL_MAX_RETRIES = 60


def _extract_source_strip(fill_quadrant: np.ndarray, bg: int) -> np.ndarray:
    """Extract the unique non-black legend strip from LL."""
    coords = np.argwhere(fill_quadrant != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    strip = fill_quadrant[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(strip != bg)
    assert strip.shape[0] == 1 or strip.shape[1] == 1
    assert max(strip.shape) >= 2
    assert strip.shape[0] < fill_quadrant.shape[0] or strip.shape[1] < fill_quadrant.shape[1]

    mask = np.zeros(fill_quadrant.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert not np.any((fill_quadrant != bg) & ~mask)
    return strip


def _expand_source_tile(source_tile: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Expand a smaller source tile to the stencil period by integer repetition."""
    sh, sw = source_tile.shape
    th, tw = target_shape
    assert th % sh == 0 and tw % sw == 0
    scale_r = th // sh
    scale_c = tw // sw
    return np.repeat(np.repeat(source_tile, scale_r, axis=0), scale_c, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper quadrants as size controls, then repeat a stencil whose holes reveal an upscaled local legend pattern.

    Input structure:
    - The input is an even square split into four equal quadrants, with black as background.
    - The upper-left and upper-right quadrants still control the output height and width through their non-black counts.
    - The lower-right quadrant is a full stencil tile.
    - The lower-left quadrant is mostly black and contains one smaller non-black legend strip whose bbox is either 1xn or nx1.

    Rule:
    - Output height equals the number of non-black cells in the upper-left quadrant.
    - Output width equals the number of non-black cells in the upper-right quadrant.
    - First extract the small legend strip from the lower-left quadrant and enlarge it by integer repetition until it matches the stencil tile's size.
    - Tile the lower-right stencil over the output. Whenever a repeated stencil cell is black, replace it with the enlarged lower-left color at the same local coordinate inside the stencil period, using enlarged[r % q][c % q]. If the stencil is empty, tile the enlarged legend map directly.

    Key insight:
    - The top half still specifies only the output dimensions, but the bottom half now requires a three-step decode: find the tiny legend strip, upscale it to the stencil's resolution, and then use local within-stencil coordinates so every repeated block shows the same internal background color layout.
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
    enlarged = _expand_source_tile(source_strip, (qh, qw))
    assert np.all(enlarged != bg)

    out = np.empty((out_h, out_w), dtype=int)
    has_stencil = bool(np.any(stencil != bg))
    for r in range(out_h):
        local_r = r % qh
        for c in range(out_w):
            local_c = c % qw
            if has_stencil:
                value = int(stencil[local_r, local_c])
                if value == bg:
                    value = int(enlarged[local_r, local_c])
            else:
                value = int(enlarged[local_r, local_c])
            assert value != bg
            out[r, c] = value

    return out


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    items = list(pool)
    random.shuffle(items)
    return items[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from the palette."""
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


def _make_source_quadrant(q: int, source_tile: np.ndarray) -> np.ndarray:
    """Place the chosen small source strip in one corner of a black qxq quadrant."""
    quadrant = np.zeros((q, q), dtype=int)
    sh, sw = source_tile.shape
    if sw == 1:
        r0 = random.choice([0, q - sh])
        c0 = random.choice([0, q - 1])
        quadrant[r0 : r0 + sh, c0 : c0 + sw] = source_tile
    else:
        r0 = random.choice([0, q - 1])
        c0 = random.choice([0, q - sw])
        quadrant[r0 : r0 + sh, c0 : c0 + sw] = source_tile
    return quadrant


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a readable stencil with both foreground and holes."""
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


def _expected_output_local(enlarged: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Oracle for the combo rule: local-coordinate fill on the enlarged legend map."""
    q = stencil.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = enlarged[rr[bg_mask] % q, cc[bg_mask] % q]
    return tiled


def _expected_output_coarse(enlarged: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Ablation oracle: D14-only style coarse lookup on the enlarged map."""
    q = stencil.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = enlarged[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _expected_output_cycle(legend: list[int], stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Ablation oracle: D3-only style blockwise legend cycling."""
    q = stencil.shape[0]
    coarse_cols = int(ceil(out_w / q))
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        block_r = r // q
        for c in range(out_w):
            block_c = c // q
            fill_color = int(legend[(block_r * coarse_cols + block_c) % len(legend)])
            value = int(stencil[r % q, c % q])
            if value == BLACK:
                value = fill_color
            out[r, c] = value
    return out


def _used_source_cells(source_tile: np.ndarray, stencil: np.ndarray) -> set[tuple[int, int]]:
    """Return which small-source cells are exposed by stencil holes after scaling."""
    q = stencil.shape[0]
    sh, sw = source_tile.shape
    scale_r = q // sh
    scale_c = q // sw
    used: set[tuple[int, int]] = set()
    for r, c in np.argwhere(stencil == BLACK):
        used.add((int(r) // scale_r, int(c) // scale_c))
    return used


def _variant_metrics(source_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> tuple[int, int]:
    """Measure distance from key ablations that would miss the fusion logic."""
    enlarged = _expand_source_tile(source_tile, stencil.shape)
    local = _expected_output_local(enlarged, stencil, out_h, out_w)
    coarse = _expected_output_coarse(enlarged, stencil, out_h, out_w)
    cycle = _expected_output_cycle(
        [int(v) for v in (source_tile[0, :] if source_tile.shape[0] == 1 else source_tile[:, 0])],
        stencil,
        out_h,
        out_w,
    )
    delta_coarse = int(np.count_nonzero(local != coarse))
    delta_cycle = int(np.count_nonzero(local != cycle))
    return delta_coarse, delta_cycle


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid fused-variant input."""
    n = GRID_SIDE
    q = QUADRANT

    legend = _palette(LEGEND_LEN)
    source_tile = np.array([legend], dtype=int) if random.random() < 0.5 else np.array([[legend[0]], [legend[1]]], dtype=int)

    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
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
    out_h, out_w = random.choice(valid_dims)

    for _ in range(STENCIL_MAX_RETRIES):
        stencil = _make_stencil(q, stencil_palette)
        if len(_used_source_cells(source_tile, stencil)) < source_tile.size:
            continue

        delta_coarse, delta_cycle = _variant_metrics(source_tile, stencil, out_h, out_w)
        if delta_coarse < MIN_LOCAL_VS_COARSE_DELTA:
            continue
        if delta_cycle < MIN_LOCAL_VS_CYCLE_DELTA:
            continue

        lower_left = _make_source_quadrant(q, source_tile)
        strip = _extract_source_strip(lower_left, BLACK)
        if strip.shape != source_tile.shape:
            continue
        if [int(v) for v in (strip[0, :] if strip.shape[0] == 1 else strip[:, 0])] != legend:
            continue

        upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
        upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

        grid = np.zeros((n, n), dtype=int)
        grid[:q, :q] = upper_left
        grid[:q, q:] = upper_right
        grid[q:, :q] = lower_left
        grid[q:, q:] = stencil

        expected = _expected_output_local(_expand_source_tile(strip, stencil.shape), stencil, out_h, out_w)
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if not set(legend).issubset(set(map(int, np.unique(expected)))):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo input")
