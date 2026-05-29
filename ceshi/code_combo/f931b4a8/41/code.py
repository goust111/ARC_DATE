# Combo variant of f931b4a8 (index 41)
# combo_spec: D13/1+D14/0
# Name: 递归铺贴+颜色源缩小

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color

# Design analysis:
# - concrete_behavior:
#   The grid stays split into four equal quadrants, but the lower-left is no longer a full qxq
#   color table. Instead it contains a smaller solid non-black rectangle (fixed to 2x2 inside a
#   4x4 quadrant), and the lower-right remains a qxq template.
#   Example:
#     source_tile = [[1, 2],
#                    [3, 4]]
#     coarse_fill = [[1, 1, 2, 2],
#                    [1, 1, 2, 2],
#                    [3, 3, 4, 4],
#                    [3, 3, 4, 4]]
#     stencil mask = [[0, 5, 5, 0],
#                     [5, 5, 5, 5],
#                     [5, 5, 5, 5],
#                     [0, 5, 5, 0]]
#   Original/D14-only behavior would keep stencil foreground color 5 and only use coarse_fill in
#   stencil holes. The fused rule changes the foreground path too:
#   every non-black stencil cell becomes a recursive micro-tile lookup
#   source_tile[local_r % 2, local_c % 2], restarted inside each 4x4 template copy.
#   Every black stencil cell still uses the coarse block lookup coarse_fill[block_r, block_c].
#   The result visibly has two nested scales: large repeated template silhouettes and a small
#   repeated 2x2 color texture living inside each silhouette.
# - complexity_guardrails:
#   D14/0 is only obvious when the source is truly smaller than the template and all four source
#   cells matter, so generation fixes N=8, q=4, source size=2, and output dimensions to 13..16.
#   That guarantees four outer template blocks in each axis, so the coarse lookup can touch every
#   expanded source cell. The source tile always uses four distinct colors so the recursive fill is
#   human-visible instead of collapsing to a solid color.
# - hidden_assumptions_rechecked:
#   The original solver assumed (1) lower-left is fully non-black and directly indexable by
#   coarse block coordinates, and (2) lower-right foreground colors survive to output. Both fail
#   here. The new solver therefore:
#   1. extracts a unique non-black rectangle from the lower-left quadrant,
#   2. requires that rectangle to be smaller than the quadrant and axis-aligned,
#   3. expands it to qxq for hole-filling only,
#   4. treats lower-right non-black cells as a mask whose interiors are recursively tiled from the
#      small source tile rather than copied verbatim.
#   The generator explicitly guarantees these preconditions.
# - legality_visibility:
#   The 2x2 source rectangle is placed wholly inside the 4x4 lower-left quadrant with no overlap
#   and black padding around it. The stencil is always non-empty and always contains black holes,
#   so both the recursive fine texture and the coarse hole-fill are visible in every sample.
#   Stencil colors are chosen disjoint from source colors, which makes it impossible for the fused
#   behavior to accidentally look identical to the non-recursive baseline on generated examples.


def _extract_source_tile(fill_quadrant: np.ndarray, bg: int) -> np.ndarray:
    """Extract the unique solid non-background rectangle from the lower-left quadrant."""
    coords = np.argwhere(fill_quadrant != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    tile = fill_quadrant[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(tile != bg)

    mask = np.zeros(fill_quadrant.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert not np.any((fill_quadrant != bg) & ~mask)
    assert tile.shape[0] < fill_quadrant.shape[0] and tile.shape[1] < fill_quadrant.shape[1]
    return tile


def _expand_source_tile(source_tile: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Repeat each source cell over an equal block to match the template's coarse coordinates."""
    sh, sw = source_tile.shape
    th, tw = target_shape
    assert th % sh == 0 and tw % sw == 0
    scale_r = th // sh
    scale_c = tw // sw
    return np.repeat(np.repeat(source_tile, scale_r, axis=0), scale_c, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use upper-quadrant counts to size an output whose outer template is filled by a smaller recursive source tile.

    Input structure:
    - The input is an even square split into four equal quadrants.
    - The upper-left and upper-right quadrants are control regions; only their non-black counts matter.
    - The lower-left quadrant contains one smaller solid non-black rectangle inside black padding.
    - The lower-right quadrant is a non-empty template tile whose black cells mark holes.

    Rule:
    - Output height is the non-black count in the upper-left quadrant.
    - Output width is the non-black count in the upper-right quadrant.
    - First extract the small lower-left source rectangle and enlarge it to the template's qxq size by uniform repetition; this enlarged map is used only for coarse hole-filling.
    - Tile the lower-right template across the output. At each output cell, if the template cell is black, fill from the enlarged source using the outer block index. If the template cell is non-black, ignore that template color and instead fill from the original small source using local coordinates repeated inside the current template copy.

    Key insight:
    - The bottom half now encodes two linked resolutions of the same source: a coarse upscaled map for template holes and a fine recursive micro-tile for template foreground, so the output is a nested tiling rather than a single direct stencil copy.
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

    source_tile = _extract_source_tile(fill_quadrant, bg)
    coarse_fill = _expand_source_tile(source_tile, (qh, qw))
    sh, sw = source_tile.shape

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(source_tile[r % sh, c % sw])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            local_r = r % qh
            local_c = c % qw
            block_r = r // qh
            block_c = c // qw
            if int(stencil[local_r, local_c]) == bg:
                assert block_r < qh and block_c < qw
                value = int(coarse_fill[block_r, block_c])
            else:
                value = int(source_tile[local_r % sh, local_c % sw])
            assert value != bg
            out[r, c] = value
    return out


# --------------- generator (o5x) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIDE = 8
QUADRANT = GRID_SIDE // 2
SOURCE_SIZE = QUADRANT // 2

CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 13
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


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


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else \
        random.randint(2, min(max_colors, count, len(pool)))
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


def _make_small_source(palette: list[int]) -> np.ndarray:
    """Build a 2x2 source tile with four distinct visible colors."""
    assert len(palette) == SOURCE_SIZE * SOURCE_SIZE
    vals = palette[:]
    random.shuffle(vals)
    tile = np.array(vals, dtype=int).reshape(SOURCE_SIZE, SOURCE_SIZE)
    for _ in range(random.randrange(4)):
        tile = np.rot90(tile)
    if random.random() < 0.5:
        tile = np.fliplr(tile)
    if random.random() < 0.5:
        tile = np.flipud(tile)
    assert np.unique(tile).size == SOURCE_SIZE * SOURCE_SIZE
    return tile


def _make_fill_quadrant(q: int, source_tile: np.ndarray) -> np.ndarray:
    """Place the smaller source tile inside the lower-left quadrant with black padding."""
    size = source_tile.shape[0]
    quadrant = np.zeros((q, q), dtype=int)
    offsets = [(r, c) for r in range(q - size + 1) for c in range(q - size + 1) if (r, c) != (0, 0)]
    r0, c0 = random.choice(offsets)
    quadrant[r0:r0 + size, c0:c0 + size] = source_tile
    return quadrant


def _stencil_masks() -> list[np.ndarray]:
    """Template masks chosen to expose both coarse holes and fine recursive fill."""
    return [
        np.array([
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ], dtype=bool),
        np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ], dtype=bool),
        np.array([
            [0, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
        ], dtype=bool),
    ]


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the template tile; its colors are only a mask for recursive filling."""
    assert q == QUADRANT
    mask = random.choice(_stencil_masks())
    tile = np.zeros((q, q), dtype=int)
    if len(palette) == 1:
        tile[mask] = palette[0]
    else:
        for r, c in np.argwhere(mask):
            tile[r, c] = palette[(r + c + random.randrange(len(palette))) % len(palette)]
    assert np.any(tile == BLACK) and np.any(tile != BLACK)
    return tile


def _expected_output(source_tile: np.ndarray, stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the fused solver logic for oracle validation."""
    q = stencil.shape[0]
    coarse_fill = _expand_source_tile(source_tile, (q, q))
    sh, sw = source_tile.shape
    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            local_r = r % q
            local_c = c % q
            if int(stencil[local_r, local_c]) == BLACK:
                out[r, c] = coarse_fill[r // q, c // q]
            else:
                out[r, c] = source_tile[local_r % sh, local_c % sw]
    return out


def _baseline_output(source_tile: np.ndarray, stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    """Non-recursive baseline: keep stencil foreground colors, only use source for holes."""
    q = stencil.shape[0]
    coarse_fill = _expand_source_tile(source_tile, (q, q))
    rr, cc = np.mgrid[:out_h, :out_w]
    out = stencil[rr % q, cc % q].copy()
    bg_mask = out == BLACK
    out[bg_mask] = coarse_fill[rr[bg_mask] // q, cc[bg_mask] // q]
    return out


def _recursive_local_usage_count(stencil: np.ndarray, source_shape: tuple[int, int]) -> int:
    """How many source cells are sampled by template foreground within one outer template copy."""
    sh, sw = source_shape
    used = {
        (r % sh, c % sw)
        for r in range(stencil.shape[0])
        for c in range(stencil.shape[1])
        if int(stencil[r, c]) != BLACK
    }
    return len(used)


def _coarse_usage_count(stencil: np.ndarray, out_h: int, out_w: int,
                        source_shape: tuple[int, int]) -> int:
    """How many distinct source cells are reached by coarse hole-filling across the output."""
    q = stencil.shape[0]
    sh, sw = source_shape
    scale_r = q // sh
    scale_c = q // sw
    used: set[tuple[int, int]] = set()
    block_rows = (out_h + q - 1) // q
    block_cols = (out_w + q - 1) // q
    for br in range(block_rows):
        row_span = min(q, out_h - br * q)
        for bc in range(block_cols):
            col_span = min(q, out_w - bc * q)
            if np.any(stencil[:row_span, :col_span] == BLACK):
                used.add((br // scale_r, bc // scale_c))
    return len(used)


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate fused sample."""
    n = GRID_SIDE
    q = QUADRANT

    source_palette = _palette(SOURCE_SIZE * SOURCE_SIZE)
    stencil_palette = _palette(random.choice([1, 2]), [c for c in COLORS if c not in source_palette])
    used_input_output = set(source_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in used_input_output]
    if len(control_pool) < 2:
        return None

    out_h = random.randint(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX)
    out_w = random.randint(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX)
    if out_h * out_w > OUTPUT_AREA_MAX:
        return None

    source_tile = _make_small_source(source_palette)
    fill_quadrant = _make_fill_quadrant(q, source_tile)
    stencil = _make_stencil(q, stencil_palette)

    if _recursive_local_usage_count(stencil, source_tile.shape) < source_tile.size:
        return None
    if _coarse_usage_count(stencil, out_h, out_w, source_tile.shape) < source_tile.size:
        return None

    expected = _expected_output(source_tile, stencil, out_h, out_w)
    baseline = _baseline_output(source_tile, stencil, out_h, out_w)
    diff_count = int(np.count_nonzero(expected != baseline))
    if diff_count < (out_h * out_w) // 3:
        return None
    if np.unique(expected).size < source_tile.size:
        return None

    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_quadrant
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
