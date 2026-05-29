# Variant of f931b4a8
# Dimension: D14 — 颜色源与模板的尺寸关系 (meta)
# Change: 颜色源和模板的尺寸相同（均为半网格大小），两者一一对应地参与铺贴... → 颜色源尺寸大于模板（需降采样）

from __future__ import annotations

from math import ceil
import random

import numpy as np


# Design analysis for the D14 variant
# -----------------------------------
# a) Concretization and behavior trace
#    - The full input is still an even square split into four quadrants.
#    - The lower-left quadrant remains the color source, but it is now higher
#      resolution than the effective stencil. We realize this by making the
#      lower-left quadrant a block-constant 8x8 area whose 2x2 monochrome blocks
#      correspond to a hidden 4x4 coarse grid.
#    - The lower-right quadrant no longer uses its whole area as the stencil.
#      Instead, it contains a smaller active stencil patch embedded in black
#      padding. The active patch is recovered by taking the bbox of non-black
#      stencil pixels. Inside that bbox, black cells are still meaningful holes.
#    - Example trace with q=8 and active stencil 4x4:
#      old rule: stencil period = 8, fill lookup index = (r//8, c//8)
#      new rule: extract 4x4 stencil, downsample 8x8 source into 4x4 coarse
#      colors, then use stencil period = 4 and fill lookup index = (r//4, c//4).
#      The difference is visible because the source has twice the resolution of
#      the effective stencil and must be reduced before hole filling.
# b) Complexity calibration
#    - Template size is fixed to half the quadrant side, which keeps the scaling
#      rule easy to infer from a few examples while making the size mismatch
#      obvious.
#    - The source is never fully solid: it uses 2-4 colors arranged in regular
#      coarse patterns so the downsampling target is visually legible.
#    - The stencil always has both foreground and holes, so the variant behavior
#      is exercised instead of degenerating to direct tiling.
# c) Old implicit assumptions that break
#    - Using the entire lower-right quadrant as the stencil is no longer valid;
#      the solver must crop the active bbox first.
#    - Using q as both stencil period and fill-grid size is no longer valid.
#      After the change, the period is the active stencil size, while the source
#      must be block-downsampled to that smaller size.
#    - Directly tiling the lower-left quadrant when the stencil is empty would
#      hide the variant and makes the smaller stencil size unrecoverable. The
#      generator therefore always produces a non-empty active stencil.
# d) Placement legality
#    - The active stencil patch is embedded fully inside the lower-right quadrant.
#    - Border coverage is enforced so its bbox equals the intended smaller size.
#    - The source is created by exact integer block expansion, so downsampling is
#      well-defined and lossless.
# e) Visual observability
#    - The right-bottom quadrant visibly contains a smaller motif inside a larger
#      black field.
#    - The left-bottom quadrant visibly has larger 2x2 color blocks, signaling
#      that it should be read at a coarser granularity than its raw pixel size.
#    - Output sizes are chosen so at least two coarse fill positions are used.
# f) Visual tidiness
#    - The source uses structured rows/cols/checker/diagonal coarse layouts
#      instead of noisy pixel scatter.
#    - The stencil hole pattern is constrained to a 4x4 patch with clear borders.
# g) Color inferability
#    - Colors are copied, never remapped by id.
#    - Distinct source colors reappear in hole-filled regions with repeated block
#      structure, giving a human solver positional evidence for the downsampling.
# h) New solver assumptions and generator guarantees
#    - The active stencil bbox must be non-empty: guaranteed by generation.
#    - The source dimensions must be divisible by the active stencil dimensions:
#      guaranteed by choosing a fixed 2x scale.
#    - Each source block must be monochrome so one representative samples it:
#      guaranteed by expanding a coarse grid with np.kron.
#    - Output height/width must stay within active_size^2 so fill indices fit the
#      downsampled source: guaranteed when choosing control counts.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)
SOURCE_SCALE = 2
SOURCE_COLOR_RANGE = (2, 4)
STENCIL_COLOR_RANGE = (1, 3)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 5
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper controls to size an output whose holes are filled from a downsampled source.

    Input structure:
    - The grid is an even square divided into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants still control output height and width.
    - The lower-left quadrant is a larger color source made of repeated monochrome blocks, while the lower-right quadrant contains a smaller active stencil patch inside black padding.

    Rule:
    - Output height is the number of non-background pixels in the upper-left quadrant.
    - Output width is the number of non-background pixels in the upper-right quadrant.
    - Recover the active stencil by cropping the bounding box of non-black pixels in the lower-right quadrant.
    - Partition the lower-left quadrant into equal blocks so it downsamples to the stencil's height and width; tile the stencil over the output, and whenever a stencil cell is black, fill it with the downsampled source color chosen by the repeated-tile index.

    Key insight:
    - The lower quadrants no longer operate at the same resolution: the source must first be reduced to the smaller stencil grain, and only then can it provide the coarse colors that fill stencil holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    source = g[qh:, :qw]
    raw_stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    assert out_h > 0 and out_w > 0

    stencil = _extract_active_stencil(raw_stencil)
    sh, sw = stencil.shape
    coarse_source = _downsample_source(source, sh, sw)

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % sh, c % sw])
            if value == BLACK:
                fill_r = r // sh
                fill_c = c // sw
                assert fill_r < sh and fill_c < sw
                value = int(coarse_source[fill_r, fill_c])
                assert value != BLACK
            out[r, c] = value

    return out


def _extract_active_stencil(raw_stencil: np.ndarray) -> np.ndarray:
    coords = np.argwhere(raw_stencil != BLACK)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    stencil = raw_stencil[int(r0):int(r1) + 1, int(c0):int(c1) + 1]
    assert stencil.shape[0] < raw_stencil.shape[0] or stencil.shape[1] < raw_stencil.shape[1]
    return stencil


def _downsample_source(source: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = source.shape
    assert src_h % out_h == 0 and src_w % out_w == 0
    block_h = src_h // out_h
    block_w = src_w // out_w
    assert block_h > 1 or block_w > 1

    coarse = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            block = source[r * block_h:(r + 1) * block_h, c * block_w:(c + 1) * block_w]
            value = int(block[0, 0])
            assert value != BLACK
            assert np.all(block == value)
            coarse[r, c] = value
    return coarse


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


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
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_coarse_source(q: int, palette: list[int],
                        required_coords: list[tuple[int, int]] | None = None) -> np.ndarray:
    tile = np.empty((q, q), dtype=int)
    mode = random.choice(["rows", "cols", "checker", "diagonal", "blocks"])
    k = len(palette)

    if mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = palette[(c + offset) % k]
    elif mode == "checker":
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + random.randrange(k)) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color
    elif mode == "diagonal":
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + 2 * cc + random.randrange(k)) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color
    else:
        tile = _random_color_array((q, q), palette)

    existing = set(map(int, np.unique(tile)))
    missing = [c for c in palette if c not in existing]
    if missing:
        fix_coords = list(required_coords) if required_coords is not None else \
            [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color

    if required_coords:
        used_colors = {int(tile[r, c]) for r, c in required_coords}
        if len(used_colors) == 1:
            r, c = random.choice(required_coords)
            alternatives = [color for color in palette if color != int(tile[r, c])]
            if alternatives:
                tile[r, c] = random.choice(alternatives)
    return tile


def _expand_source(coarse_source: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(coarse_source, np.ones((scale, scale), dtype=int))


def _make_template(q: int, palette: list[int]) -> np.ndarray:
    tile = np.zeros((q, q), dtype=int)
    mode = random.choice(["checker", "row_stripe", "col_stripe", "frame", "random"])

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "frame":
        tile[:, :] = random.choice(palette)
        inner_color = random.choice(palette)
        tile[1:q - 1, 1:q - 1] = BLACK
        tile[random.randrange(1, q - 1), random.randrange(1, q - 1)] = inner_color
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
        fg_count = random.randint(max(5, q + 1), q * q - 3)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)

    # Force border support so bbox matches the intended active template size.
    if not np.any(tile[0, :] != BLACK):
        tile[0, random.randrange(q)] = random.choice(palette)
    if not np.any(tile[q - 1, :] != BLACK):
        tile[q - 1, random.randrange(q)] = random.choice(palette)
    if not np.any(tile[:, 0] != BLACK):
        tile[random.randrange(q), 0] = random.choice(palette)
    if not np.any(tile[:, q - 1] != BLACK):
        tile[random.randrange(q), q - 1] = random.choice(palette)

    # Keep several visible holes inside the active stencil.
    interior = [(r, c) for r in range(1, q - 1) for c in range(1, q - 1)]
    random.shuffle(interior)
    current_holes = int(np.count_nonzero(tile == BLACK))
    needed_holes = max(3, q - 1) - current_holes
    for r, c in interior[:max(0, needed_holes)]:
        tile[r, c] = BLACK

    assert np.any(tile != BLACK)
    assert np.any(tile == BLACK)
    return tile


def _embed_template(quadrant_size: int, template: np.ndarray) -> np.ndarray:
    qh, qw = template.shape
    raw = np.zeros((quadrant_size, quadrant_size), dtype=int)
    r0 = random.randint(0, quadrant_size - qh)
    c0 = random.randint(0, quadrant_size - qw)
    raw[r0:r0 + qh, c0:c0 + qw] = template
    return raw


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
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


def _expected_output(source: np.ndarray, raw_stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    stencil = _extract_active_stencil(raw_stencil)
    sh, sw = stencil.shape
    coarse_source = _downsample_source(source, sh, sw)

    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % sh, cc % sw].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = coarse_source[rr[bg_mask] // sh, cc[bg_mask] // sw]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    template_q = q // SOURCE_SCALE

    source_k = random.randint(*SOURCE_COLOR_RANGE)
    source_palette = _palette(source_k)
    stencil_k = random.randint(*STENCIL_COLOR_RANGE)
    stencil_palette = _palette(stencil_k)

    output_colors = set(source_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if len(control_pool) < 2:
        return None

    dim_lo = max(OUTPUT_DIM_MIN, template_q)
    dim_hi = min(OUTPUT_DIM_MAX, template_q * template_q)
    valid_dims = [
        (h, w) for h in range(dim_lo, dim_hi + 1) for w in range(dim_lo, dim_hi + 1)
        if h * w <= OUTPUT_AREA_MAX
        and ceil(h / template_q) * ceil(w / template_q) >= max(2, source_k)
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / template_q)) for c in range(ceil(out_w / template_q))]
    coarse_source = _make_coarse_source(template_q, source_palette, used_fill)
    source = _expand_source(coarse_source, SOURCE_SCALE)
    template = _make_template(template_q, stencil_palette)
    raw_stencil = _embed_template(q, template)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = source
    grid[q:, q:] = raw_stencil

    expected = _expected_output(source, raw_stencil, out_h, out_w)
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
