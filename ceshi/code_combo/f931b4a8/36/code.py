# Combo variant of f931b4a8 (index 36)
# combo_spec: D4/1+D14/1
# Name: 颜色源大于模板+多色模板

from __future__ import annotations

from math import ceil
import random

import numpy as np


# Design analysis for the fused D4/1 + D14/1 variant
# ---------------------------------------------------
# a) Concretization and behavior trace
#    - The global layout stays as four quadrants inside an even square.
#    - The upper quadrants still encode output height and width via foreground
#      pixel counts.
#    - The lower-right quadrant no longer uses its whole area as the template.
#      Instead it contains a smaller active template patch inside black padding;
#      the solver must crop the non-black bounding box to recover that patch.
#    - The lower-left quadrant is larger than the active template. It is built as
#      a block-expanded image: for q=8 and active template size 4, each 2x2 block
#      in the source corresponds to one coarse color cell.
#    - Example trace with q=8, template 4x4, output 9x11:
#      old task: use the full 8x8 lower-right tile as period; when the stencil is
#      black, fill from lower-left[r//8, c//8].
#      fused task: crop a 4x4 active template, downsample the 8x8 source to a 4x4
#      coarse grid, tile that 4x4 template, and fill black template cells from
#      coarse_source[r//4, c//4].
#      The added D4 change matters because the visible template foreground now has
#      at least two distinct non-black colors, so the solver must preserve those
#      colors while only using the downsampled source for black holes.
# b) Complexity calibration
#    - The active template size is fixed to half of the source quadrant side,
#      making the scale mismatch obvious but still easy to infer.
#    - The template is forced to have both holes and at least two foreground
#      colors, so examples visibly mix template-owned colors with source-owned
#      colors instead of collapsing to a one-color stencil.
#    - The source is generated from 2-4 colors on the coarse grid, then expanded
#      into exact monochrome blocks, so the intended downsampling target is legible.
# c) Old implicit assumptions that break
#    - `stencil[r % q, c % q]` is no longer valid because the usable template is
#      smaller than the quadrant and must be cropped from padding first.
#    - `fill_tile[r // q, c // q]` is no longer valid because the fill lookup
#      operates at template grain after source downsampling, not at quadrant grain.
#    - Treating every non-black template pixel as interchangeable is no longer
#      valid: multiple foreground colors belong to the template itself and must
#      survive unchanged in the output.
# d) Placement legality
#    - The active template is embedded fully inside the lower-right quadrant with
#      black padding around it, so the crop is well-defined and in-bounds.
#    - Border support is enforced on the active template, guaranteeing that its
#      recovered bounding box matches the intended template size.
#    - The source is created by exact integer expansion with scale 2, so every
#      downsample block is monochrome and the solver's assertions are justified.
# e) Visual observability
#    - The lower-right quadrant visibly contains a smaller motif floating inside a
#      larger black field, signaling that the effective period is smaller.
#    - The lower-left quadrant visibly consists of larger repeated color blocks,
#      signaling that it must be read more coarsely than pixel-by-pixel.
#    - The output always shows at least two template foreground colors and also
#      uses multiple source coarse cells, so both halves of the fused rule are
#      needed to explain the samples.
# f) Visual tidiness
#    - Both the source and template use structured motifs such as rows, checker,
#      diagonals, or frames instead of unconstrained noise.
#    - Template and source palettes are sampled disjointly, which keeps "template
#      structure colors" visually separate from "hole fill colors".
# g) Change inferability
#    - Colors are copied directly; there is no hidden color remapping.
#    - Because the template foreground colors reappear exactly while black holes
#      reveal a coarser repeated source pattern, a human can infer which colors
#      come from which quadrant across 2-4 examples.
# h) New solver assumptions and generator guarantees
#    - Assumption: the active template crop is non-empty and smaller than the
#      source quadrant. Guarantee: generation always embeds a non-empty 4x4 patch
#      inside an 8x8 lower-right quadrant.
#    - Assumption: source dimensions are divisible by template dimensions.
#      Guarantee: source size is 8x8 and template size is fixed to 4x4.
#    - Assumption: each source downsampling block is monochrome and non-black.
#      Guarantee: generation expands a coarse non-black grid with `np.kron`.
#    - Assumption: output dimensions do not index beyond the coarse source when
#      using `(r // template_h, c // template_w)`. Guarantee: output sizes are
#      capped at 16, which matches the 4x4 coarse grid coverage.
#    - Assumption: the fused behavior is not degenerate to the original task.
#      Guarantee: generation rejects any sample whose fused output matches the
#      output produced by the original uncropped / undownsampled rule.


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (16,)
SOURCE_SCALE = 2
SOURCE_COLOR_RANGE = (2, 4)
TEMPLATE_COLOR_RANGE = (2, 3)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 5
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the upper control regions to size an output built from a multicolor template and a downsampled source.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants control output height and width by foreground count.
    - The lower-left quadrant is a larger color source made of repeated monochrome blocks, while the lower-right quadrant contains a smaller active template patch inside black padding.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - Recover the active template by cropping the bounding box of non-black pixels in the lower-right quadrant.
    - Downsample the lower-left quadrant into the template's shape by reading equal monochrome blocks.
    - Tile the active template over the output; keep every non-black template color as-is, and whenever the template cell is black, fill it with the downsampled source color selected by the repeated coarse-tile index.

    Key insight:
    - The lower quadrants operate at two different resolutions, and the template itself now carries multiple foreground colors, so solving requires both cropping/scaling and distinguishing template-owned colors from source-filled holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    source = g[qh:, :qw]
    raw_template = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    assert out_h > 0 and out_w > 0

    template = _extract_active_template(raw_template)
    th, tw = template.shape
    assert source.shape[0] > th or source.shape[1] > tw
    coarse_source = _downsample_source(source, th, tw)

    out = np.empty((out_h, out_w), dtype=int)
    for r in range(out_h):
        for c in range(out_w):
            value = int(template[r % th, c % tw])
            if value == BLACK:
                fill_r = r // th
                fill_c = c // tw
                assert fill_r < th and fill_c < tw
                value = int(coarse_source[fill_r, fill_c])
                assert value != BLACK
            out[r, c] = value

    return out


def _extract_active_template(raw_template: np.ndarray) -> np.ndarray:
    coords = np.argwhere(raw_template != BLACK)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    template = raw_template[int(r0):int(r1) + 1, int(c0):int(c1) + 1]
    assert template.shape[0] < raw_template.shape[0] or template.shape[1] < raw_template.shape[1]
    assert np.any(template == BLACK)
    assert len(set(map(int, np.unique(template[template != BLACK])))) >= 2
    return template


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
    colors = list(pool)
    random.shuffle(colors)
    return colors[:k]


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
    k = len(palette)
    mode = random.choice(["rows", "cols", "checker", "diagonal", "blocks"])

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
    missing = [color for color in palette if color not in existing]
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

    assert np.all(tile != BLACK)
    return tile


def _expand_source(coarse_source: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(coarse_source, np.ones((scale, scale), dtype=int))


def _make_multicolor_template(q: int, palette: list[int]) -> np.ndarray:
    assert len(palette) >= 2
    tile = np.zeros((q, q), dtype=int)
    k = len(palette)
    mode = random.choice(["checker", "row_bands", "col_bands", "frame", "diagonals"])

    if mode == "checker":
        rr, cc = np.mgrid[:q, :q]
        fg_mask = (rr + cc + random.randrange(2)) % 2 == 0
        idx = (2 * rr + cc + random.randrange(k)) % k
        for i, color in enumerate(palette):
            tile[fg_mask & (idx == i)] = color
    elif mode == "row_bands":
        parity = random.randrange(2)
        offset = random.randrange(k)
        for r in range(q):
            if r % 2 == parity:
                for c in range(q):
                    tile[r, c] = palette[(r + c + offset) % k]
    elif mode == "col_bands":
        parity = random.randrange(2)
        offset = random.randrange(k)
        for c in range(q):
            if c % 2 == parity:
                for r in range(q):
                    tile[r, c] = palette[(r + c + offset) % k]
    elif mode == "frame":
        offset = random.randrange(k)
        for c in range(q):
            tile[0, c] = palette[(c + offset) % k]
            tile[q - 1, c] = palette[(c + offset + 1) % k]
        for r in range(q):
            tile[r, 0] = palette[(r + offset + 1) % k]
            tile[r, q - 1] = palette[(r + offset) % k]
        tile[1:q - 1, 1:q - 1] = BLACK
        tile[random.randrange(1, q - 1), random.randrange(1, q - 1)] = palette[(offset + 2) % k]
    else:
        offset = random.randrange(k)
        for r in range(q):
            tile[r, r] = palette[(r + offset) % k]
            tile[r, q - 1 - r] = palette[(r + offset + 1) % k]

    _force_template_bbox(tile, palette)

    fg_coords = [(r, c) for r in range(q) for c in range(q) if tile[r, c] != BLACK]
    random.shuffle(fg_coords)
    existing = set(map(int, np.unique(tile[tile != BLACK])))
    missing = [color for color in palette if color not in existing]
    for color, (r, c) in zip(missing, fg_coords):
        tile[r, c] = color

    visible_colors = set(map(int, np.unique(tile[tile != BLACK])))
    if len(visible_colors) < 2:
        coords = fg_coords[:2]
        if len(coords) < 2:
            coords = [(0, 0), (q - 1, q - 1)]
        tile[coords[0]] = palette[0]
        tile[coords[1]] = palette[1]

    if not np.any(tile == BLACK):
        interior = [(r, c) for r in range(1, q - 1) for c in range(1, q - 1)]
        random.shuffle(interior)
        hole_r, hole_c = interior[0]
        tile[hole_r, hole_c] = BLACK

    assert np.any(tile != BLACK)
    assert np.any(tile == BLACK)
    assert len(set(map(int, np.unique(tile[tile != BLACK])))) >= 2
    return tile


def _force_template_bbox(tile: np.ndarray, palette: list[int]) -> None:
    q = tile.shape[0]
    if not np.any(tile[0, :] != BLACK):
        tile[0, random.randrange(q)] = random.choice(palette)
    if not np.any(tile[q - 1, :] != BLACK):
        tile[q - 1, random.randrange(q)] = random.choice(palette)
    if not np.any(tile[:, 0] != BLACK):
        tile[random.randrange(q), 0] = random.choice(palette)
    if not np.any(tile[:, q - 1] != BLACK):
        tile[random.randrange(q), q - 1] = random.choice(palette)


def _embed_template(quadrant_size: int, template: np.ndarray) -> np.ndarray:
    th, tw = template.shape
    raw = np.zeros((quadrant_size, quadrant_size), dtype=int)
    r0 = random.randint(0, quadrant_size - th)
    c0 = random.randint(0, quadrant_size - tw)
    raw[r0:r0 + th, c0:c0 + tw] = template
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


def _expected_output(source: np.ndarray, raw_template: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    template = _extract_active_template(raw_template)
    th, tw = template.shape
    coarse_source = _downsample_source(source, th, tw)

    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = template[rr % th, cc % tw].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = coarse_source[rr[bg_mask] // th, cc[bg_mask] // tw]
    return tiled


def _original_expected_output(source: np.ndarray, raw_template: np.ndarray,
                              out_h: int, out_w: int) -> np.ndarray:
    q = source.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = raw_template[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = source[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _sample_disjoint_palettes() -> tuple[list[int], list[int], list[int]] | None:
    source_k = random.randint(*SOURCE_COLOR_RANGE)
    template_k = random.randint(*TEMPLATE_COLOR_RANGE)

    colors = list(COLORS)
    random.shuffle(colors)
    total_needed = source_k + template_k
    if len(colors) <= total_needed:
        return None

    source_palette = colors[:source_k]
    template_palette = colors[source_k:source_k + template_k]
    control_pool = colors[source_k + template_k:]
    if not control_pool:
        return None
    return source_palette, template_palette, control_pool


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    template_q = q // SOURCE_SCALE

    sampled = _sample_disjoint_palettes()
    if sampled is None:
        return None
    source_palette, template_palette, control_pool = sampled

    valid_dims = [
        (h, w)
        for h in range(OUTPUT_DIM_MIN, min(OUTPUT_DIM_MAX, template_q * template_q) + 1)
        for w in range(OUTPUT_DIM_MIN, min(OUTPUT_DIM_MAX, template_q * template_q) + 1)
        if h * w <= OUTPUT_AREA_MAX
        and ceil(h / template_q) * ceil(w / template_q) >= max(3, len(source_palette))
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(ceil(out_h / template_q)) for c in range(ceil(out_w / template_q))]
    coarse_source = _make_coarse_source(template_q, source_palette, used_fill)
    source = _expand_source(coarse_source, SOURCE_SCALE)
    template = _make_multicolor_template(template_q, template_palette)
    raw_template = _embed_template(q, template)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = source
    grid[q:, q:] = raw_template

    expected = _expected_output(source, raw_template, out_h, out_w)
    baseline = _original_expected_output(source, raw_template, out_h, out_w)

    template_colors = set(map(int, np.unique(template[template != BLACK])))
    output_colors = set(map(int, np.unique(expected)))
    used_source_colors = set(map(int, np.unique(coarse_source[:ceil(out_h / template_q), :ceil(out_w / template_q)])))

    if len(template_colors) < 2:
        return None
    if len(output_colors & template_colors) < 2:
        return None
    if len(output_colors & used_source_colors) < 2:
        return None
    if np.array_equal(expected, baseline):
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
