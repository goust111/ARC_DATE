# Combo variant of b0039139 (index 29)
# combo_spec: D4/1+D7/1
# Name: 二维网格中分量形状决定副本变换

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for combo D4/1 + D7/1
# --------------------------------------
# a) Concrete behavior and trace
#    - The original task splits the input into four blue-separated sections:
#      template, stencil, foreground swatch, background swatch. It crops the
#      template, recolors it, counts stencil components, then repeats one
#      identical copy along a single axis.
#    - This fused variant changes the expansion stage in two coordinated ways.
#      First, the number of stencil components N is mapped to a 2D grid using a
#      factorization rule instead of a 1D strip. We use the largest factor not
#      exceeding sqrt(N) as the row count, and `cols = N // rows`.
#      Examples: `4 -> 2x2`, `6 -> 2x3`.
#    - Second, each green stencil component contributes one tile whose transform
#      is selected by that component's own geometry:
#        * 1x3 horizontal bar -> keep the tile unchanged
#        * 3x1 vertical bar   -> rotate the tile 90 degrees clockwise
#        * 2x2 square block   -> mirror the tile left-to-right
#    - Example trace for N=6:
#        * Source logic: any six stencil components would yield the same tile
#          copied six times in one line.
#        * Fused logic: the six components are first sorted in row-major order,
#          grouped into a `2x3` matrix, then their shapes pick per-slot views
#          such as `[id, mirror, rot; rot, id, mirror]`.
#      The crop/recolor stages are unchanged; only the "how many copies and what
#      each copy looks like" stage is replaced by a 2D puzzle assembly.
# b) Complexity calibration
#    - Only three component classes are used, matching the single D7 variant, so
#      the transform vocabulary stays learnable.
#    - The generator samples only composite repeat counts with both factors > 1,
#      so every sample is genuinely 2D rather than collapsing back to a strip.
#    - Every sample includes all three component classes at least once, which
#      gives repeated support for each transform mapping in a small number of
#      examples.
# c) Hidden-assumption audit of main()
#    - `_split_sections` still assumes full blue separator rows or columns. That
#      remains valid.
#    - The original `n_repeats = len(components...)` assumed the stencil encoded
#      only multiplicity; that is no longer valid because component geometry now
#      matters for the corresponding tile.
#    - The original output code assumed a single concatenation axis chosen by the
#      separator orientation. That assumption is broken by D4/1; output layout is
#      now determined by factorization of the component count, not by separator
#      direction.
#    - Rotation changes height/width unless the tile is square. The solver
#      therefore pads the cropped tile to a square canvas before transforming.
#      The generator already produces square, asymmetric crops so this is visible
#      rather than a no-op.
#    - Row-major grouping assumes the stencil components occupy a visible 2D
#      lattice with stable row bands and column bands. The generator guarantees
#      that by placing each component into a dedicated slot with gaps.
# d) Placement legality
#    - The stencil is generated on a slot lattice with one component per slot and
#      at least one black cell separating neighboring slots horizontally and
#      vertically, so components never touch or merge.
#    - Template, stencil, and swatches are assembled with the original full-line
#      blue separators. The template canvas is enlarged when needed so the shared
#      dimension can hold a true 2D stencil even in side-by-side section layouts.
# e) Visual observability
#    - The stencil itself visibly shows both cues humans need: total component
#      count determines the matrix shape, and each component's shape determines
#      the transformed appearance of its matching output copy.
#    - The template crop is forced to be asymmetric and to differ under identity,
#      rotation, and mirror, so the transform effect is obvious in the output.
# f) Visual cleanliness
#    - The output uses a single background-colored separator row and column
#      between tile cells, yielding a clean board-like layout rather than a dense
#      collage.
#    - The stencil stays black-background with isolated green parts; no mixed
#      color annotation is added inside the same region.
# g) Non-degeneracy checks
#    - Fusion would degenerate if N were prime (1xN strip) or if the template
#      were symmetric enough that rotation/mirror matched the original. The
#      generator explicitly avoids both cases.
#    - A useful sample-level metric is: `rows > 1 and cols > 1` plus
#      `fused_output != plain_2d_repeat_output`. The post-write validation script
#      measures both so we can confirm the two dimensions are active together.


COMPONENT_SHAPES = {
    "horizontal": np.full((1, 3), 3, dtype=int),
    "vertical": np.full((3, 1), 3, dtype=int),
    "square": np.full((2, 2), 3, dtype=int),
}


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = 1

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == sep))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        return "vertical", [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == sep))]
    cuts = [-1] + sep_cols + [w]
    return "horizontal", [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _square_tile(tile: np.ndarray, background: int) -> np.ndarray:
    h, w = tile.shape
    if h == w:
        return tile
    side = max(h, w)
    out = np.full((side, side), background, dtype=int)
    r0 = (side - h) // 2
    c0 = (side - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = tile
    return out


def _component_kind(coords: np.ndarray) -> str:
    rows = coords[:, 0]
    cols = coords[:, 1]
    height = int(rows.max() - rows.min() + 1)
    width = int(cols.max() - cols.min() + 1)
    if width > height:
        return "horizontal"
    if height > width:
        return "vertical"
    return "square"


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    rows = 1
    for candidate in range(int(math.isqrt(n)), 0, -1):
        if n % candidate == 0:
            rows = candidate
            break
    cols = n // rows
    return int(rows), int(cols)


def _component_center(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _sorted_components_row_major(pattern: np.ndarray) -> list[np.ndarray]:
    comps = components_mask(pattern != 0, connectivity=4)
    if not comps:
        return []

    rows, cols = _grid_shape(len(comps))
    if rows <= 1 or cols <= 1:
        return sorted(comps, key=lambda comp: _component_center(comp))

    by_row = sorted(comps, key=lambda comp: _component_center(comp))
    ordered: list[np.ndarray] = []
    for r in range(rows):
        row_group = by_row[r * cols : (r + 1) * cols]
        row_group = sorted(row_group, key=lambda comp: (_component_center(comp)[1], _component_center(comp)[0]))
        ordered.extend(row_group)
    return ordered


def _transform_tile(tile: np.ndarray, kind: str) -> np.ndarray:
    if kind == "vertical":
        return np.rot90(tile, k=3)
    if kind == "square":
        return np.fliplr(tile)
    return tile


def _join_tiles_grid(tiles: list[np.ndarray], rows: int, cols: int, background: int) -> np.ndarray:
    if not tiles:
        return np.empty((0, 0), dtype=int)

    tile_h, tile_w = tiles[0].shape
    out = np.full((rows * tile_h + (rows - 1), cols * tile_w + (cols - 1)), background, dtype=int)
    for idx, tile in enumerate(tiles):
        if tile.shape != (tile_h, tile_w):
            raise ValueError("all transformed tiles must have the same shape")
        r = idx // cols
        c = idx % cols
        r0 = r * (tile_h + 1)
        c0 = c * (tile_w + 1)
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and place transformed copies into a factorized 2D grid.

    Input structure:
    - Full blue (1) separator rows or columns split the input into four sections.
    - Section 1 is a black-background template, section 2 is a green component board,
      and sections 3-4 are solid swatches that provide the output foreground/background.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels, recolor that crop with the
      swatches, and pad it to a square base tile if needed.
    - Find the 4-connected components in section 2. Let N be the component count, map
      `N` to `(rows, cols)` by taking the largest factor not exceeding `sqrt(N)` as the
      row count, and order the components in row-major reading order.
    - For each ordered component, use its shape to choose the tile view: horizontal bar
      -> unchanged, vertical bar -> rotate 90 degrees clockwise, square block -> mirror
      left-to-right. Place the resulting copies into the `rows x cols` output grid with
      one background-colored separator row and column between neighboring cells.

    Key insight: The stencil no longer says only "how many copies"; its component count
    chooses the 2D grid shape, while each component's own geometry chooses the transform
    of the tile occupying the matching grid slot.
    """
    g = np.asarray(grid, dtype=int)
    _orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)
    tile = _square_tile(tile, background)

    components = _sorted_components_row_major(repetition_pattern)
    if not components:
        return tile

    rows, cols = _grid_shape(len(components))
    transformed_tiles = [_transform_tile(tile, _component_kind(comp)) for comp in components]
    return _join_tiles_grid(transformed_tiles, rows, cols, background)


# --------------- generator (c) ---------------

REPEAT_COUNTS = (4, 6)
CROP_SIZE_RANGE = (3, 5)
DENSITY_RANGE = (0.50, 0.80)
COLOR_POOL = list(range(2, 10))
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 8)
GEN_RETRIES = 400
SHAPE_RETRIES = 150
SLOT_SPAN = 4


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for section in sections[1:]:
            parts.extend([sep, section])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for section in sections[1:]:
        parts.extend([sep, section])
    return np.vstack(parts)


def _sample_shape(size: int) -> np.ndarray | None:
    target_bbox = (0, 0, size - 1, size - 1)
    for _ in range(SHAPE_RETRIES):
        density = random.uniform(*DENSITY_RANGE)
        shape = random_sprite(
            size,
            size,
            density=density,
            color_palette=[TEMPLATE_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        mask = shape != 0
        if bbox(mask) != target_bbox:
            continue

        base = mask.astype(int)
        variants = {
            base.tobytes(),
            np.rot90(base, k=3).tobytes(),
            np.fliplr(base).tobytes(),
        }
        if len(variants) < 3:
            continue
        return shape
    return None


def _sample_component_kinds(n_blobs: int) -> list[str]:
    kinds = ["horizontal", "vertical", "square"]
    while len(kinds) < n_blobs:
        kinds.append(random.choice(["horizontal", "vertical", "square"]))
    random.shuffle(kinds)
    return kinds


def _stencil_dims(rows: int, cols: int) -> tuple[int, int]:
    return 2 + rows * SLOT_SPAN, 2 + cols * SLOT_SPAN


def _build_stencil(kinds: list[str], rows: int, cols: int) -> np.ndarray:
    h, w = _stencil_dims(rows, cols)
    grid = np.zeros((h, w), dtype=int)
    for idx, kind in enumerate(kinds):
        slot_r = idx // cols
        slot_c = idx % cols
        comp = COMPONENT_SHAPES[kind]
        ch, cw = comp.shape
        jitter_r = random.randint(0, max(0, SLOT_SPAN - 1 - ch))
        jitter_c = random.randint(0, max(0, SLOT_SPAN - 1 - cw))
        r0 = 1 + slot_r * SLOT_SPAN + jitter_r
        c0 = 1 + slot_c * SLOT_SPAN + jitter_c
        grid[r0 : r0 + ch, c0 : c0 + cw] = comp
    return grid


def _embed_template(shape: np.ndarray, canvas_size: int) -> np.ndarray:
    crop_size = shape.shape[0]
    out = np.zeros((canvas_size, canvas_size), dtype=int)
    r0 = (canvas_size - crop_size) // 2
    c0 = (canvas_size - crop_size) // 2
    out[r0 : r0 + crop_size, c0 : c0 + crop_size] = shape
    return out


def _plain_grid_repeat(tile: np.ndarray, rows: int, cols: int, background: int) -> np.ndarray:
    return _join_tiles_grid([tile.copy() for _ in range(rows * cols)], rows, cols, background)


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.choice(REPEAT_COUNTS)
    rows, cols = _grid_shape(n_repeats)
    kinds = _sample_component_kinds(n_repeats)

    crop_size = random.randint(*CROP_SIZE_RANGE)
    shape = _sample_shape(crop_size)
    if shape is None:
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    transformed_tiles = [_transform_tile(tile, kind) for kind in kinds]
    output = _join_tiles_grid(transformed_tiles, rows, cols, bg_color)

    baseline = _plain_grid_repeat(tile, rows, cols, bg_color)
    if np.array_equal(output, baseline):
        return None

    stencil = _build_stencil(kinds, rows, cols)
    shared_dim = stencil.shape[0] if horizontal else stencil.shape[1]
    template = _embed_template(shape, shared_dim)
    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
