# Variant of b0039139
# Dimension: D7 — 计数板中连通分量的空间排布 (src)
# Change: 区段2中的色3连通分量排布在网格内，其数量决定重复次数，但分量的具体位置和形状不影响输出... → 分量的形状决定副本的变换

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D7 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The four-section layout stays the same: template, stencil, fg swatch, bg swatch.
#    - The template is still cropped and recolored, but the stencil no longer contributes
#      only a count. Each green connected component is now one of three solid shapes:
#        * 1x3 horizontal bar  -> keep the tile unchanged
#        * 3x1 vertical bar    -> rotate the tile 90 degrees clockwise
#        * 2x2 square block    -> mirror the tile left-to-right
#    - Copies are emitted in the stencil's reading order along the output axis:
#        * side-by-side sections  -> sort stencil components left-to-right
#        * stacked sections       -> sort stencil components top-to-bottom
#    - Example trace:
#        * Old rule: three stencil blobs of any geometry => tile, tile, tile.
#        * New rule: [horizontal bar, square, vertical bar] =>
#          [tile, mirrored tile, rotated tile].
#      The changed step is exactly the "expand repetitions" stage; the split/crop/recolor
#      stages remain the same.
# b) Complexity calibration
#    - The variant uses only three geometry classes, which is small enough to infer from
#      a few examples but rich enough to create visibly different outputs.
#    - To keep all three branches learnable, the generator uses 3-6 components and makes
#      every sample contain all three shape classes at least once.
# c) Hidden-assumption audit of main()
#    - _split_sections still assumes blue full-width/full-height separators; unchanged.
#    - The original `n_repeats = len(components...)` assumed all copies were identical.
#      That assumption is invalid here, so main() must iterate over the components rather
#      than only count them.
#    - The original concatenation assumed a single tile shape for all copies. Rotation can
#      change dimensions, so main() now squares the cropped tile canvas before applying
#      transforms. The generator keeps the crop square already, making this harmless.
#    - Classification by bbox aspect ratio assumes stencil components are solid bars or a
#      solid square; the generator guarantees exactly those shapes.
# d) Placement legality
#    - Stencil components are placed sequentially along the repeat axis with >=2 cells of
#      black gap, plus a black border, so they never touch or merge.
#    - The orthogonal coordinate is randomized within bounds, so components stay in-bounds
#      while preserving a clean ordering that the solver can recover by sorting.
# e) Visual observability
#    - The template crop is forced to be square and asymmetrical, and its identity /
#      rotation / mirror views must all differ. Without this, the D7 transformation would
#      often be invisible.
#    - The order cue is purely spatial and visible; no invisible color-id ordering is used.
# f) Visual cleanliness
#    - The stencil contains only one green component per local region on black, with ample
#      spacing. The output still uses one-color separators, so transformed copies remain
#      easy to segment by eye.
# g) Color-change inferability
#    - Colors keep their original role: one swatch supplies the foreground, one the
#      background. The new rule is purely geometric, so no extra color mapping burden is
#      introduced.


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


def _sorted_components(pattern: np.ndarray, orientation: str) -> list[np.ndarray]:
    comps = components_mask(pattern != 0, connectivity=4)
    if orientation == "horizontal":
        return sorted(comps, key=lambda comp: (int(comp[:, 1].min()), int(comp[:, 0].min())))
    return sorted(comps, key=lambda comp: (int(comp[:, 0].min()), int(comp[:, 1].min())))


def _transform_tile(tile: np.ndarray, kind: str) -> np.ndarray:
    if kind == "vertical":
        return np.rot90(tile, k=3)
    if kind == "square":
        return np.fliplr(tile)
    return tile


def _join_tiles(tiles: list[np.ndarray], background: int, horizontal: bool) -> np.ndarray:
    if len(tiles) == 1:
        return tiles[0]
    if horizontal:
        sep = np.full((tiles[0].shape[0], 1), background, dtype=int)
        pieces = [tiles[0]]
        for tile in tiles[1:]:
            pieces.extend([sep, tile])
        return np.hstack(pieces)
    sep = np.full((1, tiles[0].shape[1]), background, dtype=int)
    pieces = [tiles[0]]
    for tile in tiles[1:]:
        pieces.extend([sep, tile])
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and repeat transformed copies according to stencil geometry.

    Input structure:
    - Full blue (1) separator rows or columns divide the input into four sections.
    - Section 1 contains a black-background template; section 2 contains green components.
    - Sections 3 and 4 are solid swatches supplying the output foreground and background.

    Rule:
    - Crop section 1 to its non-black bbox, recolor it with the two swatches, and interpret
      that crop as the base tile.
    - Sort the 4-connected components in section 2 along the output axis. A horizontal bar
      keeps the tile unchanged, a vertical bar rotates it 90 degrees clockwise, and a square
      block mirrors it left-to-right.
    - Emit one transformed copy per component, separated by a single background-colored
      row or column.

    Key insight: The second section no longer stores only a repeat count; each component's
    geometry selects the transform for its corresponding copy, and its position fixes the
    copy order.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
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

    components = _sorted_components(repetition_pattern, orientation)
    if not components:
        return tile

    transformed_tiles = [_transform_tile(tile, _component_kind(comp)) for comp in components]
    return _join_tiles(transformed_tiles, background, orientation == "horizontal")


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (3, 6)
CROP_SIZE_RANGE = (3, 5)
DENSITY_RANGE = (0.50, 0.80)
COLOR_POOL = list(range(2, 10))
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
GEN_RETRIES = 400
SHAPE_RETRIES = 150

COMPONENT_SHAPES = {
    "horizontal": np.full((1, 3), STENCIL_COLOR, dtype=int),
    "vertical": np.full((3, 1), STENCIL_COLOR, dtype=int),
    "square": np.full((2, 2), STENCIL_COLOR, dtype=int),
}


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
        mask = (shape != 0)
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
    if n_blobs == 3:
        kinds = ["horizontal", "vertical", "square"]
        random.shuffle(kinds)
        return kinds

    kinds = ["horizontal", "vertical", "square"]
    while len(kinds) < n_blobs:
        kinds.append(random.choice(["horizontal", "vertical", "square"]))
    random.shuffle(kinds)
    return kinds


def _build_stencil(kinds: list[str], shared_dim: int, horizontal: bool) -> np.ndarray:
    components = [COMPONENT_SHAPES[kind] for kind in kinds]
    axis_sizes = [comp.shape[1] if horizontal else comp.shape[0] for comp in components]
    leading_gap = random.randint(1, 2)
    trailing_gap = random.randint(1, 2) + random.randint(0, STENCIL_VARYING_EXTRA)
    inner_gaps = [random.randint(2, 3) for _ in range(len(components) - 1)]
    varying_dim = leading_gap + sum(axis_sizes) + sum(inner_gaps) + trailing_gap

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    pos = leading_gap
    for idx, comp in enumerate(components):
        ch, cw = comp.shape
        if horizontal:
            row = random.randint(1, h - ch - 1)
            col = pos
            grid[row : row + ch, col : col + cw] = comp
            pos += cw
        else:
            row = pos
            col = random.randint(1, w - cw - 1)
            grid[row : row + ch, col : col + cw] = comp
            pos += ch
        if idx < len(inner_gaps):
            pos += inner_gaps[idx]

    return grid


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)
    kinds = _sample_component_kinds(n_repeats)

    crop_size = random.randint(*CROP_SIZE_RANGE)
    shape = _sample_shape(crop_size)
    if shape is None:
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    transformed_tiles = [_transform_tile(tile, kind) for kind in kinds]
    output = _join_tiles(transformed_tiles, bg_color, horizontal)

    template = np.zeros((crop_size + 2, crop_size + 2), dtype=int)
    template[1 : crop_size + 1, 1 : crop_size + 1] = shape

    shared_dim = crop_size + 2
    stencil = _build_stencil(kinds, shared_dim, horizontal)
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
