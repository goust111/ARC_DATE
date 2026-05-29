# Variant of b0039139
# Dimension: D2 — 模板形状定义方式 (src)
# Change: 区段1包含一个用色4绘制的形状模板，裁剪为其非背景像素的外接矩形... → 模板水平镜像后再平铺

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite

# Design analysis for the D2 reflection variant
# ---------------------------------------------
# concretization:
#   layout, stencil semantics, and color swatches stay identical to the source task:
#   four sections separated by blue 1-lines; section 2 still stores only the repeat
#   count via 4-connected green blobs; sections 3 and 4 still provide the output
#   foreground/background colors. The only rule change is inside section 1 handling:
#   after cropping the yellow template to its non-black bbox, the cropped mask is
#   flipped left-right before recoloring. Example:
#     source crop mask  [[1,1,0],      mirrored mask [[0,1,1],
#                        [1,0,0],  ->                 [0,0,1],
#                        [1,1,1]]                     [1,1,1]]
#   The original task would recolor the left matrix directly; this variant recolors
#   the mirrored one and repeats that mirrored tile according to the same stencil.
# complexity calibration:
#   A pure reflection change is too weak if the template is horizontally symmetric,
#   because the mirrored tile would equal the original tile and the puzzle would
#   collapse back to b0039139. Therefore generation must enforce visible horizontal
#   asymmetry, but should not otherwise add new objects or extra codes. Using the
#   existing 2-5 x 2-5 random connected template keeps difficulty close to source.
# implicit-assumption audit:
#   - _split_sections only assumes full separator rows/cols of color 1; unchanged.
#   - bbox(template != 0) still works because the template remains a single yellow
#     shape on black and still needs cropping.
#   - The original `tile = where(tile_mask, fg, bg)` silently assumed the cropped
#     mask itself is the final geometry. That assumption is invalid now; we must
#     insert `np.fliplr(...)` before recoloring.
#   - Repetition count from section 2 and orientation from separator direction remain
#     valid; no other part of main should change.
# placement legality:
#   The template still sits inside a 1-cell black border, so cropping is well-defined.
#   The mirrored output uses the same bbox dimensions as the cropped template, so
#   shared_dim calculations and swatch joining logic remain correct. Stencil blobs
#   are still separated by black to keep component counting unambiguous.
# visual observability:
#   Each generated template is required to differ from its horizontal mirror. With
#   distinct foreground/background colors, the output tile visibly changes side-to-
#   side, letting a human infer "mirror then tile" from a few examples.
# visual cleanliness:
#   No new overlay or mixed encoding is introduced. Reflection happens only after
#   cropping, so outputs remain regular repeated rectangles separated by a single
#   background row/column exactly as in the source task.
# color inference:
#   Colors still come only from the two swatches. Reflection does not depend on color
#   ids, only on visible left-right geometry, so the change remains inferable from the
#   structure alone.
# new main() assumptions and generator guarantees:
#   - main now assumes section 1's cropped mask is not horizontally symmetric if the
#     variant should be observable. generate_input() enforces that explicitly.
#   - main assumes there are at least four sections and section 2 components are
#     isolated enough to count correctly. The generator preserves the original layout
#     construction, which guarantees both properties.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror a cropped template horizontally, recolor it, and repeat it from a count stencil.

    Input structure:
    - Full blue (1) separator rows or columns divide the grid into four sections.
    - Section 1 is a yellow-on-black template; section 2 is a non-black component stencil.
    - Sections 3 and 4 are solid swatches giving the output foreground and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels, flip that crop left-right, then recolor it.
    - Count the 4-connected non-black components in section 2 to get the repeat count.
    - Tile the mirrored recolored crop along the separator axis, inserting one background-colored row or column between copies.

    Key insight: The stencil still contributes only a number, while the geometric change
    comes from recognizing that the template must be reflected horizontally before tiling.
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
    cropped_mask = crop_bbox(template_mask, template_bbox)
    mirrored_mask = np.fliplr(cropped_mask)
    tile = np.where(mirrored_mask, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    if orientation == "horizontal":
        sep = np.full((tile.shape[0], 1), background, dtype=int)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), background, dtype=int)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)          # blob count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # fg/bg color choices (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
STENCIL_COLOR = 3                    # blob color in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200             # per-blob placement attempts in stencil
SHAPE_RETRIES = 80                   # attempts to get a visibly mirror-changing template
GEN_RETRIES = 300                    # outer generate_input retry limit


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with 1-pixel bg_color separator."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    """Create a solid color swatch. shared_dim is the fixed axis; varying axis is random."""
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
    """Join sections with single-pixel separator lines of SEPARATOR_COLOR."""
    if horizontal:
        sep = np.full((shared_dim, 1), SEPARATOR_COLOR, dtype=int)
        parts = [sections[0]]
        for s in sections[1:]:
            parts.extend([sep, s])
        return np.hstack(parts)
    sep = np.full((1, shared_dim), SEPARATOR_COLOR, dtype=int)
    parts = [sections[0]]
    for s in sections[1:]:
        parts.extend([sep, s])
    return np.vstack(parts)


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place n_blobs 2-cell blobs (1x2 or 2x1) of STENCIL_COLOR on a black grid.
    Blobs are separated by >=1 cell of black in all directions."""
    min_varying = max(4, n_blobs * 2 + 2)
    varying_dim = random.randint(min_varying, min_varying + STENCIL_VARYING_EXTRA)

    h, w = (shared_dim, varying_dim) if horizontal else (varying_dim, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    placed = 0
    for _ in range(BLOB_PLACE_RETRIES):
        if placed >= n_blobs:
            break
        br, bc = (1, 2) if random.choice([True, False]) else (2, 1)

        if h - 1 - br < 1 or w - 1 - bc < 1:
            continue
        r = random.randint(1, h - 1 - br)
        c = random.randint(1, w - 1 - bc)

        r0, c0 = max(0, r - 1), max(0, c - 1)
        r1, c1 = min(h, r + br + 1), min(w, c + bc + 1)
        if np.any(grid[r0:r1, c0:c1] != 0):
            continue

        grid[r:r + br, c:c + bc] = STENCIL_COLOR
        placed += 1

    return grid if placed == n_blobs else None


def _is_visible_reflection(shape: np.ndarray) -> bool:
    """Require the mirrored crop to differ from the original crop."""
    mask = shape != 0
    return bool(mask.any()) and not np.array_equal(mask, np.fliplr(mask))


def _make_shape() -> np.ndarray | None:
    """Generate a connected yellow template whose horizontal mirror is visibly different."""
    for _ in range(SHAPE_RETRIES):
        crop_h = random.randint(*CROP_SIZE_RANGE)
        crop_w = random.randint(*CROP_SIZE_RANGE)
        density = random.uniform(*DENSITY_RANGE)
        shape = random_sprite(
            crop_h,
            crop_w,
            density=density,
            color_palette=[TEMPLATE_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        if _is_visible_reflection(shape):
            return shape
    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape = _make_shape()
    if shape is None:
        return None
    crop_h, crop_w = map(int, shape.shape)

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    mirrored_shape = np.fliplr(shape != 0)
    tile = np.where(mirrored_shape, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1:crop_h + 1, 1:crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

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
    raise RuntimeError("Failed to generate valid input after %d attempts" % GEN_RETRIES)
