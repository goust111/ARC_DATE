# Variant of b0039139
# Dimension: D1 — 模板重着色规则 (transform)
# Change: 模板像素(色4)替换为区段3的颜色，模板内背景像素(色0)替换为区段4的颜色，形成双色裁剪图案...
#         → 颜色互换：前景变背景、背景变前景

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D1 variant
# a) Concretization and behavior trace
#    - The input layout stays identical: template, stencil, swatch-3, swatch-4, split by full blue lines.
#    - The template section still contains a yellow(4) connected shape on black(0), padded by a 1-cell black border.
#    - The stencil section still encodes only the repeat count via the number of 4-connected green(3) 2-cell blobs.
#    - The only semantic change is the palette application on the cropped template bbox:
#      old task: shape cells -> section 3 color, cropped black cells -> section 4 color, separators -> section 4 color.
#      new task: shape cells -> section 4 color, cropped black cells -> section 3 color, separators -> section 3 color.
#    - Small trace:
#      template crop = [[4,4],[4,0]], section 3 = red(2), section 4 = maroon(9)
#      old tile      = [[2,2],[2,9]]
#      new tile      = [[9,9],[9,2]]
#      Thus both the tile interior and the between-copy separator color flip roles.
#
# b) Complexity calibration
#    - The rule is still single-step and local, but not trivial because the two swatches must be interpreted
#      against opposite roles from the original intuition.
#    - To make the inversion visible, the generator rejects full solid crops and keeps shapes with at least one
#      black cell inside the crop bbox, so both swapped colors appear inside each tile, not just in separators.
#
# c) Implicit-assumption audit of the original main()
#    - _split_sections assumes full-width/full-height blue separator lines and four non-empty sections.
#      This remains valid; the generator still builds exactly that.
#    - template_mask = template != 0 assumes the template's informative pixels are all non-black.
#      Still valid because the template drawing color remains fixed at 4.
#    - bbox(template_mask) assumes at least one template pixel exists. The generator still creates a connected shape.
#    - _solid_color(...) assumes each swatch block is effectively solid. The generator preserves this.
#    - The original names foreground/background encoded the old mapping and would now be misleading.
#      The solver must instead treat section 3 as output background/separator color and section 4 as shape color.
#    - The repeat separator color must follow the new output background (section 3); leaving it on section 4 would
#      silently mix old and new semantics.
#
# d) Placement legality
#    - Template remains in a fixed (crop_h+2)x(crop_w+2) box with a 1-cell border, so cropping is unambiguous.
#    - Stencil blobs are still placed with a black moat, so connected-component counting is stable.
#    - Swatches remain solid rectangles sharing the cross-axis dimension with other sections, so joining by single
#      separator rows/cols yields a legal rectangular input.
#
# e) Visual observability
#    - The inversion is human-visible because section 3 now matches the holes/negative space and the inter-copy gaps,
#      while section 4 fills the actual template shape.
#    - Generator support: every sampled tile has both shape pixels and cropped-background pixels.
#
# f) Visual cleanliness
#    - Sections remain spatially separated by blue lines; no overlays are introduced.
#    - The output is still a clean 1D repetition, so the color-role swap is easy to compare across examples.
#
# g) Color-change inferability
#    - Only two non-blue/non-black swatch colors participate, and their roles are tied to explicit sections.
#    - No hidden ordering by numeric color id is used; the mapping is entirely spatial and repeated consistently.
#
# Post-change solver assumption audit
#    - main() now assumes section 3's solid color is the output background/separator color.
#      generate_input() guarantees this by constructing the oracle output with section 3 in those positions.
#    - main() assumes the cropped template bbox can contain both True and False cells for strong visual evidence.
#      This is not required for correctness, but the generator enforces it to prevent weak/degenerate examples.


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
    Crop a template, apply the two swatches with reversed roles, and repeat it by stencil count.

    Input structure:
    - Solid blue (1) separator rows or columns divide the grid into four ordered sections.
    - Section 1 is the yellow-on-black template; section 2 is the repeat-count stencil.
    - Sections 3 and 4 are solid color swatches whose roles are intentionally reversed in the output.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels.
    - Recolor the cropped template so template pixels take section 4's color, while cropped black pixels take section 3's color.
    - Count 4-connected non-black components in section 2 and repeat that tile many times, using section 3's color as the spacer.

    Key insight: The stencil still supplies only a repetition count, but the swatch interpretation is inverted:
    the visually "foreground" template shape is painted from section 4, and section 3 becomes the output background.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, section3_block, section4_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    output_background = _solid_color(section3_block)
    output_shape = _solid_color(section4_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, output_shape, output_background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    if orientation == "horizontal":
        sep = np.full((tile.shape[0], 1), output_background, dtype=int)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), output_background, dtype=int)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)          # blob count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # swatch colors (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
STENCIL_COLOR = 3                    # blob color in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200             # per-blob placement attempts in stencil
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

        # Check blob cells + 1-cell neighborhood are all black.
        r0, c0 = max(0, r - 1), max(0, c - 1)
        r1, c1 = min(h, r + br + 1), min(w, c + bc + 1)
        if np.any(grid[r0:r1, c0:c1] != 0):
            continue

        grid[r : r + br, c : c + bc] = STENCIL_COLOR
        placed += 1

    return grid if placed == n_blobs else None


def _shape_has_cropped_background(shape: np.ndarray) -> bool:
    """Require both shape and background cells inside the eventual crop for visual clarity."""
    return bool(np.any(shape == 0) and np.any(shape == TEMPLATE_COLOR))


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    # Generate a connected tile shape and keep only crops that visibly use both swapped colors.
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
    if not _shape_has_cropped_background(shape):
        return None

    # Pick distinct swatch colors: section 3 becomes output background, section 4 becomes output shape.
    section3_color, section4_color = random.sample(COLOR_POOL, 2)

    # Build output under the inverted palette rule.
    tile = np.where(shape != 0, section4_color, section3_color).astype(int)
    output = _tile_repeat(tile, n_repeats, section3_color, horizontal)

    # Build template: shape in (crop_h+2) x (crop_w+2) black border.
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    # Build stencil.
    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    # Assemble: template | stencil | section3_swatch | section4_swatch
    section3_swatch = _make_swatch(shared_dim, section3_color, horizontal)
    section4_swatch = _make_swatch(shared_dim, section4_color, horizontal)
    inp = _join_sections([template, stencil, section3_swatch, section4_swatch], shared_dim, horizontal)

    # Oracle check against the new solver.
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
