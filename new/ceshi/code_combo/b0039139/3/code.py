# Combo variant of b0039139 (index 3)
# combo_spec: D1/0+D2/0
# Name: 双重反转：负空间加颜色互换

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant D1/0 + D2/0
# a) Concretization and behavior trace
#    - Layout stays the same as the seed task: four ordered sections separated by full
#      blue lines. Section 1 is the template source, section 2 is the repetition stencil,
#      sections 3 and 4 are solid swatches.
#    - D2/0 flips how section 1 should be read: black cells are now the shape, embedded
#      as holes inside a yellow plaque. Therefore the solver must use `template == 0`
#      as the shape mask and crop that zero-mask bbox.
#    - D1/0 flips how the swatches recolor the crop: section 4 colors the recovered
#      shape cells, while section 3 colors the crop background and the between-copy gaps.
#    - Small trace:
#      negative template section =
#        4 4 4 4
#        4 0 0 4
#        4 0 4 4
#        4 4 4 4
#      section 3 = red(2), section 4 = maroon(9)
#      old seed reading would use nonzero as shape, crop the whole 4x4 plaque, and paint
#      it mostly with section 3. The fused rule instead reads only the zero holes, crops
#      to [[0,0],[0,4]], and outputs [[9,9],[9,2]].
#      The two inversions do not cancel: one changes which pixels define the tile, and
#      the other changes which swatch colors those pixels receive.
#
# b) Complexity calibration
#    - The task remains a single clean pipeline: decode template -> recolor -> count
#      stencil components -> tile along the separator axis. There is no branching on
#      color ids or shape categories.
#    - To keep both inversions observable, the generator rejects solid crops and keeps
#      only shapes whose bbox contains both shape cells and non-shape cells. That makes
#      the swapped swatch roles visible inside each tile rather than only in separators.
#
# c) Implicit-assumption audit of the original main()
#    - `_split_sections(...)` assumed full separator rows/cols and four non-empty blocks.
#      This still holds; the generator preserves the same layout contract.
#    - `template_mask = template != 0` no longer holds. On the fused task that would read
#      the yellow plaque as shape, enlarge the bbox to the full framed section, and erase
#      the negative-space semantics. It must become `template == 0`.
#    - `bbox(template_mask)` still assumes at least one shape cell exists. The generator
#      guarantees at least one zero hole because it starts from a connected positive mask.
#    - `_solid_color(...)` assumes the swatch sections are effectively solid. The generator
#      still makes them single-color rectangles.
#    - The original variable names foreground/background become misleading. Under the fused
#      rule, section 3 is the output background/separator color and section 4 is the output
#      shape color.
#    - The repeat separator color must match section 3, not section 4, otherwise the D1/0
#      inversion would only apply inside the tile and the output semantics would be mixed.
#
# d) Placement legality
#    - The negative template is built as a `(crop_h + 2) x (crop_w + 2)` yellow plaque
#      with the inner `crop_h x crop_w` window holding 0 on shape cells and 4 elsewhere.
#      This keeps the zero-mask bbox strictly inside the border and avoids overlap.
#    - The stencil keeps the original 2-cell green blobs with a one-cell black moat, so
#      component counting stays stable and no blobs merge accidentally.
#    - Swatches remain solid rectangles sharing the cross-axis dimension with the other
#      sections, so joining with single separator rows/cols always yields a legal rectangle.
#
# e) Visual observability
#    - A human can see the fused rule from a few examples because section 1 visibly looks
#      like a yellow plaque with black cut-outs, while the output paints those cut-outs
#      using section 4 rather than section 3.
#    - The output background and the between-copy gaps consistently match section 3. This
#      repeated cue supports the color-role inversion without relying on numeric color ids.
#
# f) Visual cleanliness
#    - The four semantic roles remain segregated into four sections; no extra overlays or
#      markers are added. The only local complexity is inside the template plaque, which
#      still uses just two colors and a border.
#    - Output remains a one-dimensional repetition, so the negative-space decoding and the
#      color swap are easy to compare across repeated copies.
#
# g) Change inferability
#    - Only two swatch colors participate in the final image, and their roles are tied to
#      fixed spatial sections rather than hidden properties.
#    - The generator ensures both output colors appear inside the tile, so the section-3 /
#      section-4 role swap can be inferred from structure instead of from separators alone.
#
# Post-change solver assumption audit
#    - `main()` assumes section 1 contains at least one zero hole and at least one yellow
#      cell inside the cropped bbox. `generate_input()` enforces this by rejecting solid
#      shapes and embedding them as negative space.
#    - `main()` assumes section 3 and section 4 are distinct; otherwise the D1 inversion
#      would be invisible. `generate_input()` samples the two swatch colors without replacement.
#    - `main()` assumes the repeated separator uses the same color as the tile background.
#      The generator constructs the oracle output with section 3 in both positions.


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
    Decode a negative-space template, swap the swatch roles, and repeat the result by stencil count.

    Input structure:
    - Solid blue (1) separator rows or columns divide the grid into four ordered sections.
    - Section 1 is a yellow plaque whose black holes define the template shape; section 2 is a repeat-count stencil.
    - Sections 3 and 4 are solid color swatches, but their output roles are reversed from the seed task.

    Rule:
    - Read section 1 by treating black cells as the shape, crop to the bbox of those black cells, and keep the crop's black/non-black pattern.
    - Recolor that crop so shape cells take section 4's color while the crop background takes section 3's color.
    - Count 4-connected non-black components in section 2 and repeat the recolored crop that many times, inserting one section-3-colored row or column between copies.

    Key insight: The stencil still encodes only how many copies to make, but two independent inversions govern the tile itself:
    the template is recovered from negative space, and the usual foreground/background swatch roles are swapped.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, section3_block, section4_block = sections[:4]

    template_mask = template == 0
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
TEMPLATE_COLOR = 4                   # yellow plaque color; zeros carve the shape
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
    """Place n_blobs 2-cell blobs (1x2 or 2x1) of STENCIL_COLOR on a black grid."""
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

        grid[r : r + br, c : c + bc] = STENCIL_COLOR
        placed += 1

    return grid if placed == n_blobs else None


def _generate_shape_mask(crop_h: int, crop_w: int) -> np.ndarray:
    """Generate a connected shape whose bbox contains both shape and non-shape cells."""
    for _ in range(GEN_RETRIES):
        density = random.uniform(*DENSITY_RANGE)
        shape = random_sprite(
            crop_h,
            crop_w,
            density=density,
            color_palette=[TEMPLATE_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        shape_mask = shape != 0
        if np.any(shape_mask) and np.any(~shape_mask):
            return shape_mask
    raise RuntimeError("Failed to generate a non-solid template shape")


def _make_negative_template(shape_mask: np.ndarray) -> np.ndarray:
    """Embed the positive shape as black holes inside a yellow framed plaque."""
    crop_h, crop_w = shape_mask.shape
    template = np.full((crop_h + 2, crop_w + 2), TEMPLATE_COLOR, dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = np.where(shape_mask, 0, TEMPLATE_COLOR)
    return template


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    shape_mask = _generate_shape_mask(crop_h, crop_w)

    # Section 3 becomes output background; section 4 becomes output shape.
    section3_color, section4_color = random.sample(COLOR_POOL, 2)

    tile = np.where(shape_mask, section4_color, section3_color).astype(int)
    output = _tile_repeat(tile, n_repeats, section3_color, horizontal)

    template = _make_negative_template(shape_mask)

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    section3_swatch = _make_swatch(shared_dim, section3_color, horizontal)
    section4_swatch = _make_swatch(shared_dim, section4_color, horizontal)
    inp = _join_sections([template, stencil, section3_swatch, section4_swatch], shared_dim, horizontal)

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
