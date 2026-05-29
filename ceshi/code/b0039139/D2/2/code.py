# Variant of b0039139
# Dimension: D2 — 模板形状定义方式 (src)
# Change: 区段1包含一个用色4绘制的形状模板，裁剪为其非背景像素的外接矩形... → 模板旋转90度后再平铺

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D2 rotation variant
# - concrete_behavior:
#   The input still has four sections separated by full blue lines: template, stencil,
#   foreground swatch, background swatch. The template remains a single yellow shape on
#   black with a 1-cell black border. The output still repeats one recolored tile N times,
#   where N is the number of 4-connected green blobs in the stencil. The only changed step
#   is that the cropped template mask is rotated 90 degrees clockwise before recoloring.
# - example_trace:
#   Original: a cropped 2x3 mask shaped like [[1,1,0],[0,1,1]] becomes a 2x3 recolored
#   tile and is tiled along the separator axis.
#   Variant: the same crop is first rotated to a 3x2 mask [[0,1],[1,1],[1,0]], then those
#   foreground/background cells are recolored and tiled. The repeat count and separator
#   color stay unchanged; only the tile geometry changes.
# - complexity_calibration:
#   This keeps a single deterministic transform (always clockwise 90 degrees), matching the
#   source idea of a visible rotation without adding multiple branches such as mirror vs.
#   counterclockwise. To keep the change observable, the generator rejects templates whose
#   cropped mask is identical to its 90-degree rotation.
# - old_assumptions_audit:
#   In the original main(), `tile_mask = crop_bbox(template_mask, template_bbox)` assumed
#   that the cropped orientation was already the final tile orientation. That is no longer
#   valid. Section splitting, swatch extraction, component counting, and repeat-axis logic
#   do not depend on that assumption and remain correct.
# - placement_legality:
#   Rotating happens only after cropping inside the solver, so the input template placement
#   and the four-section assembly remain legal. The generator still gives the template a
#   1-cell border and sizes the stencil/swatches by the template section's shared axis, so
#   no input section overlaps or goes out of bounds.
# - visual_observability:
#   A solver can infer the new rule from 2-4 examples because the stencil and swatches keep
#   their old meanings while the output tile has the same internal pattern but turned. The
#   generator enforces `rotated_crop != crop`, preventing silent fallback to the original
#   task on symmetric samples.
# - cleanliness_and_color_inference:
#   No extra markers are added. Colors keep the same structural cues as the base task:
#   green encodes count, the two solid swatches encode output colors, blue only separates
#   sections, and yellow only marks the template source.
# - new_solver_assumptions_review:
#   The new main() assumes the template has at least one non-black pixel so a bbox exists,
#   and assumes clockwise rotation is always the intended transform. The generator guarantees
#   the first by creating a connected template sprite, and guarantees the second by building
#   the oracle output from that exact rotation.


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


def _rotate_clockwise(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, k=3)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate a cropped template, recolor it, and repeat it from a component-count stencil.

    Input structure:
    - Full blue (1) separator rows or columns divide the grid into four sections.
    - Section 1 is a black-background template drawn in color 4; section 2 is the repeat-count stencil.
    - Sections 3 and 4 are solid swatches giving the output foreground and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels, rotate that crop 90 degrees clockwise, and recolor it with the two swatches.
    - Count the 4-connected non-black components in section 2.
    - Repeat the rotated recolored tile that many times, inserting one background-colored separator row/column between copies.

    Key insight: The stencil still contributes only the repeat count and the separator layout still sets
    the tiling axis, but the visible tile must be inferred from a clockwise rotation of the cropped template.
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
    tile_mask = _rotate_clockwise(tile_mask)
    tile = np.where(tile_mask, foreground, background).astype(int)

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
SHAPE_RETRIES = 80                   # retries to get a visibly rotated template
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


def _rotation_is_visible(shape: np.ndarray) -> bool:
    return not np.array_equal(shape, _rotate_clockwise(shape))


def _make_shape() -> np.ndarray | None:
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
        if _rotation_is_visible(shape != 0):
            return shape
    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape = _make_shape()
    if shape is None:
        return None
    rotated_shape = _rotate_clockwise(shape != 0)

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    tile = np.where(rotated_shape, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    crop_h, crop_w = shape.shape
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
