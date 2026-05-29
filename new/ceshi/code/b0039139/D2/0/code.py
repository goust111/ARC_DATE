# Variant of b0039139
# Dimension: D2 — 模板形状定义方式 (src)
# Change: 区段1包含一个用色4绘制的形状模板，裁剪为其非背景像素的外接矩形... → 模板为负空间：背景像素定义形状

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior and trace:
#    The original task stores the template as yellow shape cells (4) on black (0),
#    so `template != 0` is the shape mask and cropping its bbox recovers the tile.
#    In this variant, section 1 is visually inverted: a yellow field (4) contains a
#    black cut-out (0), and the black cut-out is the true shape. Example:
#      original crop source:      variant crop source:
#        0 4 4 0                    4 0 0 4
#        0 4 0 0                    4 0 4 4
#    The old logic would read the yellow cells as shape; the new logic must read the
#    zero cells as shape instead. Because the variant also keeps a 1-cell yellow border
#    around the cut-out, using `template != 0` would expand the bbox to the whole
#    section and produce the complement with an extra border, which is visibly wrong.
#
# b) Complexity calibration:
#    Only one conceptual axis changes: how section 1 encodes the template. The stencil,
#    swatches, section ordering, and tiling axis all stay unchanged. To keep the
#    inversion visually obvious, the generator rejects fully filled sprites so the
#    negative-space crop contains both 0 and 4 inside the cropped bbox.
#
# c) Hidden assumptions in the original main():
#    - `template_mask = template != 0` assumed non-black means shape. This is false now.
#    - `bbox(template_mask)` assumed background outside the shape is 0. This is false now
#      because the template section background is 4, so the old bbox would include the
#      whole framed section.
#    - `np.where(tile_mask, foreground, background)` is still valid after replacing the
#      mask definition, so recoloring does not need any further change.
#
# d) Placement legality:
#    The template section is still `(crop_h + 2) x (crop_w + 2)`, but now it is filled
#    with 4 and the inner `crop_h x crop_w` window stores 0 for shape cells and 4 for
#    non-shape cells. This guarantees no out-of-bounds writes and guarantees that the
#    solver's zero-mask bbox is exactly the intended inner crop.
#
# e) Visual observability:
#    Across examples, the output tile keeps the same recolor-and-repeat behavior while
#    section 1 consistently shows a silhouette as black holes inside a yellow plaque.
#    That makes the inversion legible from a few examples without introducing extra
#    branches or color-dependent exceptions.
#
# f) Clean visual layout:
#    The four-section layout and separator scheme are unchanged, so the negative-space
#    template remains isolated from the stencil and swatches. There is no overlap or
#    multi-layer drawing inside a single area beyond the two template colors.
#
# g) Color inference:
#    Colors still come from dedicated swatch sections. The variant does not add any new
#    color mapping; it only changes which template color is interpreted as foreground.


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
    Recover a negative-space template, recolor it, and repeat it by stencil count.

    Input structure:
    - Solid blue (1) separator rows or columns divide the grid into four sections.
    - Section 1 is a yellow (4) plaque whose black (0) holes define the template shape;
      section 2 is a component-count stencil.
    - Sections 3 and 4 are solid color swatches for the output foreground and background.

    Rule:
    - Read section 1 by treating black cells as the template, crop to the bbox of those
      black cells, and recolor the crop using the two swatches.
    - Count 4-connected non-black components in section 2.
    - Repeat the recolored crop that many times, inserting one background-colored
      row/column between copies.

    Key insight: The counting mechanism is unchanged; the only conceptual twist is that
    section 1 must be interpreted as a cut-out silhouette, not as painted foreground.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template == 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
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
TEMPLATE_COLOR = 4                   # template plaque color; zeros carve the shape
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

        r0, c0 = max(0, r - 1), max(0, c - 1)
        r1, c1 = min(h, r + br + 1), min(w, c + bc + 1)
        if np.any(grid[r0:r1, c0:c1] != 0):
            continue

        grid[r:r + br, c:c + bc] = STENCIL_COLOR
        placed += 1

    return grid if placed == n_blobs else None


def _generate_shape_mask(crop_h: int, crop_w: int) -> np.ndarray:
    """Generate a connected shape whose bbox still contains some non-shape cells."""
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
        if np.any(~shape_mask):
            return shape_mask
    raise RuntimeError("Failed to generate a non-solid template shape")


def _make_negative_template(shape_mask: np.ndarray) -> np.ndarray:
    """Embed the shape as black cut-outs inside a yellow framed section."""
    crop_h, crop_w = shape_mask.shape
    template = np.full((crop_h + 2, crop_w + 2), TEMPLATE_COLOR, dtype=int)
    template[1:crop_h + 1, 1:crop_w + 1] = np.where(shape_mask, 0, TEMPLATE_COLOR)
    return template


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    shape_mask = _generate_shape_mask(crop_h, crop_w)

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    tile = np.where(shape_mask, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = _make_negative_template(shape_mask)

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
