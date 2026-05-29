# Variant of b0039139
# Dimension: D12 — 模板中特殊像素的处理 (meta)
# Change: 模板区段中只有两种像素值：色4(模板形状)和色0(背景)，不存在其他颜色... → 模板包含第三种颜色作为透明标记

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D12 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The input still has four sections separated by solid color-1 lines:
#      template, stencil, foreground swatch, background swatch.
#    - The template is no longer binary. Inside its 1-cell black border, the cropped
#      pattern contains exactly three semantic pixel types:
#        * 4: paint with the foreground swatch color.
#        * 0: paint with the background swatch color.
#        * 1: transparent marker; keep the output cell at the default color 0.
#    - Small trace:
#        template crop [[4, 1, 0],
#                       [0, 4, 4]]
#        fg=8, bg=6
#        old logic (`template != 0`) would produce [[8, 8, 6],
#                                                   [6, 8, 8]]
#        new logic must produce               [[8, 0, 6],
#                                             [6, 8, 8]]
#      The repeat-count stencil and tiling axis are unchanged.
# b) Complexity calibration
#    - The variant is only meaningful if all three template states appear in the
#      cropped area, so the generator rejects degenerate crops without any internal
#      0 cells or without any transparent 1 cells.
#    - Foreground/background swatches remain a simple two-color cue; the only new
#      inference burden is distinguishing "painted" 4 from "transparent" 1.
# c) Hidden assumptions in the original solver
#    - `template_mask = template != 0` used to be harmless because every nonzero
#      template cell meant "foreground". With transparent 1s, that assumption fails.
#    - `tile = np.where(tile_mask, foreground, background)` then incorrectly paints
#      transparent cells with the foreground color.
#    - The bbox step can stay based on `template != 0`, but now that means
#      "all marked cells" rather than "all foreground cells". The generator ensures
#      that behavior is intentional by allowing 1s to participate in the crop.
# d) Placement legality
#    - The template still sits inside a 1-cell black border and never overlaps other
#      sections because sections are joined only by separator rows/cols.
#    - The stencil placement rules are unchanged: each 2-cell blob stays separated by
#      at least one black cell, so the solver's component count remains stable.
# e) Visual observability
#    - Every generated crop contains at least one 1 cell and at least one 0 cell
#      inside the cropped bbox. This makes the difference between "background inside
#      the crop" and "transparent marker" visible in the output: bg-color vs black.
# f) Visual cleanliness
#    - Transparency is encoded only inside the template section; no extra legends or
#      overlapping marks are added elsewhere, so the four-section layout remains easy
#      to parse.
# g) Color-change inferability
#    - Output colors are still mostly controlled by the two swatches. The only fixed
#      color is transparent output = 0, which is supported by repeated black holes
#      across the tiled copies.
# h) Quantitative anti-degeneration target
#    - On generated samples, transparent pixel count in the cropped template must be
#      >= 1 and the new solver output must differ from the original binary-template
#      solver output. The validation script checks this distribution directly.
# i) New solver assumptions to audit
#    - The solver assumes template colors are limited to {0,1,4}; the generator
#      enforces that.
#    - The solver assumes the cropped template contains at least one nonzero cell;
#      the generator enforces that by deriving the template from a nonempty sprite.
#    - The solver assumes swatches are solid-color rectangles and the stencil count is
#      the repeat count; those original guarantees are preserved unchanged.


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


TEMPLATE_COLOR = 4
TRANSPARENT_MARKER = 1
OUTPUT_DEFAULT_COLOR = 0


def _render_cropped_template(template: np.ndarray, foreground: int, background: int) -> np.ndarray:
    template_bbox = bbox(template != 0)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    cropped = crop_bbox(template, template_bbox)
    tile = np.full(cropped.shape, background, dtype=int)
    tile[cropped == TEMPLATE_COLOR] = foreground
    tile[cropped == TRANSPARENT_MARKER] = OUTPUT_DEFAULT_COLOR
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a three-state template, recolor it, and repeat it by a component-count stencil.

    Input structure:
    - Solid blue (1) separator rows or columns divide the grid into four sections.
    - Section 1 is a black-background template that may contain yellow shape pixels (4)
      and blue transparent-marker pixels (1); section 2 is the repeat-count stencil.
    - Sections 3 and 4 are solid color swatches giving the output foreground and background.

    Rule:
    - Crop section 1 to the bbox of all marked template cells (both 4 and 1).
    - Inside that crop, recolor 4 -> foreground swatch, 0 -> background swatch, and
      leave transparent-marker 1 cells at the default output color 0.
    - Count 4-connected non-black components in section 2 and repeat the recolored crop
      that many times, with one background-colored row/column between copies.

    Key insight: The stencil still encodes only the repetition count, but the template
    is no longer binary: some nonzero cells define transparent holes instead of painted mass.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile = _render_cropped_template(template, foreground, background)
    if tile.size == 0:
        return tile

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
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of support bbox
COLOR_POOL = list(range(2, 10))      # fg/bg color choices (2-9 inclusive)
STENCIL_COLOR = 3                    # blob color in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200             # per-blob placement attempts in stencil
TEMPLATE_ATTEMPTS = 120              # retries for making a visible 3-state template
GEN_RETRIES = 300                    # outer generate_input retry limit
TRANSPARENT_RATIO_RANGE = (0.2, 0.45)


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


def _make_template_core(crop_h: int, crop_w: int) -> np.ndarray | None:
    for _ in range(TEMPLATE_ATTEMPTS):
        density = random.uniform(*DENSITY_RANGE)
        support = random_sprite(
            crop_h,
            crop_w,
            density=density,
            color_palette=[TEMPLATE_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        support_mask = support == TEMPLATE_COLOR
        if support_mask.sum() < 2:
            continue
        if not np.any(~support_mask):
            continue

        coords = [tuple(x) for x in np.argwhere(support_mask)]
        random.shuffle(coords)
        max_transparent = min(len(coords) - 1, max(1, int(round(len(coords) * TRANSPARENT_RATIO_RANGE[1]))))
        min_transparent = max(1, int(np.ceil(len(coords) * TRANSPARENT_RATIO_RANGE[0])))
        min_transparent = min(min_transparent, max_transparent)
        n_transparent = random.randint(min_transparent, max_transparent)

        template_core = np.zeros((crop_h, crop_w), dtype=int)
        template_core[support_mask] = TEMPLATE_COLOR
        for r, c in coords[:n_transparent]:
            template_core[r, c] = TRANSPARENT_MARKER

        if not np.any(template_core == TEMPLATE_COLOR):
            continue
        if not np.any(template_core == TRANSPARENT_MARKER):
            continue
        if not np.any(template_core == 0):
            continue
        return template_core

    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    template_core = _make_template_core(crop_h, crop_w)
    if template_core is None:
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    tile = _render_cropped_template(template_core, fg_color, bg_color)
    if tile.size == 0 or not np.any(tile == OUTPUT_DEFAULT_COLOR):
        return None
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1:crop_h + 1, 1:crop_w + 1] = template_core

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
