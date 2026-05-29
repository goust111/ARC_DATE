# Variant of b0039139
# Dimension: D7 — 计数板中连通分量的空间排布 (src)
# Change: 区段2中的色3连通分量排布在网格内，其数量决定重复次数，但分量的具体位置和形状不影响输出... → 分量的颜色各不相同，决定副本的前景色
#
# Design analysis for this D7 variant
# a) Concrete behavior:
#    - Keep the original 4-section layout and the same template/background mechanics.
#    - Section 2 still contains isolated 2-cell blobs, but now each blob has its own
#      color. The output contains one recolored copy per blob, and that copy's
#      foreground is the blob's own color.
#    - Because the output is linear while the stencil is 2D, the blobs must also be
#      ordered. The cleanest D7-specific change is to read them in layout order:
#      left-to-right for horizontal outputs, top-to-bottom for vertical outputs
#      (with the orthogonal coordinate as a tiebreaker).
#    - Small trace:
#      original: stencil has green blobs at columns 2, 6, 10; solver counts 3 and
#      paints every copy with section-3's single color.
#      variant: the same three blobs may be colors 3, 6, 8; solver reads them as
#      [3, 6, 8] by position and paints copy1/copy2/copy3 with 3/6/8 respectively.
#      The copied shape and shared background still come from sections 1 and 4.
# b) Complexity calibration:
#    - The variant is stronger than a mere count because section 2 now controls both
#      copy count and per-copy foreground sequence.
#    - To keep the rule human-inferable, each blob color is unique and the output
#      uses the same ordering rule throughout a sample.
# c) Old-code assumptions that break here:
#    - `foreground = _solid_color(foreground_block)` assumed one foreground for all
#      copies. That is no longer valid.
#    - `n_repeats = len(components_mask(repetition_pattern != 0, ...))` used section 2
#      only as a scalar count. The variant needs per-component color extraction and a
#      stable ordering, not just the total.
#    - Building one `tile` once and repeating it unchanged would collapse all copy
#      colors to one value and silently erase the new signal.
# d) Placement legality:
#    - The generator still uses 2-cell blobs with a 1-cell black moat, so blobs never
#      merge, never overlap, and remain easy to segment by color.
#    - Shared axis sizing and separator logic stay unchanged, so all four sections
#      remain well-aligned and the solver can split them exactly as before.
# e) Visual observability:
#    - Every generated sample has at least two repetitions, and blob colors are unique.
#      The output therefore visibly alternates foreground colors across copies.
#    - Section 3 is kept as the legacy solid swatch, but it is only a partial clue:
#      it matches one blob color rather than controlling every copy, so training pairs
#      visibly contradict the original rule.
# f) Visual cleanliness:
#    - Section 2 keeps sparse, isolated 2-cell blobs on black; section 1 stays a
#      single yellow template on black; section 4 stays a solid background swatch.
#      No region mixes multiple semantics on top of each other.
# g) Color inferability:
#    - The foreground colors are copied directly, with no hidden arithmetic on color
#      ids. A human can see the same colors in section 2 and in the output copies.
# New solver assumptions and generator guarantees:
#    - The solver assumes section 2's intended output order can be read from component
#      positions. The generator guarantees this by producing isolated blobs and using
#      the same layout axis in the output.
#    - The solver assumes section 4 is a solid shared background color. The generator
#      preserves the original solid-swatch construction.
#    - The solver ignores section 3 for painting. The generator keeps it as a solid
#      legacy swatch chosen from the blob colors so the input remains tidy without
#      introducing a second, unrelated color-mapping rule.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


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


def _ordered_stencil_colors(stencil: np.ndarray, horizontal: bool) -> list[int]:
    entries: list[tuple[int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(stencil) if int(v) != 0):
        for coords in components_mask(stencil == color, connectivity=4):
            mins = coords.min(axis=0)
            r0, c0 = int(mins[0]), int(mins[1])
            entries.append((c0, r0, color) if horizontal else (r0, c0, color))
    entries.sort()
    return [color for _, _, color in entries]


def _stack_tiles(tiles: list[np.ndarray], background: int, horizontal: bool) -> np.ndarray:
    if not tiles:
        return np.empty((0, 0), dtype=int)
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
    Crop a template and replay it once per colored stencil blob, using each blob's own color.

    Input structure:
    - Solid blue (1) separator rows or columns split the grid into four sections.
    - Section 1 is a black-background template; section 2 is a black stencil with isolated
      non-black blobs whose colors differ from one another.
    - Section 3 remains a solid legacy swatch, while section 4 is a solid background swatch.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels and keep that binary shape mask.
    - Read the 4-connected blobs from section 2 in layout order, and for each blob recolor
      the cropped mask with the blob's own color over the shared background from section 4.
    - Concatenate those per-blob copies along the layout axis, separating adjacent copies by
      one row or column of the same background color.

    Key insight: section 2 is no longer only a repeat counter; its component colors, read in
    spatial order, directly specify the foreground-color sequence of the output copies.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, _legacy_foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    horizontal = orientation == "horizontal"
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)

    colors_in_order = _ordered_stencil_colors(repetition_pattern, horizontal)
    tiles = [np.where(tile_mask, color, background).astype(int) for color in colors_in_order]
    return _stack_tiles(tiles, background, horizontal)


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)           # blob count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)              # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)          # fill ratio of shape bounding box
COMPONENT_COLOR_POOL = [2, 3, 5, 6, 7, 8, 9]
TEMPLATE_COLOR = 4                    # shape drawn in yellow inside template
SEPARATOR_COLOR = 1                   # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)             # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4             # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200              # per-blob placement attempts in stencil
GEN_RETRIES = 300                     # outer generate_input retry limit


def _make_colored_tile(tile_mask: np.ndarray, fg_color: int, bg_color: int) -> np.ndarray:
    return np.where(tile_mask, fg_color, bg_color).astype(int)


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


def _build_stencil(component_colors: list[int], shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place one isolated 2-cell blob per color on a black grid."""
    n_blobs = len(component_colors)
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

        grid[r : r + br, c : c + bc] = component_colors[placed]
        placed += 1

    return grid if placed == n_blobs else None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

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

    chosen_colors = random.sample(COMPONENT_COLOR_POOL, n_repeats + 1)
    component_colors = chosen_colors[:n_repeats]
    bg_color = chosen_colors[-1]

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(component_colors, shared_dim, horizontal)
    if stencil is None:
        return None

    colors_in_order = _ordered_stencil_colors(stencil, horizontal)
    legacy_fg_color = random.choice(colors_in_order)

    tile_mask = shape != 0
    tiles = [_make_colored_tile(tile_mask, fg_color, bg_color) for fg_color in colors_in_order]
    output = _stack_tiles(tiles, bg_color, horizontal)

    fg_swatch = _make_swatch(shared_dim, legacy_fg_color, horizontal)
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
