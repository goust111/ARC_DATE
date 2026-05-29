# Combo variant of b0039139 (index 17)
# combo_spec: D3/3+D5/3
# Name: 最大分量面积决定重复加分量面积决定分隔宽度

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D3/3 + D5/3 combo variant:
# - Concrete grid change:
#   The four-section layout is unchanged: template, stencil, foreground swatch,
#   background swatch, separated by full blue rows or columns. The fusion only
#   changes what section 2 means. Instead of many equal 2-cell blobs whose count
#   alone matters, section 2 now contains an ordered chain of green 1-cell-thick
#   bars. Their left-to-right order (or top-to-bottom order for vertical layouts)
#   defines the sequence of separator widths in the output, and the largest bar
#   area also defines how many template copies are produced.
# - Small traced example:
#   Suppose a horizontal sample has stencil bar areas [2, 5, 1, 3].
#   Original task:
#     * component count = 4, so the output has 4 copies;
#     * every gap between copies is width 1.
#   Fused task:
#     * max area = 5, so the output has 5 copies;
#     * the 4 gaps between copies have widths [2, 5, 1, 3] in that same order.
#   The same components therefore play two different roles: globally, the maximum
#   decides repetition count; locally, each component area decides one separator.
# - Complexity calibration:
#   The generator only uses 3..5 repeats, so there are always at least two bars
#   and at least two visible separator bands. It also creates exactly N-1 bars for
#   N output copies, which keeps the mapping "one ordered bar -> one ordered gap"
#   human-readable rather than forcing any hidden reuse rule. One bar is always the
#   unique longest bar of length N, while the others are shorter, so the D3/3 cue
#   and the D5/3 cue are both visually active in every sample.
# - Main() assumption audit:
#   `_split_sections` still assumes full blue separator rows or columns; unchanged.
#   `bbox(template != 0)` still assumes section 1 has one non-black template on a
#   black background; unchanged.
#   `_solid_color` still assumes sections 3 and 4 are solid swatches; unchanged.
#   The old `len(components_mask(...))` assumed every component contributed the same
#   information and only the count mattered. That is false now: `main()` must keep
#   the components ordered along the tiling axis, measure each area, use the maximum
#   area as `n_repeats`, and use the ordered area list as separator widths.
#   This introduces one new assumption: the stencil has exactly `n_repeats - 1`
#   ordered components. The generator guarantees that directly.
# - Placement legality:
#   The generator draws each stencil component as a separate horizontal or vertical
#   bar with at least one black cell between neighboring bars, so 4-connectivity
#   never merges them. All bars stay inside a black frame with 1-cell margin, so
#   they do not touch section borders or go out of bounds.
# - Visual observability:
#   Humans can read area immediately because each component is a 1-cell-thick bar,
#   so area equals visible length. The longest bar explains why there are more copies
#   than components, while the unequal gap widths show that the full ordered area
#   sequence matters rather than just the maximum.
# - Visual cleanliness:
#   Section roles remain cleanly separated. The stencil contains only black plus one
#   row/column of green bars, which avoids clutter and makes ordering obvious. The
#   output contains only recolored template copies plus solid background separators,
#   so different gap widths are easy to compare.
# - Generator guarantees for the new solver:
#   `generate_input()` always builds exactly `N - 1` stencil components ordered along
#   the output axis, with one component of area `N` and all others in `1..N-1`.
#   Therefore:
#     * D3/3 is active because `max_area = N` but `component_count = N - 1`;
#     * D5/3 is active because at least one separator width is > 1, and the output
#       gaps visibly follow the per-component area sequence.


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


def _ordered_component_areas(mask: np.ndarray, orientation: str) -> list[int]:
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return []
    if orientation == "horizontal":
        comps.sort(key=lambda pts: (int(np.min(pts[:, 1])), int(np.min(pts[:, 0]))))
    else:
        comps.sort(key=lambda pts: (int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))))
    return [int(comp.shape[0]) for comp in comps]


def _tile_repeat_with_widths(
    tile: np.ndarray,
    sep_widths: list[int],
    bg_color: int,
    horizontal: bool,
) -> np.ndarray:
    if not sep_widths:
        return tile

    pieces: list[np.ndarray] = [tile]
    for width in sep_widths:
        if horizontal:
            sep = np.full((tile.shape[0], width), bg_color, dtype=int)
            pieces.extend([sep, tile])
        else:
            sep = np.full((width, tile.shape[1]), bg_color, dtype=int)
            pieces.extend([sep, tile])
    return np.hstack(pieces) if horizontal else np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the cropped template and tile it using ordered stencil areas twice over.

    Input structure:
    - Solid blue (1) separator rows or columns divide the input into four sections.
    - Section 1 is a black-background shape template; section 2 is a green stencil
      made of several separated 4-connected components arranged along the output axis.
    - Sections 3 and 4 are solid color swatches that supply the output foreground
      and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels and recolor that crop with
      the foreground/background swatches.
    - In section 2, isolate the green components, keep them in left-to-right order
      for horizontal layouts or top-to-bottom order for vertical layouts, and measure
      each component area.
    - Repeat the recolored crop a number of times equal to the largest component area.
      Between adjacent copies, insert background-colored separator bands whose widths
      are the ordered component areas.

    Key insight: the stencil components are reused in two different ways. Their maximum
    area gives one global count for how many copies to make, while the full ordered list
    of component areas gives the local widths of the separator bands between copies.
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

    ordered_areas = _ordered_component_areas(repetition_pattern == STENCIL_COLOR, orientation)
    n_repeats = max(ordered_areas, default=1)
    if n_repeats <= 1:
        return tile

    needed = n_repeats - 1
    if len(ordered_areas) >= needed:
        sep_widths = ordered_areas[:needed]
    else:
        fill = ordered_areas[-1] if ordered_areas else 1
        sep_widths = ordered_areas + [fill] * (needed - len(ordered_areas))

    return _tile_repeat_with_widths(tile, sep_widths, background, orientation == "horizontal")


# --------------- generator (combo D3/3 + D5/3) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (3, 5)          # largest component area N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # fg/bg color choices (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
STENCIL_COLOR = 3                    # green components in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_COMPONENT_GAP_RANGE = (1, 2)
STENCIL_TRAILING_EXTRA = 4
GEN_RETRIES = 300


def _tile_repeat(tile: np.ndarray, sep_widths: list[int], bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` with variable-width separator bands."""
    return _tile_repeat_with_widths(tile, sep_widths, bg_color, horizontal)


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


def _choose_separator_widths(n_repeats: int) -> list[int]:
    """
    Choose ordered component areas.

    There are exactly n_repeats - 1 components, one of which has area n_repeats.
    The resulting list simultaneously serves as:
    - ordered stencil component areas;
    - ordered output separator widths.
    """
    widths = [n_repeats]
    for _ in range(n_repeats - 2):
        widths.append(random.randint(1, n_repeats - 1))
    random.shuffle(widths)
    return widths


def _build_stencil(areas: list[int], shared_dim: int, horizontal: bool) -> np.ndarray:
    """Build an ordered chain of 1-cell-thick bars whose lengths equal component areas."""
    gaps = [random.randint(*STENCIL_COMPONENT_GAP_RANGE) for _ in range(max(0, len(areas) - 1))]
    trailing = random.randint(0, STENCIL_TRAILING_EXTRA)
    varying_dim = 2 + sum(areas) + sum(gaps) + trailing

    if horizontal:
        grid = np.zeros((shared_dim, varying_dim), dtype=int)
        r = max(1, min(shared_dim - 2, shared_dim // 2))
        c = 1
        for i, area in enumerate(areas):
            grid[r, c : c + area] = STENCIL_COLOR
            c += area
            if i < len(gaps):
                c += gaps[i]
        return grid

    grid = np.zeros((varying_dim, shared_dim), dtype=int)
    c = max(1, min(shared_dim - 2, shared_dim // 2))
    r = 1
    for i, area in enumerate(areas):
        grid[r : r + area, c] = STENCIL_COLOR
        r += area
        if i < len(gaps):
            r += gaps[i]
    return grid


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

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)

    sep_widths = _choose_separator_widths(n_repeats)
    output = _tile_repeat(tile, sep_widths, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(sep_widths, shared_dim, horizontal)

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
