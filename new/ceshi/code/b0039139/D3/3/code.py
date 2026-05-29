# Variant of b0039139
# Dimension: D3 — 连通分量计数方式 (src)
# Change: 计算区段2中色3像素的4-连通分量数量，该数量决定模板副本的重复次数... → 最大连通分量的面积作为重复次数

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D3 variant:
# - Concrete grid change:
#   The four-section layout stays the same: template, stencil, foreground swatch,
#   background swatch, separated by blue rows or columns. The only semantic change
#   is section 2. In the original task, section 2 contained several same-sized green
#   components, so the solver only needed to count how many components existed.
#   That cannot express "largest area" because identical 2-cell blobs would always
#   have the same maximum area. Therefore section 2 must now contain multiple green
#   components with different areas.
# - Small traced example:
#   If section 2 contains green components with areas [5, 2, 1], the old solver
#   would produce 3 template copies because it counts components. The new solver
#   must instead inspect each component, compare their areas, and produce 5 copies
#   because the largest component has 5 pixels.
# - Complexity calibration:
#   The generator forces at least two stencil components and forces the number of
#   components to differ from the largest area. This prevents accidental regression
#   to the old behavior on generated samples. The largest component is also unique,
#   while the smaller components are strictly smaller, so the "take the maximum"
#   cue is visually stable.
# - Main() assumption audit:
#   `_split_sections` still assumes blue full rows or columns divide the input into
#   four sections; the variant keeps that unchanged.
#   `bbox(template != 0)` still assumes the template foreground is the only
#   non-black object in section 1; unchanged.
#   `_solid_color` still assumes sections 3 and 4 are solid swatches; unchanged.
#   The original `len(components_mask(...))` relied on every stencil component being
#   equally informative and only the count mattering. That assumption is now false:
#   components can have different areas, so `main()` must iterate through components
#   and measure `len(component)` before deciding the repeat count.
# - Placement legality:
#   The new stencil uses 1-cell-thick green bars. For horizontal layouts the bars
#   are horizontal and packed left-to-right; for vertical layouts they are vertical
#   and packed top-to-bottom. Bars are separated by at least one black column/row,
#   so they never merge into a single 4-connected component and never overlap.
# - Visual observability:
#   Bars keep the component area directly readable because area = visible length.
#   This avoids hiding the new rule inside irregular polyomino geometry. With only
#   2-4 examples, a human can plausibly notice that the output matches the longest
#   green bar rather than the number of green bars.
# - Visual cleanliness:
#   Section boundaries, template crop, and swatches are unchanged, so the grid stays
#   clean. The stencil area contains only black plus green bars, with gaps between
#   bars and a 1-cell border, so components remain easy to distinguish.
# - Generator guarantees for the new solver:
#   `generate_input()` always builds at least one green component, always makes the
#   largest component area equal the intended repeat count, and always keeps the
#   component count different from that repeat count. Those constraints match the
#   new `main()` logic exactly.


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


def _largest_component_area(mask: np.ndarray) -> int:
    comps = components_mask(mask, connectivity=4)
    return max((int(comp.shape[0]) for comp in comps), default=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the cropped template and repeat it using the largest stencil component.

    Input structure:
    - Solid blue (1) separator rows or columns divide the input into four sections.
    - Section 1 is a black-background shape template; section 2 is a green stencil.
    - Sections 3 and 4 are solid color swatches that provide the output foreground
      and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black pixels and recolor that crop with
      the foreground/background swatches.
    - In section 2, isolate the green pixels, split them into 4-connected components,
      and find the component with the largest area.
    - Repeat the recolored crop that many times, using one background-colored
      separator row or column between adjacent copies.

    Key insight: section 2 no longer encodes the answer by how many green pieces it
    contains; it encodes the answer by how large the biggest green piece is, so the
    solver must measure component areas before tiling the template.
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

    n_repeats = _largest_component_area(repetition_pattern == STENCIL_COLOR)
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
REPEAT_COUNT_RANGE = (2, 6)          # largest component area N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # fg/bg color choices (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
STENCIL_COLOR = 3                    # green components in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_COMPONENT_COUNT_RANGE = (2, 4)
STENCIL_GAP_RANGE = (1, 2)
STENCIL_VARYING_EXTRA = 4
GEN_RETRIES = 300


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


def _choose_stencil_areas(n_repeats: int) -> list[int]:
    min_count, max_count = STENCIL_COMPONENT_COUNT_RANGE
    count_choices = [k for k in range(min_count, max_count + 1) if k != n_repeats]
    n_components = random.choice(count_choices)
    areas = [n_repeats]
    for _ in range(n_components - 1):
        areas.append(random.randint(1, n_repeats - 1))
    random.shuffle(areas)
    return areas


def _build_stencil(areas: list[int], shared_dim: int, horizontal: bool) -> np.ndarray:
    """
    Build a stencil whose 4-connected components are separated 1-cell-thick bars.

    Each bar area equals its length, so the largest component area is immediately
    visible and equals the intended repeat count.
    """
    gaps = [random.randint(*STENCIL_GAP_RANGE) for _ in range(max(0, len(areas) - 1))]
    trailing = random.randint(0, STENCIL_VARYING_EXTRA)

    if horizontal:
        varying_dim = 2 + sum(areas) + sum(gaps) + trailing
        grid = np.zeros((shared_dim, varying_dim), dtype=int)
        c = 1
        for i, area in enumerate(areas):
            r = random.randint(1, shared_dim - 2)
            grid[r, c : c + area] = STENCIL_COLOR
            c += area
            if i < len(gaps):
                c += gaps[i]
        return grid

    varying_dim = 2 + sum(areas) + sum(gaps) + trailing
    grid = np.zeros((varying_dim, shared_dim), dtype=int)
    r = 1
    for i, area in enumerate(areas):
        c = random.randint(1, shared_dim - 2)
        grid[r : r + area, c] = STENCIL_COLOR
        r += area
        if i < len(gaps):
            r += gaps[i]
    return grid


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    # Generate connected tile shape.
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

    # Pick distinct fg/bg colors.
    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    # Build output from the intended repeat count.
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    # Build template: shape in (crop_h+2) x (crop_w+2) black border.
    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1:crop_h + 1, 1:crop_w + 1] = shape

    # Build stencil: largest green component area equals n_repeats.
    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    areas = _choose_stencil_areas(n_repeats)
    stencil = _build_stencil(areas, shared_dim, horizontal)

    # Assemble: template | stencil | fg_swatch | bg_swatch.
    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    # Oracle check.
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
