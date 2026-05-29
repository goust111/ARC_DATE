# Combo variant of b0039139 (index 10)
# combo_spec: D3/0+D5/3
# Name: 8-连通计数配分量面积分隔

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant
# =====================================
# concretization_and_trace:
# - The input still has four sections split by blue (1) full rows/columns in the
#   order template / stencil / fg swatch / bg swatch.
# - The fusion changes section 2 in two linked ways. We no longer read green pixels
#   with 4-connectivity; we read 8-connected green groups. Also, the width of each
#   internal separator in the output is the pixel area of the corresponding ordered
#   8-connected group.
# - A concrete stencil example for a horizontal layout is:
#       110  110  110
#       001  000  001
#            011  001
#   with black gaps between the three clusters. Interpreting each cluster:
#   first area = 3, second area = 4, third area = 4. Under the source task, these
#   are six 4-connected mini-pieces, so the output would repeat the tile six times
#   with 1-cell separators. Under this combo, they become three 8-connected groups,
#   so the output has three copies; the first separator is 3 cells wide and the
#   second separator is 4 cells wide. The last group's area is still visible in the
#   stencil and still contributes to the repeat count, but there is no separator
#   after the final copy because only internal gaps are drawn.
#
# complexity_calibration:
# - The generator uses exactly two area families, 3 and 4, built from diagonal-only
#   joins. This keeps the rule easy to observe: humans can count green pixels inside
#   each diagonal cluster and compare them against separator widths.
# - Freedom is intentionally limited. Every sample contains three ordered 8-connected
#   groups, and the first two visible separators are always widths 3 and 4 in some
#   order. That repeated support makes the fused rule inferable from few examples.
# - The cluster shapes differ only in height, not width, so input size stays compact
#   while the relevant area change remains visually legible.
#
# implicit_assumption_audit_for_main:
# - `_split_sections` assumes only the true separators are full blue rows/columns.
#   The generator never uses color 1 inside any section.
# - `template_mask = template != 0` assumes section 1 contains only black background
#   plus one non-black template shape. The generator preserves that.
# - `bbox(template_mask)` assumes the template is non-empty. `random_sprite` provides
#   a non-empty connected shape, so the bbox always exists in generated samples.
# - `_ordered_components` assumes the intended stencil objects are green (3) and are
#   spatially ordered along the repeat axis without ties. The generator places each
#   cluster in a left-to-right or top-to-bottom sequence with fixed blank gaps.
# - `_repeat_with_component_gaps` assumes the gap widths come from the first N-1
#   ordered 8-connected components. This is the only non-symmetric part of the rule;
#   the final component still affects the copy count but cannot create an external
#   trailing gap because the output only contains separators between adjacent copies.
#
# placement_legality:
# - Each stencil cluster is a small boolean mask placed wholly inside the stencil.
# - Consecutive clusters are separated by two blank columns/rows along the layout
#   axis, so no accidental 8-neighbor merge can occur across different components.
# - Cluster heights/widths are at most 3, while the shared axis is `crop_dim + 2`
#   with `crop_dim` in 2..3, so every placement fits comfortably.
# - The generator rejects any sample whose final input or output would exceed a
#   30x30 grid, keeping the construction inside ordinary ARC size limits.
#
# observability_and_cleanliness:
# - The stencil is clean: black background plus three green diagonal clusters with
#   obvious empty gaps between them. Nothing overlaps, and each cluster can be
#   counted both by group count and by pixel area with the naked eye.
# - The output is equally clean because only one template crop is repeated, and the
#   only varying feature is the thickness of the background-colored separator bands.
# - Because the visible separators are always one width-3 band and one width-4 band,
#   the area-to-gap correspondence has repeated support instead of appearing only as
#   a rare accidental size difference.
#
# new_solver_assumptions_after_change:
# - `main()` now depends on the ordered 8-connected green components being the true
#   control signal. The generator guarantees this with dedicated color 3 clusters.
# - `main()` also assumes diagonal touching is meaningful rather than accidental. The
#   generator constructs every component from explicit diagonal-only joins, so 8- and
#   4-connectivity produce different readings in every sample.


TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1

REPEAT_COUNT = 3
CROP_SIZE_RANGE = (2, 3)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = list(range(2, 10))
SWATCH_DIM_RANGE = (3, 3)
STENCIL_GAP = 2
GEN_RETRIES = 300

BASE_CLUSTER_BY_AREA = {
    3: np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    4: np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == SEPARATOR_COLOR))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        return "vertical", [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == SEPARATOR_COLOR))]
    cuts = [-1] + sep_cols + [w]
    return "horizontal", [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _ordered_components(mask: np.ndarray, horizontal: bool) -> list[np.ndarray]:
    comps = components_mask(mask, connectivity=8)
    axis = 1 if horizontal else 0
    return sorted(comps, key=lambda coords: int(np.min(coords[:, axis])))


def _repeat_with_component_gaps(
    tile: np.ndarray, component_areas: list[int], bg_color: int, horizontal: bool
) -> np.ndarray:
    if len(component_areas) <= 1:
        return tile

    gap_widths = component_areas[:-1]
    pieces = [tile]
    for width in gap_widths:
        if horizontal:
            gap = np.full((tile.shape[0], width), bg_color, dtype=int)
            pieces.extend([gap, tile])
        else:
            gap = np.full((width, tile.shape[1]), bg_color, dtype=int)
            pieces.extend([gap, tile])

    return np.hstack(pieces) if horizontal else np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop and recolor a template, then space its repeats using 8-connected stencil areas.

    Input structure:
    - Blue (1) full rows or columns split the input into four consecutive sections.
    - Section 1 is a black template holding one non-black sprite; section 2 is a green stencil on black.
    - Sections 3 and 4 are solid color swatches that supply the output foreground and background.

    Rule:
    - Trim section 1 to the bbox of its non-black pixels and recolor that crop with the two swatch colors.
    - In section 2, find the green (3) components using 8-connectivity and order them along the repeat axis.
    - Make one copy of the recolored crop per ordered 8-connected component; between copy i and copy i+1,
      insert a background-colored band whose thickness equals the area of component i.

    Key insight:
    - Diagonal joins in the stencil matter twice: they collapse several apparent mini-pieces into one repeat,
      and the total number of pixels in each merged group becomes the width of the next separator band.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    horizontal = orientation == "horizontal"
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile = np.where(tile_mask, foreground, background).astype(int)

    comps = _ordered_components(repetition_pattern == STENCIL_COLOR, horizontal)
    if not comps:
        return tile
    component_areas = [int(len(comp)) for comp in comps]
    return _repeat_with_component_gaps(tile, component_areas, background, horizontal)


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


def _make_cluster(area: int, horizontal: bool) -> np.ndarray:
    cluster = BASE_CLUSTER_BY_AREA[area].copy()
    if random.choice([True, False]):
        cluster = np.fliplr(cluster)
    if random.choice([True, False]):
        cluster = np.flipud(cluster)
    if not horizontal:
        cluster = cluster.T
    return np.ascontiguousarray(cluster)


def _sample_component_areas() -> list[int]:
    visible = [3, 4]
    random.shuffle(visible)
    tail = random.choice([3, 4])
    return visible + [tail]


def _build_stencil(component_areas: list[int], shared_dim: int, horizontal: bool) -> np.ndarray:
    clusters = [_make_cluster(area, horizontal) for area in component_areas]
    margins = 2
    varying_extent = margins + STENCIL_GAP * (len(clusters) - 1)
    varying_extent += sum(cluster.shape[1] if horizontal else cluster.shape[0] for cluster in clusters)

    h, w = (shared_dim, varying_extent) if horizontal else (varying_extent, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    cursor = 1
    for idx, cluster in enumerate(clusters):
        ch, cw = cluster.shape
        if horizontal:
            r0 = random.randint(0, shared_dim - ch)
            view = grid[r0 : r0 + ch, cursor : cursor + cw]
            view[cluster] = STENCIL_COLOR
            cursor += cw
        else:
            c0 = random.randint(0, shared_dim - cw)
            view = grid[cursor : cursor + ch, c0 : c0 + cw]
            view[cluster] = STENCIL_COLOR
            cursor += ch
        if idx < len(clusters) - 1:
            cursor += STENCIL_GAP

    return grid


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])

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
    component_areas = _sample_component_areas()
    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    output = _repeat_with_component_gaps(tile, component_areas, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(component_areas, shared_dim, horizontal)

    ordered = _ordered_components(stencil == STENCIL_COLOR, horizontal)
    ordered_areas = [int(len(comp)) for comp in ordered]
    count4 = len(components_mask(stencil == STENCIL_COLOR, connectivity=4))
    count8 = len(ordered)
    if ordered_areas != component_areas or count8 != REPEAT_COUNT or count4 <= count8:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, horizontal)

    if max(inp.shape) > 30 or max(output.shape) > 30:
        return None

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
