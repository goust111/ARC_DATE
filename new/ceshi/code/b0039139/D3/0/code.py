# Variant of b0039139
# Dimension: D3 — 连通分量计数方式 (src)
# Change: 计算区段2中色3像素的4-连通分量数量，该数量决定模板副本的重复次数... → 使用8-连通而非4-连通计数

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D3 variant
# ==================================
# concretization_and_trace:
# - Layout stays identical to the source task: four sections split by blue (1)
#   full rows or columns, in the order template / stencil / fg swatch / bg swatch.
# - The only semantic change is in section 2. Instead of counting orthogonally
#   separated green pieces with 4-connectivity, we now count 8-connected green
#   groups, so diagonal corner-touching pieces merge.
# - To make that visible, each intended repeat is encoded as one diagonal cluster
#   made from two separate 2-cell mini-blobs, e.g.
#       330
#       003
#       003
#   Here the left horizontal domino and the right vertical domino touch only at a
#   corner. Old logic: 2 components. New logic: 1 component.
# - Small trace: if section 2 contains three such diagonal clusters separated from
#   each other by black gaps, the original solver would produce 6 copies while the
#   variant must produce 3 copies. All other steps (crop template, recolor with
#   swatches, tile with bg separators along the layout axis) are unchanged.
#
# complexity_calibration:
# - The variant needs a stable visual cue, not random accidental diagonal contacts.
# - Generator therefore uses a small family of repeated diagonal-cluster motifs.
#   This keeps the puzzle readable: humans can see "paired green bits that belong
#   together by diagonal touch", while 4-connected counting would clearly overcount.
# - We keep freedom limited to orientation, template shape, colors, cluster flips,
#   and cross-axis offsets. The core counting rule has repeated support in every
#   sample instead of depending on one brittle special case.
#
# implicit_assumption_audit_for_main:
# - `_split_sections` assumes no internal section contains a full blue row/column.
#   Generator preserves that by never using color 1 except separators.
# - `template_mask = template != 0` assumes template foreground is the only
#   non-black content in section 1. That remains true.
# - `bbox(template_mask)` assumes the template has at least one foreground cell.
#   `random_sprite` is used with connected non-empty sprites, so this holds.
# - Counting now depends on `repetition_pattern == 3`, not merely non-black.
#   This matches the task statement exactly and avoids hidden dependence on other
#   possible colors inside section 2.
# - Tiling logic still assumes the count is >= 1. Generator keeps repeat counts in
#   2..6, so the degenerate zero-template path remains unused but harmless.
#
# placement_legality:
# - Each diagonal cluster is placed wholly inside the stencil bounds.
# - Clusters are separated by at least two empty columns/rows along the varying
#   axis, which is enough to prevent accidental 8-neighborhood merging between
#   different clusters even when their cross-axis offsets differ.
# - The cluster's shared-axis size is at most 3, so it always fits because the
#   shared dimension is `crop_dim + 2`, and `crop_dim` is at least 2.
#
# observability_and_cleanliness:
# - Every generated stencil satisfies `count4 > count8`, usually `count4 = 2*count8`.
#   So the new rule is not a silent internal change; it changes the visible repeat
#   count relative to the number of apparent mini-blobs in section 2.
# - Sections remain visually separated, the stencil uses only black and green, and
#   clusters never overlap or interleave. The puzzle stays neat and local.
#
# new_solver_assumptions_after_change:
# - `main()` now assumes the intended counted pixels are specifically color 3.
#   `generate_input()` guarantees that all stencil foreground cells use color 3.
# - `main()` also assumes diagonal-only joins are meaningful. Generator enforces
#   that by constructing section 2 from explicit diagonal clusters instead of
#   relying on random placement that might collapse back to the 4-connected case.


TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and repeat it using an 8-connected stencil count.

    Input structure:
    - Blue (1) full separator rows or columns divide the input into four sections.
    - Section 1 is a black template with a yellow shape; section 2 is a green stencil.
    - Sections 3 and 4 are solid swatches that provide the output foreground and background colors.

    Rule:
    - Crop section 1 to the bounding box of its non-black shape pixels and recolor the crop with the two swatches.
    - In section 2, count the 8-connected components formed by green (3) pixels, so diagonal corner contacts also join pieces.
    - Repeat the recolored crop that many times along the layout axis, separating neighboring copies by one row or column of the background color.

    Key insight:
    - The stencil does not count isolated orthogonal blobs; diagonal green groups in section 2 each stand for one copy, and the separator orientation tells you whether copies stack vertically or horizontally.
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

    n_repeats = len(components_mask(repetition_pattern == STENCIL_COLOR, connectivity=8))
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

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = list(range(2, 10))
SWATCH_DIM_RANGE = (3, 8)
STENCIL_GAP_RANGE = (2, 4)
GEN_RETRIES = 300


BASE_DIAGONAL_CLUSTER = np.array(
    [
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ],
    dtype=bool,
)


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with a 1-pixel bg separator."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _make_swatch(shared_dim: int, color: int, horizontal: bool) -> np.ndarray:
    varying = random.randint(*SWATCH_DIM_RANGE)
    if horizontal:
        return np.full((shared_dim, varying), color, dtype=int)
    return np.full((varying, shared_dim), color, dtype=int)


def _join_sections(sections: list[np.ndarray], shared_dim: int, horizontal: bool) -> np.ndarray:
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


def _make_diagonal_cluster(horizontal: bool) -> np.ndarray:
    cluster = BASE_DIAGONAL_CLUSTER.copy()
    if random.choice([True, False]):
        cluster = np.fliplr(cluster)
    if random.choice([True, False]):
        cluster = np.flipud(cluster)
    if not horizontal:
        cluster = cluster.T
    return np.ascontiguousarray(cluster)


def _build_stencil(n_groups: int, shared_dim: int, horizontal: bool) -> np.ndarray:
    clusters = [_make_diagonal_cluster(horizontal) for _ in range(n_groups)]
    gaps = [random.randint(*STENCIL_GAP_RANGE) for _ in range(max(0, n_groups - 1))]
    lead_margin = random.randint(1, 2)
    tail_margin = random.randint(1, 2)

    varying_extent = lead_margin + tail_margin
    varying_extent += sum(cluster.shape[1] if horizontal else cluster.shape[0] for cluster in clusters)
    varying_extent += sum(gaps)

    h, w = (shared_dim, varying_extent) if horizontal else (varying_extent, shared_dim)
    grid = np.zeros((h, w), dtype=int)

    cursor = lead_margin
    for i, cluster in enumerate(clusters):
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
        if i < len(gaps):
            cursor += gaps[i]

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
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)

    count8 = len(components_mask(stencil == STENCIL_COLOR, connectivity=8))
    count4 = len(components_mask(stencil == STENCIL_COLOR, connectivity=4))
    if count8 != n_repeats or count4 <= count8:
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
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
