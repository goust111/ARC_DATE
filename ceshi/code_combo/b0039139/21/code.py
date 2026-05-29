# Combo variant of b0039139 (index 21)
# combo_spec: D3/0+D4/3+D5/0
# Name: 8-连通计数配无间隔镜像交替平铺

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant
# =====================================
# concretization_and_trace:
# - The four-section layout remains the same as the source task: template / stencil /
#   fg swatch / bg swatch, split by full blue rows or full blue columns.
# - Section 1 still gives a single yellow shape on black that must be cropped to its
#   tight bbox and then recolored by the two swatches.
# - Two later stages change jointly:
#   1) Section 2 is now counted with 8-connectivity on green pixels, so diagonal-only
#      corner contacts merge.
#   2) The output copies are not separated by a bg stripe. Instead they form a 1D
#      chain of original / mirrored / original / mirrored ... copies, each pushed as
#      far inward as possible without foreground collisions.
# - Small horizontal trace with a 3x3 crop:
#       mask =
#       110
#       010
#       011
#   Suppose the stencil contains three diagonal clusters of green pixels, each made
#   from two dominoes touching only at a corner. Old counting: 6 components; new
#   counting: 3 components. Old output: 6 plain copies with 1 bg column between them.
#   New output: 3 copies only, arranged as
#       original, top-bottom mirror, original
#   with zero separator columns and with positive overlap whenever silhouettes allow.
#   So the combo visibly changes both the copy count and the packing geometry.
#
# complexity_calibration:
# - The 8-connected cue must be obvious, so the stencil is built from a small repeated
#   family of explicit diagonal clusters rather than accidental diagonal contact.
# - The mirror rule must also be obvious, so generator rejects shapes that are symmetric
#   under the relevant mirror. That keeps the alternation visible in only 2-4 examples.
# - Freedom stays controlled: orientation, crop shape, colors, cluster flips, and
#   compact overlap amount vary, but the logical pattern is stable across samples.
#
# implicit_assumption_audit_for_main:
# - `_split_sections` assumes blue appears only as separator lines. Generator uses color 1
#   nowhere else.
# - `template_mask = template != 0` assumes section 1 contains only black background and
#   one non-black template shape. Generator preserves that exactly.
# - `bbox(template_mask)` assumes at least one template foreground cell exists. Random
#   sprite generation always yields a non-empty connected shape.
# - Component counting can no longer use `!= 0` loosely; the intended counted pixels are
#   specifically green (3). `main()` therefore counts `repetition_pattern == STENCIL_COLOR`.
# - The old tiling branch assumed a fixed separator stripe. That assumption is invalid
#   under D4/3 + D5/0, so it is replaced by collision-checked gapless packing.
# - The new packing loop assumes each later copy should still extend the chain by at least
#   one column/row. Generator guarantees that by rejecting shapes whose packed sequence
#   would hide a later copy entirely inside the earlier canvas.
#
# placement_legality:
# - Every stencil cluster is placed wholly in-bounds and separated from other clusters by
#   at least two empty cells along the varying axis, preventing accidental 8-merging
#   between intended groups.
# - Packed output placement never goes out of bounds because the canvas grows only when a
#   copy extends past the current right or bottom edge.
# - Foreground overlap is forbidden during packing; overlap is allowed only where at least
#   one side contributes background.
#
# observability_and_inference:
# - Each accepted sample satisfies `count4 > count8`, so a solver must notice that some
#   diagonal-touching green pieces count as one.
# - Each accepted sample also makes the mirror rule visible: the crop is asymmetric under
#   the used mirror, and the packed result differs from both the original gapped repeat
#   and the plain gapless unmirrored repeat.
# - No rule depends on hidden numeric ordering of colors. Human-visible structure does the
#   work: section positions define roles, diagonal touch defines grouping, and alternating
#   silhouettes define the output chain.
#
# cleanliness_and_non_degeneracy:
# - Output uses only the fg/bg swatch colors, with no extra markers, so the result stays
#   visually clean despite packing.
# - The generator demands at least one positive overlap between adjacent packed copies.
#   That makes the D4 compactness effect visible, while D5 already removes separator bands.
#
# new_solver_assumptions_after_change:
# - `main()` assumes the intended alternating mirror direction is perpendicular to the
#   chain axis: horizontal chains use `flipud`, vertical chains use `fliplr`.
# - `generate_input()` uses the same construction, checks oracle equality, and rejects
#   cases where the combo would collapse back to a visually near-base example.


TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1

REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = list(range(2, 10))
SWATCH_DIM_RANGE = (3, 8)
STENCIL_GAP_RANGE = (2, 4)
GEN_RETRIES = 500
SHAPE_RETRIES = 120


BASE_DIAGONAL_CLUSTER = np.array(
    [
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
    ],
    dtype=bool,
)


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape

    sep_rows = [r for r in range(h) if bool(np.all(grid[r, :] == SEPARATOR_COLOR))]
    if sep_rows:
        cuts = [-1] + sep_rows + [h]
        parts = [grid[a + 1 : b, :] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
        return "vertical", parts

    sep_cols = [c for c in range(w) if bool(np.all(grid[:, c] == SEPARATOR_COLOR))]
    cuts = [-1] + sep_cols + [w]
    parts = [grid[:, a + 1 : b] for a, b in zip(cuts, cuts[1:]) if b > a + 1]
    return "horizontal", parts


def _solid_color(block: np.ndarray) -> int:
    vals, counts = np.unique(block, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _piece_sequence(base_mask: np.ndarray, n_repeats: int, horizontal: bool) -> list[np.ndarray]:
    base = base_mask.astype(bool)
    if horizontal:
        return [base if i % 2 == 0 else np.flipud(base) for i in range(n_repeats)]
    return [base if i % 2 == 0 else np.fliplr(base) for i in range(n_repeats)]


def _pack_alternating_mask(
    base_mask: np.ndarray, n_repeats: int, horizontal: bool
) -> tuple[np.ndarray, list[int]]:
    """Pack alternating mirrored copies with no forced separator stripe."""
    pieces = _piece_sequence(base_mask, n_repeats, horizontal)
    packed = pieces[0].copy()
    overlaps: list[int] = []

    for piece in pieces[1:]:
        cur_h, cur_w = packed.shape
        piece_h, piece_w = piece.shape
        if horizontal:
            start_min = max(0, cur_w - piece_w + 1)
            for start in range(start_min, cur_w + 1):
                overlap_w = max(0, min(cur_w, start + piece_w) - start)
                if overlap_w > 0 and np.any(packed[:, start : start + overlap_w] & piece[:, :overlap_w]):
                    continue

                new_w = max(cur_w, start + piece_w)
                canvas = np.zeros((cur_h, new_w), dtype=bool)
                canvas[:, :cur_w] = packed
                canvas[:, start : start + piece_w] |= piece
                packed = canvas
                overlaps.append(cur_w - start)
                break
            else:
                raise RuntimeError("Failed to pack a horizontal mirrored copy.")
        else:
            start_min = max(0, cur_h - piece_h + 1)
            for start in range(start_min, cur_h + 1):
                overlap_h = max(0, min(cur_h, start + piece_h) - start)
                if overlap_h > 0 and np.any(packed[start : start + overlap_h, :] & piece[:overlap_h, :]):
                    continue

                new_h = max(cur_h, start + piece_h)
                canvas = np.zeros((new_h, cur_w), dtype=bool)
                canvas[:cur_h, :] = packed
                canvas[start : start + piece_h, :] |= piece
                packed = canvas
                overlaps.append(cur_h - start)
                break
            else:
                raise RuntimeError("Failed to pack a vertical mirrored copy.")

    return packed, overlaps


def _gapped_plain_repeat_mask(base_mask: np.ndarray, n_repeats: int, horizontal: bool) -> np.ndarray:
    tile = base_mask.astype(bool)
    if n_repeats <= 1:
        return tile
    if horizontal:
        sep = np.zeros((tile.shape[0], 1), dtype=bool)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.zeros((1, tile.shape[1]), dtype=bool)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _gapless_plain_repeat_mask(base_mask: np.ndarray, n_repeats: int, horizontal: bool) -> np.ndarray:
    tile = base_mask.astype(bool)
    if n_repeats <= 1:
        return tile
    if horizontal:
        return np.hstack([tile.copy() for _ in range(n_repeats)])
    return np.vstack([tile.copy() for _ in range(n_repeats)])


def _is_axis_asymmetric(mask: np.ndarray, horizontal: bool) -> bool:
    mirrored = np.flipud(mask) if horizontal else np.fliplr(mask)
    return not np.array_equal(mask, mirrored)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and compactly pack alternating mirrored copies using an 8-connected stencil count.

    Input structure:
    - Full blue rows or columns divide the input into four sections.
    - Section 1 is a black template containing one yellow shape; section 2 is a green counting stencil.
    - Sections 3 and 4 are solid swatches that specify the output foreground and background colors.

    Rule:
    - Crop section 1 to the bounding box of its non-black shape and recolor that crop with the two swatches.
    - In section 2, count 8-connected green components, so diagonal corner-touching green pieces belong to one group.
    - Create that many output copies along the axis implied by the separator direction, alternating original and mirrored crops.
      Horizontal chains use top-bottom flips, vertical chains use left-right flips, and adjacent copies are packed with no separator stripe.

    Key insight:
    - The stencil still encodes only a repeat count, but the intended groups are diagonal green clusters and the output is a tight alternating mirror chain rather than a spaced simple repetition.
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
    tile_mask = crop_bbox(template_mask, template_bbox).astype(bool)

    n_repeats = len(components_mask(repetition_pattern == STENCIL_COLOR, connectivity=8))
    packed_mask, _overlaps = _pack_alternating_mask(
        tile_mask, max(1, n_repeats), horizontal=(orientation == "horizontal")
    )
    return np.where(packed_mask, foreground, background).astype(int)


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


def _generate_valid_shape(horizontal: bool, n_repeats: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
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
        mask = shape != 0
        if not _is_axis_asymmetric(mask, horizontal):
            continue

        packed_mask, overlaps = _pack_alternating_mask(mask, n_repeats, horizontal)
        if not overlaps or max(overlaps) <= 0:
            continue

        gapped_plain = _gapped_plain_repeat_mask(mask, n_repeats, horizontal)
        gapless_plain = _gapless_plain_repeat_mask(mask, n_repeats, horizontal)
        if np.array_equal(packed_mask, gapped_plain):
            continue
        if np.array_equal(packed_mask, gapless_plain):
            continue
        return shape, packed_mask, overlaps
    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape_info = _generate_valid_shape(horizontal, n_repeats)
    if shape_info is None:
        return None
    shape, packed_mask, _overlaps = shape_info

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    output = np.where(packed_mask, fg_color, bg_color).astype(int)

    crop_h, crop_w = shape.shape
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
