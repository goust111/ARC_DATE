# Variant of b0039139
# Dimension: D4 — 副本平铺方向 (grid_shape)
# Change: 平铺方向由分隔线方向决定——水平分隔线(行)则垂直平铺，垂直分隔线(列)则水平平铺... → 紧凑打包：交替正反排列节省空间

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for this D4 variant
# -----------------------------------
# a) Concrete behavior:
#    - Input stays as the original four sections: template, count stencil, fg swatch, bg swatch.
#    - The separator orientation still selects the output's main axis:
#      column separators -> output grows left/right; row separators -> output grows up/down.
#    - The changed step is only the copy layout. Original behavior was:
#      tile + 1-cell bg gap + tile + ...
#      The variant builds copies in alternating orientations:
#      original, perpendicular-axis mirror, original, perpendicular-axis mirror, ...
#      Then each new copy is shifted toward the previous output as far as possible without any
#      foreground-foreground collision, so copies interlock and the total extent shrinks.
#      Concretely: left/right chains alternate between the original crop and its top-bottom flip;
#      up/down chains alternate between the original crop and its left-right flip.
#    - Small trace, horizontal case, width 4, N=3:
#      original: 4 + 1 + 4 + 1 + 4 = 14 columns.
#      variant: 4 columns for the first tile, then mirrored copies packed with overlaps such
#      as [1, 2], so the width becomes 4 + (4-1) + (4-2) = 9.
#      The visual result is still a 1D chain, but now with alternating left/right silhouettes.
# b) Complexity calibration:
#    - Only one transform family is used per sample: a single mirror direction determined by the
#      packing axis. No rotations or mixed transform tables are introduced.
#    - The stencil still encodes only an integer count via connected components, so the user
#      can isolate the new idea from familiar parts of the puzzle.
# c) Review of original assumptions that break under the variant:
#    - The original `tile` construction still works unchanged; color recovery and cropping do
#      not rely on the old spacing rule.
#    - The original `n_repeats` counting still works unchanged.
#    - The old branch
#        `sep = bg row/col; pieces = [tile, sep, tile, ...]`
#      assumed every copy occupied a disjoint rectangle with a fixed 1-cell spacer. That is
#      exactly the obsolete assumption, so it is replaced by collision-checked packing.
# d) Placement legality:
#    - Packing always keeps copies in-bounds because the output canvas is expanded only when a
#      copy would extend past the current right/bottom edge.
#    - Copy overlap is allowed only where at least one of the two cells is background, so no
#      foreground information is destroyed.
#    - Generator-side validity checks guarantee at least one positive overlap in the chosen axis.
# e) Visual observability:
#    - The generator rejects shapes symmetric under the chosen mirror, otherwise the alternation
#      would be invisible.
#    - It also rejects cases where packing degenerates to zero overlap, otherwise the result would
#      look like a plain repeated strip.
# f) Visual cleanliness:
#    - Output still uses only the original fg/bg pair, with no extra markers or layered regions.
#      The silhouette alone carries the new rule, so the image remains easy to parse.
# g) Color inference:
#    - Colors are unchanged from the base task: two swatches directly specify output fg/bg.
#      No new color-conditioned behavior is added.
# h) Variant verification target:
#    - A generated sample is considered non-degenerate when the packed layout differs from the
#      old simple repeat and every adjacent placement has overlap >= 1.
# i) New solver assumptions and generator guarantees:
#    - Solver assumes all copies share the same row alignment in horizontal outputs or the same
#      column alignment in vertical outputs; generator always constructs outputs that way.
#    - Solver assumes the intended compact placement is the leftmost/topmost collision-free one
#      for each successive mirrored copy; generator uses the same construction, so oracle checks
#      stay exact.


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


def _pack_alternating_mask(
    base_mask: np.ndarray, n_repeats: int, horizontal: bool
) -> tuple[np.ndarray, list[int]]:
    """Pack original / mirrored / original / ... copies along one axis as tightly as possible."""
    if n_repeats <= 1:
        return base_mask.astype(bool), []

    base = base_mask.astype(bool)
    packed = base
    overlaps: list[int] = []

    if horizontal:
        pieces = [base if i % 2 == 0 else np.flipud(base) for i in range(n_repeats)]
        packed = pieces[0].copy()
        for piece in pieces[1:]:
            cur_h, cur_w = packed.shape
            piece_w = piece.shape[1]
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
                raise RuntimeError("Failed to pack mirrored copy without collision.")
        return packed, overlaps

    pieces = [base if i % 2 == 0 else np.fliplr(base) for i in range(n_repeats)]
    packed = pieces[0].copy()
    for piece in pieces[1:]:
        cur_h, cur_w = packed.shape
        piece_h = piece.shape[0]
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
            raise RuntimeError("Failed to pack mirrored copy without collision.")
    return packed, overlaps


def _simple_repeat_mask(base_mask: np.ndarray, n_repeats: int, horizontal: bool) -> np.ndarray:
    """Baseline from the original task: one-cell gaps between axis-aligned copies."""
    if n_repeats <= 1:
        return base_mask.astype(bool)

    tile = base_mask.astype(bool)
    if horizontal:
        sep = np.zeros((tile.shape[0], 1), dtype=bool)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.zeros((1, tile.shape[1]), dtype=bool)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _is_axis_asymmetric(mask: np.ndarray, horizontal: bool) -> bool:
    mirrored = np.flipud(mask) if horizontal else np.fliplr(mask)
    return not np.array_equal(mask, mirrored)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and pack alternating mirrored copies according to a component count.

    Input structure:
    - Solid blue separator rows or columns divide the input into four sections.
    - Section 1 is a black-background template, section 2 is a component-count stencil.
    - Sections 3 and 4 are solid swatches that specify the output foreground and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black cells and recolor that crop with the two swatches.
    - Count 4-connected non-black components in section 2 to get the number of output copies.
    - Along the axis implied by the separator direction, place copies as original / mirrored / original / ...
      where horizontal chains use top-bottom flips and vertical chains use left-right flips.
      Shift each new copy inward as far as possible while preventing any foreground overlap.

    Key insight: The stencil still provides only a repeat count; the new behavior is that the output
    uses compact collision-free packing of alternating flips instead of fixed one-cell separators.
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

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    packed_mask, _overlaps = _pack_alternating_mask(
        tile_mask, n_repeats, horizontal=(orientation == "horizontal")
    )
    return np.where(packed_mask, foreground, background).astype(int)


# --------------- generator (D4 packed grid variant) ---------------

# --- CFG: base task parameters kept where possible ---
REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (2, 5)
DENSITY_RANGE = (0.55, 0.85)
COLOR_POOL = list(range(2, 10))
TEMPLATE_COLOR = 4
STENCIL_COLOR = 3
SEPARATOR_COLOR = 1
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
GEN_RETRIES = 400

# Variant-specific guards for observability.
MIN_PACKED_OVERLAP = 1


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


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
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


def _generate_valid_shape(horizontal: bool, n_repeats: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    for _ in range(80):
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
        if not overlaps or min(overlaps) < MIN_PACKED_OVERLAP:
            continue

        simple_mask = _simple_repeat_mask(mask, n_repeats, horizontal)
        if np.array_equal(packed_mask, simple_mask):
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
