# Combo variant of b0039139 (index 9)
# combo_spec: D4/3+D5/0
# Name: 紧凑镜像交替拼接无间隔

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant D4/3 + D5/0
# --------------------------------------------------
# a) Concrete behavior:
#    - The input keeps the original four-section format: template, component-count stencil,
#      foreground swatch, background swatch, split by full blue separator rows or columns.
#    - The stable part of the seed task is unchanged:
#      1. crop the template to the bbox of its non-black shape,
#      2. recolor that crop with the two swatches,
#      3. count 4-connected stencil components to get N.
#    - The changed part is the copy layout. The fused rule is not the old
#      "tile + bg gap + tile + ...", and it is also not the overlap-based packed reference.
#      Here the chain is built by direct concatenation with no spacer cells at all:
#      original, mirror, original, mirror, ...
#    - The mirror direction is chosen so adjacent copies face each other across the seam:
#      horizontal output chains use top-bottom flips; vertical output chains use left-right
#      flips. This keeps every copy the same size, so D5/0 can force exact edge-to-edge
#      stitching with no empty gutter.
#    - Small horizontal trace with a 3x2 crop A and N=3:
#      original seed: [A][bg][A][bg][A], width = 3 * 2 + 2.
#      fused combo:   [A][flipud(A)][A], width = 3 * 2.
#      The seam is now a mirror seam rather than a background separator seam.
# b) Complexity calibration:
#    - Only one binary choice is introduced per sample: whether the input separators imply a
#      horizontal or vertical chain. Mirror direction then follows deterministically.
#    - No rotations, no mixed transform tables, and no hidden color-dependent cases are added.
#      The only new visual burden is recognizing alternating mirrored copies with zero spacing.
# c) Review of original assumptions that break:
#    - Cropping and swatch decoding still hold exactly; those lines do not depend on spacing.
#    - Component counting still holds exactly; the stencil still encodes only an integer.
#    - The obsolete assumption is the original repeat branch that inserted a 1-cell background
#      separator between copies. D5/0 invalidates the spacer, and D4/3 invalidates identical
#      copy orientation, so that branch must be replaced wholesale.
#    - The overlap-search helper from the standalone D4/3 reference is intentionally not reused:
#      this combo's text says copies are directly stitched and edge pixels align, so the layout
#      must be exact concatenation rather than collision-based overlap compression.
# d) Placement legality:
#    - Generator outputs are built from same-sized crops concatenated along one axis, so copies
#      cannot overlap and cannot go out of bounds.
#    - The input stencil still places 2-cell blobs with a one-cell black moat, so component
#      counts remain unambiguous.
# e) Visual observability:
#    - The generator rejects masks symmetric under the mirror axis chosen for that sample.
#      Otherwise the mirrored copy would look identical and the D4 effect would disappear.
#    - It also rejects cases where the fused output equals a plain no-gap repetition of the
#      unmirrored crop. This guards against accidental degeneration.
#    - D5/0 remains visible because the output extent loses exactly N-1 separator cells relative
#      to the seed task.
# f) Visual cleanliness:
#    - The output uses only the swatch colors, with no extra markers.
#    - Because the template crop is tight, each tile touches the seam with at least one
#      foreground pixel on both its first and last row/column set, so the direct stitching is
#      legible rather than separated by accidental blank margins.
# g) Color inference:
#    - Colors behave exactly as in the seed task: section 3 is output foreground and section 4
#      is output background. The combo changes geometry only.
# h) Quantitative non-degeneracy target:
#    - For each generated sample, let fused = alternating mirrored no-gap chain and plain =
#      unmirrored no-gap chain. We require `np.count_nonzero(fused != plain) > 0`.
#      Separately, the axis extent difference versus the seed output is always `N - 1`.
# i) New solver assumptions and generator guarantees:
#    - Solver assumes the intended chain uses the separator-implied axis and alternates exactly
#      every other copy; generator always constructs that pattern.
#    - Solver assumes there is no inter-copy spacer at all; generator never inserts one.
#    - Solver assumes the relevant mirror is perpendicular to the chain axis; generator matches
#      that convention and enforces axis asymmetry so the rule remains human-visible.


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


def _mirror_piece(tile: np.ndarray, horizontal: bool) -> np.ndarray:
    """Mirror a tile across the chain axis so adjacent copies face each other at the seam."""
    return np.fliplr(tile) if horizontal else np.flipud(tile)


def _alternating_chain(tile: np.ndarray, n_repeats: int, horizontal: bool) -> np.ndarray:
    """Build original / mirrored / original / ... copies with no separator cells."""
    if n_repeats <= 1:
        return np.array(tile, copy=True)

    mirror = _mirror_piece(tile, horizontal)
    pieces = [tile if i % 2 == 0 else mirror for i in range(n_repeats)]
    return np.hstack(pieces) if horizontal else np.vstack(pieces)


def _simple_repeat_mask(base_mask: np.ndarray, n_repeats: int, horizontal: bool, gap: int) -> np.ndarray:
    """Baseline renderer for generator-side degeneration checks."""
    tile = base_mask.astype(bool)
    if n_repeats <= 1:
        return tile.copy()

    if horizontal:
        if gap > 0:
            sep = np.zeros((tile.shape[0], gap), dtype=bool)
            pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
            return np.hstack(pieces)
        return np.hstack([tile for _ in range(n_repeats)])

    if gap > 0:
        sep = np.zeros((gap, tile.shape[1]), dtype=bool)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.vstack(pieces)
    return np.vstack([tile for _ in range(n_repeats)])


def _is_axis_asymmetric(mask: np.ndarray, horizontal: bool) -> bool:
    return not np.array_equal(mask, _mirror_piece(mask.astype(bool), horizontal))


def _alternation_delta(mask: np.ndarray, n_repeats: int, horizontal: bool) -> int:
    fused = _alternating_chain(mask.astype(bool), n_repeats, horizontal)
    plain = _simple_repeat_mask(mask, n_repeats, horizontal, gap=0)
    return int(np.count_nonzero(fused != plain))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template and stitch alternating mirrored copies into one compact chain.

    Input structure:
    - Full blue separator rows or columns divide the input into four ordered sections.
    - Section 1 is a black-background template, section 2 is a component-count stencil.
    - Sections 3 and 4 are solid swatches that supply the output foreground and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black cells and recolor that crop with the two swatches.
    - Count 4-connected non-black components in section 2 to obtain the number of copies.
    - Along the axis implied by the separator direction, place copies as original / mirrored /
      original / mirrored / ... with no separator rows or columns between neighbors.
      Horizontal chains use left-right flips; vertical chains use top-bottom flips.

    Key insight: The stencil still encodes only how many copies to make, but the output is no
    longer separated repeats. The new signal is the combination of alternating mirror seams and
    zero-gap edge-to-edge concatenation.
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

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    return _alternating_chain(tile, n_repeats, horizontal=(orientation == "horizontal")).astype(int)


# --------------- generator (combo D4/3 + D5/0) ---------------

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
SHAPE_RETRIES = 120


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


def _generate_valid_shape(horizontal: bool, n_repeats: int) -> tuple[np.ndarray, np.ndarray] | None:
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

        alt_mask = _alternating_chain(mask.astype(bool), n_repeats, horizontal)
        if _alternation_delta(mask, n_repeats, horizontal) <= 0:
            continue

        seed_mask = _simple_repeat_mask(mask, n_repeats, horizontal, gap=1)
        if alt_mask.shape == seed_mask.shape and np.array_equal(alt_mask, seed_mask):
            continue

        return shape, alt_mask
    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape_info = _generate_valid_shape(horizontal, n_repeats)
    if shape_info is None:
        return None
    shape, alt_mask = shape_info

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    output = np.where(alt_mask, fg_color, bg_color).astype(int)

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
