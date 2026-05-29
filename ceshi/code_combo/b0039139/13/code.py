# Combo variant of b0039139 (index 13)
# combo_spec: D8/0+D10/0
# Name: 正方形模板强制正方形输出

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
ROOT = next((parent for parent in THIS_FILE.parents if (parent / "solvers" / "arc_utils.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Could not locate solvers/arc_utils.py from combo variant path")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for combo D8/0 + D10/0
# ---------------------------------------
# a) Concrete behavior and trace
#    - Source task:
#      * take the template's tight non-black bbox,
#      * recolor that rectangle,
#      * repeat it N times as a 1-row/1-col-separated strip.
#    - This combo:
#      * first replace the tight rectangle with the smallest enclosing square tile,
#        implemented as "bbox crop, then extend the shorter axis with background cells";
#      * then place the repeated strip into a square canvas whose side is
#        (square_tile_side * N), keeping the strip at the top-left;
#      * because the strip still contains N-1 separator bands, it is usually longer than the
#        square side on the repeat axis, so the far end is truncated; the unused orthogonal
#        region of the square is filled with the background color.
#    - Small trace:
#      * suppose the visible shape bbox is 2x4 and N=3, horizontal layout.
#      * source tile: 2x4, source output: 2x14.
#      * combo tile: 4x4 (same 2x4 shape plus 2 background rows below).
#      * repeated strip before square forcing: 4x14.
#      * combo output canvas: 12x12, so rows 0:4 hold the strip clipped to width 12 and
#        rows 4:12 are pure background.
#    - The fusion is real rather than additive-only: D8 changes the tile side that D10 uses,
#      and D10 then turns that square tile into a larger square canvas with both clipping and fill.
#
# b) Complexity calibration
#    - Only two new geometric ideas are introduced: square crop and square output.
#    - Both are anchored the same way (top-left preserved, extra space goes toward the lower/right
#      side), so the learner sees one consistent convention rather than many branches.
#    - We deliberately keep the repeat count range and stencil language from the source task so
#      the new difficulty comes from geometry, not from additional symbolic cases.
#
# c) Hidden assumption review against the original main()
#    - Original assumption: bbox dimensions directly become output tile dimensions.
#      Broken by D8; tile height and width now both equal max(bbox_h, bbox_w).
#    - Original assumption: output cross-axis equals the crop cross-axis.
#      Broken by D10; both output axes now equal square_tile_side * N.
#    - Original assumption: repeating the strip never overflows.
#      Broken by D10; overflow is now expected and must be clipped on the repeat axis.
#    - Original assumption: N<=1 can just return the tile.
#      The solver still supports it, but the generator keeps N>=2 so square forcing is visible.
#
# d) Placement legality
#    - The template section still uses a 1-cell black border around the non-square source bbox,
#      so no object overlap is introduced in the input.
#    - The stencil keeps the original separated 2-cell blobs, so component counting remains valid.
#    - In the output, the repeated strip is pasted onto a preallocated square canvas and clipped
#      by slicing, so there is no out-of-bounds write.
#
# e) Visual observability
#    - The generator forbids square source bboxes, so the square-crop rule is visible.
#    - N is always at least 2, so the square-output rule is also visible: the result is never the
#      original thin strip, and the far edge of the strip is clipped while the orthogonal side
#      shows a large uniform background band.
#    - No rule depends on hidden properties such as numeric color order.
#
# f) Visual cleanliness
#    - The four-section input layout is unchanged, so the clue structure stays readable.
#    - The output contains only three roles: recolored tile content, separator slivers, and the
#      square canvas fill, all using the same fg/bg pair as the source task.
#    - Clipping happens only at the far right or far bottom edge, never in the middle.
#
# g) Change inferability
#    - There is no new color remapping beyond the original swatches.
#    - The new pattern is structural and repeated across samples:
#      "rectangle template becomes square tile" and "strip becomes square canvas".
#
# h) New main() assumptions after the fusion
#    - _square_crop_mask() assumes at least one non-black template pixel exists; generate_input()
#      always uses random_sprite(), which returns a connected non-empty shape.
#    - _force_square_output() assumes the tile is square because its side length determines both
#      output axes; main() enforces that by always calling _square_crop_mask() first.
#    - The clipping convention assumes the repeated strip is anchored at the top-left of the
#      square canvas; generate_input() uses the same oracle, so all training examples match it.


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


def _split_sections(grid: np.ndarray) -> tuple[str, list[np.ndarray]]:
    h, w = grid.shape
    sep = SEPARATOR_COLOR

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


def _square_crop_mask(template_mask: np.ndarray) -> np.ndarray:
    """Crop to the tight bbox, then extend the shorter axis with False cells to make a square."""
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=bool)

    rect = crop_bbox(template_mask, template_bbox).astype(bool)
    h, w = rect.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=bool)
    square[:h, :w] = rect
    return square


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    if n <= 1:
        return tile.copy()
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def _force_square_output(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Place the repeated strip into a square canvas of side tile_side * n, clipping overflow."""
    side = int(tile.shape[0]) * max(int(n), 1)
    out = np.full((side, side), bg_color, dtype=int)
    strip = _tile_repeat(tile, n, bg_color, horizontal)
    hh = min(side, strip.shape[0])
    ww = min(side, strip.shape[1])
    out[:hh, :ww] = strip[:hh, :ww]
    return out


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a template after promoting its tight crop to a square tile, then place the repeated strip into a forced square canvas.

    Input structure:
    - Blue separator rows or columns split the grid into four sections.
    - Section 1 is a black template carrying one yellow shape; section 2 is the component-count stencil.
    - Sections 3 and 4 are solid swatches that supply the output foreground and background colors.

    Rule:
    - Count the 4-connected non-black components in section 2 to obtain the repetition count.
    - Crop the template to the tight non-black bbox, but then enlarge that crop to the smallest square by extending only the shorter axis with background cells.
    - Recolor the resulting square mask with the foreground/background swatches.
    - Repeat that square tile along the separator-controlled orientation with 1-cell background separators, then place the strip at the top-left of a square output whose side length is tile_side times the repetition count; clip any overflow and leave the rest as background.

    Key insight:
    - The stencil still says how many copies to make, but the template no longer yields a rectangle: it yields a square tile, and that same square side also determines the final square canvas that can both truncate the strip and leave a padded empty zone.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)

    tile_mask = _square_crop_mask(template != 0)
    if tile_mask.size == 0:
        return np.empty((0, 0), dtype=int)

    tile = np.where(tile_mask, foreground, background).astype(int)
    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    return _force_square_output(tile, n_repeats, background, orientation == "horizontal")


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    if crop_h == crop_w:
        return None

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

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    tile_mask = _square_crop_mask(template != 0)
    tile = np.where(tile_mask, fg_color, bg_color).astype(int)
    output = _force_square_output(tile, n_repeats, bg_color, horizontal)

    shared_dim = template.shape[0] if horizontal else template.shape[1]
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
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
