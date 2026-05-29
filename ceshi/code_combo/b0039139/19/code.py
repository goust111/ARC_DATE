# Combo variant of b0039139 (index 19)
# combo_spec: D2/2+D4/2+D8/0
# Name: 旋转后方形裁剪再按宽高比平铺

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant
# - concretization_and_trace:
#   The input still has four blue-separated sections: template, stencil, foreground swatch,
#   background swatch. The template is first cropped to its tight non-black bbox exactly as
#   before, but the visible tile is no longer that crop. The crop is rotated 90 degrees
#   clockwise, then expanded to the smallest square that can contain it by padding only on
#   the lower or right side. Example: a 2x4 crop becomes a 4x2 rotated mask, then a 4x4
#   square with two padded columns on the right. The stencil still contributes only the copy
#   count N. The repeat axis is chosen from the final tile aspect ratio; because the final
#   tile is always square, the square tie-break is the actual control and is fixed to
#   horizontal repetition.
# - complexity_calibration:
#   The rule stays low-branch: one rotation direction, one square-padding anchor, one tie-
#   break. To keep each sub-change visible, the generator uses only non-square crops, forces
#   a section layout whose old axis would disagree with the new tie-break, and further keeps
#   only the aspect branch where rotate-then-square is visually distinguishable from both
#   square-only and square-then-rotate alternatives.
# - implicit_assumption_audit:
#   `_split_sections(...)` still only parses the four blocks; its layout result is no longer
#   the output axis. `bbox(template != 0)` still needs a non-empty template, which generation
#   guarantees. The old assumption that `crop_bbox(...)` already yields the final tile is
#   broken twice here: rotation changes orientation, and square padding changes extent. The
#   new `_square_pad_top_left(...)` introduces a new convention that extra area is appended
#   on the short side at the bottom or right; the generator builds oracle outputs with that
#   same convention. `_repeat_is_horizontal(...)` now resolves squares horizontally, and the
#   generator always uses N >= 2 so that the tie-break affects a visible repeated layout.
# - placement_legality:
#   The square expansion happens only in the solver/output logic, so the input template can
#   keep the old bordered rectangular section. The generator fixes the section layout to
#   stacked rows, which lets every section share the template width and prevents accidental
#   reuse of the base task's horizontal tiling rule. Stencil blobs keep the original one-cell
#   separation checks, so component counting remains stable and non-overlapping.
# - visual_observability:
#   Humans can see three stable cues across examples: the output motif is rotated relative to
#   the yellow template, the repeated tile has become square with an added blank strip on the
#   right, and stacked input sections still lead to a horizontal row of copies. The generator
#   intentionally preserves only this branch because it keeps the "rotate first, then square-
#   pad" order visually separable from the ambiguous square-then-rotate branch.
# - cleanliness_and_inference:
#   No extra markers are added. Blue still only separates sections, green still only encodes
#   the copy count, and the two solid swatches still directly provide output colors. The new
#   padding is background-colored, so it reads as a geometric canvas extension rather than a
#   new object layer.
# - post_edit_solver_assumptions:
#   `main()` assumes the tight crop is non-empty, that clockwise rotation is always intended,
#   that square padding is top-left anchored, and that square crops tile horizontally. The
#   generator enforces each assumption explicitly and rejects samples where any one of the
#   three fused controls would be behaviorally silent.


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


def _rotate_clockwise(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, k=3)


def _square_pad_top_left(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    size = max(h, w)
    out = np.zeros((size, size), dtype=bool)
    out[:h, :w] = mask
    return out


def _repeat_is_horizontal(tile_mask: np.ndarray) -> bool:
    h, w = tile_mask.shape
    if w > h:
        return True
    if h > w:
        return False
    return True


def _build_fused_tile_mask(crop_mask: np.ndarray) -> np.ndarray:
    return _square_pad_top_left(_rotate_clockwise(crop_mask))


def _crop_template_mask(template: np.ndarray) -> np.ndarray:
    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=bool)
    return crop_bbox(template_mask, template_bbox)


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with 1-pixel bg_color separators."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate a cropped template, expand it to a square, recolor it, and repeat it from a stencil count.

    Input structure:
    - Full blue separator rows or columns divide the input into four sections.
    - Section 1 contains a yellow-on-black template, section 2 contains the green repeat-count stencil.
    - Sections 3 and 4 are solid swatches that specify the output foreground and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black cells, rotate that crop 90 degrees clockwise, then pad its short side to obtain the smallest enclosing square; the added area is background.
    - Recolor the resulting square tile with the two swatch colors.
    - Count the 4-connected non-black components in section 2 and repeat the tile that many times with one background-colored spacer between copies; use horizontal repetition for tiles that are wider than tall, vertical for taller tiles, and horizontal when width equals height.

    Key insight:
    - The separator layout only segments the four input blocks. The real geometry comes from a rotate-then-square transform on the template, and because that transform makes the final tile square, the square tie-break determines the copy direction.
    """
    g = np.asarray(grid, dtype=int)
    _layout, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    crop_mask = _crop_template_mask(template)
    if crop_mask.size == 0:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = _build_fused_tile_mask(crop_mask)
    tile = np.where(tile_mask, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    return _tile_repeat(tile, n_repeats, background, _repeat_is_horizontal(tile_mask))


# --------------- generator (combo variant) ---------------

# --- CFG: tunable generation parameters ---
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
SHAPE_RETRIES = 120
GEN_RETRIES = 500


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


def _choose_crop_size() -> tuple[int, int]:
    sizes = list(range(CROP_SIZE_RANGE[0], CROP_SIZE_RANGE[1] + 1))
    pairs = [(h, w) for h in sizes for w in sizes if h != w]
    return random.choice(pairs)


def _square_then_rotate(mask: np.ndarray) -> np.ndarray:
    return _rotate_clockwise(_square_pad_top_left(mask))


def _fusion_flags(crop_mask: np.ndarray, layout_horizontal: bool) -> dict[str, bool]:
    fused = _build_fused_tile_mask(crop_mask)
    rotated_only = _rotate_clockwise(crop_mask)
    square_only = _square_pad_top_left(crop_mask)
    square_then_rotate = _square_then_rotate(crop_mask)
    return {
        "square_effective": rotated_only.shape[0] != rotated_only.shape[1],
        "rotation_effective": not np.array_equal(fused, square_only),
        "order_effective": not np.array_equal(fused, square_then_rotate),
        "tie_break_effective": _repeat_is_horizontal(fused) != layout_horizontal,
    }


def _make_shape() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        crop_h, crop_w = _choose_crop_size()
        density = random.uniform(*DENSITY_RANGE)
        shape = random_sprite(
            crop_h,
            crop_w,
            density=density,
            color_palette=[TEMPLATE_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        rotated = _rotate_clockwise(shape != 0)
        if rotated.shape[0] <= rotated.shape[1]:
            continue
        flags = _fusion_flags(shape != 0, layout_horizontal=False)
        if all(flags.values()):
            return shape
    return None


def _baseline_output(shape_mask: np.ndarray, fg_color: int, bg_color: int, n_repeats: int, horizontal: bool) -> np.ndarray:
    tile = np.where(shape_mask, fg_color, bg_color).astype(int)
    return _tile_repeat(tile, n_repeats, bg_color, horizontal)


def _gen_attempt() -> np.ndarray | None:
    layout_horizontal = False
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    shape = _make_shape()
    if shape is None:
        return None
    crop_mask = shape != 0
    tile_mask = _build_fused_tile_mask(crop_mask)

    fg_color, bg_color = random.sample(COLOR_POOL, 2)
    output = _baseline_output(tile_mask, fg_color, bg_color, n_repeats, _repeat_is_horizontal(tile_mask))

    template = np.zeros((shape.shape[0] + 2, shape.shape[1] + 2), dtype=int)
    template[1 : shape.shape[0] + 1, 1 : shape.shape[1] + 1] = shape

    shared_dim = template.shape[1]
    stencil = _build_stencil(n_repeats, shared_dim, layout_horizontal)
    if stencil is None:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, layout_horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, layout_horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, layout_horizontal)

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    old_output = _baseline_output(crop_mask, fg_color, bg_color, n_repeats, layout_horizontal)
    square_only_output = _baseline_output(
        _square_pad_top_left(crop_mask),
        fg_color,
        bg_color,
        n_repeats,
        _repeat_is_horizontal(_square_pad_top_left(crop_mask)),
    )
    square_then_rotate_output = _baseline_output(
        _square_then_rotate(crop_mask),
        fg_color,
        bg_color,
        n_repeats,
        _repeat_is_horizontal(_square_then_rotate(crop_mask)),
    )
    if np.array_equal(output, old_output):
        return None
    if np.array_equal(output, square_only_output):
        return None
    if np.array_equal(output, square_then_rotate_output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
