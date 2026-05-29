# Combo variant of b0039139 (index 2)
# combo_spec: D2/2+D4/2
# Name: 旋转模板决定平铺方向

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D2/2 + D4/2 combo variant
# - concrete_behavior:
#   The input still consists of four blue-separated sections: template, stencil, foreground
#   swatch, background swatch. The template is cropped to its non-black bbox exactly as in
#   the base task, but that crop is then rotated 90 degrees clockwise before it is recolored
#   or used for any geometry reasoning.
#   Example trace: if the visible template crop is a 2x4 mask, the base task would recolor
#   that 2x4 mask and use the separator layout to choose the repeat axis. In this combo, the
#   2x4 crop first becomes a 4x2 mask, then the 4x2 shape is recolored, and because the
#   rotated mask is taller than wide it is stacked vertically. Rotation therefore changes
#   both the tile appearance and the axis decision.
# - complexity_calibration:
#   The rule has only one geometric transform and one aspect-ratio classifier, so it stays
#   learnable, but the two steps are coupled: the aspect check must be applied after the
#   rotation, not before. The generator samples only non-square template crops so the
#   rotation reliably swaps wide/tall and visibly affects the output axis.
# - implicit_assumption_review:
#   `_split_sections(...)` still works only as a parser for the four blocks; its layout label
#   is no longer semantically tied to output orientation. `crop_bbox(...)` still recovers the
#   intended template crop because `random_sprite(...)` touches every border of its own canvas.
#   The original assumption that the cropped mask can be tiled directly is broken and replaced
#   by a new assumption: the rotated crop is the true tile prototype and also the source for
#   the aspect-ratio decision.
# - placement_legality:
#   Rotation happens only inside the solver/output construction, so the input layout remains
#   a legal four-section assembly with matching shared height or width. Stencil blobs still
#   use the original spaced placement rule, preventing overlaps and preserving their count
#   under 4-connectivity.
# - visual_observability:
#   Humans can see the yellow template, imagine the 90-degree turn, and then notice that the
#   repeated output direction follows the turned shape rather than the blue separator layout.
#   Because layout orientation is randomized independently, examples can show the same input
#   parsing layout producing different output axes.
# - cleanliness_and_color_inference:
#   No extra markers are introduced. Blue still means separators, green still means repeat
#   count blobs, yellow still marks the source template, and the two solid swatches still
#   define the recoloring. The fusion changes only geometry, so color inference stays clean.
# - fusion_non_degeneracy:
#   The generator intentionally rejects square template crops. That ensures the clockwise
#   rotation flips wide vs tall, so the D2 change truly controls the D4 decision. On every
#   generated sample, the repeat axis inferred from the rotated crop differs from the axis
#   that would have been inferred from the unrotated crop.
# - new_solver_assumptions_review:
#   `main(...)` now assumes the template section has at least one non-black pixel, the crop is
#   intended to be rotated clockwise, and wide/tall comparison is performed on the rotated
#   crop with a fixed square tie-break of horizontal. The generator guarantees the first two
#   directly and avoids square crops so the tie-break branch is defined but not needed.


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


def _repeat_is_horizontal(tile_mask: np.ndarray) -> bool:
    h, w = tile_mask.shape
    if w > h:
        return True
    if h > w:
        return False
    return True


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the chosen axis with 1-pixel bg separators."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate a cropped template, recolor it, and tile it by a stencil count using the rotated geometry.

    Input structure:
    - Full blue (1) separator rows or columns divide the canvas into four sections.
    - Section 1 is a black-backed template drawn in color 4; section 2 is the repeat-count stencil.
    - Sections 3 and 4 are solid swatches that specify the output foreground and background colors.

    Rule:
    - Crop section 1 to the bbox of its non-black cells, rotate that crop 90 degrees clockwise, and recolor the rotated mask with the two swatches.
    - Count the 4-connected non-black components in section 2 to get the number of copies.
    - Repeat the rotated tile with one background-colored spacer between copies; choose a horizontal row when the rotated tile is wider than tall, a vertical stack when it is taller than wide, and horizontal for square rotated tiles.

    Key insight:
    - The separator layout only tells you how to split the four input sections. The real repeat axis is decided by the rotated template's aspect ratio, so the rotation step must be inferred before the tiling direction can be recovered.
    """
    g = np.asarray(grid, dtype=int)
    _layout, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template != 0
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    tile_mask = crop_bbox(template_mask, template_bbox)
    tile_mask = _rotate_clockwise(tile_mask)
    tile = np.where(tile_mask, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    return _tile_repeat(tile, n_repeats, background, _repeat_is_horizontal(tile_mask))


# --------------- generator (c) ---------------

# --- CFG: tunable generation parameters ---
REPEAT_COUNT_RANGE = (2, 6)          # blob count N (= tile repetitions)
CROP_SIZE_RANGE = (2, 5)             # crop height/width bounds
DENSITY_RANGE = (0.55, 0.85)         # fill ratio of shape bounding box
COLOR_POOL = list(range(2, 10))      # fg/bg color choices (2-9 inclusive)
TEMPLATE_COLOR = 4                   # shape drawn in yellow inside template
STENCIL_COLOR = 3                    # blob color in stencil section
SEPARATOR_COLOR = 1                  # blue separator rows/cols
SWATCH_DIM_RANGE = (3, 8)            # varying-axis size for color swatches
STENCIL_VARYING_EXTRA = 4            # max extra cells beyond minimum stencil size
BLOB_PLACE_RETRIES = 200             # per-blob placement attempts in stencil
GEN_RETRIES = 300                    # outer generate_input retry limit
ASPECT_MODE_WEIGHTS = {
    "wide": 1,
    "tall": 1,
}


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


def _build_stencil(n_blobs: int, shared_dim: int, horizontal: bool) -> np.ndarray | None:
    """Place n_blobs 2-cell blobs (1x2 or 2x1) of STENCIL_COLOR on a black grid."""
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
    wide_pairs = [(h, w) for h in sizes for w in sizes if w > h]
    tall_pairs = [(h, w) for h in sizes for w in sizes if h > w]
    mode = random.choices(
        list(ASPECT_MODE_WEIGHTS.keys()),
        weights=list(ASPECT_MODE_WEIGHTS.values()),
        k=1,
    )[0]
    if mode == "wide":
        return random.choice(wide_pairs)
    return random.choice(tall_pairs)


def _gen_attempt() -> np.ndarray | None:
    layout_horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

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
    rotated_mask = _rotate_clockwise(shape != 0)
    if layout_horizontal == _repeat_is_horizontal(rotated_mask):
        return None

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    tile = np.where(rotated_mask, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, _repeat_is_horizontal(rotated_mask))

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

    shared_dim = (crop_h + 2) if layout_horizontal else (crop_w + 2)
    stencil = _build_stencil(n_repeats, shared_dim, layout_horizontal)
    if stencil is None:
        return None

    fg_swatch = _make_swatch(shared_dim, fg_color, layout_horizontal)
    bg_swatch = _make_swatch(shared_dim, bg_color, layout_horizontal)
    inp = _join_sections([template, stencil, fg_swatch, bg_swatch], shared_dim, layout_horizontal)

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
