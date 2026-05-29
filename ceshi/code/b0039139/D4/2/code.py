# Variant of b0039139
# Dimension: D4 — 副本平铺方向 (grid_shape)
# Change: 平铺方向由分隔线方向决定——水平分隔线(行)则垂直平铺，垂直分隔线(列)则水平平铺... → 平铺方向由模板宽高比决定

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


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


def _repeat_is_horizontal(tile_mask: np.ndarray) -> bool:
    """Use the cropped template geometry to choose the replication axis."""
    h, w = tile_mask.shape
    if w > h:
        return True
    if h > w:
        return False
    return True


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with 1-pixel bg_color separators."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


# Design analysis for the D4 variant:
# - concretization:
#   - The input still has four sections separated by solid blue full rows or full columns.
#   - The first section contains a bordered template. After cropping away the black border,
#     the bbox dimensions of that crop decide the output axis: wide -> horizontal copies,
#     tall -> vertical copies, square -> horizontal by fixed tie-break.
#   - Example: if the sections are stacked top-to-bottom but the cropped template is 3x5,
#     the original task would repeat vertically from the separator rows; this variant
#     repeats horizontally because the visible template is wider than tall.
# - complexity calibration:
#   - Section layout remains random, but it no longer encodes the answer. Generator
#     samples wide, tall, and some square crops so the solver must inspect the template.
#   - The rule stays low-entropy: only three aspect cases exist, with a single fixed
#     tie-break for squares.
# - implicit assumption review:
#   - `_split_sections(...)` is still valid, but only for extracting the four blocks.
#     Its returned layout string must not be reused as the output axis.
#   - `bbox(template != 0)` must recover the intended crop size. Generator preserves the
#     original one-cell black border and uses `random_sprite(...)`, whose sprite touches
#     every side of its own canvas, so the crop dimensions stay exactly as chosen.
#   - Counting stencil components is unchanged; D4 only changes the axis used after the
#     count has already been recovered.
# - placement legality:
#   - Input assembly still uses separator layout only to enforce matching shared height or
#     width across sections, so the four blocks align cleanly and never overlap.
#   - Stencil blob placement keeps the original spacing and border checks, preventing
#     collisions and ensuring `components_mask(..., connectivity=4)` returns the target N.
# - visual observability:
#   - The deciding cue is visible geometry, not a hidden id. Humans can crop the template
#     mentally and compare width vs height.
#   - Generator samples section-layout orientation independently from template aspect, so
#     both separator layouts can produce either output axis across a sample batch; this
#     prevents degeneration back to the old rule.
# - cleanliness and color inference:
#   - No extra symbols are added. The template, stencil, and two swatches remain separated
#     into the same four clean blocks.
#   - Foreground/background recoloring is still read directly from the swatches; colors do
#     not participate in sorting or tie-breaking.
# - new solver assumption audit:
#   - Square crops always resolve to horizontal repetition. Generator includes square cases
#     and oracle-checks every constructed input with `main(...)` to guarantee consistency.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped template, then repeat it by a stencil-driven count using the template's aspect ratio.

    Input structure:
    - Full blue separator rows or columns split the canvas into four blocks.
    - Block 1 is a black-backed template, block 2 is a non-black component stencil.
    - Blocks 3 and 4 are solid swatches that provide the output foreground and background colors.

    Rule:
    - Crop block 1 to the bounding box of its non-black cells and recolor that crop with the two swatch colors.
    - Count the 4-connected non-black components in block 2 to get the number of copies.
    - Tile the recolored crop with one background-colored spacer between copies; use a horizontal row of copies when the cropped template is wider than tall, a vertical stack when it is taller than wide, and horizontal for square crops.

    Key insight:
    - The separator layout only tells you how to parse the four input blocks. The repeat axis is inferred from the cropped template's own geometry, not from whether the separators are rows or columns.
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
    "wide": 4,
    "tall": 4,
    "square": 2,
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
    square_pairs = [(s, s) for s in sizes]
    mode = random.choices(
        list(ASPECT_MODE_WEIGHTS.keys()),
        weights=list(ASPECT_MODE_WEIGHTS.values()),
        k=1,
    )[0]
    if mode == "wide":
        return random.choice(wide_pairs)
    if mode == "tall":
        return random.choice(tall_pairs)
    return random.choice(square_pairs)


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

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    tile = np.where(shape != 0, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, _repeat_is_horizontal(shape != 0))

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
    raise RuntimeError("Failed to generate valid input after %d attempts" % GEN_RETRIES)
