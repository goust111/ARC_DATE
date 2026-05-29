# Variant of b0039139
# Dimension: D6 — 区段布局与分隔线结构 (meta)
# Change: 四个区段由色1(蓝色)的分隔线(整行或整列)分隔，可以是全水平排列或全垂直排列或混合排列... → 区段数量可变：支持多个色板

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     - The first two sections keep their old roles: template then repetition stencil.
#     - The tail is no longer exactly two swatches. It now contains 3 or 4 swatches:
#       fill, bbox-background, edge, and optional accent.
#     - Therefore the input has 5 or 6 sections instead of a fixed 4.
#   concrete_pixel_behavior:
#     - The cropped template is no longer treated as one binary mask.
#     - Inside the crop, color 4 marks body/fill pixels, color 5 marks border pixels,
#       color 6 optionally marks a small accent, and color 0 remains the local background.
#     - Output recolors each symbolic region with the matching swatch while keeping the
#       original repeat-count logic from the stencil.
#   traced_difference:
#     - Original flow: `template != 0` collapsed every non-black pixel into one foreground,
#       so a 4/5/6 multi-symbol template would lose its structure and only use two colors.
#     - Variant flow: crop the raw template colors, map 0/4/5/(6) separately, then tile.
#     - Visible consequence: repeated copies now preserve a third color border and sometimes
#       a fourth accent, so extra swatches materially affect the output.
#   complexity_calibration:
#     - Only two palette counts are used in generation: 3 swatches (5 sections total) and
#       4 swatches (6 sections total). This keeps the branch count low enough for ARC.
#     - The border role is mandatory in every sample; the accent role is optional but only
#       generated when the shape has enough interior area to keep all regions visible.
#   implicit_assumptions_audit:
#     - `_split_sections` still assumes one global separator orientation. The generator keeps
#       the layout purely horizontal or purely vertical, so this remains valid.
#     - `main()` assumes section order is template, stencil, then semantic swatches in a
#       fixed order. The generator preserves that order exactly.
#     - `main()` assumes the cropped template uses only symbolic colors from {0,4,5,6}.
#       The generator constructs templates from these labels only.
#     - `main()` assumes the template crop visibly contains background, border, and fill;
#       the generator retries until those regions are all non-empty. Accent, when present,
#       is also guaranteed non-empty and does not erase the whole fill region.
#   placement_legality:
#     - The template is embedded in a 1-cell black frame exactly as before.
#     - Stencil blobs keep the old non-overlap and padding rules.
#     - Swatches are solid rectangles sharing the common axis length, so joining sections
#       never causes out-of-bounds writes or malformed separators.
#   visual_observability:
#     - Humans can infer the extra mapping because the extra swatch is paired with a visibly
#       distinct border ring that repeats in every copy.
#     - The optional accent appears only as an extra interior mark, so 5-section and 6-section
#       cases differ by an obvious additional swatch and an obvious additional region.
#   cleanliness:
#     - No section overlays another; each object family lives in its own block.
#     - Colors are chosen as distinct visible palette ids, not by any hidden sorting rule.


SEPARATOR_COLOR = 1
STENCIL_COLOR = 3
TEMPLATE_FILL_COLOR = 4
TEMPLATE_EDGE_COLOR = 5
TEMPLATE_ACCENT_COLOR = 6

# --- CFG: tuned for visible multi-region templates ---
REPEAT_COUNT_RANGE = (2, 6)
CROP_SIZE_RANGE = (4, 6)
DENSITY_RANGE = (0.60, 0.82)
COLOR_POOL = [2, 3, 4, 5, 6, 7, 8, 9]
SWATCH_DIM_RANGE = (3, 8)
STENCIL_VARYING_EXTRA = 4
BLOB_PLACE_RETRIES = 200
GEN_RETRIES = 400
SHAPE_RETRIES = 200
USE_ACCENT_PROB = 0.5


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


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    edge = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                edge[r, c] = True
                continue
            if not (mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]):
                edge[r, c] = True
    return edge


def _recolor_symbolic_tile(symbolic_tile: np.ndarray, palette_colors: list[int]) -> np.ndarray:
    fill_color = palette_colors[0]
    background_color = palette_colors[1]
    edge_color = palette_colors[2]
    accent_color = palette_colors[3] if len(palette_colors) >= 4 else fill_color

    tile = np.full(symbolic_tile.shape, background_color, dtype=int)
    tile[symbolic_tile == TEMPLATE_FILL_COLOR] = fill_color
    tile[symbolic_tile == TEMPLATE_EDGE_COLOR] = edge_color
    tile[symbolic_tile == TEMPLATE_ACCENT_COLOR] = accent_color
    return tile


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
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


def _pick_accent(fill_mask: np.ndarray) -> np.ndarray | None:
    coords = [tuple(x) for x in np.argwhere(fill_mask)]
    if len(coords) < 2:
        return None

    seed_r, seed_c = random.choice(coords)
    accent = np.zeros_like(fill_mask, dtype=bool)
    accent[seed_r, seed_c] = True

    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = seed_r + dr, seed_c + dc
        if 0 <= rr < fill_mask.shape[0] and 0 <= cc < fill_mask.shape[1] and fill_mask[rr, cc]:
            neighbors.append((rr, cc))

    if neighbors and int(fill_mask.sum()) >= 4 and random.choice([True, False]):
        rr, cc = random.choice(neighbors)
        accent[rr, cc] = True

    if int(np.sum(fill_mask & ~accent)) == 0:
        return None
    return accent


def _make_symbolic_template(crop_h: int, crop_w: int, use_accent: bool) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        density = random.uniform(*DENSITY_RANGE)
        shape = random_sprite(
            crop_h,
            crop_w,
            density=density,
            color_palette=[TEMPLATE_FILL_COLOR],
            connectivity=4,
            symmetry="not_symmetric",
        )
        shape_mask = shape != 0
        if not np.any(~shape_mask):
            continue

        edge = _edge_mask(shape_mask)
        fill = shape_mask & ~edge
        if int(edge.sum()) == 0 or int(fill.sum()) == 0:
            continue

        symbolic = np.zeros((crop_h, crop_w), dtype=int)
        symbolic[fill] = TEMPLATE_FILL_COLOR
        symbolic[edge] = TEMPLATE_EDGE_COLOR

        if use_accent:
            accent = _pick_accent(fill)
            if accent is None:
                continue
            symbolic[accent] = TEMPLATE_ACCENT_COLOR
            symbolic[fill & ~accent] = TEMPLATE_FILL_COLOR

        return symbolic
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a cropped symbolic template and repeat it from a component-count stencil.

    Input structure:
    - Solid blue (1) separator rows or columns split the grid into 5 or 6 sections.
    - Section 1 is a black-framed symbolic template; section 2 is a component-count stencil.
    - The remaining sections are solid swatches in fixed order: fill, local background,
      border, and optionally accent.

    Rule:
    - Crop section 1 to the bbox of all non-black template cells, preserving the symbolic
      colors inside that crop.
    - Recolor symbolic fill / background / border / optional accent cells from the swatches.
    - Count 4-connected non-black components in section 2 and repeat the recolored crop that
      many times, separating copies with one row or column of the mapped background color.

    Key insight:
    - The stencil still controls only the repeat count, but the template is no longer binary:
      extra swatch sections matter because different symbolic regions inside the crop are now
      mapped independently before tiling.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    if len(sections) < 5:
        raise ValueError("expected at least 5 sections")

    template = sections[0]
    repetition_pattern = sections[1]
    palette_blocks = sections[2:]
    palette_colors = [_solid_color(block) for block in palette_blocks]
    if len(palette_colors) < 3:
        raise ValueError("expected fill/background/edge swatches")

    template_bbox = bbox(template != 0)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    symbolic_tile = crop_bbox(template, template_bbox)
    background_color = palette_colors[1]
    tile = _recolor_symbolic_tile(symbolic_tile, palette_colors)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    return _tile_repeat(tile, n_repeats, background_color, horizontal=(orientation == "horizontal"))


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)

    use_accent = random.random() < USE_ACCENT_PROB
    crop_h = random.randint(*CROP_SIZE_RANGE)
    crop_w = random.randint(*CROP_SIZE_RANGE)
    symbolic_crop = _make_symbolic_template(crop_h, crop_w, use_accent)
    if symbolic_crop is None:
        return None

    n_swatches = 4 if use_accent else 3
    palette_colors = random.sample(COLOR_POOL, n_swatches)
    tile = _recolor_symbolic_tile(symbolic_crop, palette_colors)
    output = _tile_repeat(tile, n_repeats, palette_colors[1], horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = symbolic_crop

    shared_dim = crop_h + 2 if horizontal else crop_w + 2
    stencil = _build_stencil(n_repeats, shared_dim, horizontal)
    if stencil is None:
        return None

    swatches = [_make_swatch(shared_dim, color, horizontal) for color in palette_colors]
    inp = _join_sections([template, stencil, *swatches], shared_dim, horizontal)

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
