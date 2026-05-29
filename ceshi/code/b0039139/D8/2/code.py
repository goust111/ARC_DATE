# Variant of b0039139
# Dimension: D8 — 模板裁剪方式 (transform)
# Change: 将区段1中的模板裁剪到色4像素的最小外接矩形(bbox)... → 裁剪后填充为矩形：凸包填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the D8 variant
# a) Concrete behavior and trace:
#    The input layout stays identical to the base task: four sections split by blue separator
#    rows/cols, ordered as template / stencil / foreground swatch / background swatch.
#    The change happens only when section 1 is turned into the tile. Previously, the yellow
#    template pixels were cropped to their bbox and recolored directly. Now the cropped yellow
#    silhouette is first completed to its convex hull, then recolored. Example: a 2x2 L-shape
#    template `[[4,4],[4,0]]` used to stay an L after recoloring; in this variant it becomes a
#    full 2x2 block because the missing corner lies inside the convex hull of the occupied pixel
#    squares. The stencil still contributes only the repeat count, and the swatches still define
#    foreground/background colors.
# b) Complexity calibration:
#    If the generator allowed mostly already-convex templates, the new rule would collapse back to
#    the original crop-and-recolor task. To avoid that, generation rejects shapes whose convex-hull
#    completion adds no cells. It also mixes two visible subfamilies: some completions fill the
#    entire bbox, others produce a smaller convex polygon inside the bbox. This keeps the rule
#    observable without introducing a second independent transformation dimension.
# c) Hidden-assumption review of the original main():
#    - `template_mask = template != 0` relied on "all non-black pixels in section 1 are template".
#      The variant is explicitly about color-4 template pixels, so the new solver uses
#      `template == TEMPLATE_COLOR`.
#    - `tile_mask = crop_bbox(template_mask, template_bbox)` assumed the cropped mask itself is the
#      output geometry. That assumption is no longer valid; we must replace it with convex-hull
#      completion on the cropped mask.
#    - Repetition counting and orientation logic do not depend on the old crop rule, so they stay.
# d) Placement legality:
#    The template is still embedded inside a 1-cell black border, so its bbox crop is well-defined.
#    The convex hull is computed only inside that crop, therefore the output tile size matches the
#    original crop size and remains compatible with the original stencil sizing and tiling code.
#    Stencil blobs and swatches retain the same non-overlap and separator constraints as before.
# e) Visual observability:
#    Solvers can infer the new rule from the fact that concave yellow templates become convex after
#    recoloring. This is visible in pixel geometry alone; no ordering by color id or hidden numeric
#    convention is used. Mixed samples where the hull is sometimes the full bbox and sometimes not
#    help distinguish "convex completion" from a weaker "fill every bbox" shortcut.
# f) Visual cleanliness:
#    Only one section changes semantics, and it remains a single monochrome template on black.
#    The output still contains repeated copies separated by the background color, so the display is
#    as clean as in the source task.
# g) Color inference:
#    Colors remain structurally explained by the two solid swatches. The geometric change is
#    independent of the chosen color ids, so no extra color-mapping burden is introduced.
#
# Post-change solver assumptions checked against the generator:
# - main() assumes section 1 contains a non-empty color-4 template component. generate_input()
#   guarantees this by using random_sprite with TEMPLATE_COLOR.
# - main() assumes blue separators split the grid into exactly four usable sections. The generator
#   assembles the grid with the original four-section layout.
# - main() assumes the convex hull is computed on the cropped template only. The generator always
#   keeps the meaningful shape entirely inside the black-bordered template panel.


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


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _point_in_convex_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if not polygon:
        return False
    if len(polygon) == 1:
        return point == polygon[0]

    sign = 0
    eps = 1e-9
    for i, a in enumerate(polygon):
        b = polygon[(i + 1) % len(polygon)]
        cross = _cross(a, b, point)
        if abs(cross) <= eps:
            continue
        cur = 1 if cross > 0 else -1
        if sign == 0:
            sign = cur
        elif sign != cur:
            return False
    return True


def _convex_hull_fill(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros_like(mask, dtype=bool)

    corners: list[tuple[float, float]] = []
    for r, c in coords:
        rr, cc = int(r), int(c)
        corners.extend(
            [
                (cc, rr),
                (cc + 1, rr),
                (cc, rr + 1),
                (cc + 1, rr + 1),
            ]
        )

    polygon = _convex_hull(corners)
    h, w = mask.shape
    filled = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if _point_in_convex_polygon((c + 0.5, r + 0.5), polygon):
                filled[r, c] = True
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a cropped template to its convex hull, recolor it, and repeat it by stencil count.

    Input structure:
    - Blue (1) separator rows or columns divide the grid into four sections.
    - Section 1 is a black template panel containing yellow (4) shape pixels; section 2 is a
      component-count stencil; sections 3 and 4 are solid swatches for output foreground/background.

    Rule:
    - Crop section 1 to the bbox of its yellow pixels, then fill the convex hull of that cropped
      silhouette before recoloring it with the two swatch colors.
    - Count the 4-connected non-black components in section 2.
    - Repeat the completed/recolored tile that many times, separating copies by one row or column
      of the output background color.

    Key insight: The stencil still encodes only how many copies to make; the new step is geometric
    completion of the cropped template into its convex hull before any recoloring or tiling.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template == TEMPLATE_COLOR
    template_bbox = bbox(template_mask)
    if template_bbox is None:
        return np.empty((0, 0), dtype=int)

    foreground = _solid_color(foreground_block)
    background = _solid_color(background_block)
    cropped_mask = crop_bbox(template_mask, template_bbox)
    tile_mask = _convex_hull_fill(cropped_mask)
    tile = np.where(tile_mask, foreground, background).astype(int)

    n_repeats = len(components_mask(repetition_pattern != 0, connectivity=4))
    if n_repeats <= 1:
        return tile

    if orientation == "horizontal":
        sep = np.full((tile.shape[0], 1), background, dtype=int)
        pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
        return np.hstack(pieces)

    sep = np.full((1, tile.shape[1]), background, dtype=int)
    pieces = [tile] + [x for _ in range(n_repeats - 1) for x in (sep, tile)]
    return np.vstack(pieces)


# --------------- generator (convex_hull_completion) ---------------

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
SHAPE_RETRIES = 300                  # per-sample retries to obtain a usable concave template
GEN_RETRIES = 400                    # outer generate_input retry limit
TARGET_HULL_MODES = ("full_bbox", "partial")


def _tile_repeat(tile: np.ndarray, n: int, bg_color: int, horizontal: bool) -> np.ndarray:
    """Repeat `tile` n times along the given axis with 1-pixel bg_color separator."""
    if horizontal:
        sep = np.full((tile.shape[0], 1), bg_color, dtype=int)
        pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
        return np.hstack(pieces)
    sep = np.full((1, tile.shape[1]), bg_color, dtype=int)
    pieces = [tile] + [x for _ in range(n - 1) for x in (sep, tile)]
    return np.vstack(pieces)


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
    """Place n_blobs 2-cell blobs (1x2 or 2x1) of STENCIL_COLOR on a black grid.
    Blobs are separated by >=1 cell of black in all directions."""
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


def _sample_shape(target_mode: str) -> tuple[np.ndarray, np.ndarray] | None:
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
        shape_mask = shape == TEMPLATE_COLOR
        hull_mask = _convex_hull_fill(shape_mask)
        added = int(hull_mask.sum() - shape_mask.sum())
        if added < 1:
            continue

        is_full_bbox = bool(hull_mask.all())
        if target_mode == "full_bbox" and not is_full_bbox:
            continue
        if target_mode == "partial" and is_full_bbox:
            continue
        return shape, hull_mask
    return None


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)
    target_mode = random.choice(TARGET_HULL_MODES)

    sample = _sample_shape(target_mode)
    if sample is None:
        return None
    shape, hull_mask = sample
    crop_h, crop_w = shape.shape

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    tile = np.where(hull_mask, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = np.zeros((crop_h + 2, crop_w + 2), dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = shape

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
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after %d attempts" % GEN_RETRIES)
