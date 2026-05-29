# Combo variant of b0039139 (index 8)
# combo_spec: D2/0+D8/2
# Name: 负空间的凸包填充

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, components_mask, crop_bbox, random_sprite


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The base task reads section 1 as painted yellow foreground on black, crops that
#    painted shape, recolors it, and repeats it by the blob count in section 2. This
#    combo changes both the source geometry and the crop result. In section 1, the true
#    shape is the black cut-out inside a yellow plaque, so zeros are the template and
#    yellow is merely the surrounding material. After cropping the zero-mask bbox, that
#    negative-space shape is completed to its convex hull before recoloring.
#    Small trace:
#      section-1 inner crop:        zero-mask crop:         convex-hull fill:
#        0 0 4                        1 1                    1 1
#        0 4 4                        1 0                    1 1
#        4 4 4
#    D2-only would output the L-shape from the zero-mask crop. This combo instead fills
#    the missing corner and outputs a 2x2 block. The original base solver would be more
#    wrong still: it would read the yellow plaque as foreground and keep the complement.
#
# b) Complexity calibration:
#    If the generator allowed convex or solid negative-space shapes, the hull step would
#    be invisible and the task would collapse into the D2-only inversion. To keep the
#    fusion observable, generation only accepts negative-space masks whose convex hull
#    adds at least one cell. It also samples two hull-result families: some completions
#    fill the entire cropped bbox, while others form a smaller convex polygon, which
#    distinguishes "convex hull" from the simpler but incorrect shortcut "fill bbox".
#
# c) Hidden assumptions in the original main():
#    - `template_mask = template != 0` assumed non-black cells are the true shape. The
#      combo breaks this completely; in the new input, non-black cells are the plaque.
#    - `bbox(template_mask)` assumed the template background is black. Now the outside of
#      the negative-space shape is yellow, so the old bbox would expand to the whole
#      plaque instead of the intended cut-out crop.
#    - `tile_mask = crop_bbox(template_mask, template_bbox)` assumed the cropped mask is
#      already the final output geometry. The combo invalidates that too: the cropped
#      negative-space mask must be convex-hull-filled before recoloring.
#    - Counting stencil components and tiling orientation do not rely on the old template
#      convention, so those parts remain structurally valid.
#
# d) Placement legality:
#    The template section remains a framed rectangle of size `(crop_h + 2) x (crop_w + 2)`.
#    The inner `crop_h x crop_w` area stores the negative-space pattern, with zeros for the
#    true shape and color 4 elsewhere. Because `random_sprite()` guarantees foreground on
#    every border of the sampled mask, the zero-mask bbox equals the whole inner crop, so
#    the shared dimension used by stencil and swatches is consistent with the solver crop.
#    Stencil blobs still sit on a separate black panel with a one-cell safety margin, so
#    there is no overlap or out-of-bounds placement anywhere in the assembled input.
#
# e) Visual observability:
#    From a few examples, a human can see that section 1 is not "paint this yellow thing";
#    it is "read the hole punched out of the yellow plaque". The output then becomes more
#    convex than that cut-out, often filling an obvious missing corner. Because the stencil
#    still only controls repetition count, the new visual cue is concentrated in one place
#    instead of being diluted across multiple unrelated changes.
#
# f) Visual cleanliness:
#    Each section keeps a single role: plaque template, blob-count stencil, foreground
#    swatch, background swatch. The combo does not stack multiple markings in one panel;
#    it only changes how section 1 is interpreted. Repeated output tiles are still spaced
#    by the chosen background color, so the final grids remain easy to parse.
#
# g) Change inferability:
#    Color mapping stays simple and structurally grounded: sections 3 and 4 are still solid
#    swatches that directly supply output foreground and background colors. The new logic is
#    entirely geometric. Since the generator forces visible hull additions, the difference
#    between "read the holes" and "fill the holes' convex hull" can be inferred from a
#    handful of input/output examples without relying on hidden color-id conventions.
#
# Post-change solver assumptions checked against the generator:
# - main() assumes section 1 contains at least one zero cell inside a color-4 plaque.
#   generate_input() constructs exactly that.
# - main() assumes the zero-mask crop is non-empty and meaningful as a standalone shape.
#   The sampled negative-space mask is always connected and bbox-tight.
# - main() assumes convex-hull filling is applied only after cropping the zero-mask bbox.
#   The generator builds outputs from that same cropped negative-space hull.
# - main() assumes four sections split cleanly by blue separator rows or columns.
#   The generator still assembles the original four-panel layout.


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
SHAPE_RETRIES = 300
GEN_RETRIES = 400
TARGET_HULL_MODES = ("full_bbox", "partial")


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
    Recover a negative-space template, convex-hull-fill it, recolor it, and repeat it by stencil count.

    Input structure:
    - Solid blue (1) separator rows or columns divide the grid into four sections.
    - Section 1 is a yellow (4) plaque whose black (0) cut-out defines the template shape;
      section 2 is a component-count stencil.
    - Sections 3 and 4 are solid color swatches for the output foreground and background.

    Rule:
    - Read section 1 by treating black cells as the template, crop to the bbox of those
      black cells, then fill the convex hull of that cropped negative-space silhouette.
    - Recolor the completed mask with the two swatch colors, count 4-connected non-black
      components in section 2, and repeat the tile that many times with one background-
      colored separator row/column between copies.

    Key insight: The template is encoded as empty space rather than painted foreground, and
    the output geometry is not the raw cut-out but its convex completion after cropping.
    """
    g = np.asarray(grid, dtype=int)
    orientation, sections = _split_sections(g)
    template, repetition_pattern, foreground_block, background_block = sections[:4]

    template_mask = template == 0
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


# --------------- generator (negative_space_convex_hull) ---------------


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


def _sample_negative_space_shape(target_mode: str) -> tuple[np.ndarray, np.ndarray] | None:
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
        shape_mask = shape != 0
        hull_mask = _convex_hull_fill(shape_mask)
        added = int(hull_mask.sum() - shape_mask.sum())
        if added < 1:
            continue

        is_full_bbox = bool(hull_mask.all())
        if target_mode == "full_bbox" and not is_full_bbox:
            continue
        if target_mode == "partial" and is_full_bbox:
            continue
        return shape_mask, hull_mask
    return None


def _make_negative_template(shape_mask: np.ndarray) -> np.ndarray:
    crop_h, crop_w = shape_mask.shape
    template = np.full((crop_h + 2, crop_w + 2), TEMPLATE_COLOR, dtype=int)
    template[1 : crop_h + 1, 1 : crop_w + 1] = np.where(shape_mask, 0, TEMPLATE_COLOR)
    return template


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice([True, False])
    n_repeats = random.randint(*REPEAT_COUNT_RANGE)
    target_mode = random.choice(TARGET_HULL_MODES)

    sample = _sample_negative_space_shape(target_mode)
    if sample is None:
        return None
    shape_mask, hull_mask = sample
    crop_h, crop_w = shape_mask.shape

    fg_color, bg_color = random.sample(COLOR_POOL, 2)

    tile = np.where(hull_mask, fg_color, bg_color).astype(int)
    output = _tile_repeat(tile, n_repeats, bg_color, horizontal)

    template = _make_negative_template(shape_mask)

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
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
