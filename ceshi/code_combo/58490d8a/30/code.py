# Combo variant of 58490d8a (index 30)
# combo_spec: D1/0+D2/3
# Name: 形状匹配+中空精灵

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage

from arc_utils import bbox_coords, components_mask, crop_bbox, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   baseline_reference:
#     - Original 58490d8a used one colored key pixel per row and counted outside components by
#       exact color equality.
#     - D1/0 replaces that lookup with exact shape matching against legend templates.
#     - D2/3 requires the outside components themselves to be hollow rings / frames with internal
#       background holes, so the visible colored cells are no longer the same mask as the template.
#   concrete_fused_rule:
#     - The legend remains the largest black rectangle and still doubles as the output canvas.
#     - Each key is now a solid 5x5 template silhouette inside the legend.
#     - Outside sprites are hollow boundary-only versions of those silhouettes, recolored with a
#       small non-key palette. Matching is therefore: fill the sprite's hole(s), recover its outer
#       silhouette, then compare that silhouette to a legend template; color is irrelevant.
#     - Output stays tally-based: on each template row, draw one dot per matching outside sprite,
#       stepping rightward by 2 cells from the template center.
#   concrete_trace:
#     - Original row logic:
#       legend key = red pixel, outside red blobs = 4, output row = red dots at cols 1,3,5,7.
#     - Fused row logic:
#       legend key = green solid clipped-corner 5x5 template, outside matching sprites may be
#       orange / blue / orange / blue hollow frames, count = 4, output row = green dots at
#       cols 3,5,7,9.
#     - Difference:
#       the raw outside mask no longer equals the legend mask because the sprite has holes, and the
#       outside color no longer predicts the row. The solver must recover outer shape first.
#   complexity_calibration:
#     - The template library is fixed to 6 full-bbox 5x5 silhouettes. This is large enough for
#       visible variation but small enough that 2-4 templates per sample remain learnable.
#     - All outside sprites are hollow and all legend templates are solid, so the fusion is
#       visually obvious in every sample rather than only occasionally.
#     - Outside colors come from a small shared non-key pool, which creates repeated color reuse
#       across different shapes and blocks any fallback to the original color-count rule.
#   implicit_assumption_audit_for_main:
#     - The old `key_col` scan assumed each key is a single pixel in one common column. False now:
#       the new solver must extract non-black components inside the legend and use their bbox
#       centers as tally anchors.
#     - The old `(g == color)` counting assumed grouping is color-based. False now: grouping is by
#       silhouette signature after filling holes inside each outside component.
#     - The old global-background shortcut is brittle when legend keys become large. The new solver
#       derives background from cells outside the legend box only.
#     - The new hole-filling step assumes outside sprites are closed 8-connected frames. Generator
#       guarantees this by deriving every outside sprite from an eroded solid template boundary.
#   placement_legality:
#     - Legend templates sit in 5x5 slots separated by one empty row, so they stay distinct under
#       8-connectivity even though the legend box is black.
#     - Outside sprites are pasted with one-cell 8-neighborhood padding, so no two sprites merge
#       and none touches the legend box.
#     - Grid sizes are enlarged relative to the seed because hollow 5x5 sprites with counts up to
#       five per key need substantially more packing room.
#   visual_observability:
#     - A human sees solid templates in the box and hollow frames outside, making the fused change
#       from the seed immediately visible.
#     - Key colors never appear on outside sprites, so color matching is contradicted directly.
#     - Filled-hole outer contours are stable across samples, and multiple repeated sprites per key
#       provide enough evidence to infer that the count belongs to the template shape class.
#   new_main_assumptions_and_generator_guarantees:
#     - Every legend template has a unique full-bbox 5x5 silhouette. Generator samples distinct
#       templates within one instance, so rows remain unambiguous.
#     - Every outside sprite is the exact hollow boundary of one library silhouette with no
#       rotation or reflection. Filling holes therefore recovers the intended template exactly.
#     - The template center is a valid tally anchor because all legend keys share the same 5x5
#       support and legend width is sampled from the target maximum count.


BLACK = 0
COLOR_POOL = tuple(range(1, 10))

# Legend / tally geometry.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
TEMPLATE_SIZE = 5
KEY_TOP_FIRST = 1
KEY_TOP_STEP = 6
TEMPLATE_LEFT = 1
TEMPLATE_CENTER_OFFSET = 2
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_SLACK_RANGE = (0, 3)

# Grid sizing and placement.
GRID_HEIGHT_RANGE = (36, 44)
GRID_WIDTH_RANGE = (36, 44)
LEGEND_EDGE_MARGIN = 2

# Sprite placement.
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8
DISTRACTOR_PROB = 0.5
GENERATION_ATTEMPTS = 500

FILL_STRUCTURE = np.ones((3, 3), dtype=bool)

# Fixed 5x5 silhouette library. Every mask has at least one interior cell so its hollow boundary
# version contains hole(s), and every bbox remains the full 5x5 support for a stable anchor.
SOLID_LIBRARY = (
    np.array(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        dtype=bool,
    ),
    np.array(
        [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]],
        dtype=bool,
    ),
    np.array(
        [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]],
        dtype=bool,
    ),
)


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _shape_signature_from_mask(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), rel


def _shape_signature_from_coords(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), rel


def _coords_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _filled_shape_signature(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    mask = _coords_mask(coords)
    filled = ndimage.binary_fill_holes(mask, structure=FILL_STRUCTURE)
    return _shape_signature_from_mask(filled)


def _component_color(grid: np.ndarray, coords: np.ndarray) -> int:
    r, c = map(int, coords[0])
    return int(grid[r, c])


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    best = None
    for coords in components_mask(grid == BLACK, connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(grid, bbox)
        if (sub != BLACK).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, bbox)))
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return tuple(map(int, best[1]))


def _outside_bg(grid: np.ndarray, outside: np.ndarray) -> int:
    vals, counts = np.unique(grid[outside], return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _extract_keys(box: np.ndarray) -> list[dict]:
    keys = []
    for coords in components_mask(box != BLACK, connectivity=8):
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        keys.append(
            {
                "row": int((r0 + r1) // 2),
                "col": int((c0 + c1) // 2),
                "color": _component_color(box, coords),
                "shape": _shape_signature_from_coords(coords),
            }
        )
    keys.sort(key=lambda item: int(item["row"]))
    return keys


def _solid_to_hollow(mask: np.ndarray) -> np.ndarray:
    interior = ndimage.binary_erosion(mask, structure=FILL_STRUCTURE, border_value=0)
    return mask & ~interior


HOLLOW_LIBRARY = tuple(_solid_to_hollow(mask) for mask in SOLID_LIBRARY)


def _validate_shape_library() -> None:
    solid_sigs = {_shape_signature_from_mask(mask) for mask in SOLID_LIBRARY}
    assert len(solid_sigs) == len(SOLID_LIBRARY)
    for solid, hollow in zip(SOLID_LIBRARY, HOLLOW_LIBRARY):
        comps = components_mask(hollow, connectivity=8)
        assert len(comps) == 1
        assert _filled_shape_signature(comps[0]) == _shape_signature_from_mask(solid)
        assert _shape_signature_from_mask(hollow) != _shape_signature_from_mask(solid)


_validate_shape_library()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count hollow outside sprites by recovered outer shape and write tallies inside the legend box.

    Input structure:
    - A large black rectangle contains several solid monochrome template silhouettes, one template per row band.
    - Outside the rectangle, scattered sprites are hollow rings / frames whose outer contours reuse those template shapes, but their colors come from a separate palette.

    Rule:
    - Ignore outside sprite color and treat each legend template as a shape class.
    - For each outside connected component, fill its enclosed hole(s), compare the recovered silhouette to the legend templates, and count exact matches.
    - Return a black canvas the size of the legend box, placing one same-color dot per match every other column to the right from the template center.

    Key insight: The colored cells outside are only the boundary of each object, so the solver must reconstruct the hidden filled silhouette and match that geometry to the legend template; the legend color matters only when drawing the tally.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    r0, c0, r1, c1 = _find_legend_bbox(g)
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    keys = _extract_keys(box)
    if not keys:
        return np.zeros_like(box)

    outside = np.ones((H, W), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    bg = _outside_bg(g, outside)

    counts_by_shape: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    mask = outside & (g != bg) & (g != BLACK)
    for coords in components_mask(mask, connectivity=8):
        sig = _filled_shape_signature(coords)
        counts_by_shape[sig] = counts_by_shape.get(sig, 0) + 1

    out = np.full((bh, bw), BLACK, dtype=int)
    for key in keys:
        count = counts_by_shape.get(key["shape"], 0)
        for i in range(count):
            cc = int(key["col"] + TALLY_COL_STEP * i)
            if 0 <= cc < bw:
                out[int(key["row"]), cc] = int(key["color"])
    return out


def _key_top(i: int) -> int:
    return KEY_TOP_FIRST + KEY_TOP_STEP * i


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_TOP_FIRST + KEY_TOP_STEP * (k - 1) + TEMPLATE_SIZE + LEGEND_HEIGHT_TAIL
    anchor_col = TEMPLATE_LEFT + TEMPLATE_CENTER_OFFSET
    min_bw = max(TEMPLATE_LEFT + TEMPLATE_SIZE, anchor_col + TALLY_COL_STEP * (max_count - 1) + 1)
    bw = min_bw + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _shape_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    bh, bw = _legend_shape(len(key_colors), max(counts))
    out = np.full((bh, bw), BLACK, dtype=int)
    anchor_col = TEMPLATE_LEFT + TEMPLATE_CENTER_OFFSET
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_top(i) + TEMPLATE_CENTER_OFFSET
        for j in range(count):
            out[rr, anchor_col + TALLY_COL_STEP * j] = int(color)
    return out


def _build_legend(shape: tuple[int, int], key_colors: list[int], key_shapes: list[np.ndarray]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, (color, mask) in enumerate(zip(key_colors, key_shapes)):
        patch = _shape_patch(mask, color, BLACK)
        paste(legend, patch, _key_top(i), TEMPLATE_LEFT, transparent=BLACK)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> np.ndarray | None:
    bh, bw = map(int, legend.shape)
    H = _randint(GRID_HEIGHT_RANGE)
    W = _randint(GRID_WIDTH_RANGE)
    min_h = bh + 2 * LEGEND_EDGE_MARGIN
    min_w = bw + 2 * LEGEND_EDGE_MARGIN
    if H < min_h or W < min_w:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    r0 = random.randint(LEGEND_EDGE_MARGIN, H - bh - LEGEND_EDGE_MARGIN)
    c0 = random.randint(LEGEND_EDGE_MARGIN, W - bw - LEGEND_EDGE_MARGIN)
    paste(grid, legend, r0, c0)
    return grid


def _sample_palette() -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    sprite_candidates = [c for c in remaining if c not in key_colors]
    pool_size = random.randint(2, min(3, len(sprite_candidates)))
    sprite_colors = random.sample(sprite_candidates, pool_size)
    return bg, key_colors, sprite_colors


def _sample_counts(k: int) -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _build_sprite_queue(
    key_shape_ids: list[int],
    counts: list[int],
    sprite_colors: list[int],
    distractor_shape_id: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for shape_id, count in zip(key_shape_ids, counts):
        hollow_mask = HOLLOW_LIBRARY[int(shape_id)]
        for _ in range(count):
            sprites.append(_shape_patch(hollow_mask, random.choice(sprite_colors), bg))
    if distractor_shape_id is not None:
        sprites.append(_shape_patch(HOLLOW_LIBRARY[int(distractor_shape_id)], random.choice(sprite_colors), bg))
    random.shuffle(sprites)
    return sprites


def _place_sprites(grid: np.ndarray, sprites: list[np.ndarray], bg: int) -> None:
    for sprite in sprites:
        sr, sc = random_free_location(
            grid,
            sprite,
            transparent=bg,
            padding=SPRITE_PADDING,
            padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
        )
        paste(grid, sprite, sr, sc, transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    bg, key_colors, sprite_colors = _sample_palette()
    k = len(key_colors)
    key_shape_ids = random.sample(range(len(SOLID_LIBRARY)), k)
    key_shapes = [SOLID_LIBRARY[i].copy() for i in key_shape_ids]
    remaining_ids = [i for i in range(len(SOLID_LIBRARY)) if i not in key_shape_ids]
    distractor_shape_id = random.choice(remaining_ids) if remaining_ids and random.random() < DISTRACTOR_PROB else None

    counts = _sample_counts(k)
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors, key_shapes)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_shape_ids, counts, sprite_colors, distractor_shape_id, bg)
    try:
        _place_sprites(grid, sprites, bg)
    except ValueError:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a combo variant input")
