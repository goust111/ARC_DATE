# Variant of 58490d8a
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: s.color == d.color — 图例框内的关键像素颜色与框外精灵颜色一致即为匹配组... → 形状匹配替代颜色匹配

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, paste, random_free_location


# Design analysis for the D1 variant
# ----------------------------------
# spec:
#   baseline_reference:
#     - Original 58490d8a grouped outside objects by color and used one legend pixel per row.
#     - Reference task 16b78196 shows the intended shape-comparison mechanism:
#       crop each component to its bbox and compare relative occupied coordinates exactly,
#       without any color dependence and without rotation / flip search.
#   concrete_variant:
#     - The legend is still the largest black rectangle and still doubles as the output canvas.
#     - Each legend key is now a 3x3 monochrome template shape placed in a fixed slot.
#     - Outside the legend, every sprite is also a 3x3 monochrome component, but its color is
#       sampled from a pool that excludes all legend colors. Matching is therefore forced to
#       depend on geometry, not pigment.
#     - The output stays close to the original task: each key gets a same-color tally row of
#       evenly spaced dots. The row anchor is the template's center cell, so the template slot
#       still visually marks where tally dots start.
#   concrete_trace:
#     - Original logic on one row:
#       legend key = red pixel, outside red objects = 3, output row = red dots at cols 1,3,5.
#     - Variant logic on one row:
#       legend key = green L-shape template, outside matching L-shapes may be orange / blue /
#       orange, count = 3, output row = green dots at cols 2,4,6.
#     - Difference:
#       grouping no longer follows object color, and the legend cue is multi-cell geometry rather
#       than a single identifying pixel.
#   complexity_calibration:
#     - Shape library size is fixed and small (6 exact 3x3 masks) so the rule remains learnable.
#     - Each sample uses 2-4 distinct templates, and multiple outside sprites repeat each chosen
#       shape, which gives the human solver repeated evidence for the grouping rule.
#     - Shapes keep a full 3x3 bbox and are mostly asymmetric, so the visual change is obvious.
#   implicit_assumption_audit_for_main:
#     - The old `key_col` scan assumed every key was a single pixel in one shared column. This is
#       false now; the new solver must enumerate non-black components inside the legend box.
#     - The old `(g == color)` counting assumed color equality is the only grouping criterion.
#       This is false now; the new solver must normalize each component shape and ignore color.
#     - The old global `get_bg(grid)` happened to work because the legend was small. With larger
#       3x3 templates, black can occupy more cells, so the new solver derives background from the
#       area outside the legend instead of relying on the whole-grid mode.
#     - Drawing dots is still safe because generator width is chosen from the target max count and
#       each key anchor column is fixed at the template center.
#   placement_legality:
#     - Legend templates are separated by a 1-row black gap, so they remain distinct 8-connected
#       components inside the legend.
#     - Outside sprites are pasted with 1-cell 8-neighborhood padding, so no two sprites merge and
#       none touches the legend box.
#     - Every sprite shares the same 3x3 support size, which keeps matching and placement simple.
#   visual_observability:
#     - Key colors never appear on outside sprites, eliminating the original color-matching cue.
#     - Outside sprites reuse a small non-key color pool, so the same outside color can belong to
#       different shape classes; a color-based hypothesis is visibly contradicted.
#     - The legend/output area stays clean: black box on one side, scattered sprites elsewhere,
#       one tally row per template.
#   new_main_assumptions_and_generator_guarantees:
#     - Every legend template occupies a full 3x3 bbox. Generator uses a fixed library with that
#       property, so anchor = bbox center is reliable.
#     - Outside sprites are exact copies of legend masks with no rotation or reflection. Generator
#       copies masks verbatim and uses a distractor from a different mask.
#     - Template shapes are unique within one sample. Generator samples distinct masks.


BLACK = 0
COLOR_POOL = tuple(range(1, 10))

# Legend / tally geometry.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
TEMPLATE_SIZE = 3
KEY_TOP_FIRST = 1
KEY_TOP_STEP = 4
TEMPLATE_LEFT = 1
TEMPLATE_CENTER_OFFSET = 1
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_SLACK_RANGE = (0, 3)

# Grid sizing and placement.
GRID_HEIGHT_RANGE = (22, 27)
GRID_WIDTH_RANGE = (22, 27)
LEGEND_EDGE_MARGIN = 2

# Sprite placement.
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8
DISTRACTOR_PROB = 0.5
GENERATION_ATTEMPTS = 400

# Fixed exact-match shape library. Every mask is connected and spans a full 3x3 bbox.
SHAPE_LIBRARY = (
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
)


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _shape_signature(coords: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return int(r1 - r0 + 1), int(c1 - c0 + 1), tuple(rel)


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
                "shape": _shape_signature(coords),
            }
        )
    keys.sort(key=lambda item: int(item["row"]))
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites by template shape and encode the tallies inside the legend box.

    Input structure:
    - A large black rectangle contains several small monochrome template shapes, one template per row band.
    - Outside the rectangle, scattered monochrome sprites reuse those shapes but can have unrelated colors.

    Rule:
    - Treat each legend template as a shape class and ignore sprite color when grouping.
    - Count outside connected components whose bbox-normalized shape exactly matches the template.
    - Return a black canvas the size of the legend box, placing one same-color dot per match at every other column from the template center.

    Key insight: The legend no longer labels rows by color alone; the solver must compare normalized geometry of components and use the template color only when drawing the tally.
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
        sig = _shape_signature(coords)
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
    min_bw = max(2 * max_count + 1, TEMPLATE_LEFT + TEMPLATE_SIZE)
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
    key_shapes: list[np.ndarray],
    counts: list[int],
    sprite_colors: list[int],
    distractor_shape: np.ndarray | None,
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for mask, count in zip(key_shapes, counts):
        for _ in range(count):
            sprites.append(_shape_patch(mask, random.choice(sprite_colors), bg))
    if distractor_shape is not None:
        sprites.append(_shape_patch(distractor_shape, random.choice(sprite_colors), bg))
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
    key_shapes = [shape.copy() for shape in random.sample(SHAPE_LIBRARY, k)]
    remaining_shapes = [shape for shape in SHAPE_LIBRARY if not any(np.array_equal(shape, key) for key in key_shapes)]
    distractor_shape = random.choice(remaining_shapes).copy() if remaining_shapes and random.random() < DISTRACTOR_PROB else None

    counts = _sample_counts(k)
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors, key_shapes)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_shapes, counts, sprite_colors, distractor_shape, bg)
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
    raise RuntimeError("failed to generate a valid 58490d8a D1 variant input")
