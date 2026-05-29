# Combo variant of 58490d8a (index 16)
# combo_spec: D1/0+D14/0
# Name: 形状匹配+排序重排

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - Baseline 58490d8a used one colored key pixel per legend row, matched outside objects by
#      that color, and redrew the tally on the same row.
#    - This combo changes both the grouping cue and the output row assignment:
#      each legend row band now contains a 3x3 monochrome template shape, outside sprites are
#      grouped by exact bbox-normalized shape, and only after counting are the legend row slots
#      reassigned from largest count to smallest.
#    - Example trace:
#        input legend rows top->bottom = [green L, yellow zigzag, blue U]
#        outside matching counts          = [2, 5, 3]
#        unsorted shape tally             = [green x2, yellow x5, blue x3]
#        combo output                     = [yellow x5, blue x3, green x2]
#      The first change replaces color equality with geometry comparison; the second change
#      permutes which row slot receives each tally.
#
# b) Complexity calibration
#    - The shape library is small and fixed (6 full-3x3 masks), so the visual vocabulary stays
#      learnable across 2-4 training pairs instead of exploding into many ad-hoc geometries.
#    - Each sample uses 2-4 distinct templates, and every chosen template appears multiple times
#      outside according to a sampled count in 1..5, giving repeated evidence for shape grouping.
#    - Output ranking uses distinct counts and the generator rejects already-descending input
#      orders, so the row permutation is visible rather than collapsing back to the unsorted
#      version.
#
# c) Implicit assumption audit
#    - Original assumption: keys lie in one shared column as single pixels. Broken by D1/0.
#      Fix: enumerate non-black connected components inside the legend box and use each template's
#      bbox center as its tally anchor.
#    - Original assumption: `(g == color)` identifies a count group. Broken by D1/0.
#      Fix: normalize component coordinates to a shape signature and ignore outside color.
#    - Original assumption: input row == output row. Broken by D14/0.
#      Fix: keep original slot rows separately, count first, sort `(orig_row, color, shape, count)`
#      records by `(-count, orig_row)`, then redraw onto ranked slots.
#    - New combo interaction: once rows are permuted, the solver must carry both the template
#      color and the template anchor column through sorting; generator keeps all templates in the
#      same column so the anchor remains valid after reordering.
#
# d) Placement legality
#    - Legend templates occupy disjoint 3x3 slots with a full blank row between them, so they stay
#      separate under 8-connectivity.
#    - Outside sprites are pasted with one-cell 8-neighborhood padding, preventing overlap,
#      merges, and contact with the legend box.
#    - Grid size is increased relative to the baseline so the taller 3x3-slot legend still fits
#      with edge margins and with enough free area for all sprites.
#
# e) Visual observability
#    - Outside sprite colors are sampled from a pool that excludes legend colors, so a color-match
#      hypothesis visibly fails.
#    - Multiple shape classes may share the same outside colors, which gives positive evidence that
#      geometry, not pigment, determines grouping.
#    - Ranking depends only on the visible multiplicity of matching shapes. Ties are broken by
#      original row order for determinism, but the generator samples distinct counts so the primary
#      rule is count-based sorting.
#
# f) Visual cleanliness
#    - The black legend remains a single dedicated reasoning/output region, while all counted
#      sprites stay outside it on a uniform background.
#    - Tally marks are still one row of evenly spaced dots per ranked slot, so the output is no
#      busier than the original task despite the extra reasoning step.
#
# g) Generator guarantees for new solver assumptions
#    - Every legend template and every counted sprite is an exact copy of one library mask, with no
#      rotations or reflections. Exact signature comparison is therefore sufficient.
#    - Chosen key templates are unique within a sample, so each shape class maps to one legend row.
#    - Counts are distinct and the input legend order is not already descending, guaranteeing that
#      sorting changes at least one row slot.
#    - Outside colors never equal key colors, guaranteeing that the original color-based rule would
#      disagree with the combo behavior instead of silently succeeding.


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
GENERATION_ATTEMPTS = 500

# Fixed exact-match shape library. Each mask is connected and spans a full 3x3 bbox.
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


def _extract_keys(box: np.ndarray) -> list[dict[str, object]]:
    keys: list[dict[str, object]] = []
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
    keys.sort(key=lambda item: (int(item["row"]), int(item["col"])))
    return keys


def _sorted_records(keys: list[dict[str, object]], counts_by_shape: dict[tuple[int, int, tuple[tuple[int, int], ...]], int]) -> list[dict[str, object]]:
    records = []
    for key in keys:
        shape = key["shape"]
        records.append(
            {
                "orig_row": int(key["row"]),
                "anchor_col": int(key["col"]),
                "color": int(key["color"]),
                "shape": shape,
                "count": int(counts_by_shape.get(shape, 0)),
            }
        )
    records.sort(key=lambda item: (-int(item["count"]), int(item["orig_row"])))
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites by template shape, then rank the legend rows by those counts.

    Input structure:
    - A large black rectangle contains several small monochrome template shapes, one template in each horizontal row band.
    - Outside the rectangle, scattered monochrome sprites reuse some of those shapes but can have unrelated colors.

    Rule:
    - Treat each legend template as a shape class, count outside connected components whose bbox-normalized shape matches it exactly, and ignore sprite color while counting.
    - Sort the legend templates by count from largest to smallest, breaking ties by their original top-to-bottom order.
    - Return a black canvas the size of the legend box, drawing one same-color dot per counted match at every other column from the ranked row's template center.

    Key insight: The legend now serves two purposes in sequence: its shapes define which outside sprites belong together, and its row slots are then reused as sorted ranking positions rather than fixed destinations.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    r0, c0, r1, c1 = _find_legend_bbox(g)
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    keys = _extract_keys(box)
    if not keys:
        return np.zeros_like(box)

    slot_rows = [int(key["row"]) for key in keys]
    outside = np.ones((H, W), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    bg = _outside_bg(g, outside)

    counts_by_shape: dict[tuple[int, int, tuple[tuple[int, int], ...]], int] = {}
    mask = outside & (g != bg) & (g != BLACK)
    for coords in components_mask(mask, connectivity=8):
        sig = _shape_signature(coords)
        counts_by_shape[sig] = counts_by_shape.get(sig, 0) + 1

    records = _sorted_records(keys, counts_by_shape)
    out = np.full((bh, bw), BLACK, dtype=int)
    for new_idx, record in enumerate(records):
        rr = int(slot_rows[new_idx])
        anchor_col = int(record["anchor_col"])
        color = int(record["color"])
        count = int(record["count"])
        for i in range(count):
            cc = anchor_col + TALLY_COL_STEP * i
            if 0 <= cc < bw:
                out[rr, cc] = color
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


def _sorted_key_records(key_colors: list[int], key_shapes: list[np.ndarray], counts: list[int]) -> list[tuple[int, int, np.ndarray, int]]:
    records = [
        (int(i), int(color), key_shapes[i], int(counts[i]))
        for i, color in enumerate(key_colors)
    ]
    records.sort(key=lambda item: (-int(item[3]), int(item[0])))
    return records


def _build_expected(key_colors: list[int], key_shapes: list[np.ndarray], counts: list[int]) -> np.ndarray:
    bh, bw = _legend_shape(len(key_colors), max(counts))
    out = np.full((bh, bw), BLACK, dtype=int)
    anchor_col = TEMPLATE_LEFT + TEMPLATE_CENTER_OFFSET
    for out_i, (_orig_i, color, _shape, count) in enumerate(_sorted_key_records(key_colors, key_shapes, counts)):
        rr = _key_top(out_i) + TEMPLATE_CENTER_OFFSET
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
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    counts = random.sample(choices, k)
    while counts == sorted(counts, reverse=True):
        random.shuffle(counts)
    return [int(v) for v in counts]


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
    expected = _build_expected(key_colors, key_shapes, counts)
    legend = _build_legend(expected.shape, key_colors, key_shapes)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_shapes, counts, sprite_colors, distractor_shape, bg)
    try:
        _place_sprites(grid, sprites, bg)
    except ValueError:
        return None

    solved = main(grid)
    if np.array_equal(solved, legend):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
