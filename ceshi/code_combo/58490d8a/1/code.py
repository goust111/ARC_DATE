# Combo variant of 58490d8a (index 1)
# combo_spec: D4/0+D14/0
# Name: 排序后条形填充

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant (D4/0 + D14/0)
#
# a) Concrete behavior and trace
#    - The input still has one black legend rectangle plus scattered sprites
#      outside it. Inside the legend, the leftmost non-black column contains one
#      key pixel on every other row.
#    - Original task:
#        key rows stay in place;
#        count n becomes dots at key_col, key_col+2, key_col+4, ...
#    - Combo task:
#        Step 1: count outside connected components for each key color.
#        Step 2: sort the key colors by count descending.
#        Step 3: redraw the legend rows in that ranked order, and render each
#                tally as a solid bar from key_col through key_col+n-1.
#    - Example with input legend rows [blue, red, yellow] and counts [2, 5, 3]:
#        original output rows = [blue dots x2, red dots x5, yellow dots x3]
#        combo output rows    = [red bar len5, yellow bar len3, blue bar len2]
#      So both the row assignment and the within-row geometry change.
#
# b) Complexity calibration
#    - If the counts were already descending, the sorting stage would be
#      invisible; the generator rejects that.
#    - If every count were 1, bars would collapse to single pixels and the D4
#      change would be invisible; the generator uses distinct counts sampled
#      from 1..5, which guarantees at least one count >= 2 for K >= 2.
#    - Keeping K in 2..4 and counts in 1..5 preserves the original search space
#      while making the fused rule visually apparent.
#
# c) Implicit-assumption audit
#    - Old assumption: output row == input key row. Broken by D14, so main must
#      first collect (original_row, color, count), sort those records, then draw
#      onto the fixed slot rows in ranked order.
#    - Old assumption: tally cells are separated by one black column. Broken by
#      D4, so drawing must use a contiguous slice instead of key_col + 2*i.
#    - Legend detection as "largest black CC whose bbox contains non-black
#      pixels", key-column detection as the leftmost non-black column, and
#      outside-object counting with 8-connectivity still remain valid.
#    - The sorted redraw assumes a deterministic tie break. main uses original
#      top-to-bottom legend order; generate_input mostly avoids ties so the human
#      can infer the primary descending-count rule.
#
# d) Placement legality
#    - Sprites are still generated as connected 2x2..3x3 monochrome shapes and
#      placed with one-cell 8-neighborhood padding, so separate copies do not
#      merge into one component and never touch the legend.
#    - The legend width is resized for bars: it must fit key_col + max_count
#      cells plus a right tail/slack. This replaces the old spaced-dot width.
#    - The legend is pasted before sprites and kept away from canvas edges, so
#      every sampled placement stays in-bounds.
#
# e) Visual observability
#    - The ranking key is fully visible: number of same-color outside sprites.
#    - The rendering key is also visible: counted quantity becomes horizontal
#      run length, not isolated dots.
#    - Because counts are distinct and not already descending, examples show a
#      nontrivial row permutation and at least one multi-cell bar.
#
# f) Visual cleanliness
#    - Output remains a sparse black legend panel with one active row every two
#      rows, so the reordered bars do not overlap and are easy to parse.
#    - Outside sprites remain separated by padding, which keeps counting by eye
#      manageable even after adding the sorting step.
#
# g) Change inferability
#    - No hidden numeric feature such as raw color id participates in sorting.
#      Colors only identify which outside objects to count.
#    - A color keeps its identity when moved to a new ranked row, so the legend
#      itself explains the correspondence between outside sprites and output bar.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count legend colors, rank them by frequency, and rewrite the legend as bars.

    Input structure:
    - A black rectangular legend patch contains one colored key pixel on each active row in its leftmost non-black column.
    - Outside that patch, disconnected monochrome sprites of the key colors are scattered across the background.

    Rule:
    - Count how many outside connected components belong to each key color.
    - Sort the key colors by count from largest to smallest, breaking ties by their original top-to-bottom legend order, then redraw the legend rows in that ranked order using contiguous same-color bars of length n starting at the key column.

    Key insight:
    - The legend rows are measurement slots first and output slots second: first recover each color's count from the outside sprites, then reuse the fixed row positions as rank positions and encode each count as an unbroken horizontal run.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    best = None  # (area, bbox)
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    keys.sort(key=lambda item: int(item[0]))
    slot_rows = [int(rr) for rr, _col in keys]

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    records: list[tuple[int, int, int]] = []
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(rr), int(col), int(count)))

    records.sort(key=lambda item: (-int(item[2]), int(item[0])))
    for new_idx, (_orig_rr, col, count) in enumerate(records):
        rr = int(slot_rows[new_idx])
        end = min(int(bw), int(key_col + count))
        out[int(rr), int(key_col) : end] = int(col)

    return out


COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 3)

GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 300


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    masks = []
    for cells in combinations(range(h * w), area):
        mask = np.zeros((h, w), dtype=bool)
        for cell in cells:
            mask[cell // w, cell % w] = True
        spans_edges = mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()
        if not spans_edges:
            continue
        _, count = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if count == 1:
            masks.append(mask)
    return tuple(masks)


def _make_sprite(color: int, bg: int) -> np.ndarray:
    area = _randint(SPRITE_AREA_RANGE)
    dims = []
    for h, w in SPRITE_BBOXES:
        if h * w >= area and _valid_masks(h, w, area):
            dims.append((h, w))
    h, w = random.choice(dims)
    mask = random.choice(_valid_masks(h, w, area))
    sprite = np.full((h, w), int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _sample_palette() -> tuple[int, list[int], int | None]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color


def _sample_counts(k: int) -> list[int]:
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    counts = random.sample(choices, k)
    while counts == sorted(counts, reverse=True):
        random.shuffle(counts)
    return [int(v) for v in counts]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = KEY_COLUMN + max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _sorted_key_records(key_colors: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    records = [(int(i), int(color), int(count)) for i, (color, count) in enumerate(zip(key_colors, counts))]
    records.sort(key=lambda item: (-int(item[2]), int(item[0])))
    return records


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for out_i, (_orig_i, color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        rr = _key_row(out_i)
        expected[rr, KEY_COLUMN : KEY_COLUMN + count] = int(color)
    return expected


def _build_unsorted_bar_output(shape: tuple[int, int], key_colors: list[int], counts: list[int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        out[rr, KEY_COLUMN : KEY_COLUMN + count] = int(color)
    return out


def _build_sorted_dot_output(shape: tuple[int, int], key_colors: list[int], counts: list[int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for out_i, (_orig_i, color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        rr = _key_row(out_i)
        for j in range(count):
            cc = KEY_COLUMN + 2 * j
            if cc < shape[1]:
                out[rr, cc] = int(color)
    return out


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
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


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for color, count in zip(key_colors, counts):
        sprite = _make_sprite(color, bg)
        for _ in range(count):
            sprites.append(sprite.copy())
    if distractor_color is not None:
        sprites.append(_make_sprite(distractor_color, bg))
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
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)

    unsorted_bar = _build_unsorted_bar_output(expected.shape, key_colors, counts)
    sorted_dots = _build_sorted_dot_output(expected.shape, key_colors, counts)
    if np.array_equal(expected, unsorted_bar):
        return None
    if np.array_equal(expected, sorted_dots):
        return None

    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, bg)
    try:
        _place_sprites(grid, sprites, bg)
    except ValueError:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
