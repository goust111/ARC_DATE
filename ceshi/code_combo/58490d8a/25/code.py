# Combo variant of 58490d8a (index 25)
# combo_spec: D4/0+D10/0+D14/0
# Name: 紧凑排序直方图

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for combo D4/0 + D10/0 + D14/0
#
# a) Concrete behavior and trace
#    - The input still has one black legend rectangle and several disconnected
#      monochrome sprites outside it.
#    - D10 compresses the key rows vertically: instead of rows 1,3,5,... the
#      colored key pixels occupy rows 1,2,3,... inside the box.
#    - D4 changes the rendering primitive: a count n becomes a contiguous bar
#      of length n starting at the key column, not spaced dots.
#    - D14 inserts a reordering step after counting: rows in the output are
#      reassigned from highest count to lowest count.
#    - Example with compact key rows [blue, red, yellow] at legend rows [1,2,3]
#      and outside counts [2,5,3]:
#         original task output on those rows:
#           row 1 -> blue dots at 1,3
#           row 2 -> red dots at 1,3,5,7,9
#           row 3 -> yellow dots at 1,3,5
#         fused output:
#           row 1 -> red bar    at 1..5
#           row 2 -> yellow bar at 1..3
#           row 3 -> blue bar   at 1..2
#      The fusion is not additive noise: counting feeds sorting, and sorting
#      chooses which compact slot receives each continuous histogram bar.
#
# b) Complexity calibration
#    - If counts were tied, D14 would become visually weak, so generator uses
#      distinct counts.
#    - If all counts were 1, D4 would collapse into single pixels, so distinct
#      counts across K>=2 also guarantees at least one bar length > 1.
#    - If the input order were already descending by count, D14 would be hidden,
#      so generator rejects that arrangement.
#    - K stays in 2..4 and counts stay in 1..5, preserving the original task's
#      scale while making the fusion clearly visible in a few examples.
#
# c) Implicit assumption audit for main()
#    - Legend detection as the largest black connected component with some
#      non-black pixels in its bbox still holds because compact key rows do not
#      disconnect the black frame/interior.
#    - Key-column detection as "leftmost non-black column in the legend" still
#      holds; neither sorting nor bar drawing changes the input encoding.
#    - The old assumption "output row == input key row" is broken by D14, so the
#      solver must first record counts, then sort records, then redraw onto the
#      observed slot rows.
#    - The old assumption "tally marks are spaced by two columns" is broken by
#      D4, so the redraw stage must fill slices `[key_col:key_col+count)`.
#    - Enumerating all non-black rows in the key column remains valid for D10
#      because the compact row positions are read from the input rather than
#      inferred from a hard-coded odd-row pattern.
#
# d) Placement legality
#    - The legend width is sampled as `KEY_COLUMN + max_count + 1 + slack`, so
#      every contiguous bar fits fully inside the box.
#    - The compact legend height is `KEY_ROW_FIRST + K + LEGEND_HEIGHT_TAIL`,
#      which leaves one black row above and one below the consecutive key rows.
#    - Sprites are pasted with one-cell 8-connected padding, so copies of the
#      same color remain separate components and do not touch the legend.
#
# e) Visual observability
#    - Humans can see three concurrent cues: consecutive key rows, descending
#      bar lengths, and the fact that colors move away from their input rows.
#    - Sorting uses only visible component counts, never raw color ids or other
#      hidden numeric properties.
#    - Distinct counts make the descending staircase unambiguous from 2-4 I/O
#      examples.
#
# f) Visual cleanliness
#    - The output occupies only the legend rectangle, on a black canvas, with
#      one bar per row. No overlays or competing marker systems are introduced.
#    - Compact rows reduce empty space vertically but keep enough structure to
#      distinguish each bar because K is capped at 4.
#
# g) Change inferability
#    - Colors are preserved, not remapped. A color's bar simply moves to the row
#      determined by its count rank.
#    - Because the same colors also appear on the counted outside sprites, the
#      solver can infer the full chain "identify color -> count sprites -> rank
#      by count -> draw continuous bar" from examples alone.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites and rewrite the legend as a compact sorted histogram.

    Input structure:
    - A black rectangular legend box contains one colored key pixel per consecutive active row in its leftmost non-black column.
    - Outside that box, disconnected monochrome sprites of the key colors are scattered on the background, with an optional distractor color.

    Rule:
    - For each key color, count the connected components of that color outside the legend box.
    - Sort the key colors by count from largest to smallest, breaking ties by original top-to-bottom key order, then draw each result as a contiguous horizontal bar starting at the key column on the ranked compact row.

    Key insight: The legend rows are only measurement slots in the input; after counting, those same compact slots become ranking positions, and the tally shape changes from spaced dots to solid bars.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    best = None  # (area, bbox)
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(g, bbox)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, bbox)))
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
    keys.sort(key=lambda t: int(t[0]))
    slot_rows = [int(rr) for rr, _col in keys]

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    records: list[tuple[int, int, int]] = []
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(rr), int(col), int(count)))

    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    for new_idx, (_orig_rr, col, count) in enumerate(records):
        rr = int(slot_rows[new_idx])
        out[int(rr), int(key_col) : int(min(bw, key_col + count))] = int(col)

    return out


COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 1
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
        _, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if n == 1:
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
    bh = KEY_ROW_FIRST + k + LEGEND_HEIGHT_TAIL
    bw = KEY_COLUMN + max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _sorted_key_records(key_colors: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    records = [(int(i), int(color), int(count)) for i, (color, count) in enumerate(zip(key_colors, counts))]
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return records


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for out_i, (_orig_i, color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        rr = _key_row(out_i)
        expected[rr, KEY_COLUMN : KEY_COLUMN + count] = int(color)
    return expected


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


def _compact_key_rows(shape: tuple[int, int], key_colors: list[int]) -> list[int]:
    legend = _build_legend(shape, key_colors)
    return [int(rr) for rr in range(legend.shape[0]) if int(legend[rr, KEY_COLUMN]) != int(BLACK)]


def _gen_attempt() -> np.ndarray | None:
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
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

    input_rows = _compact_key_rows(expected.shape, key_colors)
    output_colors = [int(solved[rr, KEY_COLUMN]) for rr in input_rows]
    if output_colors == [int(c) for c in key_colors]:
        return None
    if max(counts) <= 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
