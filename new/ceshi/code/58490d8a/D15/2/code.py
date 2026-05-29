# Variant of 58490d8a
# Dimension: D15 — 圆点颜色 (transform)
# Change: 每行圆点的颜色与该行关键色相同... → 圆点颜色交替排列

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# Design analysis for the D15 variant.
#
# 1) Concreteized behavior:
#    The original task turns object counts into same-colored tally dots on each key row.
#    This variant changes only the tally coloring rule: the first counted slot stays in the
#    key color, the second counted slot becomes black, the third returns to the key color,
#    and so on. Because black dots would disappear on an all-black canvas, the generator
#    adds a single neutral "peg" color on every potential tally slot to the right of each
#    key pixel. Input rows therefore look like:
#      [key, peg, peg, peg, ...] on columns 1,3,5,7,...
#    and output rows recolor only the counted prefix:
#      count=4 -> [key, black, key, black, peg, peg, ...]
#
# 2) Local trace of the changed stage:
#    Original row logic for a red key with 4 outside objects:
#      input legend row:  [red, ., ., ., ...]
#      output row:        [red, red, red, red, ...] on every other column
#    Variant row logic:
#      input legend row:  [red, gray, gray, gray, ...]
#      output row:        [red, black, red, black, ...]
#    Counting itself is unchanged; only the rendering of the counted prefix differs.
#
# 3) Complexity calibration:
#    The peg color is a single repeated neutral color, distinct from background, keys,
#    and distractors, so the alternation is visible but not combinatorially noisy.
#    The generator enforces at least one row with count >= 3 so examples visibly contain
#    both color states of the alternating rule instead of collapsing to a single key pixel.
#
# 4) Solver assumptions reviewed against the variant:
#    - Legend detection still relies on the largest black connected component whose bbox
#      contains non-black cells. This remains valid because the legend is still a black
#      rectangle with internal non-black holes.
#    - The leftmost non-black legend column must still be the key column. Pegs therefore
#      start at key_col + 2, never earlier.
#    - Counting outside connected components by key color is unchanged and still safe.
#    - Rendering can no longer start from an all-black canvas if black tally slots must
#      be visible, so the solver first restores the neutral peg rail and then overwrites
#      the counted prefix with alternating key/black cells.
#
# 5) Placement legality and cleanliness:
#    Outside sprites are unchanged: they remain non-touching 8-connected components placed
#    away from the legend. The only extra structure lives inside the legend bbox, so no new
#    placement collisions are introduced.
#
# 6) Observability for human solvers:
#    The neutral peg rail makes black tally slots visibly different from empty background.
#    Across 2-4 examples, rows with counts >= 3 show a repeated key/black/key pattern,
#    which is a directly observable transform rather than a hidden color-id convention.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and repaint a legend row with alternating tally colors.

    Input structure:
    - A black rectangular legend patch contains one key-color pixel on each key row.
    - On those same rows, the remaining tally slots are pre-marked by a repeated neutral peg color.
    - Outside the legend, separate same-colored sprites for each key are scattered on the background.

    Rule:
    - For each key color, count its connected components outside the legend box.
    - Rebuild the legend-sized output: keep the peg rail, then recolor the first counted slots on that row,
      alternating key color, black, key color, black, ... from left to right.

    Key insight: The legend still indexes colors and rows, but the tally prefix is no longer monochrome —
    the visible peg rail turns black into an intentional second tally color instead of invisible background.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    best = None
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
    keys.sort(key=lambda t: int(t[0]))

    slot_samples = []
    for rr, _col in keys:
        for cc in range(key_col + 2, bw, 2):
            val = int(box[int(rr), int(cc)])
            if val != int(black):
                slot_samples.append(val)
    slot_color = int(Counter(slot_samples).most_common(1)[0][0]) if slot_samples else int(black)

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    out = np.zeros((bh, bw), dtype=int)
    if slot_color != int(black):
        for rr, _col in keys:
            for cc in range(key_col + 2, bw, 2):
                out[int(rr), int(cc)] = int(slot_color)

    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for i in range(int(nc)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col) if i % 2 == 0 else int(black)

    return out


COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
TALLY_COL_STEP = 2
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
MIN_VISIBLE_ALTERNATION_COUNT = 3


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


def _sample_palette() -> tuple[int, list[int], int | None, int]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]

    slot_color = random.choice(remaining)
    remaining = [c for c in remaining if c != slot_color]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color, slot_color


def _sample_counts(k: int) -> list[int]:
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= MIN_VISIBLE_ALTERNATION_COUNT:
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int], slot_color: int) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for cc in range(KEY_COLUMN + TALLY_COL_STEP, bw, TALLY_COL_STEP):
            expected[rr, cc] = int(slot_color)
        for j in range(count):
            cc = KEY_COLUMN + TALLY_COL_STEP * j
            expected[rr, cc] = int(color) if j % 2 == 0 else BLACK
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int], slot_color: int) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        rr = _key_row(i)
        legend[rr, KEY_COLUMN] = int(color)
        for cc in range(KEY_COLUMN + TALLY_COL_STEP, shape[1], TALLY_COL_STEP):
            legend[rr, cc] = int(slot_color)
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
    bg, key_colors, distractor_color, slot_color = _sample_palette()
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(key_colors, counts, slot_color)
    legend = _build_legend(expected.shape, key_colors, slot_color)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, bg)
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
    raise RuntimeError("failed to generate a valid 58490d8a D15 input")
