# Variant of 58490d8a
# Dimension: D4 — 变换操作：计数到圆点序列 (transform)
# Change: 每种关键色的框外连通分量数量n转换为n个同色圆点，从关键列开始每隔一列放置一个... → 圆点替换为条形填充

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


"""
Design analysis for the D4 variant

a) Concrete behavior and trace
   - Input still contains one black legend rectangle and several same-color sprites outside it.
   - The leftmost non-black column inside the box is still the key column, with one colored key pixel on every other row.
   - The only behavioral change is inside the output box:
     original: count n -> place pixels at key_col, key_col+2, key_col+4, ...
     variant : count n -> fill key_col..key_col+n-1 continuously
   - Small trace with key_col=1 and count=3:
     original row: [0, C, 0, C, 0, C, 0]
     variant row : [0, C, C, C, 0, 0, 0]
     The counting stage is unchanged; only the rendering stage changes from spaced dots to a solid bar.

b) Complexity calibration
   - If every count were 1, the variant would visually collapse to the original because a length-1 bar equals a single dot.
   - To keep D4 observable, generation guarantees at least one key color has count >= 2 in every sample.
   - Key count, sprite count, palette size, and object geometry remain in the original ranges, so the task does not gain unrelated freedom.

c) Audit of old assumptions that must change
   - Old solver line `cc = key_col + 2 * i` assumed fixed gaps between tally marks. This is invalid for the bar variant.
   - Old legend-width formula `2*max_count + tail + slack` only existed to fit spaced dots. The variant instead needs width for a contiguous run from the key column.
   - Old expected-output builder wrote one isolated pixel per count. It must now write a contiguous slice.
   - Black-box detection, key-column detection, outside masking, and connected-component counting still rely on assumptions that remain true.

d) Placement legality
   - Outside sprites are still placed by `random_free_location` with padding=1 and 8-connectivity, so they do not touch each other or the legend.
   - The legend box is still placed away from edges using the same margin logic.
   - The new bar stays entirely inside the legend because width is sampled from `KEY_COLUMN + max_count + 1 + slack`.

e) Visual observability
   - Human-solvable contrast comes from adjacency: the second counted cell is now immediately next to the key pixel rather than separated by black.
   - Enforcing at least one count >= 2 ensures every generated example exhibits this adjacency cue.

f) Visual cleanliness
   - Bars live on the same sparse odd rows as before, with empty black rows between them.
   - No extra overlays or new marker types are introduced, so the legend region remains easy to parse.

g) Color inference
   - No new color mapping is introduced. Each bar keeps the same color as its key pixel and corresponding outside sprites.
   - Distractor colors remain outside the key set and never appear in the legend output.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside objects and rewrite the legend as horizontal bars.

    Input structure:
    - A black rectangular legend area contains one key-color pixel on each active row in its leftmost non-black column.
    - Outside that box, disconnected sprites of the key colors are scattered across the background.

    Rule:
    - For each key color, count the number of same-colored connected components outside the legend box.
    - Output a black canvas with the legend's shape, then fill a contiguous same-color bar of length n starting at the key column, where n is that count.

    Key insight: The legend rows still encode color-to-count correspondence, but the tally is shown as a solid run from the key position instead of separated dots.
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
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labels, nc = label_mask(mask, connectivity=8)
        end = min(int(bw), int(key_col + nc))
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
    counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
    if max(counts) == 1:
        counts[random.randrange(k)] = random.randint(2, TARGET_COUNT_RANGE[1])
    return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = KEY_COLUMN + max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
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

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a D4 variant input")
