# Combo variant of 58490d8a (index 2)
# combo_spec: D3/1+D5/2
# Name: 水平图例+递增间距

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Original task: key pixels occupy one interior column of the black legend at rows 1,3,5,... and each
#      counted tally grows rightward with fixed stride 2.
#    - D3/1 rotates the legend logic: key pixels instead occupy the top interior row at cols 1,3,5,... and
#      each color's tally grows downward in the same column.
#    - D5/2 replaces the constant stride with cumulative gaps. For tally index j=0,1,2,3,... the offsets are
#      0,2,5,9,..., so the vertical gaps between consecutive dots are 2,3,4,...
#    - Fusion example with key colors [red, blue, green] and counts [4,2,1]:
#      original output rows would be red at cols 1/3/5/7, blue at 1/3, green at 1;
#      D3-only output columns would be red at rows 1/3/5/7, blue at 1/3, green at 1;
#      combo output columns are red at rows 1/3/6/10, blue at 1/3, green at 1.
#      The legend must therefore be read left-to-right, while each matched tally descends with widening gaps.
# b) Complexity calibration:
#    - The rotated top-row legend is already visible when K >= 2, so we keep the original 2..4 key colors.
#    - The growing-gap rule becomes indistinguishable from fixed stride when every count is <= 2, so the
#      generator forces at least one count >= 3 in every sample.
#    - Counts still remain in 1..5 and distractors stay optional, which preserves the original object-counting
#      difficulty instead of adding unrelated axes.
# c) Implicit-assumption audit of the original solver:
#    - "Leftmost non-black column contains the keys" breaks; the solver must now search for the topmost non-black row.
#    - Keys are no longer indexed by row, so the mapping list changes from `(row, color)` to `(col, color)`.
#    - `key_col + 2*i` is invalid because the tally is vertical and non-uniform; it becomes `_tally_row(key_row, i)`.
#    - The legend bbox heuristic still works because the legend remains a single largest black component whose bbox
#      contains non-black pixels.
# d) Placement legality:
#    - The new limiting dimension is legend height, because counts up to 5 produce rows 1,3,6,10,15.
#    - `_legend_shape()` computes the last required tally row exactly and only adds a small slack range so the
#      legend still fits inside the original 20..25 canvas with the required outer margin.
#    - Sprite placement remains legal because the legend is still pasted as one opaque patch and
#      `random_free_location(..., padding=1, padding_connectivity=8)` prevents contact with the legend and peers.
# e) Visual observability:
#    - Every sample has a horizontal key row and at least one tally column with three or more dots, so the solver
#      can visually detect both the 90-degree reorientation and the widening vertical spacing from just a few examples.
#    - The telltale row pattern 1/3/6 (or 1/3/6/10) cannot be confused with the old every-other-row pattern 1/3/5.
# f) Visual cleanliness:
#    - Keys remain separated by one black column each, and each tally occupies only its own column, so different
#      colors never overlap inside the legend.
#    - The legend output stays a sparse black patch, which keeps the counting signal easy to read.
# g) Change inferability:
#    - Colors still act as direct identity links from outside sprites to legend entries; there is no hidden color remap.
#    - The only new reasoning burden is geometric: top-row lookup plus increasing vertical gaps.
# h) New solver assumptions and generator support:
#    - The solver assumes exactly one topmost non-black key row and no other non-black legend pixels below it.
#      `_build_legend()` enforces that structure.
#    - The solver assumes the legend is tall enough for the cumulative vertical positions implied by the true count.
#      `_legend_shape()` uses the same tally-row helper as the renderer, so generation and solving stay aligned.


def _tally_offset(index: int) -> int:
    """Row offset of the zero-based tally dot under growing inter-dot gaps."""
    idx = int(index)
    return idx * (idx + 3) // 2


def _tally_row(key_row: int, index: int) -> int:
    """Absolute row for the zero-based tally dot."""
    return int(key_row) + _tally_offset(index)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and encode the totals in a rotated legend box.

    Input structure:
    - A black rectangular legend patch contains one horizontal row of key-color pixels on its topmost non-black row.
    - Outside the legend, separate monochrome sprites of those key colors are scattered across the background.

    Rule:
    - Count the connected components of each key color outside the legend box.
    - Repaint the cropped legend area in black, then place one dot per counted object in the matching key column,
      starting at the key row and continuing downward with increasing gaps: +2 rows, then +3, then +4, ...

    Key insight:
    - The legend is still both the color index and the output canvas, but now you read the keys left-to-right and
      grow each tally top-to-bottom according to a cumulative spacing sequence rather than a fixed stride.
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

    key_row = None
    for rr in range(bh):
        if (box[int(rr), :] != int(black)).any():
            key_row = int(rr)
            break
    if key_row is None:
        return np.zeros_like(box)

    keys = [(int(cc), int(box[int(key_row), int(cc)])) for cc in range(bw) if int(box[int(key_row), int(cc)]) != int(black)]
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for cc, col in keys:
        mask = (g == int(col)) & outside
        _lab, count = label_mask(mask, connectivity=8)
        for i in range(int(count)):
            rr = _tally_row(key_row, i)
            if 0 <= rr < bh:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (o5x) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
VISIBLE_MAX_COUNT_MIN = 3
KEY_ROW = 1
KEY_COLUMN_FIRST = 1
KEY_COLUMN_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_HEIGHT_SLACK_RANGE = (0, 2)

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
    """Return an inclusive randint from a configured (low, high) pair."""
    return random.randint(bounds[0], bounds[1])


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    """Return connected masks that span all four sides of the sprite bbox."""
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
    """Sample one connected same-color sprite drawn on the background color."""
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
    """Choose background, key colors, and optional distractor color."""
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color


def _sample_counts(k: int) -> list[int]:
    """Sample outside-component counts while forcing a visible growing-gap tally."""
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= VISIBLE_MAX_COUNT_MIN:
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the horizontal-key, vertical-tally layout."""
    last_dot_row = _tally_row(KEY_ROW, max_count - 1)
    bh = last_dot_row + 1 + LEGEND_HEIGHT_TAIL + _randint(LEGEND_HEIGHT_SLACK_RANGE)
    bw = KEY_COLUMN_STEP * k + LEGEND_WIDTH_TAIL
    return bh, bw


def _key_col(index: int) -> int:
    """Map a zero-based key index to its column inside the legend/output box."""
    return KEY_COLUMN_FIRST + KEY_COLUMN_STEP * index


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and counts."""
    bh, bw = _legend_shape(len(key_colors), max(counts))
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        cc = _key_col(i)
        for j in range(count):
            expected[_tally_row(KEY_ROW, j), cc] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with only the top key-row pixels filled."""
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[KEY_ROW, _key_col(i)] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> np.ndarray | None:
    """Sample a grid and paste the legend away from all canvas edges."""
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
    """Create canonical key sprites, optional distractor sprite, and shuffle them."""
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
    """Paste sprites into random non-touching free locations in queue order."""
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
    """Run one output-first generation attempt and reject solver mismatches."""
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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
