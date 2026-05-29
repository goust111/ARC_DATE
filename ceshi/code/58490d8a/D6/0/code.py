# Variant of 58490d8a
# Dimension: D6 — 图例关键列位置 (symbol_instructions)
# Change: 关键色像素固定放在图例框内的第2列（index=1），是框内最左侧的非黑列... → 关键色放在最右列

"""
Design analysis for the D6 variant.

a) Concrete behavior and trace
spec:
  preserved:
    - The input still contains one black legend rectangle plus scattered same-color sprites outside it.
    - Each key color still appears once inside the legend and as 1-5 external 8-connected components.
    - The output is still a black canvas with one tally row per key color.
  changed:
    - The key pixels are no longer in the second column. They are placed in the legend's rightmost column.
    - The tally does not grow rightward anymore. It starts at that right-edge key pixel and advances left by 2 cells.

Small trace:
  original:
    - If a legend row had key color red at column 1 and there were 3 red components outside,
      the output row was red at columns 1, 3, 5.
  variant:
    - If the legend width is 8, the red key pixel sits at column 7.
    - With 3 red components, the output row is red at columns 7, 5, 3.
  difference:
    - The same counting subproblem remains, but both the anchor side and the drawing direction flip.

b) Complexity calibration
  - Keep 2-4 key rows and 1-5 objects per key color so the task still has repeated evidence.
  - Force at least one sampled count to be >= 2 so the leftward growth is visible in the output, not just the right-edge anchor.
  - Keep one optional distractor color only; more distractors would add clutter without helping the D6 idea.

c) Hidden-assumption audit of the original main()
  - Legend detection by "largest black CC whose bbox contains non-black pixels" still works.
    The right-edge key pixels punch holes in the box boundary, but the black region remains a single large CC.
  - The original scan `for cc in range(bw)` assumed the key column was the first non-black column. That is false now.
    It must scan from the right.
  - The original drawing step `key_col + 2*i` assumed the tally extends rightward. That is false now.
    It must become `key_col - 2*i`.
  - Returning `np.zeros_like(box)` when no key column exists remains safe because black is still 0.

d) Placement legality
  - The legend is pasted first onto a background-colored grid.
  - Sprites are still placed with `random_free_location(..., padding=1, padding_connectivity=8)`,
    so they cannot overlap or even touch each other or the legend box.
  - Because the key column is on the right edge, the legend width formula changes to `2*max_count + slack`.
    This guarantees every leftward tally fits: with `max_count=m`, columns `bw-1, bw-3, ..., bw-(2m-1)` stay in-bounds.

e) Visual observability
  - The input already shows that the colored legend markers moved from the left side to the right boundary.
  - The output shows dots propagating left from that boundary.
  - Requiring at least one count >= 2 ensures the reversed direction is visible in most rows or at least one row.

f) Visual cleanliness
  - The layout remains sparse: one legend area, a few small external sprites, at most one distractor.
  - Key rows stay on alternating odd rows, so tallies do not visually collide.
  - Extra width slack is added on the left side only, which keeps the "anchor on the right edge" easy to read.

g) Color inference
  - No color remapping is introduced.
  - Colors still serve only as direct identity links between legend row and external components, which is easy to infer.
"""

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count exterior colored objects and write the tallies into a right-anchored legend.

    Input structure:
    - A black rectangular legend box contains one colored key pixel per active row, and those key pixels sit in the box's rightmost column.
    - Outside the box, separated same-colored sprites of the key colors are scattered on the background.

    Rule:
    - For each key color, count the 8-connected components of that color outside the legend box.
    - Output a black box of the same size where each key row shows one dot per counted component, starting at the right-edge key column and stepping left every other cell.

    Key insight:
    - The legend pixels still specify which colors to count and which rows to use, but their extreme-right placement also tells you the tally direction has flipped and must grow leftward.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
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

    # STEP 2) Inside the legend bbox, find the key column:
    # rightmost column with any non-black pixels (the key pixels).
    key_col = None
    for cc in range(bw - 1, -1, -1):
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

    # STEP 3/4) Count CCs of each key color outside the legend, then draw 1 dot per CC.
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for i in range(int(nc)):
            cc = int(key_col - 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (o5x) ---------------

# Palette and legend color choices.
COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

# Key rows and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_LEFT_SLACK_RANGE = (0, 3)

# Canvas size and legend placement bounds.
GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

# Sprite shape and placement constraints.
SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

# Retry budget for rejection sampling.
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
    """Sample target counts and force at least one multi-dot row for visible directionality."""
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= 2:
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest leftward tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + _randint(LEGEND_LEFT_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _key_col(width: int) -> int:
    """The key column is always the legend's rightmost column."""
    return int(width - 1)


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)
    key_col = _key_col(bw)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, key_col - TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with only the rightmost key-column pixels filled."""
    legend = np.full(shape, BLACK, dtype=int)
    key_col = _key_col(shape[1])
    for i, color in enumerate(key_colors):
        legend[_key_row(i), key_col] = int(color)
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
    raise RuntimeError("failed to generate a valid 58490d8a input")
