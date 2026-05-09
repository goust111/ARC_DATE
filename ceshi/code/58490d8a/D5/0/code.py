# Variant of 58490d8a
# Dimension: D5 — 圆点放置间距 (transform)
# Change: 相邻圆点之间间隔1个背景像素（步长为2），即放置在关键列、关键列+2、关键列+4等位置... → 无间隔连续放置

from __future__ import annotations

import random
import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# Design analysis for the D5 variant
# ----------------------------------
# source_task: none
#
# spec_variant:
#   changed_dimension:
#     name: D5
#     old_behavior: tally dots on each key row are separated by one black cell
#     new_behavior: tally dots occupy consecutive columns with no black gaps
#   concrete_grid_effect:
#     input:
#       - unchanged from the base task: a black legend rectangle contains one colored key
#         pixel per active row at column 1
#       - outside the legend, each key color appears as 1-5 disconnected 8-connected sprites
#       - optional distractor sprites still exist and are ignored by the solver
#     output:
#       - unchanged canvas role: output is the cropped legend box only
#       - changed tally geometry: for a row with count 4, dots now appear at columns
#         1,2,3,4 instead of 1,3,5,7
#   behavior_trace:
#     example:
#       - key row color red, outside red component count = 3
#       - base task writes red at cols [1,3,5]
#       - variant writes red at cols [1,2,3]
#       - difference: the count is the same, only the horizontal step changes
#   complexity_calibration:
#     - keep K in 2-4 colors and counts in 1-5 so the puzzle still requires counting
#     - force at least one count >= 2 in every generated instance; otherwise the new step
#       is invisible because a single dot looks identical in both versions
#     - preserve optional distractors so color filtering remains necessary
#   old-code-assumption-audit:
#     - legend detection via "largest black component whose bbox contains non-black pixels"
#       remains valid because the input legend shape is unchanged
#     - key column detection via "leftmost non-black column in the legend bbox" remains valid
#     - only the dot-placement formula `key_col + 2*i` depends on the old D5 setting; it must
#       become `key_col + i`
#     - the legend width formula also depended on the old step indirectly; it must be recomputed
#       from the new last-used column so the right-side black tail still exists
#   placement_legality:
#     - sprites are still placed with 8-connected padding=1, so no sprite can touch another
#       sprite or the legend
#     - the narrower legend only reduces occupied area; it does not introduce new collisions
#   visual_observability:
#     - contiguous runs of same-color dots are easy to distinguish from alternating-dot rows
#     - forcing at least one multi-dot row guarantees the D5 change is visible in every sample
#   cleanliness:
#     - only one symbolic region (the legend box) is used for output
#     - outside sprites remain separated and do not overlap the legend or each other
#   new_solver_assumptions_to_satisfy:
#     - each key color's outside objects stay disconnected under 8-connectivity
#     - the legend still contains exactly one active color pixel per key row in the key column
#     - the legend width must be at least one black column wider than the longest contiguous tally


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored sprites and encode the totals inside the legend box.

    Input structure:
    - A solid black legend rectangle contains one colored key pixel per active row in its leftmost non-black column.
    - Outside that rectangle, disconnected sprites of those key colors are scattered on the background.

    Rule:
    - For each key color, count the number of same-colored connected components outside the legend.
    - Repaint the cropped legend box as black, then place that many dots on the matching key row in consecutive columns starting at the key column.

    Key insight: The legend identifies which colors to count, and the answer is a compact left-to-right run whose length equals each color's outside component count.
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
    keys.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, col in keys:
        mask = (g == int(col)) & outside
        _lab, count = label_mask(mask, connectivity=8)
        for i in range(int(count)):
            cc = int(key_col + i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
TALLY_COL_STEP = 1
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
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
        if max(counts) >= 2:
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    min_bw = KEY_COLUMN + TALLY_COL_STEP * (max_count - 1) + 1 + LEGEND_WIDTH_TAIL
    bw = min_bw + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
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
    raise RuntimeError("failed to generate a valid 58490d8a D5 input")
