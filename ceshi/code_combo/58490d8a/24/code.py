# Combo variant of 58490d8a (index 24)
# combo_spec: D3/1+D4/0+D5/0
# Name: 垂直柱状图

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


# Design analysis for combo D3/1 + D4/0 + D5/0
# a) Concrete behavior and trace:
#    - Base task: inside the black legend box, key pixels occupy one vertical key column at
#      rows 1,3,5,... and each row encodes a spaced horizontal dot sequence:
#      count 3 -> columns [1,3,5].
#    - D3/1 rotates the legend logic: key pixels move to one horizontal key row at
#      columns 1,3,5,... and each key would now own a vertical tally direction.
#    - D4/0 turns a dot sequence into a solid fill segment, and D5/0 removes the
#      one-cell gaps. After rotation, these combine into a continuous vertical bar.
#    - Example with key colors [red, blue] and counts [3,1]:
#      base output:
#        row red  -> red at (1,1),(1,3),(1,5)
#        row blue -> blue at (3,1)
#      rotated-only output:
#        red column at rows (1,1),(3,1),(5,1); blue only at (1,3)
#      fused output:
#        red column fills rows 1..3 at col 1; blue occupies row 1 at col 3.
#      So the count target is unchanged, but the rendered tally is now a top-down
#      compact column whose height equals the number of outside components.
# b) Complexity calibration:
#    - Keep key-count, palette, sprite geometry, and distractor rules from the source task.
#    - Preserve column spacing of 2 between neighboring key colors so adjacent bars do not
#      merge visually once they become solid vertical columns.
#    - Force at least one count >= 2 in every generated sample; otherwise the fused variant
#      could visually collapse into a mere top-row key display with no visible bar growth.
# c) Implicit-assumption audit:
#    - Legend detection as "largest black component whose bbox contains non-black pixels"
#      still holds because the legend remains one black rectangle with colored holes.
#    - The old solver assumption "leftmost non-black column is the key index" breaks;
#      we must instead use the topmost non-black row.
#    - Key identities are no longer stored as (row,color); they become (col,color).
#    - The old tally formula `key_col + 2*i` is invalid twice over: the growth axis changes
#      from columns to rows, and the spacing changes from step 2 to step 1.
#    - The output height formula must now fit a contiguous vertical bar starting at row 1,
#      while the width formula must still fit 2-4 horizontally spaced key columns.
# d) Placement legality:
#    - Generator keeps output-first construction: sample counts, build the expected legend
#      crop, derive the input legend by keeping only the top key row, then place sprites.
#    - Legend size uses:
#        width  = 2*K + 1
#        height = key_row + max_count + 1 + slack
#      which guarantees each bar fits with at least one black row below the tallest bar.
#    - Sprite placement still uses `random_free_location(..., padding=1, connectivity=8)`,
#      so sprites never touch each other or the legend box.
# e) Visual observability:
#    - Human solvers can see two simultaneous changes: keys are arranged left-to-right on
#      the top row, and the answers extend downward as solid columns rather than dotted rows.
#    - Requiring at least one height > 1 ensures the vertical-bar rule appears explicitly.
# f) Visual cleanliness:
#    - Only one semantic region exists inside the legend: the top row for keys and the
#      columns beneath them for tallies. Empty black columns remain between bars.
#    - The background region still contains separated sprites only, with no overlapping marks.
# g) Change inferability:
#    - No color remapping is introduced. A color seen in a key pixel matches the outside
#      sprites to count and the output bar to draw.
#    - Because each sample contains 2-4 key colors and repeated sprite evidence, the
#      color-to-column correspondence is directly visible without relying on color ids.
# h) New solver assumptions to guarantee in generation:
#    - Exactly one legend row contains non-black pixels, and it is the top key row.
#    - Every key color appears in a unique column of that row.
#    - The sampled height is sufficient for every contiguous bar, and at least one bar has
#      length >= 2 so the fused behavior differs from the rotated dotted variant.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored sprites and rewrite the legend as vertical bars.

    Input structure:
    - A black rectangular legend box contains one horizontal row of key-color pixels in its topmost non-black row.
    - Outside that box, disconnected sprites of those key colors are scattered on the background, plus an optional distractor color.

    Rule:
    - For each key color, count the number of same-colored connected components outside the legend box.
    - Output the cropped legend box repainted in black, then fill a same-colored vertical bar downward from the matching key pixel, with bar height equal to that count.

    Key insight:
    - The legend still serves as both the color index and the answer canvas, but it must now be read left-to-right across the top row while each tally grows top-to-bottom as one continuous column.
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
    keys.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for cc, col in keys:
        mask = (g == int(col)) & outside
        _lab, count = label_mask(mask, connectivity=8)
        end = min(int(bh), int(key_row + count))
        out[int(key_row) : end, int(cc)] = int(col)

    return out


COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW = 1
KEY_COLUMN_FIRST = 1
KEY_COLUMN_STEP = 2
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
    bh = KEY_ROW + max_count + LEGEND_HEIGHT_TAIL + _randint(LEGEND_SLACK_RANGE)
    bw = KEY_COLUMN_STEP * k + LEGEND_WIDTH_TAIL
    return bh, bw


def _key_col(i: int) -> int:
    return KEY_COLUMN_FIRST + KEY_COLUMN_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        cc = _key_col(i)
        expected[KEY_ROW : KEY_ROW + count, cc] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[KEY_ROW, _key_col(i)] = int(color)
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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
