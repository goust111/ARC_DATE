# Combo variant of 58490d8a (index 3)
# combo_spec: D1/1+D11/2
# Name: 尺寸匹配+最大面积计数

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    get_bg,
    label_mask,
    paste,
    random_free_location,
)

# ---------------------------------------------------------------------------
# Design analysis for the fused size-match + maximum-area variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - The black legend box keeps its original role, but each key row now begins with a
#   short horizontal bar rather than a single key pixel. The bar length is not a tally:
#   it names an area bucket.
# - We use three visible buckets. Each bucket covers several adjacent areas so that
#   "size matching" and "take the maximum" are both active:
#       bar length 2 -> sprite areas 4-6
#       bar length 3 -> sprite areas 7-9
#       bar length 4 -> sprite areas 10-12
# - Outside the box, monochrome sprites are still isolated 8-connected components.
#   Matching ignores outside colors completely; each sprite is assigned only by its
#   measured pixel area falling into one of the legend buckets.
# - After grouping by bucket, the solver does not count how many sprites are in the
#   group. Instead it finds the largest actual area inside that bucket and writes that
#   many dots on every legend row carrying the same bar length.
# - Small trace:
#   original task: a red key pixel with 3 red components outside -> 3 red dots.
#   naive literal D1+D11 fusion: a 5-cell bar would match only area-5 sprites, so the
#   "largest area in that group" would still be 5 and the max step would do nothing.
#   fused task here: a 3-cell bar means bucket 7-9. If the outside bucket members have
#   areas [7, 9, 8], the output row gets 9 dots, not 3 dots and not "one dot per
#   matching sprite".
#
# b) Complexity calibration
# - The bar-length vocabulary is only three values, so humans can infer the category
#   system from 2-4 examples.
# - Outside sprites all live in the same 4x4 bbox family. That suppresses cheap bbox
#   matching and forces the intended "count filled cells" reasoning.
# - Every used bucket is generated with at least one smaller member and one larger
#   member, so the max-area step changes the answer and never collapses to a simple
#   presence test.
#
# c) Audit of original main() assumptions that break under the fusion
# - Original assumption: the key is one pixel and its color identifies the target
#   outside group. Broken. We must read a same-color horizontal run to recover the
#   area bucket, and the key-row color now serves only as output paint.
# - Original assumption: each same-color outside component contributes 1 tally dot.
#   Broken. We must aggregate all outside components across colors, map them into
#   buckets by measured area, then keep only the maximum area per bucket.
# - Original assumption: legend width only needs to fit component counts up to 5.
#   Broken. The output now uses actual component areas up to 12, so width must scale
#   with the maximum reported area instead.
#
# d) Placement legality
# - Legend bars are horizontal runs anchored at the same key column, with at least one
#   trailing black cell, so the legend's black region stays as one large 8-connected
#   component and remains easy for the solver to locate.
# - Outside sprites are placed with 8-neighborhood padding from each other and from the
#   legend. Because of that spacing, labeling the combined non-background outside mask
#   is safe even when different sprite colors are reused.
#
# e) Visual observability
# - The task differs visibly from the original in two ways at once: the legend now uses
#   short bars instead of single pixels, and outside colors no longer line up with
#   legend colors.
# - It also differs visibly from pure size-counting: within one bar category, several
#   sprites can appear, but the output dot count matches the largest measured area, not
#   the number of sprites.
#
# f) Visual cleanliness
# - The legend interior contains only one colored bar per odd row and otherwise stays
#   black, preserving the original "clean tally box" aesthetic.
# - All outside sprites are compact 4x4 monochrome components with enforced padding, so
#   the grid remains readable even when multiple buckets and colors are present.
#
# g) Change inferability
# - No hidden ordering on color ids is used anywhere.
# - Legend colors and sprite colors are sampled from disjoint palettes so the solver
#   cannot accidentally fall back to color matching.
# - The only reusable structural cue is: bar length -> area bucket -> largest actual
#   area in that bucket -> number of dots on that row.

COLOR_POOL = tuple(range(1, 10))
BLACK = 0

KEY_COUNT_RANGE = (2, 4)
COMPONENT_COUNT_RANGE = (2, 4)
DISTRACTOR_PROB = 0.45

KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 2)

GRID_HEIGHT_RANGE = (22, 28)
GRID_WIDTH_RANGE = (30, 36)
LEGEND_EDGE_MARGIN = 2

SPRITE_BBOX = (4, 4)
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 600

# Area buckets encoded by legend-bar length.
AREA_BUCKETS: dict[int, tuple[int, ...]] = {
    2: (4, 5, 6),
    3: (7, 8, 9),
    4: (10, 11, 12),
}
BAR_LENGTHS = tuple(sorted(AREA_BUCKETS))


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


@lru_cache(maxsize=None)
def _valid_masks(area: int) -> tuple[np.ndarray, ...]:
    """Connected 4x4 masks with the requested area and the full 4x4 bbox."""
    h, w = SPRITE_BBOX
    masks = []
    for cells in combinations(range(h * w), int(area)):
        mask = np.zeros((h, w), dtype=bool)
        for cell in cells:
            mask[cell // w, cell % w] = True
        spans_edges = (
            mask[0, :].any()
            and mask[-1, :].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        )
        if not spans_edges:
            continue
        _lab, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if int(n) == 1:
            masks.append(mask)
    return tuple(masks)


def _area_to_bar_len(area: int) -> int | None:
    for bar_len, areas in AREA_BUCKETS.items():
        if int(area) in areas:
            return int(bar_len)
    return None


def _make_sprite(color: int, bg: int, area: int) -> np.ndarray:
    mask = random.choice(_valid_masks(int(area)))
    sprite = np.full(SPRITE_BBOX, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _legend_shape(k: int, max_reported_area: int, max_bar_len: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    tally_width = KEY_COLUMN + TALLY_COL_STEP * (max_reported_area - 1) + LEGEND_WIDTH_TAIL
    marker_width = KEY_COLUMN + max_bar_len + LEGEND_WIDTH_TAIL
    bw = max(tally_width, marker_width) + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside sprites to legend size buckets and encode each bucket's largest area.

    Input structure:
    - A black rectangular legend box contains one colored horizontal bar on each key row.
    - Outside the box, isolated monochrome sprites of unrelated colors are scattered across the background.

    Rule:
    - Read each legend-bar length as an area bucket, then assign every outside sprite to the bucket containing its pixel area.
    - For each legend row, find the largest actual sprite area among the outside sprites in that bucket and draw that many same-color dots on the row, spaced every other cell from the bar start.

    Key insight: The legend does not identify outside colors anymore; it encodes coarse size classes, and the answer for each class is the maximum measured component area within that class.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    best = None  # (area, bbox)
    for coords in components_mask(g == int(BLACK), connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != int(BLACK)).any():
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
        if (box[:, int(cc)] != int(BLACK)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys: list[tuple[int, int, int]] = []
    for rr in range(bh):
        color = int(box[int(rr), int(key_col)])
        if color == int(BLACK):
            continue
        span = 0
        cc = int(key_col)
        while cc < bw and int(box[int(rr), int(cc)]) == color:
            span += 1
            cc += 1
        keys.append((int(rr), color, int(span)))
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    outside_mask = (g != int(bg)) & (g != int(BLACK)) & outside

    max_area_by_bucket: dict[int, int] = {}
    for coords in components_mask(outside_mask, connectivity=8):
        area = int(coords.shape[0])
        bar_len = _area_to_bar_len(area)
        if bar_len is None:
            continue
        max_area_by_bucket[bar_len] = max(int(max_area_by_bucket.get(bar_len, 0)), int(area))

    out = np.zeros((bh, bw), dtype=int)
    for rr, color, bar_len in keys:
        reported_area = int(max_area_by_bucket.get(int(bar_len), 0))
        for i in range(reported_area):
            cc = int(key_col + TALLY_COL_STEP * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(color)

    return out


def _sample_palette(k: int) -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    remaining = [c for c in colors if c != bg]
    legend_colors = random.sample(remaining, k)

    remaining = [c for c in remaining if c not in legend_colors]
    sprite_palette_size = min(len(remaining), _randint((2, 3)))
    sprite_colors = random.sample(remaining, sprite_palette_size)
    return bg, legend_colors, sprite_colors


def _sample_bars(k: int) -> list[int]:
    if k <= len(BAR_LENGTHS):
        bars = random.sample(list(BAR_LENGTHS), k)
    else:
        bars = list(BAR_LENGTHS)
        while len(bars) < k:
            bars.append(random.choice(BAR_LENGTHS))
        random.shuffle(bars)
    if len(set(bars)) == 1 and len(BAR_LENGTHS) > 1:
        bars[-1] = random.choice([b for b in BAR_LENGTHS if b != bars[0]])
        random.shuffle(bars)
    return bars


def _sample_bucket_program(used_bars: set[int]) -> dict[int, list[int]]:
    """
    For each used bucket, build a small multiset of areas with a meaningful maximum.

    Every used bucket includes:
    - one sampled bucket maximum, chosen from the upper part of the bucket,
    - at least one smaller sprite from the same bucket.
    """
    program: dict[int, list[int]] = {}
    for bar_len in used_bars:
        areas = list(AREA_BUCKETS[int(bar_len)])
        max_area = random.choice(areas[1:])
        smaller = random.choice([area for area in areas if area < max_area])
        count = _randint(COMPONENT_COUNT_RANGE)
        sampled = [max_area, smaller]
        while len(sampled) < count:
            sampled.append(random.choice([area for area in areas if area <= max_area]))
        random.shuffle(sampled)
        program[int(bar_len)] = sampled
    return program


def _build_expected(
    legend_colors: list[int],
    bars: list[int],
    bucket_program: dict[int, list[int]],
) -> np.ndarray:
    reported_areas = [max(bucket_program[int(bar_len)]) for bar_len in bars]
    bh, bw = _legend_shape(len(legend_colors), max(reported_areas), max(bars))
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, area) in enumerate(zip(legend_colors, reported_areas)):
        rr = _key_row(i)
        for j in range(area):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(
    shape: tuple[int, int], legend_colors: list[int], bars: list[int]
) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, (color, bar_len) in enumerate(zip(legend_colors, bars)):
        rr = _key_row(i)
        legend[int(rr), KEY_COLUMN : KEY_COLUMN + int(bar_len)] = int(color)
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
    bucket_program: dict[int, list[int]],
    sprite_colors: list[int],
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for row_idx, bar_len in enumerate(sorted(bucket_program)):
        areas = bucket_program[int(bar_len)]
        start = random.randrange(len(sprite_colors))
        for j, area in enumerate(areas):
            color = sprite_colors[(start + row_idx + j) % len(sprite_colors)]
            sprites.append(_make_sprite(color, bg, int(area)))

    unused_bars = [bar for bar in BAR_LENGTHS if bar not in bucket_program]
    if unused_bars and random.random() < DISTRACTOR_PROB:
        dbar = random.choice(unused_bars)
        darea = random.choice(AREA_BUCKETS[int(dbar)])
        dcolor = random.choice(sprite_colors)
        sprites.append(_make_sprite(dcolor, bg, int(darea)))

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
    k = _randint(KEY_COUNT_RANGE)
    bars = _sample_bars(k)
    bucket_program = _sample_bucket_program(set(bars))
    bg, legend_colors, sprite_colors = _sample_palette(k)

    expected = _build_expected(legend_colors, bars, bucket_program)
    legend = _build_legend(expected.shape, legend_colors, bars)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(bucket_program, sprite_colors, bg)
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
