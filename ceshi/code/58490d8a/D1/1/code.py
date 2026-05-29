# Variant of 58490d8a
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: s.color == d.color — 图例框内的关键像素颜色与框外精灵颜色一致即为匹配组... → 尺寸匹配：按精灵面积分组

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
# Design analysis for the D1 size-match variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - The black legend box stays in the same role and position regime as the original
#   task. The changed part is the row marker: instead of one colored key pixel, each
#   key row now starts with a horizontal colored bar whose length encodes an area
#   category. Example: a 5-cell bar means "count all outside sprites whose area is 5".
# - Outside sprites are still separate monochrome connected components, but matching
#   now ignores their color completely. The solver must measure each component's
#   pixel area and tally it onto the legend row whose bar length equals that area.
# - Small trace:
#   original row marker = one red pixel, outside has 3 red components -> output row
#   gets 3 red dots.
#   variant row marker = a 5-cell red bar, outside has blue/green/yellow components
#   with areas [5, 4, 5, 7] -> output row gets 2 red dots, because only the two
#   area-5 components match. Color agreement is irrelevant.
#
# b) Complexity calibration
# - Area categories are kept to the original sprite-area scale: 4, 5, 6, 7 pixels.
#   This is small enough for human counting, but large enough that rows visibly differ.
# - To make the change observable, outside sprites are all cropped to the same 3x3
#   bbox family, and only the number of filled cells changes. This prevents bbox size
#   from replacing the intended "count pixels" reasoning.
# - Legend colors remain distinct so output rows are still easy to read, but outside
#   sprite colors are sampled from a separate palette so the old color-matching rule
#   cannot explain the examples.
#
# c) Audit of original main() assumptions that break under the variant
# - Original code assumed "row color == queried outside color" and built per-color
#   masks. That must be replaced with a single outside-component scan plus area counts.
# - Original code treated the key as one pixel at (row, key_col). That is no longer
#   enough; the solver must measure the contiguous same-color run starting at key_col
#   to recover the legend row's target area.
# - Original code could label each key color independently without merge risk. Under
#   size matching, different-colored sprites may belong to the same group, so the
#   solver must aggregate all outside components first, then group by component area.
#
# d) Placement legality
# - The legend bars always start at the same left key column and leave at least one
#   black cell on their right, so the surrounding black legend region stays 8-connected
#   and is still detectable as one large black component.
# - Outside sprites are placed with 8-neighborhood padding, so counting connected
#   components on the combined non-background mask is safe even when colors vary.
#
# e) Visual observability
# - The changed rule is visible in two ways: legend rows now have different bar lengths,
#   and outside sprite colors do not match legend colors. A human has to use size,
#   not color, to align rows with counted objects.
# - Row order is randomized; the rule is not a hidden numeric sort. Each row can be
#   solved independently from its visible bar length.
#
# f) Visual cleanliness
# - The black box remains uncluttered: one colored bar per odd row, nothing else.
# - Outside sprites stay separated by padding and all share the same 3x3 footprint
#   family, so the canvas does not become visually noisy despite mixed colors.
#
# g) Color-inference safety
# - No invisible color-id ordering is used.
# - Legend colors and outside sprite colors come from disjoint sets. This makes the
#   size-based grouping legible and prevents accidental regression to the original
#   color-matching task on generated samples.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites by area and write the tallies inside a legend box.

    Input structure:
    - A black rectangular legend box contains one colored horizontal bar on each key row.
    - Outside the box, separated monochrome sprites of various colors are scattered across the grid.

    Rule:
    - The length of a legend-row bar gives the target sprite area for that row.
    - Count outside connected components whose pixel area matches that bar length, then draw one dot per matching component on the same row, spaced every other cell from the bar start.

    Key insight: The legend no longer names groups by color; it encodes them numerically by bar length, so the solver must measure component areas and tally by size.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
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

    assert not (best is None)

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

    keys = []
    for rr in range(bh):
        color = int(box[int(rr), int(key_col)])
        if color == int(black):
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

    comp_areas: dict[int, int] = {}
    outside_mask = (g != int(bg)) & (g != int(black)) & outside
    for coords in components_mask(outside_mask, connectivity=8):
        area = int(coords.shape[0])
        comp_areas[area] = int(comp_areas.get(area, 0) + 1)

    out = np.zeros((bh, bw), dtype=int)
    for rr, color, target_area in keys:
        count = int(comp_areas.get(int(target_area), 0))
        for i in range(count):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(color)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 4)
AREA_CHOICES = (4, 5, 6, 7)
DISTRACTOR_PROB = 0.5

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

SPRITE_BBOX = (3, 3)
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 400


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


@lru_cache(maxsize=None)
def _valid_masks(area: int) -> tuple[np.ndarray, ...]:
    """Connected 3x3 masks with the requested area and full 3x3 bbox."""
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


def _make_sprite(color: int, bg: int, area: int) -> np.ndarray:
    mask = random.choice(_valid_masks(int(area)))
    sprite = np.full(SPRITE_BBOX, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _sample_palette(k: int) -> tuple[int, list[int], list[int]]:
    """Choose background, legend colors, and an outside-sprite color pool."""
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    remaining = [c for c in colors if c != bg]
    legend_colors = random.sample(remaining, k)

    remaining = [c for c in remaining if c not in legend_colors]
    sprite_palette_size = min(len(remaining), _randint((2, 3)))
    sprite_colors = random.sample(remaining, sprite_palette_size)
    return bg, legend_colors, sprite_colors


def _sample_rows() -> tuple[list[int], list[int]]:
    k = _randint(KEY_COUNT_RANGE)
    areas = random.sample(list(AREA_CHOICES), k)
    counts = [_randint(TARGET_COUNT_RANGE) for _ in range(k)]
    return areas, counts


def _legend_shape(k: int, max_count: int, max_area: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    tally_width = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL
    marker_width = KEY_COLUMN + max_area + LEGEND_WIDTH_TAIL
    bw = max(tally_width, marker_width) + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(
    legend_colors: list[int], areas: list[int], counts: list[int]
) -> np.ndarray:
    bh, bw = _legend_shape(len(legend_colors), max(counts), max(areas))
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(legend_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(
    shape: tuple[int, int], legend_colors: list[int], areas: list[int]
) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, (color, area) in enumerate(zip(legend_colors, areas)):
        rr = _key_row(i)
        legend[int(rr), KEY_COLUMN : KEY_COLUMN + int(area)] = int(color)
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
    areas: list[int],
    counts: list[int],
    sprite_colors: list[int],
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for row_idx, (area, count) in enumerate(zip(areas, counts)):
        start = random.randrange(len(sprite_colors))
        for j in range(count):
            color = sprite_colors[(start + row_idx + j) % len(sprite_colors)]
            sprites.append(_make_sprite(color, bg, int(area)))

    remaining_areas = [a for a in AREA_CHOICES if a not in set(areas)]
    if remaining_areas and random.random() < DISTRACTOR_PROB:
        distractor_area = random.choice(remaining_areas)
        distractor_color = random.choice(sprite_colors)
        sprites.append(_make_sprite(distractor_color, bg, int(distractor_area)))

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
    areas, counts = _sample_rows()
    bg, legend_colors, sprite_colors = _sample_palette(len(areas))

    expected = _build_expected(legend_colors, areas, counts)
    legend = _build_legend(expected.shape, legend_colors, areas)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(areas, counts, sprite_colors, bg)
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
    raise RuntimeError("failed to generate a valid 58490d8a D1 size-match input")
