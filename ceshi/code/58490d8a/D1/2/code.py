# Variant of 58490d8a
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: s.color == d.color — 图例框内的关键像素颜色与框外精灵颜色一致即为匹配组... → 空间区域匹配：按精灵所在象限分组

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# Design analysis for the D1 variant
# a) Concrete behavior:
#    The original task used each legend-row color as a lookup key for outside sprites of the same color.
#    Here the legend still has two colored key rows, but those colors are only row labels. The upper legend
#    row now summarizes the upper half of the grid, and the lower legend row summarizes the lower half.
#    Small trace: if the input has 2 blue + 1 yellow sprites above the midline and 3 teal sprites below it,
#    the original solver would have counted by color, while the variant writes 3 dots on the first row and
#    3 dots on the second row, regardless of the sprite colors.
# b) Complexity calibration:
#    Two spatial groups are enough to make the change visible, but the generator forces repeated evidence:
#    both halves always contain sprites, and at least one sprite color appears in both halves. This prevents
#    the scene from collapsing into a trivial color-coded tally.
# c) Implicit-assumption audit:
#    The original `for rr, col in keys` loop assumed that each row could isolate its source objects by color.
#    That is no longer true. The new logic treats the sorted key rows as ordered slots (top first, bottom
#    second) and counts all outside components by vertical position. This means the solver now assumes that
#    sprites do not straddle the half-divider; the generator enforces that.
# d) Placement legality:
#    Sprites are still connected 2x2..3x3 monochrome objects with 8-neighbor padding. They are placed
#    separately into the top and bottom slices, with a spare divider row between slices, so they never overlap,
#    never touch each other, and never cross the classifier boundary used by the solver.
# e) Visual observability:
#    The row-to-region mapping is readable from top-to-bottom ordering: the first legend row counts upper
#    sprites, the second counts lower sprites. Because sprite colors are independent from legend colors, the
#    new grouping rule is visible instead of being masked by the old color rule.
# f) Visual cleanliness:
#    The legend stays a clean black box, sprites remain sparse and non-touching, and only one simple spatial
#    split is used. No extra markers are added on top of the original scene.
# g) Color inference:
#    No hidden color-id ordering is used. Legend colors are distinct row labels, while sprite colors come from
#    a separate pool and intentionally repeat across both halves so the human solver can reject color matching.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites by vertical region and encode the tallies inside the legend.

    Input structure:
    - A black legend rectangle contains two colored key pixels in the left tally column.
    - Outside the box, disconnected monochrome sprites are scattered in both the upper and lower halves of the grid.

    Rule:
    - Ignore outside sprite color when deciding which legend row to use.
    - Count all outside connected components whose centers fall in the upper half, draw that many evenly spaced dots on the upper legend row, and do the same for the lower half on the lower row.

    Key insight: The legend rows are ordered spatial buckets rather than color buckets, so position relative to the grid midline determines the matching row.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
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
    # leftmost column with any non-black pixels (the "row labels").
    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    keys.sort(key=lambda t: int(t[0]))
    assert len(keys) >= 2

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    # STEP 3) Count all outside components, ignoring color, and bucket them by upper/lower half.
    split_row = int(H // 2)
    region_counts = [0, 0]
    outside_fg = outside & (g != int(bg)) & (g != int(black))
    for coords in components_mask(outside_fg, connectivity=8):
        cr0, _cc0, cr1, _cc1 = bbox_coords(coords)
        center_r = 0.5 * (int(cr0) + int(cr1))
        region_counts[0 if center_r < split_row else 1] += 1

    # STEP 4) Render tally dots onto the ordered legend rows: top row then bottom row.
    for region_idx, (rr, col) in enumerate(keys[:2]):
        for i in range(region_counts[int(region_idx)]):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (D1 spatial-region variant) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0

# The variant uses exactly two legend rows: upper half and lower half.
KEY_COUNT = 2
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
REGION_GAP_ROWS = 1

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


def _sample_palette() -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)
    remaining = [c for c in colors if c != bg]

    key_colors = random.sample(remaining, KEY_COUNT)
    remaining = [c for c in remaining if c not in key_colors]

    sprite_color_count = random.randint(2, min(3, len(remaining)))
    sprite_colors = random.sample(remaining, sprite_color_count)
    return bg, key_colors, sprite_colors


def _sample_counts() -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(KEY_COUNT)]


def _legend_shape(max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * KEY_COUNT + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(max_count)
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

    split_row = int(H // 2)
    r0 = int(split_row - 2)
    if r0 < LEGEND_EDGE_MARGIN or r0 + bh > H - LEGEND_EDGE_MARGIN:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    c0 = random.randint(LEGEND_EDGE_MARGIN, W - bw - LEGEND_EDGE_MARGIN)
    paste(grid, legend, r0, c0)
    return grid


def _sample_region_colors(counts: list[int], sprite_colors: list[int]) -> list[list[int]]:
    # Guarantee at least one repeated color across both halves, so color alone cannot explain the tally rows.
    bridge = random.choice(sprite_colors)
    top_colors = [bridge]
    bottom_colors = [bridge]
    others = [c for c in sprite_colors if c != bridge]

    if others:
        alt = random.choice(others)
        if counts[0] > 1:
            top_colors.append(alt)
        elif counts[1] > 1:
            bottom_colors.append(alt)

    while len(top_colors) < counts[0]:
        top_colors.append(random.choice(sprite_colors))
    while len(bottom_colors) < counts[1]:
        bottom_colors.append(random.choice(sprite_colors))

    random.shuffle(top_colors)
    random.shuffle(bottom_colors)
    return [top_colors, bottom_colors]


def _build_region_sprites(region_colors: list[list[int]], bg: int) -> list[list[np.ndarray]]:
    palette = sorted({int(c) for colors in region_colors for c in colors})
    base_sprites = {int(color): _make_sprite(int(color), bg) for color in palette}
    return [[base_sprites[int(color)].copy() for color in colors] for colors in region_colors]


def _region_slices(H: int) -> tuple[tuple[int, int], tuple[int, int]]:
    split_row = int(H // 2)
    return (0, split_row), (split_row + REGION_GAP_ROWS, H)


def _place_region_sprites(grid: np.ndarray, region_sprites: list[list[np.ndarray]], bg: int) -> None:
    (top_a, top_b), (bot_a, bot_b) = _region_slices(grid.shape[0])
    for (row_a, row_b), sprites in zip(((top_a, top_b), (bot_a, bot_b)), region_sprites):
        random.shuffle(sprites)
        subgrid = grid[int(row_a) : int(row_b), :]
        for sprite in sprites:
            sr, sc = random_free_location(
                subgrid,
                sprite,
                transparent=bg,
                border=1,
                padding=SPRITE_PADDING,
                padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
            )
            paste(grid, sprite, int(row_a) + int(sr), int(sc), transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    bg, key_colors, sprite_colors = _sample_palette()
    counts = _sample_counts()
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    region_colors = _sample_region_colors(counts, sprite_colors)
    region_sprites = _build_region_sprites(region_colors, bg)
    try:
        _place_region_sprites(grid, region_sprites, bg)
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
    raise RuntimeError("failed to generate a valid 58490d8a D1 input")
