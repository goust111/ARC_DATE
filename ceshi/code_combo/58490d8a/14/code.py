# Combo variant of 58490d8a (index 14)
# combo_spec: D1/2+D9/1
# Name: 区域匹配+精灵消失

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

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The source task used legend-row color as the matching key and returned only the cropped legend box.
#    - This combo keeps the same visual ingredients, but the two legend rows now stand for ordered spatial
#      regions: upper half first, lower half second. Example: if the top half contains 3 sprites in mixed
#      colors and the bottom half contains 2 sprites, the original task would have tried to count only
#      same-colored sprites for each legend row and would output a small black crop. The combo instead writes
#      3 dots on the upper legend row and 2 on the lower legend row, returns the full grid, and erases all
#      outside sprites after counting them.
#    - So the fused change is not "count by region" plus an independent canvas tweak. The region buckets decide
#      both which legend row gets the tally and which outside sprites disappear from the preserved scene.
# b) Complexity calibration:
#    - The generator fixes exactly two legend rows because the spatial split has only two buckets.
#    - Legend colors are sampled from a palette disjoint from all outside sprite colors, and at least one
#      sprite color is reused in both halves. That makes the old color-matching explanation visibly impossible.
#    - Counts are forced to differ across the two halves and at least one count is > 1, so the legend rows
#      change in a readable way instead of collapsing into two identical single dots.
# c) Implicit-assumption audit:
#    - The original `for rr, col in keys` counting loop assumed each row could isolate its source set by color.
#      That assumption is invalid here; rows are ordered slots, not color groups.
#    - The original cropped `out = np.zeros((bh, bw))` is also invalid because D9 changes the output canvas to
#      the full grid. The solver must preserve the whole scene, erase counted sprites outside the legend, then
#      repaint the legend tally in place.
#    - New solver assumptions are: there are exactly two meaningful legend rows, the legend straddles the
#      horizontal split so the upper row naturally corresponds to the upper half and the lower row to the lower
#      half, and sprites do not cross the split line. The generator enforces all three constraints.
# d) Placement legality:
#    - Sprites remain connected 2x2..3x3 monochrome components with 8-neighborhood padding.
#    - They are placed separately into top and bottom slices with one unused divider row between them, so no
#      sprite crosses the classifier boundary and components stay isolated.
#    - The legend width is still derived from the maximum regional count, so every tally run fits after the
#      legend is repainted.
# e) Visual observability:
#    - Human solvers can see that legend colors never appear outside the box, so the rows cannot be matching by
#      color. They can also see that one outside sprite color repeats across both halves, ruling out a hidden
#      palette ordering. The only stable grouping cue is vertical position.
#    - The output also makes the erase effect explicit: both halves become empty where the counted sprites used
#      to be, while the full grid shape stays unchanged.
# f) Visual cleanliness:
#    - No extra markers or overlapping overlays are added. The output is simply the original scene with the
#      outside sprites removed and the legend converted into the tally panel.
#    - Because sprites are sparse and non-touching, the erased regions read as clean disappearances rather than
#      ambiguous merged gaps.
# g) Color inference:
#    - No invisible color-id ordering is used. Legend colors are only row labels, and sprite colors are chosen
#      independently and repeated across halves to provide visible evidence that color is irrelevant.


def _find_legend_bbox(grid: np.ndarray, black: int) -> tuple[int, int, int, int]:
    """Return the legend bbox: the largest black component whose bbox encloses non-black cells."""
    best = None  # (area, bbox)
    for coords in components_mask(grid == int(black), connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(grid, bbox)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, bbox)))
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return tuple(map(int, best[1]))


def _extract_keys(box: np.ndarray, black: int) -> tuple[int | None, list[tuple[int, int]]]:
    """Return the leftmost key column and ordered (row, color) legend entries."""
    bh, bw = map(int, box.shape)
    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return None, []
    keys = [
        (int(rr), int(box[int(rr), int(key_col)]))
        for rr in range(bh)
        if int(box[int(rr), int(key_col)]) != int(black)
    ]
    keys.sort(key=lambda t: int(t[0]))
    return key_col, keys


def _outside_mask(shape: tuple[int, int], legend_bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Return a mask that is True exactly outside the legend rectangle."""
    r0, c0, r1, c1 = map(int, legend_bbox)
    outside = np.ones(shape, dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    return outside


def _count_region_components(grid: np.ndarray, outside_fg: np.ndarray) -> list[int]:
    """Count outside connected components by upper and lower halves of the full grid."""
    H, _W = map(int, grid.shape)
    split_row = int(H // 2)
    counts = [0, 0]
    for coords in components_mask(outside_fg, connectivity=8):
        cr0, _cc0, cr1, _cc1 = bbox_coords(coords)
        center_r = 0.5 * (int(cr0) + int(cr1))
        counts[0 if center_r < split_row else 1] += 1
    return counts


def _build_tally_patch(shape: tuple[int, int], keys: list[tuple[int, int]], key_col: int, region_counts: list[int]) -> np.ndarray:
    """Render the black legend patch with one every-other-cell dot per region count."""
    bh, bw = map(int, shape)
    patch = np.zeros((bh, bw), dtype=int)
    for region_idx, (rr, col) in enumerate(keys[:2]):
        for i in range(int(region_counts[int(region_idx)])):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                patch[int(rr), int(cc)] = int(col)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites by vertical region, erase them, and rewrite the legend in place.

    Input structure:
    - A black legend rectangle contains two colored key pixels in its left tally column.
    - Outside the box, disconnected monochrome sprites appear in both the upper and lower halves of a single-background grid.

    Rule:
    - Ignore outside sprite color when matching to legend rows; instead group every outside connected component by whether it lies in the upper or lower half of the grid.
    - Output the full grid: erase all outside sprites back to the background, repaint the legend box black, then draw one evenly spaced dot per counted sprite on the upper legend row for the upper half and on the lower legend row for the lower half.

    Key insight: The legend rows are ordered spatial buckets rather than color buckets, and the counted sprites disappear from those buckets after their regional totals are written into the legend.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    black = 0

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains non-black pixels.
    r0, c0, r1, c1 = _find_legend_bbox(g, black)
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Inside the legend bbox, find the key column:
    # leftmost column with any non-black pixels (the ordered region labels).
    key_col, keys = _extract_keys(box, black)
    if key_col is None or len(keys) < 2:
        return g.copy()

    outside = _outside_mask(g.shape, (r0, c0, r1, c1))

    # STEP 3) Count all outside components, ignoring their color, and bucket them by upper/lower half.
    outside_fg = outside & (g != int(bg)) & (g != int(black))
    region_counts = _count_region_components(g, outside_fg)

    # STEP 4) Keep the full scene, erase all counted sprites, then repaint the legend tally.
    out = g.copy()
    out[outside_fg] = int(bg)

    tally = _build_tally_patch((bh, bw), keys, int(key_col), region_counts)
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(black)
    paste(out, tally, int(r0), int(c0))
    return out


# --------------- generator (combo D1/2 + D9/1) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0

# The combo uses exactly two legend rows: upper half and lower half.
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
    """Choose background, legend-row colors, and a disjoint outside-sprite palette."""
    colors = list(COLOR_POOL)
    bg = random.choice(colors)
    remaining = [c for c in colors if c != bg]

    key_colors = random.sample(remaining, KEY_COUNT)
    remaining = [c for c in remaining if c not in key_colors]

    sprite_color_count = random.randint(2, min(3, len(remaining)))
    sprite_colors = random.sample(remaining, sprite_color_count)
    return bg, key_colors, sprite_colors


def _sample_counts() -> list[int]:
    """Choose distinct upper/lower region counts and force at least one visible extension."""
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(KEY_COUNT)]
        if counts[0] != counts[1] and any(count > 1 for count in counts):
            return counts


def _legend_shape(max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * KEY_COUNT + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_tally_canvas(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(max_count)
    tally = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            tally[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return tally


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Place the two-row legend so it straddles the global horizontal split."""
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
    return grid, (int(r0), int(c0), int(r0 + bh - 1), int(c0 + bw - 1))


def _sample_region_colors(counts: list[int], sprite_colors: list[int]) -> list[list[int]]:
    """Assign sprite colors per region while forcing cross-region color repetition."""
    bridge = random.choice(sprite_colors)
    top_colors = [bridge]
    bottom_colors = [bridge]
    others = [c for c in sprite_colors if c != bridge]

    if others:
        alt = random.choice(others)
        if counts[0] > 1:
            top_colors.append(alt)
        if counts[1] > 1:
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


def _build_expected_full(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], tally: np.ndarray, bg: int) -> np.ndarray:
    """Apply the combo transform: erase all outside sprites and repaint the legend tally."""
    r0, c0, r1, c1 = map(int, legend_bbox)
    expected = np.array(grid, dtype=int, copy=True)
    outside = _outside_mask(expected.shape, legend_bbox)
    outside_fg = outside & (expected != int(bg)) & (expected != int(BLACK))

    expected[outside_fg] = int(bg)
    expected[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(BLACK)
    paste(expected, tally, int(r0), int(c0))
    return expected


def _color_match_counts(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], key_colors: list[int]) -> list[int]:
    """Reference metric: counts produced by the original color-matching rule."""
    outside = _outside_mask(grid.shape, legend_bbox)
    return [int(label_mask((grid == int(color)) & outside, connectivity=8)[1]) for color in key_colors]


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    bg, key_colors, sprite_colors = _sample_palette()
    counts = _sample_counts()
    tally = _build_tally_canvas(key_colors, counts)
    legend = _build_legend(tally.shape, key_colors)
    placed = _new_grid_with_legend(bg, legend)
    if placed is None:
        return None
    grid, legend_bbox = placed

    region_colors = _sample_region_colors(counts, sprite_colors)
    region_sprites = _build_region_sprites(region_colors, bg)
    try:
        _place_region_sprites(grid, region_sprites, bg)
    except ValueError:
        return None

    expected = _build_expected_full(grid, legend_bbox, tally, bg)

    # Reject weak or degenerate samples that could still look color-driven.
    original_counts = _color_match_counts(grid, legend_bbox, key_colors)
    if original_counts == counts:
        return None

    outside = _outside_mask(grid.shape, legend_bbox)
    outside_fg = outside & (grid != int(bg)) & (grid != int(BLACK))
    erased_pixels = int(outside_fg.sum())
    if erased_pixels <= 0:
        return None

    if not np.array_equal(main(grid), expected):
        return None

    meta = {
        "counts": counts,
        "original_counts": original_counts,
        "erased_pixels": erased_pixels,
        "key_colors": key_colors,
        "sprite_colors": sprite_colors,
        "region_colors": region_colors,
    }
    return grid, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        triple = _gen_attempt()
        if triple is not None:
            grid, _expected, _meta = triple
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
