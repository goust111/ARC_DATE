# Combo variant of 58490d8a (index 10)
# combo_spec: D1/2+D3/2
# Name: 区域匹配+分散图例框

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location


COLOR_POOL = tuple(range(1, 10))
BLACK = 0

# The fused region-matching rule uses exactly two spatial buckets: upper and lower halves.
REGION_COUNT = 2
TARGET_COUNT_RANGE = (1, 5)

LEGEND_HEIGHT = 3
LEGEND_KEY_ROW = 1
LEGEND_KEY_COL = 1
TALLY_COL_STEP = 2
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 2)

GRID_HEIGHT_RANGE = (24, 30)
GRID_WIDTH_RANGE = (24, 30)
LEGEND_BORDER = 1
LEGEND_PADDING = 1
REGION_GAP_ROWS = 1

SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 400


# Design analysis for the D1/2 + D3/2 combo
# a) Concrete behavior and trace:
#    - Original task: one big black legend box is cropped out, and each key color counts same-colored sprites anywhere outside.
#    - D3/2 breaks the single destination into many scattered mini-legends. D1/2 breaks color matching and replaces it with
#      spatial grouping. The fused task therefore becomes: each mini-legend represents the half of the grid it lives in.
#    - Small trace: suppose the upper half has 3 sprites (blue, yellow, blue) and the lower half has 2 sprites (teal, teal).
#      The top mini-legend gets 3 dots of its own label color, and the bottom mini-legend gets 2 dots of its own label color.
#      A color-matching solver would fail, because the legend colors are intentionally absent from the outside sprites.
# b) Complexity calibration:
#    - Use exactly two regions so the half-based rule is visually stable across examples.
#    - Keep varied 2..5 tally ranges and legend widths so counting remains nontrivial while every region still offers
#      enough evidence for humans to infer the spatial grouping rule from a small number of examples.
#    - Force at least one shared sprite color across both halves, so grouping by sprite color is visibly wrong.
#    - Force different top/bottom counts, so the two mini-legends usually display different tally lengths.
# c) Implicit-assumption audit:
#    - The original "largest black component is the legend" assumption is invalid: there are now two separate black boxes.
#    - The original "legend row matches sprite color" assumption is invalid: legend colors are only local dot colors.
#    - The original cropped-box output shape is invalid: the new output is a full-grid repaint with both boxes preserved in place.
#    - The new solver assumes every legend is a 3-row black component with one colored tally row at row 1 and key column 1.
#      The generator enforces that exact local structure.
#    - The new solver assigns components by vertical region, so it assumes sprites and legends do not straddle the midline.
#      The generator places them inside disjoint upper/lower slices with a spare divider row.
# d) Placement legality:
#    - Legends are placed first, one inside each half-slice, using collision-aware placement with borders.
#    - Sprites are then placed inside their designated half-slices with 8-neighbor padding, so no two sprites touch and no
#      sprite touches a legend.
#    - Legend width is derived from the target count of that region, guaranteeing enough room for all tally dots.
# e) Visual observability:
#    - Humans can see two separate black boxes rather than one cropped legend, satisfying the D3 change.
#    - Because legend colors do not appear outside, the examples strongly suggest that box location, not color, decides which
#      sprites belong to which tally, satisfying the D1 change.
#    - Shared sprite colors across both halves provide repeated evidence that the grouping rule is regional.
# f) Visual cleanliness:
#    - Each half contains one box plus sparse non-touching sprites, avoiding clutter.
#    - The divider row prevents ambiguous near-midline placements, and the legends remain compact 3-row rectangles.
# g) Change inferability:
#    - No hidden ordering by numeric color id is used. Region identity is visible from top-vs-bottom placement, and the output
#      color for each tally is directly given by the box's own seed pixel.


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _split_row(H: int) -> int:
    return int(H // 2)


def _region_slices(H: int) -> tuple[tuple[int, int], tuple[int, int]]:
    split = _split_row(H)
    return (0, split), (split + REGION_GAP_ROWS, H)


def _region_index_from_center(center_r: float, H: int) -> int:
    return 0 if center_r < _split_row(H) else 1


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
        _lab, n = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
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


def _find_legends(grid: np.ndarray) -> tuple[list[tuple[tuple[int, int, int, int], int, int, int, int]], np.ndarray]:
    H, W = grid.shape
    legends: list[tuple[tuple[int, int, int, int], int, int, int, int]] = []
    legends_mask = np.zeros((H, W), dtype=bool)

    for coords in components_mask(grid == int(BLACK), connectivity=8):
        if coords.size == 0:
            continue
        bbox = tuple(map(int, bbox_coords(coords)))
        patch = crop_bbox(grid, bbox)
        ph, pw = map(int, patch.shape)
        nonblack = np.argwhere(patch != int(BLACK))
        if nonblack.size == 0 or ph != LEGEND_HEIGHT:
            continue

        rows = {int(r) for r, _c in nonblack}
        colors = {int(patch[int(r), int(c)]) for r, c in nonblack}
        if rows != {LEGEND_KEY_ROW} or len(colors) != 1:
            continue

        cols = sorted(int(c) for _r, c in nonblack)
        key_col = cols[0]
        if key_col != LEGEND_KEY_COL:
            continue
        if any((cc - key_col) % TALLY_COL_STEP != 0 for cc in cols):
            continue

        color = next(iter(colors))
        legends.append((bbox, ph, pw, key_col, color))
        r0, c0, r1, c1 = bbox
        legends_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    legends.sort(key=lambda item: (item[0][0] + item[0][2], item[0][1]))
    return legends, legends_mask


def _count_region_components(grid: np.ndarray, legends_mask: np.ndarray, bg: int) -> list[int]:
    H, _W = grid.shape
    counts = [0, 0]
    outside_fg = (~legends_mask) & (grid != int(bg)) & (grid != int(BLACK))
    for coords in components_mask(outside_fg, connectivity=8):
        if coords.size == 0:
            continue
        r0, _c0, r1, _c1 = bbox_coords(coords)
        center_r = 0.5 * (int(r0) + int(r1))
        counts[_region_index_from_center(center_r, H)] += 1
    return counts


def _scattered_color_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    legends, legends_mask = _find_legends(g)
    out = np.full((H, W), int(bg), dtype=int)
    for bbox, ph, pw, key_col, color in legends:
        r0, c0, _r1, _c1 = bbox
        mask = (g == int(color)) & (~legends_mask)
        _lab, n = label_mask(mask, connectivity=8)
        patch = np.full((ph, pw), int(BLACK), dtype=int)
        for i in range(int(n)):
            cc = int(key_col + TALLY_COL_STEP * i)
            if 0 <= cc < pw:
                patch[LEGEND_KEY_ROW, cc] = int(color)
        paste(out, patch, r0, c0)
    return out


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))
    legends, legends_mask = _find_legends(g)
    region_counts = _count_region_components(g, legends_mask, bg)

    same_color_counts = []
    for _bbox, _ph, _pw, _key_col, color in legends:
        mask = (g == int(color)) & (~legends_mask)
        _lab, n = label_mask(mask, connectivity=8)
        same_color_counts.append(int(n))

    spread_gap = 0
    if legends:
        bboxes = [bbox for bbox, _ph, _pw, _key_col, _color in legends]
        total_area = sum((b[2] - b[0] + 1) * (b[3] - b[1] + 1) for b in bboxes)
        r0 = min(b[0] for b in bboxes)
        c0 = min(b[1] for b in bboxes)
        r1 = max(b[2] for b in bboxes)
        c1 = max(b[3] for b in bboxes)
        spread_gap = int((r1 - r0 + 1) * (c1 - c0 + 1) - total_area)

    color_gap = 0
    legend_regions = []
    for bbox, _ph, _pw, _key_col, _color in legends:
        center_r = 0.5 * (bbox[0] + bbox[2])
        region_idx = _region_index_from_center(center_r, H)
        legend_regions.append(region_idx)
    for region_idx, same_color_count in zip(legend_regions, same_color_counts):
        color_gap += abs(int(region_counts[region_idx]) - int(same_color_count))

    matches_color_baseline = int(np.array_equal(main(g), _scattered_color_baseline(g)))
    return {
        "legend_count": int(len(legends)),
        "spread_gap": int(spread_gap),
        "color_gap": int(color_gap),
        "top_count": int(region_counts[0]),
        "bottom_count": int(region_counts[1]),
        "matches_color_baseline": int(matches_color_baseline),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites by region and write the totals into scattered local legend boxes.

    Input structure:
    - Two separate black mini-legends are scattered apart, one in the upper half of the grid and one in the lower half, each with a single colored seed pixel on its middle row.
    - Outside those boxes, small disconnected sprites of several non-black colors are scattered across the same upper and lower halves.

    Rule:
    - Ignore sprite color when deciding which legend box a sprite belongs to.
    - Count all outside connected components whose centers lie in the same half of the grid as a given legend box, then output a full-grid repaint that keeps each box in place and fills its middle row with that many evenly spaced dots of the box's own color.

    Key insight: Each mini-legend is a regional tally marker rather than a color key, so the box location determines the matching sprite set and the seed color only determines the color of the dots drawn inside that box.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    legends, legends_mask = _find_legends(g)
    assert len(legends) >= 2

    region_counts = _count_region_components(g, legends_mask, bg)
    out = np.full((H, W), int(bg), dtype=int)

    for bbox, ph, pw, key_col, color in legends:
        r0, c0, r1, c1 = bbox
        center_r = 0.5 * (int(r0) + int(r1))
        region_idx = _region_index_from_center(center_r, H)
        count = int(region_counts[region_idx])

        patch = np.full((ph, pw), int(BLACK), dtype=int)
        for i in range(count):
            cc = int(key_col + TALLY_COL_STEP * i)
            if 0 <= cc < pw:
                patch[LEGEND_KEY_ROW, cc] = int(color)
        paste(out, patch, r0, c0)

    return out


def _sample_palette() -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)
    remaining = [c for c in colors if c != bg]

    legend_colors = random.sample(remaining, REGION_COUNT)
    remaining = [c for c in remaining if c not in legend_colors]

    sprite_color_count = random.randint(2, min(3, len(remaining)))
    sprite_colors = random.sample(remaining, sprite_color_count)
    return bg, legend_colors, sprite_colors


def _sample_counts() -> list[int]:
    while True:
        counts = [_randint(TARGET_COUNT_RANGE) for _ in range(REGION_COUNT)]
        if counts[0] != counts[1] and min(counts) >= 2:
            return counts


def _legend_width(count: int) -> int:
    return TALLY_COL_STEP * count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)


def _build_input_legend(color: int, width: int) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, width), int(BLACK), dtype=int)
    patch[LEGEND_KEY_ROW, LEGEND_KEY_COL] = int(color)
    return patch


def _build_output_legend(color: int, count: int, width: int) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, width), int(BLACK), dtype=int)
    for j in range(count):
        patch[LEGEND_KEY_ROW, LEGEND_KEY_COL + TALLY_COL_STEP * j] = int(color)
    return patch


def _new_grids(bg: int) -> tuple[np.ndarray, np.ndarray]:
    H = _randint(GRID_HEIGHT_RANGE)
    W = _randint(GRID_WIDTH_RANGE)
    empty = np.full((H, W), int(bg), dtype=int)
    return empty.copy(), empty.copy()


def _place_legends(
    inp: np.ndarray,
    expected: np.ndarray,
    legend_specs: list[tuple[int, int, np.ndarray, np.ndarray]],
    bg: int,
) -> bool:
    slices = _region_slices(inp.shape[0])
    for region_idx, (_color, _count, in_patch, out_patch) in enumerate(legend_specs):
        row_a, row_b = slices[region_idx]
        sub_inp = inp[int(row_a) : int(row_b), :]
        try:
            rr, cc = random_free_location(
                sub_inp,
                in_patch,
                transparent=int(bg),
                border=LEGEND_BORDER,
                padding=LEGEND_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        paste(inp, in_patch, int(row_a) + int(rr), int(cc))
        paste(expected, out_patch, int(row_a) + int(rr), int(cc))
    return True


def _sample_region_colors(counts: list[int], sprite_colors: list[int]) -> list[list[int]]:
    bridge = random.choice(sprite_colors)
    alternates = [color for color in sprite_colors if color != bridge]
    assert alternates

    region_colors = []
    for region_idx, count in enumerate(counts):
        # Seed each half with one shared color plus one alternate color so color-based grouping is visibly wrong.
        colors = [bridge, alternates[region_idx % len(alternates)]]
        while len(colors) < count:
            colors.append(random.choice(sprite_colors))
        random.shuffle(colors)
        region_colors.append(colors)
    return region_colors


def _build_region_sprites(region_colors: list[list[int]], bg: int) -> list[list[np.ndarray]]:
    palette = sorted({int(color) for colors in region_colors for color in colors})
    base_sprites = {int(color): _make_sprite(int(color), bg) for color in palette}
    return [[base_sprites[int(color)].copy() for color in colors] for colors in region_colors]


def _place_region_sprites(grid: np.ndarray, region_sprites: list[list[np.ndarray]], bg: int) -> bool:
    for region_idx, sprites in enumerate(region_sprites):
        row_a, row_b = _region_slices(grid.shape[0])[region_idx]
        subgrid = grid[int(row_a) : int(row_b), :]
        random.shuffle(sprites)
        for sprite in sprites:
            try:
                sr, sc = random_free_location(
                    subgrid,
                    sprite,
                    transparent=int(bg),
                    border=1,
                    padding=SPRITE_PADDING,
                    padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
                )
            except ValueError:
                return False
            paste(grid, sprite, int(row_a) + int(sr), int(sc), transparent=int(bg))
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg, legend_colors, sprite_colors = _sample_palette()
    counts = _sample_counts()

    inp, expected = _new_grids(bg)
    legend_specs = []
    for color, count in zip(legend_colors, counts):
        width = _legend_width(count)
        legend_specs.append(
            (
                int(color),
                int(count),
                _build_input_legend(color, width),
                _build_output_legend(color, count, width),
            )
        )

    if not _place_legends(inp, expected, legend_specs, bg):
        return None

    region_colors = _sample_region_colors(counts, sprite_colors)
    region_sprites = _build_region_sprites(region_colors, bg)
    if not _place_region_sprites(inp, region_sprites, bg):
        return None

    if not np.array_equal(main(inp), expected):
        return None

    metrics = _combo_metrics(inp)
    if metrics["legend_count"] != REGION_COUNT:
        return None
    if metrics["spread_gap"] <= 0:
        return None
    if metrics["color_gap"] <= 0:
        return None
    if metrics["matches_color_baseline"] != 0:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid combo input for 58490d8a")
