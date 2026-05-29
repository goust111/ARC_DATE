# Combo variant of 58490d8a (index 20)
# combo_spec: D1/1+D5/1
# Name: 尺寸匹配+面积决定间距

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
# Design analysis for the combo variant (size match + area-group width spacing)
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - The black legend box is still the destination/output canvas, but each key row is
#   now a colored horizontal bar instead of a single key pixel.
# - The bar length is the queried sprite area. Example: a 5-cell red bar means
#   "collect all outside components whose filled area is 5 pixels", regardless of
#   their color.
# - After that area match is found, spacing is no longer fixed. The dots for that row
#   use the bbox width shared by the matched area-group sprites. Example:
#   legend rows = red bar of length 5, teal bar of length 4.
#   outside components = two area-5 sprites of bbox width 3, three area-4 sprites of
#   bbox width 2, in mixed colors.
#   original task: would have matched by color and used step 2 everywhere.
#   combo task: red row gets 2 dots at cols 1,4; teal row gets 3 dots at cols 1,3,5.
# - The two changes interact in sequence rather than independently: area selects the
#   source group, then that same group supplies the spacing.
#
# b) Complexity calibration
# - Area categories stay in the small original range 4..7 so pixel counting remains
#   human-solvable.
# - Bbox widths are restricted to 2 or 3, which makes the spacing difference clearly
#   visible without making the legend too wide for 20..25 sized grids.
# - Each generated sample contains at least one width-2 row and one width-3 row, and
#   both have count >= 2, so the spacing rule is observable rather than latent.
# - Legend colors and outside sprite colors are disjoint, and outside colors are
#   reused across different area groups, preventing accidental fallback to color match.
#
# c) Audit of original main() assumptions that break here
# - Original assumption: the key is one colored pixel. Broken. The solver must now
#   read a horizontal same-color run length to recover the target area.
# - Original assumption: row color identifies which outside mask to count. Broken.
#   The solver must scan all outside components together and aggregate by area.
# - Original assumption: dot step is always 2. Broken. The solver must infer one
#   step per area group from matched components' bbox widths.
# - New combo constraint: all components in the same area group need a stable bbox
#   width, otherwise the row spacing would be ambiguous. The generator enforces this
#   by using one canonical shape per area group.
#
# d) Placement legality
# - The legend bar always starts at the same key column and leaves black space on the
#   right, so the surrounding black region remains a single detectable component.
# - Outside sprites are pasted with 8-neighborhood padding, so connected-component
#   counting on the union of all non-bg, non-black pixels remains valid.
# - Legend width is computed from both the longest bar and the widest needed tally row,
#   so no intended dots fall outside the legend bbox.
#
# e) Visual observability
# - A solver looking only at examples can see that legend rows differ by bar length,
#   while outside colors do not align with legend colors. That points toward area-
#   based grouping.
# - Samples also show two spacing regimes (2 and 3), tied to the matched sprite
#   shapes. This makes the second dimension visible on the canvas itself.
#
# f) Visual cleanliness
# - The legend remains uncluttered: one bar on each odd row, black everywhere else.
# - Outside sprites are small, isolated, and uniformly padded, so area and bbox width
#   are readable by eye.
#
# g) Inference safety / new solver assumptions
# - No sorting by hidden color ids is used; row order is just the visible legend order.
# - main() assumes the largest black component containing non-black cells is the legend,
#   that each legend row encodes one contiguous bar, and that each represented area has
#   at least one outside component with a consistent bbox width. generate_input()
#   constructs inputs to satisfy exactly those assumptions.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tally outside sprites by area and write the counts back into a legend box.

    Input structure:
    - A black rectangular legend box contains one colored horizontal bar on each key row.
    - Outside the box, disconnected monochrome sprites of mixed colors are scattered across the grid.

    Rule:
    - The length of a legend-row bar gives the target sprite area for that row.
    - Count all outside connected components with that area, measure their shared bbox width, then place one dot per matching component on the row using that width as the horizontal step.

    Key insight: The legend row does not name a color group anymore; it names an area group, and that matched area group also dictates the dot spacing through its sprite width.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    black = 0

    best = None  # (area, bbox)
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        box_bbox = bbox_coords(coords)
        sub = crop_bbox(g, box_bbox)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, box_bbox)))
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

    keys: list[tuple[int, int, int]] = []
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
    keys.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    area_counts: dict[int, int] = {}
    area_widths: dict[int, set[int]] = {}
    outside_mask = (g != int(bg)) & (g != int(black)) & outside
    for coords in components_mask(outside_mask, connectivity=8):
        area = int(coords.shape[0])
        comp_bbox = bbox_coords(coords)
        width = int(comp_bbox[3] - comp_bbox[1] + 1)
        area_counts[area] = int(area_counts.get(area, 0) + 1)
        if area not in area_widths:
            area_widths[area] = set()
        area_widths[area].add(width)

    out = np.full((bh, bw), BLACK, dtype=int)
    for rr, color, target_area in keys:
        count = int(area_counts.get(int(target_area), 0))
        widths = area_widths.get(int(target_area), set())
        if count <= 0 or not widths:
            continue
        step = int(max(widths))
        for i in range(count):
            cc = int(key_col + step * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(color)

    return out


COLOR_POOL = tuple(range(1, 10))
BLACK = 0

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 4)
AREA_CHOICES = (4, 5, 6, 7)
DISTRACTOR_PROB = 0.5

KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
LEGEND_HEIGHT_TAIL = 1
LEGEND_RIGHT_MARGIN = 2
LEGEND_SLACK_RANGE = (0, 3)

GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 500


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    """Connected masks that fully occupy their chosen bbox extent."""
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


@lru_cache(maxsize=None)
def _shape_refs(area: int, bbox_width: int) -> tuple[tuple[int, int, int], ...]:
    """Return references into _valid_masks() for masks with the requested area and width."""
    refs = []
    for h, w in SPRITE_BBOXES:
        if int(w) != int(bbox_width):
            continue
        masks = _valid_masks(h, w, int(area))
        for idx in range(len(masks)):
            refs.append((int(h), int(w), int(idx)))
    return tuple(refs)


@lru_cache(maxsize=None)
def _width_choices(area: int) -> tuple[int, ...]:
    widths = sorted({int(w) for _h, w, _idx in _shape_refs(int(area), 2)} | {int(w) for _h, w, _idx in _shape_refs(int(area), 3)})
    return tuple(widths)


def _sample_palette(k: int) -> tuple[int, list[int], list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    remaining = [c for c in colors if c != bg]
    legend_colors = random.sample(remaining, k)

    remaining = [c for c in remaining if c not in legend_colors]
    sprite_palette_size = min(len(remaining), _randint((2, 3)))
    sprite_colors = random.sample(remaining, sprite_palette_size)
    return bg, legend_colors, sprite_colors


def _sample_rows() -> tuple[list[int], list[int], list[int]]:
    k = _randint(KEY_COUNT_RANGE)
    areas = random.sample(list(AREA_CHOICES), k)

    step_by_area: dict[int, int] = {}
    area_for_step2 = random.choice([area for area in areas if 2 in _width_choices(int(area))])
    step_by_area[area_for_step2] = 2

    step3_candidates = [area for area in areas if 3 in _width_choices(int(area)) and area != area_for_step2]
    if not step3_candidates:
        return _sample_rows()
    step_by_area[random.choice(step3_candidates)] = 3

    for area in areas:
        if area not in step_by_area:
            step_by_area[area] = random.choice(_width_choices(int(area)))

    counts = [_randint(TARGET_COUNT_RANGE) for _ in areas]
    for step in (2, 3):
        idxs = [i for i, area in enumerate(areas) if int(step_by_area[area]) == int(step)]
        if idxs:
            pick = random.choice(idxs)
            counts[pick] = max(counts[pick], 2)

    steps = [int(step_by_area[area]) for area in areas]
    return areas, steps, counts


def _legend_shape(areas: list[int], counts: list[int], steps: list[int]) -> tuple[int, int]:
    bh = KEY_ROW_STEP * len(areas) + LEGEND_HEIGHT_TAIL
    needed = []
    for area, count, step in zip(areas, counts, steps):
        marker_need = KEY_COLUMN + int(area) + 1
        tally_need = KEY_COLUMN + int(step) * (int(count) - 1) + LEGEND_RIGHT_MARGIN
        needed.append(max(marker_need, tally_need))
    bw = max(needed) + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(
    legend_colors: list[int],
    areas: list[int],
    counts: list[int],
    steps: list[int],
) -> np.ndarray:
    bh, bw = _legend_shape(areas, counts, steps)
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count, step) in enumerate(zip(legend_colors, counts, steps)):
        rr = _key_row(i)
        for j in range(count):
            expected[int(rr), KEY_COLUMN + int(step) * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], legend_colors: list[int], areas: list[int]) -> np.ndarray:
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


def _sample_shape(area: int, step: int) -> tuple[int, int, np.ndarray]:
    refs = _shape_refs(int(area), int(step))
    h, w, idx = random.choice(refs)
    return int(h), int(w), _valid_masks(int(h), int(w), int(area))[int(idx)]


def _render_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    sprite = np.full(mask.shape, int(bg), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _build_sprite_queue(
    areas: list[int],
    counts: list[int],
    steps: list[int],
    sprite_colors: list[int],
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    row_shapes = [_sample_shape(int(area), int(step)) for area, step in zip(areas, steps)]

    for row_idx, ((_, _, mask), count) in enumerate(zip(row_shapes, counts)):
        start = random.randrange(len(sprite_colors))
        for j in range(count):
            color = sprite_colors[(start + row_idx + j) % len(sprite_colors)]
            sprites.append(_render_sprite(mask, int(color), int(bg)))

    remaining_areas = [area for area in AREA_CHOICES if area not in set(areas)]
    if remaining_areas and random.random() < DISTRACTOR_PROB:
        distractor_area = random.choice(remaining_areas)
        distractor_step = random.choice(_width_choices(int(distractor_area)))
        _h, _w, mask = _sample_shape(int(distractor_area), int(distractor_step))
        sprites.append(_render_sprite(mask, random.choice(sprite_colors), int(bg)))

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
    areas, steps, counts = _sample_rows()
    bg, legend_colors, sprite_colors = _sample_palette(len(areas))

    expected = _build_expected(legend_colors, areas, counts, steps)
    legend = _build_legend(expected.shape, legend_colors, areas)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(areas, counts, steps, sprite_colors, bg)
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
