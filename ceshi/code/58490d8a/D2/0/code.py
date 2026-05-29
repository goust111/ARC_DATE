# Variant of 58490d8a
# Dimension: D2 — 源对象：框外精灵的定义 (src)
# Change: 框外各关键颜色的独立连通分量（8连通），每种颜色的所有精灵形状相同，外接矩形2x2至3x3，面积4-7像素... → 多色精灵替代单色精灵

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location


# Design analysis for the D2 variant
# ----------------------------------
# concrete_behavior:
#   legend:
#     unchanged: the black legend box still contains one key pixel per row and the
#       output is still a black tally canvas with evenly spaced dots.
#   src_objects:
#     changed: each counted outside object is now a two-color sprite. The dominant
#       color is the legend-matching "body" color; a smaller accent color acts as a
#       bridge joining body fragments into one 8-connected object.
#     visible_templates:
#       - 2x3 rectangle with a 2-cell accent stripe in the middle column
#       - 3x2 rectangle with a 2-cell accent stripe in the middle row
#       - 3x3 sparse shape with two dominant columns plus one accent center
#       - 3x3 sparse shape with two dominant rows plus one accent center
#     effect_on_reasoning:
#       the old solver counted connected components of the key color directly.
#       That is no longer valid because the dominant color is intentionally split
#       into two disconnected islands inside one multicolor sprite.
# example_trace:
#   original:
#     a 2x3 red rectangle counted as one red connected component.
#   variant:
#     the same bbox may contain red / blue / red on both rows.
#     The full 6-pixel object is one sprite, but the red pixels form two separate
#     8-connected red components. Counting red-only CCs would wrongly return 2;
#     counting whole-object CCs first and then reading the dominant color returns 1.
# complexity_calibration:
#   why_not_too_simple:
#     if the accent were just a harmless decoration attached to a still-connected
#     dominant region, the original logic would continue to work and the variant
#     would be visually different but logically degenerate.
#   why_not_too_free:
#     each sprite still uses only two colors, small 2x2..3x3 footprints, one
#     canonical template per primary color, and strict 8-gap placement. The only
#     new inference is "group whole object first, then read majority color".
# implicit_assumption_audit:
#   old_main_dependencies:
#     - legend detection assumes black appears only in the legend box. Kept true.
#     - key extraction assumes the first non-black legend column stores the keys.
#       Kept true.
#     - outside counting previously assumed every counted sprite was monochrome and
#       could be recovered by masking one color. Broken by the variant and replaced
#       with all-color object extraction plus dominant-color classification.
#   new_main_dependencies:
#     - every outside counted sprite is one 8-connected non-bg object. The generator
#       enforces padding=1 so objects never touch each other or the legend.
#     - each sprite has a unique dominant color count. Templates guarantee body >
#       accent, so majority-color classification is stable.
# placement_legality:
#   generator uses random_free_location with transparent=bg and 8-connected padding=1.
#   This prevents overlap, touching, and legend contact while allowing background
#   holes inside 3x3 sparse templates.
# observability_and_cleanliness:
#   every counted sprite is visibly two-colored, and the dominant color is repeated
#   enough times to be recognizable. The accent is small but structurally essential,
#   so the samples look neat while still forcing a new parsing step.
# adaptive_non_d2_tweak:
#   legend right slack is widened to 2..3 cells instead of 0..3. This keeps the
#   D2 change observable even when a true count is 1: an obsolete monochrome solver
#   would try to place an extra tally dot, and the box is wide enough for that error
#   to become visible rather than silently clipping away.


def _legend_bbox(grid: np.ndarray, black: int) -> tuple[int, int, int, int]:
    """Return the legend-box bbox as the largest black CC whose bbox encloses key pixels."""
    best = None
    for coords in components_mask(grid == int(black), connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(grid, b)
        if (sub != int(black)).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or area > best[0]:
                best = cand
    assert best is not None
    return tuple(map(int, best[1]))


def _outside_sprite_components(grid: np.ndarray, outside: np.ndarray, bg: int) -> list[np.ndarray]:
    """Return full multicolor outside sprites as 8-connected non-background objects."""
    mask = outside & (grid != int(bg)) & (grid != BLACK)
    return components_mask(mask, connectivity=8)


def _dominant_color(grid: np.ndarray, coords: np.ndarray) -> int | None:
    """Return the unique majority color inside one multicolor sprite."""
    values = grid[coords[:, 0], coords[:, 1]]
    colors, counts = np.unique(values, return_counts=True)
    if colors.size == 0:
        return None
    max_count = int(counts.max())
    winners = colors[counts == max_count]
    if winners.size != 1:
        return None
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a legend box and tally dominant-color multicolor sprites.

    Input structure:
    - A black legend rectangle still stores the key colors in its first non-black internal column.
    - Outside the box, each counted source object is a two-color 8-connected sprite whose larger body color matches one legend key.

    Rule:
    - Extract each outside sprite as a whole object, infer its dominant/body color, and count how many sprites belong to each legend color.
    - Return a black tally panel matching the legend box size, placing one evenly spaced dot per counted sprite on the corresponding key row.

    Key insight:
    - The accent pixels are structural glue, not separate items: group the full two-color sprite first, then read its dominant color to decide which key row gains a dot.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    r0, c0, r1, c1 = _legend_bbox(g, BLACK)
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != BLACK).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != BLACK]
    keys.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False

    counts_by_color = {int(color): 0 for _, color in keys}
    for coords in _outside_sprite_components(g, outside, bg):
        dominant = _dominant_color(g, coords)
        if dominant in counts_by_color:
            counts_by_color[int(dominant)] += 1

    out = np.zeros((bh, bw), dtype=int)
    for rr, color in keys:
        for i in range(counts_by_color[int(color)]):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(color)
    return out


# --------------- generator (o5x) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (2, 3)

GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 300

SPRITE_TEMPLATES = (
    np.array([[1, 2, 1], [1, 2, 1]], dtype=int),
    np.array([[1, 1], [2, 2], [1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 2, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 2, 0], [1, 1, 1]], dtype=int),
)


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


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
    if H < bh + 2 * LEGEND_EDGE_MARGIN or W < bw + 2 * LEGEND_EDGE_MARGIN:
        return None
    grid = np.full((H, W), int(bg), dtype=int)
    r0 = random.randint(LEGEND_EDGE_MARGIN, H - bh - LEGEND_EDGE_MARGIN)
    c0 = random.randint(LEGEND_EDGE_MARGIN, W - bw - LEGEND_EDGE_MARGIN)
    paste(grid, legend, r0, c0)
    return grid


def _sample_palette() -> tuple[int, list[int], int | None, list[int]]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    primary_colors = list(key_colors)
    if distractor_color is not None:
        primary_colors.append(int(distractor_color))
    accent_pool = [c for c in colors if c not in {bg, *primary_colors}]
    return bg, key_colors, distractor_color, accent_pool


def _sample_counts(k: int) -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _make_sprite(primary: int, accent: int, bg: int) -> np.ndarray:
    template = random.choice(SPRITE_TEMPLATES)
    sprite = np.full(template.shape, int(bg), dtype=int)
    sprite[template == 1] = int(primary)
    sprite[template == 2] = int(accent)
    return sprite


def _build_sprite_queue(
    key_colors: list[int],
    counts: list[int],
    distractor_color: int | None,
    accent_pool: list[int],
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for color, count in zip(key_colors, counts):
        accent = random.choice(accent_pool)
        sprite = _make_sprite(int(color), int(accent), int(bg))
        for _ in range(count):
            sprites.append(sprite.copy())
    if distractor_color is not None:
        accent = random.choice(accent_pool)
        sprites.append(_make_sprite(int(distractor_color), int(accent), int(bg)))
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
    bg, key_colors, distractor_color, accent_pool = _sample_palette()
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, accent_pool, bg)
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
