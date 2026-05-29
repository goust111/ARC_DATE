# Variant of 58490d8a
# Dimension: D5 — 圆点放置间距 (transform)
# Change: 相邻圆点之间间隔1个背景像素（步长为2），即放置在关键列、关键列+2、关键列+4等位置... → 间距由精灵尺寸决定

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the D5 variant
# ----------------------------------
# variant_spec:
#   core_change:
#   - The legend still lists key colors in one column and still encodes counts as repeated dots.
#   - The only rule change is the horizontal step between consecutive dots on one row:
#     it is now the bbox width of that row's source sprite color, not a fixed value 2.
#   concretization:
#   - Outside the black legend, each key color still appears as 1-5 disconnected monochrome sprites.
#   - All sprites of one key color are still copies of one canonical shape so their bbox width is stable.
#   - Allowed bbox widths are only 2 or 3, taken directly from the visible horizontal extent of the sprite.
#   - Output row i places dots at key_col + j * sprite_width(color_i).
#   traced_example:
#   - Suppose red has 4 outside components and each red sprite spans 3 columns.
#   - Original task would draw red at columns 1,3,5,7.
#   - This variant must draw red at columns 1,4,7,10.
#   - Blue with width 2 still uses columns 1,3,5..., so one sample can show both behaviors side by side.
#   complexity_calibration:
#   - Width choices are restricted to {2, 3}; this keeps the latent relation small and learnable.
#   - Every generated grid contains both widths, and at least one width-2 row and one width-3 row have count >= 2,
#     so the spacing difference is visually observable instead of hiding behind single-dot rows.
#   implicit_assumption_review:
#   - In the original solver, `key_col + 2 * i` hard-coded the old step and is invalid here.
#   - In the original generator, legend width and expected output also assumed a fixed step 2.
#   - The solver now infers step per color from outside components; the generator guarantees one stable width per color.
#   placement_legality:
#   - Legend width is derived from the largest needed final dot column under the new per-row steps.
#   - Sprite placement still uses the same non-overlap and padding constraints, so components stay separable.
#   visual_cleanliness:
#   - The legend keeps one row per key color and black background everywhere else; no output rows overlap.
#   - Sprites remain small isolated blobs, so their bbox widths are easy to read by eye.
#   color_rule:
#   - No color remapping is introduced; colors still only identify which outside sprites feed which legend row.
#   solver_assumptions_to_match_generator:
#   - Each key color has at least one outside component.
#   - All outside components of one key color share the same bbox width.
#   - The largest black connected component containing non-black pixels is the legend box.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count keyed objects and write the tallies back into the legend box.

    Input structure:
    - A black rectangular legend box contains one colored key pixel on every other row in its leftmost non-black column.
    - Outside that box, each key color appears as several disconnected same-color sprites whose widths are visibly measurable.

    Rule:
    - For each key color, count its separate outside sprites and measure that color's sprite bbox width.
    - Repaint the legend box as black, then place one dot per counted sprite on the matching key row, stepping right by that measured width each time.

    Key insight: The outside objects provide two signals at once for each legend row — how many dots to draw, and how far apart the dots must be.
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
    keys.sort(key=lambda t: int(t[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, col in keys:
        comps = components_mask((g == int(col)) & outside, connectivity=8)
        if not comps:
            continue
        widths = [int(bbox_coords(comp)[3] - bbox_coords(comp)[1] + 1) for comp in comps]
        step = int(max(widths))
        for i in range(len(comps)):
            cc = int(key_col + step * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

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
STEP_CHOICES = (2, 3)
LEGEND_HEIGHT_TAIL = 1
LEGEND_RIGHT_MARGIN = 2
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


def _make_sprite(color: int, bg: int, bbox_width: int) -> np.ndarray:
    options = []
    for h, w in SPRITE_BBOXES:
        if w != int(bbox_width):
            continue
        for area in range(SPRITE_AREA_RANGE[0], min(SPRITE_AREA_RANGE[1], h * w) + 1):
            if _valid_masks(h, w, area):
                options.append((h, w, area))
    h, w, area = random.choice(options)
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


def _sample_steps(k: int) -> list[int]:
    steps = [2, 3]
    while len(steps) < k:
        steps.append(random.choice(STEP_CHOICES))
    random.shuffle(steps)
    return steps


def _sample_counts(steps: list[int]) -> list[int]:
    counts = [_randint(TARGET_COUNT_RANGE) for _ in steps]
    for step in STEP_CHOICES:
        idxs = [i for i, value in enumerate(steps) if value == step]
        pick = random.choice(idxs)
        counts[pick] = max(counts[pick], 2)
    return counts


def _legend_shape(k: int, counts: list[int], steps: list[int]) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    needed = [KEY_COLUMN + step * (count - 1) + LEGEND_RIGHT_MARGIN for step, count in zip(steps, counts)]
    bw = max(needed) + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int], steps: list[int]) -> np.ndarray:
    bh, bw = _legend_shape(len(key_colors), counts, steps)
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count, step) in enumerate(zip(key_colors, counts, steps)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + step * j] = int(color)
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
    steps: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for color, count, step in zip(key_colors, counts, steps):
        sprite = _make_sprite(color, bg, bbox_width=step)
        for _ in range(count):
            sprites.append(sprite.copy())
    if distractor_color is not None:
        sprites.append(_make_sprite(distractor_color, bg, bbox_width=random.choice(STEP_CHOICES)))
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
    steps = _sample_steps(len(key_colors))
    counts = _sample_counts(steps)
    expected = _build_expected(key_colors, counts, steps)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, steps, distractor_color, bg)
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
