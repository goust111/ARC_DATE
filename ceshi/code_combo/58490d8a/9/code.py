# Combo variant of 58490d8a (index 9)
# combo_spec: D5/1+D10/0
# Name: 面积决定间距+紧密行排列

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
#   - The black legend is still the destination box, but its key rows are now packed
#     consecutively at rows 1,2,3,... instead of 1,3,5,...
#   - The outside sprites still determine per-color counts, and now also determine the
#     horizontal tally step: the row uses that color's sprite bbox width.
#   - Example with three colors:
#       original rows/steps: rows [1,3,5], all rows use columns 1,3,5,...
#       combo rows/steps:    rows [1,2,3], red width 3 -> cols 1,4,7,...;
#                            blue width 2 -> cols 1,3,5,...; green width 3 similarly.
#   - So the fused change is visible in two places at once: the legend is shorter, and
#     different rows can have different dot spacing.
# b) complexity_calibration:
#   - Step values are restricted to bbox widths {2, 3}; this is small enough to infer
#     from a few examples and large enough to produce visually distinct rows.
#   - Every generated sample contains both widths, and for each width at least one row
#     has count >= 2, so the spacing evidence is not hidden behind single-dot rows.
#   - Key count remains 2..4, preserving repeated evidence without overcrowding the grid.
# c) implicit_assumption_review_for_main:
#   - Legend detection via "largest black connected component whose bbox contains
#     non-black pixels" still holds because the compact legend remains one black region.
#   - Key-column detection only relies on the leftmost non-black column in the legend,
#     so consecutive key rows do not break it.
#   - The original hard-coded expression `key_col + 2 * i` is no longer valid; the
#     solver must infer one stable bbox width per key color from outside components.
#   - Using the observed key-row positions directly is important: the solver must not
#     reconstruct odd rows or assume blank separators.
# d) placement_legality:
#   - The generator sizes legend width from the largest required final dot column under
#     the inferred per-row step, so every tally fits.
#   - Sprites are placed with one-cell 8-connected padding from each other and from the
#     legend, preserving clean component counting and bbox measurement.
# e) visual_observability:
#   - Compact rows are visible from the input alone because all key pixels occupy
#     consecutive rows and the legend height drops from about 2K+1 to K+2.
#   - Variable spacing is visible across output rows because samples always include both
#     width-2 and width-3 colors with repeated dots.
# f) visual_cleanliness:
#   - Each color owns exactly one legend row, the background stays uniform, and the
#     output remains black except for the tally dots.
#   - Small isolated sprites keep width reading simple and avoid overlapping cues.
# g) inferability_of_changes:
#   - No hidden color remapping is introduced; colors only match source sprites to
#     legend rows.
#   - The only latent numerical relation is "step equals sprite bbox width", supported
#     by repeated same-width copies per color and by both width categories appearing.
# h) new_solver_assumptions_and_generator_guarantees:
#   - Each key color has at least one outside component.
#   - All components of one key color share a canonical sprite shape, hence one bbox width.
#   - Width 2 and width 3 both appear among key colors in every sample.
#   - At least one width-2 row and one width-3 row have counts >= 2, ensuring the combo
#     differs from the fixed-step original behavior on generated examples.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the keyed sprites and rewrite the compact legend with width-based tallies.

    Input structure:
    - A black rectangular legend box contains one colored key pixel per consecutive key row in its leftmost non-black column.
    - Outside the box, each key color appears as several disconnected monochrome sprites, and the sprite width for a given color is consistent across its copies.

    Rule:
    - For each key color, count the outside connected components of that color and measure their bbox width.
    - Repaint the legend box black and place one dot per component on the matching key row, advancing rightward by that measured width each time.

    Key insight:
    - The outside sprites encode two facts simultaneously: how many marks a row needs, and the stride between those marks; the compact legend simply reads row order directly from the observed key pixels.
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


COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 1
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
    bh = KEY_ROW_FIRST + KEY_ROW_STEP * (k - 1) + LEGEND_HEIGHT_TAIL + 1
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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
