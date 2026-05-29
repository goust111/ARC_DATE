# Variant of 58490d8a
# Dimension: D3 — 目标位置：图例框的结构 (dst)
# Change: 黑色矩形框，内部第2列（index=1）每隔一行放置一个关键颜色像素作为图例条目，输出画布即为该框的裁剪区域... → 多个分散图例框替代单一图例框

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# spec_variant:
#   core_change:
#     from: one large black legend rectangle whose crop is the whole output
#     to: several scattered black mini-legends, one per key color, kept in-place
#   input_structure:
#     background: single non-black color
#     legends:
#       count: K in 2-4
#       each_legend:
#         shape: solid black 3 x W rectangle with one key pixel inside
#         key_row: row index 1
#         key_col: column index 1
#         width: 2*target_count + 1 + slack, slack in 0-2
#         fill: all black except the key/tally row
#       placement:
#         legends are placed as separate connected black components
#         legends do not touch each other or any sprite (8-connectivity gap)
#         legends are intentionally spread across the full input grid rather than merged
#     sprites:
#       same as original task: for each key color, count 1-5 connected components outside legends
#       optional one distractor sprite of a non-key color
#   output_structure:
#     canvas: same size as the input grid
#     outside_area: reset to background
#     each legend: repainted in its original position as a black box whose middle row contains
#       one colored dot per counted same-color component, at columns 1,3,5,...
#   behavior_trace:
#     original:
#       1. find the largest black component
#       2. crop that one bbox
#       3. count objects by key color and draw inside the crop
#     variant:
#       1. find every black component whose bbox contains a one-row colored tally track
#       2. union all legend bboxes into an "inside legends" mask
#       3. for each legend independently, count outside components of its color
#       4. redraw every small legend in-place on a full-grid output
#     difference:
#       the old "largest black bbox" shortcut is invalid because no single box exists anymore
#   implicit_assumption_audit:
#     - Original main assumed the biggest black CC is always the destination frame. This fails now.
#     - Original main assumed all keys share one column inside one crop. This fails now.
#     - Original output size equaled the one legend bbox. This fails now because boxes are scattered.
#     - New main assumes every legend is a 3-row black component with all non-black cells on row 1,
#       all in a single color, and spaced every other column. Generator guarantees this.
#   legality_and_cleanliness:
#     - Legends are placed first with extra padding so they remain visually distinct.
#     - Sprites are then placed with padding against both legends and other sprites.
#     - Width depends on that legend's own target count, so every tally fits without overlap.
#   observability_check:
#     - Humans can see multiple separated black boxes in the input and the same boxes filled in output.
#     - The variant is not a hidden color-id trick: colors are only matched by visible legend pixels.
#     - Repeated examples show 2-4 boxes, so "multiple independent tally boxes" is directly inferable.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and write tallies into many scattered legend boxes.

    Input structure:
    - Several separate black mini-legends are scattered around the grid, each with one key-color pixel on its middle row.
    - Outside those boxes, disconnected objects of the key colors appear on the background.

    Rule:
    - For each mini-legend, count same-colored connected components outside every legend box.
    - Output a full-grid repaint: keep each black box in place and extend its middle-row tally rightward with evenly spaced dots.

    Key insight: Every small black box is its own local counting template, so the task is to detect all legend boxes first and then solve each color count independently on the shared scene.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    black = 0

    legends: list[tuple[tuple[int, int, int, int], int, int, int, int]] = []
    legends_mask = np.zeros((H, W), dtype=bool)

    # Each legend is a separate black CC whose bbox contains a single-row tally track.
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        bbox = tuple(map(int, bbox_coords(coords)))
        patch = crop_bbox(g, bbox)
        ph, pw = map(int, patch.shape)
        nonblack = np.argwhere(patch != int(black))
        if nonblack.size == 0 or ph != 3:
            continue

        rows = {int(r) for r, _c in nonblack}
        colors = {int(patch[int(r), int(c)]) for r, c in nonblack}
        if len(rows) != 1 or len(colors) != 1:
            continue

        key_row = next(iter(rows))
        color = next(iter(colors))
        cols = sorted(int(c) for _r, c in nonblack)
        key_col = cols[0]

        if key_row != 1 or key_col != 1:
            continue
        if any((cc - key_col) % 2 != 0 for cc in cols):
            continue

        legends.append((bbox, ph, pw, key_col, color))
        r0, c0, r1, c1 = bbox
        legends_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert legends

    out = np.full((H, W), int(bg), dtype=int)

    for bbox, ph, pw, key_col, color in legends:
        r0, c0, _r1, _c1 = bbox
        mask = (g == int(color)) & (~legends_mask)
        _lab, n = label_mask(mask, connectivity=8)

        patch = np.full((ph, pw), int(black), dtype=int)
        for i in range(int(n)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < pw:
                patch[1, cc] = int(color)
        paste(out, patch, r0, c0)

    return out


# --------------- generator ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
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
LEGEND_PADDING = 2

SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

GENERATION_ATTEMPTS = 400


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


def _sample_palette() -> tuple[int, list[int], int | None]:
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor


def _sample_counts(k: int) -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _legend_width(count: int) -> int:
    return TALLY_COL_STEP * count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)


def _build_input_legend(color: int, width: int) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, width), BLACK, dtype=int)
    patch[LEGEND_KEY_ROW, LEGEND_KEY_COL] = int(color)
    return patch


def _build_output_legend(color: int, count: int, width: int) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, width), BLACK, dtype=int)
    for j in range(count):
        patch[LEGEND_KEY_ROW, LEGEND_KEY_COL + TALLY_COL_STEP * j] = int(color)
    return patch


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
    # Place wider legends first to reduce rejection.
    for color, count, in_patch, out_patch in sorted(legend_specs, key=lambda t: t[2].shape[1], reverse=True):
        try:
            r0, c0 = random_free_location(
                inp,
                in_patch,
                transparent=bg,
                border=LEGEND_BORDER,
                padding=LEGEND_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        paste(inp, in_patch, r0, c0)
        paste(expected, out_patch, r0, c0)
    return True


def _place_sprites(grid: np.ndarray, sprites: list[np.ndarray], bg: int) -> bool:
    for sprite in sprites:
        try:
            sr, sc = random_free_location(
                grid,
                sprite,
                transparent=bg,
                border=1,
                padding=SPRITE_PADDING,
                padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return False
        paste(grid, sprite, sr, sc, transparent=bg)
    return True


def _legend_metrics(grid: np.ndarray) -> tuple[int, int]:
    # Metric for non-degeneracy: number of detected legend boxes and total blank area in their union bbox.
    bboxes = []
    total_area = 0
    for coords in components_mask(grid == BLACK, connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        patch = crop_bbox(grid, bbox)
        nonblack = np.argwhere(patch != BLACK)
        if patch.shape[0] == LEGEND_HEIGHT and nonblack.size > 0:
            rows = {int(r) for r, _c in nonblack}
            if rows == {LEGEND_KEY_ROW}:
                bboxes.append(bbox)
                total_area += patch.shape[0] * patch.shape[1]
    if not bboxes:
        return 0, 0
    r0 = min(b[0] for b in bboxes)
    c0 = min(b[1] for b in bboxes)
    r1 = max(b[2] for b in bboxes)
    c1 = max(b[3] for b in bboxes)
    union_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return len(bboxes), int(union_area - total_area)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg, key_colors, distractor = _sample_palette()
    counts = _sample_counts(len(key_colors))

    inp, expected = _new_grids(bg)

    legend_specs = []
    for color, count in zip(key_colors, counts):
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

    sprites = _build_sprite_queue(key_colors, counts, distractor, bg)
    if not _place_sprites(inp, sprites, bg):
        return None

    if not np.array_equal(main(inp), expected):
        return None

    n_legends, spread_gap = _legend_metrics(inp)
    if n_legends < 2 or spread_gap <= 0:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid D3 variant input for 58490d8a")
