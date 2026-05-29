# Combo variant of 58490d8a (index 12)
# combo_spec: D10/0+D15/0
# Name: 紧密行排列+统一圆点色

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for combo D10/0 + D15/0
#
# fused_rule_summary:
#   - Keep the original counting task: for each key color, count outside connected
#     components of that color.
#   - Fuse two changes at once:
#       1. key rows are vertically packed with no blank separator rows
#       2. every tally dot in the output uses one fixed shared color
#
# concrete_behavior_and_trace:
#   - Small example with K=3 key colors and counts [3, 1, 2]:
#       original input legend rows:    1, 3, 5
#       combo input legend rows:       1, 2, 3
#       original output row colors:    [blue blue blue], [red], [yellow yellow]
#       combo output row colors:       [9 9 9], [9], [9 9]
#   - What changes per stage:
#       * legend reading:
#           original sometimes visually suggested odd-row spacing
#           combo must read every non-black pixel found in the key column because
#           consecutive rows can all be active
#       * counting:
#           unchanged; each key color still maps to the number of outside 8-connected
#           components of that same color
#       * drawing:
#           original wrote the key color itself at columns 1,3,5,...
#           combo writes one reserved tally color at columns 1,3,5,... on the same
#           observed key row
#   - The two edits reinforce each other: because dot color no longer distinguishes
#     rows, the packed colored key column becomes the only visible row-to-color index.
#
# complexity_check:
#   - The task is not simplified into mere recoloring because solving still requires
#     per-color connected-component counting outside the legend.
#   - The fusion remains visually distinct from the base task:
#       * legend height shrinks from 2*K+1 to K+2
#       * output rows no longer inherit key colors
#   - Freedom stays controlled:
#       * K remains 2..4
#       * counts remain 1..5
#       * one fixed tally color is reused across all rows
#     so a few training examples give repeated support for the same pattern.
#
# implicit_assumption_audit_for_main:
#   - Largest black component with non-black pixels inside its bbox:
#       still valid because the legend is still a single black rectangle with a few
#       colored pixels punched into it.
#   - Leftmost non-black column inside the legend:
#       still valid because generator keeps all key pixels in one dedicated column.
#   - Collect all non-black entries from that column:
#       now essential; the combo cannot assume odd rows or any fixed vertical gap.
#   - Count outside components by the key color:
#       still valid because tally color is reserved for output only and never appears
#       in the input grid.
#   - Draw tallies at key_col + 2*i:
#       still valid because horizontal spacing is unchanged and width is still sized
#       from max_count.
#   - New fused assumption:
#       row identity in the output comes only from position, so the input must visibly
#       preserve distinct colored key pixels on consecutive rows; generator guarantees
#       this by using distinct key colors and by not inserting blank rows between them.
#
# placement_legality:
#   - The legend remains at least two cells from the grid edge.
#   - Sprites are placed with one-cell 8-connected padding, so they do not touch each
#     other or the legend box.
#   - Compacting the legend reduces required height, which makes placement easier and
#     does not create new overlap modes.
#   - The fixed tally color is excluded from the input palette, preventing accidental
#     fake objects of that color outside the legend.
#
# visual_observability_and_cleanliness:
#   - A human can see the fusion from 2-3 examples:
#       * colored key pixels form a compact stack with no empty spacer rows
#       * output rows are aligned to those positions
#       * all output dots share one color
#   - No invisible ordering rule is introduced; row order is simply top-to-bottom in
#     the visible key column.
#   - The black legend and isolated outside sprites keep the scene uncluttered.
#
# fusion_metric_for_post_test_verification:
#   - Metric 1: compactness ratio = gaps_between_key_rows / (K-1), which must be 0.
#   - Metric 2: output palette on non-black cells must have size 1 and equal the
#     reserved tally color.
#   - If either metric fails on generated samples, the generator has regressed toward
#     the base task or a single-dimension variant.
#
# new_solver_assumptions_and_generator_guarantees:
#   - assumption: at least two key rows exist so compact stacking is observable
#     guarantee: KEY_COUNT_RANGE is 2..4
#   - assumption: shared tally color differs from every input role
#     guarantee: INPUT_COLOR_POOL excludes TALLY_COLOR
#   - assumption: every key row should display at least one dot
#     guarantee: TARGET_COUNT_RANGE is 1..5
#   - assumption: solver can read rows directly from the legend instead of inferring
#     them from a formula
#     guarantee: generator always writes exactly one key pixel per active row in the
#     dedicated key column
# -----------------------------------------------------------------------------

BLACK = 0
TALLY_COLOR = 9


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites and rewrite the legend as compact shared-color tallies.

    Input structure:
    - A black rectangular legend box contains one colored key pixel in its leftmost non-black column, and those key rows are packed consecutively with no empty separator rows.
    - Outside the box, disconnected sprites of the key colors are scattered on a uniform background, with an optional distractor color.

    Rule:
    - For each key row, count the 8-connected components of that row's key color outside the legend box.
    - Return a black canvas the same size as the legend box, placing one tally dot per counted component at columns `key_col`, `key_col+2`, `key_col+4`, ... on the same row, using one fixed tally color for every row.

    Key insight:
    - The colored key column still tells you which outside color to count, but once the counts are known the output keeps only row position and quantity: consecutive key rows identify the rows, and a single shared color identifies every tally dot.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    best = None
    for coords in components_mask(g == BLACK, connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(g, bbox)
        if (sub != BLACK).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, bbox)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    key_col = None
    for cc in range(bw):
        if (box[:, cc] != BLACK).any():
            key_col = cc
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(rr, int(box[rr, key_col])) for rr in range(bh) if int(box[rr, key_col]) != BLACK]
    keys.sort(key=lambda item: item[0])

    outside = np.ones((H, W), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    for rr, color in keys:
        mask = (g == color) & outside
        _labels, count = label_mask(mask, connectivity=8)
        for i in range(int(count)):
            cc = key_col + 2 * i
            if 0 <= cc < bw:
                out[rr, cc] = TALLY_COLOR

    return out


# --------------- generator (o5x) ---------------

# Palette and legend color choices.
INPUT_COLOR_POOL = tuple(range(1, TALLY_COLOR))
DISTRACTOR_PROB = 0.5

# Key rows and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 1
KEY_COLUMN = 1
TALLY_COL_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
LEGEND_SLACK_RANGE = (0, 3)

# Canvas size and legend placement bounds.
GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

# Sprite shape and placement constraints.
SPRITE_AREA_RANGE = (4, 7)
SPRITE_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3))
SPRITE_CONNECTIVITY = 8
SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8

# Retry budget for rejection sampling.
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
        _labels, count = label_mask(mask, connectivity=SPRITE_CONNECTIVITY)
        if count == 1:
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
    colors = list(INPUT_COLOR_POOL)
    bg = random.choice(colors)

    key_count = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, key_count)
    remaining = [c for c in remaining if c not in key_colors]
    distractor = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor


def _sample_counts(k: int) -> list[int]:
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_FIRST + k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(counts), max_count)
    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, count in enumerate(counts):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = TALLY_COLOR
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> np.ndarray | None:
    bh, bw = map(int, legend.shape)
    grid_h = _randint(GRID_HEIGHT_RANGE)
    grid_w = _randint(GRID_WIDTH_RANGE)
    min_h = bh + 2 * LEGEND_EDGE_MARGIN
    min_w = bw + 2 * LEGEND_EDGE_MARGIN
    if grid_h < min_h or grid_w < min_w:
        return None

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    r0 = random.randint(LEGEND_EDGE_MARGIN, grid_h - bh - LEGEND_EDGE_MARGIN)
    c0 = random.randint(LEGEND_EDGE_MARGIN, grid_w - bw - LEGEND_EDGE_MARGIN)
    paste(grid, legend, r0, c0)
    return grid


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
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(counts)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, distractor_color, bg)
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
