# Combo variant of 58490d8a (index 13)
# combo_spec: D6/1+D15/0
# Name: 中间关键列+统一圆点色
#
# Design analysis
# ---------------
# combo_variant_spec:
#   summary:
#     - Keep the original task's core counting problem: each colored key in the
#       legend refers to same-colored connected sprites outside the legend box.
#     - Fuse two changes at once:
#       1. the key pixels move from the left-side key column to the legend's
#          horizontal center column;
#       2. every output tally dot uses one shared tally color instead of the key
#          color from that row.
#   concrete_behavior_and_trace:
#     - Small example:
#       - Suppose the legend has three key rows with key colors [2, 4, 7].
#       - Outside-sprite component counts are [5, 1, 3].
#       - Original task would output row dots at columns
#         [1, 3, 5, 7, 9], [1], [1, 3, 5], colored respectively as
#         [2...], [4], [7...].
#       - This combo instead places the key pixels at the legend center column,
#         say col 5, and outputs:
#         row1 -> cols [5, 3, 7, 1, 9]
#         row2 -> cols [5]
#         row3 -> cols [5, 3, 7]
#         with every dot recolored to one fixed tally color.
#     - Therefore the counting stage is unchanged, but the rendering stage now
#       depends on both row position and center-out geometry rather than a
#       row-specific color trail starting from the left.
#   complexity_calibration:
#     - To make the bilateral expansion visually obvious and easy to infer from
#       only a few ARC examples, the generator restricts counts to odd values
#       {1, 3, 5}. That yields exact left/right symmetry around the center key.
#     - Each generated task requires at least one row with count > 1 and at
#       least two distinct counts inside the same legend, so the center-out rule
#       and the "count controls length" rule are both observable.
#     - The shared tally color is reserved away from all input roles, so the
#       fusion cannot visually collapse back into the original per-row coloring.
#   implicit_assumption_audit:
#     - Original assumption: "the key column is the leftmost non-black column in
#       the legend bbox". This no longer holds; the solver must instead detect
#       the legend's non-black column and use it as the central anchor.
#     - Original assumption: "tally dots keep the key color". This is broken by
#       D15/0; the drawing phase must always emit TALLY_COLOR.
#     - Original assumption: "writing key_col + 2*i always means growing to the
#       right". This is broken by D6/1; placement must alternate left/right
#       around the center: center, center-2, center+2, ...
#     - New solver assumption: there is exactly one non-black legend column.
#       Generator guarantee: the legend patch contains colored pixels only in
#       the center column, nowhere else inside the black box.
#     - New solver assumption: the center-out sequence fits inside the box.
#       Generator guarantee: width is odd and chosen from the farthest required
#       radius plus symmetric side slack.
#   placement_legality:
#     - Legend height still follows 2*K+1 with K in 2..4.
#     - Legend width is the minimal odd width for the max odd count, plus equal
#       slack on both sides, which preserves the exact horizontal center.
#     - External sprites still use random_free_location with padding=1 and
#       8-connectivity, so sprites do not overlap or touch one another or the
#       legend box.
#   visual_observability_and_cleanliness:
#     - The input visibly differs from the original because the only colored
#       legend pixels now sit on the middle vertical axis rather than near the
#       left edge.
#     - The output visibly differs from both the original and the single D6
#       variant because every row shares one tally color, forcing the solver to
#       use row alignment plus symmetry, not output hue, to identify counts.
#     - The scene stays clean: the legend remains a sparse black box, the
#       outside sprites remain separated, and the output remains one sparse dot
#       row per key.
#   post_change_solver_audit:
#     - Loop over legend columns: assumes at least one column contains non-black
#       pixels; generator guarantees K >= 2 key rows and all of them occupy the
#       center column.
#     - Loop over keys: assumes each key color is distinct; generator samples key
#       colors without replacement.
#     - Component counting: assumes distractor colors do not equal key colors or
#       TALLY_COLOR; generator samples them from the remaining input palette.
#     - Output writing: assumes counts are positive; generator samples only from
#       {1, 3, 5}, so each row has at least one center dot.

from __future__ import annotations

import random
import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


BLACK = 0
TALLY_COLOR = 9
INPUT_COLOR_POOL = tuple(range(1, TALLY_COLOR))
DISTRACTOR_PROB = 0.5

# Key rows and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_CHOICES = (1, 3, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
LEGEND_HEIGHT_TAIL = 1
LEGEND_SIDE_SLACK_RANGE = (0, 2)

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
    """Return an inclusive randint from a configured (low, high) pair."""
    return random.randint(bounds[0], bounds[1])


def _tally_positions(center_col: int, count: int) -> list[int]:
    """Return center-out tally columns around center_col."""
    cols = [int(center_col)]
    radius = 1
    while len(cols) < int(count):
        cols.append(int(center_col - 2 * radius))
        if len(cols) < int(count):
            cols.append(int(center_col + 2 * radius))
        radius += 1
    return cols


def _legend_min_width(max_count: int) -> int:
    """Smallest odd width that fits the farthest center-out tally dot."""
    farthest_offset = 2 * (int(max_count) // 2)
    return 2 * farthest_offset + 1


@lru_cache(maxsize=None)
def _valid_masks(h: int, w: int, area: int) -> tuple[np.ndarray, ...]:
    """Return connected masks that span all four sides of the sprite bbox."""
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
    """Sample one connected same-color sprite drawn on the background color."""
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


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _key_col(width: int) -> int:
    """Return the center column of an odd-width legend."""
    return int(width) // 2


def _sample_palette() -> tuple[int, list[int], int | None]:
    """Choose background, key colors, and optional distractor color."""
    colors = list(INPUT_COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color


def _sample_counts(k: int) -> list[int]:
    """Sample odd counts that make the center-out rule visible inside each task."""
    while True:
        counts = [random.choice(TARGET_COUNT_CHOICES) for _ in range(k)]
        if any(count > 1 for count in counts) and len(set(counts)) >= min(2, k):
            return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and odd width for the centered tally layout."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = _legend_min_width(max_count) + 2 * _randint(LEGEND_SIDE_SLACK_RANGE)
    return bh, bw


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by centered counts."""
    del key_colors
    bh, bw = _legend_shape(len(counts), max(counts))
    center_col = _key_col(bw)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, count in enumerate(counts):
        rr = _key_row(i)
        for cc in _tally_positions(center_col, int(count)):
            expected[rr, int(cc)] = TALLY_COLOR
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with centered key pixels only."""
    legend = np.full(shape, BLACK, dtype=int)
    center_col = _key_col(shape[1])
    for i, color in enumerate(key_colors):
        legend[_key_row(i), center_col] = int(color)
    return legend


def _new_grid_with_legend(bg: int, legend: np.ndarray) -> np.ndarray | None:
    """Sample a grid and paste the legend away from all canvas edges."""
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
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    """Create canonical key sprites, optional distractor sprite, and shuffle them."""
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
    """Paste sprites into random non-touching free locations in queue order."""
    for sprite in sprites:
        sr, sc = random_free_location(
            grid,
            sprite,
            transparent=bg,
            padding=SPRITE_PADDING,
            padding_connectivity=SPRITE_PADDING_CONNECTIVITY,
        )
        paste(grid, sprite, sr, sc, transparent=bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites and rewrite the legend as a centered shared-color tally.

    Input structure:
    - A black rectangular legend box contains one colored key pixel on each active row, and all of those key pixels lie in the box's horizontal middle column.
    - Outside the box, separated monochrome sprites of the key colors are scattered on the background, with an optional distractor color that is not a key.

    Rule:
    - For each key row, count connected components of that row's key color outside the legend box.
    - Produce a black output canvas the same size as the legend box, then place one tally dot per counted component on that row using a single fixed tally color; start at the middle key column and expand every other cell outward in the order center, left, right, left, right.

    Key insight: The colored legend pixels still select which external color to count, but the result is encoded only by row position and by a bilateral center-out dot pattern in one shared color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
    best = None  # (area, bbox)
    for coords in components_mask(g == BLACK, connectivity=8):
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        sub = crop_bbox(g, bbox)
        if (sub != BLACK).any():
            area = int(coords.shape[0])
            candidate = (area, tuple(map(int, bbox)))
            if best is None or candidate[0] > best[0]:
                best = candidate

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) The fused variant uses a single centered key column inside the legend.
    nonblack_cols = [int(cc) for cc in range(bw) if (box[:, int(cc)] != BLACK).any()]
    if not nonblack_cols:
        return np.zeros_like(box)
    key_col = int(nonblack_cols[len(nonblack_cols) // 2])

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != BLACK]
    keys.sort(key=lambda item: int(item[0]))

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    # STEP 3/4) Count outside CCs per key color, then draw a center-out tally in one shared color.
    for rr, color in keys:
        mask = (g == int(color)) & outside
        _, nc = label_mask(mask, connectivity=8)
        for cc in _tally_positions(key_col, int(nc)):
            if 0 <= int(cc) < bw:
                out[int(rr), int(cc)] = TALLY_COLOR

    return out


def _gen_attempt() -> np.ndarray | None:
    """Run one output-first generation attempt and reject solver mismatches."""
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(key_colors, counts)
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
