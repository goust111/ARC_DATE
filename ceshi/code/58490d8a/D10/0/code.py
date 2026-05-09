# Variant of 58490d8a
# Dimension: D10 — 关键色行间距 (meta)
# Change: 关键色像素在图例框内的第1、3、5...行（奇数行），行间距固定为2（每隔一行留一空行）... → 关键色行紧密排列（无间隔行）

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, label_mask, paste, random_free_location

# Design analysis for the D10 variant
# ----------------------------------
# concrete_behavior:
#   - The legend box is still a black rectangle with one key-color column at x=1.
#   - The only visual change is vertical packing inside that box:
#       original rows: 1,3,5,...
#       variant rows:  1,2,3,...
#   - Output keeps the same horizontal tally rule: for each key row, place dots at
#     key_col, key_col+2, key_col+4, ... according to the count of outside components.
#
# behavior_trace:
#   - Original with K=3 used legend rows [1,3,5] and height 7.
#   - Variant with the same colors/counts uses rows [1,2,3] and height 5.
#   - Counting outside sprites is unchanged; only the row receiving each tally is
#     packed upward because the blank separator rows are removed.
#
# complexity_check:
#   - Keeping K in 2..4 preserves repeated evidence that multiple consecutive rows
#     can all be active, so the compact layout is visible in a few examples.
#   - Tally spacing, key column, palette policy, sprite shapes, and distractors stay
#     unchanged so the task remains the same conceptually rather than becoming a new task.
#
# implicit_assumption_audit_for_main:
#   - Legend detection via the largest black CC still works because the black box
#     remains one connected region even when colored key pixels occupy consecutive rows.
#   - Key-column detection only assumes "leftmost column inside the box containing
#     non-black pixels"; it does not depend on odd-row spacing, so it remains valid.
#   - Enumerating every non-black row in that column is exactly what we want for the
#     packed layout, because row positions are read from the input instead of inferred
#     from a fixed step size.
#   - Horizontal tally placement still assumes two-column spacing, which is unchanged
#     in this variant and is guaranteed by the width formula below.
#
# placement_legality:
#   - The generator keeps the legend at least two cells from the grid edge.
#   - Sprites are placed with one-cell 8-connected padding, so they do not touch the
#     legend or each other and the solver's outside-mask counting remains unambiguous.
#   - Reducing legend height makes placement easier; no new collisions are introduced.
#
# visual_observability:
#   - The compact legend height is directly visible: for K key colors, the legend now
#     has height K+2 instead of 2K+1, and the colored key rows appear consecutively.
#   - No invisible ordering trick is used; row order is simply the top-to-bottom order
#     seen in the key column.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colored objects and write the tallies into a compact legend box.

    Input structure:
    - A black rectangular legend box sits on a single-color background, and its leftmost non-black column holds one colored key pixel on each consecutive key row.
    - Outside that box, disconnected same-color sprites for the key colors are scattered across the grid, with an optional distractor color.

    Rule:
    - For each key color, count the connected components of that color outside the legend box.
    - Repaint the legend box as a black output canvas and place one tally dot per component on the matching key row, stepping rightward by two columns each time.

    Key insight:
    - The legend already tells you both which colors matter and which exact rows receive the answers; the compact variant removes spacer rows, but the row-reading logic is still driven entirely by the observed key pixels.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    _bg = int(get_bg(g))

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

    assert not (best is None)

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Inside the legend bbox, find the key column:
    # leftmost column with any non-black pixels (the "key pixels").
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

    # STEP 3/4) Count CCs of each key color outside the legend, then draw 1 dot per CC.
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labc, nc = label_mask(mask, connectivity=8)
        for i in range(int(nc)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (o5x) ---------------

# Palette and legend color choices.
COLOR_POOL = tuple(range(1, 10))
BLACK = 0
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
    """Return an inclusive randint from a configured (low, high) pair."""
    return random.randint(bounds[0], bounds[1])


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


def _sample_palette() -> tuple[int, list[int], int | None]:
    """Choose background, key colors, and optional distractor color."""
    colors = list(COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color


def _sample_counts(k: int) -> list[int]:
    """Sample the target outside-component count for each key color."""
    return [_randint(TARGET_COUNT_RANGE) for _ in range(k)]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest tally."""
    bh = KEY_ROW_FIRST + k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the black output tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        rr = _key_row(i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch with only the key-column pixels filled."""
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[_key_row(i), KEY_COLUMN] = int(color)
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
    raise RuntimeError("failed to generate a valid 58490d8a D10 input")
