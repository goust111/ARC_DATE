# Variant of 58490d8a
# Dimension: D14 — 多步处理：计数与排序 (multi_step)
# Change: 单步处理 — 直接统计各颜色连通分量数量并填入对应行... → 先计数后按数量排序重排图例行

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D14 variant
#
# a) Concrete behavior and trace
#    - The input still contains one black legend rectangle and several scattered
#      same-color sprites outside it.
#    - Inside the legend, colored key pixels appear on alternating rows in some
#      original top-to-bottom order. That order is now only the "measurement"
#      order, not the final output order.
#    - Example trace:
#        input legend rows = [blue, red, yellow]
#        outside component counts = [2, 5, 3]
#        original task output rows = [blue x2, red x5, yellow x3]
#        new variant output rows = [red x5, yellow x3, blue x2]
#      The changed stage is therefore after counting and before drawing.
#
# b) Complexity calibration
#    - The new rule adds one explicit intermediate reasoning step: rank the key
#      colors by an observable property (their counted sprite totals).
#    - To keep this visually recoverable, the generator uses distinct counts and
#      rejects already-sorted legends, so the row permutation is visible in most
#      samples instead of collapsing back to the original behavior.
#
# c) Review of old solver assumptions
#    - Old assumption: each key color is drawn back onto its own input row.
#      This no longer holds; row identity must be separated from color identity.
#    - Old loop `for rr, col in keys` was safe only because output row == input
#      row. In the variant, we must first collect `(orig_row, color, count)`,
#      then sort those records, then redraw onto the ordered slot rows.
#    - The legend detection, key-column detection, outside-mask construction,
#      and connected-component counting remain valid and are intentionally kept.
#
# d) Placement legality
#    - Sprites are still placed with one-cell 8-neighborhood padding, so same-
#      color copies remain separate connected components and distractors do not
#      merge into keys.
#    - The legend remains inset from the canvas edges and is pasted before
#      sprites are placed, so random placement cannot overlap the legend.
#
# e) Visual observability
#    - Sorting never depends on invisible values such as raw color ids.
#    - The ranking key is the visible tally length in the output.
#    - Ties are broken by original legend order for determinism, but the
#      generator mostly avoids ties so the primary sorting rule is easy to infer.
#
# f) Visual cleanliness
#    - The output remains a black rectangular tally board with one colored row
#      per legend slot, so no new visual layers are added.
#    - Objects outside the legend are still sparse and non-touching, preserving
#      easy counting by eye.
#
# g) Color-rule inferability
#    - Colors are not remapped. A color simply moves to a new legend row when
#      its count rank changes. This is structurally signaled by the same color
#      appearing in the reordered tally row.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count legend colors, then rank their tally rows by frequency.

    Input structure:
    - A black rectangular legend patch contains one colored key pixel on each alternating row.
    - Outside the patch, disconnected monochrome sprites of the key colors are scattered on the background.

    Rule:
    - Count how many outside connected components belong to each key color.
    - Sort the key colors by count from largest to smallest, breaking ties by their original top-to-bottom legend order, then draw each tally row in that ranked order.

    Key insight:
    - The legend rows are not fixed destinations anymore: first use them to identify which colors to count, then reuse the same row slots as ranking positions.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
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

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Inside the legend bbox, find the key column:
    # leftmost column with any non-black pixels (the key-pixel column).
    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    keys.sort(key=lambda t: int(t[0]))
    slot_rows = [int(rr) for rr, _col in keys]

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    # STEP 3) Count outside CCs for each key color while keeping original row order.
    records: list[tuple[int, int, int]] = []
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(rr), int(col), int(count)))

    # STEP 4) Reassign legend rows by count rank, then draw 1 dot per counted CC.
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    for new_idx, (_orig_rr, col, count) in enumerate(records):
        rr = int(slot_rows[new_idx])
        for i in range(int(count)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (D14 variant) ---------------

# Palette and legend color choices.
COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

# Key rows and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
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
    """Sample distinct target counts and force input legend order to need reordering."""
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    counts = random.sample(choices, k)
    while counts == sorted(counts, reverse=True):
        random.shuffle(counts)
    return [int(v) for v in counts]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the key count and largest tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


def _key_row(i: int) -> int:
    """Map a zero-based key index to its row inside the legend/output box."""
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _sorted_key_records(key_colors: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    """Return (original_index, color, count) sorted by output rank."""
    records = [(int(i), int(color), int(count)) for i, (color, count) in enumerate(zip(key_colors, counts))]
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return records


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the sorted black output tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for out_i, (_orig_i, color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        rr = _key_row(out_i)
        for j in range(count):
            expected[rr, KEY_COLUMN + TALLY_COL_STEP * j] = int(color)
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch whose rows preserve the unsorted input order."""
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

    solved = main(grid)
    if np.array_equal(solved, legend):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a D14 input")
