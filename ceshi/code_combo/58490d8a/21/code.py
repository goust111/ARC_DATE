# Combo variant of 58490d8a (index 21)
# combo_spec: D3/1+D14/0+D15/0
# Name: 水平图例+排序+统一圆点色

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for combo D3/1 + D14/0 + D15/0
#
# a) Concrete behavior and trace
#    - Original task:
#      a black legend box stores key colors down one inner column; each key row
#      grows rightward with same-colored tally dots.
#    - Combined variant:
#      the key pixels are rotated into one top legend row at columns 1,3,5,...
#      each key color still points to same-colored outside sprites, but after
#      counting we sort the legend entries by count descending and redraw them as
#      a sorted top row of colored keys with white vertical tally columns below.
#    - Small trace:
#        input key row left->right = [red, blue, yellow]
#        outside counts            = [2, 5, 3]
#        original D3-only output   = red-height2, blue-height5, yellow-height3
#          kept in the same left->right key order and drawn in key colors
#        combo output              = blue-height5, yellow-height3, red-height2
#          with [blue,yellow,red] on the top key row and white columns below,
#          so the visible pattern is a descending skyline under sorted keys.
#    - The changed stages are therefore:
#      orientation swap -> measure per color -> sort by measured counts ->
#      discard color in the rendered tallies.
#
# b) Complexity calibration
#    - Counts stay in 1..5 and key count stays in 2..4, matching the base task.
#    - To make sorting visible despite the shared white tally color, the generator
#      samples distinct counts and rejects already-descending input orders.
#    - The layout keeps every-other-cell spacing, so multiple columns remain easy
#      to separate even when their top dots share the same color.
#
# c) Implicit-assumption audit of the original solver
#    - "leftmost non-black column holds all keys" is broken by D3/1; we must scan
#      for the topmost non-black row instead.
#    - "each key redraws onto its own row" is broken twice: D3/1 moves tallies to
#      columns, and D14/0 means output column identity is determined only after
#      sorting by count.
#    - "dot color equals key color" is broken by D15/0; only the tally dots turn
#      white, while the sorted key row stays colored so the legend remains readable.
#    - Legend detection as the largest black component with non-black pixels in
#      its bbox still holds, because the input legend remains one black rectangle
#      containing sparse colored holes.
#
# d) Placement legality
#    - Rotating the legend means width depends on K while height depends on the
#      maximum count: bw = 2*K+1 and bh = key_row + 2*max_count + tail + slack.
#    - The legend is pasted before sprites are placed, and sprite placement still
#      uses one-cell 8-neighborhood padding, so sprites cannot overlap or touch
#      the legend or each other.
#    - TALLY_COLOR is reserved for output only; key colors and background are
#      sampled from 1..9, so white tally cells never appear in the input.
#
# e) Visual observability
#    - The input visibly shows a horizontal key row, which distinguishes the D3
#      change from the source task.
#    - The output columns visibly descend left to right because counts are sorted.
#      This is human-visible and does not depend on raw numeric color ids.
#    - The shared white color removes redundant color hints from the tally cells,
#      but the mapping is still inferable because the sorted colored key row stays
#      visible and each input key color has a unique outside count.
#
# f) Visual cleanliness
#    - There is only one legend area, still isolated from the scattered sprites.
#    - Key pixels and tally cells occupy alternating coordinates, preventing
#      adjacent columns from visually merging into thick white blocks.
#    - Output keeps only one row of color plus white tally dots on black, making
#      the ranked column heights easy to read without losing legend identity.
#
# g) Change inferability and anti-degeneracy
#    - Because every tally column is white while the key row remains colored, the
#      color change is accompanied by strong structural cues: all output marks
#      live inside the legend crop, in ranked columns, with heights equal to
#      object counts.
#    - The generator forces distinct counts and a non-descending input order so
#      the sorted output cannot collapse back to the unsorted horizontal variant.
#    - Using white tally dots also guarantees the combo does not collapse back to
#      the original or D3-only colored-tally behaviors on generated samples.
#
# h) New solver assumptions and generator guarantees
#    - Assumption: the legend has at least one non-black row.
#      Guarantee: KEY_COUNT_RANGE remains 2..4.
#    - Assumption: counts fit vertically in the legend.
#      Guarantee: legend height is sized from max(counts) before input inversion.
#    - Assumption: sorting ties can be resolved deterministically by original
#      legend order.
#      Guarantee: generator uses distinct counts, but solver still keeps the
#      stable tie-break so the rule is well-defined beyond generated samples.
# -----------------------------------------------------------------------------


BLACK = 0
TALLY_COLOR = 10
INPUT_COLOR_POOL = tuple(range(1, 10))
DISTRACTOR_PROB = 0.5

# Key columns and tally dimensions for the rotated legend.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW = 1
KEY_COLUMN_FIRST = 1
KEY_COLUMN_STEP = 2
TALLY_ROW_STEP = 2
LEGEND_HEIGHT_TAIL = 2
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count keyed sprite colors, rank the tallies, and redraw them in white.

    Input structure:
    - A black rectangular legend box contains one horizontal row of key-color pixels on its topmost non-black row.
    - Outside the box, disconnected monochrome sprites of those key colors are scattered on a uniform background.

    Rule:
    - Count the outside connected components for each key color, then sort the key entries by count from largest to smallest, breaking ties by original left-to-right legend order.
    - Repaint the cropped legend box in black, place the sorted key colors back onto the top legend row, and place one white tally dot per counted component every other row below each ranked key.

    Key insight:
    - The input legend is read left-to-right only to decide which colors to count; the output reuses those column slots as rank positions, and the column heights carry all count information once color is discarded.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # STEP 1) Find the legend bbox: largest black CC whose bbox contains some non-black pixels.
    best = None  # (area, bbox)
    for coords in components_mask(g == BLACK, connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != BLACK).any():
            area = int(coords.shape[0])
            cand = (area, tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Inside the legend bbox, find the key row:
    # topmost row with any non-black pixels (the key-pixel row).
    key_row = None
    for rr in range(bh):
        if (box[int(rr), :] != BLACK).any():
            key_row = int(rr)
            break
    if key_row is None:
        return np.zeros_like(box)

    keys = [(int(cc), int(box[int(key_row), int(cc)])) for cc in range(bw) if int(box[int(key_row), int(cc)]) != BLACK]
    keys.sort(key=lambda t: int(t[0]))
    slot_cols = [int(cc) for cc, _col in keys]

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    # STEP 3) Count outside CCs for each key color while preserving the input legend order.
    records: list[tuple[int, int, int]] = []
    for cc, color in keys:
        mask = (g == int(color)) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(cc), int(color), int(count)))

    # STEP 4) Reassign columns by count rank, redraw the sorted colored key row,
    # then draw white tally columns below it.
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    for new_idx, (_orig_cc, color, count) in enumerate(records):
        cc = int(slot_cols[new_idx])
        out[int(key_row), int(cc)] = int(color)
        for i in range(int(count)):
            rr = int(key_row + TALLY_ROW_STEP * (i + 1))
            if 0 <= rr < bh:
                out[int(rr), int(cc)] = TALLY_COLOR

    return out


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
    """Choose background, key colors, and optional distractor color while reserving white for output only."""
    colors = list(INPUT_COLOR_POOL)
    bg = random.choice(colors)

    k = _randint(KEY_COUNT_RANGE)
    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if remaining and random.random() < DISTRACTOR_PROB else None
    return bg, key_colors, distractor_color


def _sample_counts(k: int) -> list[int]:
    """Sample distinct counts and force the input legend order to need sorting."""
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    counts = random.sample(choices, k)
    while counts == sorted(counts, reverse=True):
        random.shuffle(counts)
    return [int(v) for v in counts]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return rotated legend height and width for the key count and largest tally."""
    bh = KEY_ROW + TALLY_ROW_STEP * max_count + LEGEND_HEIGHT_TAIL + _randint(LEGEND_SLACK_RANGE)
    bw = KEY_COLUMN_STEP * k + LEGEND_WIDTH_TAIL
    return bh, bw


def _key_col(i: int) -> int:
    """Map a zero-based key index to its column inside the legend/output box."""
    return KEY_COLUMN_FIRST + KEY_COLUMN_STEP * i


def _sorted_key_records(key_colors: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    """Return (original_index, color, count) sorted by output rank."""
    records = [(int(i), int(color), int(count)) for i, (color, count) in enumerate(zip(key_colors, counts))]
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return records


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the sorted output: colored key row plus white tally columns."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for out_i, (_orig_i, _color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        cc = _key_col(out_i)
        expected[KEY_ROW, cc] = int(_color)
        for j in range(count):
            rr = KEY_ROW + TALLY_ROW_STEP * (j + 1)
            expected[rr, cc] = TALLY_COLOR
    return expected


def _build_unsorted_expected(shape: tuple[int, int], key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the colored-key output without sorting, for anti-degeneracy checks."""
    bh, bw = map(int, shape)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for i, (color, count) in enumerate(zip(key_colors, counts)):
        cc = _key_col(i)
        expected[KEY_ROW, cc] = int(color)
        for j in range(count):
            rr = KEY_ROW + TALLY_ROW_STEP * (j + 1)
            expected[rr, cc] = TALLY_COLOR
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return a black legend patch whose top key row preserves the unsorted input order."""
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[KEY_ROW, _key_col(i)] = int(color)
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
    """Run one output-first generation attempt and reject solver mismatches or combo collapse."""
    bg, key_colors, distractor_color = _sample_palette()
    counts = _sample_counts(len(key_colors))
    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
    unsorted_expected = _build_unsorted_expected(expected.shape, key_colors, counts)
    if np.array_equal(expected, unsorted_expected):
        return None

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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
