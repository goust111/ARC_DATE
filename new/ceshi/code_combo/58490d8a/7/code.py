# Combo variant of 58490d8a (index 7)
# combo_spec: D5/2+D14/0
# Name: 递增间距+排序重排

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# Design analysis for the D5/2 + D14/0 combo
# a) Concrete behavior and trace:
#    - The input stays in the same object family as the original task: one black legend rectangle contains
#      key-color pixels in a single internal column, and monochrome sprites of those colors sit outside it.
#    - Original behavior:
#        key rows top-to-bottom = [blue, red, yellow]
#        outside counts         = [2, 5, 3]
#        original output rows   = [blue at 1,3] / [red at 1,3,5,7,9] / [yellow at 1,3,5]
#    - D14 changes the row assignment stage:
#        sort by count descending -> [red 5, yellow 3, blue 2]
#        slot rows are still the same physical rows [1,3,5], but now row 1 receives red, row 3 yellow, row 5 blue.
#    - D5 changes the within-row placement stage:
#        count 5 no longer lands at columns 1,3,5,7,9.
#        With growing gaps, the dot columns are 1,3,6,10,15.
#    - Fused behavior:
#        top row (largest count) becomes visually longest and also shows the clearest growing-gap pattern,
#        so the solver must first count, then rank colors, then render each ranked row with cumulative spacing.
# b) Complexity calibration:
#    - If all counts were <= 2, the growing-gap rule would look identical to fixed step-2 spacing. The generator
#      therefore forces at least one sampled count >= 3.
#    - If the sampled legend order were already sorted by descending counts, the D14 reorder would be invisible.
#      The generator therefore rejects already-sorted input orders.
#    - Counts are sampled without ties in normal generation. This keeps the ranking legible from 2-4 examples and
#      avoids making the task hinge on an invisible tie-break convention, while the solver still handles ties by
#      original top-to-bottom order for determinism.
# c) Implicit-assumption audit of the original solver:
#    - The original `for rr, col in keys` loop assumed "input row == output row". This breaks under D14, so the
#      combo solver must first collect `(orig_row, color, count)` records and sort them before drawing.
#    - The original `key_col + 2 * i` assumed a constant step. This breaks under D5, so drawing now uses a
#      cumulative offset helper where successive gaps are 2, 3, 4, ...
#    - The legend finder and key-column finder still work because the input legend geometry itself is unchanged.
#    - The output row slots are still the original odd rows of the legend; only which color/count occupies each
#      slot changes. The generator preserves this by keeping one key pixel per alternating row and no extra marks
#      inside the legend box.
# d) Placement legality:
#    - Outside sprites are still placed with `random_free_location(..., padding=1, padding_connectivity=8)`,
#      which guarantees no sprite touches another sprite or the legend and therefore each copy remains a separate
#      connected component.
#    - The legend width is expanded from the original linear formula to a cumulative-spacing formula so the
#      rightmost dot of the largest count always fits with at least one trailing black column plus slack.
#    - Grid placement still respects a margin from canvas edges, so neither the legend nor the sprites can clip.
# e) Visual observability:
#    - The ranking key is the visible sprite count, not any hidden value like the numeric color id.
#    - The longest tally is moved to the top row, making the row permutation obvious.
#    - Because the top tally is also guaranteed to have at least 3 dots, the nonuniform gaps are visible in the
#      most salient row rather than hidden in a short lower row.
# f) Visual cleanliness:
#    - The output remains one black rectangular tally board with sparse colored dots. Sorting only permutes which
#      color occupies each row, and growing gaps only stretches dots horizontally within the same clean region.
#    - Alternating key rows are preserved, so even after reordering the rows remain separated by full black rows.
# g) Color-rule inferability:
#    - No recoloring is introduced. A row's color always matches one input key color and one outside sprite set.
#      The only new inference is that color identity and output row identity are no longer the same thing.
# h) New solver assumptions and generator support:
#    - Assumption: there is exactly one relevant legend bbox defined as the largest black connected component whose
#      bbox contains non-black pixels. The generator still creates exactly one such legend.
#    - Assumption: the legend contains one usable non-black key column. The generator fills only column 1 on odd
#      rows, so the leftmost non-black column remains unambiguous.
#    - Assumption: row-ranking slots are exactly the original key rows in top-to-bottom order. The generator keeps
#      those rows fixed and only changes the output expectation, never the input legend layout.
#    - Assumption: the cumulative-spacing formula fits in the legend width. `_legend_shape()` uses the same tally
#      helper as the renderer, so the sampled max count is always drawable.


def _tally_offset(index: int) -> int:
    """Return the cumulative column offset for the zero-based tally dot."""
    idx = int(index)
    return idx * (idx + 3) // 2


def _tally_col(key_col: int, index: int) -> int:
    """Return the absolute column of the zero-based tally dot."""
    return int(key_col) + _tally_offset(index)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count legend colors, reorder the tally rows by frequency, and draw them with growing gaps.

    Input structure:
    - A black rectangular legend patch contains one colored key pixel on each alternating row of its leftmost non-black column.
    - Outside the patch, disconnected monochrome sprites of those key colors are scattered across the background.

    Rule:
    - Count the outside connected components of each key color.
    - Sort the key colors by count from largest to smallest, breaking ties by their original top-to-bottom legend order, then redraw the legend as black with one tally row per ranked color. Within each row, the first dot starts at the key column and later dots shift right with increasing gaps: +2, then +3, then +4, and so on.

    Key insight:
    - The legend rows are no longer fixed color destinations: first use them to identify and count colors, then reuse the same row slots as ranking positions, with the tally columns following a cumulative-spacing pattern instead of a constant stride.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

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

    key_col = None
    for cc in range(bw):
        if (box[:, int(cc)] != int(black)).any():
            key_col = int(cc)
            break
    if key_col is None:
        return np.zeros_like(box)

    keys = [(int(rr), int(box[int(rr), int(key_col)])) for rr in range(bh) if int(box[int(rr), int(key_col)]) != int(black)]
    keys.sort(key=lambda t: int(t[0]))
    slot_rows = [int(rr) for rr, _color in keys]

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    records: list[tuple[int, int, int]] = []
    for rr, color in keys:
        mask = (g == int(color)) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(rr), int(color), int(count)))

    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    for new_idx, (_orig_rr, color, count) in enumerate(records):
        rr = int(slot_rows[new_idx])
        for dot_idx in range(int(count)):
            cc = _tally_col(key_col, dot_idx)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(color)

    return out


# --------------- generator (combo) ---------------

COLOR_POOL = tuple(range(1, 10))
BLACK = 0
DISTRACTOR_PROB = 0.5

KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
VISIBLE_MAX_COUNT_MIN = 3
KEY_ROW_FIRST = 1
KEY_ROW_STEP = 2
KEY_COLUMN = 1
LEGEND_HEIGHT_TAIL = 1
LEGEND_WIDTH_TAIL = 1
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
    """Sample distinct counts that make both sorting and growing spacing visible."""
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    while True:
        counts = [int(v) for v in random.sample(choices, k)]
        if max(counts) < VISIBLE_MAX_COUNT_MIN:
            continue
        if counts == sorted(counts, reverse=True):
            continue
        return counts


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return legend height and width for the sorted growing-gap tally."""
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    last_dot_col = _tally_col(KEY_COLUMN, max_count - 1)
    bw = last_dot_col + LEGEND_WIDTH_TAIL + 1 + _randint(LEGEND_SLACK_RANGE)
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
    """Render the fused output tally canvas implied by key colors and counts."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for out_idx, (_orig_idx, color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        rr = _key_row(out_idx)
        for dot_idx in range(count):
            expected[rr, _tally_col(KEY_COLUMN, dot_idx)] = int(color)
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
    """Run one output-first generation attempt and reject solver mismatches or degeneracy."""
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
    raise RuntimeError("failed to generate a valid 58490d8a combo input")
