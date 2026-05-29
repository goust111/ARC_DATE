# Combo variant of 58490d8a (index 17)
# combo_spec: D11/0+D14/0
# Name: 4连通计数+排序重排

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the fused D11/0 + D14/0 variant
#
# a) Concreteization and behavior trace
#    - The visible scene still has one black legend rectangle plus scattered
#      same-color motifs outside it.
#    - Change 1: outside motifs are counted with 4-connectivity, so a motif that
#      is one 8-connected blob can contribute multiple tally dots when its parts
#      meet only at corners.
#    - Change 2: legend rows no longer stay attached to their input colors. After
#      counting, rows are reused as ranked slots from largest tally to smallest.
#    - Example trace:
#        input rows = [blue, red, yellow]
#        one blue motif has 2 four-components, red has 5, yellow has 3
#        old task output = blue x1, red x5, yellow x3 on original rows
#        fused output    = red x5, yellow x3, blue x2 on ranked rows
#      The same sample therefore differs from the original task in both the
#      counting stage and the row assignment stage.
#
# b) Complexity calibration
#    - To make the 4-connectivity shift observable, the generator draws motifs
#      from a tiny fixed family of templates whose 4-component counts are known
#      and limited to 1..5.
#    - To make the sorting step observable, target totals are sampled as distinct
#      values and the input legend order is rejected when already descending.
#    - Same-color copies still reuse one canonical motif, preserving a stable
#      "these belong together" cue instead of introducing too much shape freedom.
#
# c) Hidden-assumption audit against the original solver
#    - Legend detection via "largest black 8-connected component whose bbox
#      contains non-black pixels" still holds because black remains confined to
#      the legend rectangle.
#    - Key-column detection via the leftmost non-black legend column still holds
#      because the legend layout is unchanged.
#    - The original per-row redraw assumption is broken by D14/0, so the solver
#      must separate "input row used to identify the color" from "output slot
#      row used after ranking".
#    - The original 8-connectivity count is broken by D11/0, so the solver must
#      explicitly use connectivity=4 when counting outside same-color parts.
#    - Generator placement must keep 8-neighborhood gaps between all pasted
#      motifs and the legend, otherwise independent motifs could touch and create
#      ambiguous extra 4-connected joins.
#
# d) Placement legality
#    - Every motif fits in a 2x2 or 3x3 bbox and is pasted with one-cell
#      8-neighborhood padding through `random_free_location(..., padding=1,
#      padding_connectivity=8)`.
#    - This guarantees: no overlap, no touching across motifs, no touching with
#      the legend, and no accidental merges that would corrupt the intended
#      four-component counts.
#
# e) Visual observability
#    - Sorting is driven only by visible tally lengths, never by hidden numeric
#      color ids or generation order.
#    - The 4-connectivity clue is visible because some repeated motifs contain
#      diagonal-only links yet yield more tally dots than the number of visible
#      outside blobs.
#    - Distinct totals ensure the rank order is legible from a small number of
#      examples instead of being obscured by ties.
#
# f) Visual cleanliness
#    - The output remains a sparse black tally board with one colored row per
#      slot; no new overlay system is introduced.
#    - Outside motifs stay small, separated, and color-consistent, so the two
#      reasoning steps remain decodable by eye.
#
# g) Inferability of the combined rule
#    - Colors are never remapped arbitrarily; a color simply moves to a ranked
#      row after its four-component tally is computed.
#    - The same repeated motif per color gives the solver a repeated witness for
#      "corner-touching splits" while the row permutation gives a repeated
#      witness for "largest count goes first".
#
# New implicit assumptions introduced by the fused solver and how generation
# guarantees them:
#    - Assumption: the ranked order is deterministic. Generator enforces distinct
#      totals, so descending count order is unique.
#    - Assumption: at least one key color exhibits a four-vs-eight mismatch.
#      Generator forces at least one color to use a motif whose per-copy
#      four-component count is > 1.
#    - Assumption: the sorting step changes something visible. Generator rejects
#      already-descending input count orders.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count 4-connected legend fragments, then rank the legend rows by tally size.

    Input structure:
    - A black rectangular legend patch contains one key-colored pixel on alternating rows of its leftmost non-black column.
    - Outside that patch, each key color appears in one or more small motifs, and some motifs are only tied together diagonally.

    Rule:
    - For each key color, count outside same-colored connected components using 4-connectivity rather than 8-connectivity.
    - Sort the key colors by those counts from largest to smallest, breaking ties by original top-to-bottom legend order, then draw one evenly spaced dot per counted component on the ranked rows.

    Key insight:
    - The legend first tells you which colors to measure, but its row slots are reused after counting, and diagonal-only links inside a motif must be split before deciding that ranking.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    black = 0

    # STEP 1) Find the legend bbox: the largest black CC whose bbox encloses
    # some non-black key pixels.
    best = None  # (area, bbox)
    for coords in components_mask(g == int(black), connectivity=8):
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        sub = crop_bbox(g, b)
        if (sub != int(black)).any():
            cand = (int(coords.shape[0]), tuple(map(int, b)))
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None

    r0, c0, r1, c1 = map(int, best[1])
    box = crop_bbox(g, (r0, c0, r1, c1))
    bh, bw = map(int, box.shape)

    # STEP 2) Find the key-pixel column inside the legend.
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

    # STEP 3) Count each key color outside the legend using 4-connectivity.
    records: list[tuple[int, int, int]] = []
    for rr, col in keys:
        mask = (g == int(col)) & outside
        _labels, count = label_mask(mask, connectivity=4)
        records.append((int(rr), int(col), int(count)))

    # STEP 4) Reuse the legend rows as rank slots, largest tally first.
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    for new_idx, (_orig_rr, col, count) in enumerate(records):
        rr = int(slot_rows[new_idx])
        for i in range(int(count)):
            cc = int(key_col + 2 * i)
            if 0 <= cc < bw:
                out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (combo variant) ---------------

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
LEGEND_SLACK_RANGE = (0, 3)

GRID_HEIGHT_RANGE = (20, 25)
GRID_WIDTH_RANGE = (20, 25)
LEGEND_EDGE_MARGIN = 2

SPRITE_PADDING = 1
SPRITE_PADDING_CONNECTIVITY = 8
GENERATION_ATTEMPTS = 300

# Each template is a single 8-connected motif whose 4-connected component count
# equals the dict key.
BASE_MASKS: dict[int, np.ndarray] = {
    1: np.array([[1, 1], [1, 1]], dtype=bool),
    2: np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=bool),
    3: np.array([[1, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=bool),
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
    5: np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
}


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _key_row(i: int) -> int:
    return KEY_ROW_FIRST + KEY_ROW_STEP * i


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    bh = KEY_ROW_STEP * k + LEGEND_HEIGHT_TAIL
    bw = TALLY_COL_STEP * max_count + LEGEND_WIDTH_TAIL + _randint(LEGEND_SLACK_RANGE)
    return bh, bw


@lru_cache(maxsize=None)
def _mask_variants(four_count: int) -> tuple[np.ndarray, ...]:
    base = BASE_MASKS[int(four_count)]
    seen: set[tuple[int, ...]] = set()
    variants = []
    for flip in (False, True):
        arr = np.fliplr(base) if flip else base
        for turns in range(4):
            rot = np.rot90(arr, turns)
            key = tuple(map(int, rot.flatten()))
            if key not in seen:
                seen.add(key)
                variants.append(rot.copy())
    return tuple(variants)


def _make_sprite(color: int, bg: int, four_count: int) -> np.ndarray:
    mask = random.choice(_mask_variants(int(four_count)))
    sprite = np.full(mask.shape, int(bg), dtype=int)
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


def _sample_counts(k: int) -> list[int]:
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    while True:
        counts = [int(v) for v in random.sample(choices, k)]
        if counts != sorted(counts, reverse=True):
            return counts


def _choose_units(counts: list[int]) -> list[int] | None:
    options: list[list[int]] = []
    eligible: list[int] = []
    for i, total in enumerate(counts):
        divisors = [unit for unit in sorted(BASE_MASKS) if total % unit == 0]
        options.append(divisors)
        if any(unit > 1 for unit in divisors):
            eligible.append(i)
    if not eligible:
        return None

    forced = random.choice(eligible)
    units = []
    for i, divisors in enumerate(options):
        multi_units = [unit for unit in divisors if unit > 1]
        if i == forced:
            units.append(int(random.choice(multi_units)))
        elif multi_units and random.random() < 0.6:
            units.append(int(random.choice(multi_units)))
        else:
            units.append(int(random.choice(divisors)))
    return units


def _sorted_key_records(key_colors: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    records = [(int(i), int(color), int(count)) for i, (color, count) in enumerate(zip(key_colors, counts))]
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return records


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)

    expected = np.full((bh, bw), BLACK, dtype=int)
    for out_i, (_orig_i, color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        rr = _key_row(out_i)
        for j in range(int(count)):
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
    units: list[int],
    distractor_color: int | None,
    bg: int,
) -> list[np.ndarray]:
    sprites = []
    for color, total, unit in zip(key_colors, counts, units):
        copies = int(total // unit)
        sprite = _make_sprite(int(color), int(bg), int(unit))
        for _ in range(copies):
            sprites.append(sprite.copy())
    if distractor_color is not None:
        distractor_unit = random.choice(list(BASE_MASKS))
        sprites.append(_make_sprite(int(distractor_color), int(bg), int(distractor_unit)))
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
    units = _choose_units(counts)
    if units is None:
        return None

    expected = _build_expected(key_colors, counts)
    legend = _build_legend(expected.shape, key_colors)
    grid = _new_grid_with_legend(bg, legend)
    if grid is None:
        return None

    sprites = _build_sprite_queue(key_colors, counts, units, distractor_color, bg)
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
