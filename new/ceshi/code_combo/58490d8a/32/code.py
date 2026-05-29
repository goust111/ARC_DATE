# Combo variant of 58490d8a (index 32)
# combo_spec: D3/1+D10/0+D14/0+D15/0
# Name: 水平紧密排序+统一色四维组合

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, label_mask, paste, random_free_location


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - Original task:
#      * A black legend box contains colored key pixels down one column on rows 1,3,5,...
#      * Output redraws each key on its own row, extending same-color dots rightward.
#    - This combo rotates the legend, packs the key slots, sorts them by count, and
#      recolors the count marks:
#      * Input legend now has one top key row with colors at columns 1,2,3,... and
#        those columns are intentionally in an unsorted left-to-right order.
#      * After counting outside components for each key color, output reorders the
#        key headers from largest count to smallest count.
#      * Each reordered header keeps its own color on the top row, but the counted
#        marks below it are all drawn in one reserved tally color and form a compact
#        vertical stack.
#    - Example trace with input top row [blue, red, yellow] and outside counts [2,5,3]:
#      * Original task would have produced three horizontal rows in input order.
#      * A naive D3+D14+D15 merge that recolors the full column would lose color
#        identity after sorting, because every output column would become the same
#        tally color.
#      * The fused rule instead outputs top row [red, yellow, blue], then puts
#        same-color tally dots under those headers with heights 5,3,2.
# b) Complexity calibration
#    - Key count stays in 2..4 and object counts stay in 1..5 so multiple columns
#      and multiple stack heights are visible in a small number of examples.
#    - Counts are sampled as distinct values and the input order is rejected when it
#      already matches descending count order. This keeps the D14 reordering visible.
#    - The tally color is reserved for output only, so the D15 change is also
#      unmistakable instead of accidentally matching an input sprite color.
# c) Implicit-assumption audit of the original solver
#    - "Leftmost non-black column is the legend index" breaks under D3; we must read
#      the topmost non-black row and treat columns as slots.
#    - "Each color returns to its original slot" breaks under D14; we must count
#      first, then sort records by count before drawing.
#    - "The tally marks can use the key color itself" breaks under D15; only the
#      headers stay colored while the count dots become the reserved tally color.
#    - "Alternating key spacing is fixed at step 2" breaks under D10; the relevant
#      slot columns are now consecutive and must be read directly from the row.
# d) Placement legality
#    - The generator still uses one black legend patch pasted into a larger canvas
#      before any sprites are placed.
#    - Outside sprites are placed with padding=1 under 8-connectivity, so copies of
#      the same color stay as separate components and nothing touches the legend box.
#    - The compact horizontal legend reduces width pressure, while the vertically
#      stacked dots fit by sizing the legend height from the maximum sampled count.
# e) Visual observability
#    - The sorted property depends only on visible tally heights, never on raw color ids.
#    - The top row preserves the color identities after sorting, so a human can see
#      which key color moved to which ranked column.
#    - Uniform tally-color stacks under colored headers make the D15 change obvious
#      without hiding the D14 permutation signal.
# f) Visual cleanliness
#    - The output has exactly two visual layers inside the legend: one colored header
#      row and one monochrome tally field below it.
#    - Key columns are consecutive, which makes the compact layout salient, but the
#      count dots stay in separate columns so they do not merge horizontally.
# g) Change inferability
#    - The only color change rule is structural and global: all count dots use the
#      same reserved ARC color.
#    - Because the colored header row survives and is itself reordered, humans can
#      infer that color identity comes from header position while quantity comes from
#      the uniform stack below each header.
# New solver assumptions and generator guarantees
#    - Assumption: there is exactly one active header row inside the legend.
#      Guarantee: the generator writes colored keys only on KEY_ROW.
#    - Assumption: slot identity is the left-to-right order of non-black cells in
#      that header row.
#      Guarantee: keys occupy consecutive columns starting at KEY_COLUMN_FIRST.
#    - Assumption: the reserved tally color does not appear in the input scene, so
#      output dots do not collide with any counting color.
#      Guarantee: input colors are sampled from INPUT_COLOR_POOL with TALLY_COLOR
#      excluded, and the tally color is fixed globally across all samples.


BLACK = 0
TALLY_COLOR = 8
INPUT_COLOR_POOL = tuple(color for color in range(1, 10) if color != TALLY_COLOR)

# Header columns and tally dimensions.
KEY_COUNT_RANGE = (2, 4)
TARGET_COUNT_RANGE = (1, 5)
KEY_ROW = 1
KEY_COLUMN_FIRST = 1
KEY_COLUMN_STEP = 1
TALLY_ROW_OFFSET = 1
TALLY_ROW_STEP = 1
LEGEND_HEIGHT_TAIL = 1
LEGEND_HEIGHT_SLACK_RANGE = (0, 3)
LEGEND_WIDTH_TAIL = 1

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
GENERATION_CASES = tuple((k, use_distractor) for k in range(KEY_COUNT_RANGE[0], KEY_COUNT_RANGE[1] + 1) for use_distractor in (False, True))

_generation_case_cursor = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites, rank the legend headers, and draw uniform-color stacks beneath them.

    Input structure:
    - A black rectangular legend box contains one horizontal header row of colored key pixels packed into consecutive columns.
    - Outside that box, disconnected same-colored sprites for those key colors are scattered on the background, plus at most one distractor color.

    Rule:
    - Count the outside connected components for each header color.
    - Sort the header colors from largest count to smallest count, keeping original left-to-right order as the tie break, then redraw the legend crop with those sorted colored headers on top and fixed-color tally dots stacked downward beneath each header.

    Key insight:
    - The input header row is only the lookup table; the output turns those headers into ranked column labels, while the actual quantities are carried by one-color vertical stacks below them.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

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

    key_row = None
    for rr in range(bh):
        if (box[int(rr), :] != BLACK).any():
            key_row = int(rr)
            break
    if key_row is None:
        return np.zeros_like(box)

    keys = [(int(cc), int(box[int(key_row), int(cc)])) for cc in range(bw) if int(box[int(key_row), int(cc)]) != BLACK]
    keys.sort(key=lambda t: int(t[0]))
    slot_cols = [int(cc) for cc, _color in keys]

    outside = np.ones((H, W), dtype=bool)
    outside[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = False
    out = np.zeros((bh, bw), dtype=int)

    records: list[tuple[int, int, int]] = []
    for cc, color in keys:
        mask = (g == int(color)) & outside
        _labels, count = label_mask(mask, connectivity=8)
        records.append((int(cc), int(color), int(count)))

    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    for new_idx, (_orig_cc, color, count) in enumerate(records):
        cc = int(slot_cols[new_idx])
        out[int(key_row), int(cc)] = int(color)
        for i in range(int(count)):
            rr = int(key_row + TALLY_ROW_OFFSET + TALLY_ROW_STEP * i)
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


def _sample_palette(key_count: int, want_distractor: bool) -> tuple[int, list[int], int | None]:
    """Choose background, key colors, and an optional distractor while reserving the tally color for output."""
    colors = list(INPUT_COLOR_POOL)
    bg = random.choice(colors)

    remaining = [c for c in colors if c != bg]
    key_colors = random.sample(remaining, key_count)
    remaining = [c for c in remaining if c not in key_colors]
    distractor_color = random.choice(remaining) if want_distractor and remaining else None
    return bg, key_colors, distractor_color


def _sample_counts(k: int) -> list[int]:
    """Sample distinct component counts and force the input order to need sorting."""
    choices = list(range(TARGET_COUNT_RANGE[0], TARGET_COUNT_RANGE[1] + 1))
    counts = random.sample(choices, k)
    while counts == sorted(counts, reverse=True):
        random.shuffle(counts)
    return [int(v) for v in counts]


def _legend_shape(k: int, max_count: int) -> tuple[int, int]:
    """Return compact horizontal-legend dimensions for the key count and tallest stack."""
    bh = KEY_ROW + TALLY_ROW_OFFSET + TALLY_ROW_STEP * max(0, max_count - 1) + LEGEND_HEIGHT_TAIL + _randint(LEGEND_HEIGHT_SLACK_RANGE)
    bw = KEY_COLUMN_FIRST + KEY_COLUMN_STEP * (k - 1) + LEGEND_WIDTH_TAIL + 1
    return bh, bw


def _key_col(i: int) -> int:
    """Map a zero-based key index to its packed column inside the legend/output box."""
    return KEY_COLUMN_FIRST + KEY_COLUMN_STEP * i


def _sorted_key_records(key_colors: list[int], counts: list[int]) -> list[tuple[int, int, int]]:
    """Return (original_index, color, count) sorted by output rank."""
    records = [(int(i), int(color), int(count)) for i, (color, count) in enumerate(zip(key_colors, counts))]
    records.sort(key=lambda t: (-int(t[2]), int(t[0])))
    return records


def _build_expected(key_colors: list[int], counts: list[int]) -> np.ndarray:
    """Render the sorted output: colored headers on top and uniform tally-color stacks below."""
    max_count = max(counts)
    bh, bw = _legend_shape(len(key_colors), max_count)
    expected = np.full((bh, bw), BLACK, dtype=int)

    for out_i, (_orig_i, color, count) in enumerate(_sorted_key_records(key_colors, counts)):
        cc = _key_col(out_i)
        expected[KEY_ROW, cc] = int(color)
        for j in range(count):
            rr = KEY_ROW + TALLY_ROW_OFFSET + TALLY_ROW_STEP * j
            expected[rr, cc] = TALLY_COLOR
    return expected


def _build_legend(shape: tuple[int, int], key_colors: list[int]) -> np.ndarray:
    """Return an unsorted black legend patch with only the packed header row filled."""
    legend = np.full(shape, BLACK, dtype=int)
    for i, color in enumerate(key_colors):
        legend[KEY_ROW, _key_col(i)] = int(color)
    return legend


def _next_generation_case() -> tuple[int, bool]:
    """Cycle through key-count and distractor regimes so repeated calls cover all branches."""
    global _generation_case_cursor
    case = GENERATION_CASES[_generation_case_cursor % len(GENERATION_CASES)]
    _generation_case_cursor += 1
    return case


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


def _gen_attempt(key_count: int, want_distractor: bool) -> np.ndarray | None:
    """Run one output-first generation attempt and reject mismatches or weak reorder cases."""
    bg, key_colors, distractor_color = _sample_palette(key_count, want_distractor)
    counts = _sample_counts(key_count)
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
    if np.array_equal([color for color in key_colors], [color for _orig_i, color, _count in _sorted_key_records(key_colors, counts)]):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    key_count, want_distractor = _next_generation_case()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(key_count, want_distractor)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 58490d8a combo-32 input")
