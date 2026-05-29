# Combo variant of d8e07eb2 (index 21)
# combo_spec: D2/3+D9/0
# Name: 含干扰查询+固定4个有效查询
#
# Design analysis
# ---
# combo_variant_spec:
#   concrete_pixels:
#     original_query_band: four table-aligned 5x5 cells, all visible glyphs are real queries.
#     fused_query_band: the fixed 22-column canvas cannot fit 5-6 separate 5x5 cells, so the
#       table and verdict regions stay unchanged while the query band is compressed into six fixed
#       3x3 windows arranged as a staggered 2x3 pattern:
#       (0,1), (0,8), (0,15), (2,4), (2,11), (2,18).
#     occupancy_rule: exactly 5 or 6 windows are occupied; among them exactly 4 are genuine table
#       glyphs and the remaining 1-2 are distractors whose shape does not appear anywhere in the
#       reference table.
#   behavior_trace:
#     original_example: top band shows four true glyphs A/B/C/D; the solver treats every visible
#       query as relevant, highlights those four table cells, and succeeds only if they form a full
#       row or column.
#     fused_example: visible top glyphs are [A, X, B, C, Y, D], where X/Y are distractors. The
#       original logic would count all six visible items, so success would be impossible even when
#       A/B/C/D form a complete line. The fused solver instead normalizes signatures, filters the
#       six windows through the table-signature map, ignores X/Y, and evaluates only the surviving
#       four matches.
#     D9_interaction: the four real queries are fixed in both success and failure cases. Therefore
#       failure is no longer caused by "too few true queries"; it comes only from the geometry of
#       the four matched cells after distractor removal.
#   complexity_calibration:
#     real_queries: always 4 distinct table glyphs, which keeps the row/column hypothesis stable.
#     distractors: always 1 or 2, enough to make filtering necessary without flooding the query band.
#     distractor_style: distractors reuse normal glyph colors but use 3x3 shapes absent from the
#       current table, so humans can reject them by visible shape mismatch instead of hidden color id
#       tricks.
#     slot_count: six candidate windows give two observable regimes (5 filled or 6 filled), which
#       makes the "ignore extras" rule visible across a few examples.
#   implicit_assumption_review:
#     old_query_extraction: assumed every non-empty query position is valid. Broken by D2/3, so the
#       new solver must build a table signature dictionary first and keep only query signatures found
#       there.
#     old_success_check: assumed visible query count itself decides completeness. Broken by both
#       dimensions, because the top band now shows 5-6 glyphs while the true logical query count is
#       always 4.
#     old_signature_frame: compared 5x5 query cells directly against 5x5 table cells. Broken by the
#       compressed source region, so the new solver must normalize glyph coordinates before matching.
#     new_assumption: query glyphs live only inside the six fixed 3x3 windows. The generator enforces
#       this exact geometry and may leave at most one window empty.
#   placement_legality:
#     all six query windows lie inside rows 0..4 and cols 1..20, with no overlapping pixels.
#     the 4x4 table remains at the original 5x5 coordinates, so the table/verdict geometry stays
#       legal and unchanged.
#     generator validation runs main(input) and compares with the constructed expected output before
#       accepting a sample.
#   observability:
#     in successful examples, the output highlights exactly four table cells even when 5-6 glyphs are
#       visible above, making the distractor filter directly observable.
#     across failure examples, four table cells still highlight, but they bend away from a single
#       row/column; this isolates the new decision criterion from mere query count.
#   visual_cleanliness:
#     the staggered query windows avoid stacking symbols on top of the table and keep every top-band
#       glyph as a single 3x3 patch.
#     distractors are not marked separately in the input, so the only extra visual burden is a small
#       number of additional glyphs in otherwise empty teal space.
#   inference_check:
#     no invisible ordering, sorting, or color arithmetic is used. Matching depends only on visible
#       color+shape equality, and the output transformation has only two observable branches:
#       complete line -> green top+bottom, otherwise red bottom.
#   post_edit_main_assumptions:
#     separator detection still assumes exactly the full-width non-background rows are separators.
#     table inference still assumes a regular 5x5 lattice below the first separator.
#     query filtering now assumes each real query signature is unique in the table; the generator
#       guarantees that because every table (shape, color) pair is unique.

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# -- Grid layout --
GRID_H, GRID_W = 30, 22
TOP_SEP_ROW = 5
BOTTOM_SEP_ROW = 27
QUERY_BAND_H = 5
VERDICT_ROW = 28

# -- Colors --
BG = 8
GREEN = 3
RED = 2
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)

# -- Query geometry --
QUERY_SIZE = 3
QUERY_SLOTS = ((0, 1), (0, 8), (0, 15), (2, 4), (2, 11), (2, 18))

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
DISTRACTOR_COUNTS = (1, 2)
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

# -- Generation --
MAX_RETRIES = 300


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _normalized_signature(patch: np.ndarray, bg: int) -> tuple[tuple[int, int, int], ...] | None:
    mask = patch != int(bg)
    if not bool(mask.any()):
        return None
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    return tuple((int(r - r0), int(c - c0), int(patch[r, c])) for r, c in coords)


def _cell_signature(
    g: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    bg: int,
) -> tuple[tuple[int, int, int], ...] | None:
    patch = g[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)]
    return _normalized_signature(patch, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter a noisy query band down to the four real table lookups and test whether they make a full line.

    Input structure:
    - The canvas keeps the original teal background and two full-width pink separator rows.
    - Between the separators is the same 4x4 table of 5x5 cells, each cell containing one centered
      3x3 monochrome glyph identified by its color and shape.
    - Above the first separator are six fixed compact 3x3 query windows in a staggered layout. Exactly
      5-6 windows are occupied, but only four occupied glyphs actually exist in the table; the others
      are distractors.

    Rule:
    - Normalize glyph signatures so compact query glyphs can be compared directly with centered table glyphs.
    - Ignore any occupied query window whose color+shape signature does not appear in the table.
    - Green-fill the background inside each matched table cell. If the four surviving matched cells form
      one complete table row or one complete table column, also green-fill the query band and the verdict
      band; otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: visible query count is a decoy in this variant. The solver must first discard the
    non-table glyphs, then evaluate the original row/column condition on exactly the four remaining
    table-backed queries.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = get_bg(g)

    sep_rows = _constant_non_bg_rows(g, bg)
    assert len(sep_rows) >= 2, "expected two full-width separator rows"
    top_sep = sep_rows[0]
    bottom_sep = sep_rows[-1]

    table_region = g[top_sep + 1 : bottom_sep, :]
    table_fg = np.argwhere(table_region != bg)
    assert table_fg.size, "reference table must contain glyphs"

    first_glyph_r = top_sep + 1 + int(table_fg[:, 0].min())
    first_glyph_c = int(table_fg[:, 1].min())
    row0 = first_glyph_r - 1
    col0 = first_glyph_c - 1

    row_starts = [r for r in range(row0, bottom_sep, CELL) if r + CELL <= bottom_sep]
    col_starts = [c for c in range(col0, w, CELL) if c + CELL <= w]
    assert row_starts and col_starts, "could not infer table cell grid"

    table_map: dict[tuple[tuple[int, int, int], ...], tuple[int, int]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, CELL, CELL, bg)
            if sig is None:
                continue
            table_map[sig] = (int(ri), int(ci))

    query_sigs = [
        sig
        for r, c in QUERY_SLOTS
        if (sig := _cell_signature(g, r, c, QUERY_SIZE, QUERY_SIZE, bg)) is not None
    ]
    real_query_sigs = {sig for sig in query_sigs if sig in table_map}

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for sig in real_query_sigs:
        ri, ci = table_map[sig]
        selected.append((ri, ci))
        r = row_starts[ri]
        c = col_starts[ci]
        cell = out[r : r + CELL, c : c + CELL]
        cell[cell == bg] = GREEN

    needed = len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = len(real_query_sigs) == needed and len(selected) == needed and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED

    return out


def _has_full_bbox(mask: int) -> bool:
    bits = [(mask >> i) & 1 for i in range(9)]
    return (
        any(bits[0:3])
        and any(bits[6:9])
        and any(bits[i] for i in (0, 3, 6))
        and any(bits[i] for i in (2, 5, 8))
    )


PATTERNS_BY_FILL = {
    k: tuple(m for m in range(1 << 9) if m.bit_count() == k and _has_full_bbox(m))
    for k in range(FILL_MIN, FILL_MAX + 1)
}
ALL_QUERY_PATTERNS = tuple(mask for masks in PATTERNS_BY_FILL.values() for mask in masks)


def _balanced_color_list() -> list[int]:
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks() -> list[int]:
    counts = dict(SHAPE_BASE_COUNTS)
    extras = TABLE_DIM * TABLE_DIM - sum(counts.values())
    for _ in range(extras):
        choices = [k for k in range(FILL_MIN, FILL_MAX) if counts[k] < MAX_SHAPES_PER_FILL]
        counts[random.choice(choices)] += 1

    masks: list[int] = []
    for k, n in counts.items():
        masks.extend(random.sample(PATTERNS_BY_FILL[k], n))
    random.shuffle(masks)
    return masks


def _table_glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _query_glyph(mask: int, color: int) -> np.ndarray:
    glyph = np.full((QUERY_SIZE, QUERY_SIZE), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    glyph[bits] = color
    return glyph


def _signature_from_mask(mask: int, color: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(r), int(c), int(color))
        for r in range(3)
        for c in range(3)
        if (mask >> (r * 3 + c)) & 1
    )


def _line_cells() -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells() -> list[tuple[int, int]]:
    mode = random.choice(FAIL_MODES)
    if mode == "rect":
        rs = random.sample(range(TABLE_DIM), 2)
        cs = random.sample(range(TABLE_DIM), 2)
        cells = [(r, c) for r in rs for c in cs]
    elif mode == "bent_row":
        r = random.randrange(TABLE_DIM)
        dropped_c = random.randrange(TABLE_DIM)
        new_r = random.choice([rr for rr in range(TABLE_DIM) if rr != r])
        cells = [(r, c) for c in range(TABLE_DIM) if c != dropped_c] + [(new_r, dropped_c)]
    elif mode == "bent_col":
        c = random.randrange(TABLE_DIM)
        dropped_r = random.randrange(TABLE_DIM)
        new_c = random.choice([cc for cc in range(TABLE_DIM) if cc != c])
        cells = [(r, c) for r in range(TABLE_DIM) if r != dropped_r] + [(dropped_r, new_c)]
    elif mode == "scatter_rows":
        base_c = random.randrange(TABLE_DIM)
        cols = [base_c] * TABLE_DIM
        for r in random.sample(range(TABLE_DIM), random.choice((1, 2, 3))):
            cols[r] = random.choice([c for c in range(TABLE_DIM) if c != base_c])
        cells = [(r, cols[r]) for r in range(TABLE_DIM)]
    else:
        base_r = random.randrange(TABLE_DIM)
        rows = [base_r] * TABLE_DIM
        for c in random.sample(range(TABLE_DIM), random.choice((1, 2, 3))):
            rows[c] = random.choice([r for r in range(TABLE_DIM) if r != base_r])
        cells = [(rows[c], c) for c in range(TABLE_DIM)]

    assert len(cells) == TABLE_DIM and len(set(cells)) == TABLE_DIM
    assert len({r for r, _c in cells}) > 1 and len({c for _r, c in cells}) > 1
    random.shuffle(cells)
    return cells


def _pick_distractors(
    table_masks: list[int],
    used_query_sigs: set[tuple[tuple[int, int, int], ...]],
    count: int,
) -> list[tuple[int, int]]:
    distractors: list[tuple[int, int]] = []
    table_mask_set = set(table_masks)
    mask_choices = [mask for mask in ALL_QUERY_PATTERNS if mask not in table_mask_set]
    while len(distractors) < count:
        mask = random.choice(mask_choices)
        color = random.choice(GLYPH_COLORS)
        sig = _signature_from_mask(mask, color)
        if sig in used_query_sigs:
            continue
        used_query_sigs.add(sig)
        distractors.append((mask, color))
    return distractors


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    masks = _balanced_shape_masks()
    colors = _balanced_color_list()
    glyph_specs: list[tuple[int, int]] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph_specs.append((mask, color))
        glyph = _table_glyph(mask, color)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    query_cells = _line_cells() if success else _fail_cells()
    distractor_count = random.choice(DISTRACTOR_COUNTS)

    used_query_sigs = {
        _signature_from_mask(glyph_specs[r * TABLE_DIM + c][0], glyph_specs[r * TABLE_DIM + c][1])
        for r, c in query_cells
    }
    distractors = _pick_distractors(masks, used_query_sigs, distractor_count)

    query_specs = [glyph_specs[r * TABLE_DIM + c] for r, c in query_cells] + distractors
    random.shuffle(query_specs)
    slot_indices = random.sample(range(len(QUERY_SLOTS)), len(query_specs))
    for slot_index, (mask, color) in zip(slot_indices, query_specs, strict=True):
        r0, c0 = QUERY_SLOTS[slot_index]
        paste(grid, _query_glyph(mask, color), r0, c0)

    expected = _expected_output(grid, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
