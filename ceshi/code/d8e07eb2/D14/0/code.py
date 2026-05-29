# Variant of d8e07eb2
# Dimension: D14 — 未匹配单元格的处理 (transform)
# Change: 参考表中未被匹配的单元格在输出中保持不变（字形和背景色不变）... → 未匹配单元格被擦除

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis:
#   variant: D14 / erase unmatched reference-table glyphs
#   source_task_baseline:
#     task 58f5dbd5 uses background-colored carving as the key transform:
#       matched structure is kept, while selected pixels are turned back into bg.
#     This variant borrows only that erase mechanic, not its packing/layout logic.
#   a_specific_behavior_trace:
#     input_pixels:
#       - 30x22 teal canvas.
#       - row 5 and row 27 are full-width separator rows in color 6.
#       - rows 7,12,17,22 x cols 1,6,11,16 define a 4x4 table of 5x5 cells.
#       - each cell contains a centered 3x3 monochrome glyph on teal bg.
#       - the query band (rows 0..4) contains 3 or 4 glyph copies sampled from the table.
#     original_output_step:
#       - matched cells keep glyph pixels and their bg turns green.
#       - unmatched cells are copied through unchanged.
#     variant_output_step:
#       - matched cells still keep glyph pixels and their bg turns green.
#       - unmatched cells are filtered out: every non-bg glyph pixel inside those
#         cells becomes bg, so the whole 5x5 cell looks empty.
#       - success/fail verdict logic remains unchanged.
#     concrete_example:
#       - if query glyphs match table cells {(0,1), (2,1), (3,1)}:
#         original output leaves the other 13 table glyphs visible.
#         variant output blanks those 13 cells entirely, so only the three matched
#         cells remain visible against the table frame.
#   b_complexity_calibration:
#     - Keep the 4x4 table and 16 unique (shape,color) signatures from the original
#       task to avoid ambiguous matching.
#     - No extra rule branches are added; the only changed degree of freedom is
#       the visual treatment of unmatched cells.
#     - The effect is strong because every sample erases at least 12 of 16 cells.
#   c_implicit_assumption_review:
#     - Separator detection still relies on two constant full-width non-bg rows.
#       Generator keeps rows 5 and 27 as the only such rows.
#     - Cell signatures must be read from the input grid, not the evolving output.
#       Otherwise erased unmatched cells would destroy later comparisons.
#     - Query glyphs must stay distinct, because success tests compare the number
#       of selected table cells against the number of query signatures.
#     - Erasing only non-bg pixels is important: a matched green-filled cell must
#       not later be overwritten by any blanket table clearing step.
#   d_placement_legality:
#     - The table/query geometry is unchanged from the original fixed layout.
#     - Each pasted glyph fits inside a 5x5 cell with a 1-pixel border.
#     - Query glyphs occupy distinct top-band slots aligned with table columns.
#   e_visual_observability:
#     - In 2-4 examples, the new rule is directly visible because unmatched table
#       cells disappear completely instead of remaining as colorful distractors.
#     - Success vs fail is still readable from the verdict band and query-band fill.
#   f_visual_cleanliness:
#     - Removing unmatched glyphs reduces clutter rather than adding overlays.
#     - Matched cells remain emphasized by green backgrounds, so the eye can isolate
#       the selected row/column or scattered set immediately.
#   g_color_inference:
#     - No hidden color remapping is introduced.
#     - Green still means selected/success context; red still means fail verdict.
#   post_change_main_assumptions:
#     - Every table cell is either matched once or erased once.
#     - Generator guarantees many unmatched cells exist, so the D14 effect does not
#       degenerate back to the original behavior.


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _cell_signature(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[tuple[int, int, int], ...] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    return tuple(
        (int(r), int(c), int(cell[r, c]))
        for r, c in zip(*np.where(mask), strict=False)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the query glyphs against the reference table and filter the table to only the hits.

    Input structure:
    - A teal background with two full-width separator rows in a non-background color.
    - The top band holds up to four 5x5 query cells, each containing one centered 3x3 glyph.
    - The middle band is a 4x4 table of the same 5x5 cells; a glyph is identified by its
      exact 3x3 shape together with its color.

    Rule:
    - For every table cell whose glyph appears in the query band, keep the glyph and fill that
      cell's background pixels green.
    - For every table cell that is not matched, erase the glyph back to the teal background.
    - If the matched cells are exactly a full table row or a full table column with all four
      query slots occupied, also green-fill the query band and the verdict band; otherwise only
      the verdict band turns red.

    Key insight: The table becomes a visual filter result rather than an annotated lookup table:
    matched entries survive as green-highlighted cells, while every unmatched glyph disappears.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = get_bg(g)
    cell_size = 5

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

    row_starts = list(range(row0, bottom_sep, cell_size))
    row_starts = [r for r in row_starts if r + cell_size <= bottom_sep]
    col_starts = list(range(col0, w, cell_size))
    col_starts = [c for c in col_starts if c + cell_size <= w]
    assert row_starts and col_starts, "could not infer table cell grid"

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, bg)
            cell = out[r : r + cell_size, c : c + cell_size]
            if sig in query_set:
                selected.append((int(ri), int(ci)))
                cell[cell == bg] = 3
            else:
                cell[cell != bg] = bg

    complete_query = len(query_sigs) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_sigs) and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator (o5x) ---------------

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

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

# -- Generation --
MAX_RETRIES = 300


def _has_full_bbox(mask: int) -> bool:
    """True iff the 3x3 bitmask touches top, bottom, left, and right edges."""
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


def _balanced_color_list() -> list[int]:
    """Assign 16 cell colors so each palette color appears 1-3 times."""
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks() -> list[int]:
    """Pick 16 distinct 3x3 masks with a balanced spread of fill counts."""
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


def _glyph(mask: int, color: int) -> np.ndarray:
    """Render a 3x3 bitmask as a 5x5 cell with a 1-pixel teal border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _line_cells() -> list[tuple[int, int]]:
    """Pick one complete table row or column and shuffle the query order."""
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    """Pick `count` cells that do not form a full single row or column."""
    if count == 3:
        rs = random.sample(range(TABLE_DIM), random.choice((2, 3)))
        cs = random.sample(range(TABLE_DIM), random.choice((2, 3)))
        cells = [(rs[0], cs[0]), (rs[1], cs[1])]
        pool = [(r, c) for r in rs for c in cs if (r, c) not in cells]
        cells.append(random.choice(pool))
    else:
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

    assert len(cells) == count and len(set(cells)) == count
    assert len({r for r, _c in cells}) > 1 and len({c for _r, c in cells}) > 1
    random.shuffle(cells)
    return cells


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    """Erase unmatched table glyphs, highlight matches, then color the verdict areas."""
    out = grid.copy()
    selected = set(cells)
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            if (r_i, c_i) in selected:
                cell[cell == BG] = GREEN
            else:
                cell[cell != BG] = BG
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    """Generate one valid instance and reject it if the solver oracle disagrees."""
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    masks = _balanced_shape_masks()
    colors = _balanced_color_list()
    glyphs: list[np.ndarray] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], 0, COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    # D14 anti-degeneration check: on every valid sample, at least one unmatched
    # glyph pixel must be erased relative to the input table.
    table_in = grid[ROWS[0] : ROWS[-1] + CELL, COLS[0] : COLS[-1] + CELL]
    table_out = actual[ROWS[0] : ROWS[-1] + CELL, COLS[0] : COLS[-1] + CELL]
    erased_pixels = int(np.count_nonzero((table_in != BG) & (table_out == BG)))
    if erased_pixels <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 D14 input")
