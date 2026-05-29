# Combo variant of d8e07eb2 (index 9)
# combo_spec: D4/3+D14/0
# Name: 擦除匹配字形+擦除未匹配字形

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis:
#   combo_variant:
#     spec: D4/3 + D14/0
#     intent:
#       - matched reference-table cells lose their glyphs and keep only a green fill
#       - unmatched reference-table cells also lose their glyphs, but revert to pure bg
#       - the table therefore becomes a binary board: green blank cells for matches,
#         teal blank cells for non-matches
#   a_specific_behavior_trace:
#     input_pixels:
#       - 30x22 canvas, bg 8, separator rows at r=5 and r=27 in color 6
#       - rows 0..4 hold up to four copied 5x5 query glyph cells
#       - rows 7..26 hold a 4x4 table of 5x5 cells; each cell contains one centered
#         3x3 monochrome glyph with a 1-pixel bg border
#     original_behavior_step_by_step:
#       - read query glyph signatures from the top band
#       - find exact color+shape matches in the table
#       - for matched cells only, recolor bg pixels 8->3 while preserving glyph pixels
#       - leave unmatched cells unchanged
#     fused_behavior_step_by_step:
#       - read the same query signatures and find the same matched coordinates
#       - if a table cell is matched: recolor its bg pixels 8->3, then erase the glyph
#         pixels back to bg; result is a solid 5x5 green square
#       - if a table cell is unmatched: erase its glyph pixels back to bg; result is a
#         solid 5x5 teal square
#       - success/fail verdict logic is unchanged: only the table rendering is altered
#     concrete_example:
#       - suppose queries match table cells {(1,0), (1,1), (1,2), (1,3)}
#       - original output shows row 1 as green backgrounds with four colored glyphs still
#         visible, while the other 12 table glyphs remain visible too
#       - fused output shows row 1 as four fully green 5x5 blocks and the other 12 cells as
#         fully teal 5x5 blanks; no table glyph survives anywhere
#       - the difference is visible at two levels: matched cells lose their inner pattern,
#         and unmatched cells also lose theirs, so the table becomes a pure occupancy mask
#   b_complexity_calibration:
#     - Keep the original 4x4 geometry, 16 distinct glyph signatures, and 3-or-4 query
#       choices. This preserves enough structural variety for line-vs-nonline decoding.
#     - The fused transform is not over-simplified because humans must still infer matching
#       from the input table and queries; the output only reveals the selected positions.
#     - The effect is strong in every sample because there are always at least 12 unmatched
#       cells and at least 3 matched cells, so many glyph pixels are erased.
#   c_implicit_assumption_review:
#     - `_constant_non_bg_rows()` assumes the separator rows are the only full-width constant
#       non-bg rows. Generator preserves rows 5 and 27 exactly for that purpose.
#     - `_cell_signature()` assumes glyph identity is read from the input grid, where cells
#       still contain visible monochrome glyphs. The solver must never re-read signatures from
#       the mutated output, because the fused transform removes all table glyphs.
#     - Query uniqueness still matters: success checks compare `len(selected)` with the number
#       of query signatures, so generator keeps all query glyphs distinct.
#     - Table-grid inference assumes the top-left table cell has a 1-pixel border around the
#       first glyph, so `row0 = first_glyph_r - 1` and `col0 = first_glyph_c - 1` remain valid.
#     - The old branch structure that only touched matched cells is no longer sufficient:
#       unmatched cells must also be explicitly erased, otherwise D14/0 would be missing.
#     - Erase ordering matters only locally. For matched cells, `cell[cell == bg] = 3` must run
#       before `cell[cell != 3] = bg`, or a naive `cell[cell != bg] = bg` after filling could wipe
#       the green block itself. The implementation therefore uses mutually exclusive masks.
#   d_placement_legality:
#     - The table uses fixed origins ROWS=(7,12,17,22), COLS=(1,6,11,16); every 5x5 paste fits
#       inside the 30x22 canvas.
#     - Query glyphs are pasted only into the aligned top-band slots starting at the same COLS;
#       three-query cases leave only the trailing slot empty, matching the original task spec.
#     - No extra objects are introduced, so there is no new overlap mode to manage.
#   e_visual_observability:
#     - Across 2-4 examples, a human can see that every output table cell is blank, with matches
#       marked only by green solid blocks and non-matches by teal blanks.
#     - This is more observable than the original because distracting colored table glyphs vanish.
#     - Success is still legible from the green query band and green verdict band; failure remains
#       legible from the unchanged query band and red verdict band.
#   f_visual_cleanliness:
#     - The fused output removes rather than adds overlays, so the mid-table is visually cleaner
#       than both the original task and the single D14 variant.
#     - Because matched and unmatched cells collapse to only two appearances, the resulting board
#       is easy to parse as a selected-position mask.
#   g_color_inference:
#     - No hidden numeric color rule is introduced. Green means selected table position and, on
#       success, also paints the query-band background and verdict band. Red remains fail-only.
#     - Input glyph palette still excludes green, so every green table block is guaranteed to be
#       an output-only marker rather than an input artifact.
#   quantifiable_non_degeneracy:
#     - Let `surviving_original_fg` be the count of table pixels whose non-bg input color survives
#       unchanged at the same location in the output. In this combo it must be 0, because every
#       table glyph pixel is either turned green (matched cell) or turned back to bg (unmatched).
#     - Let `green_table_pixels` be the number of green pixels inside the output table. It should
#       equal `25 * matched_cell_count`, because each matched cell becomes a solid 5x5 green block.
#     - These metrics distinguish the combo from the original task: the original preserves matched
#       glyph colors and leaves unmatched glyphs visible, so `surviving_original_fg` would be >0.
#   post_change_main_assumptions_checked_against_generator:
#     - Every table cell contains exactly one non-empty glyph in the input.
#     - The table always has at least one unmatched cell, because the query count is at most 4
#       and the table has 16 cells.
#     - Success samples always use 4 queries that form one complete row or column; fail samples
#       use 3 queries or 4 non-collinear/non-co-row queries.


GREEN = 3
RED = 2
BG = 8
CELL = 5
TABLE_DIM = 4
GRID_H, GRID_W = 30, 22
TOP_SEP_ROW = 5
BOTTOM_SEP_ROW = 27
QUERY_BAND_H = 5
VERDICT_ROW = 28
SEP = 6
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5
COLOR_MIN, COLOR_MAX = 1, 3
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")
MAX_RETRIES = 300


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
    Match the query glyphs against a reference table and reduce the table to blank hit/miss cells.

    Input structure:
    - A teal canvas is split by two full-width separator rows in color 6.
    - The top band contains up to four 5x5 query cells, each holding one centered 3x3 glyph.
    - The middle band is a 4x4 table of the same 5x5 cells; glyph identity depends on exact
      shape together with color.

    Rule:
    - For every matched table cell, erase the glyph and leave a solid green 5x5 block.
    - For every unmatched table cell, erase the glyph back to the teal background so the cell
      becomes blank.
    - If all four query slots are occupied and the matched cells make one complete table row or
      one complete table column, also green-fill the query-band background and the verdict band;
      otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: The lookup still uses color+shape equality from the input, but the output table
    no longer preserves any glyphs at all; it becomes a clean occupancy map of matches vs misses.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = int(get_bg(g))
    cell_size = CELL

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
                cell[:, :] = GREEN
            else:
                cell[cell != bg] = bg

    complete_query = len(query_sigs) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_sigs) and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED

    return out


# --------------- generator (o5x) ---------------


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
    """Assign 16 colors so each palette color appears 1-3 times."""
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
    """Erase all table glyphs; matched cells become solid green blocks."""
    out = grid.copy()
    selected = set(cells)
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            if (r_i, c_i) in selected:
                cell[:, :] = GREEN
            else:
                cell[cell != BG] = BG
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _table_metrics(inp: np.ndarray, out: np.ndarray, matched_count: int) -> tuple[int, int]:
    """Return surviving original-fg count and green pixel count in the output table."""
    table_in = inp[ROWS[0] : ROWS[-1] + CELL, COLS[0] : COLS[-1] + CELL]
    table_out = out[ROWS[0] : ROWS[-1] + CELL, COLS[0] : COLS[-1] + CELL]
    surviving_original_fg = int(np.count_nonzero((table_in != BG) & (table_out == table_in)))
    green_table_pixels = int(np.count_nonzero(table_out == GREEN))
    assert green_table_pixels == matched_count * CELL * CELL
    return surviving_original_fg, green_table_pixels


def _gen_attempt() -> np.ndarray | None:
    """Generate one valid combo sample and reject it if solver or metrics disagree."""
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

    surviving_original_fg, green_table_pixels = _table_metrics(grid, actual, len(query_cells))
    if surviving_original_fg != 0:
        return None
    if green_table_pixels <= 0:
        return None

    # Anti-degeneration: compared with the original task, matched glyph pixels must no longer
    # survive in their original colors; they are absorbed into solid green blocks instead.
    matched_fg_converted = 0
    for r_i, c_i in query_cells:
        cell_in = grid[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell_out = actual[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        matched_fg_converted += int(np.count_nonzero((cell_in != BG) & (cell_out == GREEN)))
    if matched_fg_converted <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
