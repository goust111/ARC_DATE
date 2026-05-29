# Variant of d8e07eb2
# Dimension: D5 — 判定条件（行/列完整性） (match_cond)
# Change: 若所有查询字形的匹配单元格恰好构成参考表的一整行或一整列，则判定为绿色（通过）；否则判定为红色（失败）... → 对角线完整性判定

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D5 variant
# ==================================
# a) Concrete behavior trace
#    - The canvas stays exactly the same: a 5-row query band, a 4x4 reference
#      table of 5x5 cells, and a 2-row verdict band separated by two full-width
#      pink rows.
#    - Each query glyph is still an exact color+shape copy of one table cell.
#      The solver still highlights every matched table cell by filling that
#      cell's teal background with green.
#    - The only rule change is the verdict condition:
#        original: success iff the 4 matched cells are one whole row or column
#        variant : success iff the 4 matched cells are exactly one whole diagonal
#                  (main diagonal or anti-diagonal)
#    - Small trace on a 4x4 table:
#        original success example: {(1,0),(1,1),(1,2),(1,3)}  -> whole row
#        variant  failure there  : same four cells are not a diagonal
#        variant  success example: {(0,0),(1,1),(2,2),(3,3)}  -> main diagonal
#      In both versions the same cells are highlighted in the table; the only
#      difference is whether the query band and verdict band also turn green.
#
# b) Complexity calibration
#    - The table is still 4x4, so a diagonal means exactly four matched cells.
#      This keeps the task comparable to the original and visually simple.
#    - To make the D5 change visible, the generator deliberately includes fail
#      cases that are complete rows or complete columns. Those would pass under
#      the original rule but must fail here, so the variant cannot silently
#      collapse back to the old behavior.
#    - Query glyph count remains 3-4. Three-glyph queries are still useful as
#      obvious failures because they show that a diagonal must be complete.
#
# c) Old assumptions that must change
#    - `same_row` / `same_col` were the old success test. Those assumptions are
#      no longer valid and must be replaced by explicit diagonal-set matching.
#    - `complete_query = len(query_sigs) == len(col_starts)` still holds: a full
#      diagonal needs one query in every column slot, so successes still require
#      all 4 query slots to be occupied.
#    - Matching by exact glyph signature is unchanged; the variant does not alter
#      object identity, only how matched positions are evaluated.
#
# d) Placement legality
#    - Query glyphs are pasted only into the four pre-existing 5x5 slots in the
#      top band, exactly as before, so there is no overlap or boundary risk.
#    - Table glyphs remain one per 5x5 cell at fixed coordinates.
#    - Fail-mode samplers are constrained so selected cells are distinct and
#      never equal either full diagonal when `success=False`.
#
# e) Visual observability
#    - Humans can infer the new rule from a small sample set because:
#        1. success cases show four highlighted cells lying on one diagonal;
#        2. some fail cases show four highlighted cells lying on a row/column,
#           yet the verdict is red, which directly rejects the old hypothesis.
#    - No hidden sorting by color id or any non-visible quantity is introduced.
#
# f) Visual cleanliness
#    - The output marking scheme is unchanged: only background pixels are turned
#      green in matched cells, separators stay pink, and verdict rows are solid.
#      This preserves the clean region separation of the original task.
#
# g) Color-rule audit
#    - There is no new color remapping. Green still means "highlight/success"
#      and red still means "failure". The variant only changes which geometric
#      arrangement triggers the green verdict.


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


def _diagonal_cell_sets(dim: int) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    main_diag = frozenset((i, i) for i in range(dim))
    anti_diag = frozenset((i, dim - 1 - i) for i in range(dim))
    return main_diag, anti_diag


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up query glyphs in a reference table and mark whether they form a diagonal.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a query row of 5x5 glyph cells.
    - Between separators is a square reference table of the same 5x5 cells; a
      glyph is identified by both its color and its 3x3 shape.

    Rule:
    - Green-fill the background inside every table cell whose glyph exactly
      appears in the query row.
    - If the matched table cells are exactly one complete main diagonal or one
      complete anti-diagonal, also green-fill the query band and the bottom
      verdict band.
    - Otherwise leave the query band unchanged and fill the bottom verdict band red.

    Key insight: The bottom color is a yes/no verdict about the GEOMETRY of the
    matched positions in the table; four matches only pass when they trace one
    of the table's two full diagonals.
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
            if _cell_signature(g, r, c, cell_size, bg) not in query_set:
                continue
            selected.append((int(ri), int(ci)))
            cell = out[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = 3

    square_table = len(row_starts) == len(col_starts)
    complete_query = len(query_sigs) == len(col_starts)
    if square_table:
        diag_sets = _diagonal_cell_sets(len(row_starts))
        is_complete_diagonal = frozenset(selected) in diag_sets
    else:
        is_complete_diagonal = False
    success = complete_query and len(selected) == len(query_sigs) and is_complete_diagonal

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
P_SUCCESS = 0.5
P_QUERY_3 = 0.35
FAIL_MODES = ("row", "row", "col", "col", "off_main", "off_anti", "rect", "perm")

# -- Generation --
MAX_RETRIES = 300


def _has_full_bbox(mask: int) -> bool:
    """True if the 3x3 bitmask spans all four edges (non-empty top/bottom/left/right rows)."""
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

NON_DIAGONAL_PERMS = tuple(
    perm
    for perm in itertools.permutations(range(TABLE_DIM))
    if perm not in ((0, 1, 2, 3), (3, 2, 1, 0))
)


def _balanced_color_list() -> list[int]:
    """Assign 16 colors (one per cell) so each of the 8 palette colors appears 1-3 times."""
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks() -> list[int]:
    """Pick 16 distinct 3x3 shape masks with a balanced distribution of fill counts."""
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
    """Render a 3x3 bitmask as a CELL x CELL array with 1-pixel border of BG."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _main_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, i) for i in range(TABLE_DIM)]


def _anti_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, TABLE_DIM - 1 - i) for i in range(TABLE_DIM)]


def _diagonal_cells() -> list[tuple[int, int]]:
    """Pick one complete diagonal of 4 cells (shuffled order)."""
    cells = list(random.choice((_main_diagonal_cells(), _anti_diagonal_cells())))
    random.shuffle(cells)
    return cells


def _is_full_diagonal(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != TABLE_DIM:
        return False
    return frozenset(cells) in _diagonal_cell_sets(TABLE_DIM)


def _scatter_fail_triplet() -> list[tuple[int, int]]:
    rs = random.sample(range(TABLE_DIM), random.choice((2, 3)))
    cs = random.sample(range(TABLE_DIM), random.choice((2, 3)))
    cells = [(rs[0], cs[0]), (rs[1], cs[1])]
    pool = [(r, c) for r in rs for c in cs if (r, c) not in cells]
    cells.append(random.choice(pool))
    assert len(set(cells)) == 3
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    """Pick distinct cells that do NOT form a complete main/anti diagonal."""
    if count == 3:
        if random.random() < 0.5:
            cells = random.sample(random.choice((_main_diagonal_cells(), _anti_diagonal_cells())), 3)
        else:
            cells = _scatter_fail_triplet()
    else:
        mode = random.choice(FAIL_MODES)
        if mode == "row":
            r = random.randrange(TABLE_DIM)
            cells = [(r, c) for c in range(TABLE_DIM)]
        elif mode == "col":
            c = random.randrange(TABLE_DIM)
            cells = [(r, c) for r in range(TABLE_DIM)]
        elif mode == "off_main":
            diag = set(_main_diagonal_cells())
            removed = random.choice(list(diag))
            replacement_pool = [
                (r, c)
                for r in range(TABLE_DIM)
                for c in range(TABLE_DIM)
                if (r, c) not in diag
            ]
            cells = list((diag - {removed}) | {random.choice(replacement_pool)})
        elif mode == "off_anti":
            diag = set(_anti_diagonal_cells())
            removed = random.choice(list(diag))
            replacement_pool = [
                (r, c)
                for r in range(TABLE_DIM)
                for c in range(TABLE_DIM)
                if (r, c) not in diag
            ]
            cells = list((diag - {removed}) | {random.choice(replacement_pool)})
        elif mode == "rect":
            rs = random.sample(range(TABLE_DIM), 2)
            cs = random.sample(range(TABLE_DIM), 2)
            cells = [(r, c) for r in rs for c in cs]
        else:
            perm = random.choice(NON_DIAGONAL_PERMS)
            cells = [(r, perm[r]) for r in range(TABLE_DIM)]

    assert len(cells) == count and len(set(cells)) == count
    assert not _is_full_diagonal(cells)
    random.shuffle(cells)
    return cells


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    """Build the expected solver output by green-filling matched cells and verdict band."""
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
    """Try to generate one valid input grid; return None on oracle mismatch."""
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
        query_cells = _diagonal_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], 0, COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 D5-2 variant input")
