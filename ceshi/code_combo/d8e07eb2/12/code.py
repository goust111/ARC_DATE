# Combo variant of d8e07eb2 (index 12)
# combo_spec: D5/2+D6/2
# Name: 对角线判定+渐变颜色反馈

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
#    - The canvas layout stays unchanged: a 5-row query band, a 4x4 table of
#      5x5 cells, and a 2-row verdict band separated by two full-width pink rows.
#    - As in the base task, each query glyph is an exact color+shape copy of one
#      reference-table cell, and every matched table cell is highlighted by
#      turning only its teal background pixels green.
#    - The fused rule changes two linked stages:
#        1. geometry test: success is no longer "one full row/column"; it is now
#           "exactly one full main diagonal or anti-diagonal".
#        2. verdict encoding: the bottom band is no longer binary. It encodes the
#           best diagonal overlap count:
#             4 cells on one diagonal -> green  (3)
#             3 cells on one diagonal -> yellow (4)
#             2 cells on one diagonal -> orange (7)
#             0 or 1 cell             -> red    (2)
#    - Small trace on a 4x4 table:
#        original success: {(1,0),(1,1),(1,2),(1,3)} because it is a full row.
#        combo output    : still highlight those four cells, but verdict is red,
#                          because that row hits each diagonal only once.
#        partial combo   : three cells from {(0,0),(1,1),(2,2),(3,3)} give a
#                          yellow verdict, showing "almost a diagonal" rather
#                          than the old all-or-nothing failure.
#        full combo      : {(0,0),(1,1),(2,2),(3,3)} turns both query band and
#                          verdict band green.
#
# b) Complexity calibration
#    - The table is still 4x4, so the diagonal target remains visually simple:
#      exactly four positions are relevant on each candidate diagonal.
#    - The graded verdict uses only four colors, which is enough to communicate
#      distance-to-target without creating an unlearnable color code.
#    - The generator explicitly produces all four overlap levels. Yellow and
#      orange examples prevent the task from collapsing back to a binary rule,
#      while red row/column examples reject the original row/column hypothesis.
#
# c) Implicit assumption audit
#    - The old `same_row` / `same_col` logic is invalid and must be replaced by
#      exact diagonal-set matching for success plus a separate overlap counter
#      for partial feedback.
#    - `complete_query = len(query_sigs) == len(col_starts)` still matters, but
#      only for success. Partial verdicts remain meaningful when there are only
#      3 query glyphs.
#    - The solver now assumes the verdict color depends on visible geometry, not
#      just pass/fail. The generator therefore has to control the matched-cell
#      sets by diagonal-overlap bucket, not merely by success boolean.
#    - Exact glyph matching by color+shape is unchanged, so the generator still
#      guarantees all 16 table signatures are unique and all query signatures are
#      drawn from distinct table cells.
#
# d) Placement legality
#    - Query glyphs are pasted only into the four existing 5x5 slots in rows
#      0..4, columns 1/6/11/16, so no overlap or boundary risk is introduced.
#    - Table glyphs remain one per fixed 5x5 cell in the 4x4 table.
#    - Partial-overlap samplers add non-diagonal cells only from coordinates that
#      are on neither diagonal, preventing accidental escalation from 2->3 or
#      3->4 overlap.
#
# e) Visual observability
#    - A human can recover the fused rule from a few examples because:
#        1. green outputs always show a full diagonal;
#        2. yellow/orange outputs show the same type of highlighted cells, but
#           with fewer cells lying on one diagonal;
#        3. some red outputs are full rows/columns, which would have passed in
#           the original task and therefore expose the D5 change directly.
#    - No hidden use of color ids, ordering, or other non-visual metadata is
#      introduced: only visible positions and the solid verdict-band color matter.
#
# f) Visual cleanliness
#    - The table highlight remains local to matched cells, separators stay pink,
#      and the verdict band is a single solid color, so the output still has
#      three cleanly separated semantic regions.
#    - The query band is only flood-filled on full diagonal success. Partial
#      progress is communicated solely by the bottom band, which avoids mixing
#      multiple competing color signals in the same region.
#
# g) Change inferability
#    - The new color mapping is intentionally structured around one visible count:
#      "how many matched cells belong to the same full diagonal?".
#    - Because the generator samples 4, 3, 2, and 0/1 overlap cases repeatedly,
#      each verdict color gets concrete geometric support across examples instead
#      of appearing as an arbitrary remapping.


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


def _diagonal_cell_sets(
    dim: int,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    main_diag = frozenset((i, i) for i in range(dim))
    anti_diag = frozenset((i, dim - 1 - i) for i in range(dim))
    return main_diag, anti_diag


def _best_diagonal_overlap(cells: set[tuple[int, int]] | frozenset[tuple[int, int]], dim: int) -> int:
    if not cells:
        return 0
    return max(len(cells & diag) for diag in _diagonal_cell_sets(dim))


def _verdict_color(overlap: int) -> int:
    if overlap >= 4:
        return 3
    if overlap == 3:
        return 4
    if overlap == 2:
        return 7
    return 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up query glyphs in a reference table and grade how completely they trace a diagonal.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a query row of 5x5 glyph cells.
    - Between separators is a square reference table built from the same 5x5
      cells; each glyph is identified by its 3x3 shape together with its color.

    Rule:
    - Green-fill the background inside every table cell whose glyph exactly
      appears in the query row.
    - If the matched table cells are exactly one full main diagonal or one full
      anti-diagonal, also green-fill the query band.
    - Fill the bottom verdict band by best diagonal coverage: 4 aligned matches
      -> green, 3 -> yellow, 2 -> orange, 0 or 1 -> red.

    Key insight: The highlighted table cells reveal WHICH reference entries were
    queried, while the bottom color reveals HOW CLOSE those positions are to one
    complete diagonal; only an exact 4-cell diagonal also floods the query band.
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

    success = False
    best_overlap = 0
    if len(row_starts) == len(col_starts):
        selected_set = frozenset(selected)
        diag_sets = _diagonal_cell_sets(len(row_starts))
        best_overlap = max(len(selected_set & diag) for diag in diag_sets)
        success = (
            len(query_sigs) == len(col_starts)
            and len(selected) == len(query_sigs)
            and selected_set in diag_sets
        )

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3

    out[bottom_sep + 1 :, :] = _verdict_color(best_overlap)
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
YELLOW = 4
RED = 2
ORANGE = 7
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
OVERLAP_LEVELS = (4, 3, 2, 1)
OVERLAP_WEIGHTS = (6, 5, 5, 6)

# -- Generation --
MAX_RETRIES = 300

MAIN_DIAGONAL = tuple((i, i) for i in range(TABLE_DIM))
ANTI_DIAGONAL = tuple((i, TABLE_DIM - 1 - i) for i in range(TABLE_DIM))
DIAGONAL_CHOICES = (MAIN_DIAGONAL, ANTI_DIAGONAL)
DIAGONAL_SETS = tuple(frozenset(cells) for cells in DIAGONAL_CHOICES)
ALL_CELLS = tuple((r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM))
NON_DIAGONAL_CELLS = tuple(
    cell for cell in ALL_CELLS if cell not in DIAGONAL_SETS[0] and cell not in DIAGONAL_SETS[1]
)


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


def _balanced_color_list() -> list[int]:
    """Assign 16 colors so each glyph color appears 1-3 times."""
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks() -> list[int]:
    """Pick 16 distinct 3x3 masks with a balanced fill-count distribution."""
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
    """Render a 3x3 bitmask as a 5x5 array with a 1-pixel teal border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _is_full_diagonal(cells: list[tuple[int, int]]) -> bool:
    return len(cells) == TABLE_DIM and frozenset(cells) in DIAGONAL_SETS


def _full_diagonal_cells() -> list[tuple[int, int]]:
    cells = list(random.choice(DIAGONAL_CHOICES))
    random.shuffle(cells)
    return cells


def _three_overlap_cells() -> list[tuple[int, int]]:
    diag = list(random.choice(DIAGONAL_CHOICES))
    core = random.sample(diag, 3)
    if random.random() < 0.5:
        cells = core
    else:
        cells = core + [random.choice(NON_DIAGONAL_CELLS)]
    assert _best_diagonal_overlap(frozenset(cells), TABLE_DIM) == 3
    assert not _is_full_diagonal(cells)
    random.shuffle(cells)
    return cells


def _two_overlap_cells() -> list[tuple[int, int]]:
    diag = list(random.choice(DIAGONAL_CHOICES))
    count = random.choice((3, 4))
    cells = random.sample(diag, 2) + random.sample(NON_DIAGONAL_CELLS, count - 2)
    assert _best_diagonal_overlap(frozenset(cells), TABLE_DIM) == 2
    assert not _is_full_diagonal(cells)
    random.shuffle(cells)
    return cells


def _one_or_zero_overlap_cells() -> list[tuple[int, int]]:
    if random.random() < 0.7:
        if random.random() < 0.5:
            r = random.randrange(TABLE_DIM)
            cells = [(r, c) for c in range(TABLE_DIM)]
        else:
            c = random.randrange(TABLE_DIM)
            cells = [(r, c) for r in range(TABLE_DIM)]
    else:
        count = random.choice((3, 4))
        cells = random.sample(NON_DIAGONAL_CELLS, count)
    assert _best_diagonal_overlap(frozenset(cells), TABLE_DIM) <= 1
    assert not _is_full_diagonal(cells)
    random.shuffle(cells)
    return cells


def _sample_query_cells() -> list[tuple[int, int]]:
    target = random.choices(OVERLAP_LEVELS, weights=OVERLAP_WEIGHTS, k=1)[0]
    if target == 4:
        return _full_diagonal_cells()
    if target == 3:
        return _three_overlap_cells()
    if target == 2:
        return _two_overlap_cells()
    return _one_or_zero_overlap_cells()


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    """Build the expected output for the fused diagonal-overlap rule."""
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN

    if _is_full_diagonal(cells):
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN

    overlap = _best_diagonal_overlap(frozenset(cells), TABLE_DIM)
    out[VERDICT_ROW:, :] = _verdict_color(overlap)
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

    query_cells = _sample_query_cells()
    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], 0, COLS[slot])

    expected = _expected_output(grid, query_cells)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 D5/2+D6/2 combo input")
