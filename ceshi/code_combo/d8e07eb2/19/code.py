# Combo variant of d8e07eb2 (index 19)
# combo_spec: D5/2+D9/0
# Name: 对角线判定+固定4查询

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
#    - The canvas geometry remains the original 30x22 layout: a 5-row query
#      band, a full-width separator, a 4x4 table of 5x5 cells, another full-
#      width separator, then a 2-row verdict band.
#    - Every query glyph is still an exact copy of one table glyph, identified
#      by the visible pair (3x3 shape, glyph color). Matching cells in the
#      table are highlighted exactly as before by turning only their teal
#      background pixels green.
#    - The fused change combines two edits:
#        1. success geometry changes from "one full row/column" to
#           "one full main diagonal or anti-diagonal";
#        2. query count is no longer 3-4; it is always exactly 4.
#    - Small trace on a 4x4 table:
#        original success: {(2,0),(2,1),(2,2),(2,3)} -> whole row
#        fused result   : same four cells must fail because they are not a
#                         diagonal, even though all four query slots are filled.
#        fused success  : {(0,0),(1,1),(2,2),(3,3)} or
#                         {(0,3),(1,2),(2,1),(3,0)}
#      This difference is the core interaction: fixing the query count to 4
#      removes the old "short query" failure signal, so geometry alone decides.
#
# b) Complexity calibration
#    - Fixing the query count to 4 is safe because the table is 4x4, so a full
#      diagonal also has cardinality 4. The task stays visually simple while
#      still supporting both success and failure.
#    - To keep the D5 diagonal change observable, the generator intentionally
#      emits many row/column fail cases. Those would pass under the original
#      solver but fail here, which makes the fused rule distinguishable from
#      both the baseline and a trivial "all 4 queries mean success" shortcut.
#    - Fail modes are limited to a small set of visibly distinct 4-cell layouts
#      (rows, columns, near-diagonals, rectangles, non-diagonal permutations),
#      which gives repeated evidence without introducing too many categories.
#
# c) Implicit-assumption audit
#    - `_constant_non_bg_rows` still assumes the two separators are the only
#      full-width uniform non-background rows. The generator preserves that.
#    - Table-grid inference still depends on the first table glyph being inset
#      by one background pixel within a 5x5 cell. The generator still centers
#      every 3x3 glyph inside a 5x5 cell, so `row0/col0` inference remains valid.
#    - Query extraction originally tolerated 3 or 4 glyphs. In the fused rule,
#      `main()` must not assume fewer queries imply failure by count alone.
#      Instead it checks `len(query_sigs) == len(col_starts)` and then evaluates
#      the selected positions against the two diagonal sets.
#    - The old `same_row` / `same_col` test is invalid after D5/2 and must be
#      replaced by exact set comparison with the main and anti diagonals.
#    - Matching still assumes each query signature maps to exactly one table
#      cell. The generator guarantees global uniqueness of (shape, color) pairs
#      by using distinct shapes across all 16 cells.
#
# d) Placement legality
#    - Query glyphs only occupy the four pre-existing top-band slots at columns
#      1, 6, 11, 16. With fixed count 4, all four slots are filled and there is
#      still no overlap or out-of-bounds risk.
#    - The 4x4 table layout remains fixed at rows (7,12,17,22) and cols
#      (1,6,11,16), so each 5x5 cell fits exactly between the separators.
#    - Every fail sampler produces four distinct cells and explicitly rejects
#      either diagonal set, so generator labels cannot drift.
#
# e) Visual observability
#    - From 2-4 examples, a human can see that all samples always contain four
#      query glyphs, so query count is no longer informative.
#    - Success samples show the highlighted table cells lying on one diagonal.
#      Fail samples can still show four highlighted cells on a complete row or
#      column, which directly disproves the original rule.
#    - No invisible ordering by numeric color id is used; all logic depends on
#      visible glyph identity and visible table geometry.
#
# f) Visual cleanliness
#    - Region semantics stay clean: query band shows the copied glyphs, the
#      table shows matches via green background fill, separators stay pink, and
#      the verdict band stays a solid green/red slab.
#    - There is no extra overlay inside glyph strokes; only background pixels of
#      selected cells are recolored, so glyph shapes remain readable.
#
# g) Change inferability
#    - The fused task does not add any new color mapping. Green still marks
#      matched-cell highlighting and successful verdict, red still marks failure.
#    - The new inference burden is purely structural: "exactly four queries" and
#      "diagonal completeness instead of row/column completeness". The generator
#      provides both diagonal successes and row/column failures to make that
#      structure learnable from few examples.


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
    Match four query glyphs against a reference table and test whether they trace one full diagonal.

    Input structure:
    - A teal canvas is split by two full-width pink separator rows into a top
      query band, a middle 4x4 reference table of 5x5 cells, and a bottom
      verdict band.
    - Each occupied query slot contains a 3x3 colored glyph centered inside a
      5x5 cell, and the reference table uses the same glyph format.

    Rule:
    - Find every table cell whose glyph exactly matches one of the four query
      glyphs, and green-fill only the teal background inside those matched cells.
    - If the four matched cells are exactly the table's main diagonal or exactly
      its anti-diagonal, also green-fill the query band and the verdict band.
    - Otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: Because the query count is always fixed at four, the verdict is
    determined purely by the matched geometry inside the 4x4 table: a complete
    diagonal passes, while complete rows or columns still fail.
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
        is_complete_diagonal = frozenset(selected) in _diagonal_cell_sets(len(row_starts))
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
FAIL_MODES = ("row", "row", "col", "col", "off_main", "off_anti", "rect", "perm")

# -- Generation --
MAX_RETRIES = 300


def _has_full_bbox(mask: int) -> bool:
    """True iff the 3x3 bitmask touches all four outer sides of the 3x3 box."""
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
    """Assign 16 colors so each allowed glyph color appears between 1 and 3 times."""
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks() -> list[int]:
    """Pick 16 distinct 3x3 masks with a roughly balanced fill-count distribution."""
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
    """Render a 3x3 bitmask as a 5x5 glyph cell with a 1-pixel teal border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _main_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, i) for i in range(TABLE_DIM)]


def _anti_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, TABLE_DIM - 1 - i) for i in range(TABLE_DIM)]


def _diagonal_cells() -> list[tuple[int, int]]:
    """Pick one complete diagonal of four cells in random query-slot order."""
    cells = list(random.choice((_main_diagonal_cells(), _anti_diagonal_cells())))
    random.shuffle(cells)
    return cells


def _is_full_diagonal(cells: list[tuple[int, int]]) -> bool:
    return len(cells) == TABLE_DIM and frozenset(cells) in _diagonal_cell_sets(TABLE_DIM)


def _fail_cells() -> list[tuple[int, int]]:
    """Pick four distinct cells that are not either full diagonal."""
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

    assert len(cells) == TABLE_DIM and len(set(cells)) == TABLE_DIM
    assert not _is_full_diagonal(cells)
    random.shuffle(cells)
    return cells


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    """Build the target output by highlighting matched cells and the verdict area."""
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
    """Try to generate one valid fused-variant input; return None on oracle mismatch."""
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
    query_cells = _diagonal_cells() if success else _fail_cells()

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
    raise RuntimeError("failed to generate a valid d8e07eb2 combo-19 input")
