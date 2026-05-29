# Combo variant of d8e07eb2 (index 18)
# combo_spec: D5/1+D9/0
# Name: 2x2子矩阵判定+固定4查询

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, paste


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


def _forms_2x2_block(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    cells = {(int(r), int(c)) for r, c in cells}
    if len(cells) != 4:
        return False
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    return (
        len(rows) == 2
        and len(cols) == 2
        and rows[1] == rows[0] + 1
        and cols[1] == cols[0] + 1
    )


def _is_complete_line(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    cells = {(int(r), int(c)) for r, c in cells}
    if len(cells) != 4:
        return False
    rows = {r for r, _c in cells}
    cols = {c for _r, c in cells}
    return len(rows) == 1 or len(cols) == 1


# Design analysis
# ----------------
# a) Concrete behavior:
#    - The physical layout stays the same as the seed task: a 5-row query band,
#      separator row, one blank spacer row, a 4x4 table of 5x5 cells, another
#      separator row, then a 2-row verdict band.
#    - Each visible glyph is still identified by the exact pair (3x3 shape,
#      color). Matching remains one-to-one because the 16 table signatures are
#      unique.
#    - The fused change affects both the query count and the success geometry:
#      there are now always exactly four queries, and success happens only when
#      the four matched table cells are one contiguous 2x2 block.
#    - Small trace:
#        old success example: queries copied from {(1,0), (1,1), (1,2), (1,3)}
#        old verdict: green, because that is a full row
#        combo verdict: red, because one row is not a 2x2 block
#        combo success example: queries copied from {(1,1), (1,2), (2,1), (2,2)}
#        combo verdict: green, because rows {1,2} and cols {1,2} form one
#        adjacent 2x2 submatrix
#      So the visible difference from the original task is not the number of
#      queries anymore; it is the spatial arrangement of the highlighted table
#      cells.
#
# b) Complexity calibration:
#    - Fixing the query count at four avoids trivial count-based failure and
#      keeps every sample in the same "four-way lookup" regime.
#    - The new success predicate remains compact and human-visible: a contiguous
#      square in the 4x4 table is easy to spot, but it is meaningfully different
#      from the original row/column line test.
#    - The fail generator deliberately includes complete rows and columns among
#      its failure modes. That gives explicit counterexamples showing that "four
#      queries" and even "full line" are no longer enough.
#
# c) Old implicit assumptions that must change:
#    - In the seed solver, `len(query_sigs) == len(col_starts)` effectively meant
#      "all four slots are filled, therefore a line is possible". In the combo,
#      all four slots are always filled by construction, but that is only a
#      necessary condition; the matched coordinates must additionally satisfy the
#      adjacency test for a 2x2 block.
#    - The original `same_row or same_col` test is invalid. It must be replaced
#      by a check for exactly two adjacent row indices and two adjacent column
#      indices.
#    - Generator logic that used 3-query failures is no longer legal under D9/0.
#      Every generated sample must place four distinct query glyphs.
#    - Generator logic that used an adjacent 2x2 rectangle as a generic failure
#      pattern is now invalid because that pattern is precisely the new success
#      case.
#
# d) Placement legality:
#    - Table anchors, cell size, glyph borders, and separator rows remain
#      unchanged from the seed specification, so the scene still fits in 30x22.
#    - Query glyphs are pasted only into the four existing top slots, one per
#      slot, and all copied source cells are distinct. That prevents overlap.
#    - Failure patterns are sampled from the 4x4 lattice only after checking they
#      are not contiguous 2x2 blocks, so the solver never sees an accidental
#      success mislabeled as failure.
#
# e) Visual observability:
#    - A human can infer the combo from a few examples because successful outputs
#      always highlight a compact 2x2 square in the table, while some failures
#      highlight four cells that still make a perfect row or column but do not
#      turn the query band green.
#    - This directly separates the new rule from both the original task and from
#      a naive "all four queries present means success" hypothesis.
#
# f) Visual cleanliness:
#    - No new markers are introduced. The only overlays remain green fills inside
#      matched table cells, optional green fill of query-band background, and the
#      bottom verdict color.
#    - Because the generator still uses the original spacing and centered 3x3
#      glyphs, the 2x2-vs-line distinction stays readable rather than crowded.
#
# g) Change inferability:
#    - There is no arbitrary color remapping. Colors still only help identify
#      which table glyph matches which query glyph.
#    - The new inference burden is purely spatial: among four exact matches, only
#      a contiguous 2x2 table block triggers the green verdict.
#
# Post-change solver assumptions reviewed against the generator:
#    - There are exactly two separator rows spanning the full width: guaranteed.
#    - The table is a 4x4 grid of 5x5 cells aligned to the inferred lattice:
#      guaranteed by fixed ROWS/COLS placement.
#    - Every query glyph is copied from a distinct table cell, so query
#      signatures are distinct and each has exactly one match in the table:
#      guaranteed by construction.
#    - Every generated sample has exactly four queries: guaranteed.
#    - Success samples always form one contiguous 2x2 block, and failure samples
#      never do: guaranteed by `_block_cells()` and `_fail_cells()`.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match four query glyphs against the table and accept only contiguous 2x2 hits.

    Input structure:
    - A teal canvas is divided by two full-width pink separator rows.
    - The top band contains exactly four 5x5 query cells.
    - The middle band is a 4x4 reference table of the same 5x5 cells, each
      holding a centered 3x3 colored glyph that must be matched by both shape
      and color.

    Rule:
    - Find every table cell whose glyph exactly appears among the four queries
      and recolor that cell's background pixels green.
    - If the four matched table positions are exactly one contiguous 2x2
      submatrix, also green-fill the background of the query band and the entire
      verdict band.
    - Otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: Four queries are always present, so the verdict is determined
    by the 2D arrangement of the four matched table cells, not by the count and
    not by a one-dimensional row/column line.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = get_bg(g)
    cell_size = 5
    required_matches = 4

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

    success = (
        len(query_sigs) == required_matches
        and len(selected) == required_matches
        and _forms_2x2_block(selected)
    )

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
REQUIRED_QUERIES = 4
BLOCK_SIDE = 2

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_FAIL_LINE = 0.45
P_FAIL_GAPPED_RECT = 0.25
P_FAIL_BENT = 0.15

# -- Generation --
MAX_RETRIES = 300


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


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _block_cells() -> list[tuple[int, int]]:
    top = random.randrange(TABLE_DIM - 1)
    left = random.randrange(TABLE_DIM - 1)
    cells = [(top + dr, left + dc) for dr in range(BLOCK_SIDE) for dc in range(BLOCK_SIDE)]
    random.shuffle(cells)
    return cells


def _line_cells() -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _gapped_rect_cells() -> list[tuple[int, int]]:
    row_pairs = [(a, b) for a in range(TABLE_DIM) for b in range(a + 1, TABLE_DIM)]
    col_pairs = [(a, b) for a in range(TABLE_DIM) for b in range(a + 1, TABLE_DIM)]
    while True:
        r0, r1 = random.choice(row_pairs)
        c0, c1 = random.choice(col_pairs)
        cells = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if not _forms_2x2_block(cells):
            random.shuffle(cells)
            return cells


def _bent_cells() -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        dropped_c = random.randrange(TABLE_DIM)
        new_r = random.choice([rr for rr in range(TABLE_DIM) if rr != r])
        cells = [(r, c) for c in range(TABLE_DIM) if c != dropped_c] + [(new_r, dropped_c)]
    else:
        c = random.randrange(TABLE_DIM)
        dropped_r = random.randrange(TABLE_DIM)
        new_c = random.choice([cc for cc in range(TABLE_DIM) if cc != c])
        cells = [(r, c) for r in range(TABLE_DIM) if r != dropped_r] + [(dropped_r, new_c)]
    assert len(cells) == REQUIRED_QUERIES and len(set(cells)) == REQUIRED_QUERIES
    assert not _forms_2x2_block(cells)
    random.shuffle(cells)
    return cells


def _scatter_cells() -> list[tuple[int, int]]:
    universe = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM)]
    while True:
        cells = random.sample(universe, REQUIRED_QUERIES)
        if not _forms_2x2_block(cells) and not _is_complete_line(cells):
            random.shuffle(cells)
            return cells


def _fail_cells() -> list[tuple[int, int]]:
    roll = random.random()
    if roll < P_FAIL_LINE:
        cells = _line_cells()
    elif roll < P_FAIL_LINE + P_FAIL_GAPPED_RECT:
        cells = _gapped_rect_cells()
    elif roll < P_FAIL_LINE + P_FAIL_GAPPED_RECT + P_FAIL_BENT:
        cells = _bent_cells()
    else:
        cells = _scatter_cells()

    assert len(cells) == REQUIRED_QUERIES
    assert len(set(cells)) == REQUIRED_QUERIES
    assert not _forms_2x2_block(cells)
    return cells


def _expected_output(
    grid: np.ndarray, cells: list[tuple[int, int]], success: bool
) -> np.ndarray:
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
    glyphs: list[np.ndarray] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    query_cells = _block_cells() if success else _fail_cells()

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
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
