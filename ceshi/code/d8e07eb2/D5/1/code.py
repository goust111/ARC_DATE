# Variant of d8e07eb2
# Dimension: D5 — 判定条件（行/列完整性） (match_cond)
# Change: 若所有查询字形的匹配单元格恰好构成参考表的一整行或一整列，则判定为绿色（通过）；否则判定为红色（失败）... → 要求2x2子矩阵排列

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
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


# Design analysis for the D5 variant
# a) Concrete behavior:
#    The canvas layout stays unchanged: query band on top, 4x4 reference table in
#    the middle, verdict band at the bottom. Each query glyph is still copied from
#    one table cell and matched by exact color+shape. The only semantic change is
#    the success geometry. Originally, four matches passed when they occupied one
#    entire row or one entire column. In this variant, four matches pass only when
#    they occupy the four corners of one contiguous 2x2 table block, i.e. rows
#    {r, r+1} crossed with cols {c, c+1}. Example: {(1,1),(1,2),(2,1),(2,2)} now
#    passes, while {(1,0),(1,1),(1,2),(1,3)} used to pass but now fails.
# b) Complexity calibration:
#    The table remains 4x4 with 16 unique glyph signatures, so matching is still
#    unambiguous. Requiring a full 2x2 block keeps the rule visually legible while
#    being meaningfully different from a 1D line. Fail examples explicitly include
#    full rows/columns so the dataset does not silently collapse back to the
#    original task.
# c) Old implicit assumptions that must change:
#    The original solver relied on "all selected cells share one row/col" and on
#    "four queries means complete line". Here, four queries are still necessary,
#    but not sufficient; we must also require exactly two adjacent row indices and
#    two adjacent column indices. Generator-side fail modes that previously used a
#    2x2 rectangle can no longer be accepted blindly, because an adjacent rectangle
#    is now the success case.
# d) Placement legality:
#    Cell anchors, separators, and glyph borders are preserved. Queries are still
#    pasted only into the four top slots, the table is still aligned to the 5x5
#    lattice, and all chosen table cells are distinct, so no overlaps or
#    out-of-bounds placements occur.
# e) Visual observability:
#    Humans can now distinguish successful samples by seeing four highlighted table
#    cells forming a compact square rather than a straight line. Including some
#    failing full-row/full-column queries makes the new condition directly
#    observable from a few examples.
# f) Visual cleanliness:
#    No extra markers are added. The only overlays remain the original green cell
#    fills plus the verdict band, so the grid stays sparse and readable.
# g) Color inference:
#    No new color mapping is introduced. Colors still identify glyph instances only;
#    the novel behavior is purely spatial and visible from the arrangement of the
#    matched cells.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the query glyphs against the table and test whether they make a 2x2 block.

    Input structure:
    - The background is teal, split by two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells.
    - The middle band is a 4x4 table of the same 5x5 cells, each carrying a
      centered 3x3 glyph identified jointly by shape and color.

    Rule:
    - For every table cell whose glyph appears in the query band, recolor that
      cell's background pixels green.
    - If there are exactly four queried glyphs and their matched table positions
      are exactly one contiguous 2x2 submatrix, also green-fill the query band
      background and the verdict band.
    - Otherwise keep the query band as-is and fill the verdict band red.

    Key insight: The verdict depends on the matched cells forming a compact
    square block in the reference table, not a straight line and not the count alone.
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
BLOCK_SIDE = 2
BLOCK_AREA = BLOCK_SIDE * BLOCK_SIDE

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("line", "gapped_rect", "bent", "scatter")

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
    assert len(cells) == BLOCK_AREA and len(set(cells)) == BLOCK_AREA
    assert not _forms_2x2_block(cells)
    random.shuffle(cells)
    return cells


def _scatter_cells(count: int) -> list[tuple[int, int]]:
    universe = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM)]
    while True:
        cells = random.sample(universe, count)
        if not _forms_2x2_block(cells):
            random.shuffle(cells)
            return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    if count == 3:
        return _scatter_cells(count)

    mode = random.choice(FAIL_MODES)
    if mode == "line":
        return _line_cells()
    if mode == "gapped_rect":
        return _gapped_rect_cells()
    if mode == "bent":
        return _bent_cells()
    return _scatter_cells(count)


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
    glyphs: list[np.ndarray] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _block_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else BLOCK_AREA
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
    raise RuntimeError("failed to generate a valid d8e07eb2 input")
