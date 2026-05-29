# Combo variant of d8e07eb2 (index 20)
# combo_spec: D10/2+D12/1
# Name: 行统一颜色+两轮匹配

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        sys.path.append(str(_arc_utils.parent))
        break

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
#    The outer scaffold stays the original 30x22 layout: top query band, two
#    full-width separator rows, a 4x4 table of 5x5 cells, and a 2-row verdict
#    band. The fused change is inside the table logic:
#    - Every table row uses one shared glyph color.
#    - Every table column uses one shared 3x3 shape.
#    - Queries are exact table glyph copies, so each query is read as
#      (row-color, column-shape).
#    Example: if row 2 is blue and column 1 is a hollow-L shape, then any blue
#    query with that shape resolves to table cell (2, 1). Round 1 highlights the
#    whole blue row yellow; round 2 upgrades only the exact blue hollow-L cell
#    to green. This differs from the original exact-signature lookup because the
#    intermediate row-level candidate set becomes visible in the output.
#
# b) Complexity calibration
#    The generator uses exactly 4 row colors and 4 column shapes. That keeps the
#    factorization small enough to infer while still creating repetition:
#    every query color repeats 4 times in the table, and every query shape
#    repeats 4 times. Output-only colors 3 and 4 are reserved so green/yellow
#    remain unambiguous.
#
# c) Implicit-assumption audit
#    The original solver assumed one-step exact matching by full cell signature.
#    That breaks here because color is no longer globally identifying. The new
#    solver instead assumes:
#    - each populated cell is monochrome;
#    - each table row has a single color, distinct across rows;
#    - each table column has a single shape, distinct across columns;
#    - query cells resolve uniquely by the pair (row color, column shape).
#    The generator enforces all four assumptions directly.
#
# d) Placement legality
#    Cell geometry is unchanged: 5x5 slots at fixed coordinates with centered
#    3x3 glyphs. Query glyphs are pasted only into the four top slots, so there
#    is no overlap or out-of-bounds risk.
#
# e) Visual observability
#    The fused rule is visible in a few examples because the table shows four
#    horizontal color bands crossed by four vertical shape families, and the
#    output keeps both stages: whole candidate rows turn yellow first, then only
#    the query-specific cells turn green.
#
# f) Visual cleanliness
#    Yellow and green appear only as background fills inside table cells and in
#    the usual success/verdict regions. No extra marker dots are added, so the
#    grid remains readable and the two rounds do not obscure glyph silhouettes.
#
# g) Change inferability
#    A degenerate generator could still pass if it produced row-success cases
#    whose four green cells overwrite an entire yellow row. To keep the two-step
#    behavior observable, successful samples are generated from full columns.
#    The solver still supports full-row success for valid external inputs, but
#    the generator biases toward column success so yellow-only residue is always
#    present in training-style samples.


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _cell_shape_signature(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[tuple[int, int], ...] | None:
    inner = g[int(r0) + 1 : int(r0) + int(size) - 1, int(c0) + 1 : int(c0) + int(size) - 1]
    mask = inner != int(bg)
    if not bool(mask.any()):
        return None
    return tuple((int(r), int(c)) for r, c in zip(*np.where(mask), strict=False))


def _cell_color(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> int | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    vals = sorted({int(v) for v in cell[cell != int(bg)]})
    if not vals:
        return None
    assert len(vals) == 1, "glyph cells must be monochrome"
    return vals[0]


def _cell_features(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    shape = _cell_shape_signature(g, r0, c0, size, bg)
    if shape is None:
        return None
    color = _cell_color(g, r0, c0, size, bg)
    assert color is not None
    return int(color), shape


def _infer_cell_grid(g: np.ndarray, bg: int, cell_size: int) -> tuple[int, int, list[int], list[int]]:
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

    row_starts = [r for r in range(row0, bottom_sep, cell_size) if r + cell_size <= bottom_sep]
    col_starts = [c for c in range(col0, g.shape[1], cell_size) if c + cell_size <= g.shape[1]]
    assert row_starts and col_starts, "could not infer table cell grid"
    return top_sep, bottom_sep, row_starts, col_starts


def _fill_cell_bg(out: np.ndarray, r0: int, c0: int, size: int, bg: int, color: int) -> None:
    cell = out[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    cell[cell == int(bg)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each query in two visible stages: row by color first, then column by shape.

    Input structure:
    - Teal background with two full-width separator rows.
    - Above the first separator is a query band containing up to four 5x5 glyph cells.
    - Between separators is a 4x4 reference table of the same cell size; each
      table row shares one glyph color and each table column shares one centered
      3x3 shape.

    Rule:
    - For every query color, mark the entire matching table row yellow as the
      first-pass candidate set.
    - Then, for every query glyph, use its color to pick the row and its 3x3
      shape to pick the column, and upgrade that exact table cell to green.
    - If four resolved green cells form one complete table row or one complete
      table column, also green-fill the query band and verdict band; otherwise
      leave the query band unchanged and fill the verdict band red.

    Key insight:
    The answer is not found by global glyph identity. Color only narrows the
    search to a row, shape finishes the lookup inside that row, and the yellow
    residue shows the intermediate candidate rows that were considered.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5

    top_sep, bottom_sep, row_starts, col_starts = _infer_cell_grid(g, bg, cell_size)

    row_colors: list[int] = []
    col_shapes: list[tuple[tuple[int, int], ...]] = []
    for ri, r in enumerate(row_starts):
        row_features = [_cell_features(g, r, c, cell_size, bg) for c in col_starts]
        assert all(feat is not None for feat in row_features), "table cells must all be populated"
        row_features = [feat for feat in row_features if feat is not None]
        colors = [color for color, _shape in row_features]
        shapes = [shape for _color, shape in row_features]
        assert len(set(colors)) == 1, "each table row must use one shared color"
        row_colors.append(colors[0])
        if ri == 0:
            col_shapes = shapes
        else:
            assert shapes == col_shapes, "each table column must repeat one shared shape"

    assert len(set(row_colors)) == len(row_starts), "row colors must distinguish rows"
    assert len(set(col_shapes)) == len(col_starts), "column shapes must distinguish columns"
    color_to_row = {color: ri for ri, color in enumerate(row_colors)}
    shape_to_col = {shape: ci for ci, shape in enumerate(col_shapes)}

    query_features = [
        feat
        for c in col_starts
        if (feat := _cell_features(g, 0, c, cell_size, bg)) is not None
    ]
    assert len(query_features) == len(set(query_features)), "query glyphs should be distinct"

    out = g.copy()
    query_rows: set[int] = set()
    for color, shape in query_features:
        assert color in color_to_row, "query color must match a table row"
        assert shape in shape_to_col, "query shape must match a table column"
        query_rows.add(color_to_row[color])
    for ri in query_rows:
        for c in col_starts:
            _fill_cell_bg(out, row_starts[ri], c, cell_size, bg, YELLOW)

    selected: list[tuple[int, int]] = []
    for color, shape in query_features:
        ri = color_to_row[color]
        ci = shape_to_col[shape]
        selected.append((int(ri), int(ci)))
        _fill_cell_bg(out, row_starts[ri], col_starts[ci], cell_size, bg, GREEN)

    assert len(selected) == len(set(selected)), "query glyphs should resolve to distinct cells"
    complete_query = len(query_features) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_features) and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED

    return out


# --------------- generator (combo variant) ---------------

# -- Grid layout --
GRID_H, GRID_W = 30, 22
TOP_SEP_ROW = 5
BOTTOM_SEP_ROW = 27
QUERY_BAND_H = 5
VERDICT_ROW = 28

# -- Colors --
BG = 8
RED = 2
GREEN = 3
YELLOW = 4
SEP = 6
GLYPH_COLORS = (0, 1, 2, 5, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
COLUMN_FILL_COUNTS = (5, 6, 7, 8)

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35

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


def _row_colors() -> list[int]:
    return random.sample(GLYPH_COLORS, TABLE_DIM)


def _column_masks() -> list[int]:
    fills = list(COLUMN_FILL_COUNTS)
    random.shuffle(fills)
    return [random.choice(PATTERNS_BY_FILL[k]) for k in fills]


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _is_full_line(cells: list[tuple[int, int]]) -> bool:
    return len(cells) == TABLE_DIM and (
        len({r for r, _c in cells}) == 1 or len({c for _r, c in cells}) == 1
    )


def _success_cells() -> list[tuple[int, int]]:
    # Column successes always leave yellow-only residue in every selected row.
    c = random.randrange(TABLE_DIM)
    cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    pool = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM)]
    for _ in range(500):
        cells = random.sample(pool, count)
        if len({r for r, _c in cells}) == 1:
            continue
        if len({c for _r, c in cells}) == 1:
            continue
        if _is_full_line(cells):
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample a valid fail set")


def _expected_output(
    grid: np.ndarray,
    query_cells: list[tuple[int, int]],
    success: bool,
) -> np.ndarray:
    out = grid.copy()
    query_rows = {r for r, _c in query_cells}
    for r_i in query_rows:
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            cell[cell == BG] = YELLOW
    for r_i, c_i in query_cells:
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

    row_colors = _row_colors()
    col_masks = _column_masks()
    glyphs: dict[tuple[int, int], np.ndarray] = {}
    for r_i, row_color in enumerate(row_colors):
        for c_i, mask in enumerate(col_masks):
            glyph = _glyph(mask, row_color)
            glyphs[(r_i, c_i)] = glyph
            paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    query_cells = _success_cells() if success else _fail_cells(3 if random.random() < P_QUERY_3 else 4)
    for slot, cell in enumerate(query_cells):
        paste(grid, glyphs[cell], 0, COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    if not np.any(expected == YELLOW):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
