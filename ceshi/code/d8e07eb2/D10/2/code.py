# Variant of d8e07eb2
# Dimension: D10 — 颜色在参考表中的分布规则 (meta)
# Change: 参考表中每种颜色出现次数不固定（如颜色2可能出现多次但形状不同），颜色不唯一标识字形... → 每行使用统一颜色

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D10 variant
# ----------------------------------
# a) Concretization and behavior trace
#    The canvas geometry stays identical to the original task: a 5-row query
#    band, a separator, a 4x4 table of 5x5 cells, another separator, and a
#    2-row verdict band. The changed part is the internal organization of the
#    reference table:
#    - Every table row uses a single glyph color across its 4 cells.
#    - Every table column uses a single 3x3 shape across its 4 cells.
#    Therefore a cell is identified compositionally as (row color, column
#    shape), not by a globally unique color or a globally unique shape.
#    Example: if row 1 is orange and column 2 is shape C, then the cell at
#    (1,2) is "orange+C". The matching query cell shows the same centered 3x3
#    shape in orange, plus a same-color corner dot used only as a query marker.
#    Variant logic: ignore that corner dot, read color->row and shape->column,
#    then resolve the target cell. This small query-side adaptation is needed so
#    the task truly requires factorized decoding instead of the original exact
#    full-cell signature lookup.
#
# b) Complexity calibration
#    The table uses exactly 4 distinct row colors and exactly 4 distinct column
#    shapes. Each row color is repeated 4 times, and each column shape is
#    repeated 4 times. This gives enough repetition for the new rule to be
#    visible, while keeping the number of latent categories small enough for a
#    human to infer from a few examples.
#
# c) Audit of original assumptions
#    The original solver relied on full-cell signatures, then used set
#    membership against the query glyphs. That no longer works here because
#    query cells contain an extra same-color corner marker that table cells do
#    not have. The new solver must treat the centered 3x3 motif as the shape
#    channel and the cell color as the row channel. Concretely it builds:
#    - a map from row color to table row index
#    - a map from shape mask to table column index
#    It also adds new explicit assertions that the generator guarantees:
#    row colors are uniform within a row and distinct across rows; shape masks
#    are uniform within a column and distinct across columns.
#
# d) Placement legality
#    Object placement stays on the original 30x22 fixed scaffold. Table cells
#    are still the same 5x5 slots with a centered 3x3 glyph, so there is no
#    overlap risk. Query glyphs are still pasted into the four top slots, with
#    the fourth slot optionally empty for 3-query failures.
#
# e) Visual observability
#    The variant is visible without reading color ids: the table looks like four
#    horizontal color stripes crossed with four vertical shape families, while
#    query cells visibly carry a tiny same-color corner tag. A successful row
#    query shows four same-colored glyphs; a successful column query shows four
#    identical silhouettes in different colors.
#
# f) Visual cleanliness
#    No new markers are added. The only highlighting remains the original green
#    background fill inside matched cells plus the verdict band, so the table
#    structure remains readable.
#
# g) Color-change inferability
#    Only 4 glyph colors participate in each instance, and each one appears in
#    an entire row. This repetition makes the role of color as a row label easy
#    to infer. Shapes likewise repeat down columns, making the shape->column
#    mapping discoverable from a few samples.


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _cell_shape_signature(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[tuple[int, int], ...] | None:
    cell = g[int(r0) + 1 : int(r0) + int(size) - 1, int(c0) + 1 : int(c0) + int(size) - 1]
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    return tuple((int(r), int(c)) for r, c in zip(*np.where(mask), strict=False))


def _cell_color(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> int | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    vals = sorted({int(v) for v in cell[cell != int(bg)]})
    if not vals:
        return None
    assert len(vals) == 1, "glyph cells must remain monochrome"
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each query glyph by row-color and column-shape, then test line completion.

    Input structure:
    - Teal background with two full-width separator rows.
    - Above the first separator is a query band of 5x5 glyph slots; each
      non-empty query cell has a same-color corner marker in addition to the
      centered 3x3 glyph.
    - Between separators is a 4x4 reference table of the same 5x5 cells; each
      table row shares one color while each table column shares one 3x3 shape.

    Rule:
    - For every query glyph, ignore the corner marker, use its color to choose
      a table row and its centered 3x3 shape to choose a table column, then
      green-fill the background of that table cell.
    - If four queries are present and the resolved cells make one full table row
      or one full table column, also green-fill the query band and verdict band.
    - Otherwise leave the query band unchanged and fill the verdict band red.

    Key insight:
    A query glyph is interpreted compositionally: color is the row index and
    shape is the column index, so the answer depends on the row/column pattern
    formed by those decoded positions rather than on a global glyph catalogue.
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
    selected: list[tuple[int, int]] = []
    for color, shape in query_features:
        assert color in color_to_row, "query color must match a table row"
        assert shape in shape_to_col, "query shape must match a table column"
        ri = color_to_row[color]
        ci = shape_to_col[shape]
        selected.append((int(ri), int(ci)))
        r = row_starts[ri]
        c = col_starts[ci]
        cell = out[r : r + cell_size, c : c + cell_size]
        cell[cell == bg] = 3

    assert len(selected) == len(set(selected)), "query glyphs should resolve to distinct cells"
    complete_query = len(query_features) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator (D10 variant) ---------------

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
COLUMN_FILL_COUNTS = (5, 6, 7, 8)

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

# -- Generation --
MAX_RETRIES = 300


def _has_full_bbox(mask: int) -> bool:
    """True iff the 3x3 bitmask touches top/bottom/left/right edges."""
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
    """Pick 4 distinct colors, one per row, so color alone identifies the row."""
    return random.sample(GLYPH_COLORS, TABLE_DIM)


def _column_masks() -> list[int]:
    """Pick 4 distinct shapes, one per column, with varied fill counts for visibility."""
    fills = list(COLUMN_FILL_COUNTS)
    random.shuffle(fills)
    return [random.choice(PATTERNS_BY_FILL[k]) for k in fills]


def _glyph(mask: int, color: int) -> np.ndarray:
    """Render a 3x3 bitmask as a CELL x CELL array with a 1-pixel BG border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _query_glyph(mask: int, color: int) -> np.ndarray:
    """Render a query glyph with the same centered shape plus a same-color corner marker."""
    cell = _glyph(mask, color)
    cell[0, 0] = color
    return cell


def _line_cells() -> list[tuple[int, int]]:
    """Pick a full row or full column of 4 cells, returned in shuffled query order."""
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    """Pick distinct cells that do not form one complete row or one complete column."""
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
    """Build the expected output used by the output-first generator."""
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
    """Try to generate one valid D10-variant input; return None on oracle mismatch."""
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    row_colors = _row_colors()
    col_masks = _column_masks()
    glyphs: dict[tuple[int, int], np.ndarray] = {}
    query_glyphs: dict[tuple[int, int], np.ndarray] = {}
    for r_i, row_color in enumerate(row_colors):
        for c_i, mask in enumerate(col_masks):
            glyph = _glyph(mask, row_color)
            glyphs[(r_i, c_i)] = glyph
            query_glyphs[(r_i, c_i)] = _query_glyph(mask, row_color)
            paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    for slot, cell in enumerate(query_cells):
        paste(grid, query_glyphs[cell], 0, COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 D10 input")
