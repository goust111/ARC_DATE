# Variant of d8e07eb2
# Dimension: D3 — 参考表结构 (dst)
# Change: 参考表为4行4列的5x5单元格网格，每个单元格包含一个唯一颜色+形状的3x3字形，单元格之间用背景色(8)分隔... → 参考表尺寸可变（如3x3或5x5）

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import get_bg, paste

# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The visual grammar stays the same: a 5-row query band, a full-width separator,
#      one blank spacer row, then a square table of 5x5 cells, then another separator
#      and a 2-row verdict band.
#    - The changed part is the table size. Instead of always showing 4 columns and 4
#      rows, an example may show 3x3, 4x4, or 5x5 cells. Because each cell is still
#      5x5, the whole canvas width/height must change with the chosen dimension.
#    - Example trace:
#      original logic: in a 4x4 table, four queried glyphs can succeed only if they
#      occupy one full row or one full column.
#      variant logic: in a 3x3 table, three queried glyphs are enough for success if
#      they cover one full row/column; in a 5x5 table, even four aligned glyphs are
#      still a failure because a complete line there has length five.
#      difference: the solver must infer the table size before deciding whether the
#      matched cells form a full line.
# b) Complexity calibration
#    - The new freedom is intentionally limited to three visible sizes: 3, 4, or 5.
#      This is enough to make the variation obvious without creating too many cases.
#    - Colors and shapes stay constrained exactly as before: every table cell holds a
#      centered 3x3 monochrome glyph, and each (color, shape) pair is unique.
#    - Failure samples still use a small set of recognizable patterns: short queries
#      (`dim-1` glyphs) or "almost a line but bent" queries of length `dim`.
# c) Hidden assumptions audited from the original solver
#    - The original generator hard-coded `GRID_H`, `GRID_W`, `TABLE_DIM`, `ROWS`,
#      `COLS`, and a fixed 4-cell success length. Those assumptions break as soon as a
#      5x5 table is allowed because 25 columns of cell area no longer fit in width 22.
#    - The original solver computed cell starts by stepping across the whole width and
#      height. That happens to work for the old fixed canvas, but here the code makes
#      the inferred row/column counts explicit from the table foreground bbox.
#    - The bbox inference relies on each glyph touching all four sides of its inner 3x3
#      footprint. The generator preserves that by sampling only masks with full 3x3
#      edge coverage.
# d) Placement legality
#    - Layout formula:
#      height = 5 (query) + 1 (separator) + 1 (blank) + dim*5 (table) + 1 (separator)
#               + 2 (verdict) = 10 + 5*dim
#      width  = 1 (left margin) + dim*5 (table/query slots) + 1 (right margin)
#               = 2 + 5*dim
#    - Table cells start at row 7 and column 1, with no overlap and no out-of-bounds
#      writes. Query glyphs are pasted only into the first `count` table-aligned slots.
# e) Visual observability
#    - The new rule is visible without any hidden ordering: the whole reference table
#      is physically smaller or larger, and the number of glyphs needed for a green
#      verdict changes with that visible size.
# f) Visual cleanliness
#    - Regions remain separated exactly as in the seed task. Green/red fills only touch
#      background pixels, so glyph ink is never overwritten.
# g) Color-change inferability
#    - No new color mapping is introduced. Green still means "matched / success" and
#      red still means "failure", exactly as in the original task.
# h) Variant verification target
#    - After implementation we should see generated samples across multiple dimensions,
#      not only 4x4. A practical metric is the distribution of inferred table sizes in
#      a batch of generated inputs.
# i) New solver assumptions and generator guarantees
#    - Solver assumption: all table cells contain one glyph and the glyph touches all
#      four sides of its inner 3x3 bbox.
#      Generator guarantee: every sampled mask passes `_has_full_bbox`, and every cell
#      is populated.
#    - Solver assumption: queried glyphs occupy table-aligned 5x5 slots in the top band.
#      Generator guarantee: each query glyph is pasted into one of those aligned slots.


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


def _infer_table_starts(
    g: np.ndarray, top_sep: int, bottom_sep: int, cell_size: int, bg: int
) -> tuple[list[int], list[int]]:
    table_region = g[top_sep + 1 : bottom_sep, :]
    table_fg = np.argwhere(table_region != bg)
    assert table_fg.size, "reference table must contain glyphs"

    min_r = int(table_fg[:, 0].min())
    max_r = int(table_fg[:, 0].max())
    min_c = int(table_fg[:, 1].min())
    max_c = int(table_fg[:, 1].max())

    table_h = max_r - min_r + 1
    table_w = max_c - min_c + 1
    assert (table_h + 2) % cell_size == 0 and (table_w + 2) % cell_size == 0

    row_count = (table_h + 2) // cell_size
    col_count = (table_w + 2) // cell_size
    row0 = top_sep + 1 + min_r - 1
    col0 = min_c - 1

    row_starts = [row0 + cell_size * i for i in range(row_count)]
    col_starts = [col0 + cell_size * i for i in range(col_count)]
    assert row_starts and col_starts, "could not infer table cell grid"
    assert row_starts[-1] + cell_size <= bottom_sep
    assert col_starts[-1] + cell_size <= g.shape[1]
    return row_starts, col_starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the query glyphs against a variable-sized reference table and judge line completion.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a query band made of table-aligned 5x5 glyph slots.
    - Between separators is a square reference table of 5x5 cells; the number of rows
      and columns can vary across examples, while each glyph is still identified by its
      color together with its centered 3x3 shape.

    Rule:
    - Green-fill the background inside every table cell whose glyph exactly matches one
      of the query glyphs.
    - If the matched cells cover one entire table row or one entire table column for
      that example's inferred table size, also green-fill the query band and the verdict band.
    - Otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: The verdict depends on the visible size of the reference table; a full
    line means "all columns" or "all rows" of the current table, not a fixed count of four.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5

    sep_rows = _constant_non_bg_rows(g, bg)
    assert len(sep_rows) >= 2, "expected two full-width separator rows"
    top_sep = sep_rows[0]
    bottom_sep = sep_rows[-1]

    row_starts, col_starts = _infer_table_starts(g, top_sep, bottom_sep, cell_size, bg)
    row_count = len(row_starts)
    col_count = len(col_starts)

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

    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    full_row = len(query_sigs) == col_count and len(selected) == col_count and same_row
    full_col = len(query_sigs) == row_count and len(selected) == row_count and same_col
    success = full_row or full_col

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator (o5x) ---------------

# -- Colors --
BG = 8
GREEN = 3
RED = 2
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)

# -- Shared geometry --
CELL = 5
QUERY_BAND_H = 5
LEFT_MARGIN = 1
TOP_SEP_ROW = QUERY_BAND_H
TABLE_GAP_ROW = 1
VERDICT_H = 2
TABLE_DIMS = (3, 4, 5)

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_SHORT_FAIL = 0.35
FAIL_MODES = ("bent_row", "bent_col", "scatter")

# -- Generation --
MAX_RETRIES = 400


def _has_full_bbox(mask: int) -> bool:
    """True if the 3x3 bitmask spans all four edges."""
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


def _geometry(dim: int) -> dict[str, int | tuple[int, ...]]:
    width = LEFT_MARGIN + dim * CELL + LEFT_MARGIN
    table_top = TOP_SEP_ROW + 1 + TABLE_GAP_ROW
    bottom_sep = table_top + dim * CELL
    height = bottom_sep + 1 + VERDICT_H
    row_starts = tuple(table_top + CELL * i for i in range(dim))
    col_starts = tuple(LEFT_MARGIN + CELL * i for i in range(dim))
    return {
        "height": height,
        "width": width,
        "top_sep": TOP_SEP_ROW,
        "bottom_sep": bottom_sep,
        "verdict_row": bottom_sep + 1,
        "row_starts": row_starts,
        "col_starts": col_starts,
    }


def _balanced_color_list(total_cells: int) -> list[int]:
    """Assign one color per cell while keeping all eight colors visible."""
    counts = {c: 1 for c in GLYPH_COLORS}
    extras = total_cells - len(GLYPH_COLORS)
    max_per_color = max(2, math.ceil(total_cells / len(GLYPH_COLORS)) + 1)
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < max_per_color]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    assert len(colors) == total_cells
    return colors


def _balanced_shape_masks(total_cells: int) -> list[int]:
    """Pick distinct 3x3 masks with coverage spread across fill counts."""
    counts = {k: 1 for k in range(FILL_MIN, FILL_MAX + 1)}
    extras = total_cells - len(counts)
    for _ in range(extras):
        choices = [k for k in range(FILL_MIN, FILL_MAX + 1) if counts[k] < len(PATTERNS_BY_FILL[k])]
        counts[random.choice(choices)] += 1

    masks: list[int] = []
    for k, n in counts.items():
        masks.extend(random.sample(PATTERNS_BY_FILL[k], n))
    random.shuffle(masks)
    assert len(masks) == total_cells and len(set(masks)) == total_cells
    return masks


def _glyph(mask: int, color: int) -> np.ndarray:
    """Render a 3x3 bitmask as a CELL x CELL array with a 1-pixel BG border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _line_cells(dim: int) -> list[tuple[int, int]]:
    """Pick a full row or column of the current table."""
    if random.random() < 0.5:
        r = random.randrange(dim)
        cells = [(r, c) for c in range(dim)]
    else:
        c = random.randrange(dim)
        cells = [(r, c) for r in range(dim)]
    random.shuffle(cells)
    return cells


def _scatter_fail_cells(dim: int, count: int) -> list[tuple[int, int]]:
    """Pick distinct non-line cells that span multiple rows and columns."""
    all_cells = [(r, c) for r in range(dim) for c in range(dim)]
    for _ in range(200):
        cells = random.sample(all_cells, count)
        if len({r for r, _c in cells}) == 1:
            continue
        if len({c for _r, c in cells}) == 1:
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample scattered fail cells")


def _fail_cells(dim: int, count: int) -> list[tuple[int, int]]:
    """Pick fail cells for the current dimension."""
    if count < dim:
        return _scatter_fail_cells(dim, count)

    mode = random.choice(FAIL_MODES)
    if mode == "bent_row":
        r = random.randrange(dim)
        dropped_c = random.randrange(dim)
        new_r = random.choice([rr for rr in range(dim) if rr != r])
        cells = [(r, c) for c in range(dim) if c != dropped_c] + [(new_r, dropped_c)]
    elif mode == "bent_col":
        c = random.randrange(dim)
        dropped_r = random.randrange(dim)
        new_c = random.choice([cc for cc in range(dim) if cc != c])
        cells = [(r, c) for r in range(dim) if r != dropped_r] + [(dropped_r, new_c)]
    else:
        cells = _scatter_fail_cells(dim, count)

    assert len(cells) == count and len(set(cells)) == count
    assert len({r for r, _c in cells}) > 1 and len({c for _r, c in cells}) > 1
    random.shuffle(cells)
    return cells


def _expected_output(
    grid: np.ndarray,
    row_starts: tuple[int, ...],
    col_starts: tuple[int, ...],
    top_sep: int,
    bottom_sep: int,
    cells: list[tuple[int, int]],
    success: bool,
) -> np.ndarray:
    """Build the expected solver output for oracle checking."""
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[row_starts[r_i] : row_starts[r_i] + CELL, col_starts[c_i] : col_starts[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == BG] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try to generate one valid input grid; return None on oracle mismatch."""
    dim = random.choice(TABLE_DIMS)
    geom = _geometry(dim)
    row_starts = geom["row_starts"]
    col_starts = geom["col_starts"]
    assert isinstance(row_starts, tuple) and isinstance(col_starts, tuple)

    grid = np.full((geom["height"], geom["width"]), BG, dtype=int)
    grid[geom["top_sep"], :] = SEP
    grid[geom["bottom_sep"], :] = SEP

    total_cells = dim * dim
    masks = _balanced_shape_masks(total_cells)
    colors = _balanced_color_list(total_cells)
    glyphs: list[np.ndarray] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        r_i, c_i = divmod(i, dim)
        paste(grid, glyph, row_starts[r_i], col_starts[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells(dim)
    else:
        count = dim - 1 if random.random() < P_SHORT_FAIL else dim
        query_cells = _fail_cells(dim, count)

    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * dim + c_i], 0, col_starts[slot])

    expected = _expected_output(
        grid,
        row_starts,
        col_starts,
        int(geom["top_sep"]),
        int(geom["bottom_sep"]),
        query_cells,
        success,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 input")
