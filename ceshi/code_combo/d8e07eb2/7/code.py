# Combo variant of d8e07eb2 (index 7)
# combo_spec: D3/0+D5/2
# Name: 可变表尺寸+对角线判定

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import get_bg, paste

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The scene keeps the seed task's three-band layout: query band, separator,
#      square reference table, separator, verdict band. Each visible glyph is still
#      a 3x3 monochrome pattern centered inside a 5x5 cell.
#    - The fused changes are:
#        1. the table size is no longer fixed; examples can show 3x3, 4x4, or 5x5;
#        2. success is no longer "one full row/column", but "one full main or anti
#           diagonal" of that inferred table size.
#    - Trace example:
#        original task on a 4x4 table: four queried cells forming row 2 succeed.
#        combo task on a 4x4 table: the same four matches fail, because a row is not
#        a diagonal. On a 3x3 table, three queries can succeed only if they are
#        {(0,0),(1,1),(2,2)} or {(0,2),(1,1),(2,0)}. On a 5x5 table, even four
#        diagonal-looking matches still fail because the visible table requires five.
#    - Difference from the seed: the solver must first infer the table dimension from
#      the grid geometry, then compare the matched coordinates against the two full
#      diagonals of exactly that dimension.
# b) Complexity calibration
#    - The size freedom is intentionally limited to three values: 3, 4, 5. That is
#      enough to make examples visibly different while keeping the task learnable.
#    - Query length is limited to either `dim` or `dim - 1`. This preserves a small,
#      interpretable set of cases: complete diagonal success, incomplete diagonal
#      failure, and complete-but-wrong-shape failure.
#    - Fail samples deliberately include full rows and columns of length `dim`, so the
#      fused task does not collapse into the old row/column rule.
# c) Hidden assumptions audited from the original solver
#    - The original code assumed a fixed 4x4 table, fixed canvas size, fixed row/col
#      starts, and a fixed success length of four. Those assumptions all break once
#      3x3 and 5x5 tables are allowed.
#    - The original row/column verdict logic (`same_row` / `same_col`) is invalid
#      under diagonal judging and must be replaced by exact set membership in the two
#      diagonal coordinate sets.
#    - The bbox-based table inference assumes each cell contains a glyph whose 3x3 ink
#      touches all four sides of the inner footprint. The generator preserves that by
#      sampling only masks that span top, bottom, left, and right edges.
# d) Placement legality
#    - Geometry is derived from `dim`:
#        height = 5 + 1 + 1 + 5*dim + 1 + 2 = 10 + 5*dim
#        width  = 1 + 5*dim + 1 = 2 + 5*dim
#    - With `dim in {3,4,5}`, all pasted 5x5 cells stay disjoint and inside bounds.
#      Query glyphs are pasted only into the first `count` query slots aligned with the
#      table columns, so there is no overlap.
# e) Visual observability
#    - A human can see both fused controls directly: the table is physically 3x3, 4x4,
#      or 5x5, and the green verdict appears only when highlighted matches trace one of
#      the two long diagonals of that exact square.
#    - Because failures include complete rows/columns that still produce red verdicts,
#      the samples actively reject the original hypothesis instead of silently matching it.
# f) Visual cleanliness
#    - Region separation is unchanged from the seed task. Green highlighting only fills
#      background pixels inside matched cells and the verdict/query backgrounds, so the
#      glyph ink remains legible and no competing overlay system is introduced.
# g) Change inferability
#    - No hidden numeric ordering or color arithmetic is used. Colors still identify
#      glyphs only through visible equality, and green/red keep their old meanings as
#      success/failure markers.
# h) Fusion verification target
#    - After implementation we should see a mixed distribution of inferred table sizes
#      and a non-trivial count of "row/column fail" examples where the combo output is
#      red even though the old task would have produced green.
# i) New solver assumptions and generator guarantees
#    - Solver assumption: the reference table is square and fully populated with one
#      glyph per 5x5 cell.
#      Generator guarantee: it always builds a square `dim x dim` table and pastes a
#      glyph into every cell.
#    - Solver assumption: query glyphs are distinct exact copies of table glyphs placed
#      in table-aligned slots in the top band.
#      Generator guarantee: query glyphs are sampled from distinct table cells and
#      pasted into those aligned slots only.
#    - Solver assumption: successes require all diagonal positions of the current size.
#      Generator guarantee: success samples use exactly one full main or anti diagonal.


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


def _diagonal_cell_sets(
    dim: int,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    main_diag = frozenset((i, i) for i in range(dim))
    anti_diag = frozenset((i, dim - 1 - i) for i in range(dim))
    return main_diag, anti_diag


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyphs against a variable-sized table and judge diagonal completion.

    Input structure:
    - Teal background with two full-width separator rows.
    - Above the first separator is a query band made of table-aligned 5x5 slots that
      may hold a copied glyph or stay empty.
    - Between separators is a square reference table of 5x5 cells. The table can be
      3x3, 4x4, or 5x5, and each cell contains a unique glyph identified by both its
      color and its centered 3x3 shape.

    Rule:
    - Green-fill the background inside every table cell whose glyph exactly appears in
      the query band.
    - If those matched cells are exactly one complete main diagonal or one complete
      anti-diagonal of the inferred table size, also green-fill the query band and the
      verdict band.
    - Otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: The verdict depends on two visible facts at once: how large the table
    is in this example, and whether the matched positions cover all cells of one full
    diagonal of that size rather than merely forming some other alignment.
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

    square_table = row_count == col_count
    complete_query = len(query_sigs) == col_count
    if square_table:
        is_complete_diagonal = frozenset(selected) in _diagonal_cell_sets(row_count)
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
P_SUCCESS = 0.45
P_SHORT_FAIL = 0.35
FAIL_MODES = ("row", "col", "off_main", "off_anti", "perm", "scatter")

# -- Generation --
MAX_RETRIES = 500


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


def _diagonal_cells(dim: int) -> list[tuple[int, int]]:
    cells = list(random.choice(_diagonal_cell_sets(dim)))
    random.shuffle(cells)
    return cells


def _is_full_diagonal(dim: int, cells: list[tuple[int, int]]) -> bool:
    return len(cells) == dim and frozenset(cells) in _diagonal_cell_sets(dim)


def _scatter_fail_cells(dim: int, count: int) -> list[tuple[int, int]]:
    """Pick distinct cells that visibly span multiple rows/cols and are not a full diagonal."""
    all_cells = [(r, c) for r in range(dim) for c in range(dim)]
    for _ in range(300):
        cells = random.sample(all_cells, count)
        if len({r for r, _c in cells}) == 1:
            continue
        if len({c for _r, c in cells}) == 1:
            continue
        if _is_full_diagonal(dim, cells):
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample scattered fail cells")


def _perm_fail_cells(dim: int) -> list[tuple[int, int]]:
    """Pick one cell per row using a non-diagonal column permutation."""
    cols = list(range(dim))
    for _ in range(300):
        perm = random.sample(cols, dim)
        if perm == cols or perm == list(reversed(cols)):
            continue
        cells = [(r, perm[r]) for r in range(dim)]
        if not _is_full_diagonal(dim, cells):
            random.shuffle(cells)
            return cells
    raise RuntimeError("failed to sample permutation fail cells")


def _near_diagonal_fail(dim: int, anti: bool) -> list[tuple[int, int]]:
    """Take one full diagonal and replace one position with a non-diagonal cell."""
    diag = set((i, dim - 1 - i) if anti else (i, i) for i in range(dim))
    removed = random.choice(tuple(diag))
    pool = [
        (r, c)
        for r in range(dim)
        for c in range(dim)
        if (r, c) not in diag
    ]
    cells = list((diag - {removed}) | {random.choice(pool)})
    assert len(cells) == dim and not _is_full_diagonal(dim, cells)
    random.shuffle(cells)
    return cells


def _fail_cells(dim: int, count: int) -> list[tuple[int, int]]:
    """Pick distinct query source cells that fail the diagonal rule."""
    if count < dim:
        if random.random() < 0.5:
            cells = random.sample(list(random.choice(_diagonal_cell_sets(dim))), count)
            random.shuffle(cells)
            return cells
        return _scatter_fail_cells(dim, count)

    mode = random.choice(FAIL_MODES)
    if mode == "row":
        r = random.randrange(dim)
        cells = [(r, c) for c in range(dim)]
    elif mode == "col":
        c = random.randrange(dim)
        cells = [(r, c) for r in range(dim)]
    elif mode == "off_main":
        return _near_diagonal_fail(dim, anti=False)
    elif mode == "off_anti":
        return _near_diagonal_fail(dim, anti=True)
    elif mode == "perm":
        return _perm_fail_cells(dim)
    else:
        return _scatter_fail_cells(dim, count)

    assert len(cells) == count and not _is_full_diagonal(dim, cells)
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
    dim = random.choice(TABLE_DIMS)
    geom = _geometry(dim)
    row_starts = geom["row_starts"]
    col_starts = geom["col_starts"]
    assert isinstance(row_starts, tuple) and isinstance(col_starts, tuple)

    grid = np.full((geom["height"], geom["width"]), BG, dtype=int)
    grid[int(geom["top_sep"]), :] = SEP
    grid[int(geom["bottom_sep"]), :] = SEP

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
        query_cells = _diagonal_cells(dim)
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
    raise RuntimeError("failed to generate a valid d8e07eb2 combo variant input")
