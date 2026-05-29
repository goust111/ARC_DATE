# Combo variant of d8e07eb2 (index 33)
# combo_spec: D5/2+D11/0
# Name: 对角线判定+裁剪输出

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant (D5/2 + D11/0)
#
# a) Concrete behavior trace
#    - Input stays in the original 30x22 format: a 5-row query band, a pink
#      separator, a 4x4 table of 5x5 cells, another separator, and a 2-row
#      verdict area that is now ignored by the output.
#    - Matching logic is still exact color+shape lookup. Each query glyph is a
#      copy of one distinct table glyph, and each matched table cell gets its
#      teal background filled green.
#    - Fusion change 1: success is no longer "one full row/column"; it is now
#      "exactly one full main diagonal or anti-diagonal" with all 4 query slots
#      occupied.
#    - Fusion change 2: instead of returning the full 30x22 canvas, the solver
#      returns only the 20x20 reference table crop, plus one extra row and one
#      extra column that host a tiny L-shaped verdict badge.
#    - Example trace:
#        original rule success: {(2,0),(2,1),(2,2),(2,3)} -> whole row
#        combo result there   : same four cells are highlighted in the table
#                               crop, but badge is red because a row is not a
#                               diagonal.
#        combo success case   : {(0,0),(1,1),(2,2),(3,3)} -> table crop shows
#                               those 4 cells highlighted and the badge is green.
#      The crucial visible difference is that the output keeps only the table,
#      so the diagonal pattern must be read directly inside the cropped payload.
#
# b) Complexity calibration
#    - The table remains 4x4, so the diagonal criterion has only two possible
#      success shapes. This is structured enough for humans to infer, but still
#      different from the original row/column rule.
#    - Fail generation intentionally includes complete rows and complete columns.
#      Those would have been green under the source task but must be red here,
#      which makes the D5 change observable instead of silently degenerating.
#    - Output cropping is also unmistakable: every output is 21x21 rather than
#      30x22, with the highlighted cells preserved and a fixed-shape badge in
#      the lower-right corner.
#
# c) Implicit-assumption audit
#    - `_constant_non_bg_rows` still assumes the two separator rows are constant,
#      full-width, and non-background. The generator preserves that.
#    - Table origin inference still relies on every 5x5 cell having a 1-pixel
#      teal border around the 3x3 glyph. The generator preserves that too.
#    - Query extraction from row 0 across the inferred column starts still works
#      because the query band layout is unchanged.
#    - The old `same_row` / `same_col` success check is invalid in this combo
#      and must be replaced by exact set membership in the two diagonal sets.
#    - The crop logic introduces a new assumption: the reference table is one
#      contiguous `len(row_starts)*5` by `len(col_starts)*5` block. This is
#      guaranteed by the fixed 4x4-by-5x5 table geometry.
#    - The badge logic assumes there is an extra output-only row and column, so
#      the verdict marker never overwrites table pixels. The solver constructs
#      that fresh canvas directly; the generator mirrors it in `_expected_output`.
#
# d) Placement legality
#    - Input placement is unchanged from the source task, so glyphs stay inside
#      the existing 5x5 slots and never overlap.
#    - The generator samples distinct table cells for queries, preserving the
#      one-to-one lookup assumption used by the solver.
#    - Output placement is deterministic: the table crop occupies the top-left
#      20x20 region and the badge only touches the appended last row/column.
#
# e) Visual observability
#    - A solver looking only at examples can recover the rule because:
#        1. output size always collapses to the table crop plus a tiny badge;
#        2. success examples show 4 highlighted cells on one diagonal;
#        3. some fail examples show 4 highlighted cells on a row/column with a
#           red badge, disproving the original task hypothesis.
#    - No hidden ordering by numeric color ids or any non-visual quantity is
#      introduced. Only visible shape, color, position, and crop size matter.
#
# f) Visual cleanliness
#    - The crop removes the noisiest unused regions (query band, separators, and
#      verdict band), so the transformed output is cleaner than the original.
#    - The badge uses a fixed 5-pixel L-shape in the bottom-right corner, which
#      stays visually separate from the 20x20 table content.
#
# g) Change inferability
#    - Colors keep their original semantics: green = matched-cell highlight and
#      success badge; red = failure badge.
#    - Because the badge shape is fixed and only its color changes, the learner
#      can infer that the color encodes a binary verdict, while the highlighted
#      cell arrangement in the cropped table reveals the diagonal condition.
# ---------------------------------------------------------------------------


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


def _infer_table_geometry(
    g: np.ndarray, bg: int, cell_size: int
) -> tuple[int, int, list[int], list[int]]:
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
    col_starts = list(range(col0, g.shape[1], cell_size))
    col_starts = [c for c in col_starts if c + cell_size <= g.shape[1]]
    assert row_starts and col_starts, "could not infer table cell grid"
    return top_sep, bottom_sep, row_starts, col_starts


def _matched_table_and_success(
    g: np.ndarray, bg: int, cell_size: int
) -> tuple[np.ndarray, bool, list[int], list[int], list[tuple[int, int]]]:
    _top_sep, _bottom_sep, row_starts, col_starts = _infer_table_geometry(g, bg, cell_size)

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    marked = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            if _cell_signature(g, r, c, cell_size, bg) not in query_set:
                continue
            selected.append((int(ri), int(ci)))
            cell = marked[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = GREEN

    complete_query = len(query_sigs) == len(col_starts)
    square_table = len(row_starts) == len(col_starts)
    is_complete_diagonal = square_table and frozenset(selected) in _diagonal_cell_sets(len(row_starts))
    success = complete_query and len(selected) == len(query_sigs) and is_complete_diagonal
    return marked, bool(success), row_starts, col_starts, selected


def _verdict_badge(table_h: int, table_w: int, color: int, bg: int) -> np.ndarray:
    out = np.full((int(table_h) + 1, int(table_w) + 1), int(bg), dtype=int)
    pts = (
        (table_h - 2, table_w),
        (table_h - 1, table_w),
        (table_h, table_w - 2),
        (table_h, table_w - 1),
        (table_h, table_w),
    )
    for r, c in pts:
        out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the query glyphs against the reference table, then emit only the cropped table result.

    Input structure:
    - The input canvas keeps the original 30x22 layout: a top query band, two
      full-width separator rows, and a 4x4 table of 5x5 glyph cells in between.
    - Each glyph is identified jointly by its 3x3 shape and its color, with a
      teal background border around the glyph inside each 5x5 cell.

    Rule:
    - Find every table cell whose glyph exactly appears in the query band and
      green-fill the background pixels inside those matched cells.
    - Treat the verdict as success only when all 4 query slots are occupied and
      the matched cells are exactly one full main diagonal or one full anti-diagonal.
    - Return only the cropped table area, plus a small bottom-right L-shaped
      badge: green for success and red for failure.

    Key insight: The matching mechanism is unchanged, but the solver must fuse a
    new geometric test (full diagonal only) with a new output framing that keeps
    only the reference table and a compact verdict marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5

    marked, success, row_starts, col_starts, _selected = _matched_table_and_success(g, bg, cell_size)
    table_r0 = row_starts[0]
    table_c0 = col_starts[0]
    table_h = len(row_starts) * cell_size
    table_w = len(col_starts) * cell_size
    cropped = marked[table_r0 : table_r0 + table_h, table_c0 : table_c0 + table_w].copy()

    out = _verdict_badge(table_h, table_w, GREEN if success else RED, bg)
    out[:table_h, :table_w] = cropped
    return out


# --------------- generator (combo variant) ---------------

# -- Grid layout --
GRID_H, GRID_W = 30, 22
TOP_SEP_ROW = 5
BOTTOM_SEP_ROW = 27

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
TABLE_TOP = ROWS[0]
TABLE_LEFT = COLS[0]
TABLE_H = TABLE_DIM * CELL
TABLE_W = TABLE_DIM * CELL

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.5
P_QUERY_3 = 0.3
FAIL_MODES = ("row", "row", "col", "col", "off_main", "off_anti", "rect", "perm")

# -- Generation --
MAX_RETRIES = 300


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

NON_DIAGONAL_PERMS = tuple(
    perm
    for perm in itertools.permutations(range(TABLE_DIM))
    if perm not in ((0, 1, 2, 3), (3, 2, 1, 0))
)


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


def _main_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, i) for i in range(TABLE_DIM)]


def _anti_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, TABLE_DIM - 1 - i) for i in range(TABLE_DIM)]


def _diagonal_cells() -> list[tuple[int, int]]:
    cells = list(random.choice((_main_diagonal_cells(), _anti_diagonal_cells())))
    random.shuffle(cells)
    return cells


def _is_full_diagonal(cells: list[tuple[int, int]]) -> bool:
    return len(cells) == TABLE_DIM and frozenset(cells) in _diagonal_cell_sets(TABLE_DIM)


def _scatter_fail_triplet() -> list[tuple[int, int]]:
    rs = random.sample(range(TABLE_DIM), random.choice((2, 3)))
    cs = random.sample(range(TABLE_DIM), random.choice((2, 3)))
    cells = [(rs[0], cs[0]), (rs[1], cs[1])]
    pool = [(r, c) for r in rs for c in cs if (r, c) not in cells]
    cells.append(random.choice(pool))
    assert len(set(cells)) == 3
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    """Pick distinct cells that do not form a complete main/anti diagonal."""
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
            removed = random.choice(tuple(diag))
            replacement_pool = [
                (r, c)
                for r in range(TABLE_DIM)
                for c in range(TABLE_DIM)
                if (r, c) not in diag
            ]
            cells = list((diag - {removed}) | {random.choice(replacement_pool)})
        elif mode == "off_anti":
            diag = set(_anti_diagonal_cells())
            removed = random.choice(tuple(diag))
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
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN

    cropped = out[TABLE_TOP : TABLE_TOP + TABLE_H, TABLE_LEFT : TABLE_LEFT + TABLE_W].copy()
    final = _verdict_badge(TABLE_H, TABLE_W, GREEN if success else RED, BG)
    final[:TABLE_H, :TABLE_W] = cropped
    return final


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
    raise RuntimeError("failed to generate a valid d8e07eb2 combo variant input")
