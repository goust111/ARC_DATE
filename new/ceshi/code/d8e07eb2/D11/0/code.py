# Variant of d8e07eb2
# Dimension: D11 — 网格输出尺寸 (grid_shape)
# Change: 输出网格与输入尺寸完全相同(30x22)，所有变换都在原位进行
#   → 裁剪输出仅保留参考表区域，并在裁剪结果右下角附带一个小的判定标记

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for D11 (tight_crop-style output)
#
# spec_variant:
#   input:
#     same_as_original: true
#     grid: 30 x 22
#     layout:
#       - rows 0..4: query band, up to 4 copied glyph cells
#       - row 5: separator
#       - row 6: empty bg spacer
#       - rows 7..26, cols 1..20: 4x4 table of 5x5 cells
#       - row 27: separator
#       - rows 28..29: unused verdict band in the input
#   output:
#     main_payload:
#       - tight crop of only the 4x4 reference table area
#       - size 20 x 20 because the table is exactly 4*5 by 4*5
#       - matched cells still get bg->green filling exactly as in the original task
#     verdict_mark:
#       - append one extra row and one extra column, creating a 21 x 21 canvas
#       - place a 5-pixel L-shaped marker in the bottom-right corner
#       - green (3) if the queried cells form a complete row/column with 4 queries
#       - red (2) otherwise
#
#   behavior_trace:
#     original:
#       - solver marks matched table cells in-place
#       - success additionally floods the query band and bottom verdict band
#       - failure colors the full bottom verdict band red
#     variant:
#       - solver still marks matched table cells the same way
#       - it discards the query band, separator rows, spacer row, and verdict band
#       - it returns only the cropped table plus a tiny success/fail badge
#     concrete_example:
#       - if queries match all 4 cells of table row 2, original output is still 30x22,
#         with that row highlighted, query band green, and bottom band green
#       - in the variant, output is 21x21: the 20x20 table crop with the same 4
#         highlighted cells, plus a green L badge at the lower-right corner
#
#   complexity_check:
#     - The table content, glyph uniqueness, and success/fail branching stay unchanged,
#       so the reasoning load remains matching-based rather than collapsing into a
#       trivial crop.
#     - The verdict mark has only two colors and one fixed shape, so it is easy to
#       infer from a few examples.
#     - Cropping to the exact table prevents accidental regression to the original
#       same-size output.
#
#   implicit_assumptions_audit:
#     - `_constant_non_bg_rows`: still valid because input layout is unchanged and the
#       two separator rows remain full-width constant rows.
#     - table origin inference via first glyph minus one-cell border: still valid
#       because each 5x5 cell keeps a centered 3x3 glyph with a 1-pixel bg border.
#     - query extraction from row 0 and table columns: still valid because queries
#       remain pasted into the same 5x5 slots at the top.
#     - success logic requires exactly 4 queries and all selected cells lying on one
#       row or one column: unchanged from the original task.
#     - new assumption: the table occupies a contiguous 20x20 crop. The generator
#       guarantees the fixed 4x4-by-5x5 table geometry, so slicing that bbox is safe.
#     - new assumption: the added badge uses the background color around it. The
#       generator does not place any other content in output-only space, so no overlap
#       ambiguity exists.
#
#   placement_legality:
#     - input placement matches the original fixed coordinates, so cells never overlap
#       and never leave the 30x22 canvas.
#     - output placement is deterministic: table crop at (0,0), badge only on the
#       extra last row/column, with no overwrite of table pixels.
#
#   visual_observability:
#     - The most visible change is dimensional: every output shrinks from 30x22 to 21x21.
#     - The table highlight pattern is preserved, so the crop is clearly tied to the
#       reference table rather than being an unrelated subwindow.
#     - The badge is small but consistent, avoiding clutter while making success/fail
#       directly observable.
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


def _verdict_badge(color: int, bg: int) -> np.ndarray:
    badge = np.full((21, 21), int(bg), dtype=int)
    pts = ((18, 20), (19, 20), (20, 18), (20, 19), (20, 20))
    for r, c in pts:
        badge[int(r), int(c)] = int(color)
    return badge


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyphs to a lookup table, then return only the cropped table result.

    Input structure:
    - Same 30x22 layout as the original task: a top query band, two full-width
      separator rows, and a 4x4 table of 5x5 glyph cells in the middle.
    - Each glyph is defined by both its 3x3 shape and its color inside the cell.

    Rule:
    - Find every table cell whose glyph exactly appears in the query band and
      green-fill that cell's background pixels.
    - Decide success exactly as before: all 4 query slots must be occupied and
      the matched cells must form one complete table row or one complete column.
    - Output only the 20x20 table crop, plus a small bottom-right badge: green
      for success, red for failure.

    Key insight: The logical matching problem is unchanged; the variant only
    changes which spatial region survives into the final output.
    """
    g = np.asarray(grid, dtype=int)
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
    col_starts = list(range(col0, g.shape[1], cell_size))
    col_starts = [c for c in col_starts if c + cell_size <= g.shape[1]]
    assert row_starts and col_starts, "could not infer table cell grid"

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    table_marked = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            if _cell_signature(g, r, c, cell_size, bg) not in query_set:
                continue
            selected.append((int(ri), int(ci)))
            cell = table_marked[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = 3

    complete_query = len(query_sigs) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_sigs) and (same_row or same_col)

    table_r0 = row_starts[0]
    table_c0 = col_starts[0]
    table_h = len(row_starts) * cell_size
    table_w = len(col_starts) * cell_size
    cropped = table_marked[table_r0 : table_r0 + table_h, table_c0 : table_c0 + table_w].copy()

    out = _verdict_badge(3 if success else 2, bg)
    out[:table_h, :table_w] = cropped
    return out


# --------------- generator (crop variant) ---------------

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
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

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


def _line_cells() -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
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
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN

    cropped = out[TABLE_TOP : TABLE_TOP + TABLE_H, TABLE_LEFT : TABLE_LEFT + TABLE_W].copy()
    final = _verdict_badge(GREEN if success else RED, BG)
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
        query_cells = _line_cells()
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
    raise RuntimeError("failed to generate a valid d8e07eb2 D11 input")
