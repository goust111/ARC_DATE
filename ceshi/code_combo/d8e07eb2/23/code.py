# Combo variant of d8e07eb2 (index 23)
# combo_spec: D2/0+D7/2
# Name: 查询在底部+单分隔行

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste

# Design analysis for combo D2/0 + D7/2
# 1. Concrete fused layout and behavior:
#    - The original task had three vertical bands:
#        query rows 0..4, separator row 5, table rows 7..26, separator row 27,
#        verdict rows 28..29.
#    - This combo must do two things simultaneously:
#        * D2/0 moves the query cells to the bottom side of the canvas.
#        * D7/2 removes one separator so query and verdict now share one side of the table.
#    - The fused layout used here is:
#        table rows 2..21, single separator row 22, query rows 23..27,
#        verdict rows 28..29.
#      Rows 0..1 stay as plain background. This keeps the 4x4 5x5 table unchanged,
#      preserves 5x5 query cells, preserves the 2-row verdict strip, and makes the
#      bottom zone visibly compact.
#    - Example trace:
#        original success: four query glyphs at top match one full table row -> matched
#        cells turn green, top query band turns green, bottom verdict turns green.
#        fused success: the same four glyphs now sit below the separator; the matched
#        upper table row still turns green, but now rows 23..27 and 28..29 turn green.
#        fused failure: matched cells still turn green, rows 23..27 stay unchanged, and
#        rows 28..29 turn red.
#
# 2. Complexity / observability calibration:
#    - The reference table keeps the original 4x4 structure and the original joint
#      identity rule "shape + color". This avoids simplification into pure-color lookup.
#    - Query count remains 3 or 4, so the same row/column-completion logic still matters.
#    - The visual difference from the base task is strong and immediate: humans now see
#      the dense table above a single separator and must search downward for the queries.
#
# 3. Hidden assumptions broken by the combo:
#    - `len(sep_rows) >= 2` is no longer valid; there must be exactly one separator row.
#    - `table_region = g[top_sep+1:bottom_sep]` is invalid because the table is now above
#      the separator, not between two separators.
#    - Reading queries from row 0 is invalid; they now start at `sep + 1`.
#    - Writing the verdict below the last separator is invalid; it is now always the last
#      two rows of the grid, directly below the query band.
#    - The old success fill `out[:top_sep, :] = green` would incorrectly paint the table;
#      success must now fill only the bottom query band.
#
# 4. Placement legality guaranteed by the generator:
#    - The table uses fixed starts rows (2, 7, 12, 17) and cols (1, 6, 11, 16), so all
#      16 cells remain in-bounds and non-overlapping.
#    - Query glyphs reuse the same 5x5 column slots at row 23, so they do not collide
#      with the table or verdict strip.
#    - The single separator at row 22 spans the full width and never overlaps glyphs.
#
# 5. Non-degeneracy / fusion verification target:
#    - A generated sample is considered fused only if it has exactly one full-width
#      non-background separator row and at least one query glyph located below it.
#      The post-test script checks this metric over many random samples to reject silent
#      regression back to the original two-separator, top-query layout.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match bottom query glyphs against an upper reference table and judge alignment.

    Input structure:
    - A single full-width pink separator splits the canvas into two zones.
    - Above the separator is the 4x4 reference table of 5x5 cells, with only a small
      teal margin above it.
    - Below the separator, the next 5 rows hold the query glyph cells and the final
      2 rows form the verdict strip.

    Rule:
    - Find every table cell whose glyph exactly matches one of the bottom query glyphs
      in both shape and color, then paint that cell's teal background green.
    - If all four query slots are occupied and the matched table cells make one entire
      table row or one entire table column, also paint the query band green and paint
      the verdict strip green.
    - Otherwise keep the query band unchanged and paint only the verdict strip red.

    Key insight:
    - The single separator is now just the border between the lookup table and the
      compact query+verdict zone. The logic still depends on exact glyph identity and
      whether the selected table positions complete a full straight line.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    cell_size = 5
    verdict_h = 2

    sep_rows = _constant_non_bg_rows(g, bg)
    assert len(sep_rows) == 1, "expected exactly one full-width separator row"
    sep = sep_rows[0]

    table_region = g[:sep, :]
    table_fg = np.argwhere(table_region != bg)
    assert table_fg.size, "reference table must contain glyphs above the separator"

    first_glyph_r = int(table_fg[:, 0].min())
    first_glyph_c = int(table_fg[:, 1].min())
    row0 = first_glyph_r - 1
    col0 = first_glyph_c - 1

    row_starts = list(range(row0, sep, cell_size))
    row_starts = [r for r in row_starts if r + cell_size <= sep]
    col_starts = list(range(col0, w, cell_size))
    col_starts = [c for c in col_starts if c + cell_size <= w]
    assert len(row_starts) == 4 and len(col_starts) == 4, "could not infer 4x4 table"

    verdict_top = h - verdict_h
    query_top = verdict_top - cell_size
    assert query_top == sep + 1, "expected query band directly below separator"

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, query_top, c, cell_size, bg)) is not None
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

    complete_query = len(query_sigs) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_sigs) and (same_row or same_col)

    if success:
        query_band = out[query_top:verdict_top, :]
        query_band[query_band == bg] = 3
        out[verdict_top:, :] = 3
    else:
        out[verdict_top:, :] = 2

    return out


# --------------- generator (combo D2/0 + D7/2) ---------------

# -- Grid layout --
GRID_H, GRID_W = 30, 22
SEP_ROW = 22
QUERY_TOP = 23
VERDICT_TOP = 28
QUERY_BAND_H = 5
VERDICT_H = 2

# -- Colors --
BG = 8
GREEN = 3
RED = 2
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (2, 7, 12, 17)
COLS = (1, 6, 11, 16)

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
    """Assign 16 colors so each palette color appears 1-3 times."""
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks() -> list[int]:
    """Pick 16 distinct masks with a balanced fill-count distribution."""
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
    """Render a 3x3 bitmask as a 5x5 cell with a 1-pixel teal border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _line_cells() -> list[tuple[int, int]]:
    """Pick a complete row or column of 4 table cells."""
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    """Pick `count` cells that do not form one full row or one full column."""
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
    """Oracle for the fused layout."""
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[QUERY_TOP:VERDICT_TOP, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_TOP:, :] = GREEN
    else:
        out[VERDICT_TOP:, :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try to generate one valid fused input; return None on oracle mismatch."""
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[SEP_ROW, :] = SEP

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
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], QUERY_TOP, COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo D2/0+D7/2 input")
