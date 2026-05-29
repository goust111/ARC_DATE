# Variant of d8e07eb2
# Dimension: D11 — 网格输出尺寸 (grid_shape)
# Change: 输出网格与输入尺寸完全相同(30x22)，所有变换都在原位进行... → 输出仅保留匹配的行或列

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D11 crop variant
# ========================================
# a) Concrete behavior:
# - Input stays identical to the original task: a 30x22 canvas with a 4x4 table
#   of 5x5 cells, query cells above, and separator rows.
# - The matching rule also stays identical: compare each query glyph to table
#   glyphs by exact (shape + color) signature.
# - The changed step is the output stage. In the original task, success painted
#   the whole 30x22 canvas. Here, success discards the query band, separators,
#   empty margins, and verdict band, keeping only the matched complete row or
#   matched complete column from the reference table.
# - Because each table cell is 5x5 and there are 4 cells per successful line,
#   the success output is either 5x20 (row) or 20x5 (column). Each kept cell has
#   its background recolored to green, preserving the original glyph pixels.
# - Failure is concretized as a visible 1x1 red output instead of an empty grid.
#   This keeps the D11 size change observable in every failing example.
# - Small trace:
#   Original success: 4 query glyphs match table row 2 -> highlight those 4
#   table cells, paint query band green, paint bottom band green, return 30x22.
#   Variant success: the same 4 matches are found, but the solver returns only
#   the four highlighted row-2 cells concatenated into a 5x20 strip.
#   Original failure: return 30x22 with red verdict band. Variant failure:
#   return [[2]].
#
# b) Complexity calibration:
# - The table/query construction is unchanged, so there are still 16 distinct
#   (color, shape) pairs and 3-4 queries. This preserves the original matching
#   difficulty and avoids simplifying the task into a trivial crop.
# - Success keeps both orientations (row and column), so the new size reasoning
#   is not a single fixed size.
#
# c) Implicit-assumption audit of the original main():
# - Separator detection via uniform non-background rows still holds because the
#   input layout is unchanged.
# - Grid inference from the first glyph's centered 3x3 footprint still works
#   because the generator keeps the same 5x5 cells with a 1-pixel border.
# - Query signatures remain distinct; otherwise set membership would become
#   ambiguous. The generator still guarantees unique (color, shape) pairs.
# - The original success branch freely painted whole bands because output size
#   was fixed. That assumption is invalid here; the new solver must avoid any
#   whole-canvas write and instead crop from the matched table line only.
#
# d) Placement legality:
# - Generator placement is unchanged for inputs, so cells never overlap and all
#   objects fit the 30x22 canvas exactly as in the base task.
# - Success output crops already-existing 5x5 cells, so it cannot create
#   out-of-bounds artifacts.
#
# e) Visual observability:
# - The D11 change is obvious across examples: outputs become either a horizontal
#   green strip, a vertical green strip, or a single red pixel.
# - The retained glyphs keep their original colors and shapes, so the user can
#   still see which table line was selected.
#
# f) Visual cleanliness:
# - The output contains exactly one object family: the cropped matched line.
#   This removes the original overlapping highlights/query/verdict regions and
#   makes the transformed result cleaner rather than noisier.
#
# g) Color-change inference:
# - No new hidden color rule is introduced. Green still means "matched cell
#   background" and red still means "failure", only the canvas is cropped.


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _cell_signature(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[tuple[int, int, int], ...] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    return tuple((int(r), int(c), int(cell[r, c])) for r, c in zip(*np.where(mask), strict=False))


def _highlighted_cell(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> np.ndarray:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)].copy()
    cell[cell == int(bg)] = 3
    return cell


def _cropped_success_output(
    g: np.ndarray,
    selected: list[tuple[int, int]],
    row_starts: list[int],
    col_starts: list[int],
    cell_size: int,
    bg: int,
) -> np.ndarray:
    rows = {int(r) for r, _c in selected}
    cols = {int(c) for _r, c in selected}
    if len(rows) == 1:
        ri = next(iter(rows))
        parts = [
            _highlighted_cell(g, row_starts[ri], col_starts[ci], cell_size, bg)
            for ci in range(len(col_starts))
        ]
        return np.concatenate(parts, axis=1)
    assert len(cols) == 1, "success must be a full row or full column"
    ci = next(iter(cols))
    parts = [
        _highlighted_cell(g, row_starts[ri], col_starts[ci], cell_size, bg)
        for ri in range(len(row_starts))
    ]
    return np.concatenate(parts, axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up query glyphs in a reference table and keep only the successful line.

    Input structure:
    - A teal background contains two full-width pink separator rows.
    - The top band holds up to four 5x5 query cells, each containing a centered
      3x3 colored glyph.
    - Between the separators sits a 4x4 table of the same 5x5 cells; identity is
      determined by the exact glyph shape together with its color.

    Rule:
    - Find every table cell whose glyph exactly matches one of the query glyphs.
    - If the queries fill all four slots and those matched cells are one complete
      table row or one complete table column, recolor the matched cells'
      background to green and output only that row/column as a tight crop.
    - Otherwise output a single red pixel.

    Key insight: The original line-detection criterion is unchanged; only the
    representation of the answer changes from an in-place annotation to a crop.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5

    sep_rows = _constant_non_bg_rows(g, bg)
    assert len(sep_rows) >= 2, "expected two full-width separator rows"
    top_sep = sep_rows[0]
    bottom_sep = sep_rows[-1]

    table_region = g[top_sep + 1 : bottom_sep, :]
    table_fg = np.argwhere(table_region != int(bg))
    assert table_fg.size, "reference table must contain glyphs"

    first_glyph_r = top_sep + 1 + int(table_fg[:, 0].min())
    first_glyph_c = int(table_fg[:, 1].min())
    row0 = first_glyph_r - 1
    col0 = first_glyph_c - 1

    row_starts = [r for r in range(row0, bottom_sep, cell_size) if r + cell_size <= bottom_sep]
    col_starts = [c for c in range(col0, g.shape[1], cell_size) if c + cell_size <= g.shape[1]]
    assert row_starts and col_starts, "could not infer table cell grid"

    query_sigs = [sig for c in col_starts if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            if _cell_signature(g, r, c, cell_size, bg) in query_set:
                selected.append((int(ri), int(ci)))

    complete_query = len(query_sigs) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_sigs) and (same_row or same_col)

    if not success:
        return np.array([[2]], dtype=int)
    return _cropped_success_output(g, selected, row_starts, col_starts, cell_size, bg)


# --------------- generator (o5x -> D11 crop variant) ---------------

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
    bits = [(mask >> i) & 1 for i in range(9)]
    return any(bits[0:3]) and any(bits[6:9]) and any(bits[i] for i in (0, 3, 6)) and any(
        bits[i] for i in (2, 5, 8)
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
    if not success:
        return np.array([[RED]], dtype=int)

    row_starts = list(ROWS)
    col_starts = list(COLS)
    return _cropped_success_output(grid, cells, row_starts, col_starts, CELL, BG)


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
