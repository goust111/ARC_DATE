# Variant of d8e07eb2
# Dimension: D12 — 多步处理顺序 (multi_step)
# Change: 单步处理：同时将所有查询字形匹配到参考表，再一次性判定行/列完整性... → 两轮匹配：先色后形

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


# Design analysis for the D12 variant
# ===================================
# concretization:
# - The canvas, separators, query band, and 4x4 table stay unchanged.
# - The visible change happens inside the table cells in the output:
#   round 1 marks every same-color candidate cell with yellow background (4),
#   round 2 upgrades only exact same-color+same-shape matches to green (3).
# - The output therefore keeps both stages at once: non-winning color candidates
#   stay yellow, while the exact subset becomes green.
#
# behavior trace:
# - Original task: if a query is a blue zigzag, only the one table cell with that
#   exact blue zigzag turns green.
# - Variant: all blue table cells first become yellow; then only the blue zigzag
#   cell turns green. Any other blue cell remains yellow as a trace of the first pass.
# - The row/column verdict still depends only on the green exact matches, not on
#   the yellow color candidates.
#
# complexity calibration:
# - To make the first-pass effect reliably visible, glyph colors 3 and 4 are
#   removed from the input palette so green/yellow remain purely output markers.
# - The generator uses a repeated-color table pattern: every query color appears
#   elsewhere in the table, so yellow-only leftovers are guaranteed.
# - Query glyphs are chosen with distinct colors. This avoids overlapping
#   first-pass buckets and makes the two-stage filtering easy to read by eye.
#
# implicit-assumption audit against the original solver:
# - `_cell_signature(...) not in query_set` used to be the only selection test.
#   That no longer captures the first stage, so the new solver separately tracks
#   query colors and exact query signatures.
# - The original green fill directly wrote final matches into the table. In the
#   variant, color-only candidates must be written first, then exact matches must
#   overwrite those candidates with green.
# - The original glyph palette allowed yellow glyphs. That was harmless before,
#   but it would make yellow candidate highlighting ambiguous now, so the
#   generator reserves yellow and green for output-only meanings.
#
# placement legality:
# - The table geometry remains the original 4x4 arrangement of 5x5 cells at
#   fixed coordinates, so there is no new overlap risk.
# - Query glyphs still occupy the same 5x5 slots in the top band and are copied
#   directly from table cells, preserving alignment and cell inference.
#
# observability / cleanliness:
# - The yellow-vs-green distinction lives only in table backgrounds, never on the
#   separator rows or within the verdict band, so the layered process is readable.
# - Because query colors are distinct and repeated in the table, a human can see
#   that yellow means "same color family" and green means "the exact member".
#
# new solver assumptions and how the generator satisfies them:
# - Each glyph cell is monochrome. The generator renders each 3x3 shape with a
#   single non-background color, so color extraction is well-defined.
# - Every query glyph has exactly one exact table match. The generator keeps all
#   16 shapes distinct, so matching by full signature is unique.
# - Every query color has at least one extra table occurrence. The repeated-color
#   layout guarantees at least two cells per query color, so round 1 is visible.


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


def _signature_color(sig: tuple[tuple[int, int, int], ...]) -> int:
    colors = {int(v) for _r, _c, v in sig}
    assert len(colors) == 1, "glyphs must be monochrome"
    return next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter query glyphs through a reference table in two visible passes.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a query row of 5x5 glyph cells.
    - Between separators is a 4x4 table of the same cell size; table colors can
      repeat, so color alone may point to multiple candidate cells.

    Rule:
    - First mark every table cell whose glyph color appears in the query row by
      filling that cell's background yellow.
    - Then, inside those color candidates, upgrade the cells whose full 3x3
      colored glyph exactly matches a query glyph to green.
    - Use only the green exact matches for the final verdict: if four queries are
      present and those exact matches form a complete table row or column, also
      green-fill the query band and verdict band; otherwise fill the verdict band red.

    Key insight: Yellow is the intermediate color filter, not the answer. The
    solution comes from reading the green exact matches as a refined subset of
    the yellow candidate pool.
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

    row_starts = [r for r in range(row0, bottom_sep, cell_size) if r + cell_size <= bottom_sep]
    col_starts = [c for c in range(col0, w, cell_size) if c + cell_size <= w]
    assert row_starts and col_starts, "could not infer table cell grid"

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"
    query_colors = {_signature_color(sig) for sig in query_sigs}

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, bg)
            if sig is None:
                continue
            cell = out[r : r + cell_size, c : c + cell_size]
            if _signature_color(sig) in query_colors:
                cell[cell == bg] = 4
            if sig in query_set:
                selected.append((int(ri), int(ci)))
                cell[cell == bg] = 3

    complete_query = len(query_sigs) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_sigs) and (same_row or same_col)

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
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Fixed repeated-color scaffold --
COLOR_INDEX_PATTERN = (
    (0, 1, 2, 3),
    (4, 5, 6, 0),
    (1, 2, 3, 4),
    (5, 6, 0, 1),
)

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


def _color_grid() -> list[list[int]]:
    palette = list(GLYPH_COLORS)
    random.shuffle(palette)
    return [[palette[idx] for idx in row] for row in COLOR_INDEX_PATTERN]


def _line_cells() -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _cells_have_distinct_colors(cells: list[tuple[int, int]], color_grid: list[list[int]]) -> bool:
    colors = [color_grid[r][c] for r, c in cells]
    return len(colors) == len(set(colors))


def _is_full_line(cells: list[tuple[int, int]]) -> bool:
    return len(cells) == TABLE_DIM and (
        len({r for r, _c in cells}) == 1 or len({c for _r, c in cells}) == 1
    )


def _fail_cells(count: int, color_grid: list[list[int]]) -> list[tuple[int, int]]:
    pool = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM)]
    for _ in range(500):
        cells = random.sample(pool, count)
        if _is_full_line(cells):
            continue
        if len({r for r, _c in cells}) == 1 or len({c for _r, c in cells}) == 1:
            continue
        if not _cells_have_distinct_colors(cells, color_grid):
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample a distinct-color fail set")


def _expected_output(
    grid: np.ndarray,
    query_cells: list[tuple[int, int]],
    color_grid: list[list[int]],
    success: bool,
) -> np.ndarray:
    out = grid.copy()
    query_colors = {color_grid[r][c] for r, c in query_cells}
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            if color_grid[r_i][c_i] in query_colors:
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

    masks = _balanced_shape_masks()
    color_grid = _color_grid()
    glyphs: list[np.ndarray] = []
    for i, mask in enumerate(masks):
        r_i, c_i = divmod(i, TABLE_DIM)
        color = color_grid[r_i][c_i]
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    query_cells = _line_cells() if success else _fail_cells(3 if random.random() < P_QUERY_3 else 4, color_grid)
    assert _cells_have_distinct_colors(query_cells, color_grid), "query colors must stay distinct"

    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], 0, COLS[slot])

    expected = _expected_output(grid, query_cells, color_grid, success)
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
    raise RuntimeError("failed to generate a valid d8e07eb2 D12 input")
