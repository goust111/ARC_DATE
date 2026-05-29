# Combo variant of d8e07eb2 (index 11)
# combo_spec: D4/1+D14/2
# Name: 纯绿匹配+灰色未匹配

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "solvers"))

from arc_utils import get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original matched cell: teal background pixels become green, but the 3x3 glyph
#      itself keeps its original color. Original unmatched cell: unchanged.
#    - Fused matched cell: the entire 5x5 cell becomes solid green, so the glyph
#      disappears. Fused unmatched cell: only glyph pixels are recolored to gray,
#      while teal background pixels stay teal.
#    - Example: if a cell contains a maroon L-shape on teal and it is queried,
#      output is a 5x5 green block; if the same cell is not queried, output keeps
#      the teal border and turns the maroon L-shape into a gray L-shape.
# b) Complexity calibration:
#    - The original 4x4 table, 3-4 query glyphs, and row/column success test stay
#      unchanged. This preserves the matching task and only changes how evidence is
#      rendered, producing a clean heatmap rather than simplifying the reasoning.
# c) Implicit assumptions affected by the fusion:
#    - Matching must still be computed from the input grid, not from partially
#      transformed output cells, otherwise graying could corrupt signatures.
#    - Full-cell green overwrite is safe because input glyph colors never use green.
#    - Gray is allowed as an input glyph color, but at most 3 cells use it while
#      12-13 cells remain unmatched, so the fade effect remains visible in samples.
# d) Placement legality:
#    - Cells remain fixed 5x5 slots in a 4x4 table, with query glyphs pasted only
#      into the top four slots. No new objects are introduced, so no overlap risk
#      beyond the original layout exists.
# e) Visual observability:
#    - The table now has exactly two output states: solid green blocks for matches
#      and gray glyphs on teal for non-matches. Across 2-4 examples this is easier
#      to infer than the original mixed "green background plus colored glyph" style.
# f) Visual neatness:
#    - Because matched cells are uniform and unmatched cells keep only one glyph
#      layer, the table is less cluttered than the base task and the query/verdict
#      bands remain visually separated by the pink divider rows.
# g) New solver assumptions audited against the generator:
#    - `main()` assumes every table slot contains one glyph and query glyphs are
#      distinct exact copies of table cells. `generate_input()` still enforces both
#      by creating 16 unique (shape, color) pairs and sampling distinct query cells.


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
    Match query glyphs against a reference table and render a green/gray heatmap.

    Input structure:
    - The canvas is teal with two full-width pink separator rows.
    - The top band holds up to four 5x5 query cells, each containing one centered
      3x3 glyph copied from the table.
    - The middle band is a 4x4 table of the same 5x5 cells; each glyph is defined
      jointly by its 3x3 shape and its color.

    Rule:
    - Every table cell whose glyph appears in the query band turns into a solid
      green 5x5 block.
    - Every table cell whose glyph is not queried keeps its shape but its glyph
      pixels are recolored to gray.
    - If the queried cells occupy a complete table row or column and all four query
      slots are filled, green-fill the query-band background and the verdict band;
      otherwise leave the query band as-is and color the verdict band red.

    Key insight: The combinational change only affects table rendering. Matching and
    row/column validation are still based on exact input glyph identity, while the
    output compresses the table into "green hit" versus "gray miss" states.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5
    green = 3
    gray = 5
    red = 2

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

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, bg)
            assert sig is not None, "every table slot should contain a glyph"
            cell = out[r : r + cell_size, c : c + cell_size]
            if sig in query_set:
                selected.append((int(ri), int(ci)))
                cell[:, :] = green
            else:
                cell[cell != bg] = gray

    complete_query = len(query_sigs) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_sigs) and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = green
        out[bottom_sep + 1 :, :] = green
    else:
        out[bottom_sep + 1 :, :] = red

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
GRAY = 5
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
    matched = set(cells)
    for ri, r in enumerate(ROWS):
        for ci, c in enumerate(COLS):
            cell = out[r : r + CELL, c : c + CELL]
            if (ri, ci) in matched:
                cell[:, :] = GREEN
            else:
                cell[cell != BG] = GRAY
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
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
