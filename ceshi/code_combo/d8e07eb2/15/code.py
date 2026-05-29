# Combo variant of d8e07eb2 (index 15)
# combo_spec: D7/1+D11/2
# Name: 垂直分隔+双面板输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the fused variant:
# 1. Concrete behavior:
#    The original task used horizontal bands. Here the input is rotated into
#    vertical strips: query column | separator | table | separator | verdict
#    column. The output is not an in-place recolor anymore: it is rebuilt as two
#    side-by-side panels. The left panel shows only the 4x4 reference table with
#    matched cells highlighted; the right panel shows only the query strip and
#    the verdict strip.
# 2. Complexity calibration:
#    The 4x4 table, 5x5 cell size, and 3-4 query count are preserved. This keeps
#    the original matching logic intact while making the rotated layout and panel
#    relocation visually obvious in 2-3 examples.
# 3. Hidden assumptions replaced:
#    The old solver assumed full-width separator rows, query cells on row 0, and
#    verdict rows at the bottom. The new solver instead assumes full-height
#    separator columns, query cells aligned to table rows, and an output canvas
#    whose left panel is a clean square table frame.
# 4. Placement legality:
#    The generator fixes a 22x30 canvas so 1 + 4*5 + 1 rows and 5 + 1 + 1 +
#    4*5 + 1 + 2 columns all fit exactly. Query, table, and verdict areas are
#    disjoint; the output panels also remain disjoint.
# 5. Visual observability and neatness:
#    The rotation is visible in the input because the query glyphs now stack
#    vertically. The frame-separation change is visible in the output because the
#    table leaves the center strip and reappears as a clean left panel, while the
#    query/verdict content is isolated on the right.


def _constant_non_bg_cols(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(c)
        for c in range(g.shape[1])
        if bool(np.all(g[:, c] == g[0, c])) and int(g[0, c]) != int(bg)
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


def _filled_cell(src: np.ndarray, bg: int, fill: int | None) -> np.ndarray:
    if fill is None:
        return np.array(src, copy=True)
    out = np.full(src.shape, int(fill), dtype=int)
    mask = src != int(bg)
    out[mask] = src[mask]
    return out


def _infer_layout(g: np.ndarray, bg: int, cell_size: int) -> dict[str, int | list[int]]:
    h, w = g.shape
    sep_cols = _constant_non_bg_cols(g, bg)
    assert len(sep_cols) >= 2, "expected two full-height separator columns"
    left_sep = sep_cols[0]
    right_sep = sep_cols[-1]

    table_region = g[:, left_sep + 1 : right_sep]
    table_fg = np.argwhere(table_region != bg)
    assert table_fg.size, "reference table must contain glyphs"

    first_glyph_r = int(table_fg[:, 0].min())
    first_glyph_c = left_sep + 1 + int(table_fg[:, 1].min())
    row0 = first_glyph_r - 1
    col0 = first_glyph_c - 1

    row_starts = [r for r in range(row0, h, cell_size) if r + cell_size <= h]
    col_starts = [c for c in range(col0, right_sep, cell_size) if c + cell_size <= right_sep]
    assert row_starts and col_starts, "could not infer table cell grid"

    query_c0 = left_sep - cell_size
    verdict_c0 = right_sep + 1
    verdict_w = w - verdict_c0
    left_panel_w = cell_size * len(col_starts) + 2
    right_panel_w = cell_size + verdict_w
    assert query_c0 >= 0, "query strip must fit to the left of the first separator"
    assert left_panel_w + 1 + right_panel_w == w, "output panel widths do not fit canvas"

    return {
        "h": h,
        "w": w,
        "left_sep": left_sep,
        "right_sep": right_sep,
        "query_c0": query_c0,
        "verdict_c0": verdict_c0,
        "verdict_w": verdict_w,
        "left_panel_w": left_panel_w,
        "out_sep": left_panel_w,
        "out_table_c0": 1,
        "out_query_c0": left_panel_w + 1,
        "out_verdict_c0": left_panel_w + 1 + cell_size,
        "row_starts": row_starts,
        "col_starts": col_starts,
    }


def _compose_output(
    g: np.ndarray,
    layout: dict[str, int | list[int]],
    matched: set[tuple[int, int]],
    success: bool,
    bg: int,
    cell_size: int,
) -> np.ndarray:
    h = int(layout["h"])
    w = int(layout["w"])
    out = np.full((h, w), int(bg), dtype=int)
    out[:, int(layout["out_sep"])] = 6

    row_starts = list(layout["row_starts"])
    col_starts = list(layout["col_starts"])
    query_c0 = int(layout["query_c0"])
    out_table_c0 = int(layout["out_table_c0"])
    out_query_c0 = int(layout["out_query_c0"])
    out_verdict_c0 = int(layout["out_verdict_c0"])

    if success:
        out[:, out_query_c0 : out_query_c0 + cell_size] = 3
        out[:, out_verdict_c0:] = 3
    else:
        out[:, out_verdict_c0:] = 2

    for ri, r in enumerate(row_starts):
        query_sig = _cell_signature(g, r, query_c0, cell_size, bg)
        if query_sig is None:
            continue
        src = g[r : r + cell_size, query_c0 : query_c0 + cell_size]
        patch = _filled_cell(src, bg, 3 if success else None)
        paste(out, patch, r, out_query_c0)

    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            src = g[r : r + cell_size, c : c + cell_size]
            patch = _filled_cell(src, bg, 3 if (ri, ci) in matched else None)
            paste(out, patch, r, out_table_c0 + ci * cell_size)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match vertically listed query glyphs against a reference table, then rebuild
    the answer as two horizontal panels.

    Input structure:
    - Teal background with two full-height pink separator columns.
    - Left of the first separator is a vertical query strip of 5x5 glyph cells.
    - Between separators is the 4x4 reference table; each 5x5 cell contains a
      centered 3x3 glyph identified by both color and shape.
    - Right of the second separator is an empty verdict strip.

    Rule:
    - Find every table cell whose glyph exactly appears in the query strip and
      green-fill that cell's background.
    - If the query strip contains all four slots and the matched table cells make
      one complete table row or one complete table column, the query strip is
      also green-backed and the verdict strip is green; otherwise the verdict
      strip is red and the query strip keeps its teal background.
    - Present the result as a left table panel and a right query+verdict panel,
      separated by one vertical pink divider.

    Key insight: The logical test is still "do the queried glyphs form a full
    line in the table?", but the evidence is now split across a rotated input and
    a reformatted output, so solving requires separating content matching from
    spatial presentation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5
    layout = _infer_layout(g, bg, cell_size)

    row_starts = list(layout["row_starts"])
    col_starts = list(layout["col_starts"])
    query_c0 = int(layout["query_c0"])

    query_sigs = [
        sig
        for r in row_starts
        if (sig := _cell_signature(g, r, query_c0, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    matched: set[tuple[int, int]] = set()
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            if _cell_signature(g, r, c, cell_size, bg) in query_set:
                matched.add((int(ri), int(ci)))

    complete_query = len(query_sigs) == len(row_starts)
    same_row = len({r for r, _c in matched}) == 1
    same_col = len({c for _r, c in matched}) == 1
    success = complete_query and len(matched) == len(query_sigs) and (same_row or same_col)

    return _compose_output(g, layout, matched, success, bg, cell_size)


# --------------- generator (combo D7/1 + D11/2) ---------------

# -- Input canvas --
GRID_H, GRID_W = 22, 30
QUERY_SEP_COL = 5
TABLE_SEP_COL = 27

# -- Colors --
BG = 8
GREEN = 3
RED = 2
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (1, 6, 11, 16)
TABLE_COLS = (7, 12, 17, 22)
QUERY_COL = 0
VERDICT_COL = 28

# -- Output panel geometry --
OUT_TABLE_PANEL_W = 22
OUT_SEP_COL = 22
OUT_QUERY_COL = 23
OUT_VERDICT_COL = 28

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
    layout = _infer_layout(grid, BG, CELL)
    return _compose_output(grid, layout, set(cells), success, BG, CELL)


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[:, QUERY_SEP_COL] = SEP
    grid[:, TABLE_SEP_COL] = SEP

    masks = _balanced_shape_masks()
    colors = _balanced_color_list()
    glyphs: list[np.ndarray] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], TABLE_COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], ROWS[slot], QUERY_COL)

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
