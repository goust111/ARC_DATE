# Variant of d8e07eb2
# Dimension: D4 — 匹配成功后的单元格变换 (transform)
# Change: 匹配的参考表单元格的5x5区域中，将背景像素(8)替换为绿色(3)，字形像素保持原色不变，形成绿色底色+彩色字形的效果
# ... → 匹配后用查询字形颜色重着色边框
#
# Design analysis
# spec:
#   variant_focus:
#     changed_stage: output marking of matched reference-table cells only
#     unchanged_stages:
#       - query glyph extraction still uses exact (shape + color) matching
#       - success/fail decision still depends on whether all four query slots are filled
#         and the matched table positions make one full row or one full column
#       - success still paints the query band green and the verdict band green
#       - failure still paints only the verdict band red
#   concretization_and_trace:
#     original_behavior:
#       - each matched 5x5 table cell kept the 3x3 glyph and turned every teal background
#         pixel inside that cell green
#     new_behavior:
#       - each matched 5x5 table cell keeps its centered 3x3 glyph unchanged
#       - instead of recoloring all teal background pixels, recolor the full 5x5 perimeter
#         (top row, bottom row, left col, right col) using the matched query glyph color
#     example_trace:
#       - if the query contains a yellow glyph copied from table cell (row 1, col 2),
#         the old output turned that whole cell into green background + yellow glyph
#       - the variant output keeps the same yellow 3x3 glyph, but the 16 perimeter pixels of
#         the 5x5 cell become yellow, leaving the four interior-corner-adjacent teal pixels
#         at positions (1,1)..(3,3) subject only to the glyph itself
#       - the success test still looks only at matched coordinates, so the decision path is
#         unaffected; only the local visualization differs
#   complexity_calibration:
#     why_not_more_changes:
#       - border color comes directly from the copied query glyph, so humans can infer the
#         mapping from local evidence without extra latent rules
#       - multiple glyph colors repeat across the 4x4 table, but each (shape, color) pair is
#         unique, so matching remains unambiguous
#     visibility:
#       - a colored frame is visually distinct from the original green fill on every matched
#         cell, because the 5x5 ring uses 16 pixels and preserves teal inside the frame
#   implicit_assumption_audit:
#     still_valid:
#       - separator detection assumes the two full-width non-background rows remain intact
#       - table grid inference assumes each glyph is centered in a 5x5 cell with a 1-pixel
#         border; the generator preserves that layout
#       - query signatures are read from row 0 aligned to the same 5x5 column starts
#     changed_code_requirements:
#       - the old `cell[cell == bg] = 3` relied on "all matched background pixels receive a
#         single fixed color"; this is invalid once match color varies per glyph
#       - the solver must derive a border color from the matched signature before drawing
#       - the border drawing must not overwrite the inner 3x3 glyph, so only the perimeter is
#         recolored
#   placement_legality:
#     - all table and query glyphs remain pasted into fixed 5x5 slots
#     - the centered 3x3 glyph never touches the perimeter, so border drawing cannot overlap
#       glyph pixels
#     - query slots are still placed only in the top band's four table-aligned columns
#   visual_observability:
#     - from 2-3 examples, a solver can see that matched table cells receive frames whose
#       colors copy the query glyph colors rather than uniform green fills
#     - success/failure remains legible because the query band and verdict band use the same
#       green/red coding as the base task
#   cleanliness_and_color_inference:
#     - the border sits in a dedicated ring and does not stack on top of glyph pixels
#     - color changes are local and structurally explained by exact copied glyph identity; no
#       hidden ordering or numeric color-id rule is introduced
#   new_solver_assumptions_checked_against_generator:
#     - `_signature_color()` assumes each glyph signature is monochrome; generator still builds
#       every glyph from one color only
#     - `_draw_border()` assumes a 5x5 cell so that a one-pixel perimeter exists around the
#       3x3 glyph; generator keeps CELL == 5 and 1-pixel padding

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


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
    colors = {int(color) for _r, _c, color in sig}
    assert len(colors) == 1, "glyph signatures must be monochrome"
    return next(iter(colors))


def _draw_border(cell: np.ndarray, color: int) -> None:
    cell[0, :] = int(color)
    cell[-1, :] = int(color)
    cell[:, 0] = int(color)
    cell[:, -1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the query glyphs against the reference table and frame the hits.

    Input structure:
    - Teal background with two full-width separator rows.
    - The top band contains up to four 5x5 query slots holding centered 3x3
      monochrome glyphs.
    - The middle band is a 4x4 table of the same 5x5 slots; each glyph is
      identified by its exact 3x3 shape together with its color.

    Rule:
    - For every table cell whose glyph appears in the query band, draw a
      one-pixel border around that 5x5 cell using the query glyph's color.
    - If the matched table positions are exactly one full row or one full
      column and all four query slots are filled, also paint the query band and
      the bottom verdict band green.
    - Otherwise leave the query band as-is and paint the bottom verdict band red.

    Key insight: The table lookup is unchanged; the only local transformation is
    the matched-cell marker, which becomes a copied-color frame instead of a
    uniform green fill.
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
            if sig not in query_set:
                continue
            selected.append((int(ri), int(ci)))
            _draw_border(out[r : r + cell_size, c : c + cell_size], _signature_color(sig))

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
    """Pick 16 distinct 3x3 masks with balanced fill counts."""
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
    """Pick a complete row or column of 4 cells."""
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    """Pick `count` distinct cells that do not make a full row or column."""
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


def _draw_expected_border(out: np.ndarray, r_i: int, c_i: int, color: int) -> None:
    cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
    _draw_border(cell, color)


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    """Build the expected solver output for the border-marking variant."""
    out = grid.copy()
    for r_i, c_i in cells:
        glyph = grid[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        color = int(next(iter(set(map(int, glyph[1:4, 1:4][glyph[1:4, 1:4] != BG])))))
        _draw_expected_border(out, r_i, c_i, color)
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try to generate one valid input grid; return None on oracle mismatch."""
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
    raise RuntimeError("failed to generate a valid d8e07eb2 input")
