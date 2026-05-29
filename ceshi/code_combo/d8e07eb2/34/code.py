# Combo variant of d8e07eb2 (index 34)
# combo_spec: D4/2+D13/0
# Name: 颜色边框标记+分区背景色
#
# Design analysis
# spec:
#   variant_focus:
#     changed_stages:
#       - matched table cells are marked by copied-color borders instead of uniform green fills
#       - the input no longer uses one global background color; query/table/verdict bands have
#         distinct background colors that remain meaningful in the output unless overwritten by
#         the success/fail signal
#     preserved_stages:
#       - a glyph is still identified by the exact pair (3x3 shape, glyph color)
#       - the success test is still "all four query slots filled and matched cells form exactly
#         one complete table row or one complete table column"
#       - success still recolors the query band and verdict band green; failure still recolors
#         only the verdict band red
#   concretization_and_trace:
#     original_behavior:
#       - the whole canvas background was teal, so query extraction and table extraction could
#         both ignore color 8
#       - each matched 5x5 table cell turned into green background plus the original glyph
#     combo_behavior:
#       - the query band uses gray background, the table band uses teal background, and the
#         verdict band uses black background
#       - query glyphs and table glyphs share the same foreground shape+color, but are rendered
#         on their local band backgrounds
#       - when a table glyph matches a query glyph, only the 5x5 perimeter is recolored, and
#         the border color is copied from that glyph itself
#     example_trace:
#       - suppose slot 0 of the query band contains a maroon shape copied from table cell (2, 1)
#       - the old task would paint every teal background pixel in that table cell green
#       - the combo output instead keeps the table cell interior teal except for the original
#         maroon glyph, and turns the 16 perimeter pixels of that 5x5 cell maroon
#       - if all four query slots are occupied and the four matched coordinates make one table
#         row or one table column, the gray query background flips to green and the black verdict
#         band flips to green; otherwise the query band stays gray and the verdict band becomes red
#   complexity_calibration:
#     visibility:
#       - the border marker is visibly different from the original fill marker because the cell
#         keeps an inner teal cavity around the 3x3 glyph
#       - the band backgrounds create a stable three-zone layout that is easy to notice from a
#         few examples before any output recoloring happens
#     ambiguity_control:
#       - the query background color is removed from the glyph palette so query glyph pixels
#         cannot disappear into their local background
#       - glyph identity is still shape+color, and all 16 table cells keep unique signatures, so
#         repeated colors do not create ambiguous matches
#       - only three band backgrounds are used, avoiding an overcomplicated per-row color code
#   implicit_assumption_audit:
#     broken_from_original:
#       - `get_bg()` is no longer safe because the query band background differs from the table
#         background; using one global background would treat gray query padding as foreground
#       - matched-cell painting can no longer use `cell[cell == bg] = 3` because the marker color
#         now depends on the matched glyph
#     repaired_in_combo_solver:
#       - separator rows are detected from the only repeated non-adjacent pair of full-width
#         uniform rows, not from a single global background complement
#       - query signatures are extracted with the query band's local background color
#       - table signatures are extracted with the table band's local background color
#       - border drawing touches only the perimeter, so it never overwrites the centered 3x3 glyph
#   placement_legality:
#     - grid size, separator rows, query slots, and 4x4 table geometry stay unchanged
#     - query glyphs are rendered into the same four 5x5 slots as before, with at most one empty
#       trailing slot when there are only three queries
#     - the 3x3 glyph sits inside a 1-pixel padding ring, so the new border marker never overlaps
#       the glyph and always remains inside the cell bounds
#   visual_observability:
#     - from 2-4 examples, a human can separately notice:
#       1. the top, middle, and bottom bands start with different backgrounds in the input
#       2. matched table cells get framed in the same color as the copied query glyph
#       3. green/red still encode the verdict, so the role of the bottom band stays legible
#     - the color-copy rule is structurally supported because the matching query glyph is visible
#       in the same example; no hidden numeric color ordering is needed
#   cleanliness_and_color_inference:
#     - each region has one stable base color, while each matched cell has at most one extra frame
#     - the frame sits on the outer ring and the glyph sits in the inner 3x3, so layers stay easy
#       to parse visually
#     - using three band backgrounds plus copied glyph colors is still inferable because each color
#       change is tied to either a fixed region role or a direct query-table correspondence
#   generator_guarantees_for_new_solver_assumptions:
#     - separator rows share one color and are the only non-adjacent repeated full-width uniform
#       rows, so `_separator_rows()` remains valid
#     - row 0 col 0, row 6 col 0, and row 28 col 0 are guaranteed background samples for the
#       query/table/verdict bands respectively
#     - `_signature_color()` assumes each glyph is monochrome; generator still renders each 3x3
#       glyph with exactly one foreground color
#     - query glyphs are re-rendered on the query background rather than copied as teal-backed
#       tiles, ensuring that the multi-background behavior actually appears in generated inputs

from __future__ import annotations

import random

import numpy as np

from arc_utils import paste


def _uniform_row_color(g: np.ndarray, r: int) -> int | None:
    row = g[int(r), :]
    if bool(np.all(row == row[0])):
        return int(row[0])
    return None


def _separator_rows(g: np.ndarray) -> tuple[int, int]:
    top_band_color = int(g[0, 0])
    bottom_band_color = int(g[-1, 0])
    groups: dict[int, list[int]] = {}
    for r in range(g.shape[0]):
        color = _uniform_row_color(g, r)
        if color is not None:
            groups.setdefault(int(color), []).append(int(r))
    candidates = [
        rows
        for color, rows in groups.items()
        if color not in {top_band_color, bottom_band_color} and len(rows) == 2
    ]
    assert len(candidates) == 1, "expected one repeated non-adjacent separator-row pair"
    return int(candidates[0][0]), int(candidates[0][1])


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
    Match the query glyphs to the table while preserving the three-band color code.

    Input structure:
    - The canvas is split by two full-width separator rows into a gray query band,
      a teal reference-table band, and a black verdict band.
    - The query band contains up to four 5x5 slots with centered 3x3 monochrome
      glyphs rendered on the gray local background.
    - The middle band contains a 4x4 table of the same 5x5 slots; each glyph is
      identified by its exact shape together with its foreground color.

    Rule:
    - Find every table glyph that exactly matches a query glyph in both shape and
      color, using each band's own background color when extracting signatures.
    - For each matched table cell, draw a one-pixel border around the 5x5 cell in
      that glyph's own color instead of filling the whole cell.
    - If all four query slots are filled and the matched table cells form one full
      row or one full column, recolor the query band's background pixels and the
      entire verdict band green; otherwise leave the query band unchanged and
      recolor the verdict band red.

    Key insight: The reasoning task is still a lookup followed by a row/column
    test, but the solver must now respect local band backgrounds before it can
    see the same glyph identities in the top and middle regions.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    cell_size = 5

    top_sep, bottom_sep = _separator_rows(g)
    query_bg = int(g[0, 0])
    table_bg = int(g[top_sep + 1, 0])
    verdict_bg = int(g[bottom_sep + 1, 0])

    table_region = g[top_sep + 1 : bottom_sep, :]
    table_fg = np.argwhere(table_region != table_bg)
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
        if (sig := _cell_signature(g, 0, c, cell_size, query_bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, table_bg)
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
        query_band[query_band == query_bg] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED

    return out


# --------------- generator (o5x) ---------------

# -- Grid layout --
GRID_H, GRID_W = 30, 22
TOP_SEP_ROW = 5
BOTTOM_SEP_ROW = 27
QUERY_BAND_H = 5
VERDICT_ROW = 28

# -- Colors --
QUERY_BG = 5
TABLE_BG = 8
VERDICT_BG = 0
GREEN = 3
RED = 2
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 6, 7, 9)

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
    """Assign 16 colors so each available glyph color appears 1-3 times."""
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks() -> list[int]:
    """Pick 16 distinct 3x3 masks with a balanced fill-count distribution."""
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


def _glyph(mask: int, color: int, bg: int) -> np.ndarray:
    """Render a 3x3 bitmask as a 5x5 cell with a 1-pixel local-background border."""
    cell = np.full((CELL, CELL), int(bg), dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = int(color)
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
    """Pick distinct cells that do not make a full row or full column."""
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


def _cell_index(r_i: int, c_i: int) -> int:
    return int(r_i) * TABLE_DIM + int(c_i)


def _expected_output(
    grid: np.ndarray, specs: list[tuple[int, int]], cells: list[tuple[int, int]], success: bool
) -> np.ndarray:
    """Build the expected solver output for the combo variant."""
    out = grid.copy()
    for r_i, c_i in cells:
        _mask, color = specs[_cell_index(r_i, c_i)]
        _draw_border(out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL], color)
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == QUERY_BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _base_grid() -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), TABLE_BG, dtype=int)
    grid[:TOP_SEP_ROW, :] = QUERY_BG
    grid[TOP_SEP_ROW, :] = SEP
    grid[TOP_SEP_ROW + 1 : BOTTOM_SEP_ROW, :] = TABLE_BG
    grid[BOTTOM_SEP_ROW, :] = SEP
    grid[VERDICT_ROW:, :] = VERDICT_BG
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Try to generate one valid input grid; return None on oracle mismatch."""
    grid = _base_grid()

    masks = _balanced_shape_masks()
    colors = _balanced_color_list()
    specs = list(zip(masks, colors, strict=True))

    for i, (mask, color) in enumerate(specs):
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, _glyph(mask, color, TABLE_BG), ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    for slot, (r_i, c_i) in enumerate(query_cells):
        mask, color = specs[_cell_index(r_i, c_i)]
        paste(grid, _glyph(mask, color, QUERY_BG), 0, COLS[slot])

    expected = _expected_output(grid, specs, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
