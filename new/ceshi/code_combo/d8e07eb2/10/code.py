# Combo variant of d8e07eb2 (index 10)
# combo_spec: D4/0+D14/1
# Name: 绿色字形+红色未匹配背景

# Design analysis
# spec_variant:
#   source_task: d8e07eb2
#   preserved_structure:
#     - 30x22 fixed canvas on teal background 8.
#     - Two full-width separator rows of color 6 at r=5 and r=27.
#     - Query band in rows 0..4 with up to four 5x5 glyph cells aligned to table columns.
#     - Reference table in rows 7..26 as a 4x4 grid of 5x5 cells, each with a centered 3x3 glyph.
#     - Verdict band in rows 28..29.
#   fused_change:
#     - D4/0 flips the matched-cell highlight: do not paint the matched cell background; repaint the glyph
#       foreground itself to green.
#     - D14/1 makes the complement explicit: every unmatched table cell gets a red background while its glyph
#       color stays unchanged.
#     - The two edits must act on the same partition of the reference table, so the whole 4x4 table becomes a
#       hit/miss map rather than only marking the hits.
#   concrete_trace:
#     - Suppose a matched table cell contains a yellow L-shape and an unmatched cell contains a blue T-shape.
#     - Original task: matched yellow cell becomes green-background + yellow glyph; unmatched blue cell stays teal-background + blue glyph.
#     - Combo task: matched yellow cell becomes teal-background + green glyph; unmatched blue cell becomes red-background + blue glyph.
#     - Difference: positive evidence moved from background to foreground, and negative evidence is now visible in every remaining cell.
#   complexity_check:
#     - Matching still uses the exact (shape, color, position-within-cell) signature, so there is no new lookup ambiguity.
#     - The new red miss-marker would be weakened if input glyphs could already be red, because red glyphs on red backgrounds would vanish.
#       To keep the rule legible, the generator excludes both green and red from the input glyph palette.
#     - The table still contains 16 distinct glyph shapes, so recolored matches are easy to separate from unchanged unmatched glyph shapes.
#   implicit_assumption_audit:
#     - `_constant_non_bg_rows` still assumes the only full-width constant non-background rows are the two separators; preserved.
#     - `_cell_signature` still assumes each non-empty cell contains background plus a single-color 3x3 glyph; preserved by the generator.
#     - Table-grid inference still assumes a 1-pixel border around each 3x3 glyph inside its 5x5 cell; preserved.
#     - `query_sigs` uniqueness still relies on distinct source table cells and distinct signatures; preserved.
#     - The matched-cell write now uses `cell[cell != bg] = GREEN`, which assumes green is output-only for glyphs; generator guarantees that.
#     - The unmatched-cell write uses `cell[cell == bg] = RED`, which assumes each table cell contains background pixels to recolor; the 5x5 frame guarantees 16 such pixels.
#     - Success detection remains unchanged and still depends on all four query slots being filled and on the matched cells lying in one full row or column.
#   placement_legality:
#     - Query cells remain pasted at row 0 and columns (1, 6, 11, 16), fitting exactly inside the 22-column canvas.
#     - Table cells remain pasted at rows (7, 12, 17, 22) and the same four columns, fitting exactly inside rows 7..26.
#     - The input verdict band remains empty, so later full-band recoloring cannot overwrite input objects.
#   visual_observability:
#     - Every sample contains at least 12 unmatched table cells, so the red-background effect is always abundant.
#     - Every sample contains at least 3 matched cells, so the green-glyph effect is always visible.
#     - Because red and green are reserved for output-only marking inside the table, a human can infer the partition "green glyph = queried hit, red background = queried miss" from a few examples.
#   tidy_visuals:
#     - Each table cell receives exactly one local marking mode: either green foreground or red background, never both.
#     - Query-band and verdict-band coloring remain band-level signals, so the table-local marking does not interfere with the global success/fail cue.
#   generator_guarantees_for_new_main:
#     - Input glyph colors come from a palette that excludes red and green, making both output colors semantically clean.
#     - Both success and failure branches are sampled, so the combo is visible together with both verdict outcomes.
#     - `_expected_output` uses the same hit/miss partition as `main`, and generation rejects any input whose oracle does not match.

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


def _mark_table_cell(cell: np.ndarray, bg: int, matched: bool) -> None:
    if matched:
        cell[cell != int(bg)] = GREEN
    else:
        cell[cell == int(bg)] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up query glyphs in the reference table and convert the whole table into a hit/miss map.

    Input structure:
    - A teal 30x22 canvas is split by two full-width separator rows.
    - The top band contains up to four 5x5 query cells with centered 3x3 colored glyphs.
    - The middle band contains a 4x4 table of the same 5x5 cells; each glyph is identified by its
      exact color and 3x3 shape, not just by shape alone.

    Rule:
    - Any table cell whose glyph appears in the query band is a match: recolor that cell's glyph
      pixels to green while leaving its background teal.
    - Every other table cell is a non-match: recolor its background pixels to red while leaving its
      glyph color unchanged.
    - If the matched cells occupy one complete table row or one complete table column and all four
      query slots are used, also green-fill the query-band background and the bottom verdict band.
      Otherwise keep the query band unchanged and paint the verdict band red.

    Key insight: The recognition step is the same exact query-to-table lookup as the base task, but
    the output now marks both sides of that partition at once: green foreground for hits, red
    background for misses.
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
            matched = _cell_signature(g, r, c, cell_size, bg) in query_set
            if matched:
                selected.append((int(ri), int(ci)))
            cell = out[r : r + cell_size, c : c + cell_size]
            _mark_table_cell(cell, bg, matched)

    complete_query = len(query_sigs) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_sigs) and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = GREEN
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
BG = 8
GREEN = 3
RED = 2
SEP = 6
GLYPH_COLORS = (0, 1, 4, 5, 6, 7, 9)

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
    """Assign 16 colors so each allowed glyph color appears 1-3 times."""
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


def _glyph(mask: int, color: int) -> np.ndarray:
    """Render a 3x3 bitmask as a 5x5 cell with a 1-pixel teal border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _line_cells() -> list[tuple[int, int]]:
    """Pick a complete row or column of 4 cells in shuffled order."""
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    """Pick `count` distinct cells that do not form a single full row or column."""
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


def _expected_output(
    grid: np.ndarray, cells: list[tuple[int, int]], success: bool
) -> np.ndarray:
    """Build the combo output by marking matches and misses throughout the table."""
    out = grid.copy()
    selected = set(cells)
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            _mark_table_cell(cell, BG, (r_i, c_i) in selected)
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
    raise RuntimeError("failed to generate a valid d8e07eb2 combo D4/0+D14/1 input")
