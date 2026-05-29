# Combo variant of d8e07eb2 (index 28)
# combo_spec: D4/3+D11/0+D14/0
# Name: 擦除匹配字形+擦除未匹配+裁剪输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for combo D4/3 + D11/0 + D14/0
#
# a) concretization_and_behavior_trace:
#   input stays identical to the base task:
#     - 30x22 teal canvas.
#     - row 5 and row 27 are full-width separator rows.
#     - rows 7,12,17,22 and cols 1,6,11,16 define a 4x4 table of 5x5 cells.
#     - each 5x5 cell contains a centered 3x3 monochrome glyph.
#     - the query band holds 3 or 4 exact copies sampled from the table.
#   baseline output:
#     - matched table cells keep their colored glyphs and get green background.
#     - unmatched table cells remain unchanged.
#     - success/fail is additionally shown outside the table.
#   combo output:
#     - matched cell: erase the glyph itself and leave a solid green 5x5 block.
#     - unmatched cell: erase the glyph and leave a solid teal 5x5 block.
#     - crop away everything except the 20x20 table area.
#   concrete example:
#     - if queries hit cells {(1,0), (1,1), (1,2), (1,3)}, the baseline shows a
#       colored row of four glyphs on green backgrounds and also colors the query
#       and verdict bands.
#     - the combo instead returns only the table crop, where those four cells are
#       plain green squares and the remaining 12 cells are plain teal squares.
#     - if queries hit {(0,0), (1,2), (3,1)}, the combo output is still only the
#       crop, now with three isolated green squares and 13 teal squares.
#
# b) complexity_calibration:
#   - Matching still depends on the original color+shape signatures; the task does
#     not collapse into a trivial count or positional lookup.
#   - The output is visually stronger than either single erase change alone because
#     every table cell becomes one of exactly two states: solid green or solid teal.
#   - The generator preserves both 3-query and 4-query failures plus row/column
#     successes, so examples still exhibit multiple match configurations.
#
# c) implicit_assumption_review:
#   - `_constant_non_bg_rows` still assumes the separator rows are the only full-
#     width constant non-background rows. Generator keeps that invariant.
#   - `_cell_signature` must read from the input grid, not the evolving output,
#     because the combo erases every table glyph in the output.
#   - Table-origin inference still relies on each 5x5 cell having a 1-pixel border
#     around a centered 3x3 glyph. Generator keeps the original placement.
#   - Success logic still depends on exact query cardinality and alignment, even
#     though success is no longer painted outside the table. This is fine because
#     a full green row/column remains directly visible inside the crop.
#   - New assumption: the final output contains only the contiguous 20x20 table.
#     Generator uses the fixed original table geometry, so this crop is safe.
#
# d) placement_legality:
#   - Input placement remains the original fixed layout, so glyph cells do not
#     overlap and stay within bounds.
#   - Output placement is simpler than the original: a direct 20x20 crop of the
#     reference table region, with no extra badge or overlay.
#
# e) visual_observability:
#   - Across 2-4 examples, the rule is legible: all colored glyph detail vanishes,
#     and only a green/teal occupancy pattern remains.
#   - Success cases visibly become one full green row or one full green column.
#   - Failure cases visibly become 3 or 4 non-line green cells.
#
# f) visual_cleanliness:
#   - Cropping removes the noisy query/verdict bands.
#   - Erasing both matched and unmatched glyphs prevents layered symbols or mixed
#     overlays; each cell is a single flat color.
#
# g) change_inferability:
#   - The output uses only two meaningful colors in the crop: green for matched and
#     teal for not matched. No hidden color remapping is needed.
#   - Because the green cells occupy the exact table lattice from the input, the
#     viewer can infer that the preserved structure is the reference table itself,
#     not an unrelated abstract bitmap.
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


def _table_geometry(g: np.ndarray, bg: int, cell_size: int) -> tuple[list[int], list[int]]:
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
    return row_starts, col_starts


def _combo_output_from_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5
    row_starts, col_starts = _table_geometry(g, bg, cell_size)

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    table_h = len(row_starts) * cell_size
    table_w = len(col_starts) * cell_size
    out = np.full((table_h, table_w), int(bg), dtype=int)

    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            if _cell_signature(g, r, c, cell_size, bg) not in query_set:
                continue
            rr = ri * cell_size
            cc = ci * cell_size
            out[rr : rr + cell_size, cc : cc + cell_size] = 3
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyphs against the reference table, erase all glyph detail, and keep only the cropped table.

    Input structure:
    - The input keeps the original 30x22 layout: a top query band, two full-width separator rows,
      and a middle 4x4 table of 5x5 cells on a teal background.
    - Every glyph is a centered 3x3 monochrome pattern, and a table match is defined by the exact
      pair of shape and color copied into the query band.

    Rule:
    - Find the table cells whose glyphs appear in the query band.
    - Replace each matched 5x5 table cell with a solid green block, and replace each unmatched 5x5
      table cell with a solid teal block by erasing its glyph.
    - Discard the query band, separators, spacer row, and verdict band; output only the 20x20
      reference-table crop.

    Key insight: The lookup logic is unchanged, but the output is no longer an annotated scene.
    It is a binary occupancy map of the table, where green cells mark matched signatures and all
    other cells are blanked away.
    """
    return _combo_output_from_grid(grid)


# --------------- generator (combo variant) ---------------

# -- Grid layout --
GRID_H, GRID_W = 30, 22
TOP_SEP_ROW = 5
BOTTOM_SEP_ROW = 27

# -- Colors --
BG = 8
GREEN = 3
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)
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
    """True iff the 3x3 bitmask touches top, bottom, left, and right edges."""
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


def _expected_output(cells: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((TABLE_H, TABLE_W), BG, dtype=int)
    for r_i, c_i in cells:
        rr = r_i * CELL
        cc = c_i * CELL
        out[rr : rr + CELL, cc : cc + CELL] = GREEN
    return out


def _combo_metrics(grid: np.ndarray, out: np.ndarray, cells: list[tuple[int, int]]) -> dict[str, int]:
    selected = set(cells)
    matched_green_blocks = 0
    erased_matched_glyph_cells = 0
    erased_unmatched_glyph_cells = 0
    colored_pixels_remaining = 0

    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            in_cell = grid[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            out_cell = out[r_i * CELL : (r_i + 1) * CELL, c_i * CELL : (c_i + 1) * CELL]
            had_glyph = bool(np.any(in_cell != BG))
            if (r_i, c_i) in selected:
                if bool(np.all(out_cell == GREEN)):
                    matched_green_blocks += 1
                if had_glyph and bool(np.all(out_cell == GREEN)):
                    erased_matched_glyph_cells += 1
            else:
                if had_glyph and bool(np.all(out_cell == BG)):
                    erased_unmatched_glyph_cells += 1
            colored_pixels_remaining += int(np.count_nonzero((out_cell != BG) & (out_cell != GREEN)))

    return {
        "matched_green_blocks": matched_green_blocks,
        "erased_matched_glyph_cells": erased_matched_glyph_cells,
        "erased_unmatched_glyph_cells": erased_unmatched_glyph_cells,
        "colored_pixels_remaining": colored_pixels_remaining,
    }


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

    expected = _expected_output(query_cells)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    metrics = _combo_metrics(grid, actual, query_cells)
    if actual.shape != (TABLE_H, TABLE_W):
        return None
    if metrics["matched_green_blocks"] != len(query_cells):
        return None
    if metrics["erased_matched_glyph_cells"] != len(query_cells):
        return None
    if metrics["erased_unmatched_glyph_cells"] != TABLE_DIM * TABLE_DIM - len(query_cells):
        return None
    if metrics["colored_pixels_remaining"] != 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
