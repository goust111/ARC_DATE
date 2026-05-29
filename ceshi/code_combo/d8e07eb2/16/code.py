# Combo variant of d8e07eb2 (index 16)
# combo_spec: D1/1+D14/2
# Name: 形状匹配+未匹配变灰

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# combo_summary:
#   preserved_layout:
#     - keep the original 30x22 canvas, the two full-width separator rows, the
#       4x4 reference table of 5x5 cells, and the top query band with up to 4
#       table-aligned query cells.
#     - every visible glyph still occupies the centered inner 3x3 of its 5x5 cell.
#   fused_rule:
#     - matching is by binary 3x3 silhouette only; glyph color is ignored.
#     - every matched table cell keeps its original glyph color and gets green
#       background fill inside the cell.
#     - every unmatched table cell keeps its shape but all non-background glyph
#       pixels are recolored to gray 5.
#   behavior_trace:
#     - original task:
#       query shows red shape A and blue shape B.
#       table contains red A, yellow A, blue B, orange C.
#       only exact color+shape copies count, so only red A and blue B match;
#       yellow A stays untouched because its color differs.
#     - combo task:
#       the same query matches both red A and yellow A because they share the
#       same occupancy pattern; blue B also matches.
#       matched cells keep their original colors and gain green backgrounds.
#       orange C is not matched, so its glyph remains in place but turns gray.
#     - visible difference:
#       the solver must both expand matches across colors and simultaneously
#       desaturate all non-matches, creating a table where color means either
#       "preserved because matched" or "discarded because unmatched".
#   complexity_calibration:
#     - if all 16 table shapes were unique, shape-only matching would collapse to
#       the original behavior, so the generator makes exactly 8 shape families
#       with 2 cells per family.
#     - if gray were allowed as an input glyph color, the recoloring effect could
#       disappear on some unmatched cells; therefore generator colors exclude 5.
#     - family size stays at 2 so the number of extra matches remains readable:
#       queried families can add one hidden cross-color partner, but not a large
#       swarm of matches that obscures the row/column logic.
#   implicit_assumption_audit:
#     - the original _cell_signature encoded colors, so equality implied exact
#       color+shape identity; this must become a shape-only signature.
#     - the original distinct-query assertion no longer holds because repeated
#       query exemplars are useful for requesting the same shape family twice.
#     - the original success test len(selected) == len(query_sigs) breaks once a
#       single query shape can hit two table cells; success must instead require
#       4 occupied query slots and exactly 4 matched table cells in one row/col.
#     - output mutation must never affect later matching, so all signatures are
#       read from the immutable input grid rather than from the evolving output.
#     - gray recoloring assumes unmatched cells are processed cell-by-cell; the
#       generator guarantees a full 4x4 table so every loop iteration sees a cell.
#   placement_legality:
#     - reference cells are only placed at the fixed ROWS x COLS anchors, so no
#       overlap or clipping is possible.
#     - each query slot uses the same 5x5 width as a table column and only the
#       leading 3 or 4 slots are populated, leaving any trailing slot empty.
#     - success layouts come from two shape families whose four table locations
#       exactly cover one row or one column; fail layouts are filtered so the
#       union of matched cells is never such a full line.
#   visual_observability:
#     - across examples, the solver highlights extra cross-color matches that are
#       not exact copies of the query, making the shape-only rule visible.
#     - at the same time, every untouched family fades to gray, so the "matched
#       keeps color / unmatched loses color" contrast is obvious in a few samples.
#     - no logic depends on numeric ordering of colors; humans can infer the rule
#       from repeated silhouettes and from the consistent graying of non-matches.
#   cleanliness:
#     - each table cell shows exactly one of two states: green-backed original
#       glyph if matched, or gray glyph on teal if unmatched.
#     - the top band and verdict band still use the original success/fail cues,
#       so the added gray transform does not clutter those regions.
#   generator_guarantees_for_main:
#     - every sample has a valid inferred 4x4 table grid between the separators.
#     - every queried shape family appears twice in the table with different
#       non-gray colors, ensuring at least one cross-color extra match.
#     - every sample has at least one unmatched table cell, so gray recoloring is
#       always observable and the combo does not degenerate to D1 alone.


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _cell_shape_signature(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[tuple[int, int], ...] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    return tuple((int(r), int(c)) for r, c in zip(*np.where(mask), strict=False))


def _cell_exact_signature(
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
    Match query silhouettes against the reference table, then gray out every non-match.

    Input structure:
    - A teal canvas is split by two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells with centered 3x3 glyphs.
    - Between the separators is a 4x4 table of the same cell size; some cells may
      share the same 3x3 silhouette even when their colors differ.

    Rule:
    - Compare glyphs by shape only, ignoring color, and green-fill every table cell
      whose silhouette appears in the query band while preserving that cell's glyph color.
    - In every unmatched table cell, keep the glyph shape but recolor all glyph pixels
      to gray 5.
    - If the matched cells are exactly one full table row or one full table column and
      all four query slots are occupied, also green-fill the query band and the verdict
      band; otherwise leave the query band as is and fill only the verdict band red.

    Key insight: Color is a distractor for matching but not for display. The correct
    output is found by grouping cells with the same silhouette, then preserving color
    only in the matched groups while every unmatched glyph is visibly faded to gray.
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

    query_shapes = [
        sig
        for c in col_starts
        if (sig := _cell_shape_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_shape_set = set(query_shapes)

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            shape_sig = _cell_shape_signature(g, r, c, cell_size, bg)
            cell = out[r : r + cell_size, c : c + cell_size]
            if shape_sig in query_shape_set:
                selected.append((int(ri), int(ci)))
                cell[cell == bg] = 3
            else:
                cell[(cell != bg)] = 5

    complete_query = len(query_shapes) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(col_starts) and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator ---------------

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
GLYPH_COLORS = (0, 1, 2, 4, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_PAIR_COUNTS = {5: 2, 6: 2, 7: 2, 8: 1, 9: 1}

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("two_family_non_line", "three_family", "four_family")

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


def _shape_masks() -> list[int]:
    masks: list[int] = []
    for fill_count, n in SHAPE_PAIR_COUNTS.items():
        masks.extend(random.sample(PATTERNS_BY_FILL[fill_count], n))
    random.shuffle(masks)
    return masks


def _family_color_pairs() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for _ in range(TABLE_DIM * 2):
        color_a, color_b = random.sample(GLYPH_COLORS, 2)
        pairs.append((int(color_a), int(color_b)))
    return pairs


def _glyph(mask: int, color: int) -> np.ndarray:
    """Render a 3x3 bitmask as a 5x5 cell with a 1-pixel teal border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _pair_layout() -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    """Return 8 shape families and the 2-family lines they can exactly cover."""
    orientation = random.choice(("horizontal", "vertical"))
    families: list[tuple[tuple[int, int], tuple[int, int]]] = []
    success_lines: list[tuple[int, int]] = []

    if orientation == "horizontal":
        col_order = random.sample(range(TABLE_DIM), TABLE_DIM)
        for r in range(TABLE_DIM):
            idx0 = len(families)
            families.append(((r, col_order[0]), (r, col_order[1])))
            idx1 = len(families)
            families.append(((r, col_order[2]), (r, col_order[3])))
            success_lines.append((idx0, idx1))
    else:
        row_order = random.sample(range(TABLE_DIM), TABLE_DIM)
        for c in range(TABLE_DIM):
            idx0 = len(families)
            families.append(((row_order[0], c), (row_order[1], c)))
            idx1 = len(families)
            families.append(((row_order[2], c), (row_order[3], c)))
            success_lines.append((idx0, idx1))

    return families, success_lines


def _query_sources_for_families(
    families: list[tuple[tuple[int, int], tuple[int, int]]], family_ids: list[int]
) -> list[tuple[int, int]]:
    exemplar_by_family = {
        family_id: random.choice(families[family_id]) for family_id in set(family_ids)
    }
    sources = [exemplar_by_family[family_id] for family_id in family_ids]
    random.shuffle(sources)
    return [(int(r), int(c)) for r, c in sources]


def _matched_cells_from_families(
    families: list[tuple[tuple[int, int], tuple[int, int]]], family_ids: list[int]
) -> list[tuple[int, int]]:
    cells = {
        (int(r), int(c))
        for family_id in set(family_ids)
        for r, c in families[family_id]
    }
    return sorted(cells)


def _success_query(
    families: list[tuple[tuple[int, int], tuple[int, int]]], success_lines: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    family_a, family_b = random.choice(success_lines)
    family_ids = [family_a, family_a, family_b, family_b]
    return _query_sources_for_families(families, family_ids), _matched_cells_from_families(
        families, family_ids
    )


def _non_line_family_pair(success_lines: list[tuple[int, int]]) -> tuple[int, int]:
    blocked = {frozenset(line) for line in success_lines}
    choices = [
        pair
        for pair in itertools.combinations(range(TABLE_DIM * 2), 2)
        if frozenset(pair) not in blocked
    ]
    return tuple(random.choice(choices))


def _fail_query(
    families: list[tuple[tuple[int, int], tuple[int, int]]],
    success_lines: list[tuple[int, int]],
    count: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if count == 3:
        if random.random() < 0.5:
            family_a, family_b = _non_line_family_pair(success_lines)
            family_ids = [family_a, family_a, family_b]
        else:
            family_ids = random.sample(range(TABLE_DIM * 2), 3)
    else:
        mode = random.choice(FAIL_MODES)
        if mode == "two_family_non_line":
            family_a, family_b = _non_line_family_pair(success_lines)
            family_ids = [family_a, family_a, family_b, family_b]
        elif mode == "three_family":
            family_a, family_b, family_c = random.sample(range(TABLE_DIM * 2), 3)
            family_ids = [family_a, family_a, family_b, family_c]
        else:
            family_ids = random.sample(range(TABLE_DIM * 2), 4)

    matched_cells = _matched_cells_from_families(families, family_ids)
    assert len(matched_cells) < TABLE_DIM or (
        len({r for r, _c in matched_cells}) > 1 and len({c for _r, c in matched_cells}) > 1
    )
    assert len(matched_cells) < TABLE_DIM or not (
        len({r for r, _c in matched_cells}) == 1 or len({c for _r, c in matched_cells}) == 1
    )
    return _query_sources_for_families(families, family_ids), matched_cells


def _expected_output(
    grid: np.ndarray, matched_cells: list[tuple[int, int]], success: bool
) -> np.ndarray:
    out = grid.copy()
    matched_set = {(int(r), int(c)) for r, c in matched_cells}
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            if (r_i, c_i) in matched_set:
                cell[cell == BG] = GREEN
            else:
                cell[cell != BG] = GRAY
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _cross_color_extra_matches(
    grid: np.ndarray, query_sources: list[tuple[int, int]], matched_cells: list[tuple[int, int]]
) -> int:
    query_exact = {
        _cell_exact_signature(grid, ROWS[r], COLS[c], CELL, BG) for r, c in query_sources
    }
    matched_exact = [
        _cell_exact_signature(grid, ROWS[r], COLS[c], CELL, BG) for r, c in matched_cells
    ]
    return sum(sig not in query_exact for sig in matched_exact)


def _gray_changed_unmatched_count(grid: np.ndarray, out: np.ndarray, matched_cells: list[tuple[int, int]]) -> int:
    matched_set = {(int(r), int(c)) for r, c in matched_cells}
    changed = 0
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            if (r_i, c_i) in matched_set:
                continue
            inp = grid[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            if np.array_equal(inp, cell):
                continue
            assert np.all(cell[inp != BG] == GRAY)
            assert np.all(cell[inp == BG] == BG)
            changed += 1
    return changed


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    families, success_lines = _pair_layout()
    masks = _shape_masks()
    color_pairs = _family_color_pairs()
    glyphs_by_cell: dict[tuple[int, int], np.ndarray] = {}

    for family_id, (mask, (color_a, color_b)) in enumerate(zip(masks, color_pairs, strict=True)):
        glyph_a = _glyph(mask, color_a)
        glyph_b = _glyph(mask, color_b)
        (r0, c0), (r1, c1) = families[family_id]
        glyphs_by_cell[(r0, c0)] = glyph_a
        glyphs_by_cell[(r1, c1)] = glyph_b
        paste(grid, glyph_a, ROWS[r0], COLS[c0])
        paste(grid, glyph_b, ROWS[r1], COLS[c1])

    success = random.random() < P_SUCCESS
    if success:
        query_sources, matched_cells = _success_query(families, success_lines)
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_sources, matched_cells = _fail_query(families, success_lines, count)

    for slot, (r_i, c_i) in enumerate(query_sources):
        paste(grid, glyphs_by_cell[(r_i, c_i)], 0, COLS[slot])

    out = main(grid)
    expected = _expected_output(grid, matched_cells, success)
    if not np.array_equal(out, expected):
        return None

    if _cross_color_extra_matches(grid, query_sources, matched_cells) <= 0:
        return None
    unmatched_count = TABLE_DIM * TABLE_DIM - len(matched_cells)
    if unmatched_count <= 0:
        return None
    if _gray_changed_unmatched_count(grid, out, matched_cells) != unmatched_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
