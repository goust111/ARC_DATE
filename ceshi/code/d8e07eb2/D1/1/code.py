# Variant of d8e07eb2
# Dimension: D1 — 字形匹配条件 (match_cond)
# Change: 查询字形s与参考表字形d必须同时满足颜色相同且3x3二值形状相同才算匹配... → 仅形状匹配（忽略颜色）

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D1 variant
# ----------------------------------
# variant_spec:
#   preserved_layout:
#     grid: 30x22
#     bands:
#       - rows 0..4: query cells in table-aligned 5x5 slots
#       - row 5: separator color 6
#       - rows 7..26: 4x4 table of 5x5 cells
#       - row 27: separator color 6
#       - rows 28..29: verdict band
#     cell_geometry:
#       - every visible glyph stays a centered 3x3 pattern inside a 5x5 cell
#       - highlight still means "replace background pixels inside a selected cell by green"
#   core_change:
#     original_match_unit: (shape, color)
#     new_match_unit: binary 3x3 occupancy pattern only
#     visual consequence:
#       - several table cells now intentionally share the same 3x3 shape while using different colors
#       - the query can show one colored exemplar, but the output may highlight another cell with the same shape in a different color
#   behavior_trace:
#     original:
#       - query contains red shape A
#       - table has red A and blue A
#       - old solver selects only red A because color is part of the signature
#     variant:
#       - the same query red A selects both red A and blue A
#       - if the union of all same-shaped hits forms exactly one table row/column, verdict is success
#     generator_realization:
#       - each of 8 shape families appears exactly twice in the table
#       - the two copies always use different colors
#       - success examples pick two shape families whose four member cells cover one full row or one full column
#       - query slots are filled by repeating one exemplar from each chosen family, so at least one matched cell is cross-color and not an exact query copy
#   complexity_calibration:
#     why_not_unique_shapes:
#       - if all 16 table shapes remained unique, ignoring color would have no visible effect and the variant would collapse to the original task
#     why_pairs_not_arbitrary_many:
#       - pairing each shape exactly twice keeps the search harder than the original task but still visually parsable from 2-4 examples
#       - larger multiplicities would create too many highlighted cells and bury the row/column logic
#     query_multiplicity:
#       - duplicate query glyphs are now allowed because repeating a shape exemplar is the cleanest way to keep four filled slots while still omitting one color-variant partner from the query
#   implicit_assumption_audit:
#     original_solver_dependency:
#       - _cell_signature stored pixel colors, so equality meant exact color+shape identity
#       - query uniqueness assertion assumed each useful query glyph must be different under that exact identity
#       - success check len(selected) == len(query_sigs) assumed one query glyph can hit at most one table cell
#     variant_fix:
#       - signatures now store only foreground coordinates
#       - the duplicate-query assertion is removed
#       - success instead requires four occupied query slots and exactly four matched table cells in one row or one column
#   placement_legality:
#     table_pairs:
#       - cells are still placed only at ROWS x COLS anchors, so no overlap or clipping occurs
#       - horizontal pairing yields row-based successes; vertical pairing yields column-based successes
#     query_cells:
#       - query glyphs are pasted only into the first count slots at row 0
#       - empty trailing slots remain plain background, matching the original physical layout
#   observability:
#     human_cue:
#       - outputs highlight extra table cells that share shape but not color with the query
#       - because every queried family has a hidden partner of another color, the new rule changes visible output structure on essentially every sample
#     prohibited_shortcuts:
#       - no ordering or branching depends on color ids themselves
#       - colors act only as distractors; the logical grouping is entirely visible through repeated 3x3 silhouettes
#   cleanliness:
#     - only one layer of information occupies each region: glyph ink inside cells, green background highlight, and verdict band fill
#     - the repeated-shape families are sparse enough that highlighted cells remain easy to count and align by eye
#   generator_guarantees_for_main:
#     - each query family has exactly one repeated exemplar in the query but two differently colored occurrences in the table
#     - every success sample has 4 occupied query slots, 2 unique queried shapes, and 4 matched table cells
#     - every fail sample either has 3 query slots or a matched set whose size/pattern is not a full row/column


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyph shapes against a reference table while ignoring colors.

    Input structure:
    - Teal canvas with two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells holding centered 3x3 glyphs.
    - Between the separators is a 4x4 table of the same cell size; different
      colors may reuse the same 3x3 silhouette.

    Rule:
    - Treat each glyph as a binary 3x3 shape and ignore its paint color.
    - Green-fill every table cell whose shape appears anywhere in the query band.
    - If those matched cells are exactly one full table row or one full table
      column and all four query slots are occupied, also green-fill the query
      band and the bottom verdict band; otherwise fill only the verdict band red.

    Key insight: The query glyphs are shape exemplars, not exact color keys.
    Extra matches can appear in differently colored cells that share the same shape.
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
            if _cell_shape_signature(g, r, c, cell_size, bg) not in query_shape_set:
                continue
            selected.append((int(ri), int(ci)))
            cell = out[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = 3

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
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)

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


def _color_pairs() -> list[tuple[int, int]]:
    base = list(GLYPH_COLORS)
    perm = base.copy()
    while any(a == b for a, b in zip(base, perm)):
        random.shuffle(perm)
    pairs = list(zip(base, perm, strict=True))
    random.shuffle(pairs)
    return [(int(a), int(b)) for a, b in pairs]


def _glyph(mask: int, color: int) -> np.ndarray:
    """Render a 3x3 bitmask as a CELL x CELL array with 1-pixel border of BG."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _pair_layout() -> tuple[str, list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    """Return pair families and the 2-family lines they can exactly cover."""
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

    return orientation, families, success_lines


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
    assert not (len(matched_cells) == TABLE_DIM and (
        len({r for r, _c in matched_cells}) == 1 or len({c for _r, c in matched_cells}) == 1
    ))
    return _query_sources_for_families(families, family_ids), matched_cells


def _expected_output(
    grid: np.ndarray, matched_cells: list[tuple[int, int]], success: bool
) -> np.ndarray:
    out = grid.copy()
    for r_i, c_i in matched_cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _exact_signature(cell: np.ndarray) -> tuple[tuple[int, int, int], ...] | None:
    mask = cell != BG
    if not bool(mask.any()):
        return None
    return tuple((int(r), int(c), int(cell[r, c])) for r, c in zip(*np.where(mask), strict=False))


def _cross_color_extra_matches(
    grid: np.ndarray, query_sources: list[tuple[int, int]], matched_cells: list[tuple[int, int]]
) -> int:
    query_exact = {
        _exact_signature(grid[ROWS[r] : ROWS[r] + CELL, COLS[c] : COLS[c] + CELL])
        for r, c in query_sources
    }
    matched_exact = [
        _exact_signature(grid[ROWS[r] : ROWS[r] + CELL, COLS[c] : COLS[c] + CELL])
        for r, c in matched_cells
    ]
    return sum(sig not in query_exact for sig in matched_exact)


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    _orientation, families, success_lines = _pair_layout()
    masks = _shape_masks()
    color_pairs = _color_pairs()
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

    assert _cross_color_extra_matches(grid, query_sources, matched_cells) > 0

    expected = _expected_output(grid, matched_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 D1 variant input")
