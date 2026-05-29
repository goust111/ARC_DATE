# Combo variant of d8e07eb2 (index 1)
# combo_spec: D1/1+D5/2
# Name: 形状匹配+对角线判定

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
#    - Preserved canvas: 30x22 grid, 5-row query band, separator row, 4x4 table
#      of 5x5 cells, separator row, and 2-row verdict band.
#    - Original task:
#        query glyph -> exact (shape,color) lookup -> matched cells must be one
#        full row/column to turn the query/verdict bands green.
#    - Combo task:
#        query glyph -> shape-only lookup -> every differently colored twin of
#        the same 3x3 silhouette also matches -> the union of matched cells must
#        be exactly the main diagonal or anti-diagonal.
#    - Small trace on a 4x4 table:
#        family A occupies (0,0) and (1,1) in two colors
#        family B occupies (2,2) and (3,3) in two colors
#        query slots show one exemplar of A twice and one exemplar of B twice
#      Exact matching would only recover the two copied source cells. The combo
#      solver instead expands them to four cells {(0,0),(1,1),(2,2),(3,3)} and
#      accepts because that set is a full diagonal.
#
# b) Complexity calibration
#    - Using 8 shape families with multiplicity exactly 2 keeps the shape-only
#      change visible on every sample without flooding the table with too many
#      highlights.
#    - Successes use exactly 2 families repeated across all 4 query slots, so
#      the matched union stays at 4 cells and can cleanly realize a diagonal.
#    - Fails use 2-4 families, producing 4/6/8 highlighted cells that are still
#      sparse enough to inspect by eye.
#
# c) Implicit assumption audit against the original solver
#    - The original `_cell_signature` encoded colors, so equality meant exact
#      color+shape identity. That assumption is invalid here: signatures must be
#      binary occupancy only.
#    - The original `assert len(query_sigs) == len(query_set)` is also invalid:
#      duplicate query glyphs are now intentional because repeating a family
#      exemplar is the cleanest way to occupy all 4 query slots.
#    - `len(selected) == len(query_sigs)` no longer characterizes success,
#      because one queried shape can match two table cells.
#    - `same_row` / `same_col` must be replaced by explicit diagonal-set tests.
#
# d) Placement legality
#    - Table cells still occupy the fixed ROWS x COLS anchors, so no glyph ever
#      overlaps or clips.
#    - The 8 families partition the 16 table cells exactly; each cell receives
#      one centered 3x3 glyph.
#    - Query glyphs are pasted only into the first 3 or 4 top-band slots, so the
#      trailing slot, when empty, remains plain background.
#
# e) Visual observability
#    - Every queried family has a hidden differently colored partner in the table,
#      so the output always reveals that color is not part of the lookup key.
#    - Success examples visibly highlight one full diagonal, while fail examples
#      highlight non-diagonal unions made from the same kind of cross-color
#      family expansion.
#    - Because success diagonals would fail under the old row/column rule, the
#      D5 change affects the verdict on the same inputs rather than merely being
#      a relabeling of unchanged cases.
#
# f) Visual cleanliness
#    - Highlights only recolor background pixels inside matched cells, so the
#      glyph ink remains readable after marking.
#    - Pair multiplicity is fixed at 2, which prevents dense overlapping cues and
#      keeps the table geometry easy to count.
#
# g) Inference audit
#    - No branch depends on numeric color ids; colors are pure distractors.
#    - Generator guarantees two observable combo signals:
#        1. shape-only lookup adds at least one extra table hit beyond the exact
#           query copies;
#        2. a substantial fraction of samples are diagonal successes, so the
#           diagonal verdict differs from the old row/column verdict.
#
# h) New assumptions introduced by `main()`
#    - The solver assumes the reference table is square before diagonal testing;
#      the generator always constructs a 4x4 table, satisfying that precondition.
#    - The solver assumes query/table glyphs are centered 3x3 patterns inside
#      5x5 cells. The generator preserves that geometry exactly.
#    - The solver does not assume family multiplicity 2, but the generator uses
#      it intentionally to keep the task visually crisp and the success set size
#      controllable.


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


def _diagonal_cell_sets(dim: int) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    main_diag = frozenset((i, i) for i in range(dim))
    anti_diag = frozenset((i, dim - 1 - i) for i in range(dim))
    return main_diag, anti_diag


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyph silhouettes against a reference table and judge diagonals.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a row of aligned 5x5 query cells containing
      centered 3x3 glyphs; some slots may repeat the same silhouette.
    - Between the separators is a square 4x4 reference table of the same 5x5
      cells, and multiple differently colored cells can share one silhouette.

    Rule:
    - Ignore glyph color and compare only the binary 3x3 shape.
    - Green-fill the background inside every table cell whose shape appears in
      the query band.
    - If all four query slots are occupied and the matched table cells are
      exactly one full main diagonal or one full anti-diagonal, also green-fill
      the query band and the bottom verdict band.
    - Otherwise leave the query band unchanged and fill only the verdict band red.

    Key insight: The top band supplies shape exemplars rather than exact color
    keys. The verdict depends on the geometry of the full shape-matched UNION,
    and that union only passes when it traces a complete diagonal.
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
    square_table = len(row_starts) == len(col_starts)
    if square_table:
        is_complete_diagonal = frozenset(selected) in _diagonal_cell_sets(len(row_starts))
    else:
        is_complete_diagonal = False
    success = complete_query and len(selected) == len(col_starts) and is_complete_diagonal

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
P_SUCCESS = 0.45
P_QUERY_3 = 0.35
FAIL_MODES = ("two_family_non_diag", "three_family", "four_family")

# -- Generation --
MAX_RETRIES = 300

BASE_FAMILIES = (
    ((0, 0), (1, 1)),
    ((2, 2), (3, 3)),
    ((0, 3), (1, 2)),
    ((2, 1), (3, 0)),
    ((0, 1), (1, 0)),
    ((0, 2), (1, 3)),
    ((2, 0), (3, 1)),
    ((2, 3), (3, 2)),
)
SUCCESS_FAMILY_PAIRS = ((0, 1), (2, 3))
BOARD_SYMMETRIES = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "flip_main", "flip_anti")


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
    """Render a 3x3 bitmask as a CELL x CELL array with a 1-cell BG border."""
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _transform_coord(r: int, c: int, kind: str) -> tuple[int, int]:
    n = TABLE_DIM - 1
    if kind == "id":
        return r, c
    if kind == "rot90":
        return c, n - r
    if kind == "rot180":
        return n - r, n - c
    if kind == "rot270":
        return n - c, r
    if kind == "flip_h":
        return r, n - c
    if kind == "flip_v":
        return n - r, c
    if kind == "flip_main":
        return c, r
    if kind == "flip_anti":
        return n - c, n - r
    raise ValueError(f"unknown symmetry: {kind}")


def _diagonal_family_layout() -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    kind = random.choice(BOARD_SYMMETRIES)
    families = [
        tuple(_transform_coord(r, c, kind) for r, c in family)
        for family in BASE_FAMILIES
    ]
    return families, list(SUCCESS_FAMILY_PAIRS)


def _matched_cells_from_families(
    families: list[tuple[tuple[int, int], tuple[int, int]]], family_ids: list[int]
) -> list[tuple[int, int]]:
    cells = {
        (int(r), int(c))
        for family_id in set(family_ids)
        for r, c in families[family_id]
    }
    return sorted(cells)


def _query_sources_for_families(
    families: list[tuple[tuple[int, int], tuple[int, int]]], family_ids: list[int]
) -> list[tuple[int, int]]:
    exemplar_by_family = {
        family_id: random.choice(families[family_id]) for family_id in set(family_ids)
    }
    sources = [exemplar_by_family[family_id] for family_id in family_ids]
    random.shuffle(sources)
    return [(int(r), int(c)) for r, c in sources]


def _is_full_diagonal(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != TABLE_DIM:
        return False
    return frozenset(cells) in _diagonal_cell_sets(TABLE_DIM)


def _success_query(
    families: list[tuple[tuple[int, int], tuple[int, int]]], success_pairs: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    family_a, family_b = random.choice(success_pairs)
    family_ids = [family_a, family_a, family_b, family_b]
    return _query_sources_for_families(families, family_ids), _matched_cells_from_families(
        families, family_ids
    )


def _non_diagonal_family_pair(
    families: list[tuple[tuple[int, int], tuple[int, int]]], success_pairs: list[tuple[int, int]]
) -> tuple[int, int]:
    blocked = {frozenset(pair) for pair in success_pairs}
    choices = [
        pair
        for pair in itertools.combinations(range(len(families)), 2)
        if frozenset(pair) not in blocked
        and not _is_full_diagonal(_matched_cells_from_families(families, list(pair)))
    ]
    return tuple(random.choice(choices))


def _fail_query(
    families: list[tuple[tuple[int, int], tuple[int, int]]],
    success_pairs: list[tuple[int, int]],
    count: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if count == 3:
        if random.random() < 0.5:
            family_a, family_b = _non_diagonal_family_pair(families, success_pairs)
            family_ids = [family_a, family_a, family_b]
        else:
            family_ids = random.sample(range(len(families)), 3)
    else:
        mode = random.choice(FAIL_MODES)
        if mode == "two_family_non_diag":
            family_a, family_b = _non_diagonal_family_pair(families, success_pairs)
            family_ids = [family_a, family_a, family_b, family_b]
        elif mode == "three_family":
            family_a, family_b, family_c = random.sample(range(len(families)), 3)
            family_ids = [family_a, family_a, family_b, family_c]
        else:
            family_ids = random.sample(range(len(families)), 4)

    matched_cells = _matched_cells_from_families(families, family_ids)
    assert not _is_full_diagonal(matched_cells)
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

    families, success_pairs = _diagonal_family_layout()
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
        query_sources, matched_cells = _success_query(families, success_pairs)
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_sources, matched_cells = _fail_query(families, success_pairs, count)

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
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
