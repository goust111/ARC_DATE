# Combo variant of d8e07eb2 (index 24)
# combo_spec: D1/1+D5/2+D14/2
# Name: 形状匹配+对角线+灰色未匹配

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - The canvas layout stays the same as the source task: a 5-row query band,
#      two full-width separator rows, a 4x4 table of 5x5 cells, and a 2-row
#      verdict band.
#    - The fused change hits three stages at once:
#        1. matching ignores color and reads each glyph only by its 3x3 shape;
#        2. success is no longer row/column completeness but full main/anti
#           diagonal completeness;
#        3. every non-matched table glyph is recolored to gray in the output.
#    - Small trace:
#        input table contains shape A twice, once red and once blue.
#        query shows only the red exemplar of A.
#        original task: only the red cell would match; the blue copy would stay.
#        combo task: both A-cells match because shape is enough; if the total
#        matched positions are exactly one diagonal, query+verdict turn green,
#        and every other table glyph fades to gray.
#
# b) Complexity calibration
#    - If all 16 table shapes stayed unique, ignoring color would collapse back
#      to the original task. The generator therefore builds 8 shape families,
#      each appearing exactly twice in different colors.
#    - To keep the diagonal rule visually inferable, successes use exactly two
#      queried shape families so the matched set has exactly four cells.
#    - To make the D5 change observable instead of subtle, the fail generator
#      deliberately includes exact full-row/full-column matches under a separate
#      pairing layout; those would satisfy the old rule but must stay red here.
#    - Gray is reserved as an output-only signal by excluding color 5 from input
#      glyphs. That makes "unmatched -> gray" visible on every generated sample.
#
# c) Implicit assumption audit
#    - The original `_cell_signature` encoded color, so equality meant exact
#      color+shape identity. That assumption is invalid now; signatures must
#      store only foreground coordinates.
#    - The old query uniqueness assertion also breaks: a four-slot successful
#      query now repeats two shape exemplars as A,A,B,B so that shape families
#      can still produce exactly four matched cells.
#    - The old success check `same_row or same_col` is replaced by explicit
#      comparison against the table's two diagonals.
#    - The new graying step adds a fresh assumption: unmatched cells should be
#      distinguishable from matched cells by color, so the generator forbids
#      gray input glyphs and guarantees every queried family has a hidden
#      differently colored partner.
#
# d) Placement legality
#    - All visible objects still live on the original fixed anchors `ROWS` x
#      `COLS`; no glyph is moved off-grid and no two cells overlap.
#    - The diagonal-pair layout partitions the 16 table cells into 8 disjoint
#      pairs, allowing exact 4-cell diagonal successes.
#    - The line-pair layout also partitions the same 16 cells into 8 disjoint
#      pairs, but along rows or columns, allowing exact 4-cell row/column fails.
#    - Query glyphs are always pasted into the first 3 or 4 top-band slots, so
#      empty trailing slots remain plain background exactly as in the source.
#
# e) Visual observability
#    - A human can detect the D1 change because matched outputs frequently
#      include extra highlighted cells whose colors never appear in the query but
#      whose silhouettes do.
#    - A human can detect the D5 change because some four-match failures form a
#      perfect row/column yet still get a red verdict, while successes form one
#      of the two diagonals.
#    - A human can detect D14 because all non-selected table glyphs lose their
#      original colors and become uniform gray, creating a strong contrast with
#      the preserved-color matched cells on green backgrounds.
#
# f) Visual cleanliness
#    - The output uses only three layers per table cell: original glyph color,
#      optional green background if matched, or gray recolor if unmatched. The
#      separator rows and verdict band stay visually separate from the table.
#    - Restricting shape multiplicity to pairs keeps the highlight count small
#      enough that the diagonal or line geometry remains easy to read by eye.
#
# g) Generator guarantees for `main()`
#    - Every query shape occurs in the table at least twice with different
#      colors, so shape-only matching changes the visible matched set.
#    - Every success sample has four occupied query slots and exactly one full
#      matched diagonal.
#    - Every fail sample either has fewer than four occupied query slots or a
#      matched set that is not one of the two full diagonals.
#    - Every unmatched table glyph uses a non-gray input color, so the gray
#      recolor is always an actual transformation rather than a no-op.


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
    Match query glyph SHAPES, test whether the matched cells form a diagonal,
    and gray out every non-matched table glyph.

    Input structure:
    - A 30x22 teal canvas with two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells holding centered 3x3
      monochrome glyphs.
    - Between the separators is a 4x4 reference table of the same 5x5 cells;
      several differently colored cells may share the same 3x3 silhouette.

    Rule:
    - Ignore glyph color and compare query cells to table cells by binary 3x3
      shape only.
    - For every matched table cell, green-fill that cell's teal background while
      leaving the glyph's original color intact.
    - For every unmatched table cell, keep the shape but recolor its glyph gray.
    - If the matched cells are exactly one full main diagonal or one full
      anti-diagonal and all four query slots are occupied, also green-fill the
      query band and the verdict band; otherwise fill only the verdict band red.

    Key insight: The query supplies shape exemplars rather than exact colored
    keys. Output color now encodes status: green background means matched, gray
    glyph ink means unmatched, and the bottom band says whether the matched
    positions trace a complete diagonal.
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
            cell = out[r : r + cell_size, c : c + cell_size]
            if _cell_shape_signature(g, r, c, cell_size, bg) in query_shape_set:
                selected.append((int(ri), int(ci)))
                cell[cell == bg] = GREEN
            else:
                cell[cell != bg] = GRAY

    square_table = len(row_starts) == len(col_starts)
    complete_query = len(query_shapes) == len(col_starts)
    matched_positions = frozenset(selected)
    success = square_table and complete_query and matched_positions in _diagonal_cell_sets(len(row_starts))

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED

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
INPUT_GLYPH_COLORS = (0, 1, 2, 4, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)

# -- Glyph families --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_FAMILY_COUNTS = {5: 2, 6: 2, 7: 2, 8: 1, 9: 1}
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.45
FAIL_MODES = ("line_exact", "line_exact", "non_diag_exact", "three_family", "truncated")

# -- Generation --
MAX_RETRIES = 400


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
    for fill_count, n in SHAPE_FAMILY_COUNTS.items():
        masks.extend(random.sample(PATTERNS_BY_FILL[fill_count], n))
    random.shuffle(masks)
    return masks


def _balanced_input_color_list() -> list[int]:
    counts = {c: COLOR_MIN for c in INPUT_GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(INPUT_GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in INPUT_GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _family_color_pairs() -> list[tuple[int, int]]:
    for _ in range(200):
        counts = {c: 0 for c in INPUT_GLYPH_COLORS}
        for color in _balanced_input_color_list():
            counts[color] += 1

        pairs: list[tuple[int, int]] = []
        ok = True
        for _pair_idx in range(TABLE_DIM * TABLE_DIM // 2):
            a_choices = [c for c, n in counts.items() if n > 0]
            a = random.choice(a_choices)
            counts[a] -= 1
            b_choices = [c for c, n in counts.items() if n > 0 and c != a]
            if not b_choices:
                ok = False
                break
            b = random.choice(b_choices)
            counts[b] -= 1
            pair = (int(a), int(b)) if random.random() < 0.5 else (int(b), int(a))
            pairs.append(pair)
        if ok and len(pairs) == TABLE_DIM * TABLE_DIM // 2:
            random.shuffle(pairs)
            return pairs
    raise RuntimeError("failed to build distinct family color pairs")


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _diagonal_pair_layout() -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    families = [
        ((0, 0), (1, 1)),
        ((2, 2), (3, 3)),
        ((0, 3), (1, 2)),
        ((2, 1), (3, 0)),
        ((0, 1), (0, 2)),
        ((1, 0), (1, 3)),
        ((2, 0), (2, 3)),
        ((3, 1), (3, 2)),
    ]
    success_pairs = [(0, 1), (2, 3)]
    return families, success_pairs


def _line_pair_layout() -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    orientation = random.choice(("horizontal", "vertical"))
    families: list[tuple[tuple[int, int], tuple[int, int]]] = []
    line_pairs: list[tuple[int, int]] = []

    if orientation == "horizontal":
        col_order = random.sample(range(TABLE_DIM), TABLE_DIM)
        for r in range(TABLE_DIM):
            idx0 = len(families)
            families.append(((r, col_order[0]), (r, col_order[1])))
            idx1 = len(families)
            families.append(((r, col_order[2]), (r, col_order[3])))
            line_pairs.append((idx0, idx1))
    else:
        row_order = random.sample(range(TABLE_DIM), TABLE_DIM)
        for c in range(TABLE_DIM):
            idx0 = len(families)
            families.append(((row_order[0], c), (row_order[1], c)))
            idx1 = len(families)
            families.append(((row_order[2], c), (row_order[3], c)))
            line_pairs.append((idx0, idx1))

    return families, line_pairs


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


def _is_full_diagonal(cells: list[tuple[int, int]]) -> bool:
    return frozenset(cells) in _diagonal_cell_sets(TABLE_DIM)


def _query_from_line_target(
    families: list[tuple[tuple[int, int], tuple[int, int]]], pair_ids: tuple[int, int], count: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    family_ids = [pair_ids[0], pair_ids[0], pair_ids[1], pair_ids[1]]
    if count == 3:
        family_ids.pop(random.randrange(4))
    matched_cells = _matched_cells_from_families(families, family_ids)
    return _query_sources_for_families(families, family_ids), matched_cells


def _non_success_family_pair(success_pairs: list[tuple[int, int]]) -> tuple[int, int]:
    blocked = {frozenset(pair) for pair in success_pairs}
    choices = [
        pair
        for pair in itertools.combinations(range(TABLE_DIM * TABLE_DIM // 2), 2)
        if frozenset(pair) not in blocked
    ]
    return tuple(random.choice(choices))


def _expected_output(
    grid: np.ndarray, matched_cells: list[tuple[int, int]], success: bool
) -> np.ndarray:
    out = grid.copy()
    matched_set = set(matched_cells)
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


def _all_unmatched_inputs_non_gray(
    grid: np.ndarray, matched_cells: list[tuple[int, int]]
) -> bool:
    matched_set = set(matched_cells)
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            if (r_i, c_i) in matched_set:
                continue
            cell = grid[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            fg = cell[cell != BG]
            if not np.all(fg != GRAY):
                return False
    return True


def _success_setup() -> tuple[
    list[tuple[tuple[int, int], tuple[int, int]]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    families, success_pairs = _diagonal_pair_layout()
    family_a, family_b = random.choice(success_pairs)
    family_ids = [family_a, family_a, family_b, family_b]
    return families, _query_sources_for_families(families, family_ids), _matched_cells_from_families(
        families, family_ids
    )


def _fail_setup() -> tuple[
    list[tuple[tuple[int, int], tuple[int, int]]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    mode = random.choice(FAIL_MODES)

    if mode == "line_exact":
        families, line_pairs = _line_pair_layout()
        return families, *_query_from_line_target(families, random.choice(line_pairs), count=4)

    if mode == "truncated":
        families, line_pairs = _line_pair_layout()
        return families, *_query_from_line_target(families, random.choice(line_pairs), count=3)

    families, success_pairs = _diagonal_pair_layout()
    if mode == "non_diag_exact":
        family_a, family_b = _non_success_family_pair(success_pairs)
        family_ids = [family_a, family_a, family_b, family_b]
    else:
        family_a, family_b, family_c = random.sample(range(TABLE_DIM * TABLE_DIM // 2), 3)
        family_ids = [family_a, family_a, family_b, family_c]

    matched_cells = _matched_cells_from_families(families, family_ids)
    assert not _is_full_diagonal(matched_cells)
    return families, _query_sources_for_families(families, family_ids), matched_cells


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    success = random.random() < P_SUCCESS
    if success:
        families, query_sources, matched_cells = _success_setup()
    else:
        families, query_sources, matched_cells = _fail_setup()

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

    for slot, (r_i, c_i) in enumerate(query_sources):
        paste(grid, glyphs_by_cell[(r_i, c_i)], 0, COLS[slot])

    expected = _expected_output(grid, matched_cells, success)

    assert _cross_color_extra_matches(grid, query_sources, matched_cells) > 0
    assert _all_unmatched_inputs_non_gray(grid, matched_cells)
    if success:
        assert _is_full_diagonal(matched_cells)
        assert len(query_sources) == TABLE_DIM
    else:
        assert (len(query_sources) < TABLE_DIM) or (not _is_full_diagonal(matched_cells))

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
