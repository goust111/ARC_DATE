# Combo variant of d8e07eb2 (index 27)
# combo_spec: D1/2+D5/2+D9/0
# Name: 旋转匹配+对角线+固定4查询

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for combo D1/2 + D5/2 + D9/0
# ----------------------------------------------
# a) Concretization and behavior trace
#    - The canvas geometry stays the same as the source task: a 5-row query band,
#      two full-width separator rows, a 4x4 table of 5x5 cells, and a 2-row
#      verdict band. Each 5x5 cell still contains one centered monochrome 3x3 glyph.
#    - Original behavior:
#        1. Compare each query glyph to table glyphs by exact color + exact 3x3 pattern.
#        2. Green-fill every matched table cell.
#        3. Succeed only when the 3-4 matches form one whole row or one whole column.
#    - Combo behavior:
#        1. Compare by color + 3x3 pattern up to the full dihedral family
#           (rotations 0/90/180/270, with or without reflection).
#        2. Green-fill every matched table cell, exactly as before.
#        3. Queries are always four glyphs.
#        4. Succeed only when the four matched cells are exactly the main diagonal
#           or the anti-diagonal.
#    - Small trace:
#        table cell (0,0): orange asymmetric "hook";
#        query slot 2: same orange hook rotated 90 degrees.
#        Original solver: no exact match, so cell (0,0) would stay unselected.
#        Combo solver: both reduce to the same canonical transform class, so (0,0)
#        is selected. If the four selected cells are {(0,0),(1,1),(2,2),(3,3)},
#        the query band and verdict band also turn green; if they are a full row,
#        only the table cells turn green and the verdict stays red.
#
# b) Complexity calibration
#    - Rotation/reflection matching can easily become ambiguous if two table cells
#      share the same color and the same transform-equivalence class. The generator
#      forbids that by using 16 distinct asymmetric dihedral classes, one per cell.
#    - Fixed query count = 4 is important for diagonal reasoning: it removes the
#      trivial "fewer than four can never pass" cue and keeps success/failure
#      dependent on geometry plus transformed matching.
#    - To keep the D5 change visible, fail cases deliberately include complete rows
#      and columns. Those would pass in the source task but must fail here.
#
# c) Implicit-assumption review
#    - Source `_cell_signature` encoded absolute filled coordinates inside the 5x5
#      cell. That assumption breaks under rotation/flip, so the solver instead
#      canonicalizes the centered 3x3 occupancy mask and pairs it with the glyph color.
#    - Source success logic assumed row/column completeness via `same_row` or
#      `same_col`. That is replaced with explicit comparison against the two 4-cell
#      diagonal sets.
#    - The solver now assumes each non-empty cell is monochrome and all foreground
#      pixels lie inside the centered 3x3 area. The generator constructs exactly that.
#    - The solver also assumes table keys are unique under color + transform class,
#      otherwise a single query could match multiple same-colored transformed clones.
#      The generator enforces uniqueness with asymmetric classes and one class per cell.
#
# d) Placement legality
#    - All objects use the original fixed coordinates, so there is no new overlap or
#      boundary risk. Query slots remain at columns 1, 6, 11, 16; table cells stay in
#      the 4x4 lattice below the blank spacer row.
#    - Query glyphs are rendered as transformed versions of their source table glyphs
#      but remain centered in the same 5x5 cell template, so every query stays legal.
#
# e) Visual observability
#    - A human can infer the combo from a few examples because the table highlights
#      show two simultaneous clues:
#        1. some query glyphs visibly do not exactly appear in the table, yet one
#           rotated/flipped counterpart does get highlighted;
#        2. some failures highlight a full row or full column while the verdict is red,
#           forcing the diagonal hypothesis.
#    - The generator also forces every query set to contain at least one pure rotation
#      and at least one reflection, so the transformed-match rule is not latent.
#
# f) Visual cleanliness
#    - The output marking scheme is unchanged: only background pixels inside matched
#      table cells turn green, separator rows remain color 6, and the verdict band is
#      solid green/red. No extra overlays are introduced.
#    - Restricting shapes to asymmetric 3x3 masks avoids visually degenerate cases
#      where a "transformed" query looks identical to its source.
#
# g) Inference and color audit
#    - No rule depends on numeric color ordering. Color is only an equality test
#      attached to each glyph.
#    - Green still means "selected / success" and red still means "failure"; the
#      fusion changes only object identity and the geometry of the winning pattern.
#
# h) Solver-generator contract audit
#    - `main()` scans the four query slots and may treat missing slots as empty, but
#      `generate_input()` always fills all four to satisfy D9/0 visibly.
#    - `main()` infers the table lattice from the first non-background glyph pixel,
#      so the generator must keep the original one-pixel cell margin and the blank row
#      above the table. The layout constants preserve that exact structure.


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _mask_to_int(mask: np.ndarray) -> int:
    bits = [int(x) for x in np.asarray(mask, dtype=int).reshape(-1)]
    return sum(bit << i for i, bit in enumerate(bits))


def _int_to_mask(mask: int) -> np.ndarray:
    return np.array([(int(mask) >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)


def _mask_variants(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    variants: list[np.ndarray] = []
    for rot_k in range(4):
        rotated = np.rot90(mask, rot_k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))

    out: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for arr in variants:
        key = tuple(int(x) for x in arr.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        out.append(arr.copy())
    return tuple(out)


def _variant_masks(mask: int) -> tuple[int, ...]:
    return tuple(_mask_to_int(arr) for arr in _mask_variants(_int_to_mask(mask)))


def _canonical_mask_int(mask: int) -> int:
    return min(_variant_masks(mask))


def _transform_mask(mask: int, rot_k: int, do_flip: bool) -> int:
    arr = np.rot90(_int_to_mask(mask), int(rot_k))
    if do_flip:
        arr = np.fliplr(arr)
    return _mask_to_int(arr)


def _cell_key(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[int, int] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    fg = cell != int(bg)
    if not bool(fg.any()):
        return None

    inner = fg[1:4, 1:4]
    outer = fg.copy()
    outer[1:4, 1:4] = False
    assert not bool(outer.any()), "glyph pixels must stay inside the centered 3x3 region"

    colors = {int(x) for x in cell[fg].tolist()}
    assert len(colors) == 1, "glyph cell must be monochrome"
    return next(iter(colors)), _canonical_mask_int(_mask_to_int(inner))


def _exact_cell_key(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[int, int] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    fg = cell != int(bg)
    if not bool(fg.any()):
        return None

    inner = fg[1:4, 1:4]
    outer = fg.copy()
    outer[1:4, 1:4] = False
    assert not bool(outer.any())

    colors = {int(x) for x in cell[fg].tolist()}
    assert len(colors) == 1
    return next(iter(colors)), _mask_to_int(inner)


def _diagonal_cell_sets(dim: int) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    main_diag = frozenset((i, i) for i in range(dim))
    anti_diag = frozenset((i, dim - 1 - i) for i in range(dim))
    return main_diag, anti_diag


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up transformed query glyphs in a reference table and mark whether they form a diagonal.

    Input structure:
    - Teal background with two full-width separator rows.
    - Above the first separator is a row of four 5x5 query cells.
    - Between separators is a 4x4 table of the same 5x5 cells; each non-empty
      cell contains one monochrome 3x3 glyph centered inside a 1-pixel border.

    Rule:
    - Green-fill the background inside every table cell whose glyph matches a
      query glyph by color and 3x3 shape up to rotation or reflection.
    - If the four matched cells are exactly one full main diagonal or one full
      anti-diagonal, also green-fill the query band and the verdict band.
    - Otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: The solver must combine two visible cues at once: mentally
    transform query glyphs before matching them, then judge success from the
    diagonal arrangement of the matched table positions rather than rows/columns.
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

    query_keys = [
        key
        for c in col_starts
        if (key := _cell_key(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_keys)
    assert len(query_keys) == len(query_set), "query glyphs should be distinct"

    table_keys = [
        _cell_key(g, r, c, cell_size, bg)
        for r in row_starts
        for c in col_starts
    ]
    assert all(key is not None for key in table_keys), "reference table should be fully populated"
    assert len(set(table_keys)) == len(table_keys), "table glyphs must be unique under transformed matching"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            if _cell_key(g, r, c, cell_size, bg) not in query_set:
                continue
            selected.append((int(ri), int(ci)))
            cell = out[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = 3

    complete_query = len(query_keys) == len(col_starts)
    square_table = len(row_starts) == len(col_starts)
    is_complete_diagonal = square_table and frozenset(selected) in _diagonal_cell_sets(len(row_starts))
    success = complete_query and len(selected) == len(query_keys) and is_complete_diagonal

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

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.5
FAIL_MODES = ("row", "row", "col", "col", "off_main", "off_anti", "rect", "perm")

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


def _asymmetric_shape_classes() -> tuple[int, ...]:
    reps: set[int] = set()
    for mask in range(1 << 9):
        if not (FILL_MIN <= mask.bit_count() <= FILL_MAX and _has_full_bbox(mask)):
            continue
        variants = _variant_masks(mask)
        if len(variants) != 8:
            continue
        reps.add(_canonical_mask_int(mask))

    classes = tuple(sorted(reps))
    assert len(classes) == TABLE_DIM * TABLE_DIM, "expected exactly sixteen asymmetric shape classes"
    return classes


ASYMMETRIC_CLASSES = _asymmetric_shape_classes()
ROT_OPS = ((1, False), (2, False), (3, False))
FLIP_OPS = ((0, True), (1, True), (2, True), (3, True))
NON_IDENTITY_OPS = ROT_OPS + FLIP_OPS
NON_DIAGONAL_PERMS = tuple(
    perm
    for perm in itertools.permutations(range(TABLE_DIM))
    if perm not in ((0, 1, 2, 3), (3, 2, 1, 0))
)


def _balanced_color_list() -> list[int]:
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _table_masks() -> list[int]:
    classes = list(ASYMMETRIC_CLASSES)
    random.shuffle(classes)
    masks = [random.choice(_variant_masks(rep)) for rep in classes]
    assert len({_canonical_mask_int(mask) for mask in masks}) == len(masks)
    return masks


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    cell[1:4, 1:4][_int_to_mask(mask)] = color
    return cell


def _main_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, i) for i in range(TABLE_DIM)]


def _anti_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, TABLE_DIM - 1 - i) for i in range(TABLE_DIM)]


def _diagonal_cells() -> list[tuple[int, int]]:
    cells = list(random.choice((_main_diagonal_cells(), _anti_diagonal_cells())))
    random.shuffle(cells)
    return cells


def _is_full_diagonal(cells: list[tuple[int, int]]) -> bool:
    return len(cells) == TABLE_DIM and frozenset(cells) in _diagonal_cell_sets(TABLE_DIM)


def _fail_cells() -> list[tuple[int, int]]:
    mode = random.choice(FAIL_MODES)
    if mode == "row":
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    elif mode == "col":
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    elif mode == "off_main":
        diag = set(_main_diagonal_cells())
        removed = random.choice(tuple(diag))
        pool = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM) if (r, c) not in diag]
        cells = list((diag - {removed}) | {random.choice(pool)})
    elif mode == "off_anti":
        diag = set(_anti_diagonal_cells())
        removed = random.choice(tuple(diag))
        pool = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM) if (r, c) not in diag]
        cells = list((diag - {removed}) | {random.choice(pool)})
    elif mode == "rect":
        rs = random.sample(range(TABLE_DIM), 2)
        cs = random.sample(range(TABLE_DIM), 2)
        cells = [(r, c) for r in rs for c in cs]
    else:
        perm = random.choice(NON_DIAGONAL_PERMS)
        cells = [(r, perm[r]) for r in range(TABLE_DIM)]

    assert len(cells) == TABLE_DIM and len(set(cells)) == TABLE_DIM
    assert not _is_full_diagonal(cells)
    random.shuffle(cells)
    return cells


def _query_ops() -> list[tuple[int, bool]]:
    ops: list[tuple[int, bool]] = [random.choice(ROT_OPS), random.choice(FLIP_OPS)]
    while len(ops) < TABLE_DIM:
        ops.append(random.choice(NON_IDENTITY_OPS))
    random.shuffle(ops)
    return ops[:TABLE_DIM]


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
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

    masks = _table_masks()
    colors = _balanced_color_list()
    table_specs: list[tuple[int, int]] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        table_specs.append((mask, color))
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, _glyph(mask, color), ROWS[r_i], COLS[c_i])

    canon_pairs = {(_canonical_mask_int(mask), color) for mask, color in table_specs}
    assert len(canon_pairs) == len(table_specs), "table specs must be unique under transformed matching"

    success = random.random() < P_SUCCESS
    query_cells = _diagonal_cells() if success else _fail_cells()
    ops = _query_ops()

    exact_query_keys: list[tuple[int, int]] = []
    for slot, ((r_i, c_i), (rot_k, do_flip)) in enumerate(zip(query_cells, ops, strict=True)):
        table_mask, color = table_specs[r_i * TABLE_DIM + c_i]
        query_mask = _transform_mask(table_mask, rot_k, do_flip)
        assert query_mask != table_mask, "queries must visibly rely on transformed matching"
        exact_query_keys.append((color, query_mask))
        paste(grid, _glyph(query_mask, color), 0, COLS[slot])

    exact_table_keys = {(color, mask) for mask, color in table_specs}
    assert len(exact_query_keys) == TABLE_DIM
    assert len(set(exact_query_keys)) == len(exact_query_keys), "query glyphs should stay distinct"
    assert not any(key in exact_table_keys for key in exact_query_keys), "queries should not collapse to exact copies"

    expected = _expected_output(grid, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo-27 input")
