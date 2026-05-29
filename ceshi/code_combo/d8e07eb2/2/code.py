# Combo variant of d8e07eb2 (index 2)
# combo_spec: D1/2+D5/1
# Name: 旋转匹配+2x2子矩阵判定

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, paste


# Design analysis
# ----------------
# a) concretization and behavior trace:
# - The canvas geometry stays exactly as in the source task: query band on rows
#   0..4, pink separator on row 5, a 4x4 table of 5x5 cells between rows 7..26,
#   another separator on row 27, and a 2-row verdict band on rows 28..29.
# - Each 5x5 cell still contains a monochrome 3x3 glyph centered in the inner
#   window. The first fused change is in matching: the source solver compared the
#   absolute colored pixels in a cell, so a query "L" rotated 90 degrees would not
#   match the same-color table "L". Here both are reduced to one dihedral
#   canonical signature, so the rotated or flipped query still selects that table
#   cell.
# - The second fused change is in the verdict geometry: after the transformed
#   matching step selects table cells, success now requires those cells to be
#   exactly {(r,c),(r,c+1),(r+1,c),(r+1,c+1)} for some adjacent rows and columns.
#   Example trace: four query glyphs may match table cells {(1,1),(1,2),(2,1),
#   (2,2)} only after mental rotation; this now passes, whereas the source task
#   would fail because the shapes are not exact copies and because the selected
#   cells are not a full row or column.
#
# b) complexity calibration:
# - Rotation/flip matching creates an ambiguity risk if two table cells share the
#   same color and the same dihedral shape class. The generator removes that risk
#   by filling the table with sixteen distinct asymmetric dihedral classes.
# - Symmetric shapes would make the new rule visually weak because a rotation or
#   reflection could look unchanged. Restricting the pool to asymmetric masks
#   ensures the transformed query glyphs visibly differ from their table sources.
# - The 2x2 verdict is compact and human-legible, and fail modes explicitly
#   include full rows/columns and non-adjacent rectangles so the dataset cannot
#   collapse back to the original row/column criterion.
#
# c) implicit assumptions review:
# - Source `_cell_signature` encoded exact local coordinates and therefore assumed
#   orientation mattered. That assumption is invalid under D1/2, so matching is
#   rewritten around `(color, canonical_mask)` instead.
# - The solver now assumes glyph pixels stay inside the centered 3x3 and that each
#   glyph cell is monochrome; otherwise color+canonical-mask extraction is
#   ill-posed. The generator preserves those constraints exactly.
# - The source success test assumed "four queries in one line" was enough. Under
#   D5/1 that is false; the solver must additionally require two distinct adjacent
#   row indices and two distinct adjacent column indices.
# - Combining the two changes introduces one more hidden requirement: a query must
#   not exactly equal some other same-color table glyph, or exact matching would
#   accidentally solve the puzzle. The generator forbids that by using one
#   representative per dihedral class and only non-identity query transforms.
#
# d) placement legality:
# - All objects remain on the original 5x5 lattice with the same fixed anchors, so
#   no overlap or boundary risks are added.
# - Query glyphs are pasted only into the four top slots at columns 1, 6, 11, 16.
#   The table occupies the original 4x4 block, and all chosen source cells are
#   distinct, so every placement is legal.
#
# e) visual observability:
# - Across samples, the user can see that the query glyphs are not literal copies
#   of the table glyphs, yet certain table cells are still selected, forcing the
#   rotation/flip hypothesis.
# - Successful samples highlight a compact 2x2 table block and green the query
#   band, while failing samples may highlight a full line or a scattered quartet
#   but still show a red verdict. That contrast makes the new spatial condition
#   recoverable from a small number of examples.
#
# f) visual cleanliness:
# - No extra decorations are introduced beyond the original green background fill
#   for matched cells and the green/red verdict band. The scene stays sparse: query
#   cells on top, table in the middle, verdict at the bottom.
#
# g) inferability of changes:
# - No new color mapping is introduced. Colors still serve only as part of glyph
#   identity. The learnable changes are structural: "same color + same shape up to
#   dihedral transform" for matching and "selected cells form one adjacent 2x2"
#   for success.


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
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))

    uniq: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for arr in variants:
        key = tuple(int(x) for x in arr.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(arr.copy())
    return tuple(uniq)


def _canonical_mask(mask: np.ndarray) -> tuple[int, ...]:
    return min(tuple(int(x) for x in arr.reshape(-1)) for arr in _mask_variants(mask))


def _cell_key(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[int, tuple[int, ...]] | None:
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
    color = next(iter(colors))
    return color, _canonical_mask(inner)


def _forms_2x2_block(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    cells = {(int(r), int(c)) for r, c in cells}
    if len(cells) != 4:
        return False
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    return (
        len(rows) == 2
        and len(cols) == 2
        and rows[1] == rows[0] + 1
        and cols[1] == cols[0] + 1
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match transformed query glyphs to a reference table and test for a 2x2 block.

    Input structure:
    - A teal canvas is split by two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells.
    - The middle band contains a 4x4 table of 5x5 cells, each holding one
      centered monochrome 3x3 glyph.

    Rule:
    - A query glyph matches a table glyph when the colors are equal and the 3x3
      occupancy pattern agrees after some rotation or reflection.
    - Green-fill the background of every matched table cell.
    - If there are exactly four matched query glyphs and their table positions are
      exactly one contiguous 2x2 submatrix, also green-fill the query band and the
      verdict band; otherwise leave the query band unchanged and fill the verdict
      band red.

    Key insight: The solver must first normalize glyph orientation, then judge the
    matched cells as a compact square region rather than as a row or column.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = get_bg(g)
    cell_size = 5
    required_matches = 4

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
    assert len(set(table_keys)) == len(table_keys), "table glyphs must be unique under color+transform equivalence"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            if _cell_key(g, r, c, cell_size, bg) not in query_set:
                continue
            selected.append((int(ri), int(ci)))
            cell = out[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = 3

    success = (
        len(query_keys) == required_matches
        and len(selected) == required_matches
        and _forms_2x2_block(selected)
    )

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
BLOCK_SIDE = 2
BLOCK_AREA = BLOCK_SIDE * BLOCK_SIDE

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("line", "gapped_rect", "bent", "scatter")

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


def _transform_mask(mask: int, rot_k: int, do_flip: bool) -> int:
    arr = _int_to_mask(mask)
    out = np.rot90(arr, int(rot_k))
    if do_flip:
        out = np.fliplr(out)
    return _mask_to_int(out)


def _variant_masks(mask: int) -> tuple[int, ...]:
    return tuple(_mask_to_int(arr) for arr in _mask_variants(_int_to_mask(mask)))


def _canonical_mask_int(mask: int) -> int:
    return min(_variant_masks(mask))


def _asymmetric_shape_classes() -> tuple[int, ...]:
    reps: set[int] = set()
    for mask in range(1 << 9):
        if not (FILL_MIN <= mask.bit_count() <= FILL_MAX and _has_full_bbox(mask)):
            continue
        variants = _variant_masks(mask)
        if len(variants) != 8:
            continue
        reps.add(_canonical_mask_int(mask))
    out = tuple(sorted(reps))
    assert len(out) == TABLE_DIM * TABLE_DIM, "expected exactly sixteen asymmetric shape classes"
    return out


ASYMMETRIC_CLASSES = _asymmetric_shape_classes()
ROT_OPS = ((1, False), (2, False), (3, False))
FLIP_OPS = ((0, True), (1, True), (2, True), (3, True))
NON_IDENTITY_OPS = ROT_OPS + FLIP_OPS


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


def _block_cells() -> list[tuple[int, int]]:
    top = random.randrange(TABLE_DIM - 1)
    left = random.randrange(TABLE_DIM - 1)
    cells = [(top + dr, left + dc) for dr in range(BLOCK_SIDE) for dc in range(BLOCK_SIDE)]
    random.shuffle(cells)
    return cells


def _line_cells() -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _gapped_rect_cells() -> list[tuple[int, int]]:
    row_pairs = [(a, b) for a in range(TABLE_DIM) for b in range(a + 1, TABLE_DIM)]
    col_pairs = [(a, b) for a in range(TABLE_DIM) for b in range(a + 1, TABLE_DIM)]
    while True:
        r0, r1 = random.choice(row_pairs)
        c0, c1 = random.choice(col_pairs)
        cells = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if not _forms_2x2_block(cells):
            random.shuffle(cells)
            return cells


def _bent_cells() -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        dropped_c = random.randrange(TABLE_DIM)
        new_r = random.choice([rr for rr in range(TABLE_DIM) if rr != r])
        cells = [(r, c) for c in range(TABLE_DIM) if c != dropped_c] + [(new_r, dropped_c)]
    else:
        c = random.randrange(TABLE_DIM)
        dropped_r = random.randrange(TABLE_DIM)
        new_c = random.choice([cc for cc in range(TABLE_DIM) if cc != c])
        cells = [(r, c) for r in range(TABLE_DIM) if r != dropped_r] + [(dropped_r, new_c)]
    assert len(cells) == BLOCK_AREA and len(set(cells)) == BLOCK_AREA
    assert not _forms_2x2_block(cells)
    random.shuffle(cells)
    return cells


def _scatter_cells(count: int) -> list[tuple[int, int]]:
    universe = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM)]
    while True:
        cells = random.sample(universe, count)
        if not _forms_2x2_block(cells):
            random.shuffle(cells)
            return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    if count == 3:
        return _scatter_cells(count)

    mode = random.choice(FAIL_MODES)
    if mode == "line":
        return _line_cells()
    if mode == "gapped_rect":
        return _gapped_rect_cells()
    if mode == "bent":
        return _bent_cells()
    return _scatter_cells(count)


def _query_ops(count: int) -> list[tuple[int, bool]]:
    ops: list[tuple[int, bool]] = [random.choice(ROT_OPS), random.choice(FLIP_OPS)]
    while len(ops) < count:
        ops.append(random.choice(NON_IDENTITY_OPS))
    random.shuffle(ops)
    return ops[:count]


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
    assert len(canon_pairs) == len(table_specs), "table glyphs must stay unique under transformed matching"

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _block_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else BLOCK_AREA
        query_cells = _fail_cells(count)

    ops = _query_ops(len(query_cells))
    exact_query_keys: list[tuple[int, int]] = []
    for slot, ((r_i, c_i), (rot_k, do_flip)) in enumerate(zip(query_cells, ops, strict=True)):
        table_mask, color = table_specs[r_i * TABLE_DIM + c_i]
        query_mask = _transform_mask(table_mask, rot_k, do_flip)
        assert query_mask != table_mask, "queries should visibly exercise rotation/flip matching"
        exact_query_keys.append((color, query_mask))
        paste(grid, _glyph(query_mask, color), 0, COLS[slot])

    exact_table_keys = {(color, mask) for mask, color in table_specs}
    assert not any(key in exact_table_keys for key in exact_query_keys), "queries must not collapse back to exact matching"

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
