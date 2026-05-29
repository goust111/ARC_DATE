# Variant of d8e07eb2
# Dimension: D1 — 字形匹配条件 (match_cond)
# Change: 查询字形s与参考表字形d必须同时满足颜色相同且3x3二值形状相同才算匹配... → 允许旋转/翻转的形状匹配

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis
# ----------------
# concretization:
# - Layout stays identical to the source task: a 5-row query band, two full-width
#   separator rows, a 4x4 table of 5x5 cells, and a 2-row verdict band.
# - Each 5x5 cell still contains a centered 3x3 monochrome glyph.
# - The behavioral change is only in lookup: a query glyph now matches a table
#   glyph when the colors agree and the 3x3 occupancy pattern belongs to the same
#   dihedral-equivalence class (rotation 0/90/180/270, with or without reflection).
# - Small trace:
#   original: query red "L" rotated 90 degrees does not equal the table red "L",
#   so the table cell is not selected.
#   variant: both reduce to the same canonical dihedral signature, so that cell is
#   selected and contributes to the row/column test.
#
# complexity calibration:
# - The main ambiguity risk is having two table glyphs with the same color and
#   transform-equivalent shapes. The generator forbids that by assigning each of
#   the 16 table cells a distinct asymmetric dihedral class.
# - Visual observability matters: if a glyph is symmetric, rotating/flipping it is
#   invisible and the task can collapse back to the original. Therefore the shape
#   pool is restricted to asymmetric 3x3 masks whose non-identity transforms all
#   look different.
#
# implicit assumptions review:
# - Source `_cell_signature` depended on absolute `(row, col)` coordinates inside
#   the 5x5 cell; that assumption breaks under rotation/flip and is replaced with
#   `(color, canonical 3x3 mask)`.
# - The line-test logic (`selected` cells forming one full row/column) remains
#   valid because only the lookup relation changed, not the table geometry.
# - New solver assumption: each glyph cell is monochrome and confined to the inner
#   3x3 area. The generator guarantees this exactly.
#
# placement legality:
# - All placements keep the original coordinates and sizes, so no new overlap or
#   boundary risks are introduced.
# - Query slots still occupy columns 1,6,11,16; empty slots remain plain bg.
#
# visual cleanliness / observability:
# - Every query glyph is a transformed version of a table glyph, not an exact copy.
# - Each generated puzzle forces at least one rotated query and at least one
#   reflected query, so both branches of the new equivalence rule are visible
#   across samples without adding extra visual clutter.


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
    base: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        base.append(rot)
        base.append(np.fliplr(rot))
    uniq: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for arr in base:
        key = tuple(int(x) for x in arr.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(arr.copy())
    return tuple(uniq)


def _canonical_mask(mask: np.ndarray) -> tuple[int, ...]:
    return min(tuple(int(x) for x in arr.reshape(-1)) for arr in _mask_variants(mask))


def _cell_key(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[int, tuple[int, ...]] | None:
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


def _exact_cell_key(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[int, int] | None:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up transformed query glyphs in a reference table and mark whether they form a line.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a query row of 5x5 glyph cells.
    - Between separators is a reference table of the same 5x5 cells; each cell
      contains one monochrome 3x3 glyph centered inside a 1-pixel margin.

    Rule:
    - Green-fill the background inside every table cell whose glyph matches some
      query glyph by color plus 3x3 shape up to rotation or reflection.
    - If the matched table cells form one complete row or one complete column,
      also green-fill the query band and the bottom verdict band.
    - Otherwise leave the query band unchanged and fill the bottom verdict band red.

    Key insight: The query symbols are not literal copies anymore; they must be
    mentally rotated or flipped before comparing them to the table.
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

    complete_query = len(query_keys) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_keys) and (same_row or same_col)

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
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

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
    bits = _int_to_mask(mask)
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
    glyphs: list[np.ndarray] = []
    table_specs: list[tuple[int, int]] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        table_specs.append((mask, color))
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    canon_pairs = {(_canonical_mask_int(mask), color) for mask, color in table_specs}
    assert len(canon_pairs) == len(table_specs)

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    ops = _query_ops(len(query_cells))
    exact_query_keys: list[tuple[int, int]] = []
    for slot, ((r_i, c_i), (rot_k, do_flip)) in enumerate(zip(query_cells, ops, strict=True)):
        table_mask, color = table_specs[r_i * TABLE_DIM + c_i]
        query_mask = _transform_mask(table_mask, rot_k, do_flip)
        assert query_mask != table_mask, "queries should visibly use the new transform rule"
        exact_query_keys.append((color, query_mask))
        paste(grid, _glyph(query_mask, color), 0, COLS[slot])

    exact_table_keys = {(color, mask) for mask, color in table_specs}
    assert not any(key in exact_table_keys for key in exact_query_keys), "generator must not collapse back to exact matching"

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
