# Combo variant of d8e07eb2 (index 17)
# combo_spec: D1/2+D8/2
# Name: 旋转匹配+对称字形集

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis
# ----------------
# concretization:
# - The canvas, separators, 4x4 table, and verdict logic stay exactly where the
#   source spec places them: rows 0-4 query band, row 5 separator, rows 7-26
#   table of packed 5x5 cells, row 27 separator, rows 28-29 verdict band.
# - The fusion changes two linked pieces together:
#   1. Matching is now dihedral-invariant: color must match, and the 3x3 binary
#      glyph may agree after rotation or reflection.
#   2. Every glyph comes from a symmetric shape class, so some non-identity
#      transforms visibly change the glyph while others leave it unchanged.
# - Small trace:
#   original task: a maroon table glyph and its rotated query copy do not match
#   because the exact occupied pixels differ.
#   fused task: both reduce to the same canonical dihedral signature, so that
#   table cell is selected.
#   extra fusion effect: if the glyph is a plus-sign, a non-identity transform
#   can still leave it looking identical, so some query cells are "trivially"
#   transformed while others require mental rotation.
#
# complexity calibration:
# - Restricting the shape pool to symmetric glyphs risks making the task collapse
#   back to exact matching if every query looks unchanged. To prevent that, the
#   generator forces each sample to include both:
#   - at least one query whose non-identity transform changes the exact 3x3 mask,
#   - at least one query whose non-identity transform is invisible because of
#     symmetry.
# - Restricting shape diversity too far also risks ambiguous matches when a color
#   repeats. The generator therefore only reuses a shape class across different
#   colors, never twice for the same color. Matching remains unique on
#   (color, canonical symmetric shape).
#
# implicit assumptions review:
# - Source `_cell_signature` encoded absolute foreground coordinates inside the
#   5x5 cell. That assumption breaks under rotation/reflection, so the solver now
#   canonicalizes the centered 3x3 occupancy mask over the full dihedral group.
# - `_constant_non_bg_rows` still assumes the only full-width constant non-bg
#   rows are the two separators. The generator preserves that exactly.
# - Table-grid inference still assumes the reference area is a packed lattice of
#   5x5 cells with a 1-pixel outer border around each 3x3 glyph. The generator
#   keeps the original coordinates and fills all 16 table cells.
# - The new `_cell_data` helper assumes every non-empty glyph cell is monochrome
#   and has no foreground outside the centered 3x3 box. The generator enforces
#   those constraints.
# - The success test still assumes query glyphs are distinct under the new
#   matching key. Because each selected table cell has a unique
#   (color, canonical-shape) pair, transformed queries remain distinct as well.
#
# placement legality:
# - All objects stay on the original fixed lattice: query slots at columns
#   1/6/11/16 and table cells at rows 7/12/17/22, cols 1/6/11/16.
# - Every rendered glyph is confined to rows/cols 1:4 inside its 5x5 cell, so no
#   overlap or boundary issues arise under any rotation/reflection.
# - The generator rejects any sample whose chosen queries fail to exhibit the
#   intended visible/trivial transform mix or whose expected output disagrees with
#   the solver.
#
# visual cleanliness / observability / inferability:
# - The symmetric shape pool is intentionally small and reused across colors, so a
#   human can notice that the table is built from a few recurring symmetric
#   motifs rather than 16 unrelated shapes.
# - Within each sample, some query glyphs are exact-looking copies after a
#   non-identity transform, while at least one other query visibly changes
#   orientation. This makes the fusion observable in 2-3 training examples:
#   matching cannot be pure exact equality, yet symmetry clearly matters.
# - Colors are never remapped; green/red remain purely output markers, so there is
#   no hidden numeric convention to infer.


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
    seen: set[tuple[int, ...]] = set()
    for rot_k in range(4):
        rot = np.rot90(mask, rot_k)
        for arr in (rot, np.fliplr(rot)):
            key = tuple(int(x) for x in arr.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            variants.append(arr.copy())
    return tuple(variants)


def _variant_mask_ints(mask: int) -> tuple[int, ...]:
    return tuple(_mask_to_int(arr) for arr in _mask_variants(_int_to_mask(mask)))


def _canonical_mask_int(mask: int) -> int:
    return min(_variant_mask_ints(mask))


def _transform_mask(mask: int, rot_k: int, do_flip: bool) -> int:
    arr = np.rot90(_int_to_mask(mask), int(rot_k))
    if do_flip:
        arr = np.fliplr(arr)
    return _mask_to_int(arr)


def _cell_data(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[int, int, int] | None:
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
    exact_mask = _mask_to_int(inner)
    return color, exact_mask, _canonical_mask_int(exact_mask)


def _cell_key(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[int, int] | None:
    data = _cell_data(g, r0, c0, size, bg)
    if data is None:
        return None
    color, _exact_mask, canonical_mask = data
    return color, canonical_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match symmetric query glyphs to a reference table and mark whether they form a line.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a query strip of 5x5 glyph cells.
    - Between separators is a 4x4 reference table of the same 5x5 cells; each
      cell contains one centered monochrome 3x3 glyph, and every glyph belongs
      to a symmetric shape family.

    Rule:
    - Green-fill the background inside every table cell whose glyph matches a
      query glyph by color plus 3x3 shape up to rotation or reflection.
    - If the matched table cells are exactly one full table row or one full table
      column and all four query slots are occupied, also green-fill the query
      band and the bottom verdict band.
    - Otherwise keep the query band unchanged and fill the verdict band red.

    Key insight: Because all glyphs are symmetric, some legal transforms are
    visually obvious while others are invisible, so comparison must use
    color + transform-class rather than literal pixel equality.
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

# -- Glyph pool --
FILL_MIN, FILL_MAX = 5, 9
SYMMETRIC_CLASS_POOL = (121, 186, 238, 341, 365, 381, 495)

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

# -- Generation --
MAX_RETRIES = 300
NON_IDENTITY_OPS = ((1, False), (2, False), (3, False), (0, True), (1, True), (2, True), (3, True))


def _has_full_bbox(mask: int) -> bool:
    bits = [(mask >> i) & 1 for i in range(9)]
    return (
        any(bits[0:3])
        and any(bits[6:9])
        and any(bits[i] for i in (0, 3, 6))
        and any(bits[i] for i in (2, 5, 8))
    )


def _is_symmetric(mask: int) -> bool:
    return any(_transform_mask(mask, rot_k, do_flip) == mask for rot_k, do_flip in NON_IDENTITY_OPS)


def _symmetric_shape_classes() -> tuple[int, ...]:
    reps: set[int] = set()
    for mask in range(1 << 9):
        if not (FILL_MIN <= mask.bit_count() <= FILL_MAX and _has_full_bbox(mask)):
            continue
        if not _is_symmetric(mask):
            continue
        reps.add(_canonical_mask_int(mask))
    return tuple(sorted(reps))


SYMMETRIC_CLASSES = set(_symmetric_shape_classes())
assert set(SYMMETRIC_CLASS_POOL) <= SYMMETRIC_CLASSES


def _balanced_color_list() -> list[int]:
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_classes(colors: list[int]) -> list[int]:
    per_color_used = {color: set() for color in GLYPH_COLORS}
    totals = {shape: 0 for shape in SYMMETRIC_CLASS_POOL}
    classes: list[int] = []
    for color in colors:
        choices = [shape for shape in SYMMETRIC_CLASS_POOL if shape not in per_color_used[color]]
        min_count = min(totals[shape] for shape in choices)
        lightest = [shape for shape in choices if totals[shape] == min_count]
        shape = random.choice(lightest)
        per_color_used[color].add(shape)
        totals[shape] += 1
        classes.append(shape)
    return classes


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    cell[1:4, 1:4][_int_to_mask(mask)] = color
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


def _query_ops(table_masks: list[int]) -> list[tuple[int, bool]] | None:
    stable_ops = [
        [op for op in NON_IDENTITY_OPS if _transform_mask(mask, *op) == mask]
        for mask in table_masks
    ]
    changing_ops = [
        [op for op in NON_IDENTITY_OPS if _transform_mask(mask, *op) != mask]
        for mask in table_masks
    ]
    diff_idxs = [i for i, ops in enumerate(changing_ops) if ops]
    if not diff_idxs:
        return None

    trivial_idxs = [i for i, ops in enumerate(stable_ops) if ops]
    assert len(trivial_idxs) == len(table_masks), "every symmetric glyph should have a non-identity symmetry"
    diff_idx = random.choice(diff_idxs)
    other_trivial = [i for i in trivial_idxs if i != diff_idx]
    if not other_trivial:
        return None
    trivial_idx = random.choice(other_trivial)

    chosen: list[tuple[int, bool] | None] = [None] * len(table_masks)
    chosen[diff_idx] = random.choice(changing_ops[diff_idx])
    chosen[trivial_idx] = random.choice(stable_ops[trivial_idx])

    for i in range(len(table_masks)):
        if chosen[i] is not None:
            continue
        choices = changing_ops[i] + stable_ops[i]
        chosen[i] = random.choice(choices)

    return [op for op in chosen if op is not None]


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

    colors = _balanced_color_list()
    shape_classes = _balanced_shape_classes(colors)
    table_specs: list[tuple[int, int, int]] = []
    for i, (shape_class, color) in enumerate(zip(shape_classes, colors, strict=True)):
        mask = random.choice(_variant_mask_ints(shape_class))
        table_specs.append((shape_class, mask, color))
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, _glyph(mask, color), ROWS[r_i], COLS[c_i])

    table_keys = {(color, shape_class) for shape_class, _mask, color in table_specs}
    assert len(table_keys) == len(table_specs), "table glyphs must be unique on color+canonical symmetric shape"

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    chosen_specs = [table_specs[r_i * TABLE_DIM + c_i] for r_i, c_i in query_cells]
    ops = _query_ops([mask for _shape_class, mask, _color in chosen_specs])
    if ops is None:
        return None

    visible_changes = 0
    trivial_changes = 0
    exact_query_keys: list[tuple[int, int]] = []
    canonical_query_keys: list[tuple[int, int]] = []
    for slot, ((_shape_class, table_mask, color), (rot_k, do_flip)) in enumerate(zip(chosen_specs, ops, strict=True)):
        query_mask = _transform_mask(table_mask, rot_k, do_flip)
        if query_mask == table_mask:
            trivial_changes += 1
        else:
            visible_changes += 1
        exact_query_keys.append((color, query_mask))
        canonical_query_keys.append((color, _canonical_mask_int(query_mask)))
        paste(grid, _glyph(query_mask, color), 0, COLS[slot])

    if visible_changes < 1 or trivial_changes < 1:
        return None
    if len(exact_query_keys) != len(set(exact_query_keys)):
        return None
    if len(canonical_query_keys) != len(set(canonical_query_keys)):
        return None

    exact_table_keys = {(color, mask) for _shape_class, mask, color in table_specs}
    if exact_table_keys.issuperset(exact_query_keys):
        return None

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
