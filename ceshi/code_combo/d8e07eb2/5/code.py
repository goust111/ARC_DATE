# Combo variant of d8e07eb2 (index 5)
# combo_spec: D1/2+D3/0
# Name: 旋转匹配+可变参考表尺寸

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The broad layout is preserved: a query band, a full-width separator, one blank
#      spacer row, a square table of 5x5 cells, another separator, then a 2-row verdict band.
#    - The fused change is that the table side length is no longer fixed: an example may
#      show a 3x3, 4x4, or 5x5 table, so the whole canvas grows or shrinks with that choice.
#    - Matching is also relaxed: a query glyph now matches a table glyph when the color
#      agrees and the centered 3x3 occupancy pattern is the same up to rotation/reflection.
#    - Small trace:
#      source task: in a 3x3 table, a red query "L" rotated 90 degrees does not equal the
#      red table "L", so that table cell is missed; the original fixed-4 logic also cannot
#      judge the row length correctly.
#      combo task: the solver first infers that the visible table is 3x3, canonicalizes
#      the query and table masks under dihedral transforms, selects the matching row cells,
#      and then accepts because three matches cover the whole visible row.
#
# b) Complexity calibration
#    - The size freedom is intentionally limited to three visible cases: 3, 4, and 5.
#      That is enough to force table-size inference without exploding the case count.
#    - Transform matching can become ambiguous if two table cells share the same
#      (color, canonical-shape) pair. The generator forbids that exact ambiguity.
#    - Visual observability also matters: if a glyph is symmetric, rotation/flip can look
#      unchanged and the sample may collapse back to the original task. Therefore the table
#      uses asymmetric shape classes, and every query set includes both a rotation-based and
#      a reflection-based non-identity transform.
#
# c) Implicit assumptions review
#    - The seed solver used absolute occupied coordinates inside each 5x5 cell as the key.
#      That breaks immediately under rotation/flip, so the combo solver replaces it with
#      `(glyph_color, canonical_3x3_mask)`.
#    - The seed generator and solver hard-coded a 4x4 table, a 30x22 canvas, and a success
#      length of four. Those assumptions fail once 5x5 tables are allowed, so geometry is
#      inferred from the table foreground bounding box and the success length comes from the
#      inferred side length.
#    - The bbox inference assumes each glyph touches all four sides of its inner 3x3 box.
#      The generator guarantees this with `_has_full_bbox`.
#
# d) Placement legality
#    - For side length `dim`, geometry is:
#      height = 5 query rows + 1 separator + 1 spacer + 5*dim table rows + 1 separator + 2 verdict rows
#             = 10 + 5*dim
#      width  = 1 left margin + 5*dim aligned slots + 1 right margin
#             = 2 + 5*dim
#    - Table cells start at row 7 and column 1, remain disjoint, and fit exactly inside
#      the computed canvas. Query glyphs are pasted only into aligned top-band slots.
#
# e) Visual observability
#    - Humans can see both control variables directly: the table is visibly 3x3/4x4/5x5,
#      and query glyphs are visibly transformed rather than copied.
#    - Success examples teach that "full line" means the whole current row/column, not
#      always four cells; failure examples include both short queries (`dim-1`) and bent
#      length-`dim` queries to separate these causes.
#
# f) Visual cleanliness
#    - Regions stay separated by the original separators and spacer row.
#    - Green/red highlighting still edits only background pixels, so glyph ink remains legible.
#
# g) Change inferability
#    - No new hidden color mapping is introduced. Colors still identify glyph families, green
#      still marks matched/success regions, and red still marks failure. The new reasoning load
#      is spatial: infer size, then mentally rotate/flip the query shapes.
#
# h) Verification target
#    - Post-implementation metrics should show all three table sizes in generated samples and
#      query glyphs whose canonical keys match table cells while their exact keys do not.
#
# i) New solver assumptions and generator guarantees
#    - Solver assumption: every non-empty 5x5 cell contains one monochrome glyph wholly inside
#      the centered 3x3 area. Generator guarantee: glyph rendering always follows that template.
#    - Solver assumption: the reference table is square and fully populated. Generator guarantee:
#      it always fills all `dim*dim` table cells.
#    - Solver assumption: query glyphs are placed in table-aligned slots above the separator.
#      Generator guarantee: each query glyph is pasted into one of those aligned columns.


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
        rot = np.rot90(mask, rot_k)
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


def _variant_masks(mask: int) -> tuple[int, ...]:
    return tuple(_mask_to_int(arr) for arr in _mask_variants(_int_to_mask(mask)))


def _canonical_mask(mask: np.ndarray) -> tuple[int, ...]:
    return min(tuple(int(x) for x in arr.reshape(-1)) for arr in _mask_variants(mask))


def _canonical_mask_int(mask: int) -> int:
    return min(_variant_masks(mask))


def _transform_mask(mask: int, rot_k: int, do_flip: bool) -> int:
    arr = np.rot90(_int_to_mask(mask), int(rot_k))
    if do_flip:
        arr = np.fliplr(arr)
    return _mask_to_int(arr)


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
    return next(iter(colors)), _canonical_mask(inner)


def _exact_cell_key(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[int, int] | None:
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
    return next(iter(colors)), _mask_to_int(inner)


def _infer_table_starts(
    g: np.ndarray, top_sep: int, bottom_sep: int, cell_size: int, bg: int
) -> tuple[list[int], list[int]]:
    table_region = g[top_sep + 1 : bottom_sep, :]
    table_fg = np.argwhere(table_region != bg)
    assert table_fg.size, "reference table must contain glyphs"

    min_r = int(table_fg[:, 0].min())
    max_r = int(table_fg[:, 0].max())
    min_c = int(table_fg[:, 1].min())
    max_c = int(table_fg[:, 1].max())

    table_h = max_r - min_r + 1
    table_w = max_c - min_c + 1
    assert (table_h + 2) % cell_size == 0 and (table_w + 2) % cell_size == 0

    row_count = (table_h + 2) // cell_size
    col_count = (table_w + 2) // cell_size
    row0 = top_sep + 1 + min_r - 1
    col0 = min_c - 1

    row_starts = [row0 + cell_size * i for i in range(row_count)]
    col_starts = [col0 + cell_size * i for i in range(col_count)]
    assert row_starts and col_starts, "could not infer table cell grid"
    assert row_starts[-1] + cell_size <= bottom_sep
    assert col_starts[-1] + cell_size <= g.shape[1]
    return row_starts, col_starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match transformed query glyphs against a variable-sized reference table and judge full-line coverage.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - Above the first separator is a row of table-aligned 5x5 query slots.
    - Between separators is a square reference table of 5x5 cells whose side length can
      vary by example; each non-empty cell contains one monochrome 3x3 glyph centered
      inside a 1-pixel border.

    Rule:
    - Green-fill the background inside every table cell whose glyph matches a query glyph
      by color plus 3x3 shape up to rotation or reflection.
    - If the matched cells cover one complete visible row or one complete visible column of
      the inferred table, also green-fill the query band and the verdict band.
    - Otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: Solving requires two coupled inferences: first determine the current table
    size from the layout, then compare glyphs modulo rotation/flip before checking whether
    the resulting matches span an entire row or column of that size.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5

    sep_rows = _constant_non_bg_rows(g, bg)
    assert len(sep_rows) >= 2, "expected two full-width separator rows"
    top_sep = sep_rows[0]
    bottom_sep = sep_rows[-1]

    row_starts, col_starts = _infer_table_starts(g, top_sep, bottom_sep, cell_size, bg)
    row_count = len(row_starts)
    col_count = len(col_starts)
    assert row_count == col_count, "reference table must be square"

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

    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    full_row = len(query_keys) == col_count and len(selected) == col_count and same_row
    full_col = len(query_keys) == row_count and len(selected) == row_count and same_col
    success = full_row or full_col

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator (o5x) ---------------

# -- Colors --
BG = 8
GREEN = 3
RED = 2
SEP = 6
GLYPH_COLORS = (0, 1, 2, 4, 5, 6, 7, 9)

# -- Shared geometry --
CELL = 5
QUERY_BAND_H = 5
LEFT_MARGIN = 1
TOP_SEP_ROW = QUERY_BAND_H
TABLE_GAP_ROW = 1
VERDICT_H = 2
TABLE_DIMS = (3, 4, 5)

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_SHORT_FAIL = 0.35
FAIL_MODES = ("bent_row", "bent_col", "scatter")

# -- Generation --
MAX_RETRIES = 500


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
    out = tuple(sorted(reps))
    assert len(out) >= 16, "expected a sufficiently large pool of asymmetric shape classes"
    return out


ASYMMETRIC_CLASSES = _asymmetric_shape_classes()
ROT_OPS = ((1, False), (2, False), (3, False))
FLIP_OPS = ((0, True), (1, True), (2, True), (3, True))
NON_IDENTITY_OPS = ROT_OPS + FLIP_OPS


def _geometry(dim: int) -> dict[str, int | tuple[int, ...]]:
    width = LEFT_MARGIN + dim * CELL + LEFT_MARGIN
    table_top = TOP_SEP_ROW + 1 + TABLE_GAP_ROW
    bottom_sep = table_top + dim * CELL
    height = bottom_sep + 1 + VERDICT_H
    row_starts = tuple(table_top + CELL * i for i in range(dim))
    col_starts = tuple(LEFT_MARGIN + CELL * i for i in range(dim))
    return {
        "height": height,
        "width": width,
        "top_sep": TOP_SEP_ROW,
        "bottom_sep": bottom_sep,
        "verdict_row": bottom_sep + 1,
        "row_starts": row_starts,
        "col_starts": col_starts,
    }


def _balanced_color_list(total_cells: int) -> list[int]:
    counts = {c: 1 for c in GLYPH_COLORS}
    extras = total_cells - len(GLYPH_COLORS)
    max_per_color = max(2, math.ceil(total_cells / len(GLYPH_COLORS)) + 1)
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < max_per_color]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    assert len(colors) == total_cells
    return colors


def _table_specs(total_cells: int) -> list[tuple[int, int]]:
    colors = _balanced_color_list(total_cells)
    used_classes_by_color = {color: set() for color in GLYPH_COLORS}
    specs: list[tuple[int, int]] = []
    for color in colors:
        choices = [
            canonical_mask
            for canonical_mask in ASYMMETRIC_CLASSES
            if canonical_mask not in used_classes_by_color[color]
        ]
        assert choices, "ran out of distinct canonical classes for one color"
        canonical_mask = random.choice(choices)
        used_classes_by_color[color].add(canonical_mask)
        exact_mask = random.choice(_variant_masks(canonical_mask))
        specs.append((exact_mask, color))

    canon_pairs = {(_canonical_mask_int(mask), color) for mask, color in specs}
    assert len(canon_pairs) == len(specs)
    return specs


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = _int_to_mask(mask)
    cell[1:4, 1:4][bits] = color
    return cell


def _line_cells(dim: int) -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(dim)
        cells = [(r, c) for c in range(dim)]
    else:
        c = random.randrange(dim)
        cells = [(r, c) for r in range(dim)]
    random.shuffle(cells)
    return cells


def _scatter_fail_cells(dim: int, count: int) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(dim) for c in range(dim)]
    for _ in range(200):
        cells = random.sample(all_cells, count)
        if len({r for r, _c in cells}) == 1:
            continue
        if len({c for _r, c in cells}) == 1:
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample scattered fail cells")


def _fail_cells(dim: int, count: int) -> list[tuple[int, int]]:
    if count < dim:
        return _scatter_fail_cells(dim, count)

    mode = random.choice(FAIL_MODES)
    if mode == "bent_row":
        r = random.randrange(dim)
        dropped_c = random.randrange(dim)
        new_r = random.choice([rr for rr in range(dim) if rr != r])
        cells = [(r, c) for c in range(dim) if c != dropped_c] + [(new_r, dropped_c)]
    elif mode == "bent_col":
        c = random.randrange(dim)
        dropped_r = random.randrange(dim)
        new_c = random.choice([cc for cc in range(dim) if cc != c])
        cells = [(r, c) for r in range(dim) if r != dropped_r] + [(dropped_r, new_c)]
    else:
        cells = _scatter_fail_cells(dim, count)

    assert len(cells) == count and len(set(cells)) == count
    assert len({r for r, _c in cells}) > 1 and len({c for _r, c in cells}) > 1
    random.shuffle(cells)
    return cells


def _query_ops(count: int) -> list[tuple[int, bool]]:
    assert count >= 2, "generator always needs room for one rotated and one flipped query"
    ops: list[tuple[int, bool]] = [random.choice(ROT_OPS), random.choice(FLIP_OPS)]
    while len(ops) < count:
        ops.append(random.choice(NON_IDENTITY_OPS))
    random.shuffle(ops)
    return ops[:count]


def _expected_output(
    grid: np.ndarray,
    row_starts: tuple[int, ...],
    col_starts: tuple[int, ...],
    top_sep: int,
    bottom_sep: int,
    cells: list[tuple[int, int]],
    success: bool,
) -> np.ndarray:
    out = grid.copy()
    for r_i, c_i in cells:
        cell = out[row_starts[r_i] : row_starts[r_i] + CELL, col_starts[c_i] : col_starts[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == BG] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    dim = random.choice(TABLE_DIMS)
    geom = _geometry(dim)
    row_starts = geom["row_starts"]
    col_starts = geom["col_starts"]
    assert isinstance(row_starts, tuple) and isinstance(col_starts, tuple)

    grid = np.full((geom["height"], geom["width"]), BG, dtype=int)
    grid[geom["top_sep"], :] = SEP
    grid[geom["bottom_sep"], :] = SEP

    total_cells = dim * dim
    table_specs = _table_specs(total_cells)
    glyphs: list[np.ndarray] = []
    for i, (mask, color) in enumerate(table_specs):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        r_i, c_i = divmod(i, dim)
        paste(grid, glyph, row_starts[r_i], col_starts[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells(dim)
    else:
        count = dim - 1 if random.random() < P_SHORT_FAIL else dim
        query_cells = _fail_cells(dim, count)

    ops = _query_ops(len(query_cells))
    exact_table_keys = {(color, mask) for mask, color in table_specs}
    exact_query_keys: list[tuple[int, int]] = []

    for slot, ((r_i, c_i), (rot_k, do_flip)) in enumerate(zip(query_cells, ops, strict=True)):
        table_mask, color = table_specs[r_i * dim + c_i]
        query_mask = _transform_mask(table_mask, rot_k, do_flip)
        assert query_mask != table_mask, "queries should visibly use the transform rule"
        exact_query_keys.append((color, query_mask))
        paste(grid, _glyph(query_mask, color), 0, col_starts[slot])

    assert not any(key in exact_table_keys for key in exact_query_keys), "query glyphs should not collapse to exact matching"

    expected = _expected_output(
        grid,
        row_starts,
        col_starts,
        int(geom["top_sep"]),
        int(geom["bottom_sep"]),
        query_cells,
        success,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
