# Combo variant of d8e07eb2 (index 25)
# combo_spec: D1/2+D10/2+D12/1
# Name: 旋转匹配+行颜色+两轮筛选

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        sys.path.append(str(_arc_utils.parent))
        break

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
#    The 30x22 scaffold stays the same as the source task: query band on rows
#    0..4, separator at row 5, a 4x4 table of 5x5 cells on rows 7..26, another
#    separator at row 27, and a 2-row verdict band at the bottom.
#    The fusion changes the internal coding of the table and the output process:
#    - Each table row has one uniform glyph color. Color therefore identifies
#      the row.
#    - Each table column has one shared 3x3 shape class, but the concrete mask
#      shown in each row can be a different rotation/flip variant. Shape up to
#      dihedral equivalence therefore identifies the column.
#    - The output is built in two visible rounds: first every same-color row
#      candidate is filled yellow, then the exact row+shape-class target cells
#      are upgraded to green.
#    Small trace:
#    - Original task: a query cell had to be a literal table glyph copy, so only
#      an exact color+mask match could turn green.
#    - Here: if the query is orange and its 3x3 mask is a rotated variant of the
#      column-2 family, round 1 marks the entire orange row yellow; round 2 uses
#      rotation/flip equivalence to find the one orange cell in column 2 and
#      turns only that cell green.
#
# b) Complexity calibration
#    The key ambiguity risk is having multiple columns share the same shape class
#    or having a query mask appear as an exact table mask. The generator blocks
#    both:
#    - the 4 columns use 4 distinct asymmetric dihedral classes
#    - each column uses 4 table variants plus a fifth reserved query-only variant
#      not present in the table
#    This keeps the factorization readable while making the rotation rule
#    necessary on every generated puzzle.
#
# c) Implicit-assumption review against the source solver
#    - Full-cell signature lookup no longer works. The source assumed one global
#      exact (color, shape) catalogue; now color repeats across an entire row and
#      shape repeats across an entire column modulo rotation/flip.
#    - The source wrote green directly for final matches. That breaks the two-pass
#      requirement, so the combo solver must first write yellow row candidates,
#      then overwrite resolved cells with green.
#    - The source treated 3x3 shape coordinates as absolute. Rotation/flip
#      equivalence breaks that assumption, so the combo solver canonicalizes the
#      3x3 occupancy mask before column lookup.
#
# d) Placement legality
#    No new objects are added, so the original geometry remains legal. All glyphs
#    still occupy the centered inner 3x3 of a 5x5 cell. Query glyphs still use
#    the four fixed top slots, with an optional empty slot when only 3 queries
#    are present.
#
# e) Visual observability
#    A human can see the fused logic from a few examples because:
#    - the table visibly forms horizontal color bands
#    - within each vertical column, silhouettes are recognizably the same family
#      after mental rotation/flip
#    - yellow keeps the first-pass row filter visible, while green shows the
#      second-pass shape refinement
#    The generator biases successful examples toward columns so the first-pass
#    trace remains visible even on successes; a row success would collapse yellow
#    and green into the same four cells and hide the D12 effect.
#
# f) Visual cleanliness
#    Only cell backgrounds are recolored in the output. Glyph strokes, separators,
#    and the verdict band keep their distinct visual roles. Yellow and green are
#    reserved as output-only colors, avoiding ambiguity with input glyph colors.
#
# g) Change inferability
#    The number of latent categories is intentionally small: 4 row colors and 4
#    column shape families. Each row color repeats four times and each shape
#    family repeats four times, so both mappings are supported by repetition
#    rather than hidden arbitrary lookup.
#
# New solver assumptions and generator guarantees
# ----------------------------------------------
# - Every non-empty cell is monochrome and confined to the centered inner 3x3.
#   `_cell_feature(...)` asserts this, and `_glyph(...)` generates exactly that.
# - Table rows are color-uniform and row colors are pairwise distinct.
#   `_row_colors()` samples 4 distinct colors; table assembly repeats them across
#   the row.
# - Table columns are canonical-shape-uniform and those canonical classes are
#   pairwise distinct. `_column_representatives()` chooses 4 different asymmetric
#   classes, then each row receives a different variant of the column class.
# - Every query resolves to one unique table cell. The generator chooses one row
#   color and one column class per query, and reserved query-only variants ensure
#   the original exact matcher cannot solve the instance by accident.


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
        rotated = np.rot90(mask, rot_k)
        for arr in (rotated, np.fliplr(rotated)):
            key = tuple(int(x) for x in arr.reshape(-1))
            if key in seen:
                continue
            seen.add(key)
            variants.append(arr.copy())
    return tuple(variants)


def _variant_masks(mask: int) -> tuple[int, ...]:
    return tuple(_mask_to_int(arr) for arr in _mask_variants(_int_to_mask(mask)))


def _canonical_mask_int(mask: int) -> int:
    return min(_variant_masks(mask))


def _cell_feature(
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve query glyphs through a row-color pass and a rotation-aware shape pass.

    Input structure:
    - Teal background with two full-width separator rows.
    - Above the first separator is a query row of 5x5 glyph cells.
    - Between separators is a 4x4 reference table of the same cell size; each
      row shares one glyph color, while each column shares one 3x3 shape family
      up to rotation or reflection.

    Rule:
    - First use each query glyph's color to locate its candidate table row and
      fill every background pixel in that row's 5x5 cells yellow.
    - Then, within those color-filtered candidates, use the query glyph's 3x3
      shape up to rotation/flip equivalence to choose the column, and upgrade the
      resolved cell background to green.
    - Judge success from the green cells only: if four queries are present and
      those resolved cells make one full table row or one full table column, also
      green-fill the query band and verdict band; otherwise fill the verdict band red.

    Key insight: The query glyph is decoded compositionally and in order: color
    tells you which row survives round 1, then mental rotation/flip of the 3x3
    silhouette tells you which column survives round 2.
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

    table_features: list[list[tuple[int, int, int]]] = []
    for r in row_starts:
        row_features = [_cell_feature(g, r, c, cell_size, bg) for c in col_starts]
        assert all(feat is not None for feat in row_features), "table cells must all be populated"
        table_features.append([feat for feat in row_features if feat is not None])

    row_colors: list[int] = []
    col_classes: list[int] = []
    for ri, row_features in enumerate(table_features):
        colors = [color for color, _exact, _canon in row_features]
        canons = [canon for _color, _exact, canon in row_features]
        assert len(set(colors)) == 1, "each table row must use one shared color"
        row_colors.append(colors[0])
        if ri == 0:
            col_classes = canons
        else:
            assert canons == col_classes, "each table column must repeat one shape class"

    assert len(set(row_colors)) == len(row_starts), "row colors must distinguish rows"
    assert len(set(col_classes)) == len(col_starts), "column shape classes must distinguish columns"
    color_to_row = {color: ri for ri, color in enumerate(row_colors)}
    canon_to_col = {canon: ci for ci, canon in enumerate(col_classes)}

    query_features = [
        feat
        for c in col_starts
        if (feat := _cell_feature(g, 0, c, cell_size, bg)) is not None
    ]
    assert query_features, "query band must contain at least one glyph"

    out = g.copy()

    candidate_rows: set[int] = set()
    for color, _exact, _canon in query_features:
        assert color in color_to_row, "query color must match a table row"
        candidate_rows.add(color_to_row[color])
    for ri in candidate_rows:
        r = row_starts[ri]
        for c in col_starts:
            cell = out[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = 4

    selected: list[tuple[int, int]] = []
    for color, _exact, canon in query_features:
        assert canon in canon_to_col, "query shape class must match a table column"
        ri = color_to_row[color]
        ci = canon_to_col[canon]
        selected.append((int(ri), int(ci)))
        r = row_starts[ri]
        c = col_starts[ci]
        cell = out[r : r + cell_size, c : c + cell_size]
        cell[cell == bg] = 3

    assert len(selected) == len(set(selected)), "query glyphs should resolve to distinct cells"
    complete_query = len(query_features) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator (combo variant) ---------------

# -- Grid layout --
GRID_H, GRID_W = 30, 22
TOP_SEP_ROW = 5
BOTTOM_SEP_ROW = 27
QUERY_BAND_H = 5
VERDICT_ROW = 28

# -- Colors --
BG = 8
RED = 2
GREEN = 3
YELLOW = 4
SEP = 6
GLYPH_COLORS = (0, 1, 2, 5, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
PREFERRED_FILL_COUNTS = (5, 6, 7, 8, 9)
ALL_OPS = (
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
)

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35

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


def _asymmetric_classes_by_fill() -> dict[int, tuple[int, ...]]:
    by_fill: dict[int, list[int]] = {k: [] for k in range(FILL_MIN, FILL_MAX + 1)}
    seen: set[int] = set()
    for mask in range(1 << 9):
        if not (FILL_MIN <= mask.bit_count() <= FILL_MAX and _has_full_bbox(mask)):
            continue
        variants = _variant_masks(mask)
        if len(variants) != len(ALL_OPS):
            continue
        canon = min(variants)
        if canon in seen:
            continue
        seen.add(canon)
        by_fill[mask.bit_count()].append(canon)
    return {k: tuple(sorted(v)) for k, v in by_fill.items() if v}


ASYMMETRIC_CLASSES_BY_FILL = _asymmetric_classes_by_fill()
AVAILABLE_FILLS = tuple(sorted(ASYMMETRIC_CLASSES_BY_FILL))


def _row_colors() -> list[int]:
    return random.sample(GLYPH_COLORS, TABLE_DIM)


def _column_representatives() -> list[int]:
    if len(AVAILABLE_FILLS) >= TABLE_DIM:
        fills = random.sample(AVAILABLE_FILLS, TABLE_DIM)
        random.shuffle(fills)
        return [random.choice(ASYMMETRIC_CLASSES_BY_FILL[k]) for k in fills]

    pool = [rep for reps in ASYMMETRIC_CLASSES_BY_FILL.values() for rep in reps]
    reps = random.sample(pool, TABLE_DIM)
    random.shuffle(reps)
    return reps


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = _int_to_mask(mask)
    cell[1:4, 1:4][bits] = color
    return cell


def _success_cells() -> list[tuple[int, int]]:
    c = random.randrange(TABLE_DIM)
    cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    if count == 3:
        rows = random.sample(range(TABLE_DIM), 3)
        cols = [random.randrange(TABLE_DIM) for _ in rows]
        if len(set(cols)) == 1:
            idx = random.randrange(3)
            cols[idx] = (cols[idx] + random.randrange(1, TABLE_DIM)) % TABLE_DIM
        cells = list(zip(rows, cols, strict=True))
    else:
        rows = list(range(TABLE_DIM))
        cols = [random.randrange(TABLE_DIM) for _ in rows]
        while len(set(cols)) == 1:
            cols[random.randrange(TABLE_DIM)] = random.randrange(TABLE_DIM)
        cells = list(zip(rows, cols, strict=True))
    assert len(cells) == count and len(set(cells)) == count
    random.shuffle(cells)
    return cells


def _expected_output(grid: np.ndarray, query_cells: list[tuple[int, int]], success: bool) -> np.ndarray:
    out = grid.copy()
    candidate_rows = {r for r, _c in query_cells}
    for r_i in candidate_rows:
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            cell[cell == BG] = YELLOW
    for r_i, c_i in query_cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
        out[VERDICT_ROW:, :] = GREEN
    else:
        out[VERDICT_ROW:, :] = RED
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    bg = BG
    row_colors: list[int] = []
    col_canons: list[int] = []
    table_exact: dict[tuple[int, int], int] = {}
    for ri, r in enumerate(ROWS):
        colors: list[int] = []
        canons: list[int] = []
        for ci, c in enumerate(COLS):
            feat = _cell_feature(grid, r, c, CELL, bg)
            assert feat is not None
            color, exact, canon = feat
            colors.append(color)
            canons.append(canon)
            table_exact[(ri, ci)] = exact
        row_colors.append(colors[0])
        if ri == 0:
            col_canons = canons
    color_to_row = {color: ri for ri, color in enumerate(row_colors)}
    canon_to_col = {canon: ci for ci, canon in enumerate(col_canons)}

    query_feats = [_cell_feature(grid, 0, c, CELL, bg) for c in COLS]
    query_feats = [feat for feat in query_feats if feat is not None]
    selected = [(color_to_row[color], canon_to_col[canon]) for color, _exact, canon in query_feats]
    exact_hits = sum(int(exact == table_exact[(ri, ci)]) for (color, exact, canon), (ri, ci) in zip(query_feats, selected, strict=True))
    candidate_rows = {ri for ri, _ci in selected}
    yellow_only_cells = len(candidate_rows) * TABLE_DIM - len(selected)
    return {
        "query_count": len(query_feats),
        "candidate_rows": len(candidate_rows),
        "yellow_only_cells": yellow_only_cells,
        "exact_hits": exact_hits,
    }


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    row_colors = _row_colors()
    col_reps = _column_representatives()
    table_masks = [[0] * TABLE_DIM for _ in range(TABLE_DIM)]
    query_masks: list[int] = [0] * TABLE_DIM

    for ci, rep in enumerate(col_reps):
        ops = random.sample(list(ALL_OPS), TABLE_DIM + 1)
        for ri in range(TABLE_DIM):
            table_masks[ri][ci] = _transform_mask(rep, *ops[ri])
        query_masks[ci] = _transform_mask(rep, *ops[-1])

    for ri, row_color in enumerate(row_colors):
        for ci in range(TABLE_DIM):
            paste(grid, _glyph(table_masks[ri][ci], row_color), ROWS[ri], COLS[ci])

    success = random.random() < P_SUCCESS
    query_cells = _success_cells() if success else _fail_cells(3 if random.random() < P_QUERY_3 else 4)

    for slot, (ri, ci) in enumerate(query_cells):
        paste(grid, _glyph(query_masks[ci], row_colors[ri]), 0, COLS[slot])

    metrics = _behavior_metrics(grid)
    expected = _expected_output(grid, query_cells, success)
    if metrics["yellow_only_cells"] <= 0:
        return None
    if metrics["exact_hits"] != 0:
        return None
    if not np.any(expected == YELLOW):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
