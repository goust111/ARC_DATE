# Combo variant of d8e07eb2 (index 35)
# combo_spec: D1/2+D2/3+D5/1
# Name: 干扰查询+旋转匹配+2x2判定

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
# ---------------
# a) concretization and behavior trace:
# - The 30x22 canvas keeps the original separators, verdict band, and 4x4 table of
#   5x5 cells. The only visible layout change is the query band: instead of up to
#   four 5x5 cells, it now contains five compact 3x3 glyph slots at row 1 and
#   columns 1,5,9,13,17.
# - A typical successful sample can look like
#   [real A rotated 90, distractor X, real B flipped, real C rotated 180,
#    real D flipped]. In the source task, exact 5x5-copy matching would either
#   miss A/B/C/D after transformation or would wrongly treat X as part of the
#   decisive set. In the fused task, the solver canonicalizes each glyph up to
#   rotation/reflection, discards X because it has no table counterpart, and then
#   checks whether the four surviving matches occupy one contiguous 2x2 table block.
# - The behavioral delta is therefore three-step and coupled:
#   compact query extraction -> distractor filtering -> dihedral matching ->
#   2x2 spatial verdict.
#
# b) complexity calibration:
# - The 16 table glyphs are chosen from 16 distinct asymmetric dihedral classes.
#   This keeps transform matching unambiguous and visually meaningful because every
#   non-identity transform changes the exact 3x3 picture.
# - Every sample contains at least one distractor, so the filtering rule is always
#   active. Success cases use 4 real + 1 distractor; fail cases use 3-4 real and
#   the remaining 1-2 slots as distractors.
# - Distractors come from canonical shape classes absent from the table, which
#   prevents accidental matches even under the relaxed rotation/flip rule.
#
# c) implicit assumptions review:
# - The source solver assumed query glyphs lived in 5x5 cells and that absolute
#   foreground coordinates inside the cell were the signature. That breaks once the
#   query area becomes raw 3x3 patches and matching allows transforms. The fused
#   solver replaces that with a monochrome (color, canonical_mask) key.
# - The source success logic assumed every visible query was genuine and that four
#   visible queries implied a candidate full row/column. Both assumptions fail
#   here because five visible slots include distractors and the verdict geometry is
#   now 2D rather than 1D.
# - New solver assumptions are explicit: table glyphs stay centered in the inner
#   3x3 of each 5x5 cell, query glyphs occupy full 3x3 slots, and table keys are
#   unique under color plus dihedral class. The generator constructs inputs to
#   satisfy all three.
#
# d) placement legality:
# - Query slots are fixed non-overlapping 3x3 windows fully contained in rows 1..3.
# - Table cells stay on the original 5x5 lattice; real query sources are distinct;
#   distractors are inserted only into top-band slots.
# - `generate_input()` validates every candidate by running `main()` and comparing
#   against the expected output.
#
# e) visual observability:
# - A human can see that some top-band glyphs do not appear anywhere in the table,
#   and that the matching ones are rotated/flipped rather than literal copies.
# - Successful outputs highlight a compact 2x2 region, while failing outputs can
#   still highlight lines or scattered quartets, making the new spatial rule
#   distinguishable from the original row/column task in a few examples.
#
# f) visual cleanliness:
# - The query band contains a single row of evenly spaced 3x3 glyphs with one
#   blank column between slots. No extra markers are introduced; the only overlay
#   remains the original green background fill in matched table cells and the solid
#   verdict band.
#
# g) inferability:
# - No invisible numeric ordering or color arithmetic is used. Color remains a
#   visible identity feature, while the fused reasoning stays recoverable from
#   pixels alone: ignore unmatched queries, mentally rotate/flip the rest, then
#   test whether the matched cells form a contiguous 2x2 block.


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


def _mask_variants(mask: int) -> tuple[int, ...]:
    arr = _int_to_mask(mask)
    out: list[int] = []
    seen: set[int] = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        for cand in (rot, np.fliplr(rot)):
            key = _mask_to_int(cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return tuple(out)


def _canonical_mask_int(mask: int) -> int:
    return min(_mask_variants(mask))


def _glyph_key_from_patch(patch: np.ndarray, bg: int) -> tuple[int, int] | None:
    fg = patch != int(bg)
    if not bool(fg.any()):
        return None

    colors = {int(x) for x in patch[fg].tolist()}
    assert len(colors) == 1, "glyph must be monochrome"
    color = next(iter(colors))

    if patch.shape == (3, 3):
        mask = fg
    else:
        assert patch.shape == (5, 5), "glyph patch must be either query-sized or cell-sized"
        mask = fg[1:4, 1:4]
        outer = fg.copy()
        outer[1:4, 1:4] = False
        assert not bool(outer.any()), "table glyph pixels must stay inside the centered 3x3"

    return color, _canonical_mask_int(_mask_to_int(mask))


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


def _table_starts(g: np.ndarray, top_sep: int, bottom_sep: int, cell_size: int) -> tuple[list[int], list[int]]:
    _h, w = g.shape
    bg = get_bg(g)
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
    return row_starts, col_starts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter compact query glyphs through a transformed lookup table and test for a 2x2 block.

    Input structure:
    - The background is teal with two full-width pink separator rows.
    - The top band contains five compact 3x3 glyph slots; 3-4 of them are real
      lookups and the remaining 1-2 are distractors with no table match.
    - Between separators is the original 4x4 table of 5x5 cells, each holding a
      centered monochrome 3x3 glyph.

    Rule:
    - Match top-band glyphs to table glyphs by color plus 3x3 shape up to any
      rotation or horizontal reflection, and ignore top-band glyphs that have no
      such table counterpart.
    - Green-fill the background inside every matched table cell.
    - If the surviving matched cells are exactly four cells forming one contiguous
      2x2 table submatrix, also green-fill the query band and the verdict band;
      otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: The visible query list is noisy and transformed. Solving requires
    discarding unmatched distractors first, then comparing glyphs by dihedral
    equivalence, and only then checking the compact 2x2 geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cell_size = 5
    query_size = 3
    query_step = 4

    sep_rows = _constant_non_bg_rows(g, bg)
    assert len(sep_rows) >= 2, "expected two full-width separator rows"
    top_sep = sep_rows[0]
    bottom_sep = sep_rows[-1]

    row_starts, col_starts = _table_starts(g, top_sep, bottom_sep, cell_size)

    table_map: dict[tuple[int, int], tuple[int, int]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            cell = g[r : r + cell_size, c : c + cell_size]
            key = _glyph_key_from_patch(cell, bg)
            assert key is not None, "reference table should be fully populated"
            assert key not in table_map, "table glyphs must be unique under color+transform equivalence"
            table_map[key] = (int(ri), int(ci))

    query_row = max(0, (top_sep - query_size) // 2)
    query_cols = [c for c in range(col_starts[0], g.shape[1] - query_size + 1, query_step)]
    query_keys = []
    for c in query_cols:
        patch = g[query_row : query_row + query_size, c : c + query_size]
        key = _glyph_key_from_patch(patch, bg)
        if key is not None:
            query_keys.append(key)
    assert len(query_keys) == len(set(query_keys)), "query glyphs should be distinct under the matching rule"

    real_query_keys = {key for key in query_keys if key in table_map}

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for key in real_query_keys:
        ri, ci = table_map[key]
        selected.append((ri, ci))
        r = row_starts[ri]
        c = col_starts[ci]
        cell = out[r : r + cell_size, c : c + cell_size]
        cell[cell == bg] = 3

    success = len(real_query_keys) == 4 and len(selected) == 4 and _forms_2x2_block(selected)

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

# -- Query geometry --
QUERY_SIZE = 3
QUERY_ROW = 1
QUERY_COLS = (1, 5, 9, 13, 17)

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_REAL_QUERY_3 = 0.4
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


def _all_canonical_classes() -> tuple[int, ...]:
    reps = {
        _canonical_mask_int(mask)
        for mask in range(1 << 9)
        if FILL_MIN <= mask.bit_count() <= FILL_MAX and _has_full_bbox(mask)
    }
    return tuple(sorted(reps))


def _asymmetric_shape_classes() -> tuple[int, ...]:
    reps: set[int] = set()
    for mask in range(1 << 9):
        if not (FILL_MIN <= mask.bit_count() <= FILL_MAX and _has_full_bbox(mask)):
            continue
        if len(_mask_variants(mask)) != 8:
            continue
        reps.add(_canonical_mask_int(mask))
    out = tuple(sorted(reps))
    assert len(out) == TABLE_DIM * TABLE_DIM, "expected exactly sixteen asymmetric shape classes"
    return out


ALL_CANONICAL_CLASSES = _all_canonical_classes()
ASYMMETRIC_CLASSES = _asymmetric_shape_classes()
ROT_OPS = ((1, False), (2, False), (3, False))
FLIP_OPS = ((0, True), (1, True), (2, True), (3, True))
NON_IDENTITY_OPS = ROT_OPS + FLIP_OPS


def _transform_mask(mask: int, rot_k: int, do_flip: bool) -> int:
    arr = _int_to_mask(mask)
    out = np.rot90(arr, int(rot_k))
    if do_flip:
        out = np.fliplr(out)
    return _mask_to_int(out)


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
    masks = [random.choice(_mask_variants(rep)) for rep in classes]
    assert len({_canonical_mask_int(mask) for mask in masks}) == len(masks)
    return masks


def _table_glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    cell[1:4, 1:4][_int_to_mask(mask)] = color
    return cell


def _query_glyph(mask: int, color: int) -> np.ndarray:
    glyph = np.full((QUERY_SIZE, QUERY_SIZE), BG, dtype=int)
    glyph[_int_to_mask(mask)] = color
    return glyph


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


def _pick_distractors(
    table_classes: set[int],
    used_query_keys: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    available_classes = [cls for cls in ALL_CANONICAL_CLASSES if cls not in table_classes]
    distractors: list[tuple[int, int]] = []
    while len(distractors) < count:
        canonical = random.choice(available_classes)
        mask = random.choice(_mask_variants(canonical))
        color = random.choice(GLYPH_COLORS)
        key = (color, _canonical_mask_int(mask))
        if key in used_query_keys:
            continue
        used_query_keys.add(key)
        distractors.append((mask, color))
    return distractors


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
        paste(grid, _table_glyph(mask, color), ROWS[r_i], COLS[c_i])

    table_keys = {(color, _canonical_mask_int(mask)) for mask, color in table_specs}
    table_classes = {_canonical_mask_int(mask) for mask, _color in table_specs}
    exact_table_keys = {(color, mask) for mask, color in table_specs}
    assert len(table_keys) == TABLE_DIM * TABLE_DIM
    assert len(table_classes) == TABLE_DIM * TABLE_DIM

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _block_cells()
        real_count = BLOCK_AREA
    else:
        real_count = 3 if random.random() < P_REAL_QUERY_3 else BLOCK_AREA
        query_cells = _fail_cells(real_count)
        assert not _forms_2x2_block(query_cells)

    real_ops = _query_ops(len(query_cells))
    real_queries: list[tuple[int, int]] = []
    used_query_keys = set()
    exact_query_keys: list[tuple[int, int]] = []
    for (r_i, c_i), (rot_k, do_flip) in zip(query_cells, real_ops, strict=True):
        table_mask, color = table_specs[r_i * TABLE_DIM + c_i]
        query_mask = _transform_mask(table_mask, rot_k, do_flip)
        assert query_mask != table_mask, "real queries should visibly use rotation/flip"
        key = (color, _canonical_mask_int(query_mask))
        assert key in table_keys
        assert key not in used_query_keys
        used_query_keys.add(key)
        exact_query_keys.append((color, query_mask))
        real_queries.append((query_mask, color))

    assert not any(key in exact_table_keys for key in exact_query_keys), "real queries must not collapse back to exact matching"

    distractor_count = len(QUERY_COLS) - len(real_queries)
    assert distractor_count in (1, 2)
    distractors = _pick_distractors(table_classes, used_query_keys, distractor_count)
    assert all((color, _canonical_mask_int(mask)) not in table_keys for mask, color in distractors)

    query_items = [(mask, color, True) for mask, color in real_queries] + [
        (mask, color, False) for mask, color in distractors
    ]
    random.shuffle(query_items)
    assert len(query_items) == len(QUERY_COLS)

    for slot, (mask, color, _is_real) in enumerate(query_items):
        paste(grid, _query_glyph(mask, color), QUERY_ROW, QUERY_COLS[slot])

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
