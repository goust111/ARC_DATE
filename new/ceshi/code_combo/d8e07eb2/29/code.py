# Combo variant of d8e07eb2 (index 29)
# combo_spec: D1/2+D5/1+D8/2
# Name: 旋转匹配+2x2判定+对称字形

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        ROOT = _parent
        break
assert ROOT is not None, "could not locate project root"
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, paste


# Design analysis
# ----------------
# a) concretization and behavior trace:
#    The canvas stays 30x22 with the original three-band layout: query cells in
#    rows 0..4, separator rows at 5 and 27, a 4x4 table of 5x5 cells in rows
#    7..26, and a 2-row verdict band at the bottom. Each 5x5 cell still contains
#    a centered monochrome 3x3 glyph. The fused changes are:
#    - lookup key: color + canonical dihedral class of the 3x3 occupancy mask,
#      so a query can match after rotation or reflection;
#    - verdict geometry: success now requires the four matched table cells to be
#      exactly one contiguous 2x2 block, not a full row or column;
#    - glyph vocabulary: every table glyph belongs to a symmetric shape class
#      (horizontal, vertical, or 180-degree symmetry), and the table reuses only
#      eight canonical shape classes across sixteen cells.
#    Small trace: if a blue center-symmetric zigzag appears in the table and the
#    query contains the same blue zigzag rotated 180 degrees, the original task
#    would compare raw pixel coordinates and reject it unless the raster happened
#    to be identical. Here both reduce to the same canonical class, so the cell is
#    selected. Four such selections turn the verdict green only when their table
#    coordinates are {(r,c),(r,c+1),(r+1,c),(r+1,c+1)}.
#
# b) complexity calibration:
#    Symmetry reduces the number of distinct shapes, which could make matching too
#    ambiguous if color+shape pairs repeated. The generator avoids that by using
#    each of the eight ARC glyph colors exactly twice and pairing them with eight
#    canonical symmetric shape classes via a derangement, so every (color,
#    canonical-shape) pair is unique even though shapes repeat across colors.
#    Samples also force both "trivial" and "non-trivial" transforms: at least one
#    query uses a non-identity rotation that leaves its symmetric glyph unchanged,
#    while another query uses a non-identity transform that changes the visible
#    raster. That keeps the rotation rule visible instead of collapsing to exact
#    copy matching or to arbitrary top-row decoration.
#
# c) implicit assumptions review:
#    The source `_cell_signature` encoded absolute foreground coordinates, which
#    breaks once rotation/reflection is allowed; it is replaced by `_cell_key`
#    returning `(color, canonical_mask)`. The line-based success test no longer
#    applies, so `main()` must check an adjacent 2x2 footprint instead of
#    "all rows equal" or "all cols equal". New assumptions introduced by the
#    solver are: each cell contains at most one monochrome glyph, all glyph pixels
#    stay inside the centered 3x3 window, and the reference table is unique under
#    `(color, canonical_mask)`. The generator enforces all three.
#
# d) placement legality:
#    Cell anchors, separator rows, and verdict band are unchanged from the source
#    spec, so there is no new overlap risk. Query glyphs are pasted only into the
#    four 5x5 slots at columns 1/6/11/16. Table glyphs are pasted only into the
#    fixed 4x4 lattice. All masks are 3x3 and are rendered inside the inner 3x3
#    of a 5x5 cell, so every placement remains in bounds with a one-pixel margin.
#
# e) visual observability:
#    The transformed matching is inferable because generated puzzles always mix a
#    query that visibly differs from its table counterpart with another query whose
#    symmetry makes a non-identity rotation look unchanged. The 2x2 verdict rule
#    is inferable because successful samples highlight a compact square, while fail
#    samples include rows, columns, gapped rectangles, or scattered quartets that
#    would have behaved differently in the original task.
#
# f) visual cleanliness:
#    No extra legend or marker is added. The only output overlays are the original
#    green cell backgrounds, optional green query-band background on success, and
#    a full-width verdict band. Shape repetition is limited to two instances per
#    canonical class so the table still looks structured instead of noisy.
#
# g) inference of changed variables:
#    The fusion does not introduce any hidden color remapping. Colors continue to
#    identify glyph instances, while the new reasoning load comes from dihedral
#    shape equivalence plus 2x2 spatial arrangement. Because each canonical shape
#    appears in two colors, humans can notice that color alone is insufficient,
#    and because each color appears with two different shapes, shape class alone is
#    also insufficient: both cues must be fused.


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
    raw: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        raw.append(rot)
        raw.append(np.fliplr(rot))
    uniq: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for arr in raw:
        key = tuple(int(x) for x in arr.reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(arr.copy())
    return tuple(uniq)


def _variant_masks(mask: int) -> tuple[int, ...]:
    return tuple(_mask_to_int(arr) for arr in _mask_variants(_int_to_mask(mask)))


def _canonical_mask_int(mask: int) -> int:
    return min(_variant_masks(mask))


def _transform_mask(mask: int, rot_k: int, do_flip: bool) -> int:
    arr = np.rot90(_int_to_mask(mask), int(rot_k))
    if do_flip:
        arr = np.fliplr(arr)
    return _mask_to_int(arr)


def _symmetry_types(mask: int) -> tuple[str, ...]:
    arr = _int_to_mask(mask)
    kinds: list[str] = []
    if np.array_equal(arr, np.flipud(arr)):
        kinds.append("h")
    if np.array_equal(arr, np.fliplr(arr)):
        kinds.append("v")
    if np.array_equal(arr, np.rot90(arr, 2)):
        kinds.append("c")
    return tuple(kinds)


def _cell_key(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[int, int] | None:
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
    return color, _canonical_mask_int(_mask_to_int(inner))


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
    Match transformed symmetric query glyphs to the table and test for a 2x2 block.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells.
    - The middle band is a 4x4 table of 5x5 cells; each populated cell contains
      one centered 3x3 monochrome glyph drawn from a symmetric shape family.

    Rule:
    - A query glyph matches a table glyph when the colors agree and the 3x3
      occupancy pattern matches up to rotation or reflection; matched table-cell
      background pixels are filled green.
    - If there are exactly four query glyphs and their matched table positions are
      exactly one contiguous 2x2 submatrix, also green-fill the query-band
      background and the full verdict band.
    - Otherwise keep the query band unchanged and fill the verdict band red.

    Key insight: Some query glyphs are symmetric enough that a non-identity
    rotation looks unchanged, while others must be mentally rotated or flipped;
    the verdict then depends on those matches forming a compact square block.
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

    row_starts = [r for r in range(row0, bottom_sep, cell_size) if r + cell_size <= bottom_sep]
    col_starts = [c for c in range(col0, w, cell_size) if c + cell_size <= w]
    assert row_starts and col_starts, "could not infer table cell grid"

    query_keys = [key for c in col_starts if (key := _cell_key(g, 0, c, cell_size, bg)) is not None]
    query_set = set(query_keys)
    assert len(query_keys) == len(query_set), "query glyphs should be distinct under color+canonical shape"

    table_keys = [_cell_key(g, r, c, cell_size, bg) for r in row_starts for c in col_starts]
    assert all(key is not None for key in table_keys), "reference table should be fully populated"
    assert len(set(table_keys)) == len(table_keys), "table glyphs must be unique under color+canonical shape"

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
TABLE_SHAPE_CLASSES = 8

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("line", "gapped_rect", "bent", "scatter")

# -- Generation --
MAX_RETRIES = 400

ROTATION_OPS = ((1, False), (2, False), (3, False))
FLIP_OPS = ((0, True), (1, True), (2, True), (3, True))
NON_IDENTITY_OPS = ROTATION_OPS + FLIP_OPS


def _has_full_bbox(mask: int) -> bool:
    bits = [(mask >> i) & 1 for i in range(9)]
    return (
        any(bits[0:3])
        and any(bits[6:9])
        and any(bits[i] for i in (0, 3, 6))
        and any(bits[i] for i in (2, 5, 8))
    )


def _symmetric_shape_reps() -> tuple[int, ...]:
    reps: set[int] = set()
    for mask in range(1 << 9):
        if not (FILL_MIN <= mask.bit_count() <= FILL_MAX and _has_full_bbox(mask)):
            continue
        if not _symmetry_types(mask):
            continue
        reps.add(_canonical_mask_int(mask))
    out = tuple(sorted(reps))
    assert len(out) >= TABLE_SHAPE_CLASSES
    return out


SYMMETRIC_REPS = _symmetric_shape_reps()
CLASS_VARIANTS = {rep: _variant_masks(rep) for rep in SYMMETRIC_REPS}
FULL_ROT_REPS = tuple(rep for rep in SYMMETRIC_REPS if len(set(CLASS_VARIANTS[rep])) == 1)
HALF_TURN_REPS = tuple(
    rep for rep in SYMMETRIC_REPS if _transform_mask(rep, 2, False) == rep and rep not in FULL_ROT_REPS
)
ORBIT4_REPS = tuple(rep for rep in SYMMETRIC_REPS if len(set(CLASS_VARIANTS[rep])) >= 4)


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    cell[1:4, 1:4][_int_to_mask(mask)] = color
    return cell


def _derangement(n: int) -> list[int]:
    base = list(range(n))
    while True:
        perm = base.copy()
        random.shuffle(perm)
        if all(i != perm[i] for i in base):
            return perm


def _pick_shape_reps() -> list[int]:
    chosen: list[int] = []
    chosen.append(random.choice(FULL_ROT_REPS))
    chosen.append(random.choice(HALF_TURN_REPS))

    orbit4_pool = [rep for rep in ORBIT4_REPS if rep not in chosen]
    chosen.extend(random.sample(orbit4_pool, 4))

    remaining = [rep for rep in SYMMETRIC_REPS if rep not in chosen]
    chosen.extend(random.sample(remaining, TABLE_SHAPE_CLASSES - len(chosen)))
    random.shuffle(chosen)
    return chosen


def _table_specs() -> list[tuple[int, int, int]]:
    reps = _pick_shape_reps()
    colors = list(GLYPH_COLORS)
    random.shuffle(colors)
    perm = _derangement(TABLE_SHAPE_CLASSES)

    pairs = [(colors[i], reps[i]) for i in range(TABLE_SHAPE_CLASSES)]
    pairs.extend((colors[i], reps[perm[i]]) for i in range(TABLE_SHAPE_CLASSES))
    random.shuffle(pairs)

    specs = [(random.choice(CLASS_VARIANTS[rep]), color, rep) for color, rep in pairs]
    assert len(specs) == TABLE_DIM * TABLE_DIM
    assert len({(color, rep) for _mask, color, rep in specs}) == len(specs)
    assert len({rep for _mask, _color, rep in specs}) == TABLE_SHAPE_CLASSES
    return specs


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


def _same_rotation_ops(mask: int) -> list[tuple[int, bool]]:
    return [op for op in ROTATION_OPS if _transform_mask(mask, *op) == mask]


def _changed_rotation_ops(mask: int) -> list[tuple[int, bool]]:
    return [op for op in ROTATION_OPS if _transform_mask(mask, *op) != mask]


def _flip_ops(mask: int, *, prefer_changed: bool) -> list[tuple[int, bool]]:
    ops = [op for op in FLIP_OPS if _transform_mask(mask, *op) != mask]
    if ops or prefer_changed:
        return ops
    return list(FLIP_OPS)


def _choose_query_ops(selected_masks: list[int]) -> list[tuple[int, bool]] | None:
    count = len(selected_masks)
    same_rot = {i: _same_rotation_ops(mask) for i, mask in enumerate(selected_masks)}
    changed_rot = {i: _changed_rotation_ops(mask) for i, mask in enumerate(selected_masks)}
    flip_like = {i: _flip_ops(mask, prefer_changed=False) for i, mask in enumerate(selected_masks)}

    trivial_indices = [i for i, ops in same_rot.items() if ops]
    changed_indices = [i for i, ops in changed_rot.items() if ops]
    flip_indices = [i for i, ops in flip_like.items() if ops]

    for i in trivial_indices:
        for j in changed_indices:
            if j == i:
                continue
            if count >= 3:
                for k in flip_indices:
                    if k in (i, j):
                        continue
                    ops: list[tuple[int, bool] | None] = [None] * count
                    ops[i] = random.choice(same_rot[i])
                    ops[j] = random.choice(changed_rot[j])
                    ops[k] = random.choice(flip_like[k])
                    for idx in range(count):
                        if ops[idx] is not None:
                            continue
                        candidates = changed_rot[idx] or flip_like[idx] or list(NON_IDENTITY_OPS)
                        ops[idx] = random.choice(candidates)
                    out = [op for op in ops if op is not None]
                    assert len(out) == count
                    return out
            else:
                ops = [random.choice(changed_rot[idx] or flip_like[idx] or list(NON_IDENTITY_OPS)) for idx in range(count)]
                ops[i] = random.choice(same_rot[i])
                ops[j] = random.choice(changed_rot[j])
                return ops
    return None


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

    specs = _table_specs()
    glyphs: list[np.ndarray] = []
    for i, (mask, color, _rep) in enumerate(specs):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    canon_pairs = {(color, rep) for _mask, color, rep in specs}
    assert len(canon_pairs) == len(specs)

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _block_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else BLOCK_AREA
        query_cells = _fail_cells(count)

    selected_masks = [specs[r_i * TABLE_DIM + c_i][0] for r_i, c_i in query_cells]
    ops = _choose_query_ops(selected_masks)
    if ops is None:
        return None

    saw_same = False
    saw_changed = False
    saw_flip = False
    for slot, ((r_i, c_i), (rot_k, do_flip)) in enumerate(zip(query_cells, ops, strict=True)):
        table_mask, color, _rep = specs[r_i * TABLE_DIM + c_i]
        query_mask = _transform_mask(table_mask, rot_k, do_flip)
        if query_mask == table_mask:
            saw_same = True
        else:
            saw_changed = True
        if do_flip:
            saw_flip = True
        paste(grid, _glyph(query_mask, color), 0, COLS[slot])

    if not (saw_same and saw_changed and saw_flip):
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
    raise RuntimeError("failed to generate a valid fused d8e07eb2 input")
