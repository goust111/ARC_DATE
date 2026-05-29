# Combo variant of d8e07eb2 (index 13)
# combo_spec: D1/2+D12/1
# Name: 旋转匹配+两轮筛选

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


# Design analysis
# ---------------
# concretization and behavior trace:
# - The canvas geometry stays the source 30x22 layout: a 5-row query band, two
#   full-width separator rows, a 4x4 table of 5x5 cells, and a 2-row verdict band.
# - Each occupied 5x5 cell still contains a centered 3x3 monochrome glyph. The
#   fused change is entirely in how table cells are highlighted in the output.
# - Original behavior for one query glyph:
#   1. compare full colored 3x3 pattern against table cells;
#   2. only the exact copy turns green.
# - Fused behavior for one query glyph:
#   1. first collect every table cell with the same glyph color and fill that
#      cell background yellow;
#   2. among those yellow candidates, compare 3x3 occupancy only up to the full
#      rotation/flip family and upgrade the true match to green.
# - Small trace:
#   table has a maroon asymmetric "hook" and another maroon different shape.
#   query shows the hook after a 90-degree rotation.
#   original task: no cell matches exactly, so neither maroon cell is selected.
#   fused task: both maroon cells become yellow in round 1, then only the rotated
#   equivalent hook becomes green in round 2, leaving the other maroon cell yellow.
#
# complexity calibration:
# - The main ambiguity risk is having two table cells that share both color and
#   dihedral-equivalence class. That would make round 2 non-unique. The generator
#   prevents this by assigning the 16 table cells to 16 distinct asymmetric shape
#   classes under rotation/flip.
# - To keep the two-round process visually obvious, input glyph colors exclude
#   green (3) and yellow (4), which are reserved for output-only semantics.
# - Query colors are forced to be distinct and repeated elsewhere in the table.
#   This guarantees visible yellow-only leftovers instead of all yellow cells
#   immediately turning green.
#
# implicit assumptions review:
# - The source solver relied on absolute in-cell coordinates as part of the glyph
#   key. That assumption breaks once rotation/flip is allowed, so the new solver
#   extracts the centered 3x3 boolean mask and canonicalizes it over all eight
#   dihedral transforms.
# - The new round-1 color filter assumes glyph cells are monochrome and confined
#   to the inner 3x3 region; otherwise color extraction or mask comparison would
#   be ill-defined. The generator renders exactly that structure.
# - The success test still depends only on the final green matches forming one
#   full row or column and on all four query slots being filled. Yellow candidates
#   are intermediate evidence and must not affect the verdict.
# - Because round 2 now ignores orientation, the generator must ensure every query
#   glyph is a non-identity transform of its source table glyph; otherwise some
#   samples would silently collapse back to the simpler color+exact-shape rule.
#
# placement legality:
# - All object coordinates remain the original legal coordinates: query slots at
#   columns 1/6/11/16, table cells at rows 7/12/17/22 and the same columns.
# - Every glyph stays inside its 5x5 cell with a 1-pixel border, so there is no
#   overlap across cells and no out-of-bounds placement during transforms because
#   transforms apply only to the internal 3x3 mask before rendering.
#
# visual observability and cleanliness:
# - A human can infer the fusion from few examples because the table shows two
#   simultaneous clues: yellow marks same-color families, green marks the refined
#   rotation/flip winner, and some yellow candidates remain as first-pass traces.
# - Using asymmetric query transforms makes the second round visibly different
#   from the original exact-copy task: the green table glyph need not share the
#   query's literal orientation.
# - The output stays clean because only table-cell backgrounds are layered; glyph
#   pixels themselves, separator rows, and the verdict band never mix yellow and
#   green markings in the same pixels.
# - Quantitatively, non-degenerate samples should satisfy both:
#   1. at least one yellow-only table cell exists in the expected output; and
#   2. at least one query glyph has no exact table copy, so the fused solver's
#      green matches differ from an exact-shape baseline on the same input.


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


def _variant_masks(mask: int) -> tuple[int, ...]:
    return tuple(_mask_to_int(arr) for arr in _mask_variants(_int_to_mask(mask)))


def _canonical_mask(mask: np.ndarray) -> tuple[int, ...]:
    return min(tuple(int(x) for x in arr.reshape(-1)) for arr in _mask_variants(mask))


def _canonical_mask_int(mask: int) -> int:
    return min(_variant_masks(mask))


def _cell_info(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[int, int, tuple[int, ...]] | None:
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
    return color, _mask_to_int(inner), _canonical_mask(inner)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter transformed query glyphs through the reference table in two visible rounds.

    Input structure:
    - A teal background is split by two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells, each holding a centered
      monochrome 3x3 glyph.
    - The middle band contains a 4x4 table of the same 5x5 cells; colors can
      repeat across the table, and matching must ignore glyph orientation.

    Rule:
    - Round 1: fill table-cell background yellow when that cell's glyph color
      appears anywhere in the query band.
    - Round 2: among those color candidates, treat a table glyph as a match when
      its color agrees and its 3x3 occupancy matches a query glyph after some
      rotation or reflection; overwrite those final matches with green.
    - Use only the green round-2 matches for the verdict: if four queries are
      present and the green cells form one complete table row or one complete
      table column, also green-fill the query band and verdict band; otherwise
      leave the query band unchanged and fill the verdict band red.

    Key insight: Yellow records the coarse color shortlist, while green records
    the refined rotation/flip match. The answer comes from the green subset, not
    from literal query copies.
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

    query_infos = [
        info
        for c in col_starts
        if (info := _cell_info(g, 0, c, cell_size, bg)) is not None
    ]
    query_keys = {(color, canonical) for color, _exact, canonical in query_infos}
    assert len(query_infos) == len(query_keys), "query glyphs should be distinct under color+transform matching"
    query_colors = {color for color, _canonical in query_keys}

    table_keys = [
        _cell_info(g, r, c, cell_size, bg)
        for r in row_starts
        for c in col_starts
    ]
    assert all(key is not None for key in table_keys), "reference table should be fully populated"
    assert len({(color, canonical) for color, _exact, canonical in table_keys}) == len(
        table_keys
    ), "table glyphs must be unique under color+transform equivalence"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            info = _cell_info(g, r, c, cell_size, bg)
            assert info is not None
            color, _exact, canonical = info
            cell = out[r : r + cell_size, c : c + cell_size]
            bg_mask = g[r : r + cell_size, c : c + cell_size] == bg
            if color in query_colors:
                cell[bg_mask] = 4
            if (color, canonical) in query_keys:
                selected.append((int(ri), int(ci)))
                cell[bg_mask] = 3

    complete_query = len(query_infos) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_infos) and (same_row or same_col)

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
RED = 2
GREEN = 3
YELLOW = 4
SEP = 6
GLYPH_COLORS = (0, 1, 2, 5, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9

# -- Fixed repeated-color scaffold --
COLOR_INDEX_PATTERN = (
    (0, 1, 2, 3),
    (4, 5, 6, 0),
    (1, 2, 3, 4),
    (5, 6, 0, 1),
)

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35

# -- Generation --
MAX_RETRIES = 300
ROT_OPS = ((1, False), (2, False), (3, False))
FLIP_OPS = ((0, True), (1, True), (2, True), (3, True))
NON_IDENTITY_OPS = ROT_OPS + FLIP_OPS


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


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    cell[1:4, 1:4][_int_to_mask(mask)] = color
    return cell


def _color_grid() -> list[list[int]]:
    palette = list(GLYPH_COLORS)
    random.shuffle(palette)
    return [[palette[idx] for idx in row] for row in COLOR_INDEX_PATTERN]


def _table_masks() -> list[int]:
    classes = list(ASYMMETRIC_CLASSES)
    random.shuffle(classes)
    masks = [random.choice(_variant_masks(rep)) for rep in classes]
    assert len({_canonical_mask_int(mask) for mask in masks}) == len(masks)
    return masks


def _line_cells() -> list[tuple[int, int]]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _cells_have_distinct_colors(cells: list[tuple[int, int]], color_grid: list[list[int]]) -> bool:
    colors = [color_grid[r][c] for r, c in cells]
    return len(colors) == len(set(colors))


def _is_full_line(cells: list[tuple[int, int]]) -> bool:
    return len(cells) == TABLE_DIM and (
        len({r for r, _c in cells}) == 1 or len({c for _r, c in cells}) == 1
    )


def _fail_cells(count: int, color_grid: list[list[int]]) -> list[tuple[int, int]]:
    pool = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM)]
    for _ in range(500):
        cells = random.sample(pool, count)
        if _is_full_line(cells):
            continue
        if len({r for r, _c in cells}) == 1 or len({c for _r, c in cells}) == 1:
            continue
        if not _cells_have_distinct_colors(cells, color_grid):
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample a distinct-color fail set")


def _query_ops(count: int) -> list[tuple[int, bool]]:
    ops: list[tuple[int, bool]] = [random.choice(ROT_OPS), random.choice(FLIP_OPS)]
    while len(ops) < count:
        ops.append(random.choice(NON_IDENTITY_OPS))
    random.shuffle(ops)
    return ops[:count]


def _expected_output(
    grid: np.ndarray,
    query_cells: list[tuple[int, int]],
    color_grid: list[list[int]],
    success: bool,
) -> np.ndarray:
    out = grid.copy()
    query_colors = {color_grid[r][c] for r, c in query_cells}
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            bg_mask = grid[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL] == BG
            if color_grid[r_i][c_i] in query_colors:
                cell[bg_mask] = YELLOW
    for r_i, c_i in query_cells:
        cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
        bg_mask = grid[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL] == BG
        cell[bg_mask] = GREEN
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
    color_grid = _color_grid()
    table_specs: list[tuple[int, int]] = []
    exact_table_keys: set[tuple[int, int]] = set()
    canonical_table_keys: set[tuple[int, int]] = set()

    for i, mask in enumerate(masks):
        r_i, c_i = divmod(i, TABLE_DIM)
        color = color_grid[r_i][c_i]
        table_specs.append((mask, color))
        exact_table_keys.add((color, mask))
        canonical_table_keys.add((color, _canonical_mask_int(mask)))
        paste(grid, _glyph(mask, color), ROWS[r_i], COLS[c_i])

    assert len(canonical_table_keys) == TABLE_DIM * TABLE_DIM

    success = random.random() < P_SUCCESS
    query_cells = _line_cells() if success else _fail_cells(3 if random.random() < P_QUERY_3 else 4, color_grid)
    assert _cells_have_distinct_colors(query_cells, color_grid), "query colors must stay distinct"

    ops = _query_ops(len(query_cells))
    query_exact_keys: list[tuple[int, int]] = []
    query_canonical_keys: list[tuple[int, int]] = []
    for slot, ((r_i, c_i), (rot_k, do_flip)) in enumerate(zip(query_cells, ops, strict=True)):
        table_mask, color = table_specs[r_i * TABLE_DIM + c_i]
        query_mask = _transform_mask(table_mask, rot_k, do_flip)
        assert query_mask != table_mask, "queries must visibly use non-identity transforms"
        query_exact_keys.append((color, query_mask))
        query_canonical_keys.append((color, _canonical_mask_int(query_mask)))
        paste(grid, _glyph(query_mask, color), 0, COLS[slot])

    assert len(set(query_canonical_keys)) == len(query_canonical_keys)
    assert set(query_canonical_keys).issubset(canonical_table_keys)
    assert not any(key in exact_table_keys for key in query_exact_keys), "samples must not collapse to exact matching"

    expected = _expected_output(grid, query_cells, color_grid, success)
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
