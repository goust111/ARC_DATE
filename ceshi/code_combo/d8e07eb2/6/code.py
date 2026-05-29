# Combo variant of d8e07eb2 (index 6)
# combo_spec: D2/3+D5/1
# Name: 含干扰查询+2x2子矩阵判定
#
# Design analysis
# ---
# a) Concrete behavior and trace:
#    - The 30x22 canvas keeps the original separators and 4x4 reference table.
#    - The query band cannot fit 4 real 5x5 cells plus 1-2 extra distractors, so
#      only the source region is compacted into five 3x3 windows at row 1 and
#      columns 1,5,9,13,17. The table stays unchanged as 5x5 cells containing
#      centered 3x3 glyphs.
#    - Example trace: top slots show [real A, distractor X, real B, real C, real D].
#      Original task logic would treat every visible query as meaningful and then
#      ask whether the matched cells form a line. The fused task must instead:
#      (1) normalize the compact top glyphs to the same color+shape signature used
#          by the table,
#      (2) discard X because it is absent from the table,
#      (3) highlight only A/B/C/D in the table,
#      (4) accept only if those four matches are exactly one contiguous 2x2 block.
#      So a filtered match set {(1,1),(1,2),(2,1),(2,2)} succeeds, while a filtered
#      full row {(0,0),(0,1),(0,2),(0,3)} now fails even though it would have passed
#      in the original problem.
# b) Complexity calibration:
#    - Distractors are always present, so the D2 change is visible in every sample.
#      Success cases use 4 real + 1 distractor; fail cases use 3 real + 2 distractors
#      or 4 real + 1 distractor.
#    - Table glyphs remain unique by full signature, so filtering is unambiguous.
#    - Fail generation explicitly includes line-shaped 4-match cases. That prevents
#      the task from visually collapsing back to the original row/column criterion.
# c) Implicit assumption review against the old solver:
#    - Old query extraction assumed each visible top cell was a real query and that
#      queries occupied 5x5 table-aligned slots. Both assumptions break, so the new
#      solver reads five compact 3x3 windows and normalizes them.
#    - Old success logic assumed "all visible queries matter" and "four matches in one
#      row/column means success". The fused solver instead filters by table membership
#      first, then requires exactly four surviving matches forming adjacent rows and
#      adjacent columns.
#    - Old whole-cell signatures on the query band no longer work because the query
#      glyphs no longer carry the 1-pixel border. The new normalized signature removes
#      translation so table cells and compact queries compare cleanly.
# d) Placement legality:
#    - Query windows are fixed non-overlapping 3x3 patches fully inside rows 1..3.
#    - Table cells keep the original anchors, so no object placement changes there.
#    - Distractors are drawn from shapes absent from the table, which guarantees they
#      cannot accidentally match any table cell regardless of color choice.
# e) Visual observability:
#    - Humans can see that some top glyphs have no counterpart in the table because
#      those top glyphs never produce a green table highlight.
#    - Success examples show a compact green 2x2 region in the table; fail examples
#      can still show four matched cells, including full rows/columns, but keep the
#      verdict red. That makes both dimensions inferable from a few examples.
# f) Visual cleanliness:
#    - Only one layer of top-band glyphs is present and the slots are evenly spaced.
#    - Output overlays remain the same sparse fills as the base task: green inside
#      matched cells, plus a solid green/red verdict band.
# g) New solver assumptions and generator guarantees:
#    - `main()` assumes exactly two separator rows exist, the table grid is aligned to
#      a 5x5 lattice, and query signatures are distinct. `generate_input()` constructs
#      samples to satisfy all three conditions.
#    - `main()` assumes every real query has exactly one table match. The generator
#      enforces globally unique table signatures, then selects real queries directly
#      from table cells.

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


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _normalized_signature(patch: np.ndarray, bg: int) -> tuple[tuple[int, int, int], ...] | None:
    mask = patch != int(bg)
    if not bool(mask.any()):
        return None
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    return tuple((int(r - r0), int(c - c0), int(patch[r, c])) for r, c in coords)


def _cell_signature(
    g: np.ndarray, r0: int, c0: int, h: int, w: int, bg: int
) -> tuple[tuple[int, int, int], ...] | None:
    patch = g[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)]
    return _normalized_signature(patch, bg)


def _forms_2x2_block(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> bool:
    unique = {(int(r), int(c)) for r, c in cells}
    if len(unique) != 4:
        return False
    rows = sorted({r for r, _c in unique})
    cols = sorted({c for _r, c in unique})
    return (
        len(rows) == 2
        and len(cols) == 2
        and rows[1] == rows[0] + 1
        and cols[1] == cols[0] + 1
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the top-band distractors, then test whether the surviving matches form a 2x2 block.

    Input structure:
    - The canvas keeps the original teal background and two full-width pink separator rows.
    - The top band now contains five compact 3x3 query glyph slots; 3-4 of them are genuine
      lookups copied from the table and 1-2 are distractors with no table counterpart.
    - Between the separators is the original 4x4 reference table of 5x5 cells, each holding
      one centered 3x3 glyph whose identity is the visible combination of shape and color.

    Rule:
    - Normalize each compact top glyph and compare it against the table signatures, ignoring
      any top glyph that does not exist in the reference table at all.
    - Green-fill the background of every table cell matched by the surviving real queries.
    - If exactly four real matches remain and those table positions are exactly one contiguous
      2x2 submatrix, also green-fill the query band and the verdict band; otherwise leave the
      query band unchanged and fill the verdict band red.

    Key insight: The top band is no longer a pure list of valid lookups, and the success test is
    no longer linear. The solver must first reject the distractors by visible color+shape match,
    then apply a spatial 2x2-block check only to the filtered table positions.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = get_bg(g)
    cell_size = 5
    query_size = 3
    query_step = 4
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

    table_map: dict[tuple[tuple[int, int, int], ...], tuple[int, int]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, cell_size, bg)
            if sig is None:
                continue
            assert sig not in table_map, "table glyph signatures must be unique"
            table_map[sig] = (int(ri), int(ci))

    query_r0 = max(0, (top_sep - query_size) // 2)
    query_cols = [c for c in range(col0, w - query_size + 1, query_step)]
    query_sigs = [
        sig
        for c in query_cols
        if (sig := _cell_signature(g, query_r0, c, query_size, query_size, bg)) is not None
    ]
    assert len(query_sigs) == len(set(query_sigs)), "query glyphs should be distinct"

    real_query_sigs = {sig for sig in query_sigs if sig in table_map}

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for sig in real_query_sigs:
        ri, ci = table_map[sig]
        selected.append((ri, ci))
        r = row_starts[ri]
        c = col_starts[ci]
        cell = out[r : r + cell_size, c : c + cell_size]
        cell[cell == bg] = 3

    success = len(selected) == required_matches and _forms_2x2_block(selected)

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
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

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


PATTERNS_BY_FILL = {
    k: tuple(m for m in range(1 << 9) if m.bit_count() == k and _has_full_bbox(m))
    for k in range(FILL_MIN, FILL_MAX + 1)
}
ALL_QUERY_PATTERNS = tuple(mask for masks in PATTERNS_BY_FILL.values() for mask in masks)


def _balanced_color_list() -> list[int]:
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks() -> list[int]:
    counts = dict(SHAPE_BASE_COUNTS)
    extras = TABLE_DIM * TABLE_DIM - sum(counts.values())
    for _ in range(extras):
        choices = [k for k in range(FILL_MIN, FILL_MAX) if counts[k] < MAX_SHAPES_PER_FILL]
        counts[random.choice(choices)] += 1

    masks: list[int] = []
    for k, n in counts.items():
        masks.extend(random.sample(PATTERNS_BY_FILL[k], n))
    random.shuffle(masks)
    return masks


def _table_glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _query_glyph(mask: int, color: int) -> np.ndarray:
    glyph = np.full((QUERY_SIZE, QUERY_SIZE), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    glyph[bits] = color
    return glyph


def _signature_from_mask(mask: int, color: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (int(r), int(c), int(color))
        for r in range(3)
        for c in range(3)
        if (mask >> (r * 3 + c)) & 1
    )


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


def _pick_distractors(
    table_masks: list[int],
    used_query_sigs: set[tuple[tuple[int, int, int], ...]],
    count: int,
) -> list[tuple[int, int]]:
    distractors: list[tuple[int, int]] = []
    table_mask_set = set(table_masks)
    mask_choices = [mask for mask in ALL_QUERY_PATTERNS if mask not in table_mask_set]
    while len(distractors) < count:
        mask = random.choice(mask_choices)
        color = random.choice(GLYPH_COLORS)
        sig = _signature_from_mask(mask, color)
        if sig in used_query_sigs:
            continue
        used_query_sigs.add(sig)
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

    masks = _balanced_shape_masks()
    colors = _balanced_color_list()
    glyph_specs: list[tuple[int, int]] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph_specs.append((mask, color))
        glyph = _table_glyph(mask, color)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _block_cells()
        distractor_count = 1
    else:
        real_count = 3 if random.random() < P_REAL_QUERY_3 else BLOCK_AREA
        query_cells = _fail_cells(real_count)
        distractor_count = len(QUERY_COLS) - real_count

    used_query_sigs = {
        _signature_from_mask(glyph_specs[r * TABLE_DIM + c][0], glyph_specs[r * TABLE_DIM + c][1])
        for r, c in query_cells
    }
    distractors = _pick_distractors(masks, used_query_sigs, distractor_count)

    query_items: list[tuple[str, tuple[int, int] | tuple[int, int]]] = [
        ("real", cell) for cell in query_cells
    ] + [("distractor", spec) for spec in distractors]
    random.shuffle(query_items)
    assert len(query_items) == len(QUERY_COLS)

    real_slots = 0
    for slot, (kind, payload) in enumerate(query_items):
        if kind == "real":
            r_i, c_i = payload  # type: ignore[misc]
            mask, color = glyph_specs[r_i * TABLE_DIM + c_i]
            real_slots += 1
        else:
            mask, color = payload  # type: ignore[misc]
        paste(grid, _query_glyph(mask, color), QUERY_ROW, QUERY_COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    out = main(grid)
    if real_slots != len(query_cells):
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
