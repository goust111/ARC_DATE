# Combo variant of d8e07eb2 (index 22)
# combo_spec: D6/2+D14/1
# Name: 渐变判定颜色+未匹配填红
#
# Design analysis
# a) Concrete behavior and trace:
#    - Source task:
#      * only matched table cells were green-filled;
#      * unmatched table cells stayed teal;
#      * the bottom band was binary: green for a full row/column of four, red otherwise.
#    - Fused task:
#      * every table cell is now classified, not just the matched ones:
#        matched cells get green backgrounds, unmatched cells get red backgrounds;
#      * the verdict band becomes a 4-level progress meter based on the strongest row/column
#        alignment among the matched cells: 4 -> green, 3 -> yellow, 2 -> orange, 1/0 -> red;
#      * the query band keeps the original success-only behavior and turns green only for a
#        genuine complete line of four matched cells.
#    - Small trace:
#      * if the query picks four glyphs from one table row, the table shows one full green row
#        against red elsewhere, the query band turns green, and the verdict band is green;
#      * if the query picks three glyphs from one row plus one off-row glyph, the table still
#        marks all other cells red, but now the verdict band becomes yellow instead of plain red.
# b) Complexity calibration:
#    - The gradient uses only four colors and is tied to one structural quantity: the maximum
#      number of matched cells sharing a row or sharing a column.
#    - The table coloration is equally simple: every cell background becomes either matched-green
#      or unmatched-red, so the change is strong and easy to compare across examples.
# c) Hidden assumption review:
#    - Original logic assumed unmatched table cells could stay untouched; D14/1 breaks that, so
#      the solver must visit every table cell and recolor all of their background pixels.
#    - Original success logic treated the verdict as boolean; D6/2 breaks that, so the solver now
#      needs an intermediate statistic (`line_strength`) before deciding the verdict color.
#    - Query matching still relies on unique (shape, color) signatures. The generator preserves
#      this by keeping all 16 table glyph signatures distinct.
#    - Red unmatched fills can hide red glyphs, so the generator removes glyph color 2 from its
#      palette. This is a new observability constraint introduced by the fusion, not by the source.
# d) Placement legality:
#    - The layout stays the original fixed 30x22 canvas with two separator rows and a 4x4 table
#      of non-overlapping 5x5 cells at fixed coordinates.
#    - Query glyphs are pasted only into the four legal top-band slots, with any empty slot left teal.
# e) Visual observability:
#    - The reference table always becomes a complete red/green membership map, which makes D14/1
#      visible in every sample rather than only in failures or edge cases.
#    - The generator explicitly samples all four line-strength buckets so the green/yellow/orange/red
#      verdict palette appears repeatedly across examples instead of depending on rare chance.
# f) Visual cleanliness:
#    - Table markings remain confined to background pixels inside cell borders, so glyph shapes stay
#      readable and separators / band structure remain untouched.
#    - The verdict gradient is isolated to the bottom strip and does not compete with the table colors.
# g) Color inferability:
#    - The mapping is intentionally low-arity and monotone: better row/column alignment means a warmer
#      to cooler progression from red -> orange -> yellow -> green.
#    - Because the red table backgrounds are independent of the verdict band, a human can separate
#      "membership" (table) from "alignment quality" (bottom band) using region structure alone.
# h) Generator-side new assumptions:
#    - The sampled query cell set must realize its intended maximum shared-row/shared-column count
#      exactly; this is guaranteed by enumerating legal 3-cell and 4-cell subsets up front.
#    - Query count remains 3 or 4, so solver handling of strength 0 is a defensive fallback rather
#      than a generated case.

from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from pathlib import Path

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


def _cell_signature(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[tuple[int, int, int], ...] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    return tuple(
        (int(r), int(c), int(cell[r, c]))
        for r, c in zip(*np.where(mask), strict=False)
    )


def _line_strength(cells: list[tuple[int, int]]) -> int:
    if not cells:
        return 0
    row_counts = Counter(r for r, _c in cells)
    col_counts = Counter(c for _r, c in cells)
    return max(max(row_counts.values(), default=0), max(col_counts.values(), default=0))


def _verdict_color(strength: int) -> int:
    if strength >= 4:
        return 3
    if strength == 3:
        return 4
    if strength == 2:
        return 7
    return 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyphs against a reference table, then visualize both membership and alignment strength.

    Input structure:
    - A teal background contains two full-width pink separator rows.
    - The top band holds up to four 5x5 query cells, each containing a centered 3x3 colored glyph.
    - Between the separators sits a 4x4 reference table of the same 5x5 cells; glyph identity depends
      on the exact 3x3 shape together with its color.

    Rule:
    - Recolor the background pixels inside every reference-table cell: matched glyph cells become green,
      unmatched glyph cells become red.
    - Measure how many matched cells can be aligned within one table row or one table column. Fill the
      verdict band green for 4, yellow for 3, orange for 2, and red for 1 or 0.
    - Only when all four query slots are filled and the matched table cells make one complete row or one
      complete column should the query-band background also be green-filled.

    Key insight:
    - The table is now a full match/non-match map, while the bottom band no longer answers only yes/no:
      it encodes the best row-or-column alignment score of the matched cells.
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

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, bg)
            assert sig is not None, "reference table cells must contain glyphs"
            fill = 3 if sig in query_set else 2
            cell = out[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = fill
            if sig in query_set:
                selected.append((int(ri), int(ci)))

    strength = _line_strength(selected)
    out[bottom_sep + 1 :, :] = _verdict_color(strength)

    complete_query = len(query_sigs) == len(col_starts)
    success = complete_query and len(selected) == len(query_sigs) and strength == len(col_starts)
    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3

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
YELLOW = 4
ORANGE = 7
SEP = 6
GLYPH_COLORS = (0, 1, 4, 5, 6, 7, 9)

# -- Table geometry --
TABLE_DIM = 4
CELL = 5
ROWS = (7, 12, 17, 22)
COLS = (1, 6, 11, 16)
ALL_CELLS = tuple((r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM))

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
LINE_STRENGTHS = (1, 2, 3, 4)

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


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _build_query_buckets() -> dict[int, tuple[tuple[tuple[int, int], ...], ...]]:
    buckets: dict[int, list[tuple[tuple[int, int], ...]]] = {s: [] for s in LINE_STRENGTHS}
    for count in (3, 4):
        for cells in itertools.combinations(ALL_CELLS, count):
            strength = _line_strength(list(cells))
            if strength in buckets:
                buckets[strength].append(tuple(cells))
    out = {s: tuple(bucket) for s, bucket in buckets.items()}
    assert all(out.values()), "every verdict level must have at least one realizable query set"
    return out


QUERY_BUCKETS = _build_query_buckets()


def _expected_output(grid: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    selected = set(cells)
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            cell[cell == BG] = GREEN if (r_i, c_i) in selected else RED

    strength = _line_strength(cells)
    out[VERDICT_ROW:, :] = _verdict_color(strength)
    if len(cells) == TABLE_DIM and strength == TABLE_DIM:
        query_band = out[:QUERY_BAND_H, :]
        query_band[query_band == BG] = GREEN
    return out


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    masks = _balanced_shape_masks()
    colors = _balanced_color_list()
    glyphs: list[np.ndarray] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    target_strength = random.choice(LINE_STRENGTHS)
    query_cells = list(random.choice(QUERY_BUCKETS[target_strength]))
    random.shuffle(query_cells)

    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], 0, COLS[slot])

    expected = _expected_output(grid, query_cells)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
