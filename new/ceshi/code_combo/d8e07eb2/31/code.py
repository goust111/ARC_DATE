# Combo variant of d8e07eb2 (index 31)
# combo_spec: D5/1+D12/1
# Name: 2x2判定+两轮匹配

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


# Design analysis for the fused D5/1 + D12/1 variant
# ===================================================
# a) Concrete behavior and trace:
#    The layout stays the original 30x22 canvas: query band, separator, 4x4 table,
#    separator, verdict band. The fusion changes two linked stages.
#    - Original: a query glyph matched one table cell by exact color+shape; all
#      exact matches turned green; success meant those green cells formed a full
#      row or column.
#    - Fused: round 1 uses only query colors and paints every same-color table
#      candidate yellow; round 2 refines inside that pool by exact color+shape
#      and upgrades only the true matches to green. Success is now based only on
#      the green cells, and they must be exactly one adjacent 2x2 block.
#    Example trace: if the queries come from table cells {(1,1),(1,2),(2,1),(2,2)}
#    with four distinct colors, round 1 paints all cells carrying those four
#    colors yellow, leaving several yellow-only leftovers elsewhere. Round 2 turns
#    only the four exact copies green. The verdict is green because the green
#    cells are the cross-product of rows {1,2} and cols {1,2}. If instead the
#    exact matches were a full row, the table would still show yellow traces plus
#    four green exact hits, but the verdict would be red because a line is no
#    longer sufficient.
# b) Complexity calibration:
#    The first pass needs repeated colors to be visible, while the second pass
#    needs exact matches to remain unambiguous. The generator therefore keeps all
#    16 shapes distinct, reserves yellow/green as output-only colors, and uses a
#    repeated-color scaffold where every query color appears in multiple cells.
#    Query colors are sampled to be distinct so each yellow family is readable.
# c) Implicit assumptions audit:
#    The original solver assumed exact matching was the only selection step and
#    that four matches sharing one row/column implied success. Neither remains
#    valid. The fused solver must extract query colors separately from exact
#    signatures, color-paint candidates before exact overwrites, and replace the
#    line test with an adjacency-aware 2x2 test. The generator must also avoid
#    accidentally producing ambiguous yellow output by allowing query colors to
#    overlap, so it samples only distinct-color query sets.
# d) Placement legality:
#    All objects stay on the original fixed 5x5 lattice. Query glyphs are copied
#    into the four top slots only, table cells remain inside the 4x4 region, and
#    verdict fills touch only the bottom band. Distinct sampled table cells avoid
#    overlap, and the fixed anchors avoid out-of-bounds writes.
# e) Visual observability:
#    Across examples, a human can see two stable clues at once: yellow means
#    "same color candidate" and green means "exact glyph". Successful examples
#    additionally show that the green subset forms a compact square, while failed
#    examples may still show lines or scattered greens, making the new spatial
#    condition distinguishable from the original one.
# f) Visual cleanliness:
#    The only overlays are background recolors inside table cells plus the final
#    verdict/query-band fills. Glyph pixels themselves are never rewritten, so the
#    color-family trace (yellow) and exact-match trace (green) remain legible.
# g) Inference of color changes and new solver assumptions:
#    Yellow and green have structural meaning tied to table-cell backgrounds, not
#    arbitrary remapping of glyph colors. `main()` assumes every glyph cell is
#    monochrome, each query has exactly one exact table match, and every query
#    color has at least one extra occurrence so round 1 is visible. The generator
#    guarantees these by rendering monochrome 3x3 glyphs, keeping shapes unique,
#    using a repeated-color scaffold, and sampling query sets with distinct colors.


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


def _signature_color(sig: tuple[tuple[int, int, int], ...]) -> int:
    colors = {int(v) for _r, _c, v in sig}
    assert len(colors) == 1, "glyphs must be monochrome"
    return next(iter(colors))


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
    Run a visible two-stage lookup, then judge whether the refined hits make a 2x2 block.

    Input structure:
    - A teal canvas is split by two full-width separator rows.
    - The top band contains up to four 5x5 query cells copied from the table.
    - The middle band is a 4x4 table of 5x5 cells whose centered 3x3 glyphs are
      monochrome; colors can repeat even though full color+shape signatures do not.

    Rule:
    - First, color every table cell with a query-used glyph color by filling that
      cell's background yellow.
    - Second, among those candidates, recolor the exact color+shape matches green.
    - If there are exactly four green exact matches and they occupy one contiguous
      2x2 table submatrix, also green-fill the query-band background and the
      verdict band; otherwise leave the query band unchanged and fill the verdict
      band red.

    Key insight: Yellow records a coarse color filter, but the answer depends only
    on the green refined matches, which must form a compact adjacent square.
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

    query_sigs = [
        sig
        for c in col_starts
        if (sig := _cell_signature(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set), "query glyphs should be distinct"
    query_colors = {_signature_color(sig) for sig in query_sigs}
    assert len(query_colors) == len(query_sigs), "query glyph colors must be distinct"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    saw_yellow_only = False
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, cell_size, bg)
            if sig is None:
                continue
            color = _signature_color(sig)
            cell = out[r : r + cell_size, c : c + cell_size]
            if color in query_colors:
                cell[cell == bg] = 4
                if sig not in query_set:
                    saw_yellow_only = True
            if sig in query_set:
                selected.append((int(ri), int(ci)))
                cell[cell == bg] = 3

    assert not query_sigs or saw_yellow_only, "two-pass variant expects visible yellow-only candidates"
    success = (
        len(query_sigs) == required_matches
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
BLOCK_SIDE = 2
BLOCK_AREA = BLOCK_SIDE * BLOCK_SIDE

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

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


def _color_grid() -> list[list[int]]:
    palette = list(GLYPH_COLORS)
    random.shuffle(palette)
    return [[palette[idx] for idx in row] for row in COLOR_INDEX_PATTERN]


def _cells_have_distinct_colors(cells: list[tuple[int, int]], color_grid: list[list[int]]) -> bool:
    colors = [color_grid[r][c] for r, c in cells]
    return len(colors) == len(set(colors))


def _block_cells(color_grid: list[list[int]]) -> list[tuple[int, int]]:
    blocks: list[list[tuple[int, int]]] = []
    for top in range(TABLE_DIM - 1):
        for left in range(TABLE_DIM - 1):
            cells = [(top + dr, left + dc) for dr in range(BLOCK_SIDE) for dc in range(BLOCK_SIDE)]
            if _cells_have_distinct_colors(cells, color_grid):
                blocks.append(cells)
    assert blocks, "color scaffold must support at least one distinct-color 2x2 block"
    chosen = [tuple(cell) for cell in random.choice(blocks)]
    random.shuffle(chosen)
    return chosen


def _line_cells(color_grid: list[list[int]]) -> list[tuple[int, int]]:
    lines: list[list[tuple[int, int]]] = []
    for r in range(TABLE_DIM):
        cells = [(r, c) for c in range(TABLE_DIM)]
        if _cells_have_distinct_colors(cells, color_grid):
            lines.append(cells)
    for c in range(TABLE_DIM):
        cells = [(r, c) for r in range(TABLE_DIM)]
        if _cells_have_distinct_colors(cells, color_grid):
            lines.append(cells)
    assert lines, "color scaffold must support at least one distinct-color line"
    chosen = [tuple(cell) for cell in random.choice(lines)]
    random.shuffle(chosen)
    return chosen


def _gapped_rect_cells(color_grid: list[list[int]]) -> list[tuple[int, int]]:
    choices: list[list[tuple[int, int]]] = []
    for r0 in range(TABLE_DIM):
        for r1 in range(r0 + 1, TABLE_DIM):
            for c0 in range(TABLE_DIM):
                for c1 in range(c0 + 1, TABLE_DIM):
                    cells = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
                    if _forms_2x2_block(cells):
                        continue
                    if _cells_have_distinct_colors(cells, color_grid):
                        choices.append(cells)
    assert choices, "expected a non-adjacent rectangle with distinct colors"
    chosen = [tuple(cell) for cell in random.choice(choices)]
    random.shuffle(chosen)
    return chosen


def _bent_cells(color_grid: list[list[int]]) -> list[tuple[int, int]]:
    choices: list[list[tuple[int, int]]] = []
    for r in range(TABLE_DIM):
        for dropped_c in range(TABLE_DIM):
            for new_r in range(TABLE_DIM):
                if new_r == r:
                    continue
                cells = [(r, c) for c in range(TABLE_DIM) if c != dropped_c] + [(new_r, dropped_c)]
                if _cells_have_distinct_colors(cells, color_grid) and not _forms_2x2_block(cells):
                    choices.append(cells)
    for c in range(TABLE_DIM):
        for dropped_r in range(TABLE_DIM):
            for new_c in range(TABLE_DIM):
                if new_c == c:
                    continue
                cells = [(r, c) for r in range(TABLE_DIM) if r != dropped_r] + [(dropped_r, new_c)]
                if _cells_have_distinct_colors(cells, color_grid) and not _forms_2x2_block(cells):
                    choices.append(cells)
    assert choices, "expected a bent four-cell set with distinct colors"
    chosen = [tuple(cell) for cell in random.choice(choices)]
    random.shuffle(chosen)
    return chosen


def _scatter_cells(count: int, color_grid: list[list[int]]) -> list[tuple[int, int]]:
    universe = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM)]
    for _ in range(500):
        cells = random.sample(universe, count)
        if not _cells_have_distinct_colors(cells, color_grid):
            continue
        if count == BLOCK_AREA and _forms_2x2_block(cells):
            continue
        random.shuffle(cells)
        return cells
    raise RuntimeError("failed to sample a distinct-color scattered set")


def _fail_cells(count: int, color_grid: list[list[int]]) -> list[tuple[int, int]]:
    if count == 3:
        return _scatter_cells(count, color_grid)

    mode = random.choice(FAIL_MODES)
    if mode == "line":
        return _line_cells(color_grid)
    if mode == "gapped_rect":
        return _gapped_rect_cells(color_grid)
    if mode == "bent":
        return _bent_cells(color_grid)
    return _scatter_cells(count, color_grid)


def _expected_output(
    grid: np.ndarray,
    query_cells: list[tuple[int, int]],
    color_grid: list[list[int]],
    success: bool,
) -> np.ndarray:
    out = grid.copy()
    query_colors = {color_grid[r][c] for r, c in query_cells}
    saw_yellow_only = False
    for r_i in range(TABLE_DIM):
        for c_i in range(TABLE_DIM):
            cell = out[ROWS[r_i] : ROWS[r_i] + CELL, COLS[c_i] : COLS[c_i] + CELL]
            if color_grid[r_i][c_i] in query_colors:
                cell[cell == BG] = YELLOW
                if (r_i, c_i) not in query_cells:
                    saw_yellow_only = True
    assert saw_yellow_only, "expected visible color-only candidates"
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


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    masks = _balanced_shape_masks()
    color_grid = _color_grid()
    glyphs: list[np.ndarray] = []
    for i, mask in enumerate(masks):
        r_i, c_i = divmod(i, TABLE_DIM)
        color = color_grid[r_i][c_i]
        glyph = _glyph(mask, color)
        glyphs.append(glyph)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _block_cells(color_grid)
    else:
        count = 3 if random.random() < P_QUERY_3 else BLOCK_AREA
        query_cells = _fail_cells(count, color_grid)

    assert _cells_have_distinct_colors(query_cells, color_grid), "query colors must stay distinct"

    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], 0, COLS[slot])

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
