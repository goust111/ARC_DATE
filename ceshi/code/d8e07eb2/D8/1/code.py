# Variant of d8e07eb2
# Dimension: D8 — 字形形状多样性 (meta)
# Change: 每个字形为3x3二值图案（前景色+背景色8），16种可能中使用了约10种不同的非对称图案，颜色和形状都不重复地分配到16... → 字形使用多种颜色

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the D8 variant
# ----------------------------------
# a) Concrete behavior
#    - The canvas geometry stays the same: query band, separator, 4x4 reference
#      table, separator, verdict band.
#    - Each 5x5 cell still contains a centered 3x3 glyph, but the glyph is no
#      longer monochrome. Instead, every glyph uses 2-3 foreground colors.
#    - Matching is performed on the full 5x5 cell matrix. This means the exact
#      color at each pixel position matters, not just a single glyph color plus
#      a binary support mask.
#    - To make that visible, the generator creates "twin" table entries: two
#      cells can share the same 3x3 support mask and the same multiset of colors
#      while differing only by where those colors sit inside the mask. Query
#      glyphs are copied from one twin, and only that exact matrix should match.
#    - Example trace:
#      original: a query glyph = maroon + shape S, so any table cell with the
#      same maroon pixels on the same support matches.
#      variant: two table cells may share support S and palette {yellow, blue},
#      but blue/yellow are permuted across the filled pixels. The query must
#      match the identical permutation; its sibling twin must stay unselected.
#
# b) Complexity calibration
#    - Using 8 support masks repeated twice gives 16 cells with visible shape
#      diversity, while still forcing color-arrangement disambiguation.
#    - Each family keeps the same mask and color histogram across its two twins,
#      which prevents the task from collapsing back to shape-only matching.
#    - Colors are drawn from the same 8-color palette as the original task, so
#      the visual vocabulary stays manageable.
#
# c) Hidden-assumption review for main()
#    - Separator detection via constant full-width non-background rows still
#      holds because glyph colors never occupy an entire row.
#    - Table-grid inference still relies on the centered 3x3 placement with a
#      1-pixel border. The generator preserves that exact geometry.
#    - The old sparse signature helper depended on "background pixels are
#      irrelevant". That is too weak for this variant, so the solver now stores
#      the entire 5x5 matrix.
#    - Green-filling only `cell == bg` remains safe because glyph interiors may
#      contain many colors, but the highlight is intended to replace only the
#      teal background inside matched cells.
#    - Overwriting the verdict band with a solid color remains safe because the
#      generator guarantees that band starts as plain background.
#
# d) Placement legality
#    - All cells remain 5x5 and are pasted at the original fixed row/column
#      anchors, so nothing overlaps or leaves the 30x22 grid.
#    - Query glyphs are exact copies of table cells and occupy the same four
#      column slots as before; the optional 4th fail slot may remain empty.
#
# e) Visual observability
#    - The near-twin pairs make the new rule human-visible: one twin lights up
#      while the similarly shaped/color-counted sibling does not.
#    - The row/column success condition is unchanged, so the new difficulty is
#      concentrated on glyph identity rather than on a different global rule.
#
# f) Visual cleanliness
#    - The output still highlights only backgrounds, preserving the multicolor
#      glyph interiors. This keeps the selected cells readable.
#    - Reusing the original separators and band layout prevents clutter.
#
# g) Color-change inferability
#    - No hidden ordering or arithmetic on color ids is used.
#    - Color structure is local and positional: examples show that matching
#      depends on the exact colored mini-pattern copied into the query band.


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _cell_matrix(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[tuple[int, ...], ...] | None:
    cell = np.asarray(g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)], dtype=int)
    if not bool(np.any(cell != int(bg))):
        return None
    return tuple(tuple(int(v) for v in row) for row in cell)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match multicolor query glyphs against a reference table and test for a line.

    Input structure:
    - A teal canvas is divided by two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells whose centered 3x3
      glyphs use 2-3 foreground colors.
    - The middle band is a 4x4 table of the same 5x5 cells; some cells are
      shape-twins that differ only by their internal color placement.

    Rule:
    - Treat each query/table cell as a complete 5x5 pixel matrix and green-fill
      the teal background inside every table cell whose whole matrix appears in
      the query band.
    - If four queries are present and their matched table locations make one
      complete table row or one complete table column, also green-fill the teal
      background in the query band and fill the verdict band green.
    - Otherwise leave the query band unchanged and fill the verdict band red.

    Key insight: Similar-looking glyphs are deliberately reused with the same
    support mask and color set, so exact per-pixel color placement is the only
    reliable lookup key.
    """
    g = np.asarray(grid, dtype=int)
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
    col_starts = list(range(col0, g.shape[1], cell_size))
    col_starts = [c for c in col_starts if c + cell_size <= g.shape[1]]
    assert row_starts and col_starts, "could not infer table cell grid"

    query_mats = [
        mat
        for c in col_starts
        if (mat := _cell_matrix(g, 0, c, cell_size, bg)) is not None
    ]
    query_set = set(query_mats)
    assert len(query_mats) == len(query_set), "query glyphs should be distinct"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            if _cell_matrix(g, r, c, cell_size, bg) not in query_set:
                continue
            selected.append((int(ri), int(ci)))
            cell = out[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = 3

    complete_query = len(query_mats) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(query_mats) and (same_row or same_col)

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator ---------------

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

# -- Glyph families --
FILL_COUNTS = (5, 6, 7, 8)
MASKS_PER_FILL = 2
FAMILY_SIZE = 2
MIN_GLYPH_COLORS = 2
MAX_GLYPH_COLORS = 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

# -- Generation --
MAX_RETRIES = 400


def _has_full_bbox(mask: int) -> bool:
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=int).reshape(3, 3)
    return (
        bool(bits[0, :].any())
        and bool(bits[2, :].any())
        and bool(bits[:, 0].any())
        and bool(bits[:, 2].any())
    )


def _is_asymmetric(mask: int) -> bool:
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=int).reshape(3, 3)
    transforms = (
        np.flipud(bits),
        np.fliplr(bits),
        np.rot90(bits, 1),
        np.rot90(bits, 2),
        np.rot90(bits, 3),
    )
    return all(not np.array_equal(bits, other) for other in transforms)


PATTERNS_BY_FILL = {
    k: tuple(
        mask
        for mask in range(1 << 9)
        if mask.bit_count() == k and _has_full_bbox(mask) and _is_asymmetric(mask)
    )
    for k in FILL_COUNTS
}


def _balanced_shape_masks() -> list[int]:
    masks: list[int] = []
    for fill in FILL_COUNTS:
        masks.extend(random.sample(PATTERNS_BY_FILL[fill], MASKS_PER_FILL))
    random.shuffle(masks)
    return masks


def _mask_positions(mask: int) -> list[tuple[int, int]]:
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    return [(int(r), int(c)) for r, c in zip(*np.where(bits), strict=False)]


def _render_glyph(mask: int, assignment: tuple[int, ...]) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    for color, (r, c) in zip(assignment, _mask_positions(mask), strict=True):
        cell[1 + r, 1 + c] = int(color)
    return cell


def _random_color_counts(total: int, n_colors: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), n_colors - 1))
    counts: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        counts.append(cut - prev)
        prev = cut
    random.shuffle(counts)
    return counts


def _family_glyphs(mask: int) -> list[np.ndarray]:
    positions = _mask_positions(mask)
    fill = len(positions)
    n_colors = random.randint(MIN_GLYPH_COLORS, min(MAX_GLYPH_COLORS, fill))
    palette = random.sample(GLYPH_COLORS, n_colors)
    counts = _random_color_counts(fill, n_colors)

    bag: list[int] = []
    for color, count in zip(palette, counts, strict=True):
        bag.extend([color] * count)

    assignments: set[tuple[int, ...]] = set()
    attempts = 0
    while len(assignments) < FAMILY_SIZE:
        attempts += 1
        perm = tuple(random.sample(bag, len(bag)))
        if len(set(perm)) < MIN_GLYPH_COLORS:
            continue
        assignments.add(perm)
        if attempts > 200:
            raise RuntimeError("failed to build distinct multicolor twins")

    return [_render_glyph(mask, assignment) for assignment in assignments]


def _support_signature(cell: np.ndarray) -> tuple[tuple[bool, ...], ...]:
    inner = np.asarray(cell[1:4, 1:4], dtype=int)
    return tuple(tuple(bool(v != BG) for v in row) for row in inner)


def _color_hist_signature(cell: np.ndarray) -> tuple[tuple[int, int], ...]:
    vals, counts = np.unique(cell[cell != BG], return_counts=True)
    return tuple(sorted((int(v), int(c)) for v, c in zip(vals, counts, strict=True)))


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


def _build_table_glyphs() -> list[np.ndarray]:
    glyphs: list[np.ndarray] = []
    seen_full: set[tuple[tuple[int, ...], ...]] = set()

    for mask in _balanced_shape_masks():
        family = _family_glyphs(mask)
        family_supports = {_support_signature(cell) for cell in family}
        family_hists = {_color_hist_signature(cell) for cell in family}
        assert len(family_supports) == 1, "twin family must share support mask"
        assert len(family_hists) == 1, "twin family must share color histogram"
        assert len({tuple(map(tuple, cell)) for cell in family}) == FAMILY_SIZE

        for cell in family:
            mat = tuple(tuple(int(v) for v in row) for row in cell)
            assert mat not in seen_full, "every table glyph must have a unique full matrix"
            seen_full.add(mat)
            glyphs.append(cell)

    random.shuffle(glyphs)
    assert len(glyphs) == TABLE_DIM * TABLE_DIM
    return glyphs


def _variant_score(glyphs: list[np.ndarray]) -> int:
    families: dict[tuple[tuple[bool, ...], ...], list[tuple[tuple[int, int], ...]]] = {}
    for cell in glyphs:
        key = _support_signature(cell)
        hist = _color_hist_signature(cell)
        families.setdefault(key, []).append(hist)
    return sum(
        1
        for hists in families.values()
        if len(hists) >= 2 and len(set(hists)) < len(hists)
    )


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    glyphs = _build_table_glyphs()
    assert _variant_score(glyphs) >= 8, "all support-mask families should need exact color placement"

    for i, glyph in enumerate(glyphs):
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    for slot, (r_i, c_i) in enumerate(query_cells):
        paste(grid, glyphs[r_i * TABLE_DIM + c_i], 0, COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 D8 input")
