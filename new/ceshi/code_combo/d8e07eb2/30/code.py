# Combo variant of d8e07eb2 (index 30)
# combo_spec: D1/0+D8/1
# Name: 仅颜色匹配+多色字形

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior
#    - The canvas geometry stays the original one: query band, separator, one
#      blank spacer row, 4x4 table, separator, verdict band.
#    - Every 5x5 cell still contains a centered 3x3 glyph, but each glyph now
#      uses 2-3 foreground colors instead of one. One of those colors is always
#      the unique dominant color by pixel count.
#    - Matching no longer compares the full shape. A query glyph is reduced to
#      its dominant color, and the table lookup returns the first table cell in
#      reading order whose dominant color is the same.
#    - Example trace: suppose two table cells are both yellow-dominant. One is
#      an L-shape with blue accents near the top; the later one is a zigzag with
#      maroon accents near the bottom. Copy the later zigzag into the query band.
#      The old task would re-find that exact zigzag. This combo instead ignores
#      the support mask and accents, keeps only "yellow is the majority color",
#      and therefore highlights the earlier yellow-dominant L-shape.
#
# b) Complexity calibration
#    - The generator uses exactly 8 dominant colors across 16 cells, so every
#      dominant color appears twice. That creates repeated color keys on purpose
#      and forces the "first occurrence" disambiguation to matter.
#    - Shapes remain diverse because all 16 support masks are distinct and the
#      paired same-dominant cells use different masks. The task therefore does
#      not collapse into trivial exact-copy lookup.
#    - Each glyph has only 2-3 colors and one strict majority color, which keeps
#      the new rule visible enough for humans to infer from a few examples.
#
# c) Hidden-assumption review for main()
#    - Separator detection by constant full-width non-background rows still
#      holds; multicolor glyphs never fill an entire row.
#    - Table-grid inference still depends on the fixed 5x5 cell lattice with a
#      1-pixel border around the centered 3x3 glyph. The generator preserves it.
#    - The old sparse exact signature is invalid because shape is no longer part
#      of identity. The new helper extracts only the unique dominant foreground
#      color from each occupied cell.
#    - The solver now assumes each glyph has a unique dominant color and each
#      query dominant color is distinct. The generator enforces both properties.
#    - The solver also assumes every queried dominant color appears somewhere in
#      the table and that selecting the first occurrence is enough to resolve
#      ambiguity. The generator constructs the table from dominant-color pairs so
#      that this is always true.
#
# d) Placement legality
#    - All objects remain on the original fixed 30x22 canvas. Cells are pasted
#      only at the original row/column anchors, so nothing overlaps or leaves
#      bounds.
#    - Query glyphs are still exact copies of table cells, placed into the four
#      query slots at the top. The optional fourth slot is left empty only in
#      3-query failure cases.
#
# e) Visual observability
#    - The critical fused behavior is visible because many queries are copied
#      from the later member of a same-dominant-color pair, yet the output
#      highlights the earlier partner. A human can see that shape and accent
#      placement were ignored.
#    - Repeated dominant colors appear twice per grid, providing direct evidence
#      that "same majority color" is the lookup key and "earliest such cell" is
#      the tie-break.
#
# f) Visual cleanliness
#    - Highlighting still replaces only teal background pixels with green, so
#      the multicolor glyph interiors remain legible.
#    - The layout keeps one blank spacer row above the table and uses the same
#      separator rows as the original task, so the denser glyph interiors do not
#      make the board visually messy.
#
# g) Color-rule inferability
#    - No hidden arithmetic on color ids is used. The only color role is local:
#      whichever foreground color occupies the most pixels in a glyph is the
#      lookup key.
#    - The palette size remains small and repeated. Because same-dominant-color
#      pairs recur in every sample, the "majority color beats exact shape" rule
#      is inferable from examples rather than from arbitrary one-off mappings.


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


def _dominant_color(cell: np.ndarray, bg: int) -> int | None:
    vals = np.asarray(cell, dtype=int)
    fg = vals[vals != int(bg)]
    if fg.size == 0:
        return None
    colors, counts = np.unique(fg, return_counts=True)
    top = int(np.max(counts))
    winners = [int(color) for color, count in zip(colors, counts, strict=True) if int(count) == top]
    assert len(winners) == 1, "glyph must have a unique dominant color"
    return winners[0]


def _cell_key(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> int | None:
    cell = np.asarray(g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)], dtype=int)
    return _dominant_color(cell, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match multicolor query glyphs to a reference table by dominant color only.

    Input structure:
    - A teal board is split by two full-width pink separator rows.
    - The top band contains up to four 5x5 query cells whose centered 3x3
      glyphs each use 2-3 foreground colors with one majority color.
    - The middle band is a 4x4 table of the same 5x5 cells; several cells can
      share a majority color while having different shapes and accent colors.

    Rule:
    - Reduce each non-empty query cell to its dominant foreground color.
    - In the table, find the first cell in reading order with that same dominant
      color and green-fill only the teal background inside those selected table
      cells.
    - If four queries are present and the selected table cells make one complete
      table row or one complete table column, also green-fill the teal
      background in the query band and fill the verdict band green; otherwise
      leave the query band unchanged and fill the verdict band red.

    Key insight: Exact glyph identity is a distractor now. The true lookup key
    is the unique majority color inside each multicolor glyph, and repeated keys
    are resolved by taking the earliest matching table cell.
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

    row_starts = [r for r in range(row0, bottom_sep, cell_size) if r + cell_size <= bottom_sep]
    col_starts = [c for c in range(col0, g.shape[1], cell_size) if c + cell_size <= g.shape[1]]
    assert row_starts and col_starts, "could not infer table cell grid"

    query_keys = [
        key
        for c in col_starts
        if (key := _cell_key(g, 0, c, cell_size, bg)) is not None
    ]
    assert len(query_keys) == len(set(query_keys)), "query dominant colors should be distinct"

    first_pos_by_key: dict[int, tuple[int, int]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            key = _cell_key(g, r, c, cell_size, bg)
            if key is None or key in first_pos_by_key:
                continue
            first_pos_by_key[int(key)] = (int(ri), int(ci))

    out = g.copy()
    selected = [first_pos_by_key[int(key)] for key in query_keys]
    assert len(selected) == len(query_keys), "every query key must exist in the table"

    for ri, ci in selected:
        r = row_starts[ri]
        c = col_starts[ci]
        cell = out[r : r + cell_size, c : c + cell_size]
        cell[cell == bg] = 3

    complete_query = len(query_keys) == len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = complete_query and len(selected) == len(col_starts) and (same_row or same_col)

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
TABLE_CELLS = TABLE_DIM * TABLE_DIM
PAIR_COUNT = TABLE_CELLS // 2

# -- Glyph shapes --
FILL_COUNTS = (5, 6, 7, 8)
MASKS_PER_FILL = 4
MIN_GLYPH_COLORS = 2
MAX_GLYPH_COLORS = 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_QUERY_3 = 0.35

# -- Generation --
MAX_RETRIES = 500
MIN_SECOND_QUERY_COUNT = 1
FEASIBLE_SUCCESS_LINES = tuple(
    [("row", r) for r in range(TABLE_DIM - 1)] + [("col", c) for c in range(TABLE_DIM - 1)]
)


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
    fill: tuple(
        mask
        for mask in range(1 << 9)
        if mask.bit_count() == fill and _has_full_bbox(mask) and _is_asymmetric(mask)
    )
    for fill in FILL_COUNTS
}


def _mask_positions(mask: int) -> list[tuple[int, int]]:
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    return [(int(r), int(c)) for r, c in zip(*np.where(bits), strict=False)]


def _balanced_shape_masks() -> list[int]:
    masks: list[int] = []
    for fill in FILL_COUNTS:
        masks.extend(random.sample(PATTERNS_BY_FILL[fill], MASKS_PER_FILL))
    random.shuffle(masks)
    return masks


def _dominant_partitions(total: int, n_colors: int) -> list[tuple[int, ...]]:
    if n_colors == 2:
        return [(total - accent, accent) for accent in range(1, total) if total - accent > accent]

    out: list[tuple[int, ...]] = []
    for accent_a in range(1, total - 1):
        for accent_b in range(1, total - accent_a):
            dominant = total - accent_a - accent_b
            if dominant > max(accent_a, accent_b):
                out.append((dominant, accent_a, accent_b))
    return out


PARTITIONS = {
    fill: {
        n_colors: _dominant_partitions(fill, n_colors)
        for n_colors in range(MIN_GLYPH_COLORS, MAX_GLYPH_COLORS + 1)
    }
    for fill in FILL_COUNTS
}


def _render_multicolor(mask: int, dominant: int) -> np.ndarray:
    positions = _mask_positions(mask)
    fill = len(positions)
    possible_n = [n for n in range(MIN_GLYPH_COLORS, MAX_GLYPH_COLORS + 1) if PARTITIONS[fill][n]]
    n_colors = random.choice(possible_n)
    counts = random.choice(PARTITIONS[fill][n_colors])
    accents = random.sample([c for c in GLYPH_COLORS if c != dominant], n_colors - 1)

    bag = [dominant] * counts[0]
    for color, count in zip(accents, counts[1:], strict=True):
        bag.extend([color] * count)
    random.shuffle(bag)

    cell = np.full((CELL, CELL), BG, dtype=int)
    for color, (r, c) in zip(bag, positions, strict=True):
        cell[1 + r, 1 + c] = int(color)

    assert _dominant_color(cell, BG) == dominant
    return cell


def _line_indices(kind: str, idx: int) -> list[int]:
    if kind == "row":
        return [idx * TABLE_DIM + c for c in range(TABLE_DIM)]
    return [r * TABLE_DIM + idx for r in range(TABLE_DIM)]


def _index_to_rc(idx: int) -> tuple[int, int]:
    return divmod(int(idx), TABLE_DIM)


def _pair_anchors(anchors: set[int]) -> dict[int, int] | None:
    partners = sorted(set(range(TABLE_CELLS)) - anchors)
    remaining = partners[:]
    out: dict[int, int] = {}
    for anchor in sorted(anchors, reverse=True):
        choices = [p for p in remaining if p > anchor]
        if not choices:
            return None
        partner = choices[-1]
        out[int(anchor)] = int(partner)
        remaining.remove(partner)
    return out


def _build_pair_layout() -> tuple[list[int], dict[int, int], list[int]]:
    for _ in range(300):
        kind, idx = random.choice(FEASIBLE_SUCCESS_LINES)
        success_line = _line_indices(kind, idx)
        pool = [pos for pos in range(TABLE_CELLS - 1) if pos not in success_line]
        extra_anchors = random.sample(pool, PAIR_COUNT - TABLE_DIM)
        anchors = set(success_line) | set(extra_anchors)
        assert len(anchors) == PAIR_COUNT
        pairing = _pair_anchors(anchors)
        if pairing is None:
            continue
        return sorted(anchors), pairing, success_line
    raise RuntimeError("failed to build a valid first-occurrence layout")


def _build_table() -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, int], list[int]]:
    anchors, pairing, success_line = _build_pair_layout()
    colors = random.sample(GLYPH_COLORS, PAIR_COUNT)
    masks = _balanced_shape_masks()

    glyph_by_index: dict[int, np.ndarray] = {}
    anchor_by_color: dict[int, int] = {}
    second_by_color: dict[int, int] = {}
    seen_mats: set[tuple[tuple[int, ...], ...]] = set()

    for i, color in enumerate(colors):
        anchor_idx = anchors[i]
        second_idx = pairing[anchor_idx]
        glyphs = [
            _render_multicolor(masks[2 * i], color),
            _render_multicolor(masks[2 * i + 1], color),
        ]
        random.shuffle(glyphs)
        glyph_by_index[anchor_idx] = glyphs[0]
        glyph_by_index[second_idx] = glyphs[1]
        anchor_by_color[int(color)] = int(anchor_idx)
        second_by_color[int(color)] = int(second_idx)

        for glyph in glyphs:
            mat = tuple(tuple(int(v) for v in row) for row in glyph)
            assert mat not in seen_mats, "every table glyph must have a unique matrix"
            seen_mats.add(mat)
            assert _dominant_color(glyph, BG) == color

    assert len(glyph_by_index) == TABLE_CELLS
    return glyph_by_index, anchor_by_color, second_by_color, success_line


def _choose_fail_colors(anchor_by_color: dict[int, int], count: int) -> list[int]:
    colors = list(anchor_by_color)
    for _ in range(300):
        picked = random.sample(colors, count)
        coords = [_index_to_rc(anchor_by_color[color]) for color in picked]
        if len({r for r, _c in coords}) == 1:
            continue
        if len({c for _r, c in coords}) == 1:
            continue
        random.shuffle(picked)
        return picked
    raise RuntimeError("failed to sample a failure color set")


def _query_sources(
    query_colors: list[int], anchor_by_color: dict[int, int], second_by_color: dict[int, int]
) -> dict[int, int]:
    min_second = min(len(query_colors), 2 if len(query_colors) == TABLE_DIM else MIN_SECOND_QUERY_COUNT)
    use_second = set(random.sample(query_colors, min_second))
    out: dict[int, int] = {}
    for color in query_colors:
        if color in use_second:
            out[int(color)] = int(second_by_color[color])
        else:
            out[int(color)] = int(random.choice((anchor_by_color[color], second_by_color[color])))
    assert any(out[color] != anchor_by_color[color] for color in query_colors)
    return out


def _expected_output(grid: np.ndarray, selected_indices: list[int], success: bool) -> np.ndarray:
    out = grid.copy()
    for idx in selected_indices:
        r_i, c_i = _index_to_rc(idx)
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

    glyph_by_index, anchor_by_color, second_by_color, success_line = _build_table()
    success_line_set = set(success_line)

    for idx, glyph in glyph_by_index.items():
        r_i, c_i = _index_to_rc(idx)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_colors = [color for color, idx in anchor_by_color.items() if idx in success_line_set]
        random.shuffle(query_colors)
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_colors = _choose_fail_colors(anchor_by_color, count)

    source_by_color = _query_sources(query_colors, anchor_by_color, second_by_color)
    for slot, color in enumerate(query_colors):
        src_idx = source_by_color[color]
        paste(grid, glyph_by_index[src_idx], 0, COLS[slot])

    selected_indices = [anchor_by_color[color] for color in query_colors]
    expected = _expected_output(grid, selected_indices, success)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
