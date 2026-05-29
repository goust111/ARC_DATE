# Variant of d8e07eb2
# Dimension: D2 — 查询字形来源区域 (src)
# Change: 查询字形位于第一条分隔行上方的顶部带中，每个字形占据一个5x5单元格（3x3字形+1像素边距），排成一行最多4个... → 查询字形含干扰项

# Design analysis
# ---
# variant_spec:
#   core_change:
#     original_query_band: 4 table-aligned 5x5 cells, all of them are genuine lookups
#     variant_query_band: 5 compact 3x3 glyph slots in the same top band; 3-4 are genuine
#       table lookups and the remaining 1-2 are distractors that do not exist in the table.
#     rationale: the fixed 22-column canvas cannot display 4 real queries plus 1-2 distractors
#       as separate 5x5 cells, so the minimal local adaptation is to keep the table/verdict
#       geometry unchanged and compress only the query source region.
#   concrete_pixels:
#     separators: unchanged full-width color-6 rows at y=5 and y=27
#     table: unchanged 4x4 grid of 5x5 cells with centered 3x3 monochrome glyphs
#     query_band: unchanged height 5, but now contains five 3x3 glyph windows at
#       row 1 and columns 1,5,9,13,17
#   behavior_trace:
#     original_example: query row shows four exact table glyphs -> solver treats every visible
#       query glyph as relevant -> four matched table cells decide success/fail.
#     variant_example: top band shows [real A, distractor X, real B, real C, real D].
#       The original query_set logic would incorrectly count X as part of the query and would
#       never allow success because the visible query count is now 5. The variant solver instead
#       builds the set of table signatures first, filters query signatures through that set,
#       ignores X, and evaluates only A/B/C/D.
#   complexity_calibration:
#     distractors: exactly 1 on success cases, 1 or 2 on fail cases
#     genuine_matches: still 4 for success, 3 or 4 for fail, so the original row/column concept
#       remains legible and the new reasoning step is only "filter out non-table glyphs".
#     distractor_style: distractors reuse normal table colors but use shapes absent from the
#       current table, so they are visible without introducing a noisy extra color system.
#   implicit_assumption_review:
#     old_main_query_extraction: assumes each non-empty top slot is a real query; no longer valid.
#     old_main_complete_query: assumes visible query count equals required line length (4);
#       no longer valid because there are now 5 visible glyphs.
#     old_main_cell_signature: compares whole 5x5 cells; no longer sufficient because query glyphs
#       are packed as raw 3x3 patches. The variant normalizes signatures to glyph-local coordinates.
#     unchanged_safe_parts: separator detection, table-grid inference, per-cell green fill, and
#       bottom-band verdict coloring remain compatible with the new source region.
#   placement_legality:
#     query slots are non-overlapping 3x3 windows fully inside rows 1..3 and cols 1..19.
#     table cells remain at the original 5x5 coordinates, so no overlap or boundary risk changes.
#     generator verifies every sample by running main(input) and comparing to the expected output.
#   observability:
#     humans can see that some top glyphs have no counterpart anywhere in the reference table.
#     Across examples, matched table cells are always highlighted only for the reusable glyphs,
#       making the ignore-the-extras rule visible in 2-3 demonstrations.
#   visual_cleanliness:
#     the query band contains just one layer of 3x3 glyphs with one-column gaps; there is no
#       overlaying markup before the output fill, so distractors remain readable.
#   color_inference:
#     no hidden ordering or color-id arithmetic is used; matching is determined only by visible
#       color+shape equality against the reference table, and distractors fail by visible shape mismatch.

from __future__ import annotations

import random

import numpy as np

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
    return tuple(
        (int(r - r0), int(c - c0), int(patch[r, c]))
        for r, c in coords
    )


def _cell_signature(
    g: np.ndarray, r0: int, c0: int, h: int, w: int, bg: int
) -> tuple[tuple[int, int, int], ...] | None:
    patch = g[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)]
    return _normalized_signature(patch, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match only the query glyphs that actually exist in the reference table.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - The top band contains five compact 3x3 query glyph slots; some are true
      table lookups and 1-2 are distractors with no table match.
    - Between separators is the original 4x4 reference table of 5x5 cells, each
      holding a centered 3x3 glyph identified by both shape and color.

    Rule:
    - Compare each top-band glyph to the table by exact color+shape signature,
      ignoring any glyphs that do not appear in the table at all.
    - Green-fill the background of every matched table cell.
    - If the matched cells, after filtering out distractors, make one complete
      row or one complete column of the table, also green-fill the query band
      and the verdict band; otherwise fill only the verdict band red.

    Key insight: the visible query band is no longer a pure lookup list. The
    solver must first reject top-band glyphs that have no table counterpart,
    then apply the original row/column test only to the surviving matches.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    bg = get_bg(g)
    cell_size = 5
    query_size = 3
    query_step = 4

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
            table_map[sig] = (int(ri), int(ci))

    query_r0 = max(0, (top_sep - query_size) // 2)
    query_cols = [c for c in range(col0, w - query_size + 1, query_step)]
    query_sigs = [
        sig
        for c in query_cols
        if (sig := _cell_signature(g, query_r0, c, query_size, query_size, bg)) is not None
    ]

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

    needed = len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = len(real_query_sigs) == needed and len(selected) == needed and (same_row or same_col)

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
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

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


def _pick_distractors(
    table_masks: list[int],
    used_query_sigs: set[tuple[tuple[int, int, int], ...]],
    count: int,
) -> list[tuple[int, int]]:
    distractors: list[tuple[int, int]] = []
    mask_choices = [mask for mask in ALL_QUERY_PATTERNS if mask not in set(table_masks)]
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
    table_sig_set: set[tuple[tuple[int, int, int], ...]] = set()
    glyph_specs: list[tuple[int, int]] = []
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        glyph_specs.append((mask, color))
        table_sig_set.add(_signature_from_mask(mask, color))
        glyph = _table_glyph(mask, color)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
        distractor_count = 1
    else:
        real_count = 3 if random.random() < P_REAL_QUERY_3 else 4
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

    matched_count = 0
    for slot, (kind, payload) in enumerate(query_items):
        if kind == "real":
            r_i, c_i = payload  # type: ignore[misc]
            mask, color = glyph_specs[r_i * TABLE_DIM + c_i]
            matched_count += 1
        else:
            mask, color = payload  # type: ignore[misc]
        paste(grid, _query_glyph(mask, color), QUERY_ROW, QUERY_COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    out = main(grid)
    if matched_count != len(query_cells):
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 D2 variant input")
