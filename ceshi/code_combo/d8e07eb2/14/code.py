# Combo variant of d8e07eb2 (index 14)
# combo_spec: D2/3+D3/2
# Name: 含干扰查询+参考表有重复

# Design analysis
# ---
# combo_variant:
#   summary:
#     - Keep the original 30x22 canvas, separator rows, 4x4 table, and verdict band.
#     - Apply D2/3 by shrinking the query source area to five compact 3x3 slots so
#       there is room for 1-2 distractors in the fixed top band.
#     - Apply D3/2 by forcing 1-2 genuine query glyph types to appear twice in the
#       4x4 table, which means a real query may have multiple candidate cells.
#   concrete_pixels:
#     - Rows 0..4 stay the query band, but the visible glyph windows are now 3x3
#       patches at row 1 and columns 1,5,9,13,17.
#     - Rows 7..26 remain the 4x4 table of 5x5 cells with centered 3x3 glyphs.
#     - Duplicate-table behavior is realized by copying the exact same color+shape
#       into a second table cell; the duplicate copy is always outside the intended
#       selected set, so humans can see the ambiguity directly in the reference area.
#   behavior_trace:
#     - Example sketch: top slots show [A, X, B, C, D], where X is a distractor.
#       In the table, A and C each appear twice, but one occurrence of each lies on
#       the same row as B and D.
#     - Original logic would fail in two ways:
#         1. it would count X as a real query, making a five-glyph success impossible;
#         2. it would green every A/C duplicate because table membership alone would
#            be treated as a match.
#     - New logic first filters the five visible glyphs against the table, then
#       chooses exactly one table cell per surviving glyph. Among all such
#       assignments it prefers the one with the strongest single-row alignment; if
#       no row beats columns, it prefers the strongest column alignment; remaining
#       ties go to the smaller row/column index and then the lexicographically
#       smallest cell set. This resolves duplicates into one coherent set.
#   complexity_calibration:
#     - Every generated sample contains at least one distractor and at least one
#       duplicated real-query glyph type, so both new dimensions are active.
#     - Duplicated query types are limited to 1-2, and each is duplicated exactly
#       once. That keeps the search space small and visually legible.
#     - Real queries themselves remain distinct glyph types. The ambiguity comes
#       from repeated table entries, not from duplicated query slots.
#   implicit_assumption_review:
#     - Old top-band extraction assumed every non-empty query slot was genuine.
#       D2/3 breaks that; the new solver filters by membership in the table map.
#     - Old table matching assumed a signature maps to at most one cell. D3/2
#       breaks that; the new solver stores a list of candidate cells per signature.
#     - Old highlighting loop greened every matching table cell. That is now wrong:
#       duplicate cells must be disambiguated, so the new solver enumerates
#       one-cell-per-query assignments instead.
#     - Old success test compared raw visible query count with the table width. That
#       no longer holds because visible slots include distractors. The new success
#       test uses only filtered real queries and the resolved assignment.
#     - New assumption introduced here: each real query has only a small candidate
#       list, and there is a deterministic best assignment under the row/column
#       priority rule. The generator guarantees this by construction and rejects
#       samples where the oracle chooses a different cell set than intended.
#   placement_legality:
#     - The five 3x3 query windows fit cleanly inside rows 1..3 and columns 1..19
#       with one-column gaps, so they never overlap each other or the separators.
#     - Duplicate table cells are chosen from distinct 5x5 positions in the 4x4
#       grid, and no duplicate destination overlaps a source cell.
#     - The generator always validates `main(input)` against an independently built
#       expected output before accepting the sample.
#   observability:
#     - From 2-4 examples, a human can see two separate phenomena:
#         1. some top glyphs have no counterpart anywhere in the table and are ignored;
#         2. some real top glyphs have two visible counterparts in the table, but
#            only one of them is highlighted, namely the one that best completes a
#            row/column pattern with the other real queries.
#   visual_cleanliness:
#     - The top band contains only one layer of compact glyphs on plain teal.
#     - Duplicate ambiguity is shown inside the table itself, without extra marks or
#       overlays, so the before/after relation remains readable.
#   inferability:
#     - No hidden color arithmetic is used. Matching is exact visible color+shape.
#     - The duplicate-resolution priority is structural: prefer the assignment that
#       makes the selected cells most row-like, otherwise most column-like, then use
#       top/left tie-breaks. This is observable from the greened cells.

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste

Signature = tuple[tuple[int, int, int], ...]
CellPos = tuple[int, int]


def _constant_non_bg_rows(g: np.ndarray, bg: int) -> list[int]:
    return [
        int(r)
        for r in range(g.shape[0])
        if bool(np.all(g[r, :] == g[r, 0])) and int(g[r, 0]) != int(bg)
    ]


def _normalized_signature(patch: np.ndarray, bg: int) -> Signature | None:
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
) -> Signature | None:
    patch = g[int(r0) : int(r0) + int(h), int(c0) : int(c0) + int(w)]
    return _normalized_signature(patch, bg)


def _unique_in_order(items: list[Signature]) -> list[Signature]:
    seen: set[Signature] = set()
    out: list[Signature] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _dominant_line(cells: list[CellPos]) -> tuple[str, int, int]:
    if not cells:
        return ("row", 0, 0)

    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c in cells:
        row_counts[int(r)] = row_counts.get(int(r), 0) + 1
        col_counts[int(c)] = col_counts.get(int(c), 0) + 1

    max_row = max(row_counts.values())
    best_row = min(r for r, cnt in row_counts.items() if cnt == max_row)
    max_col = max(col_counts.values())
    best_col = min(c for c, cnt in col_counts.items() if cnt == max_col)

    if max_row >= max_col:
        return ("row", int(best_row), int(max_row))
    return ("col", int(best_col), int(max_col))


def _assignment_score(cells: list[CellPos]) -> tuple[int, int, int, tuple[tuple[int, int], ...]]:
    kind, idx, count = _dominant_line(cells)
    canon = tuple(sorted((int(r), int(c)) for r, c in cells))
    inv_canon = tuple((-r, -c) for r, c in canon)
    return (
        int(count),
        1 if kind == "row" else 0,
        -int(idx),
        inv_canon,
    )


def _best_assignment(query_sigs: list[Signature], table_map: dict[Signature, list[CellPos]]) -> list[CellPos]:
    if not query_sigs:
        return []

    best_cells: list[CellPos] | None = None
    best_score: tuple[int, int, int, tuple[tuple[int, int], ...]] | None = None
    used: set[CellPos] = set()
    cur: list[CellPos] = []

    def dfs(i: int) -> None:
        nonlocal best_cells, best_score
        if i == len(query_sigs):
            score = _assignment_score(cur)
            if best_score is None or score > best_score:
                best_score = score
                best_cells = list(cur)
            return

        sig = query_sigs[i]
        for cell in sorted(table_map[sig]):
            if cell in used:
                continue
            used.add(cell)
            cur.append(cell)
            dfs(i + 1)
            cur.pop()
            used.remove(cell)

    dfs(0)
    assert best_cells is not None, "every real query should have at least one table match"
    return best_cells


def _infer_layout(g: np.ndarray, bg: int) -> tuple[int, int, list[int], list[int]]:
    h, w = g.shape
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
    assert row_starts and col_starts and h >= bottom_sep + 1, "could not infer table cell grid"
    return int(top_sep), int(bottom_sep), row_starts, col_starts


def _table_map(g: np.ndarray, bg: int, row_starts: list[int], col_starts: list[int]) -> dict[Signature, list[CellPos]]:
    out: dict[Signature, list[CellPos]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c, CELL, CELL, bg)
            if sig is None:
                continue
            out.setdefault(sig, []).append((int(ri), int(ci)))
    return out


def _query_signatures(g: np.ndarray, bg: int, col0: int, top_sep: int) -> list[Signature]:
    query_r0 = max(0, (top_sep - QUERY_SIZE) // 2)
    query_cols = [c for c in range(col0, g.shape[1] - QUERY_SIZE + 1, QUERY_STEP)]
    return [
        sig
        for c in query_cols
        if (sig := _cell_signature(g, query_r0, c, QUERY_SIZE, QUERY_SIZE, bg)) is not None
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve real query glyphs while ignoring distractors and breaking duplicate ties.

    Input structure:
    - Teal background with two full-width pink separator rows.
    - The top band contains five compact 3x3 glyph slots; 3-4 are genuine table
      lookups and the remaining 1-2 are distractors with no table match.
    - Between separators sits the original 4x4 table of 5x5 cells, but some
      cells may now repeat the exact same color+shape glyph.

    Rule:
    - Read the top-band glyphs by exact visible color+shape and discard any that
      do not occur in the table at all.
    - For each surviving query glyph, choose exactly one matching table cell.
      Among all one-cell-per-query assignments, prefer the set with the strongest
      single-row alignment; if rows do not win, prefer the strongest single
      column alignment; break remaining ties by smaller row/column index and then
      by the top-left-most cell set.
    - Green-fill the chosen table cells. If the resolved real queries are four
      glyphs and their chosen cells form a complete row or column, also
      green-fill the query band and the verdict band; otherwise fill only the
      verdict band red.

    Key insight:
    the query band is noisy and the table is ambiguous. Solving requires two
    consecutive filters: first remove top-band distractors, then collapse each
    duplicated table glyph to the occurrence that best completes a coherent line.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    top_sep, bottom_sep, row_starts, col_starts = _infer_layout(g, bg)
    table_map = _table_map(g, bg, row_starts, col_starts)
    query_sigs = _query_signatures(g, bg, col_starts[0], top_sep)
    real_query_sigs = _unique_in_order([sig for sig in query_sigs if sig in table_map])
    selected = _best_assignment(real_query_sigs, table_map) if real_query_sigs else []

    out = g.copy()
    for ri, ci in selected:
        r = row_starts[ri]
        c = col_starts[ci]
        cell = out[r : r + CELL, c : c + CELL]
        cell[cell == bg] = GREEN

    kind, _idx, count = _dominant_line(selected)
    needed = len(col_starts)
    success = len(real_query_sigs) == needed and len(selected) == needed and count == needed and kind in {"row", "col"}

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED

    return out


# --------------- generator (combo D2/3 + D3/2) ---------------

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
QUERY_STEP = 4

# -- Glyph shape --
FILL_MIN, FILL_MAX = 5, 9
SHAPE_BASE_COUNTS = {5: 2, 6: 2, 7: 2, 8: 2, 9: 1}
MAX_SHAPES_PER_FILL = 5

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.4
P_REAL_QUERY_3 = 0.4
P_DUPLICATE_TWO_FAIL = 0.45
FAIL_MODES = ("rect", "bent_row", "bent_col", "scatter_rows", "scatter_cols")

# -- Generation --
MAX_RETRIES = 500


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
ALL_TABLE_CELLS = [(r, c) for r in range(TABLE_DIM) for c in range(TABLE_DIM)]


def _balanced_color_list(total: int) -> list[int]:
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = total - len(GLYPH_COLORS) * COLOR_MIN
    assert extras >= 0
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _balanced_shape_masks(total: int) -> list[int]:
    counts = dict(SHAPE_BASE_COUNTS)
    extras = total - sum(counts.values())
    assert extras >= 0
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


def _signature_from_mask(mask: int, color: int) -> Signature:
    return tuple(
        (int(r), int(c), int(color))
        for r in range(3)
        for c in range(3)
        if (mask >> (r * 3 + c)) & 1
    )


def _line_cells() -> list[CellPos]:
    if random.random() < 0.5:
        r = random.randrange(TABLE_DIM)
        cells = [(r, c) for c in range(TABLE_DIM)]
    else:
        c = random.randrange(TABLE_DIM)
        cells = [(r, c) for r in range(TABLE_DIM)]
    random.shuffle(cells)
    return cells


def _fail_cells(count: int) -> list[CellPos]:
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
    table_masks: set[int],
    used_query_sigs: set[Signature],
    count: int,
) -> list[tuple[int, int]]:
    distractors: list[tuple[int, int]] = []
    mask_choices = [mask for mask in ALL_QUERY_PATTERNS if mask not in table_masks]
    while len(distractors) < count:
        mask = random.choice(mask_choices)
        color = random.choice(GLYPH_COLORS)
        sig = _signature_from_mask(mask, color)
        if sig in used_query_sigs:
            continue
        used_query_sigs.add(sig)
        distractors.append((mask, color))
    return distractors


def _expected_output(grid: np.ndarray, cells: list[CellPos], success: bool) -> np.ndarray:
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


def _build_specs(unique_count: int) -> list[tuple[int, int]]:
    masks = _balanced_shape_masks(unique_count)
    colors = _balanced_color_list(unique_count)
    return list(zip(masks, colors, strict=True))


def _choose_duplicate_destinations(source_cells: list[CellPos], duplicate_count: int) -> list[CellPos]:
    remaining = [cell for cell in ALL_TABLE_CELLS if cell not in source_cells]
    random.shuffle(remaining)
    remaining.sort(reverse=True)
    return remaining[:duplicate_count]


def _resolved_cells_for_grid(grid: np.ndarray) -> tuple[list[CellPos], list[Signature]]:
    bg = get_bg(grid)
    top_sep, _bottom_sep, row_starts, col_starts = _infer_layout(grid, bg)
    table_map = _table_map(grid, bg, row_starts, col_starts)
    query_sigs = _query_signatures(grid, bg, col_starts[0], top_sep)
    real_query_sigs = _unique_in_order([sig for sig in query_sigs if sig in table_map])
    selected = _best_assignment(real_query_sigs, table_map) if real_query_sigs else []
    return selected, real_query_sigs


def _gen_attempt() -> np.ndarray | None:
    grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    grid[TOP_SEP_ROW, :] = SEP
    grid[BOTTOM_SEP_ROW, :] = SEP

    success_target = random.random() < P_SUCCESS
    real_count = TABLE_DIM if success_target else (3 if random.random() < P_REAL_QUERY_3 else 4)
    source_cells = _line_cells() if success_target else _fail_cells(real_count)
    duplicate_count = 2 if success_target or random.random() < P_DUPLICATE_TWO_FAIL else 1

    unique_specs = _build_specs(TABLE_DIM * TABLE_DIM - duplicate_count)
    color_counts: dict[int, int] = {}
    for _mask, color in unique_specs:
        color_counts[color] = color_counts.get(color, 0) + 1

    safe_specs = [spec for spec in unique_specs if color_counts[spec[1]] <= COLOR_MAX - 1]
    if len(safe_specs) < duplicate_count:
        return None

    selected_specs: list[tuple[int, int]] = random.sample(safe_specs, duplicate_count)
    remaining_specs = [spec for spec in unique_specs if spec not in selected_specs]
    if len(remaining_specs) < real_count - duplicate_count:
        return None
    selected_specs.extend(random.sample(remaining_specs, real_count - duplicate_count))
    random.shuffle(selected_specs)

    duplicate_sources = random.sample(list(range(real_count)), duplicate_count)
    duplicate_destinations = _choose_duplicate_destinations(source_cells, duplicate_count)
    if len(set(duplicate_destinations)) != duplicate_count:
        return None

    assignment: dict[CellPos, tuple[int, int]] = {}
    for cell, spec in zip(source_cells, selected_specs, strict=True):
        assignment[cell] = spec

    base_pool = [spec for spec in unique_specs if spec not in selected_specs]
    non_dup_positions = [cell for cell in ALL_TABLE_CELLS if cell not in source_cells and cell not in duplicate_destinations]
    if len(base_pool) != len(non_dup_positions):
        return None
    random.shuffle(base_pool)
    for cell, spec in zip(non_dup_positions, base_pool, strict=True):
        assignment[cell] = spec

    for dst, src_idx in zip(duplicate_destinations, duplicate_sources, strict=True):
        assignment[dst] = selected_specs[src_idx]

    if set(assignment) != set(ALL_TABLE_CELLS):
        return None

    table_masks = {mask for mask, _color in assignment.values()}
    table_sig_map: dict[Signature, list[CellPos]] = {}
    for cell, (mask, color) in assignment.items():
        sig = _signature_from_mask(mask, color)
        table_sig_map.setdefault(sig, []).append(cell)
        paste(grid, _table_glyph(mask, color), ROWS[cell[0]], COLS[cell[1]])

    source_sigs = [_signature_from_mask(*assignment[cell]) for cell in source_cells]
    if len(set(source_sigs)) != real_count:
        return None

    used_query_sigs = set(source_sigs)
    distractor_count = len(QUERY_COLS) - real_count
    distractors = _pick_distractors(table_masks, used_query_sigs, distractor_count)

    query_items: list[tuple[str, CellPos | tuple[int, int]]] = [("real", cell) for cell in source_cells]
    query_items.extend(("distractor", spec) for spec in distractors)
    random.shuffle(query_items)

    for slot, (kind, payload) in enumerate(query_items):
        if kind == "real":
            cell = payload  # type: ignore[assignment]
            mask, color = assignment[cell]
        else:
            mask, color = payload  # type: ignore[misc]
        paste(grid, _query_glyph(mask, color), QUERY_ROW, QUERY_COLS[slot])

    resolved_cells, real_query_sigs = _resolved_cells_for_grid(grid)
    if len(real_query_sigs) != real_count:
        return None
    if set(resolved_cells) != set(source_cells):
        return None

    kind, _idx, count = _dominant_line(resolved_cells)
    success_actual = len(real_query_sigs) == TABLE_DIM and len(resolved_cells) == TABLE_DIM and count == TABLE_DIM and kind in {"row", "col"}
    if success_actual != success_target:
        return None

    ambiguous_real_queries = sum(1 for sig in real_query_sigs if len(table_sig_map[sig]) > 1)
    if ambiguous_real_queries < 1 or distractor_count < 1:
        return None

    expected = _expected_output(grid, resolved_cells, success_actual)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo input")
