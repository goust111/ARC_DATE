# Combo variant of d8e07eb2 (index 32)
# combo_spec: D1/2+D2/3
# Name: 旋转匹配+含干扰查询

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis
# ----------------
# concretization / behavior trace:
# - The table and verdict layout stay at the source-task coordinates: a 4x4 block
#   of 5x5 cells between two full-width separator rows, plus a 2-row verdict band.
# - The query source region follows D2/3's compact layout: five visible 3x3 glyph
#   slots in the top band at row 1 and columns 1,5,9,13,17.
# - Among those five visible glyphs, 3-4 are genuine table lookups and the rest
#   are distractors. Genuine lookups are not exact copies: each one is a rotated
#   version of a table glyph, so exact matching would fail and only transform-aware
#   matching can recover it. Distractors are chosen so that even after any rotation
#   or reflection they still have no table counterpart.
# - Small trace:
#   source task: query slot shows exact blue shape A -> literal cell signature hits
#   table cell A. Any extra visible query would incorrectly count against success.
#   combo task: query slots show [rot(A), distractor X, rot(B), rot(C), rot(D)].
#   Exact matching misses A/B/C/D because their orientations changed. The fused
#   solver instead canonicalizes each glyph by color plus dihedral-equivalence
#   class, ignores X because its canonical key is absent from the table, then tests
#   whether the surviving four matched cells form one full row or column.
#
# complexity calibration:
# - To avoid ambiguous transform-aware matches, every table glyph uses a distinct
#   asymmetric dihedral class. That guarantees at most one table cell per
#   (color, canonical-shape) key.
# - Success cases always contain 4 real queries + 1 distractor, so the output must
#   visibly demonstrate both "ignore extras" and "rotation needed". Fail cases use
#   either 3 real + 2 distractors or 4 real + 1 distractor to keep the original
#   row/column judgment legible.
# - Distractors reuse ordinary glyph colors instead of introducing a separate color
#   code. Their non-match is still human-visible because the whole rotated/flipped
#   shape class is absent from the table.
#
# implicit assumption review:
# - The source solver assumed every non-empty query slot was genuine and occupied a
#   full 5x5 cell. Both assumptions break here: query glyphs are compact 3x3
#   patches and some visible glyphs must be ignored.
# - The source `_cell_signature` relied on absolute in-cell coordinates, which is
#   incompatible with rotation/reflection matching. The fused solver replaces that
#   with a canonical dihedral key extracted from the 3x3 occupancy mask plus color.
# - The source success test used the visible query count. That no longer works
#   because success samples have 5 visible glyphs but only 4 real matches. The new
#   success condition is based on the number of filtered real matches.
# - New solver assumptions:
#   query patches and table cells are monochrome glyphs;
#   table glyph pixels stay inside the centered 3x3 region of each 5x5 cell;
#   table keys are unique under color + dihedral class.
#   The generator enforces each of these invariants explicitly.
#
# placement legality:
# - Query slots are fixed 3x3 windows fully inside rows 1..3 and cols 1..19, with
#   one blank column between neighbors. They never overlap each other or the table.
# - Table cells remain at the original 5x5 positions, so no new overlap or bounds
#   issues are introduced in the lower bands.
# - The generator still validates every candidate by running `main()` and comparing
#   against an independently constructed expected output.
#
# observability / cleanliness / inferability:
# - Every real query is visibly rotated relative to its source table glyph, so the
#   transform-aware rule is necessary in the samples instead of being latent.
# - Every sample contains visible distractors in the same top band, but the output
#   only highlights table cells for the true queries. That makes the filtering step
#   observable without extra markers.
# - The top band uses one clean layer of 3x3 glyphs with fixed spacing; there is no
#   overlap, no hidden ordering, and no reliance on raw color IDs or non-visual
#   arithmetic. The only inference is visible color + shape under rotation/flip.


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
        rot = np.rot90(arr, int(k))
        for candidate in (rot, np.fliplr(rot)):
            key = _mask_to_int(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return tuple(out)


def _canonical_mask_int(mask: int) -> int:
    return min(_mask_variants(mask))


def _table_cell_key(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[int, int] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    fg = cell != int(bg)
    if not bool(fg.any()):
        return None

    inner = fg[1:4, 1:4]
    outer = fg.copy()
    outer[1:4, 1:4] = False
    assert not bool(outer.any()), "table glyph pixels must stay inside the centered 3x3 region"

    colors = {int(x) for x in cell[fg].tolist()}
    assert len(colors) == 1, "glyph cells must be monochrome"
    return next(iter(colors)), _canonical_mask_int(_mask_to_int(inner))


def _query_patch_key(g: np.ndarray, r0: int, c0: int, size: int, bg: int) -> tuple[int, int] | None:
    patch = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    fg = patch != int(bg)
    if not bool(fg.any()):
        return None

    colors = {int(x) for x in patch[fg].tolist()}
    assert len(colors) == 1, "query glyphs must be monochrome"
    return next(iter(colors)), _canonical_mask_int(_mask_to_int(fg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match compact query glyphs to the table under transform equivalence, while ignoring decoys.

    Input structure:
    - A teal background is split by two full-width pink separator rows.
    - The top band contains five compact 3x3 query glyph slots; some are real
      lookups and the remaining 1-2 are distractors with no table counterpart.
    - Between the separators sits the original 4x4 table of 5x5 cells, each
      holding one centered monochrome 3x3 glyph.

    Rule:
    - Compare each visible query glyph to the table by color plus 3x3 shape up
      to rotation or reflection, and discard any query glyph that still has no
      match after that normalization.
    - Green-fill the background of every table cell matched by the surviving
      query glyphs.
    - If those matched cells make one complete row or one complete column of the
      table, also green-fill the query band and the verdict band; otherwise
      leave the query band unchanged and fill only the verdict band red.

    Key insight: solving now requires two filters in sequence: first mentally
    rotate/flip query glyphs to find their table counterparts, then ignore the
    extra top-band glyphs that still do not belong to the table at all.
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

    table_map: dict[tuple[int, int], tuple[int, int]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            key = _table_cell_key(g, r, c, cell_size, bg)
            assert key is not None, "reference table should be fully populated"
            assert key not in table_map, "table glyphs must be unique under color+transform equivalence"
            table_map[key] = (int(ri), int(ci))

    query_r0 = max(0, (top_sep - query_size) // 2)
    query_cols = [c for c in range(col0, w - query_size + 1, query_step)]
    query_keys = [
        key
        for c in query_cols
        if (key := _query_patch_key(g, query_r0, c, query_size, bg)) is not None
    ]
    assert len(query_keys) == len(set(query_keys)), "visible query glyphs should be distinct"

    real_query_keys = {key for key in query_keys if key in table_map}

    out = g.copy()
    selected: list[tuple[int, int]] = []
    for key in real_query_keys:
        ri, ci = table_map[key]
        selected.append((int(ri), int(ci)))
        r = row_starts[ri]
        c = col_starts[ci]
        cell = out[r : r + cell_size, c : c + cell_size]
        cell[cell == bg] = 3

    needed = len(col_starts)
    same_row = len({r for r, _c in selected}) == 1
    same_col = len({c for _r, c in selected}) == 1
    success = len(real_query_keys) == needed and len(selected) == needed and (same_row or same_col)

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


def _canonical_class_masks() -> dict[int, tuple[int, ...]]:
    by_class: dict[int, list[int]] = {}
    for mask in ALL_QUERY_PATTERNS:
        canon = _canonical_mask_int(mask)
        by_class.setdefault(canon, []).append(mask)
    return {canon: tuple(sorted(set(masks))) for canon, masks in by_class.items()}


CANONICAL_CLASS_MASKS = _canonical_class_masks()
ALL_CANONICAL_CLASSES = tuple(sorted(CANONICAL_CLASS_MASKS))


def _asymmetric_shape_classes() -> tuple[int, ...]:
    reps = [canon for canon, masks in CANONICAL_CLASS_MASKS.items() if len(_mask_variants(masks[0])) == 8]
    out = tuple(sorted(reps))
    assert len(out) == TABLE_DIM * TABLE_DIM, "expected exactly sixteen asymmetric shape classes"
    return out


ASYMMETRIC_CLASSES = _asymmetric_shape_classes()
DISTRACTOR_CLASSES = tuple(canon for canon in ALL_CANONICAL_CLASSES if canon not in set(ASYMMETRIC_CLASSES))
ROT_OPS = (1, 2, 3)


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
    masks = [random.choice(CANONICAL_CLASS_MASKS[canon]) for canon in classes]
    assert len({_canonical_mask_int(mask) for mask in masks}) == len(masks)
    return masks


def _transform_mask(mask: int, rot_k: int) -> int:
    return _mask_to_int(np.rot90(_int_to_mask(mask), int(rot_k)))


def _table_glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    cell[1:4, 1:4][_int_to_mask(mask)] = color
    return cell


def _query_glyph(mask: int, color: int) -> np.ndarray:
    glyph = np.full((QUERY_SIZE, QUERY_SIZE), BG, dtype=int)
    glyph[_int_to_mask(mask)] = color
    return glyph


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
    used_query_keys: set[tuple[int, int]],
    used_exact_specs: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    distractors: list[tuple[int, int]] = []
    while len(distractors) < count:
        canon = random.choice(DISTRACTOR_CLASSES)
        color = random.choice(GLYPH_COLORS)
        key = (color, canon)
        mask = random.choice(CANONICAL_CLASS_MASKS[canon])
        exact = (color, mask)
        if key in used_query_keys or exact in used_exact_specs:
            continue
        used_query_keys.add(key)
        used_exact_specs.add(exact)
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
    table_keys: set[tuple[int, int]] = set()
    table_exact_specs: set[tuple[int, int]] = set()
    for i, (mask, color) in enumerate(zip(masks, colors, strict=True)):
        table_specs.append((mask, color))
        table_keys.add((color, _canonical_mask_int(mask)))
        table_exact_specs.add((color, mask))
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, _table_glyph(mask, color), ROWS[r_i], COLS[c_i])

    assert len(table_specs) == TABLE_DIM * TABLE_DIM
    assert len(table_keys) == len(table_specs)

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _line_cells()
        distractor_count = 1
    else:
        real_count = 3 if random.random() < P_REAL_QUERY_3 else 4
        query_cells = _fail_cells(real_count)
        distractor_count = len(QUERY_COLS) - real_count

    real_queries: list[tuple[int, int, tuple[int, int]]] = []
    used_query_keys: set[tuple[int, int]] = set()
    used_exact_specs: set[tuple[int, int]] = set()
    for r_i, c_i in query_cells:
        table_mask, color = table_specs[r_i * TABLE_DIM + c_i]
        query_mask = _transform_mask(table_mask, random.choice(ROT_OPS))
        key = (color, _canonical_mask_int(query_mask))
        exact = (color, query_mask)
        assert key in table_keys
        assert exact not in table_exact_specs, "real queries must rely on rotated matching"
        assert query_mask != table_mask, "real queries must visibly rotate the source glyph"
        used_query_keys.add(key)
        used_exact_specs.add(exact)
        real_queries.append((query_mask, color, (r_i, c_i)))

    distractors = _pick_distractors(used_query_keys, used_exact_specs, distractor_count)

    query_items: list[tuple[str, tuple[int, int] | tuple[int, int, tuple[int, int]]]] = [
        ("real", item) for item in real_queries
    ] + [("distractor", item) for item in distractors]
    random.shuffle(query_items)
    assert len(query_items) == len(QUERY_COLS)

    for slot, (kind, payload) in enumerate(query_items):
        if kind == "real":
            mask, color, _cell = payload  # type: ignore[misc]
        else:
            mask, color = payload  # type: ignore[misc]
            assert (color, _canonical_mask_int(mask)) not in table_keys
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
