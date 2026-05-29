# Combo variant of d8e07eb2 (index 3)
# combo_spec: D1/4+D5/2
# Name: 像素计数匹配+对角线判定

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
#    - The canvas keeps the original 30x22 layout: a 5-row query band, two
#      full-width separator rows, a 4x4 table of 5x5 cells, and a 2-row verdict
#      band at the bottom.
#    - Original task:
#        1. each query glyph is an exact copy of one table glyph;
#        2. matching key = (color, exact 3x3 shape);
#        3. success iff the 4 matched cells form one full row or one full column.
#    - Combo task:
#        1. query glyphs are usually NOT exact copies of their source cells;
#        2. matching key = (color, foreground-pixel count) only;
#        3. success iff the 4 matched cells are exactly one full main or
#           anti-diagonal.
#    - Example trace:
#        table cell A at (0,0) = color 4, six painted pixels, shape "T"
#        query glyph Q0        = color 4, six painted pixels, shape "zigzag"
#      Original solver would reject Q0 because the shapes differ.
#      Combo solver accepts Q0 because color and painted-pixel count match.
#      If the four accepted cells are {(0,0),(1,1),(2,2),(3,3)}, the verdict is
#      green; if they are {(1,0),(1,1),(1,2),(1,3)}, the table cells still get
#      highlighted but the verdict is red because a row is no longer sufficient.
#
# b) Complexity calibration
#    - Ambiguity would be fatal here because matching no longer uses exact shape.
#      The generator therefore makes every table cell's (color, pixel-count)
#      pair unique, so each query glyph still points to exactly one table cell.
#    - To keep the D1 change visible, query glyphs are synthesized from an
#      alternate 3x3 mask with the SAME pixel count but a DIFFERENT shape.
#    - Pixel counts are restricted to 5-8 rather than 5-9. Count 9 would force
#      the full 3x3 block, leaving no alternate shape for the query and making
#      the D1 change visually disappear.
#    - To keep the D5 change visible, fail samples often use complete rows or
#      columns; those would pass under the old rule but must fail here.
#
# c) Implicit-assumption audit
#    - `_cell_key` assumes a non-empty glyph cell is monochrome. The generator
#      preserves that by drawing each 3x3 glyph with a single foreground color.
#    - The original `_cell_signature`-based uniqueness assumption no longer
#      matters. Instead, `main()` now assumes table keys `(color, count)` are
#      unique and query keys are distinct; the generator enforces both.
#    - `complete_query = len(query_keys) == len(col_starts)` is still valid:
#      a successful diagonal still requires exactly four occupied query slots.
#    - The old `same_row` / `same_col` test is invalid after D5/2. It is
#      replaced by exact set comparison against the two 4-cell diagonals.
#
# d) Placement legality
#    - Table cells stay at the original fixed 5x5 slots, so they never overlap
#      and always fit inside rows 7-26 and cols 1-20.
#    - Query glyphs are pasted only into the four existing top-band slots
#      starting at `COLS`, with at most one glyph per slot.
#    - Every synthesized query mask uses the same centered 3x3-with-border
#      format as the table, so no query glyph can spill outside its 5x5 cell.
#
# e) Visual observability
#    - A human can see the D1 change because top glyphs are deliberately shape-
#      mismatched against their highlighted table counterparts while keeping the
#      same color and amount of paint. This forces "count the pixels" reasoning.
#    - A human can see the D5 change because some red outputs highlight an entire
#      row or column, directly falsifying the old row/column rule.
#    - No hidden ordering by raw color id is used; color only acts as visible
#      equality, and the other key is a visibly countable number of painted cells.
#
# f) Visual cleanliness
#    - Output markings stay minimal: only background pixels inside matched table
#      cells turn green; separators keep their own color; the verdict band is a
#      clean solid strip.
#    - Query glyphs are centered in the same four slots as before, so the scene
#      remains easy to parse despite the new matching rule.
#
# g) Color-rule audit
#    - There is no color remapping. Query/table glyph colors are preserved.
#    - Green still means highlighted success state, red still means failure.
#    - The new rule depends on visible equality of colors plus visible pixel
#      counts, not on any invisible numeric property of the palette.


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


def _cell_key(
    g: np.ndarray, r0: int, c0: int, size: int, bg: int
) -> tuple[int, int] | None:
    cell = g[int(r0) : int(r0) + int(size), int(c0) : int(c0) + int(size)]
    mask = cell != int(bg)
    if not bool(mask.any()):
        return None
    colors = np.unique(cell[mask])
    assert len(colors) == 1, "glyph cells must be monochrome"
    return int(colors[0]), int(mask.sum())


def _diagonal_cell_sets(dim: int) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    main_diag = frozenset((i, i) for i in range(dim))
    anti_diag = frozenset((i, dim - 1 - i) for i in range(dim))
    return main_diag, anti_diag


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match query glyphs to the reference table by color plus painted-pixel count,
    then judge whether the matched cells trace a full diagonal.

    Input structure:
    - Teal background with two full-width separator rows.
    - The top band contains up to four 5x5 query cells, each holding one
      centered 3x3 monochrome glyph.
    - Between the separators is a 4x4 table of the same 5x5 cells; table glyphs
      may differ in shape from the query glyphs even when they are meant to match.

    Rule:
    - Ignore exact 3x3 shape when matching. A query glyph matches a table cell
      iff they have the same foreground color and the same number of painted
      pixels.
    - Green-fill the background inside every matched table cell.
    - If there are exactly four query glyphs and their matched table cells are
      exactly the main diagonal or the anti-diagonal, also green-fill the query
      band and the verdict band; otherwise leave the query band unchanged and
      fill the verdict band red.

    Key insight: The top-band glyph shapes are decoys. The real lookup key is
    visible COUNTING within each color, and the final yes/no decision depends on
    whether the matched locations form one complete diagonal of the 4x4 table.
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

    query_keys = [
        key
        for c in col_starts
        if (key := _cell_key(g, 0, c, cell_size, bg)) is not None
    ]
    query_key_set = set(query_keys)
    assert len(query_keys) == len(query_key_set), "query glyph keys should be distinct"

    out = g.copy()
    selected: list[tuple[int, int]] = []
    seen_table_keys: set[tuple[int, int]] = set()
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            key = _cell_key(g, r, c, cell_size, bg)
            if key is None:
                continue
            assert key not in seen_table_keys, "table keys must be unique under color+count matching"
            seen_table_keys.add(key)
            if key not in query_key_set:
                continue
            selected.append((int(ri), int(ci)))
            cell = out[r : r + cell_size, c : c + cell_size]
            cell[cell == bg] = 3

    square_table = len(row_starts) == len(col_starts)
    complete_query = len(query_keys) == len(col_starts)
    if square_table:
        diag_sets = _diagonal_cell_sets(len(row_starts))
        is_complete_diagonal = frozenset(selected) in diag_sets
    else:
        is_complete_diagonal = False
    success = complete_query and len(selected) == len(query_keys) and is_complete_diagonal

    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == bg] = 3
        out[bottom_sep + 1 :, :] = 3
    else:
        out[bottom_sep + 1 :, :] = 2

    return out


# --------------- generator (combo output-first) ---------------

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

# -- Glyph shape / count --
FILL_COUNTS = (5, 6, 7, 8)

# -- Color distribution --
COLOR_MIN, COLOR_MAX = 1, 3

# -- Outcome probabilities --
P_SUCCESS = 0.5
P_QUERY_3 = 0.3
FAIL_MODES = ("row", "row", "col", "col", "off_main", "off_anti", "rect", "perm")

# -- Generation --
MAX_RETRIES = 300


def _has_full_bbox(mask: int) -> bool:
    """True if the 3x3 bitmask spans all four edges."""
    bits = [(mask >> i) & 1 for i in range(9)]
    return (
        any(bits[0:3])
        and any(bits[6:9])
        and any(bits[i] for i in (0, 3, 6))
        and any(bits[i] for i in (2, 5, 8))
    )


PATTERNS_BY_FILL = {
    k: tuple(m for m in range(1 << 9) if m.bit_count() == k and _has_full_bbox(m))
    for k in FILL_COUNTS
}

NON_DIAGONAL_PERMS = tuple(
    perm
    for perm in itertools.permutations(range(TABLE_DIM))
    if perm not in ((0, 1, 2, 3), (3, 2, 1, 0))
)


def _balanced_color_list() -> list[int]:
    """Assign 16 colors so each glyph color appears 1-3 times."""
    counts = {c: COLOR_MIN for c in GLYPH_COLORS}
    extras = TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS) * COLOR_MIN
    for _ in range(extras):
        choices = [c for c in GLYPH_COLORS if counts[c] < COLOR_MAX]
        counts[random.choice(choices)] += 1
    colors = [c for c, n in counts.items() for _ in range(n)]
    random.shuffle(colors)
    return colors


def _assign_fill_counts(colors: list[int]) -> list[int]:
    """
    Assign each cell a fill count from {5,6,7,8}, with no repeated count inside
    the same color group so that every (color, count) key is unique.
    """
    groups: dict[int, list[int]] = {}
    for idx, color in enumerate(colors):
        groups.setdefault(int(color), []).append(int(idx))

    fills = [0] * len(colors)
    usage = {count: 0 for count in FILL_COUNTS}
    for color in random.sample(list(groups.keys()), len(groups)):
        available = list(FILL_COUNTS)
        chosen: list[int] = []
        for _ in groups[color]:
            min_usage = min(usage[count] for count in available)
            candidates = [count for count in available if usage[count] == min_usage]
            count = random.choice(candidates)
            chosen.append(int(count))
            usage[int(count)] += 1
            available.remove(int(count))
        random.shuffle(chosen)
        for idx, count in zip(groups[color], chosen, strict=True):
            fills[idx] = int(count)
    return fills


def _choose_table_masks(fill_counts: list[int]) -> list[int]:
    """Pick distinct table masks while respecting the chosen fill count of each cell."""
    masks = [0] * len(fill_counts)
    for count in FILL_COUNTS:
        indices = [i for i, fill in enumerate(fill_counts) if fill == count]
        picked = random.sample(PATTERNS_BY_FILL[count], len(indices))
        for idx, mask in zip(indices, picked, strict=True):
            masks[idx] = int(mask)
    return masks


def _glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[1:4, 1:4][bits] = color
    return cell


def _main_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, i) for i in range(TABLE_DIM)]


def _anti_diagonal_cells() -> list[tuple[int, int]]:
    return [(i, TABLE_DIM - 1 - i) for i in range(TABLE_DIM)]


def _diagonal_cells() -> list[tuple[int, int]]:
    cells = list(random.choice((_main_diagonal_cells(), _anti_diagonal_cells())))
    random.shuffle(cells)
    return cells


def _is_full_diagonal(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != TABLE_DIM:
        return False
    return frozenset(cells) in _diagonal_cell_sets(TABLE_DIM)


def _scatter_fail_triplet() -> list[tuple[int, int]]:
    rs = random.sample(range(TABLE_DIM), random.choice((2, 3)))
    cs = random.sample(range(TABLE_DIM), random.choice((2, 3)))
    cells = [(rs[0], cs[0]), (rs[1], cs[1])]
    pool = [(r, c) for r in rs for c in cs if (r, c) not in cells]
    cells.append(random.choice(pool))
    assert len(set(cells)) == 3
    return cells


def _fail_cells(count: int) -> list[tuple[int, int]]:
    """Pick distinct cells that do NOT form a complete main/anti diagonal."""
    if count == 3:
        if random.random() < 0.5:
            cells = random.sample(random.choice((_main_diagonal_cells(), _anti_diagonal_cells())), 3)
        else:
            cells = _scatter_fail_triplet()
    else:
        mode = random.choice(FAIL_MODES)
        if mode == "row":
            r = random.randrange(TABLE_DIM)
            cells = [(r, c) for c in range(TABLE_DIM)]
        elif mode == "col":
            c = random.randrange(TABLE_DIM)
            cells = [(r, c) for r in range(TABLE_DIM)]
        elif mode == "off_main":
            diag = set(_main_diagonal_cells())
            removed = random.choice(list(diag))
            replacement_pool = [
                (r, c)
                for r in range(TABLE_DIM)
                for c in range(TABLE_DIM)
                if (r, c) not in diag
            ]
            cells = list((diag - {removed}) | {random.choice(replacement_pool)})
        elif mode == "off_anti":
            diag = set(_anti_diagonal_cells())
            removed = random.choice(list(diag))
            replacement_pool = [
                (r, c)
                for r in range(TABLE_DIM)
                for c in range(TABLE_DIM)
                if (r, c) not in diag
            ]
            cells = list((diag - {removed}) | {random.choice(replacement_pool)})
        elif mode == "rect":
            rs = random.sample(range(TABLE_DIM), 2)
            cs = random.sample(range(TABLE_DIM), 2)
            cells = [(r, c) for r in rs for c in cs]
        else:
            perm = random.choice(NON_DIAGONAL_PERMS)
            cells = [(r, perm[r]) for r in range(TABLE_DIM)]

    assert len(cells) == count and len(set(cells)) == count
    assert not _is_full_diagonal(cells)
    random.shuffle(cells)
    return cells


def _alternate_query_mask(table_mask: int, fill_count: int) -> int:
    """Choose a same-count mask that is maximally different from the table mask."""
    candidates = [mask for mask in PATTERNS_BY_FILL[fill_count] if mask != table_mask]
    assert candidates, "every chosen fill count must admit an alternate query shape"
    best_distance = max((mask ^ table_mask).bit_count() for mask in candidates)
    best = [mask for mask in candidates if (mask ^ table_mask).bit_count() == best_distance]
    return int(random.choice(best))


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

    colors = _balanced_color_list()
    fill_counts = _assign_fill_counts(colors)
    masks = _choose_table_masks(fill_counts)

    glyph_specs: list[dict[str, int]] = []
    for i, (mask, color, fill_count) in enumerate(zip(masks, colors, fill_counts, strict=True)):
        glyph_specs.append({"mask": int(mask), "color": int(color), "fill": int(fill_count)})
        glyph = _glyph(mask, color)
        r_i, c_i = divmod(i, TABLE_DIM)
        paste(grid, glyph, ROWS[r_i], COLS[c_i])

    keys = {(spec["color"], spec["fill"]) for spec in glyph_specs}
    assert len(keys) == TABLE_DIM * TABLE_DIM

    success = random.random() < P_SUCCESS
    if success:
        query_cells = _diagonal_cells()
    else:
        count = 3 if random.random() < P_QUERY_3 else 4
        query_cells = _fail_cells(count)

    for slot, (r_i, c_i) in enumerate(query_cells):
        spec = glyph_specs[r_i * TABLE_DIM + c_i]
        query_mask = _alternate_query_mask(spec["mask"], spec["fill"])
        paste(grid, _glyph(query_mask, spec["color"]), 0, COLS[slot])

    expected = _expected_output(grid, query_cells, success)
    if not np.array_equal(main(grid), expected):
        return None

    # D1 guard: every query glyph should differ in exact shape from its matched table cell.
    for slot, (r_i, c_i) in enumerate(query_cells):
        query_sig = _cell_signature(grid, 0, COLS[slot], CELL, BG)
        table_sig = _cell_signature(grid, ROWS[r_i], COLS[c_i], CELL, BG)
        if query_sig == table_sig:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d8e07eb2 combo variant input")
