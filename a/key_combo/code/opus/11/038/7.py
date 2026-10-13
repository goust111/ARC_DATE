import itertools
import random
from collections import deque
from functools import lru_cache

import numpy as np


BG = 8
SEP = 6
GREEN = 3
RED = 2
CELL = 5
INNER = slice(1, 4)

GLYPH_COLORS = (0, 1, 4, 5, 7, 9)
TABLE_SHAPES = ((3, 4), (4, 3))
MAX_ATTEMPTS = 400


def _neighbors4(node: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = node
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(node: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = node
    return tuple((r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _is_connected(nodes: set[tuple[int, int]], *, eight: bool = False) -> bool:
    if not nodes:
        return False
    step = _neighbors8 if eight else _neighbors4
    seen = set()
    queue = deque([next(iter(nodes))])
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        for nb in step(node):
            if nb in nodes and nb not in seen:
                queue.append(nb)
    return seen == nodes


def _component_count(nodes: set[tuple[int, int]]) -> int:
    remaining = set(nodes)
    count = 0
    while remaining:
        count += 1
        start = next(iter(remaining))
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in remaining:
                continue
            remaining.remove(node)
            for nb in _neighbors4(node):
                if nb in remaining:
                    queue.append(nb)
    return count


def _same_row_or_col(nodes: set[tuple[int, int]]) -> bool:
    rows = {r for r, _ in nodes}
    cols = {c for _, c in nodes}
    return len(rows) == 1 or len(cols) == 1


def _cell_signature(grid: np.ndarray, r: int, c: int) -> tuple[int, tuple[int, ...]] | None:
    cell = grid[r : r + CELL, c : c + CELL]
    assert cell.shape == (CELL, CELL), "expected full 5x5 glyph cell"
    inner = cell[INNER, INNER]
    colors = sorted({int(v) for v in inner.flatten() if int(v) != BG})
    if not colors:
        return None
    assert len(colors) == 1, "glyphs must be monochrome inside the centered 3x3"
    color = colors[0]
    shape = tuple(int(v != BG) for v in inner.flatten())
    return color, shape


def _separator_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, "unexpected colors"

    sep_rows = _separator_rows(g)
    assert len(sep_rows) == 2, "expected exactly two full-width separator rows"
    top_sep, bottom_sep = sep_rows
    assert top_sep == 5, "query band should be one 5x5-cell row tall"
    assert bottom_sep > top_sep + CELL, "reference table must exist between separators"

    table_region = g[top_sep + 1 : bottom_sep, :]
    table_fg = np.argwhere(table_region != BG)
    assert table_fg.size, "reference table must contain glyphs"
    table_top = top_sep + int(table_fg[:, 0].min())
    table_left = int(table_fg[:, 1].min()) - 1
    assert table_top == top_sep + 1, "table cells should start immediately below the top separator"
    assert table_left >= 0, "table must fit in bounds"

    row_starts = [
        r
        for r in range(table_top, bottom_sep, CELL)
        if r + CELL <= bottom_sep and np.any(g[r : r + CELL, :] != BG)
    ]
    col_starts = [
        c
        for c in range(table_left, g.shape[1] - CELL + 1, CELL)
        if np.any(g[table_top:bottom_sep, c : c + CELL] != BG)
    ]
    dims = (len(row_starts), len(col_starts))
    assert dims in TABLE_SHAPES, f"unexpected table shape {dims}"

    query_region = g[:top_sep, :]
    query_fg = np.argwhere(query_region != BG)
    assert query_fg.size, "query band must contain glyphs"
    query_left = int(query_fg[:, 1].min()) - 1
    query_starts = [
        c
        for c in range(query_left, g.shape[1] - CELL + 1, CELL)
        if np.any(g[:top_sep, c : c + CELL] != BG)
    ]
    assert 3 <= len(query_starts) <= 4, "expected 3 or 4 query glyphs"

    table_lookup: dict[tuple[int, tuple[int, ...]], tuple[int, int, int, int]] = {}
    shape_colors: dict[tuple[int, ...], set[int]] = {}
    color_shapes: dict[int, set[tuple[int, ...]]] = {}
    for ri, r in enumerate(row_starts):
        for ci, c in enumerate(col_starts):
            sig = _cell_signature(g, r, c)
            assert sig is not None, "every table cell must contain exactly one glyph"
            assert sig not in table_lookup, "table glyphs must be unique by color+shape"
            table_lookup[sig] = (ri, ci, r, c)
            color, shape = sig
            shape_colors.setdefault(shape, set()).add(color)
            color_shapes.setdefault(color, set()).add(shape)

    query_sigs = []
    matches = []
    for c in query_starts:
        sig = _cell_signature(g, 0, c)
        assert sig is not None, "query cells must contain glyphs"
        query_sigs.append(sig)
        assert sig in table_lookup, "every query glyph must have a table match"
        ri, ci, tr, tc = table_lookup[sig]
        matches.append((ri, ci, tr, tc, sig))

    assert len(query_sigs) == len(set(query_sigs)), "query glyphs must be distinct"
    nodes = {(ri, ci) for ri, ci, _tr, _tc, _sig in matches}
    assert len(nodes) == len(query_sigs), "each query must map to a different table cell"

    return {
        "grid": g,
        "top_sep": top_sep,
        "bottom_sep": bottom_sep,
        "row_starts": row_starts,
        "col_starts": col_starts,
        "query_starts": query_starts,
        "matches": matches,
        "nodes": nodes,
        "shape_colors": shape_colors,
        "color_shapes": color_shapes,
    }


def _render_expected(
    grid: np.ndarray,
    matched_cells: list[tuple[int, int]],
    table_rows_px: list[int],
    table_cols_px: list[int],
    *,
    success: bool,
    top_sep: int,
    bottom_sep: int,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for ri, ci in matched_cells:
        r = table_rows_px[ri]
        c = table_cols_px[ci]
        cell = out[r : r + CELL, c : c + CELL]
        cell[cell == BG] = GREEN
    if success:
        query_band = out[:top_sep, :]
        query_band[query_band == BG] = GREEN
        out[bottom_sep + 1 :, :] = GREEN
    else:
        out[bottom_sep + 1 :, :] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Look up each query glyph in the reference table and judge whether the matched table cells form one 4-connected set.

    Input structure:
    - The grid has teal background and exactly two full-width pink separator rows.
    - Above the first separator is one row of 5x5 query glyph cells, each with a centered monochrome 3x3 pattern.
    - Between the separators is a 3x4 or 4x3 table of the same 5x5 glyph cells, and below the second separator is a verdict band.

    Rule:
    - Match every query glyph to the unique table cell with the same color and 3x3 shape.
    - In every matched table cell, recolor the teal background pixels inside that 5x5 cell to green while leaving the glyph itself unchanged.
    - Treat the matched table-cell indices as nodes on the table grid, with edges only between horizontally or vertically adjacent matched cells.
    - If all matched nodes are 4-connected, recolor all teal pixels in the query band to green and fill the verdict band green; otherwise leave the query band unchanged and fill the verdict band red.

    Key insight:
    - The table lookup supplies abstract grid coordinates, and the final verdict depends on graph connectivity in table-cell space rather than on a visual line or on pixel-level connectivity.
    """
    info = _analyze_grid(grid)
    out = np.array(info["grid"], copy=True)

    for _ri, _ci, r, c, _sig in info["matches"]:
        cell = out[r : r + CELL, c : c + CELL]
        cell[cell == BG] = GREEN

    success = _is_connected(info["nodes"])
    if success:
        query_band = out[: info["top_sep"], :]
        query_band[query_band == BG] = GREEN
        out[info["bottom_sep"] + 1 :, :] = GREEN
    else:
        out[info["bottom_sep"] + 1 :, :] = RED
    return out


def _mask_is_connected(mask: int) -> bool:
    nodes = {(i // 3, i % 3) for i in range(9) if (mask >> i) & 1}
    return _is_connected(nodes)


def _mask_spans(mask: int) -> bool:
    coords = [(i // 3, i % 3) for i in range(9) if (mask >> i) & 1]
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    return len(rows) >= 2 and len(cols) >= 2


MASK_POOL = tuple(
    mask
    for mask in range(1 << 9)
    if 3 <= mask.bit_count() <= 6 and _mask_is_connected(mask) and _mask_spans(mask)
)


def _render_glyph(mask: int, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    bits = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    cell[INNER, INNER][bits] = color
    return cell


@lru_cache(maxsize=None)
def _candidate_sets(table_rows: int, table_cols: int, count: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    cells = [(r, c) for r in range(table_rows) for c in range(table_cols)]
    success: list[tuple[tuple[int, int], ...]] = []
    fail_diag: list[tuple[tuple[int, int], ...]] = []
    fail_split2: list[tuple[tuple[int, int], ...]] = []
    fail_split3: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(cells, count):
        nodes = set(combo)
        if _same_row_or_col(nodes):
            continue
        if _is_connected(nodes):
            success.append(combo)
            continue
        if _is_connected(nodes, eight=True):
            fail_diag.append(combo)
            continue
        comp_count = _component_count(nodes)
        if comp_count == 2:
            fail_split2.append(combo)
        else:
            fail_split3.append(combo)
    return tuple(success), tuple(fail_diag), tuple(fail_split2), tuple(fail_split3)


def _sample_selected_cells(table_rows: int, table_cols: int, count: int, success: bool) -> list[tuple[int, int]]:
    success_sets, fail_diag, fail_split2, fail_split3 = _candidate_sets(table_rows, table_cols, count)
    if success:
        return list(random.choice(success_sets))
    buckets = [bucket for bucket in (fail_diag, fail_split2, fail_split3) if bucket]
    weights = [3 if bucket is fail_diag else 2 if bucket is fail_split2 else 1 for bucket in buckets]
    bucket = random.choices(buckets, weights=weights, k=1)[0]
    return list(random.choice(bucket))


def _sample_colors(length: int) -> list[int]:
    return [random.choice(GLYPH_COLORS) for _ in range(length)]


def _assign_glyph_specs(
    all_cells: list[tuple[int, int]],
    selected_cells: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[str, tuple[int, int]]]:
    count = len(all_cells)
    masks = random.sample(MASK_POOL, count)
    cell_to_index = {cell: i for i, cell in enumerate(all_cells)}
    outside = [cell for cell in all_cells if cell not in set(selected_cells)]
    assert outside, "table must contain distractor glyphs"

    shape_query = random.choice(selected_cells)
    shape_other = random.choice(outside)
    masks[cell_to_index[shape_other]] = masks[cell_to_index[shape_query]]

    color_query_choices = [cell for cell in selected_cells if cell != shape_query] or list(selected_cells)
    color_query = random.choice(color_query_choices)
    color_other_choices = [
        cell
        for cell in outside
        if cell != shape_other and masks[cell_to_index[cell]] != masks[cell_to_index[color_query]]
    ]
    if not color_other_choices:
        return {}, {}
    color_other = random.choice(color_other_choices)

    colors = _sample_colors(count)
    if colors[cell_to_index[shape_other]] == colors[cell_to_index[shape_query]]:
        choices = [c for c in GLYPH_COLORS if c != colors[cell_to_index[shape_query]]]
        colors[cell_to_index[shape_other]] = random.choice(choices)
    colors[cell_to_index[color_other]] = colors[cell_to_index[color_query]]

    seen = set()
    glyphs: dict[tuple[int, int], tuple[int, int]] = {}
    for cell in all_cells:
        idx = cell_to_index[cell]
        spec = (masks[idx], colors[idx])
        if spec in seen:
            return {}, {}
        seen.add(spec)
        glyphs[cell] = spec

    return glyphs, {
        "shape_query": shape_query,
        "shape_other": shape_other,
        "color_query": color_query,
        "color_other": color_other,
    }


def _render_input(
    table_rows: int,
    table_cols: int,
    verdict_h: int,
    query_cells: list[tuple[int, int]],
    glyph_specs: dict[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, list[int], list[int]]:
    query_count = len(query_cells)
    max_slots = max(query_count, table_cols)
    width = 2 + max_slots * CELL
    height = CELL + 1 + table_rows * CELL + 1 + verdict_h

    slot_cols = [1 + i * CELL for i in range(max_slots)]
    table_offset = random.randint(0, max_slots - table_cols)
    table_cols_px = slot_cols[table_offset : table_offset + table_cols]
    table_rows_px = [CELL + 1 + i * CELL for i in range(table_rows)]

    query_slots = random.sample(range(max_slots), query_count)
    grid = np.full((height, width), BG, dtype=int)
    grid[CELL, :] = SEP
    bottom_sep = CELL + 1 + table_rows * CELL
    grid[bottom_sep, :] = SEP

    for (ri, ci), (mask, color) in glyph_specs.items():
        glyph = _render_glyph(mask, color)
        r = table_rows_px[ri]
        c = table_cols_px[ci]
        grid[r : r + CELL, c : c + CELL] = glyph

    shuffled_queries = query_cells[:]
    random.shuffle(shuffled_queries)
    for slot, cell in zip(query_slots, shuffled_queries):
        mask, color = glyph_specs[cell]
        glyph = _render_glyph(mask, color)
        c = slot_cols[slot]
        grid[:CELL, c : c + CELL] = glyph

    return grid, table_rows_px, table_cols_px


def _validate_generation_spec(
    table_rows: int,
    table_cols: int,
    selected_cells: list[tuple[int, int]],
    glyph_specs: dict[tuple[int, int], tuple[int, int]],
    witness: dict[str, tuple[int, int]],
    *,
    success: bool,
) -> bool:
    nodes = set(selected_cells)
    if (table_rows, table_cols) not in TABLE_SHAPES:
        return False
    if len(selected_cells) not in (3, 4):
        return False
    if _same_row_or_col(nodes):
        return False
    if _is_connected(nodes) != success:
        return False

    mask_map = {cell: glyph_specs[cell][0] for cell in glyph_specs}
    color_map = {cell: glyph_specs[cell][1] for cell in glyph_specs}
    shape_query = witness["shape_query"]
    shape_other = witness["shape_other"]
    color_query = witness["color_query"]
    color_other = witness["color_other"]
    if mask_map[shape_query] != mask_map[shape_other] or color_map[shape_query] == color_map[shape_other]:
        return False
    if color_map[color_query] != color_map[color_other] or mask_map[color_query] == mask_map[color_other]:
        return False
    return True


def _validate_rendered_instance(
    grid: np.ndarray,
    expected_nodes: set[tuple[int, int]],
    *,
    success: bool,
    expected_output: np.ndarray,
) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    if info["nodes"] != expected_nodes:
        return False
    if _same_row_or_col(info["nodes"]):
        return False
    if _is_connected(info["nodes"]) != success:
        return False

    query_sigs = [sig for *_rest, sig in info["matches"]]
    if len(query_sigs) != len(set(query_sigs)):
        return False

    has_shape_decoy = False
    has_color_decoy = False
    for _ri, _ci, _r, _c, sig in info["matches"]:
        color, shape = sig
        has_shape_decoy |= len(info["shape_colors"].get(shape, set())) >= 2
        has_color_decoy |= len(info["color_shapes"].get(color, set())) >= 2
    if not (has_shape_decoy and has_color_decoy):
        return False

    solved = main(grid)
    return bool(np.array_equal(solved, expected_output))


def _gen_attempt() -> np.ndarray | None:
    table_rows, table_cols = random.choice(TABLE_SHAPES)
    verdict_h = random.choice((2, 3))
    query_count = random.choice((3, 4))
    success = random.random() < 0.5

    selected_cells = _sample_selected_cells(table_rows, table_cols, query_count, success)
    all_cells = [(r, c) for r in range(table_rows) for c in range(table_cols)]
    glyph_specs, witness = _assign_glyph_specs(all_cells, selected_cells)
    if not glyph_specs:
        return None
    if not _validate_generation_spec(table_rows, table_cols, selected_cells, glyph_specs, witness, success=success):
        return None

    grid, table_rows_px, table_cols_px = _render_input(
        table_rows,
        table_cols,
        verdict_h,
        selected_cells,
        glyph_specs,
    )
    expected = _render_expected(
        grid,
        selected_cells,
        table_rows_px,
        table_cols_px,
        success=success,
        top_sep=CELL,
        bottom_sep=CELL + 1 + table_rows * CELL,
    )
    if not _validate_rendered_instance(grid, set(selected_cells), success=success, expected_output=expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid connectivity-lookup ARC input")
