from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
PALETTE = tuple(range(10))
QUERY_CORNERS = ("tl", "tr", "bl", "br")
TABLE_LAYOUTS = tuple(
    (body_rows, body_cols, cell_size)
    for body_rows in range(2, 5)
    for body_cols in range(2, 5)
    for cell_size in (3, 4)
    if ((body_rows + 1) * cell_size + (body_rows + 2)) + cell_size + 3 <= 30
    and ((body_cols + 1) * cell_size + (body_cols + 2)) + cell_size + 3 <= 30
)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _cell_signature(cell: np.ndarray, bg: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in cell.tolist())


def _non_bg_color(cell: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, np.unique(cell))) - {int(bg)})
    assert len(colors) == 1, f"expected one foreground color, got {colors}"
    return int(colors[0])


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _acceptable_masks(size: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    total = size * size
    min_area = size + 2
    for bits in range(1, 1 << total):
        if bits.bit_count() < min_area or bits.bit_count() >= total:
            continue
        flat = np.array([(bits >> i) & 1 for i in range(total)], dtype=bool)
        mask = flat.reshape(size, size)
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_connected(mask):
            continue
        masks.append(mask)
    random.Random(0).shuffle(masks)
    return masks


MASK_POOL = {size: _acceptable_masks(size) for size in (3, 4)}


def _find_table(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for color in _colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        full_rows = [r for r in range(sub.shape[0]) if np.all(sub[r, :] == color)]
        full_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == color)]
        if len(full_rows) < 4 or len(full_cols) < 4:
            continue
        row_steps = np.diff(full_rows)
        col_steps = np.diff(full_cols)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        step = int(row_steps[0])
        if step != int(col_steps[0]) or step < 2:
            continue
        expected_rows = list(range(0, sub.shape[0], step))
        expected_cols = list(range(0, sub.shape[1], step))
        if full_rows != expected_rows or full_cols != expected_cols:
            continue
        if full_rows[-1] != sub.shape[0] - 1 or full_cols[-1] != sub.shape[1] - 1:
            continue
        line_mask = np.zeros_like(sub, dtype=bool)
        line_mask[full_rows, :] = True
        line_mask[:, full_cols] = True
        if not np.array_equal(sub == color, line_mask):
            continue
        candidates.append(
            {
                "sep_color": int(color),
                "bbox": (r0, c0, r1, c1),
                "rows": full_rows,
                "cols": full_cols,
                "cell_size": step - 1,
            }
        )
    assert len(candidates) == 1, f"expected one separator table, got {len(candidates)}"
    return candidates[0]


def _extract_cell(grid: np.ndarray, table: dict, cell_r: int, cell_c: int) -> np.ndarray:
    r0, c0, _r1, _c1 = table["bbox"]
    rows = table["rows"]
    cols = table["cols"]
    return grid[
        r0 + rows[cell_r] + 1 : r0 + rows[cell_r + 1],
        c0 + cols[cell_c] + 1 : c0 + cols[cell_c + 1],
    ].copy()


def _parse_query_cell(grid: np.ndarray, table: dict, bg: int) -> np.ndarray:
    sep_color = int(table["sep_color"])
    r0, c0, r1, c1 = table["bbox"]
    outside = (grid != bg) & (grid != sep_color)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere(outside)
    assert coords.size, "missing query glyph"
    qr0, qc0, qr1, qc1 = bbox_coords(coords)
    cell_size = int(table["cell_size"])
    assert (qr1 - qr0 + 1, qc1 - qc0 + 1) == (cell_size, cell_size), "query bbox must match cell size"
    patch = grid[qr0 : qr1 + 1, qc0 : qc1 + 1].copy()
    assert np.array_equal((patch != bg), outside[qr0 : qr1 + 1, qc0 : qc1 + 1]), "query must be isolated"
    return patch


def _solve_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    table = _find_table(g, bg)
    total_rows = len(table["rows"]) - 1
    total_cols = len(table["cols"]) - 1
    assert total_rows >= 3 and total_cols >= 3, "need header row/col plus body"
    body_rows = total_rows - 1
    body_cols = total_cols - 1

    row_shapes = []
    for body_r in range(body_rows):
        cell = _extract_cell(g, table, body_r + 1, 0)
        assert _non_bg_color(cell, bg) == GREY
        row_shapes.append(_mask_signature(cell != bg))
    assert len(set(row_shapes)) == len(row_shapes), "row header shapes must be unique"

    col_colors = []
    for body_c in range(body_cols):
        cell = _extract_cell(g, table, 0, body_c + 1)
        assert np.all(cell != bg), "column header must be a solid color square"
        col_colors.append(_non_bg_color(cell, bg))
    assert len(set(col_colors)) == len(col_colors), "column header colors must be unique"

    query = _parse_query_cell(g, table, bg)
    query_shape = _mask_signature(query != bg)
    query_color = _non_bg_color(query, bg)

    row_matches = [i for i, sig in enumerate(row_shapes) if sig == query_shape]
    col_matches = [i for i, color in enumerate(col_colors) if color == query_color]
    assert len(row_matches) == 1, f"expected one row match, got {row_matches}"
    assert len(col_matches) == 1, f"expected one column match, got {col_matches}"
    target_row = int(row_matches[0])
    target_col = int(col_matches[0])

    body_signatures: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
    query_body_hits = []
    for r in range(body_rows):
        for c in range(body_cols):
            cell = _extract_cell(g, table, r + 1, c + 1)
            sig = _cell_signature(cell, bg)
            body_signatures[(r, c)] = sig
            if np.array_equal(cell, query):
                query_body_hits.append((r, c))

    return {
        "bg": int(bg),
        "table": table,
        "body_rows": body_rows,
        "body_cols": body_cols,
        "row_shapes": row_shapes,
        "col_colors": col_colors,
        "query": query,
        "query_shape": query_shape,
        "query_color": int(query_color),
        "target_row": target_row,
        "target_col": target_col,
        "answer": _extract_cell(g, table, target_row + 1, target_col + 1),
        "body_signatures": body_signatures,
        "query_body_hits": query_body_hits,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the query glyph's shape to pick a row and its color to pick a column, then output the intersecting table cell.

    Input structure:
    - A separator-colored grid divides the main table into equal square cells.
    - The top-left table cell is empty; the remaining top-row cells are solid color headers and the remaining left-column cells are grey shape headers.
    - Body cells contain colored glyphs, and one isolated query glyph sits outside the table in a corner region.

    Rule:
    - Decompose the query glyph into two attributes: its binary shape and its foreground color.
    - Match the query shape against the grey row headers to choose one body row.
    - Match the query color against the solid column headers to choose one body column.
    - Return the body cell at that row-column intersection, preserving its full cell-sized mini-grid exactly.

    Key insight:
    - The query glyph is a decoy exact body glyph somewhere else in the table, so whole-glyph matching is wrong; shape and color must be used on different lookup axes.
    """
    return _solve_instance(grid)["answer"]


def _choose_colors(body_cols: int) -> tuple[int, int, list[int], list[int]]:
    usable = [color for color in PALETTE if color != GREY]
    bg = random.choice(usable)
    usable.remove(bg)
    sep = random.choice(usable)
    usable.remove(sep)
    col_colors = random.sample(usable, body_cols)
    body_palette = list(col_colors)
    extras = [color for color in usable if color not in col_colors]
    random.shuffle(extras)
    need = max(3, min(len(usable), body_cols + 1))
    body_palette.extend(extras[: max(0, need - len(body_palette))])
    return int(bg), int(sep), [int(c) for c in col_colors], [int(c) for c in body_palette]


def _pick_distinct_masks(size: int, count: int) -> list[np.ndarray]:
    pool = MASK_POOL[size]
    assert len(pool) >= count
    chosen = random.sample(pool, count)
    return [mask.copy() for mask in chosen]


def _random_signature(
    shapes: list[np.ndarray],
    colors: list[int],
    used: set[tuple[tuple[tuple[int, ...], ...], int]],
    *,
    shape_forbidden: set[tuple[tuple[int, ...], ...]] | None = None,
    color_forbidden: set[int] | None = None,
) -> tuple[np.ndarray, int]:
    shape_forbidden = set() if shape_forbidden is None else set(shape_forbidden)
    color_forbidden = set() if color_forbidden is None else set(color_forbidden)
    options = []
    for shape in shapes:
        shape_sig = _mask_signature(shape)
        if shape_sig in shape_forbidden:
            continue
        for color in colors:
            if color in color_forbidden:
                continue
            key = (shape_sig, int(color))
            if key not in used:
                options.append((shape.copy(), int(color)))
    assert options, "no signature options left"
    return random.choice(options)


def _draw_mask(cell: np.ndarray, mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(cell.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _place_query_and_table(table_h: int, table_w: int, cell_size: int, corner: str) -> tuple[int, int, int, int, int, int]:
    margin = 1
    gap = 1
    if corner == "tl":
        query_r = margin
        query_c = margin
        table_r = margin + cell_size + gap
        table_c = margin + cell_size + gap
    elif corner == "tr":
        query_r = margin
        table_r = margin + cell_size + gap
        table_c = margin
        query_c = margin + table_w + gap
    elif corner == "bl":
        table_r = margin
        query_r = margin + table_h + gap
        query_c = margin
        table_c = margin + cell_size + gap
    else:
        table_r = margin
        table_c = margin
        query_r = margin + table_h + gap
        query_c = margin + table_w + gap
    height = max(query_r + cell_size, table_r + table_h) + margin
    width = max(query_c + cell_size, table_c + table_w) + margin
    return table_r, table_c, query_r, query_c, height, width


def _render_instance(spec: dict) -> np.ndarray:
    body_rows = spec["body_rows"]
    body_cols = spec["body_cols"]
    cell_size = spec["cell_size"]
    total_rows = body_rows + 1
    total_cols = body_cols + 1
    table_h = total_rows * cell_size + (total_rows + 1)
    table_w = total_cols * cell_size + (total_cols + 1)
    table_r, table_c, query_r, query_c, height, width = _place_query_and_table(
        table_h, table_w, cell_size, spec["corner"]
    )

    grid = np.full((height, width), spec["bg"], dtype=int)
    for rr in range(table_h):
        if rr % (cell_size + 1) == 0:
            grid[table_r + rr, table_c : table_c + table_w] = spec["sep"]
    for cc in range(table_w):
        if cc % (cell_size + 1) == 0:
            grid[table_r : table_r + table_h, table_c + cc] = spec["sep"]

    def cell_bounds(cell_r: int, cell_c: int) -> tuple[int, int]:
        top = table_r + 1 + cell_r * (cell_size + 1)
        left = table_c + 1 + cell_c * (cell_size + 1)
        return top, left

    for body_r, shape in enumerate(spec["row_shapes"]):
        top, left = cell_bounds(body_r + 1, 0)
        grid[top : top + cell_size, left : left + cell_size] = _draw_mask(
            np.empty((cell_size, cell_size), dtype=int), shape, GREY, spec["bg"]
        )

    for body_c, color in enumerate(spec["col_colors"]):
        top, left = cell_bounds(0, body_c + 1)
        grid[top : top + cell_size, left : left + cell_size] = color

    for (body_r, body_c), (shape, color) in spec["body_map"].items():
        top, left = cell_bounds(body_r + 1, body_c + 1)
        grid[top : top + cell_size, left : left + cell_size] = _draw_mask(
            np.empty((cell_size, cell_size), dtype=int), shape, color, spec["bg"]
        )

    query_patch = _draw_mask(np.empty((cell_size, cell_size), dtype=int), spec["query_shape"], spec["query_color"], spec["bg"])
    grid[query_r : query_r + cell_size, query_c : query_c + cell_size] = query_patch
    return grid


def _build_spec() -> dict:
    body_rows, body_cols, cell_size = random.choice(TABLE_LAYOUTS)
    bg, sep, col_colors, body_palette = _choose_colors(body_cols)

    row_shapes = _pick_distinct_masks(cell_size, body_rows)
    row_shape_sigs = [_mask_signature(mask) for mask in row_shapes]
    extra_shapes = [mask.copy() for mask in MASK_POOL[cell_size] if _mask_signature(mask) not in set(row_shape_sigs)]
    random.shuffle(extra_shapes)
    all_shapes = row_shapes + extra_shapes[: max(6, body_rows + body_cols)]

    target_row = random.randrange(body_rows)
    target_col = random.randrange(body_cols)
    query_shape = row_shapes[target_row].copy()
    query_color = int(col_colors[target_col])

    positions = [(r, c) for r in range(body_rows) for c in range(body_cols)]
    target_pos = (target_row, target_col)
    decoy_options = [(r, c) for r, c in positions if r != target_row and c != target_col]
    decoy_pos = random.choice(decoy_options)
    remaining = [pos for pos in positions if pos not in {target_pos, decoy_pos}]
    same_shape_pos = random.choice(remaining)
    remaining.remove(same_shape_pos)
    same_color_pos = random.choice(remaining)

    used: set[tuple[tuple[tuple[int, ...], ...], int]] = set()
    body_map: dict[tuple[int, int], tuple[np.ndarray, int]] = {}

    def assign(pos: tuple[int, int], shape: np.ndarray, color: int) -> None:
        shape_sig = _mask_signature(shape)
        key = (shape_sig, int(color))
        assert key not in used
        used.add(key)
        body_map[pos] = (shape.copy(), int(color))

    assign(decoy_pos, query_shape, query_color)

    other_color_choices = [color for color in body_palette if color != query_color]
    assign(same_shape_pos, query_shape, random.choice(other_color_choices))

    other_shape_choices = [shape for shape in all_shapes if _mask_signature(shape) != _mask_signature(query_shape)]
    assign(same_color_pos, random.choice(other_shape_choices), query_color)

    target_shape, target_color = _random_signature(
        all_shapes,
        body_palette,
        used,
        shape_forbidden={_mask_signature(query_shape)},
        color_forbidden={query_color},
    )
    assign(target_pos, target_shape, target_color)

    for pos in positions:
        if pos in body_map:
            continue
        shape, color = _random_signature(all_shapes, body_palette, used)
        assign(pos, shape, color)

    return {
        "body_rows": body_rows,
        "body_cols": body_cols,
        "cell_size": cell_size,
        "bg": bg,
        "sep": sep,
        "col_colors": col_colors,
        "row_shapes": row_shapes,
        "body_map": body_map,
        "query_shape": query_shape,
        "query_color": query_color,
        "target_pos": target_pos,
        "decoy_pos": decoy_pos,
        "same_shape_pos": same_shape_pos,
        "same_color_pos": same_color_pos,
        "corner": random.choice(QUERY_CORNERS),
    }


def _add_random_padding(grid: np.ndarray, bg: int) -> np.ndarray:
    slack_h = 30 - grid.shape[0]
    slack_w = 30 - grid.shape[1]
    pad_top = random.randint(0, min(2, slack_h))
    pad_bottom = random.randint(0, min(2, slack_h - pad_top))
    pad_left = random.randint(0, min(2, slack_w))
    pad_right = random.randint(0, min(2, slack_w - pad_left))
    return np.pad(grid, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=bg)


def _validate_semantics(grid: np.ndarray, spec: dict) -> None:
    solved = _solve_instance(grid)
    tr, tc = spec["target_pos"]
    assert solved["target_row"] == tr
    assert solved["target_col"] == tc
    expected = _draw_mask(
        np.empty((spec["cell_size"], spec["cell_size"]), dtype=int),
        spec["body_map"][spec["target_pos"]][0],
        spec["body_map"][spec["target_pos"]][1],
        spec["bg"],
    )
    assert np.array_equal(solved["answer"], expected)
    assert solved["query_body_hits"] == [spec["decoy_pos"]], "query glyph must reappear exactly once as a non-answer decoy"
    assert spec["decoy_pos"] != spec["target_pos"]

    query_shape_sig = _mask_signature(spec["query_shape"])
    shape_hits = []
    color_hits = []
    signatures = set()
    for pos, (shape, color) in spec["body_map"].items():
        sig = (_mask_signature(shape), int(color))
        assert sig not in signatures, "body cell contents must be unique"
        signatures.add(sig)
        if _mask_signature(shape) == query_shape_sig:
            shape_hits.append(pos)
        if int(color) == int(spec["query_color"]):
            color_hits.append(pos)
    assert len(shape_hits) >= 2, "query shape must appear in multiple body cells"
    assert len(color_hits) >= 2, "query color must appear in multiple body cells"
    assert spec["decoy_pos"] in shape_hits and spec["decoy_pos"] in color_hits
    assert spec["target_pos"] not in solved["query_body_hits"]

    body_rows = spec["body_rows"]
    body_cols = spec["body_cols"]
    for r in range(body_rows):
        row_sigs = {
            (_mask_signature(spec["body_map"][(r, c)][0]), spec["body_map"][(r, c)][1])
            for c in range(body_cols)
        }
        assert len(row_sigs) >= 2, "each body row must contain variation"
    for c in range(body_cols):
        col_sigs = {
            (_mask_signature(spec["body_map"][(r, c)][0]), spec["body_map"][(r, c)][1])
            for r in range(body_rows)
        }
        assert len(col_sigs) >= 2, "each body column must contain variation"

    assert get_bg(grid) == spec["bg"], "background must remain the dominant color"
    assert np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        spec = _build_spec()
        grid = _render_instance(spec)
        grid = _add_random_padding(grid, spec["bg"])
        if max(grid.shape) > 30:
            continue
        try:
            _validate_semantics(grid, spec)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid dual-attribute lookup table")
