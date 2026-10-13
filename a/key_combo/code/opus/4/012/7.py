from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
SEP = 6
GREEN = 3
YELLOW = 4

CELL = 3
GAP = 1
PITCH = CELL + GAP
TABLE_DIM = 3
TABLE_SPAN = TABLE_DIM * CELL + (TABLE_DIM - 1) * GAP

GLYPH_COLORS = (1, 2, 5, 7, 9)
TOP_BAND_H = 3
BOTTOM_BAND_RANGE = (6, 8)
WIDTH_RANGE = (16, 24)
MAX_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=12)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _shape_is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask.astype(bool))) == 1


def _full_bbox_shape(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _all_glyph_shapes() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    for bits in range(1 << (CELL * CELL)):
        mask = np.array([(bits >> i) & 1 for i in range(CELL * CELL)], dtype=bool).reshape(CELL, CELL)
        filled = int(mask.sum())
        if filled < 4 or filled > 7:
            continue
        if not _full_bbox_shape(mask):
            continue
        if not _shape_is_connected(mask):
            continue
        shapes.append(mask)
    return shapes


GLYPH_SHAPES = _all_glyph_shapes()


def _render_glyph(mask: np.ndarray, color: int) -> np.ndarray:
    cell = np.full((CELL, CELL), BG, dtype=int)
    cell[mask] = int(color)
    return cell


def _glyph_signature(cell: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(cell, dtype=int).ravel())


def _extract_glyphs(region: np.ndarray, bg: int, row_offset: int = 0, col_offset: int = 0) -> list[dict]:
    glyphs: list[dict] = []
    for coords in _components(region != bg):
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 + 1 == CELL and c1 - c0 + 1 == CELL, "glyphs must occupy 3x3 cells"
        crop = region[r0 : r1 + 1, c0 : c1 + 1].copy()
        glyphs.append(
            {
                "bbox": (r0 + row_offset, c0 + col_offset, r1 + row_offset, c1 + col_offset),
                "sig": _glyph_signature(crop),
            }
        )
    return glyphs


def _separator_rows(grid: np.ndarray, bg: int) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        if bool(np.all(grid[r, :] == grid[r, 0])) and int(grid[r, 0]) != int(bg):
            rows.append(int(r))
    return rows


def _parse_query_and_table(
    grid: np.ndarray,
    bg: int,
    top_sep: int,
    bottom_sep: int,
) -> tuple[list[dict], list[dict]]:
    query_glyphs = _extract_glyphs(grid[:top_sep, :], bg)
    query_glyphs.sort(key=lambda item: item["bbox"][1])

    table_region = grid[top_sep + 1 : bottom_sep, :]
    table_glyphs = _extract_glyphs(table_region, bg, row_offset=top_sep + 1)
    assert len(table_glyphs) == TABLE_DIM * TABLE_DIM, "reference table must contain 9 glyphs"

    row_starts = sorted({int(item["bbox"][0]) for item in table_glyphs})
    col_starts = sorted({int(item["bbox"][1]) for item in table_glyphs})
    assert len(row_starts) == TABLE_DIM and len(col_starts) == TABLE_DIM, "table must be 3x3"

    row_map = {r: i for i, r in enumerate(row_starts)}
    col_map = {c: i for i, c in enumerate(col_starts)}
    for item in table_glyphs:
        r0, c0, _r1, _c1 = item["bbox"]
        item["row"] = row_map[int(r0)]
        item["col"] = col_map[int(c0)]
    return query_glyphs, table_glyphs


def _find_segment(grid: np.ndarray, bottom_start: int) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    band = grid[bottom_start:, :]
    border = np.zeros_like(band, dtype=bool)
    border[0, :] = True
    border[:, 0] = True
    comps = _components((band == YELLOW) & border)
    assert len(comps) == 1, "expected one yellow boundary segment"
    coords = [(bottom_start + r, c) for r, c in comps[0]]
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    assert (r0 == r1) ^ (c0 == c1), "segment must be straight"
    assert 2 <= len(coords) <= 3, "segment length must be 2 or 3"
    return coords, box


def _paint_stripes(
    out: np.ndarray,
    bottom_start: int,
    segment_coords: list[tuple[int, int]],
    segment_box: tuple[int, int, int, int],
    horizontal: bool,
    bg: int,
) -> None:
    for r, c in segment_coords:
        out[r, c] = bg

    h, w = out.shape
    r0, c0, r1, c1 = segment_box
    cycle = (YELLOW, YELLOW, GREEN)

    if horizontal:
        span_rows = slice(r0, r1 + 1)
        step = 0
        pos = c0
        while pos < w:
            color = cycle[step % len(cycle)]
            if color == GREEN:
                out[bottom_start:, pos] = GREEN
            else:
                out[span_rows, pos] = YELLOW
            gap = pos + 1
            if gap < w:
                out[bottom_start:, gap] = bg
            step += 1
            pos += 2
    else:
        span_cols = slice(c0, c1 + 1)
        step = 0
        pos = r0
        while pos < h:
            color = cycle[step % len(cycle)]
            if color == GREEN:
                out[pos, :] = GREEN
            else:
                out[pos, span_cols] = YELLOW
            gap = pos + 1
            if gap < h:
                out[gap, :] = bg
            step += 1
            pos += 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the query glyphs against the reference table, then use the match line type to choose the bottom stripe direction.

    Input structure:
    - Teal background with two full-width pink separator rows that split the grid into query, table, and bottom bands.
    - The query band contains several separated 3x3 glyph cells; the middle band contains a 3x3 matrix of 3x3 glyph cells.
    - A short yellow border segment appears on either the left edge or the top edge of the bottom band.

    Rule:
    - Match each query glyph to table glyphs by exact 3x3 contents, including both color and shape.
    - Green-fill the teal background inside every matched table cell and green-fill the teal background of the whole query band.
    - If the matched table cells form one complete table row, propagate yellow-yellow-green stripes to the right across the bottom band; if they form one complete column, propagate the same cycle downward.
    - Yellow stripes stay restricted to the border segment's span on the perpendicular axis, while green stripes cover the full bottom-band column or row.

    Key insight:
    - The top lookup does more than confirm a match: the geometry of the matched table cells encodes the propagation direction for the bottom-band stripe process.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    sep_rows = _separator_rows(g, bg)
    assert len(sep_rows) >= 2, "expected two separator rows"
    top_sep = int(sep_rows[0])
    bottom_sep = int(sep_rows[-1])

    query_glyphs, table_glyphs = _parse_query_and_table(g, bg, top_sep, bottom_sep)
    query_sigs = [glyph["sig"] for glyph in query_glyphs]
    query_set = set(query_sigs)
    assert len(query_sigs) == len(query_set) == TABLE_DIM, "queries must be three distinct glyphs"

    out = g.copy()
    matched: list[tuple[int, int]] = []
    for item in table_glyphs:
        if item["sig"] not in query_set:
            continue
        matched.append((int(item["row"]), int(item["col"])))
        r0, c0, r1, c1 = item["bbox"]
        cell = out[r0 : r1 + 1, c0 : c1 + 1]
        cell[cell == bg] = GREEN

    row_hits = {}
    col_hits = {}
    for row, col in matched:
        row_hits.setdefault(row, set()).add(col)
        col_hits.setdefault(col, set()).add(row)

    full_rows = [row for row, cols in row_hits.items() if len(cols) == TABLE_DIM]
    full_cols = [col for col, rows in col_hits.items() if len(rows) == TABLE_DIM]
    assert (len(full_rows) == 1) ^ (len(full_cols) == 1), "matches must form exactly one full row or one full column"
    horizontal = len(full_rows) == 1

    query_band = out[:top_sep, :]
    query_band[query_band == bg] = GREEN

    bottom_start = bottom_sep + 1
    segment_coords, segment_box = _find_segment(g, bottom_start)
    _paint_stripes(out, bottom_start, segment_coords, segment_box, horizontal, bg)
    return out


def _sample_table_cells() -> list[list[np.ndarray]]:
    masks = random.sample(GLYPH_SHAPES, TABLE_DIM * TABLE_DIM)
    colors = list(random.sample(GLYPH_COLORS, len(GLYPH_COLORS)))
    colors.extend(random.choice(GLYPH_COLORS) for _ in range(TABLE_DIM * TABLE_DIM - len(GLYPH_COLORS)))
    random.shuffle(colors)

    cells: list[list[np.ndarray]] = []
    idx = 0
    for _r in range(TABLE_DIM):
        row: list[np.ndarray] = []
        for _c in range(TABLE_DIM):
            row.append(_render_glyph(masks[idx], colors[idx]))
            idx += 1
        cells.append(row)
    return cells


def _highlight_expected(
    grid: np.ndarray,
    top_sep: int,
    bottom_sep: int,
    table_c0: int,
    selected_cells: list[tuple[int, int]],
    segment_coords: list[tuple[int, int]],
    horizontal: bool,
) -> np.ndarray:
    out = grid.copy()
    for row, col in selected_cells:
        r0 = top_sep + 1 + row * PITCH
        c0 = table_c0 + col * PITCH
        cell = out[r0 : r0 + CELL, c0 : c0 + CELL]
        cell[cell == BG] = GREEN

    query_band = out[:top_sep, :]
    query_band[query_band == BG] = GREEN

    segment_box = _bbox(segment_coords)
    _paint_stripes(out, bottom_sep + 1, segment_coords, segment_box, horizontal, BG)
    return out


def _render_input() -> tuple[np.ndarray, np.ndarray]:
    bottom_h = random.randint(*BOTTOM_BAND_RANGE)
    height = TOP_BAND_H + 1 + TABLE_SPAN + 1 + bottom_h
    width = random.randint(*WIDTH_RANGE)

    grid = np.full((height, width), BG, dtype=int)
    top_sep = TOP_BAND_H
    bottom_sep = top_sep + 1 + TABLE_SPAN
    grid[top_sep, :] = SEP
    grid[bottom_sep, :] = SEP

    query_c0 = random.randint(0, width - TABLE_SPAN)
    table_c0 = random.randint(0, width - TABLE_SPAN)
    table_cells = _sample_table_cells()

    for row in range(TABLE_DIM):
        for col in range(TABLE_DIM):
            r0 = top_sep + 1 + row * PITCH
            c0 = table_c0 + col * PITCH
            grid[r0 : r0 + CELL, c0 : c0 + CELL] = table_cells[row][col]

    horizontal = random.random() < 0.5
    if horizontal:
        sel_row = random.randrange(TABLE_DIM)
        selected_cells = [(sel_row, col) for col in range(TABLE_DIM)]
    else:
        sel_col = random.randrange(TABLE_DIM)
        selected_cells = [(row, sel_col) for row in range(TABLE_DIM)]

    query_cells = selected_cells[:]
    random.shuffle(query_cells)
    for idx, (row, col) in enumerate(query_cells):
        c0 = query_c0 + idx * PITCH
        grid[:CELL, c0 : c0 + CELL] = table_cells[row][col]

    bottom_start = bottom_sep + 1
    segment_len = random.randint(2, 3)
    segment_coords: list[tuple[int, int]] = []
    segment_kind = random.choice(("left", "top"))

    if horizontal and segment_kind == "top":
        max_start = width - 2 * TABLE_DIM
        c0 = random.randint(1, max(1, max_start))
        for dc in range(segment_len):
            segment_coords.append((bottom_start, c0 + dc))
    elif not horizontal and segment_kind == "left":
        max_offset = max(1, bottom_h - 5)
        r0 = bottom_start + random.randint(1, max_offset)
        for dr in range(segment_len):
            segment_coords.append((r0 + dr, 0))
    elif segment_kind == "top":
        c0 = random.randint(1, width - segment_len)
        for dc in range(segment_len):
            segment_coords.append((bottom_start, c0 + dc))
    else:
        r0 = bottom_start + random.randint(1, bottom_h - segment_len)
        for dr in range(segment_len):
            segment_coords.append((r0 + dr, 0))

    for r, c in segment_coords:
        grid[r, c] = YELLOW

    expected = _highlight_expected(grid, top_sep, bottom_sep, table_c0, selected_cells, segment_coords, horizontal)
    return grid, expected


def _validate_semantics(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg = _dominant_color(grid)
        sep_rows = _separator_rows(grid, bg)
        if len(sep_rows) != 2:
            return False
        top_sep, bottom_sep = sep_rows
        if top_sep != TOP_BAND_H:
            return False

        query_glyphs, table_glyphs = _parse_query_and_table(grid, bg, top_sep, bottom_sep)
        if len(query_glyphs) != TABLE_DIM or len(table_glyphs) != TABLE_DIM * TABLE_DIM:
            return False

        table_sigs = [item["sig"] for item in table_glyphs]
        if len(set(table_sigs)) != TABLE_DIM * TABLE_DIM:
            return False

        query_sigs = {item["sig"] for item in query_glyphs}
        matched = [(int(item["row"]), int(item["col"])) for item in table_glyphs if item["sig"] in query_sigs]
        if len(matched) != TABLE_DIM:
            return False

        rows = {row for row, _col in matched}
        cols = {col for _row, col in matched}
        if not ((len(rows) == 1 and len(cols) == TABLE_DIM) ^ (len(cols) == 1 and len(rows) == TABLE_DIM)):
            return False

        segment_coords, segment_box = _find_segment(grid, bottom_sep + 1)
        if len(segment_coords) != 2 and len(segment_coords) != 3:
            return False
        r0, c0, r1, c1 = segment_box
        if not ((r0 == r1 and r0 == bottom_sep + 1) or (c0 == c1 and c0 == 0)):
            return False

        observed = main(grid)
        return bool(np.array_equal(observed, expected))
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid, expected = _render_input()
        if _validate_semantics(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
