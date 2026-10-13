from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_palette_chain(
    grid: np.ndarray, bg: int, *, restrict_to_corner: bool
) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    by_col: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)

    for comp in _components4(grid != bg):
        if len(comp) != 2:
            continue
        pts = sorted(comp)
        (r0, c0), (r1, c1) = pts
        if r0 != r1 or c1 != c0 + 1:
            continue
        if restrict_to_corner:
            if r0 < h // 2 or c0 > max(1, w // 4):
                continue
        by_col[c0][r0] = (int(grid[r0, c0]), int(grid[r0, c1]))

    best_rows: list[int] | None = None
    best_col: int | None = None
    best_score: tuple[int, int, int] | None = None
    for c0, row_map in by_col.items():
        rows = sorted(row_map)
        for row in rows:
            if row - 2 in row_map:
                continue
            chain = [row]
            cur = row
            while cur + 2 in row_map:
                cur += 2
                chain.append(cur)
            score = (len(chain), -c0, chain[-1])
            if best_score is None or score > best_score:
                best_score = score
                best_rows = chain
                best_col = c0

    if best_rows is None or best_col is None:
        return {}, np.zeros_like(grid, dtype=bool)

    mapping: dict[int, int] = {}
    mask = np.zeros_like(grid, dtype=bool)
    for row in best_rows:
        dash, fill = by_col[best_col][row]
        mapping[dash] = fill
        mask[row, best_col] = True
        mask[row, best_col + 1] = True
    return mapping, mask


def _parse_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    mapping, mask = _find_palette_chain(grid, bg, restrict_to_corner=True)
    if mapping:
        return mapping, mask
    return _find_palette_chain(grid, bg, restrict_to_corner=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify dashed rectangles, recover their bounding boxes, and fill each interior
    using the bottom-left palette's dash-color -> fill-color mapping.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    mapping, palette_mask = _parse_palette(g, bg)
    assert mapping, "expected a non-empty palette in the lower-left corner"

    out = g.copy()
    for dash_color, fill_color in mapping.items():
        rect_mask = (g == dash_color) & ~palette_mask
        if not rect_mask.any():
            continue
        rows, cols = np.where(rect_mask)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
RECT_COUNT_RANGE = (1, 4)
RECT_H_RANGE = (6, 12)
RECT_W_RANGE = (6, 14)
RECT_MARGIN = 2
LEFT_RESERVED_COLS = 4
PLACEMENT_TRIES = 80
GEN_RETRIES = 400


def _dash_pattern(span: int, rng: random.Random) -> list[tuple[int, int]]:
    need = 2 if span >= 6 else 1
    best: list[tuple[int, int]] = []
    for _ in range(40):
        seg = rng.randint(2, min(4, span))
        gap = rng.randint(1, 2)
        offset = rng.randint(0, gap)
        intervals: list[tuple[int, int]] = []
        pos = offset
        while pos < span:
            end = min(span, pos + seg)
            if end - pos >= 2:
                intervals.append((pos, end))
            pos += seg + gap
        if len(intervals) >= need:
            return intervals
        if len(intervals) > len(best):
            best = intervals
    return best or [(0, min(2, span))]


def _draw_dashed_rectangle(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int, rng: random.Random
) -> None:
    horiz = _dash_pattern(c1 - c0 + 1, rng)
    vert = _dash_pattern(r1 - r0 + 1, rng)
    for a, b in horiz:
        grid[r0, c0 + a : c0 + b] = color
        grid[r1, c0 + a : c0 + b] = color
    for a, b in vert:
        grid[r0 + a : r0 + b, c0] = color
        grid[r0 + a : r0 + b, c1] = color


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - RECT_MARGIN)
    cc0 = max(0, c0 - RECT_MARGIN)
    rr1 = min(h, r1 + RECT_MARGIN + 1)
    cc1 = min(w, c1 + RECT_MARGIN + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - RECT_MARGIN)
    cc0 = max(0, c0 - RECT_MARGIN)
    rr1 = min(h, r1 + RECT_MARGIN + 1)
    cc1 = min(w, c1 + RECT_MARGIN + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _expected_output(
    grid: np.ndarray, rects: list[tuple[int, int, int, int, int]]
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1, fill_color in rects:
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(10)

    max_rectangles = min(RECT_COUNT_RANGE[1], max(1, (h * w) // 110))
    rect_count = rng.randint(RECT_COUNT_RANGE[0], max_rectangles)
    palette_height = 2 * rect_count - 1
    rect_row_limit = h - palette_height - 2
    if rect_row_limit < RECT_H_RANGE[0]:
        return None

    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    dash_colors = colors[:rect_count]
    fill_colors = colors[rect_count : 2 * rect_count]
    if len(fill_colors) < rect_count:
        return None

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, :LEFT_RESERVED_COLS] = True
    occupied[h - palette_height - 1 :, :] = True

    rects: list[tuple[int, int, int, int, int]] = []
    order = list(range(rect_count))
    rng.shuffle(order)
    for idx in order:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            rect_h = rng.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], rect_row_limit) + 1)
            rect_w = rng.randint(RECT_W_RANGE[0], min(RECT_W_RANGE[1], w - LEFT_RESERVED_COLS - 2) + 1)
            r0 = rng.randint(0, rect_row_limit - rect_h + 1)
            c0 = rng.randint(LEFT_RESERVED_COLS, w - rect_w - 1)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            if not _can_place(occupied, r0, c0, r1, c1):
                continue
            _mark_occupied(occupied, r0, c0, r1, c1)
            rects.append((r0, c0, r1, c1, fill_colors[idx]))
            _draw_dashed_rectangle(grid, r0, c0, r1, c1, dash_colors[idx], rng)
            placed = True
            break
        if not placed:
            return None

    mapping_rows = list(zip(dash_colors, fill_colors))
    rng.shuffle(mapping_rows)
    palette_top = h - palette_height
    for i, (dash_color, fill_color) in enumerate(mapping_rows):
        row = palette_top + 2 * i
        grid[row, 0] = dash_color
        grid[row, 1] = fill_color

    expected = _expected_output(grid, rects)
    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid dashed-rectangle puzzle")
