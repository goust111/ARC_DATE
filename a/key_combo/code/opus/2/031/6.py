from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLOR_PAIR_POOL = [
    (2, 1),
    (3, 4),
    (7, 9),
    (8, 6),
    (5, 2),
]
PATTERN_LIBRARY = [
    ("H", "V"),
    ("H", "H"),
    ("V", "V"),
    ("H", "V", "H"),
    ("V", "H", "V"),
    ("H", "H", "V"),
]
MAX_GAP = 12
GENERATION_RETRIES = 300
_GEN_CALL_COUNT = 0


def _bridge_params(lo: int, hi: int) -> tuple[int, int, int, int, int, int] | None:
    gap = int(hi - lo - 1)
    if gap < 4 or gap % 2 != 0:
        return None
    slid_lo = int(lo + 1)
    slid_hi = int(hi - 1)
    mid = int((slid_lo + slid_hi) // 2)
    width = int(gap // 2)
    rect_lo = int(mid - (width - 1) // 2)
    rect_hi = int(rect_lo + width - 1)
    return gap, slid_lo, slid_hi, mid, rect_lo, rect_hi


def _paint_cell(out: np.ndarray, r: int, c: int, color: int) -> None:
    if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        current = int(out[r, c])
        if current == BACKGROUND or current == int(color):
            out[r, c] = int(color)


def _draw_horizontal_bridge(
    out: np.ndarray,
    row: int,
    left_col: int,
    right_col: int,
    left_color: int,
    right_color: int,
) -> None:
    params = _bridge_params(left_col, right_col)
    if params is None:
        return

    _gap, slid_left, slid_right, mid, rect_left, rect_right = params

    for c in range(slid_left, rect_left + 1):
        _paint_cell(out, row, c, left_color)
    for c in range(rect_right, slid_right + 1):
        _paint_cell(out, row, c, right_color)

    for c in range(rect_left, rect_right + 1):
        color = left_color if c <= mid else right_color
        _paint_cell(out, row - 1, c, color)
        _paint_cell(out, row + 1, c, color)

    _paint_cell(out, row, rect_left, left_color)
    _paint_cell(out, row, rect_right, right_color)


def _draw_vertical_bridge(
    out: np.ndarray,
    top_row: int,
    bottom_row: int,
    col: int,
    top_color: int,
    bottom_color: int,
) -> None:
    params = _bridge_params(top_row, bottom_row)
    if params is None:
        return

    _gap, slid_top, slid_bottom, mid, rect_top, rect_bottom = params

    for r in range(slid_top, rect_top + 1):
        _paint_cell(out, r, col, top_color)
    for r in range(rect_bottom, slid_bottom + 1):
        _paint_cell(out, r, col, bottom_color)

    for r in range(rect_top, rect_bottom + 1):
        color = top_color if r <= mid else bottom_color
        _paint_cell(out, r, col - 1, color)
        _paint_cell(out, r, col + 1, color)

    _paint_cell(out, rect_top, col, top_color)
    _paint_cell(out, rect_bottom, col, bottom_color)


def _extract_pairs(grid: np.ndarray) -> list[tuple[str, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    pairs: list[tuple[str, int, int, int, int, int]] = []
    height, width = g.shape

    for r in range(height):
        cols = np.flatnonzero(g[r] != BACKGROUND)
        if len(cols) == 2:
            c0, c1 = (int(cols[0]), int(cols[1]))
            pairs.append(("H", int(r), c0, c1, int(g[r, c0]), int(g[r, c1])))

    for c in range(width):
        rows = np.flatnonzero(g[:, c] != BACKGROUND)
        if len(rows) == 2:
            r0, r1 = (int(rows[0]), int(rows[1]))
            pairs.append(("V", r0, r1, int(c), int(g[r0, c]), int(g[r1, c])))

    pairs.sort(key=lambda item: (item[1], item[0], item[2]))
    return pairs


def _paired_foreground_cells(pairs: list[tuple[str, int, int, int, int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for orientation, a0, a1, b, _c0, _c1 in pairs:
        if orientation == "H":
            cells.add((a0, a1))
            cells.add((a0, b))
        else:
            cells.add((a0, b))
            cells.add((a1, b))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each aligned pixel pair inward and bridge it with a gap-scaled split-color hollow rectangle.

    Input structure:
    - Black background with 2-3 isolated colored pixel pairs.
    - Each pair is either horizontal or vertical and is the only pair using its rows and columns.
    - Inside each pair, the two colors differ and the number of background cells between them is an even value of at least 4.

    Rule:
    - Find every row or column containing exactly two non-background pixels; that aligned duo is one pair.
    - Treat vertical pairs in the same canonical way as horizontal ones, with the upper pixel playing the role of the left pixel.
    - Move both pixels one step toward each other, compute the midpoint after sliding, build a 3-thick hollow bridge whose span is half of the original gap, color its left/top half with the first pixel color and its right/bottom half with the second color, then connect the slid pixels to the bridge edges.

    Key insight:
    - The original gap is the shared parameter: it determines both where the post-slide midpoint lands and how wide the hollow bridge must be, so the bridge size cannot be fixed.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    pairs = _extract_pairs(g)
    if not pairs:
        return g.copy()

    foreground = {tuple(map(int, rc)) for rc in np.argwhere(g != BACKGROUND)}
    if _paired_foreground_cells(pairs) != foreground:
        return g.copy()

    out = np.zeros_like(g)
    for orientation, a0, a1, b, color0, color1 in pairs:
        if orientation == "H":
            row = a0
            left_col = min(a1, b)
            right_col = max(a1, b)
            left_color = int(g[row, left_col])
            right_color = int(g[row, right_col])
            _draw_horizontal_bridge(out, row, left_col, right_col, left_color, right_color)
        else:
            top_row = min(a0, a1)
            bottom_row = max(a0, a1)
            col = b
            top_color = int(g[top_row, col])
            bottom_color = int(g[bottom_row, col])
            _draw_vertical_bridge(out, top_row, bottom_row, col, top_color, bottom_color)
    return out


def _next_pattern() -> tuple[str, ...]:
    global _GEN_CALL_COUNT
    pattern = PATTERN_LIBRARY[_GEN_CALL_COUNT % len(PATTERN_LIBRARY)]
    _GEN_CALL_COUNT += 1
    return pattern


def _sample_color_pairs(n_pairs: int) -> list[tuple[int, int]]:
    color_pairs = COLOR_PAIR_POOL.copy()
    random.shuffle(color_pairs)
    chosen = color_pairs[:n_pairs]
    return [pair if random.random() < 0.5 else (pair[1], pair[0]) for pair in chosen]


def _horizontal_candidates(
    height: int,
    width: int,
    reserved: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    max_gap = min(MAX_GAP, width - 2)
    for row in range(1, height - 1):
        if row in used_rows:
            continue
        for gap in range(4, max_gap + 1, 2):
            span = gap + 2
            for left_col in range(0, width - span + 1):
                right_col = left_col + gap + 1
                if left_col in used_cols or right_col in used_cols:
                    continue
                if reserved[row - 1 : row + 2, left_col : right_col + 1].any():
                    continue
                candidates.append((row, left_col, right_col))
    return candidates


def _vertical_candidates(
    height: int,
    width: int,
    reserved: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    max_gap = min(MAX_GAP, height - 2)
    for col in range(1, width - 1):
        if col in used_cols:
            continue
        for gap in range(4, max_gap + 1, 2):
            span = gap + 2
            for top_row in range(0, height - span + 1):
                bottom_row = top_row + gap + 1
                if top_row in used_rows or bottom_row in used_rows:
                    continue
                if reserved[top_row : bottom_row + 1, col - 1 : col + 2].any():
                    continue
                candidates.append((top_row, bottom_row, col))
    return candidates


def _gen_attempt(pattern: tuple[str, ...]) -> np.ndarray | None:
    pair_count = len(pattern)
    min_size = 16 if pair_count == 2 else 18
    height = random.randint(min_size, 26)
    width = random.randint(min_size, 26)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros_like(grid, dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    color_pairs = _sample_color_pairs(pair_count)

    for orientation, (color0, color1) in zip(pattern, color_pairs):
        if orientation == "H":
            candidates = _horizontal_candidates(height, width, reserved, used_rows, used_cols)
            if not candidates:
                return None
            row, left_col, right_col = random.choice(candidates)
            grid[row, left_col] = color0
            grid[row, right_col] = color1
            reserved[row - 1 : row + 2, left_col : right_col + 1] = True
            used_rows.add(row)
            used_cols.update((left_col, right_col))
        else:
            candidates = _vertical_candidates(height, width, reserved, used_rows, used_cols)
            if not candidates:
                return None
            top_row, bottom_row, col = random.choice(candidates)
            grid[top_row, col] = color0
            grid[bottom_row, col] = color1
            reserved[top_row : bottom_row + 1, col - 1 : col + 2] = True
            used_rows.update((top_row, bottom_row))
            used_cols.add(col)

    pairs = _extract_pairs(grid)
    if len(pairs) != pair_count:
        return None
    if len(_paired_foreground_cells(pairs)) != pair_count * 2:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    pattern = _next_pattern()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(pattern)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-grid input")
