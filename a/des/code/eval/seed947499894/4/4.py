from __future__ import annotations

import random

import numpy as np


def _collect_indices(a: np.ndarray, b: np.ndarray) -> list[int]:
    """Collect marked inner indices from a pair of opposite border sides."""
    marked = np.flatnonzero((np.asarray(a) != 0) | (np.asarray(b) != 0))
    return [int(x) for x in marked.tolist()]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge markers as row/column selectors, then extract the corresponding
    submatrix from the interior.

    Input structure:
    - The grid consists of an inner square region plus a 1-cell border.
    - Non-black pixels on the top/bottom border mark selected columns.
    - Non-black pixels on the left/right border mark selected rows.

    Rule:
    - Read selected columns from left to right.
    - Read selected rows from top to bottom.
    - Output the pixels at the intersections of those selected rows and columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 3 and g.shape[1] >= 3

    inner = g[1:-1, 1:-1]
    rows = _collect_indices(g[1:-1, 0], g[1:-1, -1])
    cols = _collect_indices(g[0, 1:-1], g[-1, 1:-1])

    assert rows, "expected at least one marked row"
    assert cols, "expected at least one marked column"
    return inner[np.ix_(rows, cols)]


# --------------- generator ---------------

BG = 0
COLORS = tuple(range(1, 10))

INNER_SIZE_RANGE = (6, 16)
SELECT_COUNT_RANGE = (2, 6)
DECORATION_COUNT_RANGE = (3, 8)
GEN_RETRY_LIMIT = 300
SHAPE_RETRY_LIMIT = 25
PATTERN_RETRY_LIMIT = 80


def _rand_count(limit: int) -> int:
    upper = min(SELECT_COUNT_RANGE[1], limit)
    lower = min(SELECT_COUNT_RANGE[0], upper)
    return random.randint(lower, upper)


def _paint_cells(canvas: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    if not cells:
        return False
    h, w = canvas.shape
    unique = set(cells)
    for r, c in unique:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(canvas[r, c]) != BG:
            return False
    for r, c in unique:
        canvas[r, c] = int(color)
    return True


def _shape_cells(kind: str, size: int) -> list[tuple[int, int]]:
    if kind == "rect":
        h = random.randint(1, max(1, size // 3))
        w = random.randint(1, max(1, size // 3))
        r0 = random.randint(0, size - h)
        c0 = random.randint(0, size - w)
        return [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]

    if kind == "hline":
        length = random.randint(2, max(2, size // 2))
        r = random.randint(0, size - 1)
        c0 = random.randint(0, size - length)
        return [(r, c) for c in range(c0, c0 + length)]

    if kind == "vline":
        length = random.randint(2, max(2, size // 2))
        c = random.randint(0, size - 1)
        r0 = random.randint(0, size - length)
        return [(r, c) for r in range(r0, r0 + length)]

    if kind == "diag":
        length = random.randint(2, max(2, size // 2))
        dr, dc = random.choice(((1, 1), (1, -1)))
        if dc == 1:
            r0 = random.randint(0, size - length)
            c0 = random.randint(0, size - length)
        else:
            r0 = random.randint(0, size - length)
            c0 = random.randint(length - 1, size - 1)
        return [(r0 + i * dr, c0 + i * dc) for i in range(length)]

    if kind == "plus":
        arm = random.randint(1, max(1, size // 5))
        r = random.randint(arm, size - arm - 1)
        c = random.randint(arm, size - arm - 1)
        cells = {(r, c)}
        for d in range(1, arm + 1):
            cells.add((r - d, c))
            cells.add((r + d, c))
            cells.add((r, c - d))
            cells.add((r, c + d))
        return sorted(cells)

    raise ValueError(kind)


def _add_decorations(inner: np.ndarray) -> None:
    size = inner.shape[0]
    shape_types = ("rect", "hline", "vline", "diag", "plus")
    target = random.randint(*DECORATION_COUNT_RANGE)
    placed = 0

    while placed < target:
        kind = random.choice(shape_types)
        for _ in range(SHAPE_RETRY_LIMIT):
            cells = _shape_cells(kind, size)
            color = random.choice(COLORS)
            if _paint_cells(inner, cells, color):
                placed += 1
                break
        else:
            break


def _build_pattern(rows: int, cols: int) -> np.ndarray:
    for _ in range(PATTERN_RETRY_LIMIT):
        out = np.zeros((rows, cols), dtype=int)
        palette = random.sample(COLORS, k=random.randint(2, 4))
        mode = random.choice(("checker", "bars", "frame", "diagonal", "random"))

        if mode == "checker":
            a, b = palette[:2]
            for r in range(rows):
                for c in range(cols):
                    out[r, c] = a if (r + c) % 2 == 0 else b
            mask = np.random.random((rows, cols)) < 0.2
            out[mask] = BG

        elif mode == "bars":
            if random.random() < 0.5:
                for r in range(rows):
                    out[r, :] = palette[r % len(palette)]
                gap_col = random.randrange(cols)
                out[:, gap_col] = BG
            else:
                for c in range(cols):
                    out[:, c] = palette[c % len(palette)]
                gap_row = random.randrange(rows)
                out[gap_row, :] = BG

        elif mode == "frame" and rows >= 3 and cols >= 3:
            a, b = palette[:2]
            out[0, :] = a
            out[-1, :] = a
            out[:, 0] = a
            out[:, -1] = a
            out[1:-1, 1:-1] = b
            holes = np.random.random((rows - 2, cols - 2)) < 0.35
            core = out[1:-1, 1:-1]
            core[holes] = BG

        elif mode == "diagonal":
            a, b = palette[:2]
            for r in range(rows):
                for c in range(cols):
                    if r == c:
                        out[r, c] = a
                    elif r + c == cols - 1:
                        out[r, c] = b
            if len(palette) >= 3 and rows > 2 and cols > 2:
                out[rows // 2, :] = palette[2]

        else:
            choices = [BG] + palette
            weights = [0.35] + [0.65 / len(palette)] * len(palette)
            draw = random.choices(choices, weights=weights, k=rows * cols)
            out = np.array(draw, dtype=int).reshape(rows, cols)

        nonzero = int(np.count_nonzero(out))
        unique_nonzero = len(set(map(int, out[out != 0].tolist())))
        if nonzero >= max(2, (rows * cols) // 4) and unique_nonzero >= 2:
            return out

    fallback = np.zeros((rows, cols), dtype=int)
    a, b = random.sample(COLORS, 2)
    fallback[:, 0] = a
    fallback[0, :] = b
    return fallback


def _stamp_markers(grid: np.ndarray, indices: list[int], axis: str, colors: list[int]) -> None:
    if axis == "col":
        for idx, color in zip(indices, colors, strict=False):
            grid[0, idx + 1] = int(color)
            grid[-1, idx + 1] = int(color)
    elif axis == "row":
        for idx, color in zip(indices, colors, strict=False):
            grid[idx + 1, 0] = int(color)
            grid[idx + 1, -1] = int(color)
    else:
        raise ValueError(axis)


def _attempt_generate() -> np.ndarray | None:
    size = random.randint(*INNER_SIZE_RANGE)
    row_count = _rand_count(size)
    col_count = _rand_count(size)

    rows = sorted(random.sample(range(size), row_count))
    cols = sorted(random.sample(range(size), col_count))

    inner = np.zeros((size, size), dtype=int)
    _add_decorations(inner)

    sampled = _build_pattern(row_count, col_count)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            inner[r, c] = int(sampled[i, j])

    grid = np.zeros((size + 2, size + 2), dtype=int)
    grid[1:-1, 1:-1] = inner

    top_colors = random.sample(COLORS, col_count)
    left_colors = random.sample(COLORS, row_count)
    _stamp_markers(grid, cols, "col", top_colors)
    _stamp_markers(grid, rows, "row", left_colors)

    if random.random() < 0.5:
        bottom_colors = random.sample(COLORS, col_count)
        for idx, color in zip(cols, bottom_colors, strict=False):
            grid[-1, idx + 1] = int(color)
    if random.random() < 0.5:
        right_colors = random.sample(COLORS, row_count)
        for idx, color in zip(rows, right_colors, strict=False):
            grid[idx + 1, -1] = int(color)

    expected = sampled
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.count_nonzero(actual) < max(2, (actual.size // 4)):
        return None
    if len(set(map(int, actual.ravel().tolist()))) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
