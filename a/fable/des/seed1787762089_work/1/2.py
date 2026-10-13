from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8

BLUEPRINT_COLORS = (RED, GREEN, YELLOW, ORANGE)
GRID_SIZES = (4, 5)
MAX_GENERATION_ATTEMPTS = 400


def _separator_layout(grid: np.ndarray) -> tuple[int, list[int], list[int], int]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("expected a square 2D grid")
    h, w = g.shape

    for color in np.unique(g):
        row_counts = np.count_nonzero(g == int(color), axis=1)
        col_counts = np.count_nonzero(g == int(color), axis=0)
        rows = np.flatnonzero(row_counts >= w - 1)
        cols = np.flatnonzero(col_counts >= h - 1)

        if len(rows) != len(cols) or len(rows) not in (5, 6):
            continue
        if rows[0] != 0 or cols[0] != 0 or rows[-1] != h - 1 or cols[-1] != w - 1:
            continue

        row_steps = np.diff(rows)
        col_steps = np.diff(cols)
        if row_steps.size == 0:
            continue
        if not np.all(row_steps == row_steps[0]) or not np.all(col_steps == col_steps[0]):
            continue
        if int(row_steps[0]) != int(col_steps[0]):
            continue

        n = len(rows) - 1
        step = int(row_steps[0])
        if step != n + 1 or h != n * n + n + 1:
            continue

        return int(color), rows.tolist(), cols.tolist(), int(n)

    raise ValueError("could not detect separator layout")


def _cell_slices(lines: list[int], idx: int) -> slice:
    return slice(lines[idx] + 1, lines[idx + 1])


def _marker_index(pos: int, lines: list[int]) -> int:
    for idx in range(len(lines) - 1):
        if lines[idx] < pos < lines[idx + 1]:
            return int(idx)
    raise ValueError("marker is not inside a cell span")


def _extract_cells(grid: np.ndarray, row_lines: list[int], col_lines: list[int], n: int) -> list[list[np.ndarray]]:
    cells: list[list[np.ndarray]] = []
    for i in range(n):
        row: list[np.ndarray] = []
        rs = _cell_slices(row_lines, i)
        for j in range(n):
            cs = _cell_slices(col_lines, j)
            cell = np.asarray(grid[rs, cs], dtype=int)
            if cell.shape != (n, n):
                raise ValueError("cell interior shape mismatch")
            row.append(cell.copy())
        cells.append(row)
    return cells


def _majority_and_blueprint(cells: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    flat = [cell for row in cells for cell in row]
    counts = Counter(cell.tobytes() for cell in flat)
    major_key, major_count = counts.most_common(1)[0]
    if major_count != len(flat) - 1 or len(counts) != 2:
        raise ValueError("expected one minority cell and a unique majority pattern")

    minority_key = next(key for key, count in counts.items() if count == 1)
    major = next(cell for cell in flat if cell.tobytes() == major_key).copy()
    blueprint = next(cell for cell in flat if cell.tobytes() == minority_key).copy()
    return major, blueprint


def _render_output(major_pattern: np.ndarray, blueprint: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    n = int(blueprint.shape[0])
    size = n * n + n + 1
    out = np.full((size, size), GREY, dtype=int)
    row_lines = [k * (n + 1) for k in range(n + 1)]
    col_lines = [k * (n + 1) for k in range(n + 1)]
    shifted = np.roll(blueprint, shift=(int(row_shift), int(col_shift)), axis=(0, 1))

    for i in range(n):
        rs = _cell_slices(row_lines, i)
        for j in range(n):
            cs = _cell_slices(col_lines, j)
            color = int(shifted[i, j])
            if color == BLACK:
                out[rs, cs] = major_pattern
            else:
                out[rs, cs] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique minority cell as an N x N blueprint for the whole cell grid.

    The top-border and left-border magenta markers encode the cyclic right/down
    shift applied to the blueprint before it is expanded back to cell scale.
    """
    g = np.asarray(grid, dtype=int)
    grey, row_lines, col_lines, n = _separator_layout(g)

    top_marks = np.flatnonzero(g[0] != grey)
    left_marks = np.flatnonzero(g[:, 0] != grey)
    if len(top_marks) != 1 or len(left_marks) != 1:
        raise ValueError("expected one top marker and one left marker")

    col_shift = _marker_index(int(top_marks[0]), col_lines)
    row_shift = _marker_index(int(left_marks[0]), row_lines)

    cells = _extract_cells(g, row_lines, col_lines, n)
    major_pattern, blueprint = _majority_and_blueprint(cells)
    return _render_output(major_pattern, blueprint, row_shift, col_shift)


def _neighbors4(cell: tuple[int, int], n: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n:
        out.append((r, c + 1))
    return out


def _random_major_pattern(rng: np.random.Generator, n: int) -> np.ndarray:
    target = int(rng.integers(2, min(n * n - 3, 2 * n + 2)))
    start = (int(rng.integers(0, n)), int(rng.integers(0, n)))
    cells = {start}
    frontier = set(_neighbors4(start, n))

    while len(cells) < target and frontier:
        frontier_list = list(frontier)
        pick = frontier_list[int(rng.integers(0, len(frontier_list)))]
        frontier.remove(pick)
        cells.add(pick)
        for nbr in _neighbors4(pick, n):
            if nbr not in cells:
                frontier.add(nbr)

    pattern = np.full((n, n), BLACK, dtype=int)
    for r, c in cells:
        pattern[r, c] = AZURE
    return pattern


def _random_blueprint(rng: np.random.Generator, n: int) -> np.ndarray:
    total = n * n
    num_colors = int(rng.integers(2, min(len(BLUEPRINT_COLORS), n) + 1))
    colors = rng.choice(BLUEPRINT_COLORS, size=num_colors, replace=False).astype(int)

    low = max(num_colors + 1, n)
    high = max(low + 1, min(total - 1, (2 * total) // 3 + 1))
    colored_count = int(rng.integers(low, high))
    positions = rng.choice(total, size=colored_count, replace=False)

    assignment = np.empty(colored_count, dtype=int)
    assignment[:num_colors] = colors
    if colored_count > num_colors:
        assignment[num_colors:] = rng.choice(colors, size=colored_count - num_colors, replace=True)
    rng.shuffle(assignment)

    blueprint = np.full((n, n), BLACK, dtype=int)
    flat = blueprint.reshape(-1)
    flat[positions] = assignment
    return blueprint


def _random_shift(rng: np.random.Generator, n: int) -> tuple[int, int]:
    options = [(r, c) for r in range(n) for c in range(n) if r != 0 or c != 0]
    return options[int(rng.integers(0, len(options)))]


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(rng.choice(GRID_SIZES))
    size = n * n + n + 1
    row_lines = [k * (n + 1) for k in range(n + 1)]
    col_lines = [k * (n + 1) for k in range(n + 1)]

    major_pattern = _random_major_pattern(rng, n)
    blueprint = _random_blueprint(rng, n)
    row_shift, col_shift = _random_shift(rng, n)
    shifted = np.roll(blueprint, shift=(row_shift, col_shift), axis=(0, 1))
    if np.array_equal(shifted, blueprint):
        return None

    minority_r = int(rng.integers(0, n))
    minority_c = int(rng.integers(0, n))

    inp = np.full((size, size), GREY, dtype=int)
    for i in range(n):
        rs = _cell_slices(row_lines, i)
        for j in range(n):
            cs = _cell_slices(col_lines, j)
            inp[rs, cs] = major_pattern

    inp[_cell_slices(row_lines, minority_r), _cell_slices(col_lines, minority_c)] = blueprint

    top_col = int(rng.integers(col_lines[col_shift] + 1, col_lines[col_shift + 1]))
    left_row = int(rng.integers(row_lines[row_shift] + 1, row_lines[row_shift + 1]))
    inp[0, top_col] = MAGENTA
    inp[left_row, 0] = MAGENTA

    expected = _render_output(major_pattern, blueprint, row_shift, col_shift)

    cells = _extract_cells(inp, row_lines, col_lines, n)
    flat = [cell for row in cells for cell in row]
    counts = Counter(cell.tobytes() for cell in flat).values()
    palette = set(np.unique(blueprint)) - {BLACK}

    if sorted(counts) != [1, n * n - 1]:
        return None
    if len(palette) < 2:
        return None
    if np.count_nonzero(blueprint == BLACK) < 2:
        return None
    if np.count_nonzero(np.isin(blueprint, (GREY, MAGENTA, AZURE))) != 0:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
