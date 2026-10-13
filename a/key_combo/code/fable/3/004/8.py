from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 1000


BASE_PATTERNS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _constant_lines(grid: np.ndarray, bg: int, axis: int) -> tuple[list[int], list[int]]:
    lines: list[int] = []
    colors: list[int] = []
    count = grid.shape[axis]
    for i in range(count):
        line = grid[i, :] if axis == 0 else grid[:, i]
        color = int(line[0])
        if color != bg and bool(np.all(line == color)):
            lines.append(i)
            colors.append(color)
    return lines, colors


def _intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + list(separators) + [int(length)]
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(bool).tobytes()


def _read_cells(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    row_lines, row_colors = _constant_lines(g, bg, axis=0)
    col_lines, col_colors = _constant_lines(g, bg, axis=1)
    assert row_lines and col_lines, "expected full separator rows and columns"
    sep_colors = set(row_colors + col_colors)
    assert len(sep_colors) == 1, "all separator lines must have one color"
    sep = next(iter(sep_colors))

    row_spans = _intervals(g.shape[0], row_lines)
    col_spans = _intervals(g.shape[1], col_lines)
    assert len(row_spans) == len(col_spans), "expected a square cell grid"
    n = len(row_spans)
    assert n >= 3, "expected at least a 3 by 3 cell grid"

    heights = {r1 - r0 for r0, r1 in row_spans}
    widths = {c1 - c0 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "cells must be uniform"
    assert next(iter(heights)) == next(iter(widths)), "cells must be square"

    cells: list[list[dict | None]] = [[None for _ in range(n)] for _ in range(n)]
    shape_masks: dict[tuple[int, int, bytes], np.ndarray] = {}
    colors: set[int] = set()
    empty: list[tuple[int, int]] = []

    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            cell = g[r0:r1, c0:c1]
            mask = (cell != bg) & (cell != sep)
            if not bool(mask.any()):
                empty.append((r, c))
                continue

            vals = np.unique(cell[mask])
            assert len(vals) == 1, "each nonempty cell must contain one glyph color"
            color = int(vals[0])
            key = _shape_key(mask)
            if key not in shape_masks:
                shape_masks[key] = mask.astype(bool).copy()
            cells[r][c] = {"shape": key, "color": color}
            colors.add(color)

    return bg, sep, row_spans, col_spans, cells, shape_masks, colors, empty


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete empty cells by combining the row-missing shape with the column-missing color.

    Input structure:
    - Full separator rows and columns divide the canvas into an N by N grid of
      equal square cells on a background color.
    - Each nonempty cell contains one monochrome glyph. The glyph has two
      independent attributes: its shape mask and its color.
    - A few cells are empty, no two in the same row or column. Across every full
      row the N shapes occur exactly once, and down every full column the N
      colors occur exactly once.

    Rule:
    - For each empty cell, find the one shape absent from that cell's row.
    - Independently find the one color absent from that cell's column.
    - Paint that row-missing shape in that column-missing color into the empty
      cell, leaving separators and all existing cells unchanged.

    Key insight:
    - The two attributes obey different axes: shapes are constrained by rows,
      while colors are constrained by columns. Reversing the axes gives
      contradictions or the wrong completion.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, _sep, row_spans, col_spans, cells, shape_masks, colors, empty = _read_cells(g)
    n = len(row_spans)

    assert empty, "expected at least one empty cell"
    assert len(shape_masks) == n, "expected exactly N distinct glyph shapes"
    assert len(colors) == n, "expected exactly N distinct glyph colors"

    shape_set = set(shape_masks.keys())
    color_set = set(colors)

    row_empty_counts = [0] * n
    col_empty_counts = [0] * n
    for r, c in empty:
        row_empty_counts[r] += 1
        col_empty_counts[c] += 1
    assert max(row_empty_counts) <= 1 and max(col_empty_counts) <= 1, (
        "empty cells must be in distinct rows and columns"
    )

    for r in range(n):
        present_shapes = [cells[r][c]["shape"] for c in range(n) if cells[r][c] is not None]
        assert len(present_shapes) == len(set(present_shapes)), "row shapes must not repeat"

    for c in range(n):
        present_colors = [cells[r][c]["color"] for r in range(n) if cells[r][c] is not None]
        assert len(present_colors) == len(set(present_colors)), "column colors must not repeat"

    for r, c in empty:
        row_shapes = {cells[r][cc]["shape"] for cc in range(n) if cells[r][cc] is not None}
        col_colors = {cells[rr][c]["color"] for rr in range(n) if cells[rr][c] is not None}
        missing_shapes = shape_set - row_shapes
        missing_colors = color_set - col_colors
        assert len(missing_shapes) == 1, "empty cell row must miss exactly one shape"
        assert len(missing_colors) == 1, "empty cell column must miss exactly one color"

        shape = shape_masks[next(iter(missing_shapes))]
        color = int(next(iter(missing_colors)))
        r0, r1 = row_spans[r]
        c0, c1 = col_spans[c]
        patch = out[r0:r1, c0:c1]
        patch[:, :] = bg
        patch[shape] = color

    return out


def _center_mask(pattern: np.ndarray, cell_size: int) -> np.ndarray:
    mask = np.zeros((cell_size, cell_size), dtype=bool)
    ph, pw = pattern.shape
    r0 = (cell_size - ph) // 2
    c0 = (cell_size - pw) // 2
    mask[r0 : r0 + ph, c0 : c0 + pw] = pattern
    return mask


def _sample_holes(n: int) -> set[tuple[int, int]]:
    count = random.randint(2, min(3, n - 1))
    rows = random.sample(range(n), count)
    cols = random.sample(range(n), count)
    random.shuffle(cols)
    return set(zip(rows, cols))


def _sample_shape_grid(n: int) -> list[list[int]]:
    return [random.sample(range(n), n) for _ in range(n)]


def _sample_color_grid(n: int, colors: list[int]) -> list[list[int]]:
    grid = [[0 for _ in range(n)] for _ in range(n)]
    for c in range(n):
        perm = random.sample(colors, n)
        for r, color in enumerate(perm):
            grid[r][c] = int(color)
    return grid


def _semantic_ok(
    shape_grid: list[list[int]],
    color_grid: list[list[int]],
    colors: list[int],
    holes: set[tuple[int, int]],
) -> bool:
    n = len(shape_grid)
    shape_ids = set(range(n))
    color_ids = set(colors)

    if all({shape_grid[r][c] for r in range(n)} == shape_ids for c in range(n)):
        return False
    if all({color_grid[r][c] for c in range(n)} == color_ids for r in range(n)):
        return False

    for sid in range(n):
        paired_colors = {
            color_grid[r][c]
            for r in range(n)
            for c in range(n)
            if shape_grid[r][c] == sid
        }
        if len(paired_colors) < 2:
            return False

    for color in colors:
        paired_shapes = {
            shape_grid[r][c]
            for r in range(n)
            for c in range(n)
            if color_grid[r][c] == color
        }
        if len(paired_shapes) < 2:
            return False

    hole_pairs = {(shape_grid[r][c], color_grid[r][c]) for r, c in holes}
    if len(hole_pairs) != len(holes):
        return False

    for r, c in holes:
        correct_shape = shape_grid[r][c]
        wrong_axis_shape_missing = shape_ids - {shape_grid[rr][c] for rr in range(n) if rr != r}
        if wrong_axis_shape_missing == {correct_shape}:
            return False

        correct_color = color_grid[r][c]
        wrong_axis_color_missing = color_ids - {color_grid[r][cc] for cc in range(n) if cc != c}
        if wrong_axis_color_missing == {correct_color}:
            return False

    return True


def _render_grid(
    n: int,
    cell_size: int,
    shape_masks: list[np.ndarray],
    shape_grid: list[list[int]],
    color_grid: list[list[int]],
    holes: set[tuple[int, int]],
) -> np.ndarray:
    size = n * cell_size + (n - 1)
    grid = np.full((size, size), BACKGROUND, dtype=int)

    for k in range(1, n):
        idx = k * cell_size + (k - 1)
        grid[idx, :] = SEPARATOR
        grid[:, idx] = SEPARATOR

    for r in range(n):
        for c in range(n):
            if (r, c) in holes:
                continue
            r0 = r * (cell_size + 1)
            c0 = c * (cell_size + 1)
            cell = grid[r0 : r0 + cell_size, c0 : c0 + cell_size]
            cell[shape_masks[shape_grid[r][c]]] = color_grid[r][c]

    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4, 5])
    cell_size = random.choice([5, 6]) if n <= 4 else 5
    shape_patterns = random.sample(BASE_PATTERNS, n)
    shape_masks = [_center_mask(pattern, cell_size) for pattern in shape_patterns]
    colors = random.sample(COLOR_POOL, n)

    shape_grid = _sample_shape_grid(n)
    color_grid = _sample_color_grid(n, colors)
    holes = _sample_holes(n)
    if not _semantic_ok(shape_grid, color_grid, colors, holes):
        return None

    expected = _render_grid(n, cell_size, shape_masks, shape_grid, color_grid, set())
    inp = _render_grid(n, cell_size, shape_masks, shape_grid, color_grid, holes)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-axis Latin completion input")
