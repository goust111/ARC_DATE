from __future__ import annotations

import random
from bisect import bisect_right

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_NAME = {(-1, 0): "N", (1, 0): "S", (0, -1): "W", (0, 1): "E"}
OPPOSITE = {"N": "S", "S": "N", "W": "E", "E": "W"}
PORTS = {"N": (1, 2), "S": (3, 2), "W": (2, 1), "E": (2, 3)}


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _find_lattice_lines(grid: np.ndarray) -> tuple[list[int], list[int], int]:
    h, w = grid.shape
    row_hits = [r for r in range(h) if int(np.sum(grid[r] == GREY)) > w // 2]
    col_hits = [c for c in range(w) if int(np.sum(grid[:, c] == GREY)) > h // 2]

    assert len(row_hits) >= 4 and len(col_hits) >= 4
    assert row_hits[0] == 0 and row_hits[-1] == h - 1
    assert col_hits[0] == 0 and col_hits[-1] == w - 1

    row_steps = [b - a for a, b in zip(row_hits, row_hits[1:])]
    col_steps = [b - a for a, b in zip(col_hits, col_hits[1:])]
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1
    assert row_steps[0] == col_steps[0]
    cell_size = row_steps[0] - 1
    assert cell_size == 3
    return row_hits, col_hits, cell_size


def _cell_of_coord(
    coord: tuple[int, int], line_rows: list[int], line_cols: list[int]
) -> tuple[int, int] | None:
    r, c = coord
    if r in line_rows or c in line_cols:
        return None

    ri = bisect_right(line_rows, r) - 1
    ci = bisect_right(line_cols, c) - 1
    if not (0 <= ri < len(line_rows) - 1 and 0 <= ci < len(line_cols) - 1):
        return None
    if not (line_rows[ri] < r < line_rows[ri + 1] and line_cols[ci] < c < line_cols[ci + 1]):
        return None
    return ri, ci


def _cell_class(cell: tuple[int, int], n_rows: int, n_cols: int) -> int:
    r, c = cell
    return RED if r in (0, n_rows - 1) or c in (0, n_cols - 1) else BLUE


def _visited_cells(
    grid: np.ndarray, line_rows: list[int], line_cols: list[int]
) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(grid == GREEN):
        cell = _cell_of_coord((int(r), int(c)), line_rows, line_cols)
        if cell is not None:
            cells.add(cell)
    return cells


def _cell_sides(
    grid: np.ndarray, cell: tuple[int, int], line_rows: list[int], line_cols: list[int]
) -> set[str]:
    r, c = cell
    top, bottom = line_rows[r], line_rows[r + 1]
    left, right = line_cols[c], line_cols[c + 1]

    sides = set()
    if np.any(grid[top, left + 1 : right] == GREEN):
        sides.add("N")
    if np.any(grid[bottom, left + 1 : right] == GREEN):
        sides.add("S")
    if np.any(grid[top + 1 : bottom, left] == GREEN):
        sides.add("W")
    if np.any(grid[top + 1 : bottom, right] == GREEN):
        sides.add("E")
    return sides


def _cell_graph(
    grid: np.ndarray, line_rows: list[int], line_cols: list[int]
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    cells = _visited_cells(grid, line_rows, line_cols)
    n_rows = len(line_rows) - 1
    n_cols = len(line_cols) - 1
    graph = {cell: set() for cell in cells}
    side_cache = {cell: _cell_sides(grid, cell, line_rows, line_cols) for cell in cells}

    for cell in cells:
        r, c = cell
        for direction in side_cache[cell]:
            dr, dc = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}[direction]
            nr, nc = r + dr, c + dc
            nbr = (nr, nc)
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            if nbr not in cells:
                continue
            if OPPOSITE[direction] in side_cache[nbr]:
                graph[cell].add(nbr)
    return graph


def _trace_cell_path(graph: dict[tuple[int, int], set[tuple[int, int]]]) -> list[tuple[int, int]]:
    assert graph
    degrees = {cell: len(nbrs) for cell, nbrs in graph.items()}
    assert all(deg in (1, 2) for deg in degrees.values())
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2

    start = min(endpoints)
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(len(graph)):
        order.append(cur)
        next_cells = [nbr for nbr in graph[cur] if nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1
        prev, cur = cur, next_cells[0]
    assert len(order) == len(graph)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten the cell-visitation order of a snake path through a lattice, recoloring each visited cell by perimeter-vs-interior status.

    Input structure:
    - A grey lattice fills the complete outer border and all regular internal separators, dividing the board into a full rectangular array of square 3x3 cells.
    - A green 4-connected snake runs through one-cell openings in the lattice; within each visited cell it paints a small in-cell segment, and adjacent visited cells are linked through the openings.
    - The snake visits at least six cells, forms one simple chain with exactly two endpoints, and includes both perimeter cells and strictly interior cells of the cell array.

    Rule:
    - Recover the lattice rows and columns, then identify which cells are visited by the snake and which side-openings connect neighboring visited cells.
    - Treat the visited cells themselves as the snake sequence, choose the endpoint with the smaller array row (breaking ties by array column), and traverse the cell chain to the other endpoint.
    - Output a single column whose rows follow that traversal order: color each visited cell red if it lies on the outer perimeter of the cell array, otherwise blue.

    Key insight:
    - The snake determines the readout order, but the lattice coordinates determine the recoloring, so the solver must combine path tracing with cell-position classification.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    line_rows, line_cols, _ = _find_lattice_lines(g)
    graph = _cell_graph(g, line_rows, line_cols)
    order = _trace_cell_path(graph)
    n_rows = len(line_rows) - 1
    n_cols = len(line_cols) - 1
    colors = [_cell_class(cell, n_rows, n_cols) for cell in order]
    return np.asarray(colors, dtype=int).reshape(-1, 1)


def _dfs_cell_path(
    path: list[tuple[int, int]],
    visited: set[tuple[int, int]],
    target_len: int,
    n_rows: int,
    n_cols: int,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        return list(path)

    options = [nbr for nbr in _neighbors4(path[-1]) if 0 <= nbr[0] < n_rows and 0 <= nbr[1] < n_cols and nbr not in visited]
    random.shuffle(options)
    for nxt in options:
        visited.add(nxt)
        path.append(nxt)
        result = _dfs_cell_path(path, visited, target_len, n_rows, n_cols)
        if result is not None:
            return result
        path.pop()
        visited.remove(nxt)
    return None


def _generate_cell_path(n_rows: int, n_cols: int) -> list[tuple[int, int]] | None:
    max_len = min(12, n_rows * n_cols - 1)
    if max_len < 6:
        return None

    starts = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    target_lengths = list(range(6, max_len + 1))
    random.shuffle(target_lengths)

    for target_len in target_lengths:
        random.shuffle(starts)
        for start in starts:
            path = _dfs_cell_path([start], {start}, target_len, n_rows, n_cols)
            if path is None:
                continue
            edge_flags = [cell[0] in (0, n_rows - 1) or cell[1] in (0, n_cols - 1) for cell in path]
            edge_count = sum(edge_flags)
            interior_count = len(path) - edge_count
            if edge_count < 2 or interior_count < 1:
                continue
            turn_count = 0
            for a, b, c in zip(path, path[1:], path[2:]):
                d1 = (b[0] - a[0], b[1] - a[1])
                d2 = (c[0] - b[0], c[1] - b[1])
                if d1 != d2:
                    turn_count += 1
            if turn_count == 0:
                continue
            if not any(edge_flags[i] != edge_flags[i + 1] for i in range(len(path) - 1)):
                continue
            return path
    return None


def _cell_origin(cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    return 4 * r, 4 * c


def _paint_cell_segment(grid: np.ndarray, cell: tuple[int, int], dirs: set[str]) -> None:
    top, left = _cell_origin(cell)
    grid[top + 2, left + 2] = GREEN
    for direction in dirs:
        dr, dc = PORTS[direction]
        grid[top + dr, left + dc] = GREEN


def _paint_gap(grid: np.ndarray, cell: tuple[int, int], direction: str) -> None:
    top, left = _cell_origin(cell)
    if direction == "N":
        grid[top, left + 2] = GREEN
    elif direction == "S":
        grid[top + 4, left + 2] = GREEN
    elif direction == "W":
        grid[top + 2, left] = GREEN
    elif direction == "E":
        grid[top + 2, left + 4] = GREEN
    else:
        raise AssertionError("unknown direction")


def _render_grid(path: list[tuple[int, int]], n_rows: int, n_cols: int) -> np.ndarray:
    height = 4 * n_rows + 1
    width = 4 * n_cols + 1
    grid = np.full((height, width), BG, dtype=int)

    for r in range(0, height, 4):
        grid[r, :] = GREY
    for c in range(0, width, 4):
        grid[:, c] = GREY

    for i, cell in enumerate(path):
        dirs = set()
        if i > 0:
            prev = path[i - 1]
            dirs.add(DIR_TO_NAME[(prev[0] - cell[0], prev[1] - cell[1])])
        if i + 1 < len(path):
            nxt = path[i + 1]
            dirs.add(DIR_TO_NAME[(nxt[0] - cell[0], nxt[1] - cell[1])])
        _paint_cell_segment(grid, cell, dirs)
        for direction in dirs:
            _paint_gap(grid, cell, direction)

    return grid


def _expected_output(path: list[tuple[int, int]], n_rows: int, n_cols: int) -> np.ndarray:
    order = list(path)
    if order[-1] < order[0]:
        order.reverse()
    colors = [_cell_class(cell, n_rows, n_cols) for cell in order]
    return np.asarray(colors, dtype=int).reshape(-1, 1)


def _validate_generated_grid(grid: np.ndarray, path: list[tuple[int, int]], n_rows: int, n_cols: int) -> bool:
    try:
        line_rows, line_cols, cell_size = _find_lattice_lines(grid)
        if cell_size != 3 or len(line_rows) - 1 != n_rows or len(line_cols) - 1 != n_cols:
            return False

        graph = _cell_graph(grid, line_rows, line_cols)
        if set(graph) != set(path):
            return False
        degrees = sorted(len(v) for v in graph.values())
        if degrees.count(1) != 2 or any(deg not in (1, 2) for deg in degrees):
            return False

        expected = _expected_output(path, n_rows, n_cols)
        actual = main(grid)
        return np.array_equal(actual, expected)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(3, 5)
    n_cols = random.randint(3, 5)
    path = _generate_cell_path(n_rows, n_cols)
    if path is None:
        return None

    if random.random() < 0.5:
        path = list(reversed(path))

    grid = _render_grid(path, n_rows, n_cols)
    if not _validate_generated_grid(grid, path, n_rows, n_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice snake input")
