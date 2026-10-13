from __future__ import annotations

from collections import Counter
import sys
import random
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    _file = Path(__file__).resolve()
    _arc_utils_path = None
    for parent in _file.parents:
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            _arc_utils_path = candidate.parent
            break
    if _arc_utils_path is None:
        raise
    sys.path.insert(0, str(_arc_utils_path))
    from arc_utils import get_bg


def _infer_separator_color(g: np.ndarray, bg: int) -> int:
    color_count = Counter(int(x) for x in g.ravel() if int(x) != bg)
    assert color_count, "expected separator color and centers"
    sep_color, _ = color_count.most_common(1)[0]
    return sep_color


def _separator_lines(g: np.ndarray, bg: int, sep: int, *, axis: int) -> list[int]:
    arr = g if axis == 0 else g.T
    length = arr.shape[1]
    min_sep = max(3, length // 2)
    lines: list[int] = []
    for idx in range(arr.shape[0]):
        line = arr[idx]
        other = np.logical_and(line != bg, line != sep)
        sep_count = int(np.count_nonzero(line == sep))
        if np.any(other):
            continue
        if sep_count >= min_sep:
            lines.append(idx)
    return lines


def _cell_intervals(size: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(separators)
    assert separators == sorted(set(separators))
    assert 0 <= separators[0] < size if separators else separators == []
    assert separators[-1] < size if separators else True

    out: list[tuple[int, int]] = []
    prev = -1
    for line in separators:
        out.append((prev + 1, line - 1))
        prev = line
    out.append((prev + 1, size - 1))
    assert all(r0 <= r1 for r0, r1 in out)
    return out


def _find_start_cell(
    g: np.ndarray,
    rows: list[tuple[int, int]],
    cols: list[tuple[int, int]],
    bg: int,
    sep: int,
) -> tuple[int, int]:
    starts: list[tuple[int, int]] = []
    for ri, (r0, r1) in enumerate(rows):
        for ci, (c0, c1) in enumerate(cols):
            cr = (r0 + r1) // 2
            cc = (c0 + c1) // 2
            if cr + 1 > r1 or cc + 1 > c1:
                continue
            block = g[cr : cr + 2, cc : cc + 2]
            color = int(block[0, 0])
            if color == bg or color == sep:
                continue
            if np.all(block == color):
                starts.append((ri, ci))
    assert len(starts) == 1, f"expected one 2x2 start cell, found {len(starts)}"
    return starts[0]


def _build_path_adjacency(
    rows: list[tuple[int, int]],
    cols: list[tuple[int, int]],
    row_lines: list[int],
    col_lines: list[int],
    g: np.ndarray,
    bg: int,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    h_cells = len(rows)
    w_cells = len(cols)
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {
        (r, c): set() for r in range(h_cells) for c in range(w_cells)
    }
    for ri in range(h_cells):
        for ci in range(w_cells):
            cell = (ri, ci)
            if ri + 1 < h_cells:
                line = row_lines[ri]
                c0, c1 = cols[ci]
                if np.any(g[line, c0 : c1 + 1] == bg):
                    neighbor = (ri + 1, ci)
                    adjacency[cell].add(neighbor)
                    adjacency[neighbor].add(cell)
            if ci + 1 < w_cells:
                line = col_lines[ci]
                r0, r1 = rows[ri]
                if np.any(g[r0 : r1 + 1, line] == bg):
                    neighbor = (ri, ci + 1)
                    adjacency[cell].add(neighbor)
                    adjacency[neighbor].add(cell)
    return adjacency


def _walk_chain(start: tuple[int, int], adjacency: dict[tuple[int, int], set[tuple[int, int]]]) -> list[tuple[int, int]]:
    path = [start]
    seen = {start}
    prev = None
    cur = start
    if len(adjacency.get(start, set())) > 1:
        raise AssertionError("start cell is not an endpoint")

    while True:
        neighbors = [n for n in adjacency.get(cur, set()) if n != prev]
        if not neighbors:
            break
        if len(neighbors) != 1:
            raise AssertionError("non-unique path traversal encountered")
        nxt = neighbors[0]
        if nxt in seen:
            raise AssertionError("cycle detected in traversal path")
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Traverse gap-door chain through cell separators and output the visited cell-center colors as a horizontal stripe.

    Input structure:
    - A black-background square grid split by one non-black separator color into equal square cells in a complete row/column lattice.
    - Each cell has exactly one central colored mark; one start cell uses a 2x2 same-color block.
    - Black pixels on separator lines are single-pixel gaps, and they connect adjacent cells to form one unique non-branching chain.

    Rule:
    - Detect separator lines and recover the cell partition grid.
    - Identify the unique 2x2 same-color start cell and follow the gap-defined adjacency as a single chain starting there.
    - Output a 1×n strip of the center color of each visited cell in traversal order, with the start cell at the left.

    Key insight:
    - Gaps are not noise; they are the graph edges, so the required order is the topological traversal order of cells along the unique path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "expected square grid"

    bg = get_bg(g)
    sep = _infer_separator_color(g, bg)

    row_lines = _separator_lines(g, bg, sep, axis=0)
    col_lines = _separator_lines(g, bg, sep, axis=1)
    assert row_lines and col_lines, "could not detect separator lines"

    rows = _cell_intervals(h, row_lines)
    cols = _cell_intervals(w, col_lines)
    assert len(rows) == len(cols), "expected equal square cells"
    cell_h = rows[0][1] - rows[0][0] + 1
    cell_w = cols[0][1] - cols[0][0] + 1
    assert cell_h == cell_w and cell_h > 1 and cell_w > 1

    start = _find_start_cell(g, rows, cols, bg, sep)
    adjacency = _build_path_adjacency(rows, cols, row_lines, col_lines, g, bg)
    path = _walk_chain(start, adjacency)

    colors: list[int] = []
    for ri, ci in path:
        r0, r1 = rows[ri]
        c0, c1 = cols[ci]
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        colors.append(int(g[cr, cc]))
    return np.array([colors], dtype=int)


def _generate_grid(n_cells: int, cell_size: int, line_color: int) -> tuple[np.ndarray, list[int], list[int]]:
    size = n_cells * (cell_size + 1) - 1
    grid = np.full((size, size), 0, dtype=int)
    row_lines = [k * (cell_size + 1) - 1 for k in range(1, n_cells)]
    col_lines = row_lines.copy()

    for r in row_lines:
        grid[r, :] = line_color
    for c in col_lines:
        grid[:, c] = line_color

    rows = [(k * (cell_size + 1), k * (cell_size + 1) + cell_size - 1) for k in range(n_cells)]
    cols = rows
    return grid, row_lines, col_lines


def _generate_path(length: int, n_cells: int) -> list[tuple[int, int]] | None:
    if length < 1 or length > n_cells * n_cells:
        return None
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> bool:
        if len(path) == length:
            return True
        r, c = path[-1]
        candidates = [(r + dr, c + dc) for dr, dc in dirs]
        random.shuffle(candidates)
        for nr, nc in candidates:
            if not (0 <= nr < n_cells and 0 <= nc < n_cells):
                continue
            nxt = (nr, nc)
            if nxt in used:
                continue
            used.add(nxt)
            path.append(nxt)
            if dfs(path, used):
                return True
            path.pop()
            used.remove(nxt)
        return False

    cells = [(r, c) for r in range(n_cells) for c in range(n_cells)]
    for _ in range(300):
        path: list[tuple[int, int]] = [random.choice(cells)]
        used = {path[0]}
        if dfs(path, used):
            return path
    return None


def _expected_output(path: list[tuple[int, int]], cell_colors: np.ndarray) -> np.ndarray:
    colors = [int(cell_colors[r, c]) for r, c in path]
    return np.array([colors], dtype=int)


def _paint_centers(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    cell_size: int,
    row_lines: list[int],
    col_lines: list[int],
    cell_colors: np.ndarray,
    bg: int = 0,
) -> None:
    rows = [(k * (cell_size + 1), k * (cell_size + 1) + cell_size - 1) for k in range(len(row_lines) + 1)]
    cols = rows

    for ri in range(len(rows)):
        for ci in range(len(cols)):
            r0, r1 = rows[ri]
            c0, c1 = cols[ci]
            cr = (r0 + r1) // 2
            cc = (c0 + c1) // 2
            color = int(cell_colors[ri, ci])
            if (ri, ci) == path[0]:
                grid[cr : cr + 2, cc : cc + 2] = color
            else:
                grid[cr, cc] = color


def _paint_gaps(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    cell_size: int,
    row_lines: list[int],
    col_lines: list[int],
    bg: int = 0,
) -> None:
    rows = [(k * (cell_size + 1), k * (cell_size + 1) + cell_size - 1) for k in range(len(row_lines) + 1)]
    cols = rows
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        if r0 == r1:
            line_col = col_lines[min(c0, c1)]
            rr0, rr1 = rows[r0]
            gap_row = random.randint(rr0, rr1)
            grid[gap_row, line_col] = bg
        else:
            line_row = row_lines[min(r0, r1)]
            cc0, cc1 = cols[c0]
            gap_col = random.randint(cc0, cc1)
            grid[line_row, gap_col] = bg


def _gen_attempt() -> np.ndarray | None:
    n_cells = random.randint(3, 5)
    cell_size = random.randint(3, 5)
    size = n_cells * (cell_size + 1) - 1
    if size > 30:
        return None

    line_color = random.randint(1, 9)
    bg = 0
    grid, row_lines, col_lines = _generate_grid(n_cells, cell_size, line_color)
    total_cells = n_cells * n_cells
    if total_cells <= 3:
        path_len = total_cells
    elif random.random() < 0.8:
        path_len = random.randint(2, total_cells - 1)
    else:
        path_len = total_cells

    path = _generate_path(path_len, n_cells)
    if path is None:
        return None

    palette = [color for color in range(11, 11 + 3 * total_cells)]
    random.shuffle(palette)
    colors = np.array(palette[:total_cells], dtype=int).reshape((n_cells, n_cells))
    _paint_centers(grid, path, cell_size, row_lines, col_lines, colors, bg=bg)
    _paint_gaps(grid, path, cell_size, row_lines, col_lines, bg=bg)

    out = _expected_output(path, colors)
    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _gen_attempt()
        if grid is not None:
            assert grid.ndim == 2
            assert max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate valid gateway-traversal input")
