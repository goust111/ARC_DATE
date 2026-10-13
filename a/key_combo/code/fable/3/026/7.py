from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

EMPTY_COLOR = BLACK
WALL_COLOR = GREY
TIE_COLOR = PINK
GRID_COLOR = TEAL
SEED_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE, MAROON)
INF = 10**9


def _line_indices(g: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(g.shape[0]) if np.all(g[r, :] == GRID_COLOR)]
    return [c for c in range(g.shape[1]) if np.all(g[:, c] == GRID_COLOR)]


def _regular_grid_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    rows = _line_indices(g, 0)
    cols = _line_indices(g, 1)
    assert len(rows) >= 3 and len(cols) >= 3, "expected at least a 2x2 lattice"
    assert rows[0] == 0 and rows[-1] == g.shape[0] - 1
    assert cols[0] == 0 and cols[-1] == g.shape[1] - 1

    row_steps = np.diff(rows)
    col_steps = np.diff(cols)
    assert len(set(map(int, row_steps))) == 1 and int(row_steps[0]) >= 2
    assert len(set(map(int, col_steps))) == 1 and int(col_steps[0]) >= 2
    return rows, cols


def _read_cells(g: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    rows, cols = _regular_grid_lines(g)
    cell_rows = len(rows) - 1
    cell_cols = len(cols) - 1
    cells = np.empty((cell_rows, cell_cols), dtype=int)

    for rr in range(cell_rows):
        for cc in range(cell_cols):
            block = g[rows[rr] + 1 : rows[rr + 1], cols[cc] + 1 : cols[cc + 1]]
            assert block.size > 0
            color = int(block[0, 0])
            assert np.all(block == color), "each lattice cell must be solid"
            assert color != GRID_COLOR, "grid-line color may not be used as a cell fill"
            cells[rr, cc] = color
    return cells, rows, cols


def _find_seeds(cells: np.ndarray) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            color = int(cells[r, c])
            if color not in (EMPTY_COLOR, WALL_COLOR):
                assert color != TIE_COLOR, "pink is reserved for equal-distance output"
                seeds.append((r, c, color))
    assert len(seeds) == 2, "expected exactly two seed cells"
    assert seeds[0][2] != seeds[1][2], "seed cells must have different colors"
    return seeds


def _bfs(walls: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = walls.shape
    dist = np.full((h, w), INF, dtype=int)
    sr, sc = seed
    assert not bool(walls[sr, sc])
    dist[sr, sc] = 0
    q: deque[tuple[int, int]] = deque([(sr, sc)])

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not bool(walls[nr, nc]):
                if dist[nr, nc] == INF:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
    return dist


def _paint_cell(out: np.ndarray, rows: list[int], cols: list[int], r: int, c: int, color: int) -> None:
    out[rows[r] + 1 : rows[r + 1], cols[c] + 1 : cols[c + 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Fill a walled lattice maze by shortest-path territory expansion from two colored seed cells.

    Input structure:
    A cyan one-pixel grid partitions the canvas into equal rectangular cells.
    Exactly two non-black, non-grey cells are solid seed cells with different
    colors. Some whole cells are grey walls, and the remaining whole cells are
    black empty ground.

    Rule:
    Treat the rectangular cells as the graph nodes, adjacent only up, down,
    left, and right. Grey wall cells are blocked. Every black cell is recolored
    with the seed color whose cell-graph shortest-path distance is smaller; if
    both distances are equal, it becomes pink. Seeds, grey walls, and cyan grid
    lines are preserved.

    Key insight:
    Distance is measured in lattice steps after respecting the grey walls, so a
    visually nearby seed can lose when the path must detour around the maze.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cells, rows, cols = _read_cells(g)
    seeds = _find_seeds(cells)
    walls = cells == WALL_COLOR
    (r0, c0, color0), (r1, c1, color1) = seeds

    dist0 = _bfs(walls, (r0, c0))
    dist1 = _bfs(walls, (r1, c1))

    out = g.copy()
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            if int(cells[r, c]) != EMPTY_COLOR:
                continue
            d0 = int(dist0[r, c])
            d1 = int(dist1[r, c])
            assert min(d0, d1) < INF, "every empty cell must be reachable"
            if d0 < d1:
                fill = color0
            elif d1 < d0:
                fill = color1
            else:
                fill = TIE_COLOR
            _paint_cell(out, rows, cols, r, c, fill)
    return out


# --------------- generator ---------------

CELL_ROWS_RANGE = (5, 9)
CELL_COLS_RANGE = (5, 10)
CELL_SIZE_VALUES = (2, 3)
EXTRA_WALL_RATE_RANGE = (0.04, 0.13)
MAX_GENERATION_ATTEMPTS = 2000


def _sample_dimensions(rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    cell_h = int(rng.choice(CELL_SIZE_VALUES))
    cell_w = int(rng.choice(CELL_SIZE_VALUES))
    max_cell_rows = min(CELL_ROWS_RANGE[1], (30 - 1) // (cell_h + 1))
    max_cell_cols = min(CELL_COLS_RANGE[1], (30 - 1) // (cell_w + 1))
    if max_cell_rows < CELL_ROWS_RANGE[0] or max_cell_cols < CELL_COLS_RANGE[0]:
        return None
    rows = int(rng.integers(CELL_ROWS_RANGE[0], max_cell_rows + 1))
    cols = int(rng.integers(CELL_COLS_RANGE[0], max_cell_cols + 1))
    return rows, cols, cell_h, cell_w


def _base_barrier(
    rows: int,
    cols: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str, int, int]:
    walls = np.zeros((rows, cols), dtype=bool)
    orientation = "vertical" if int(rng.integers(2)) == 0 else "horizontal"

    if orientation == "vertical":
        index = int(rng.integers(1, cols - 1))
        if cols >= 7:
            index = int(rng.integers(2, cols - 2))
        gap = int(rng.integers(0, rows))
        walls[:, index] = True
        walls[gap, index] = False
    else:
        index = int(rng.integers(1, rows - 1))
        if rows >= 7:
            index = int(rng.integers(2, rows - 2))
        gap = int(rng.integers(0, cols))
        walls[index, :] = True
        walls[index, gap] = False
    return walls, orientation, index, gap


def _choose_seeds(
    walls: np.ndarray,
    orientation: str,
    index: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rows, cols = walls.shape
    if orientation == "vertical":
        side_a = [(r, c) for r in range(rows) for c in range(index) if not walls[r, c]]
        side_b = [(r, c) for r in range(rows) for c in range(index + 1, cols) if not walls[r, c]]
    else:
        side_a = [(r, c) for r in range(index) for c in range(cols) if not walls[r, c]]
        side_b = [(r, c) for r in range(index + 1, rows) for c in range(cols) if not walls[r, c]]

    if not side_a or not side_b:
        return None
    seed_a = side_a[int(rng.integers(0, len(side_a)))]
    seed_b = side_b[int(rng.integers(0, len(side_b)))]
    if int(rng.integers(2)):
        seed_a, seed_b = seed_b, seed_a
    return seed_a, seed_b


def _add_extra_walls(
    walls: np.ndarray,
    seeds: tuple[tuple[int, int], tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    out = walls.copy()
    seed_set = set(seeds)
    rate = float(rng.uniform(*EXTRA_WALL_RATE_RANGE))
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if out[r, c] or (r, c) in seed_set:
                continue
            if rng.random() < rate:
                out[r, c] = True
    return out


def _classify_cells(
    walls: np.ndarray,
    seeds: tuple[tuple[int, int], tuple[int, int]],
    seed_colors: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist0 = _bfs(walls, seeds[0])
    dist1 = _bfs(walls, seeds[1])
    labels = np.full(walls.shape, EMPTY_COLOR, dtype=int)
    labels[walls] = WALL_COLOR
    labels[seeds[0]] = seed_colors[0]
    labels[seeds[1]] = seed_colors[1]

    for r in range(walls.shape[0]):
        for c in range(walls.shape[1]):
            if walls[r, c] or (r, c) == seeds[0] or (r, c) == seeds[1]:
                continue
            d0 = int(dist0[r, c])
            d1 = int(dist1[r, c])
            if min(d0, d1) >= INF:
                labels[r, c] = -1
            elif d0 < d1:
                labels[r, c] = seed_colors[0]
            elif d1 < d0:
                labels[r, c] = seed_colors[1]
            else:
                labels[r, c] = TIE_COLOR
    return labels, dist0, dist1


def _naive_labels(
    walls: np.ndarray,
    seeds: tuple[tuple[int, int], tuple[int, int]],
    seed_colors: tuple[int, int],
) -> np.ndarray:
    labels = np.full(walls.shape, EMPTY_COLOR, dtype=int)
    labels[walls] = WALL_COLOR
    labels[seeds[0]] = seed_colors[0]
    labels[seeds[1]] = seed_colors[1]
    for r in range(walls.shape[0]):
        for c in range(walls.shape[1]):
            if walls[r, c] or (r, c) == seeds[0] or (r, c) == seeds[1]:
                continue
            d0 = abs(r - seeds[0][0]) + abs(c - seeds[0][1])
            d1 = abs(r - seeds[1][0]) + abs(c - seeds[1][1])
            if d0 < d1:
                labels[r, c] = seed_colors[0]
            elif d1 < d0:
                labels[r, c] = seed_colors[1]
            else:
                labels[r, c] = TIE_COLOR
    return labels


def _semantic_ok(
    walls: np.ndarray,
    seeds: tuple[tuple[int, int], tuple[int, int]],
    seed_colors: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    expected, dist0, dist1 = _classify_cells(walls, seeds, seed_colors)
    if np.any(expected == -1):
        return False, None

    empty = ~(walls.copy())
    empty[seeds[0]] = False
    empty[seeds[1]] = False
    assigned = set(map(int, expected[empty].ravel()))
    if TIE_COLOR not in assigned:
        return False, None
    if seed_colors[0] not in assigned or seed_colors[1] not in assigned:
        return False, None

    naive = _naive_labels(walls, seeds, seed_colors)
    if not np.any((expected != naive) & empty):
        return False, None

    seed_distance = int(dist0[seeds[1]])
    if seed_distance >= INF or seed_distance < 4:
        return False, None
    if int(np.sum(walls)) < max(3, min(walls.shape) - 1):
        return False, None
    return True, expected


def _render_cells(cells: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray:
    rows, cols = cells.shape
    out = np.full((rows * (cell_h + 1) + 1, cols * (cell_w + 1) + 1), EMPTY_COLOR, dtype=int)
    out[:: cell_h + 1, :] = GRID_COLOR
    out[:, :: cell_w + 1] = GRID_COLOR

    for r in range(rows):
        for c in range(cols):
            color = int(cells[r, c])
            r0 = r * (cell_h + 1) + 1
            c0 = c * (cell_w + 1) + 1
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    dims = _sample_dimensions(rng)
    if dims is None:
        return None
    rows, cols, cell_h, cell_w = dims

    walls, orientation, index, _gap = _base_barrier(rows, cols, rng)
    seeds = _choose_seeds(walls, orientation, index, rng)
    if seeds is None:
        return None
    walls = _add_extra_walls(walls, seeds, rng)

    seed_colors = tuple(int(c) for c in rng.choice(SEED_COLORS, size=2, replace=False))
    ok, expected = _semantic_ok(walls, seeds, seed_colors)
    if not ok or expected is None:
        return None

    inp_cells = np.full((rows, cols), EMPTY_COLOR, dtype=int)
    inp_cells[walls] = WALL_COLOR
    inp_cells[seeds[0]] = seed_colors[0]
    inp_cells[seeds[1]] = seed_colors[1]
    inp = _render_cells(inp_cells, cell_h, cell_w)
    out = _render_cells(expected, cell_h, cell_w)

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze ripple input")
