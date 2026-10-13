from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Window = tuple[int, int]

WINDOW_SIZE = 3
SEPARATOR_STRIDE = WINDOW_SIZE + 1
BACKGROUND_COLOR = 0
SEPARATOR_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GENERATION_ATTEMPTS = 300


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Mirror arc_utils: grid shape is (rows, cols) and coords are (r, c)."""
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _mode(values: np.ndarray) -> int:
    flat = [int(v) for v in values.ravel()]
    return Counter(flat).most_common(1)[0][0]


def _runs_from_separator_mask(mask: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    start: int | None = None
    for idx, is_separator in enumerate(mask.tolist()):
        if not is_separator and start is None:
            start = idx
        elif is_separator and start is not None:
            runs.append(list(range(start, idx)))
            start = None
    if start is not None:
        runs.append(list(range(start, len(mask))))
    return runs


def _detect_window_lattice(grid: np.ndarray) -> tuple[int, list[list[int]], list[list[int]], int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int, list[list[int]], list[list[int]], np.ndarray, np.ndarray]] = []

    for color in sorted(set(map(int, g.ravel()))):
        row_mask = np.all(g == color, axis=1)
        col_mask = np.all(g == color, axis=0)
        row_blocks = _runs_from_separator_mask(row_mask)
        col_blocks = _runs_from_separator_mask(col_mask)
        if not row_blocks or not col_blocks:
            continue
        if not all(len(block) == WINDOW_SIZE for block in row_blocks):
            continue
        if not all(len(block) == WINDOW_SIZE for block in col_blocks):
            continue
        candidates.append(
            (
                int(row_mask.sum() + col_mask.sum()),
                int(color),
                row_blocks,
                col_blocks,
                row_mask,
                col_mask,
            )
        )

    assert candidates, "expected full-row/full-column separator lines delimiting 3x3 windows"
    _score, sep_color, row_blocks, col_blocks, row_mask, col_mask = max(candidates, key=lambda item: item[0])

    interior = g[np.ix_(~row_mask, ~col_mask)]
    assert interior.size > 0, "window lattice must contain non-separator cells"
    bg = _mode(interior)
    return sep_color, row_blocks, col_blocks, bg


def _parse_markers(
    grid: np.ndarray,
    row_blocks: list[list[int]],
    col_blocks: list[list[int]],
    bg: int,
) -> dict[Window, tuple[Coord, int]]:
    markers: dict[Window, tuple[Coord, int]] = {}
    for wr, rows in enumerate(row_blocks):
        for wc, cols in enumerate(col_blocks):
            block = grid[np.ix_(rows, cols)]
            marker_cells = np.argwhere(block != bg)
            assert marker_cells.shape == (1, 2), "each 3x3 window must contain exactly one visible marker"
            mr, mc = map(int, marker_cells[0])
            markers[(wr, wc)] = ((mr - 1, mc - 1), int(block[mr, mc]))
    return markers


def _resolve_terminal_colors(markers: dict[Window, tuple[Coord, int]], shape: tuple[int, int]) -> dict[Window, int]:
    parents: dict[Window, Window | None] = {}
    terminal_colors: dict[Window, int] = {}
    for cell, (offset, color) in markers.items():
        dr, dc = offset
        if (dr, dc) == (0, 0):
            parents[cell] = None
            terminal_colors[cell] = int(color)
            continue
        nr, nc = cell[0] + dr, cell[1] + dc
        assert in_bounds(shape, nr, nc), "pointer must stay inside the window array"
        parents[cell] = (nr, nc)

    memo: dict[Window, int] = {}
    visiting: set[Window] = set()

    def walk(cell: Window) -> int:
        if cell in memo:
            return memo[cell]
        assert cell not in visiting, "pointer graph must be acyclic"
        visiting.add(cell)
        parent = parents[cell]
        if parent is None:
            result = terminal_colors[cell]
        else:
            result = walk(parent)
        visiting.remove(cell)
        memo[cell] = int(result)
        return int(result)

    return {cell: walk(cell) for cell in markers}


def _paint_windows(
    grid: np.ndarray,
    row_blocks: list[list[int]],
    col_blocks: list[list[int]],
    fill_colors: dict[Window, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for (wr, wc), color in fill_colors.items():
        rows = row_blocks[wr]
        cols = col_blocks[wc]
        out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every 3x3 window with the color of the terminal reached by its direction marker.

    Input structure:
    - Grey separator rows and columns partition the grid into a lattice of 3x3 windows.
    - Each window contains exactly one colored marker pixel on an otherwise uniform window background.
    - The marker can sit at the center or at one of the other eight positions around the center.

    Rule:
    - Read each marker's offset from the window center as an 8-direction pointer to the neighboring window in that direction.
    - A marker at the exact center denotes a terminal window; its own marker color is the terminal color for that chain.
    - Follow pointers until a terminal is reached, then fill the entire originating window with that terminal color while leaving grey separators unchanged.

    Key insight:
    - The marker color matters only for terminal windows; in non-terminal windows the informative signal is the marker's relative position inside the 3x3 window.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _sep_color, row_blocks, col_blocks, bg = _detect_window_lattice(g)
    markers = _parse_markers(g, row_blocks, col_blocks, bg)
    fill_colors = _resolve_terminal_colors(markers, (len(row_blocks), len(col_blocks)))
    return _paint_windows(g, row_blocks, col_blocks, fill_colors)


def _window_neighbors(cell: Window, shape: tuple[int, int]) -> list[Window]:
    rows, cols = shape
    neighbors: list[Window] = []
    for dr, dc in DIRS8:
        nr, nc = cell[0] + dr, cell[1] + dc
        if in_bounds((rows, cols), nr, nc):
            neighbors.append((nr, nc))
    return neighbors


def _build_pointer_forest(shape: tuple[int, int], terminal_count: int) -> tuple[dict[Window, Window | None], set[Window]]:
    rows, cols = shape
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    terminals = set(random.sample(cells, terminal_count))
    parents: dict[Window, Window | None] = {cell: None for cell in terminals}
    assigned = set(terminals)

    while len(assigned) < len(cells):
        frontier = [
            cell
            for cell in cells
            if cell not in assigned and any(neighbor in assigned for neighbor in _window_neighbors(cell, shape))
        ]
        if not frontier:
            raise AssertionError("window lattice is connected, so frontier should never be empty")
        child = random.choice(frontier)
        parent_choices = [neighbor for neighbor in _window_neighbors(child, shape) if neighbor in assigned]
        parents[child] = random.choice(parent_choices)
        assigned.add(child)

    return parents, terminals


def _terminal_fill_colors(parents: dict[Window, Window | None], terminal_colors: dict[Window, int]) -> dict[Window, int]:
    memo: dict[Window, int] = {}

    def resolve(cell: Window) -> int:
        if cell in memo:
            return memo[cell]
        parent = parents[cell]
        memo[cell] = terminal_colors[cell] if parent is None else resolve(parent)
        return memo[cell]

    return {cell: resolve(cell) for cell in parents}


def _pointer_depths(parents: dict[Window, Window | None]) -> dict[Window, int]:
    memo: dict[Window, int] = {}

    def depth(cell: Window) -> int:
        if cell in memo:
            return memo[cell]
        parent = parents[cell]
        memo[cell] = 0 if parent is None else 1 + depth(parent)
        return memo[cell]

    return {cell: depth(cell) for cell in parents}


def _edge_direction_stats(parents: dict[Window, Window | None]) -> tuple[int, int]:
    orthogonal = 0
    diagonal = 0
    for cell, parent in parents.items():
        if parent is None:
            continue
        dr = parent[0] - cell[0]
        dc = parent[1] - cell[1]
        if dr == 0 or dc == 0:
            orthogonal += 1
        else:
            diagonal += 1
    return orthogonal, diagonal


def _render_input_grid(
    shape: tuple[int, int],
    parents: dict[Window, Window | None],
    terminal_colors: dict[Window, int],
    resolved_colors: dict[Window, int],
) -> np.ndarray:
    rows, cols = shape
    grid = np.full((rows * SEPARATOR_STRIDE + 1, cols * SEPARATOR_STRIDE + 1), BACKGROUND_COLOR, dtype=int)
    grid[::SEPARATOR_STRIDE, :] = SEPARATOR_COLOR
    grid[:, ::SEPARATOR_STRIDE] = SEPARATOR_COLOR

    for cell, parent in parents.items():
        r0 = 1 + cell[0] * SEPARATOR_STRIDE
        c0 = 1 + cell[1] * SEPARATOR_STRIDE
        if parent is None:
            mr, mc = 1, 1
            marker_color = terminal_colors[cell]
        else:
            dr = parent[0] - cell[0]
            dc = parent[1] - cell[1]
            mr, mc = 1 + dr, 1 + dc
            color_choices = list(PALETTE)
            if random.random() < 0.75:
                filtered = [color for color in color_choices if color != resolved_colors[cell]]
                if filtered:
                    color_choices = filtered
            marker_color = random.choice(color_choices)
        grid[r0 + mr, c0 + mc] = int(marker_color)

    return grid


def _expected_output_from_forest(shape: tuple[int, int], resolved_colors: dict[Window, int]) -> np.ndarray:
    rows, cols = shape
    out = np.full((rows * SEPARATOR_STRIDE + 1, cols * SEPARATOR_STRIDE + 1), BACKGROUND_COLOR, dtype=int)
    out[::SEPARATOR_STRIDE, :] = SEPARATOR_COLOR
    out[:, ::SEPARATOR_STRIDE] = SEPARATOR_COLOR
    for (wr, wc), color in resolved_colors.items():
        r0 = 1 + wr * SEPARATOR_STRIDE
        c0 = 1 + wc * SEPARATOR_STRIDE
        out[r0 : r0 + WINDOW_SIZE, c0 : c0 + WINDOW_SIZE] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(2, 6)
    cols = random.randint(2, 6)
    cell_count = rows * cols

    min_terminals = 2 if cell_count >= 4 else 1
    max_terminals = min(4, cell_count - 1)
    terminal_count = random.randint(min_terminals, max_terminals)

    parents, terminals = _build_pointer_forest((rows, cols), terminal_count)
    depths = _pointer_depths(parents)
    if max(depths.values()) < 2:
        return None

    orthogonal, diagonal = _edge_direction_stats(parents)
    if cell_count >= 6 and (orthogonal == 0 or diagonal == 0):
        return None

    terminal_palette = random.sample(PALETTE, terminal_count)
    terminal_colors = {cell: color for cell, color in zip(sorted(terminals), terminal_palette)}
    resolved_colors = _terminal_fill_colors(parents, terminal_colors)
    if len(set(resolved_colors.values())) < 2:
        return None

    grid = _render_input_grid((rows, cols), parents, terminal_colors, resolved_colors)
    expected = _expected_output_from_forest((rows, cols), resolved_colors)

    row_blocks = [list(range(1 + r * SEPARATOR_STRIDE, 1 + r * SEPARATOR_STRIDE + WINDOW_SIZE)) for r in range(rows)]
    col_blocks = [list(range(1 + c * SEPARATOR_STRIDE, 1 + c * SEPARATOR_STRIDE + WINDOW_SIZE)) for c in range(cols)]
    parsed = _parse_markers(grid, row_blocks, col_blocks, BACKGROUND_COLOR)
    for cell, (offset, _color) in parsed.items():
        parent = parents[cell]
        expected_offset = (0, 0) if parent is None else (parent[0] - cell[0], parent[1] - cell[1])
        if offset != expected_offset:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid window-pointer-chain input")
