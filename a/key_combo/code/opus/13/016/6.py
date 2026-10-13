from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Band = tuple[int, int]


def _prefix_run_length(row: np.ndarray, bg: int = 0) -> int:
    length = 0
    for value in row:
        if int(value) == bg:
            break
        length += 1
    return length


def _separator_color(grid: np.ndarray) -> int:
    row_colors = {
        int(grid[r, 0])
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != 0 and np.all(grid[r] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(grid.shape[1])
        if int(grid[0, c]) != 0 and np.all(grid[:, c] == grid[0, c])
    }
    shared = row_colors & col_colors
    assert len(shared) == 1, "expected one separator color shared by full rows and columns"
    return next(iter(shared))


def _bands(size: int, separators: list[int]) -> list[Band]:
    edges = [-1, *separators, size]
    bands = [(a + 1, b) for a, b in zip(edges, edges[1:])]
    assert all(start < end for start, end in bands), "invalid separator layout"
    lengths = {end - start for start, end in bands}
    assert len(lengths) == 1, "partition bands must be equal-sized"
    return bands


def _cell_foreground_color(cell: np.ndarray) -> int:
    colors = set(map(int, np.unique(cell))) - {0}
    assert len(colors) == 1, "each cell must use exactly one non-black foreground color"
    return next(iter(colors))


def _control_length(cell: np.ndarray) -> int:
    lengths = [_prefix_run_length(row) for row in cell]
    n = max(lengths)
    assert n > 0, "control cell must contain a left-aligned bar"

    rows_with_fg = [idx for idx, row in enumerate(cell) if np.any(row != 0)]
    assert rows_with_fg, "control cell must contain foreground pixels"
    assert len(rows_with_fg) == 1, "control cell must have exactly one bar row"
    row = cell[rows_with_fg[0]]
    assert np.all(row[:n] != 0) and np.all(row[n:] == 0), "bar must be contiguous and left-aligned"
    return n


def _sequence(color: int, n: int, bg: int = 0) -> list[int]:
    run_lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    seq: list[int] = []
    for i, run_len in enumerate(run_lengths):
        seq.extend([color] * run_len)
        if i != len(run_lengths) - 1:
            seq.append(bg)
    return seq


def _render_sequence(side: int, color: int, n: int) -> np.ndarray:
    seq = _sequence(color, n)
    assert len(seq) <= side * side, "sequence must fit inside the square cell"
    cell = np.zeros((side, side), dtype=int)
    cell.ravel()[: len(seq)] = np.asarray(seq, dtype=int)
    return cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each partition cell's control bar with a wrapped run-length sequence colored by the unique anomalous cell.

    Input structure:
    - A black grid is divided by complete 1-pixel separator rows and columns into equal square cells.
    - Exactly one cell is anomalous: all of its non-black pixels use a unique color that no other cell uses.
    - Every other cell contains a single-color horizontal control bar that starts at the cell's left edge.

    Rule:
    - Detect the separator lines and split the grid into its equal square cells.
    - Use the anomalous cell's unique foreground color as the output fill color, then clear that anomalous cell to black.
    - For each normal cell, read the control-bar length n and replace the bar with the 1D sequence of run lengths
      1, 2, ..., n, ..., 2, 1 separated by single black pixels, wrapped row-major inside the cell.
    - Keep the separator lines unchanged.

    Key insight:
    - The anomalous cell contributes only a global color label, while each ordinary cell contributes only a local length parameter; solving the task requires combining those two independent roles after partitioning the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_color(g)
    row_separators = [r for r in range(g.shape[0]) if np.all(g[r] == sep)]
    col_separators = [c for c in range(g.shape[1]) if np.all(g[:, c] == sep)]
    row_bands = _bands(g.shape[0], row_separators)
    col_bands = _bands(g.shape[1], col_separators)
    assert len(row_bands) == len(col_bands) >= 2, "expected a square partition"

    side = row_bands[0][1] - row_bands[0][0]
    assert all((r1 - r0) == side for r0, r1 in row_bands)
    assert all((c1 - c0) == side for c0, c1 in col_bands)

    cells: list[tuple[int, int, np.ndarray]] = []
    colors: list[int] = []
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            cell = g[r0:r1, c0:c1]
            color = _cell_foreground_color(cell)
            cells.append((br, bc, cell))
            colors.append(color)

    counts = Counter(colors)
    anomalous_colors = [color for color, count in counts.items() if count == 1]
    assert len(anomalous_colors) == 1, "expected exactly one anomalous color source"
    out_color = anomalous_colors[0]

    out = g.copy()
    for br, bc, cell in cells:
        r0, r1 = row_bands[br]
        c0, c1 = col_bands[bc]
        if _cell_foreground_color(cell) == out_color:
            out[r0:r1, c0:c1] = 0
            continue
        n = _control_length(cell)
        out[r0:r1, c0:c1] = _render_sequence(side, out_color, n)

    return out


_SEPARATOR = 5
_PALETTE = tuple(color for color in range(1, 10) if color != _SEPARATOR)
_SIDE_RANGE = (4, 8)
_GRID_BANDS = 3
_MAX_ATTEMPTS = 300


def _anomaly_shape(side: int, pattern_seed: int) -> set[Coord]:
    branch = ("rectangle", "diag", "lshape", "scatter")[pattern_seed % 4]
    coords: set[Coord] = set()

    if branch == "rectangle":
        h = max(1, side // 2)
        w = max(1, (side + 1) // 2)
        r0 = 0
        c0 = side - w
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                coords.add((r, c))
    elif branch == "diag":
        length = side
        r0 = 0
        c0 = 0
        for k in range(length):
            coords.add((r0 + k, c0 + k))
        if side >= 4:
            for k in range(length):
                coords.add((r0 + k, c0 + length - 1 - k))
    elif branch == "lshape":
        r0 = 1 if side >= 5 else 0
        c0 = 0
        vr = side - r0
        hc = max(2, side - 1)
        for r in range(r0, r0 + vr):
            coords.add((r, c0))
        for c in range(c0, c0 + hc):
            coords.add((r0 + vr - 1, c))
    else:
        for k in range(side):
            coords.add((k, (2 * k + 1) % side))
        coords.add((side // 2, side // 2))

    assert coords, "anomaly shape must be non-empty"
    return coords


def _place_control_bar(cell: np.ndarray, color: int, length: int, row: int) -> None:
    cell[row, :length] = color


def _sample_lengths(side: int, rng: random.Random) -> list[int]:
    max_len = side - 1
    base = [1, max_len]
    while len(base) < 8:
        base.append(rng.randint(1, max_len))
    rng.shuffle(base)
    return base


def _build_input(rng: random.Random) -> np.ndarray:
    side = rng.randint(*_SIDE_RANGE)
    size = _GRID_BANDS * side + (_GRID_BANDS - 1)
    grid = np.zeros((size, size), dtype=int)

    for split in range(side, size, side + 1):
        grid[split, :] = _SEPARATOR
        grid[:, split] = _SEPARATOR

    colors = list(_PALETTE)
    control_color = rng.choice(colors)
    colors.remove(control_color)
    anomaly_color = rng.choice(colors)

    anomaly_idx = rng.randrange(_GRID_BANDS * _GRID_BANDS)
    lengths = _sample_lengths(side, rng)
    length_iter = iter(lengths)

    for idx in range(_GRID_BANDS * _GRID_BANDS):
        br, bc = divmod(idx, _GRID_BANDS)
        r0 = br * (side + 1)
        c0 = bc * (side + 1)
        cell = grid[r0 : r0 + side, c0 : c0 + side]
        if idx == anomaly_idx:
            pattern_seed = anomaly_idx + anomaly_color + side
            for r, c in _anomaly_shape(side, pattern_seed):
                cell[r, c] = anomaly_color
        else:
            length = next(length_iter)
            row = (idx + length) % side
            _place_control_bar(cell, control_color, length, row)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_ATTEMPTS):
        grid = _build_input(rng)
        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, grid):
            continue

        side = (grid.shape[0] - (_GRID_BANDS - 1)) // _GRID_BANDS
        if side < 4 or max(grid.shape) > 30 or max(out.shape) > 30:
            continue

        unique_lengths = set()
        sep = _separator_color(grid)
        row_bands = _bands(grid.shape[0], [r for r in range(grid.shape[0]) if np.all(grid[r] == sep)])
        col_bands = _bands(grid.shape[1], [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)])
        colors = []
        for r0, r1 in row_bands:
            for c0, c1 in col_bands:
                cell = grid[r0:r1, c0:c1]
                colors.append(_cell_foreground_color(cell))
        anomaly_color = [color for color, count in Counter(colors).items() if count == 1][0]
        for r0, r1 in row_bands:
            for c0, c1 in col_bands:
                cell = grid[r0:r1, c0:c1]
                if _cell_foreground_color(cell) != anomaly_color:
                    unique_lengths.add(_control_length(cell))
        if len(unique_lengths) < min(3, side - 1):
            continue

        return grid
    raise RuntimeError("failed to generate valid input")
