from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
MAROON = 9

INNER_PALETTE = (GREEN, YELLOW, PINK, ORANGE, MAROON)
GRID_MIN = 14
GRID_MAX = 18
MAX_GENERATION_ATTEMPTS = 500
STEP_CHOICES = (4, 5)
CONTROL_COLS = (1, 3, 5, 7, 9, 11)

Coord = tuple[int, int]
Square = dict[str, int | Coord]

TEMPLATES: tuple[tuple[Coord, ...], ...] = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
)


def _frame_border_ok(block: np.ndarray) -> bool:
    return bool(
        np.all(block[0, :] == GREY)
        and np.all(block[2, :] == GREY)
        and np.all(block[:, 0] == GREY)
        and np.all(block[:, 2] == GREY)
    )


def _find_squares(grid: np.ndarray) -> list[Square]:
    h, w = grid.shape
    squares: list[Square] = []
    for r0 in range(h - 4):
        for c0 in range(w - 2):
            block = grid[r0 : r0 + 3, c0 : c0 + 3]
            center = int(block[1, 1])
            if center in (BACKGROUND, GREY, BLUE, RED):
                continue
            if not _frame_border_ok(block):
                continue
            squares.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + 2,
                    "c1": c0 + 2,
                    "center_r": r0 + 1,
                    "center_c": c0 + 1,
                    "color": center,
                }
            )
    return squares


def _bridge_cells(a: Square, b: Square) -> list[Coord]:
    ar0, ac0, ar1, ac1 = int(a["r0"]), int(a["c0"]), int(a["r1"]), int(a["c1"])
    br0, bc0, br1, bc1 = int(b["r0"]), int(b["c0"]), int(b["r1"]), int(b["c1"])

    row_lo = max(ar0, br0)
    row_hi = min(ar1, br1)
    col_lo = max(ac0, bc0)
    col_hi = min(ac1, bc1)

    horiz_cells: list[Coord] = []
    vert_cells: list[Coord] = []

    if row_lo <= row_hi:
        if ac1 < bc0:
            gap0, gap1 = ac1 + 1, bc0 - 1
        elif bc1 < ac0:
            gap0, gap1 = bc1 + 1, ac0 - 1
        else:
            gap0, gap1 = 1, 0
        if gap0 <= gap1:
            row = (row_lo + row_hi) // 2
            horiz_cells = [(row, c) for c in range(gap0, gap1 + 1)]

    if col_lo <= col_hi:
        if ar1 < br0:
            gap0, gap1 = ar1 + 1, br0 - 1
        elif br1 < ar0:
            gap0, gap1 = br1 + 1, ar0 - 1
        else:
            gap0, gap1 = 1, 0
        if gap0 <= gap1:
            col = (col_lo + col_hi) // 2
            vert_cells = [(r, col) for r in range(gap0, gap1 + 1)]

    if horiz_cells and vert_cells:
        return horiz_cells if len(horiz_cells) <= len(vert_cells) else vert_cells
    if horiz_cells:
        return horiz_cells
    if vert_cells:
        return vert_cells
    raise ValueError("consecutive squares must be bridge-connectable")


def _bridge_layer(sequence: list[int], by_color: dict[int, Square]) -> tuple[set[Coord], int, int]:
    cells: set[Coord] = set()
    horiz = 0
    vert = 0
    for ca, cb in zip(sequence, sequence[1:]):
        segment = _bridge_cells(by_color[ca], by_color[cb])
        if not segment:
            continue
        if segment[0][0] == segment[-1][0]:
            horiz += 1
        else:
            vert += 1
        cells.update(segment)
    return cells, horiz, vert


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the red-bridge and blue-bridge networks induced by two control sequences on the same framed squares.

    Input structure:
    - The grid is a black background containing several non-overlapping 3x3 framed squares.
    - Every square has a grey(5) border and a unique center color chosen from the non-control palette.
    - The last two rows contain two left-to-right control sequences, each listing all square-center colors once.

    Rule:
    - Read the penultimate row as the first control sequence and connect each consecutive color pair with a red(2) bridge between the matching squares.
    - Read the last row as the second control sequence and build blue(1) bridges in the same way.
    - A bridge fills only the blank gap between two aligned squares, using one-pixel width; if horizontal and vertical alignments are both available, use the shorter gap.
    - Keep the framed squares unchanged, erase both control rows to black(0), then color red-only bridge cells red(2), blue-only bridge cells blue(1), and overlapping bridge cells yellow(4).

    Key insight:
    - The output comes from comparing two dynamically constructed bridge layers, so yellow cells are created by bridge overlap rather than by any static input color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    assert h >= 3, "expected space for squares and two control rows"

    squares = _find_squares(g)
    assert len(squares) >= 2, "expected at least two framed squares"
    by_color = {int(sq["color"]): sq for sq in squares}
    assert len(by_color) == len(squares), "square colors must be unique"

    colors = set(by_color)
    seq_red = [int(v) for v in g[h - 2] if int(v) in colors]
    seq_blue = [int(v) for v in g[h - 1] if int(v) in colors]
    assert set(seq_red) == colors and len(seq_red) == len(colors), "invalid first control sequence"
    assert set(seq_blue) == colors and len(seq_blue) == len(colors), "invalid second control sequence"

    red_cells, _, _ = _bridge_layer(seq_red, by_color)
    blue_cells, _, _ = _bridge_layer(seq_blue, by_color)

    out = g.copy()
    out[h - 2 :, :] = BACKGROUND
    for r, c in red_cells - blue_cells:
        out[r, c] = RED
    for r, c in blue_cells - red_cells:
        out[r, c] = BLUE
    for r, c in red_cells & blue_cells:
        out[r, c] = YELLOW
    return out


def _adjacency(nodes: tuple[Coord, ...]) -> dict[int, list[int]]:
    idx = {pos: i for i, pos in enumerate(nodes)}
    adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for i, (r, c) in enumerate(nodes):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            pos = (r + dr, c + dc)
            if pos in idx:
                adj[i].append(idx[pos])
    return adj


def _hamiltonian_paths(adj: dict[int, list[int]]) -> list[tuple[int, ...]]:
    n = len(adj)
    seen: set[tuple[int, ...]] = set()
    paths: list[tuple[int, ...]] = []

    def dfs(path: list[int], used: set[int]) -> None:
        if len(path) == n:
            key = min(tuple(path), tuple(reversed(path)))
            if key not in seen:
                seen.add(key)
                paths.append(key)
            return
        for nxt in adj[path[-1]]:
            if nxt in used:
                continue
            used.add(nxt)
            path.append(nxt)
            dfs(path, used)
            path.pop()
            used.remove(nxt)

    for start in range(n):
        dfs([start], {start})
    return paths


def _local_to_actual(nodes: tuple[Coord, ...], h: int, w: int) -> dict[int, Coord] | None:
    row_count = 1 + max(r for r, _ in nodes)
    col_count = 1 + max(c for _, c in nodes)

    feasible_row_steps = [step for step in STEP_CHOICES if 2 + (row_count - 1) * step <= h - 4]
    feasible_col_steps = [step for step in STEP_CHOICES if 2 + (col_count - 1) * step <= w - 2]
    if not feasible_row_steps or not feasible_col_steps:
        return None

    row_step = random.choice(feasible_row_steps)
    col_step = random.choice(feasible_col_steps)
    start_r_max = h - 4 - (row_count - 1) * row_step
    start_c_max = w - 2 - (col_count - 1) * col_step
    if start_r_max < 2 or start_c_max < 2:
        return None

    start_r = random.randint(2, start_r_max)
    start_c = random.randint(2, start_c_max)
    return {
        i: (start_r + r * row_step, start_c + c * col_step)
        for i, (r, c) in enumerate(nodes)
    }


def _squares_from_centers(centers: dict[int, Coord], colors: list[int]) -> list[Square]:
    squares: list[Square] = []
    for i, color in enumerate(colors):
        cr, cc = centers[i]
        squares.append(
            {
                "r0": cr - 1,
                "c0": cc - 1,
                "r1": cr + 1,
                "c1": cc + 1,
                "center_r": cr,
                "center_c": cc,
                "color": color,
            }
        )
    return squares


def _ordered_pair_candidates(paths: list[tuple[int, ...]], by_index: dict[int, Square]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for p1 in paths:
        seq1 = list(p1)
        cells1, h1, v1 = _bridge_layer(seq1, by_index)
        for p2 in paths:
            if p1 == p2:
                continue
            seq2 = list(p2)
            cells2, h2, v2 = _bridge_layer(seq2, by_index)
            inter = cells1 & cells2
            if not inter:
                continue
            if not (cells1 - cells2):
                continue
            if not (cells2 - cells1):
                continue
            if h1 + h2 == 0 or v1 + v2 == 0:
                continue
            candidates.append((p1, p2))
    return candidates


def _paint_square(grid: np.ndarray, square: Square) -> None:
    r0, c0, r1, c1 = int(square["r0"]), int(square["c0"]), int(square["r1"]), int(square["c1"])
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    grid[int(square["center_r"]), int(square["center_c"])] = int(square["color"])


def _validate_input(grid: np.ndarray) -> None:
    squares = _find_squares(grid)
    assert 4 <= len(squares) <= 5
    by_color = {int(sq["color"]): sq for sq in squares}
    assert len(by_color) == len(squares)

    colors = set(by_color)
    seq_red = [int(v) for v in grid[-2] if int(v) in colors]
    seq_blue = [int(v) for v in grid[-1] if int(v) in colors]
    assert set(seq_red) == colors and len(seq_red) == len(colors)
    assert set(seq_blue) == colors and len(seq_blue) == len(colors)
    assert seq_red != seq_blue

    red_cells, rh, rv = _bridge_layer(seq_red, by_color)
    blue_cells, bh, bv = _bridge_layer(seq_blue, by_color)
    assert red_cells and blue_cells
    assert red_cells != blue_cells
    assert red_cells & blue_cells
    assert red_cells - blue_cells
    assert blue_cells - red_cells
    assert rh + bh > 0 and rv + bv > 0

    occupied = np.zeros(grid.shape, dtype=bool)
    for sq in squares:
        occupied[int(sq["r0"]) : int(sq["r1"]) + 1, int(sq["c0"]) : int(sq["c1"]) + 1] = True
    for r, c in red_cells | blue_cells:
        assert not occupied[r, c]
        assert r < grid.shape[0] - 2

    out = main(grid)
    assert np.all(out[-2:, :] == BACKGROUND)
    assert int(np.count_nonzero(out == RED)) > 0
    assert int(np.count_nonzero(out == BLUE)) > 0
    assert int(np.count_nonzero(out == YELLOW)) > 0
    for sq in squares:
        cr, cc = int(sq["center_r"]), int(sq["center_c"])
        assert out[cr, cc] == int(sq["color"])


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    template = random.choice(TEMPLATES)
    if len(template) > len(INNER_PALETTE):
        return None

    centers = _local_to_actual(template, h, w)
    if centers is None:
        return None

    colors = random.sample(list(INNER_PALETTE), len(template))
    squares = _squares_from_centers(centers, colors)
    by_index = {i: sq for i, sq in enumerate(squares)}

    paths = _hamiltonian_paths(_adjacency(template))
    pairs = _ordered_pair_candidates(paths, by_index)
    if not pairs:
        return None
    path_red, path_blue = random.choice(pairs)

    seq_red = [colors[i] for i in path_red]
    seq_blue = [colors[i] for i in path_blue]
    if seq_red == seq_blue:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    for sq in squares:
        _paint_square(grid, sq)

    ctrl_cols = CONTROL_COLS[: len(colors)]
    if ctrl_cols[-1] >= w:
        return None
    for col, color in zip(ctrl_cols, seq_red):
        grid[h - 2, col] = color
    for col, color in zip(ctrl_cols, seq_blue):
        grid[h - 1, col] = color

    try:
        _validate_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-sequence bridge puzzle")
