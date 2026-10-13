from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PASSAGE = BLACK
GATE_COLORS = (BLUE, YELLOW, MAGENTA, ORANGE, AZURE, MAROON)
RESERVED_COLORS = {BLACK, RED, GREEN, GREY}

GRID_MIN = 15
GRID_MAX = 29
GEN_RETRIES = 300
PORT_PAIR_TRIES = 64
GATE_COUNTS = (2, 3, 4)
SEGMENT_MARGIN = 3
GATE_SPACING = 3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
OPPOSITE_PAIRS = (
    ("top", "bottom"),
    ("bottom", "top"),
    ("left", "right"),
    ("right", "left"),
)
ADJACENT_PAIRS = tuple(
    (a, b)
    for a in SIDES
    for b in SIDES
    if a != b and (a, b) not in OPPOSITE_PAIRS
)


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _find_unique_cell(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == color)
    assert len(pts) == 1, f"expected exactly one cell of color {color}"
    return int(pts[0, 0]), int(pts[0, 1])


def _shortest_path(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = passable.shape
    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        r, c = cur
        for nr, nc in _neighbors4(r, c, h, w):
            if not passable[nr, nc] or (nr, nc) in parent:
                continue
            parent[(nr, nc)] = cur
            q.append((nr, nc))
    assert end in parent, "entrance and exit must be connected by passages"

    path_rev = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path_rev.append(cur)
        cur = parent[cur]
    path_rev.reverse()
    return path_rev


def _passable_degree(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    return sum(1 for nr, nc in _neighbors4(r, c, h, w) if grid[nr, nc] != GREY)


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    gate_indices = [
        idx
        for idx, (r, c) in enumerate(path)
        if int(grid[r, c]) not in RESERVED_COLORS
    ]

    if not gate_indices:
        for r, c in path[:-1]:
            out[r, c] = GREEN
        er, ec = path[-1]
        out[er, ec] = RED
        return out

    segment_start = 0
    segment_color = GREEN
    for gate_idx in gate_indices[:-1]:
        for i in range(segment_start, gate_idx):
            r, c = path[i]
            out[r, c] = segment_color
        gr, gc = path[gate_idx]
        segment_start = gate_idx + 1
        segment_color = int(grid[gr, gc])

    last_gate_idx = gate_indices[-1]
    for i in range(segment_start, last_gate_idx):
        r, c = path[i]
        out[r, c] = segment_color
    for i in range(last_gate_idx + 1, len(path)):
        r, c = path[i]
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the unique maze path from the green entrance to the red exit and
    recolor each path segment according to the most recently crossed gate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start = _find_unique_cell(g, GREEN)
    end = _find_unique_cell(g, RED)
    h, w = g.shape
    assert _is_border(*start, h, w), "entrance must be on the boundary"
    assert _is_border(*end, h, w), "exit must be on the boundary"
    assert start != end, "entrance and exit must be different cells"

    path = _shortest_path(g != GREY, start, end)
    return _paint_path(g, path)


def _random_odd(lo: int, hi: int) -> int:
    choices = [x for x in range(lo, hi + 1) if x % 2 == 1]
    return random.choice(choices)


def _carve_maze(height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), GREY, dtype=int)
    cell_h = (height - 1) // 2
    cell_w = (width - 1) // 2
    for rr in range(cell_h):
        for cc in range(cell_w):
            grid[2 * rr + 1, 2 * cc + 1] = PASSAGE

    start = (random.randrange(cell_h), random.randrange(cell_w))
    visited = {start}
    stack = [start]
    while stack:
        cr, cc = stack[-1]
        nxt = []
        for dr, dc in DIRS4:
            nr = cr + dr
            nc = cc + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and (nr, nc) not in visited:
                nxt.append((nr, nc, dr, dc))
        if not nxt:
            stack.pop()
            continue
        nr, nc, dr, dc = random.choice(nxt)
        wall_r = 2 * cr + 1 + dr
        wall_c = 2 * cc + 1 + dc
        grid[wall_r, wall_c] = PASSAGE
        visited.add((nr, nc))
        stack.append((nr, nc))
    return grid


def _side_ports(height: int, width: int, side: str):
    if side == "top":
        return [((0, c), (1, c)) for c in range(1, width, 2)]
    if side == "bottom":
        return [((height - 1, c), (height - 2, c)) for c in range(1, width, 2)]
    if side == "left":
        return [((r, 0), (r, 1)) for r in range(1, height, 2)]
    if side == "right":
        return [((r, width - 1), (r, width - 2)) for r in range(1, height, 2)]
    raise ValueError(f"unknown side {side}")


def _gate_candidates(grid: np.ndarray, path: list[tuple[int, int]]) -> list[int]:
    hi = len(path) - SEGMENT_MARGIN
    candidates = []
    for idx in range(SEGMENT_MARGIN, hi):
        r, c = path[idx]
        if grid[r, c] != PASSAGE:
            continue
        if _passable_degree(grid, r, c) != 2:
            continue
        candidates.append(idx)
    return candidates


def _pick_gate_indices(candidates: list[int], gate_count: int) -> list[int] | None:
    if len(candidates) < gate_count:
        return None
    candidates = sorted(candidates)

    for _ in range(200):
        picks = sorted(random.sample(candidates, gate_count))
        if all(b - a >= GATE_SPACING for a, b in zip(picks, picks[1:])):
            return picks

    picks: list[int] = []

    def backtrack(start_idx: int, last_pick: int | None) -> bool:
        if len(picks) == gate_count:
            return True
        choices = list(range(start_idx, len(candidates)))
        random.shuffle(choices)
        for idx in choices:
            pos = candidates[idx]
            if last_pick is not None and pos - last_pick < GATE_SPACING:
                continue
            picks.append(pos)
            if backtrack(idx + 1, pos):
                return True
            picks.pop()
        return False

    if backtrack(0, None):
        return picks.copy()
    return None


def _solve_with_path(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    start = _find_unique_cell(grid, GREEN)
    end = _find_unique_cell(grid, RED)
    path = _shortest_path(grid != GREY, start, end)
    return _paint_path(grid, path), path


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    height = _random_odd(GRID_MIN, GRID_MAX)
    width = _random_odd(GRID_MIN, GRID_MAX)
    maze = _carve_maze(height, width)

    pair_pool = OPPOSITE_PAIRS if random.random() < 0.5 else ADJACENT_PAIRS
    start_side, end_side = random.choice(pair_pool)
    desired_gate_count = random.choice(GATE_COUNTS)

    port_pairs = [
        (a, b)
        for a in _side_ports(height, width, start_side)
        for b in _side_ports(height, width, end_side)
    ]
    random.shuffle(port_pairs)

    for (start_border, _start_inner), (end_border, _end_inner) in port_pairs[:PORT_PAIR_TRIES]:
        inp = maze.copy()
        sr, sc = start_border
        er, ec = end_border
        inp[sr, sc] = GREEN
        inp[er, ec] = RED

        try:
            out, path = _solve_with_path(inp)
        except AssertionError:
            continue

        gate_candidates = _gate_candidates(inp, path)
        gate_indices = _pick_gate_indices(gate_candidates, desired_gate_count)
        if gate_indices is None:
            continue

        gate_colors = random.sample(GATE_COLORS, desired_gate_count)
        for gate_idx, gate_color in zip(gate_indices, gate_colors):
            gr, gc = path[gate_idx]
            inp[gr, gc] = gate_color

        expected = _paint_path(inp, path)
        if np.array_equal(main(inp), expected):
            return inp, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _sample_scene()
        if result is not None:
            inp, _expected = result
            return inp
    raise RuntimeError("failed to generate a valid maze recoloring input")
