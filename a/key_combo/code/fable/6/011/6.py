from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
ARC_COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _palette_from_bottom_row(grid: np.ndarray, bg: int) -> list[int]:
    palette = [int(value) for value in grid[-1] if int(value) != bg]
    if not palette:
        raise ValueError("palette row is empty")
    if len(set(palette)) != len(palette):
        raise ValueError("palette colors must be distinct")
    return palette


def _point_positions(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    points = np.argwhere(grid[:-1] != bg)
    return [(int(r), int(c)) for r, c in points]


def _build_graph(grid: np.ndarray, points: list[tuple[int, int]], bg: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    point_set = set(points)
    graph = {pos: [] for pos in points}
    for r, c in points:
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr = r + dr
            nc = c + dc
            mid_r = r + dr // 2
            mid_c = c + dc // 2
            neighbor = (nr, nc)
            if neighbor not in point_set:
                continue
            if not (0 <= nr < grid.shape[0] - 1 and 0 <= nc < grid.shape[1]):
                continue
            if int(grid[mid_r, mid_c]) != bg:
                continue
            graph[(r, c)].append(neighbor)
    return graph


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey lattice points by graph distance from the unique colored seed.

    The bottom row is a cyclic palette. Above it, every non-background cell is a
    lattice point. Two points are adjacent iff they are separated by exactly one
    background cell horizontally or vertically.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _background_color(grid)
    palette = _palette_from_bottom_row(grid, bg)
    points = _point_positions(grid, bg)
    if not points:
        return grid.copy()

    seed_positions = [(r, c) for r, c in points if int(grid[r, c]) != GREY]
    if len(seed_positions) != 1:
        raise ValueError("expected exactly one colored seed point")
    seed = seed_positions[0]
    seed_color = int(grid[seed])
    if seed_color not in palette:
        raise ValueError("seed color must appear in the palette")

    graph = _build_graph(grid, points, bg)
    start_phase = palette.index(seed_color)

    distances = {seed: 0}
    queue: deque[tuple[int, int]] = deque([seed])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt in distances:
                continue
            distances[nxt] = distances[node] + 1
            queue.append(nxt)

    out = grid.copy()
    for pos, dist in distances.items():
        if pos == seed:
            continue
        if int(grid[pos]) != GREY:
            continue
        out[pos] = palette[(start_phase + dist) % len(palette)]
    return out


def _logical_neighbors(pos: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = pos
    neighbors: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors


def _connected_shape(rows: int, cols: int, count: int) -> set[tuple[int, int]]:
    start = (random.randrange(rows), random.randrange(cols))
    shape = {start}
    frontier = set(_logical_neighbors(start, rows, cols))
    while len(shape) < count and frontier:
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        shape.add(nxt)
        for neighbor in _logical_neighbors(nxt, rows, cols):
            if neighbor not in shape:
                frontier.add(neighbor)
    return shape


def _generate_case() -> np.ndarray | None:
    rows = random.randint(3, 8)
    cols = random.randint(3, 8)
    top_margin = random.randint(0, 2)
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    main_height = top_margin + (2 * rows - 1) + bottom_margin

    bg = random.choice([color for color in ARC_COLORS if color != GREY])
    usable_palette = [color for color in ARC_COLORS if color not in {bg, GREY}]
    palette_len = random.randint(3, min(7, len(usable_palette)))
    palette = random.sample(usable_palette, k=palette_len)

    main_width = left_margin + (2 * cols - 1) + right_margin
    width = max(main_width, palette_len + random.randint(0, 4))
    height = main_height + 1
    if height > 30 or width > 30:
        return None

    total_sites = rows * cols
    min_count = max(4, total_sites // 3)
    count = random.randint(min_count, total_sites)
    shape = _connected_shape(rows, cols, count)
    if len(shape) < 2:
        return None

    grid = np.full((height, width), bg, dtype=int)
    seed_logical = random.choice(tuple(shape))
    seed_color = random.choice(palette)

    for lr, lc in shape:
        pr = top_margin + 2 * lr
        pc = left_margin + 2 * lc
        grid[pr, pc] = GREY
    seed_r = top_margin + 2 * seed_logical[0]
    seed_c = left_margin + 2 * seed_logical[1]
    grid[seed_r, seed_c] = seed_color

    palette_start = random.randint(0, width - palette_len)
    grid[-1, palette_start : palette_start + palette_len] = np.array(palette, dtype=int)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
