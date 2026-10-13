from __future__ import annotations

import random

import numpy as np


LEFT = {(-1, 0): (0, -1), (0, -1): (1, 0), (1, 0): (0, 1), (0, 1): (-1, 0)}
RIGHT = {v: k for k, v in LEFT.items()}


def _read_cycle(grid: np.ndarray) -> list[int]:
    cells = [int(grid[0, 0]), int(grid[0, 1]), int(grid[1, 0]), int(grid[1, 1])]
    if cells[2] != 0 or cells[3] != 0:
        return [c for c in cells if c != 0]
    return [c for c in cells[:2] if c != 0]


def _direction(green: tuple[int, int], yellow: tuple[int, int]) -> tuple[int, int]:
    return yellow[0] - green[0], yellow[1] - green[1]


def main(grid: np.ndarray) -> np.ndarray:
    """Simulate the alternating-turn particle, count cycle-colored trail pixels, and return a descending bar chart."""
    cycle = _read_cycle(grid)
    if not cycle:
        return np.zeros((0, 0), dtype=grid.dtype)
    work = np.array(grid, copy=True)
    work[:2, :2] = 0
    greens = np.argwhere(work == 3)
    yellows = np.argwhere(work == 4)
    if len(greens) == 0 or len(yellows) == 0:
        return np.zeros((1, len(cycle)), dtype=grid.dtype)
    pos = (int(greens[0, 0]), int(greens[0, 1]))
    y = (int(yellows[0, 0]), int(yellows[0, 1]))
    direction = _direction(pos, y)
    work[pos] = 0
    work[y] = 0

    painted: set[tuple[int, int]] = set()
    collisions = 0
    idx = 0
    guard = 0
    while guard < grid.size * 4:
        color = cycle[idx % len(cycle)]
        work[pos] = color
        painted.add(pos)
        idx += 1
        nr, nc = pos[0] + direction[0], pos[1] + direction[1]
        blocked = not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or work[nr, nc] == 5
        if blocked:
            collisions += 1
            work[pos] = 5
            direction = LEFT[direction] if collisions % 2 == 1 else RIGHT[direction]
            nr, nc = pos[0] + direction[0], pos[1] + direction[1]
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or work[nr, nc] == 5 or (nr, nc) in painted:
                break
        elif (nr, nc) in painted:
            break
        pos = (nr, nc)
        guard += 1

    counts = [(int(np.sum(work == color)), color) for color in cycle]
    counts.sort(key=lambda item: (-item[0], item[1]))
    height = max([count for count, _ in counts] + [1])
    out = np.zeros((height, len(counts)), dtype=grid.dtype)
    for c, (count, color) in enumerate(counts):
        out[:count, c] = color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((15, 15), dtype=int)
    grid[0, 0], grid[0, 1] = random.sample([1, 2, 6, 7], 2)
    obstacles = [(4, 7), (4, 8), (4, 9), (8, 9), (9, 9), (10, 9), (10, 5), (10, 6), (6, 5), (7, 5)]
    if random.random() < 0.5:
        obstacles += [(5, 11), (6, 11)]
    for r, c in obstacles:
        grid[r, c] = 5
    start = random.choice([(7, 7, 0, 1), (8, 7, 0, 1), (7, 6, 1, 0), (9, 6, -1, 0), (6, 6, 0, 1)])
    r, c, dr, dc = start
    grid[r, c] = 3
    grid[r + dr, c + dc] = 4
    return grid
