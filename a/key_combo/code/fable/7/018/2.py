"""ARC task: color a walled floor by shortest-path seed ownership."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


FLOOR = 7
WALL = 5
TIE = 2
SEED_COLORS = (1, 3, 4, 6, 8, 9, 0)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = 10**9


def _bfs_dist(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Four-connected shortest-path distances inside ``passable``."""
    h, w = passable.shape
    dist = np.full((h, w), INF, dtype=int)
    sr, sc = start
    dist[sr, sc] = 0
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        nd = int(dist[r, c]) + 1
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and dist[rr, cc] == INF:
                dist[rr, cc] = nd
                queue.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color every orange floor cell by its nearest seed along the walled corridors, marking ties red.

    Input structure:
    - Orange cells form a four-connected system of corridors and rooms, while grey cells are walls.
    - At least two isolated, differently colored, non-red marker pixels replace floor cells; each marker
      color occurs exactly once.

    Rule:
    - For every orange floor cell, measure four-connected shortest-path distance through floor and seed
      cells to every marker. Use the uniquely nearest marker's color, or red when two or more tie.
    - Grey walls and the original marker pixels remain unchanged.

    Key insight:
    - Nearness is measured by walking around grey barriers, not by straight-line or Manhattan distance;
      red therefore traces equal-distance boundaries in the corridor metric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"

    seed_colors = [int(v) for v in np.unique(g) if int(v) not in {FLOOR, WALL}]
    assert TIE not in seed_colors, "red is reserved for equal-distance output cells"
    assert len(seed_colors) >= 2, "expected at least two differently colored seeds"

    seeds: list[tuple[int, tuple[int, int]]] = []
    for color in seed_colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 1, "each seed color must occur at exactly one marker pixel"
        seeds.append((color, tuple(map(int, coords[0]))))

    passable = g != WALL
    distances = np.stack([_bfs_dist(passable, pos) for _, pos in seeds])
    floor_mask = g == FLOOR
    assert np.all(np.min(distances[:, floor_mask], axis=0) < INF), "every floor cell must reach a seed"

    nearest = np.min(distances, axis=0)
    winner_count = np.sum(distances == nearest[None, :, :], axis=0)
    winner = np.argmin(distances, axis=0)

    out = g.copy()
    for index, (color, _) in enumerate(seeds):
        out[floor_mask & (winner_count == 1) & (winner == index)] = color
    out[floor_mask & (winner_count > 1)] = TIE
    return out


def _carve_maze(height: int, width: int) -> np.ndarray:
    """Make a connected maze, then open a few small rooms and loops."""
    grid = np.full((height, width), WALL, dtype=int)
    start = (random.randrange(1, height, 2), random.randrange(1, width, 2))
    grid[start] = FLOOR
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack[-1]
        choices = []
        for dr, dc in DIR4:
            rr, cc = r + 2 * dr, c + 2 * dc
            if 1 <= rr < height - 1 and 1 <= cc < width - 1 and (rr, cc) not in visited:
                choices.append((rr, cc, r + dr, c + dc))
        if not choices:
            stack.pop()
            continue
        rr, cc, wr, wc = random.choice(choices)
        grid[wr, wc] = grid[rr, cc] = FLOOR
        visited.add((rr, cc))
        stack.append((rr, cc))

    # A handful of openings creates cycles and visibly wider room-like places,
    # while retaining enough barriers for corridor distance to matter.
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if grid[r, c] == WALL
        and ((grid[r - 1, c] == FLOOR and grid[r + 1, c] == FLOOR)
             or (grid[r, c - 1] == FLOOR and grid[r, c + 1] == FLOOR))
    ]
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(1, max(1, len(candidates) // 14))]:
        grid[r, c] = FLOOR

    for _ in range(random.randint(1, 3)):
        r = random.randrange(1, height - 2)
        c = random.randrange(1, width - 2)
        if np.any(grid[r : r + 2, c : c + 2] == FLOOR):
            grid[r : r + 2, c : c + 2] = FLOOR
    return grid


def _semantic_candidate(maze: np.ndarray, seeds: list[tuple[int, int]]) -> bool:
    """Reject easy cases that do not demonstrate every stated mechanism."""
    passable = maze != WALL
    distances = np.stack([_bfs_dist(passable, p) for p in seeds])
    floor = maze == FLOOR
    flat = distances[:, floor]
    if flat.size == 0 or np.any(np.min(flat, axis=0) >= INF):
        return False

    best = np.min(flat, axis=0)
    wins = flat == best[None, :]
    ties = np.sum(wins, axis=0) > 1
    floor_count = int(floor.sum())
    if not (2 <= int(ties.sum()) <= max(8, floor_count // 4)):
        return False
    # Every seed must own a visible territory, so no marker is decorative.
    if any(int(np.sum(wins[i] & ~ties)) < 5 for i in range(len(seeds))):
        return False

    coords = np.argwhere(floor)
    manhattan = np.array(
        [[abs(int(r) - sr) + abs(int(c) - sc) for r, c in coords] for sr, sc in seeds],
        dtype=int,
    )
    geo_unique = np.sum(wins, axis=0) == 1
    man_best = np.min(manhattan, axis=0)
    man_unique = np.sum(manhattan == man_best[None, :], axis=0) == 1
    geo_owner = np.argmin(flat, axis=0)
    man_owner = np.argmin(manhattan, axis=0)
    # These cells are explicit counterexamples to the straight-line shortcut.
    detour_disagreements = geo_unique & man_unique & (geo_owner != man_owner)
    if int(detour_disagreements.sum()) < 2:
        return False

    # Keep markers separated and away from trivial local domination.
    pair_dist = [int(distances[i, seeds[j][0], seeds[j][1]])
                 for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    return bool(pair_dist and min(pair_dist) >= 6)


def generate_input() -> np.ndarray:
    """Generate varied, semantically non-degenerate corridor instances."""
    for _ in range(240):
        height = random.choice((11, 13, 15, 17, 19, 21))
        width = random.choice((11, 13, 15, 17, 19, 21))
        maze = _carve_maze(height, width)
        floor_cells = [tuple(map(int, p)) for p in np.argwhere(maze == FLOOR)]
        seed_count = random.choice((2, 2, 3, 3, 4))
        if len(floor_cells) < 8 * seed_count:
            continue

        for _ in range(180):
            seeds = random.sample(floor_cells, seed_count)
            if not _semantic_candidate(maze, seeds):
                continue
            colors = random.sample(SEED_COLORS, seed_count)
            grid = maze.copy()
            for (r, c), color in zip(seeds, colors):
                grid[r, c] = color

            # Independent structural checks complement the semantic oracle above.
            out = main(grid)
            if np.array_equal(out, grid):
                continue
            if not np.array_equal(out[grid == WALL], grid[grid == WALL]):
                continue
            if any(int(out[r, c]) != color for (r, c), color in zip(seeds, colors)):
                continue
            if not np.any(out == TIE):
                continue
            return grid
    raise RuntimeError("generate_input: could not build a non-degenerate corridor puzzle")

