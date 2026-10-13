from __future__ import annotations

from collections import deque
import random

import numpy as np


class Color:
    """ARC palette ids copied in the same convention as arc_utils.py."""

    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    GREY = 5


DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
MIN_SIZE = 10
MAX_SIZE = 12
MIN_OBSTACLES = 3
MAX_OBSTACLES = 5
MIN_TURNS = 6
AREA_THRESHOLD = 4
MAX_ATTEMPTS = 500


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _blocked(grid: np.ndarray, r: int, c: int, dir_idx: int) -> bool:
    dr, dc = DIRS[dir_idx]
    nr, nc = r + dr, c + dc
    if not in_bounds(grid.shape, nr, nc):
        return True
    if int(grid[nr, nc]) in (Color.GREEN, Color.GREY):
        return True
    nr2, nc2 = r + 2 * dr, c + 2 * dc
    return in_bounds(grid.shape, nr2, nc2) and int(grid[nr2, nc2]) == Color.GREEN


def _spiral_state(grid: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...], int]:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    assert h == w, "expected a square grid"
    assert int(out[0, 0]) != Color.GREY, "top-left start must not be an obstacle"

    r = c = 0
    dir_idx = 0
    turn_count = 0
    path: list[tuple[int, int]] = []

    while True:
        assert int(out[r, c]) != Color.GREY
        out[r, c] = Color.GREEN
        path.append((r, c))

        moved = False
        for _ in range(4):
            if not _blocked(out, r, c, dir_idx):
                dr, dc = DIRS[dir_idx]
                r += dr
                c += dc
                moved = True
                break
            dir_idx = (dir_idx + 1) % 4
            turn_count += 1
        if not moved:
            break

    return out, tuple(path), turn_count


def _black_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != Color.BLACK:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_boundary = False
            touches_obstacle = False

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_boundary = True
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(grid.shape, nr, nc):
                        continue
                    val = int(grid[nr, nc])
                    if val == Color.GREY:
                        touches_obstacle = True
                    if val == Color.BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "area": len(coords),
                    "touches_boundary": touches_boundary,
                    "touches_obstacle": touches_obstacle,
                }
            )

    return comps


def _closed_regions(grid: np.ndarray) -> list[dict]:
    return [comp for comp in _black_components(grid) if not comp["touches_boundary"]]


def _recolor_regions(grid: np.ndarray, closed_regions: list[dict]) -> np.ndarray:
    out = grid.copy()
    for comp in closed_regions:
        color = Color.RED if comp["area"] >= AREA_THRESHOLD else Color.BLUE
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a clockwise green spiral around grey obstacles, then recolor enclosed black regions by area threshold.

    Input structure
    - The grid is a square black canvas with 3-5 single-cell grey obstacle pixels.
    - Obstacles are interior pixels; the top-left start cell remains free.
    - The background outside the spiral remains black.

    Rule
    - Starting at (0,0) and facing right, paint a green spiral one cell at a time.
    - Before each move, turn clockwise if the next cell would leave the grid, hit an existing green cell, hit a grey obstacle, or if the second cell ahead is already green; stop when all four directions fail.
    - Keep the grey obstacles unchanged, then find every 4-connected black region not connected to the outer border.
    - Fill each enclosed black region red when its area is at least 4, otherwise fill it blue; leave border-connected black cells unchanged.

    Key insight
    - The obstacles act as extra walls that deform the usual spiral, so the enclosed holes are no longer uniform and must be classified by their own areas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    spiral, _path, _turns = _spiral_state(g)
    closed_regions = _closed_regions(spiral)
    return _recolor_regions(spiral, closed_regions)


def _sample_obstacles(size: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _areas_signature(closed_regions: list[dict]) -> tuple[int, ...]:
    return tuple(sorted(int(comp["area"]) for comp in closed_regions))


def _scene_valid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        return False
    if not (MIN_SIZE <= g.shape[0] <= MAX_SIZE):
        return False
    if not np.all(np.isin(g, [Color.BLACK, Color.GREY])):
        return False
    if int(g[0, 0]) != Color.BLACK:
        return False

    obstacles = [tuple(map(int, rc)) for rc in np.argwhere(g == Color.GREY)]
    if not (MIN_OBSTACLES <= len(obstacles) <= MAX_OBSTACLES):
        return False
    if any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in obstacles):
        return False
    if any(max(abs(r1 - r2), abs(c1 - c2)) < 2 for i, (r1, c1) in enumerate(obstacles) for r2, c2 in obstacles[i + 1 :]):
        return False

    spiral, _path, turn_count = _spiral_state(g)
    closed_regions = _closed_regions(spiral)
    closed_areas = _areas_signature(closed_regions)
    if turn_count < MIN_TURNS:
        return False
    if len(closed_regions) < 2:
        return False
    if not any(area < AREA_THRESHOLD for area in closed_areas):
        return False
    if not any(area >= AREA_THRESHOLD for area in closed_areas):
        return False
    if not any(comp["touches_obstacle"] for comp in closed_regions):
        return False

    blank = np.full_like(g, Color.BLACK)
    blank_spiral, _blank_path, _blank_turns = _spiral_state(blank)
    blank_areas = _areas_signature(_closed_regions(blank_spiral))
    if np.array_equal(spiral == Color.GREEN, blank_spiral == Color.GREEN):
        return False
    if closed_areas == blank_areas:
        return False

    for r, c in obstacles:
        reduced = g.copy()
        reduced[r, c] = Color.BLACK
        reduced_spiral, _reduced_path, _reduced_turns = _spiral_state(reduced)
        reduced_areas = _areas_signature(_closed_regions(reduced_spiral))
        if np.array_equal(reduced_spiral == Color.GREEN, spiral == Color.GREEN) and reduced_areas == closed_areas:
            return False

    out = main(g)
    if not np.any(out == Color.RED) or not np.any(out == Color.BLUE):
        return False
    if not np.all(out[g == Color.GREY] == Color.GREY):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        size = random.randint(MIN_SIZE, MAX_SIZE)
        obstacle_count = random.randint(MIN_OBSTACLES, MAX_OBSTACLES)
        obstacles = _sample_obstacles(size, obstacle_count)
        if obstacles is None:
            continue

        grid = np.full((size, size), Color.BLACK, dtype=int)
        for r, c in obstacles:
            grid[r, c] = Color.GREY

        if _scene_valid(grid):
            return grid

    raise RuntimeError("failed to generate a valid spiral-obstacle threshold scene")
