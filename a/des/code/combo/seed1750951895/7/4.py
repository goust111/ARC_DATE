import random
from collections import deque

import numpy as np


BG = 7
GREY = 5
PAIR_COLORS = [1, 2, 3, 4, 6, 8, 9]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Connected odd-bbox shapes centered on (0, 0).
SHAPES = [
    [(0, 0)],
    [(0, -1), (0, 0), (0, 1)],
    [(-1, 0), (0, 0), (1, 0)],
    [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
    [(-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)],
    [(-1, 1), (-1, 0), (0, 0), (1, 0), (1, -1)],
]


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [r0 for r0, _ in cells]
            cols = [c0 for _, c0 in cells]
            comps.append(
                {
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _bbox_center(component: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _l_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    cells = []
    step_c = 0 if ac == bc else (1 if bc > ac else -1)
    c = ac
    while True:
        cells.append((ar, c))
        if c == bc:
            break
        c += step_c
    step_r = 0 if ar == br else (1 if br > ar else -1)
    r = ar + step_r
    while step_r != 0 and True:
        cells.append((r, bc))
        if r == br:
            break
        r += step_r
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-color pair using a solid or dashed L-path based on grey obstacles."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if color not in (BG, GREY)]
    assert colors

    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each foreground color must form exactly two objects"
        path = _l_path(_bbox_center(comps[0]), _bbox_center(comps[1]))
        blocked = any(g[r, c] == GREY for r, c in path)
        for idx, (r, c) in enumerate(path):
            if g[r, c] == GREY:
                continue
            if g[r, c] != BG:
                continue
            if blocked:
                if idx % 2 == 0:
                    out[r, c] = color
            else:
                out[r, c] = color
    return out


def _shape_cells(center: tuple[int, int], shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in shape]


def _expanded(cells: list[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _in_bounds(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _path_body(
    path: list[tuple[int, int]],
    obj1: set[tuple[int, int]],
    obj2: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [cell for cell in path if cell not in obj1 and cell not in obj2]


def _status_schedule(pair_count: int, rng: random.Random) -> list[bool]:
    blocked = [True] * (pair_count // 2) + [False] * (pair_count // 2)
    if pair_count % 2 == 1:
        blocked.append(bool(rng.randrange(2)))
    rng.shuffle(blocked)
    if pair_count == 1:
        blocked = [bool(rng.randrange(2))]
    return blocked


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 22)
    w = rng.randint(15, 22)
    grid = np.full((h, w), BG, dtype=int)

    pair_count = rng.randint(2, 4)
    colors = rng.sample(PAIR_COLORS, pair_count)
    blocked_flags = _status_schedule(pair_count, rng)

    occupied = set()
    reserved = set()

    for color, blocked in zip(colors, blocked_flags, strict=True):
        placed = False
        for _ in range(300):
            center1 = (rng.randint(2, h - 3), rng.randint(2, w - 3))
            center2 = (rng.randint(2, h - 3), rng.randint(2, w - 3))
            if abs(center1[0] - center2[0]) + abs(center1[1] - center2[1]) < 6:
                continue

            shape1 = rng.choice(SHAPES)
            shape2 = rng.choice(SHAPES)
            cells1 = _shape_cells(center1, shape1)
            cells2 = _shape_cells(center2, shape2)
            if not _in_bounds(cells1, h, w) or not _in_bounds(cells2, h, w):
                continue

            obj1 = set(cells1)
            obj2 = set(cells2)
            if obj1 & obj2:
                continue

            obj_reserved = _expanded(cells1 + cells2, h, w, radius=1)
            if obj_reserved & reserved:
                continue
            if obj1 & occupied or obj2 & occupied:
                continue

            path = _l_path(center1, center2)
            if len(path) < 5:
                continue
            body = _path_body(path, obj1, obj2)
            if len(body) < 3:
                continue
            if any(cell in occupied for cell in body):
                continue

            path_reserved = _expanded(body, h, w, radius=1)
            if path_reserved & reserved:
                continue

            obstacle_candidates = [cell for cell in body if cell not in obj_reserved]
            if blocked and not obstacle_candidates:
                continue

            obstacle_count = 0
            obstacles = []
            if blocked:
                obstacle_count = min(len(obstacle_candidates), rng.randint(1, 3))
                if obstacle_count == 0:
                    continue
                obstacles = rng.sample(obstacle_candidates, obstacle_count)

            for r, c in cells1:
                grid[r, c] = color
            for r, c in cells2:
                grid[r, c] = color
            for r, c in obstacles:
                grid[r, c] = GREY

            occupied.update(obj1)
            occupied.update(obj2)
            occupied.update(obstacles)
            reserved.update(obj_reserved)
            reserved.update(path_reserved)
            placed = True
            break
        if not placed:
            return None

    # Re-check the hidden assumptions directly against the generated grid.
    seen_blocked = []
    for color in colors:
        comps = _components_of_color(grid, color)
        if len(comps) != 2:
            return None
        path = _l_path(_bbox_center(comps[0]), _bbox_center(comps[1]))
        seen_blocked.append(any(grid[r, c] == GREY for r, c in path))

    if True not in seen_blocked or False not in seen_blocked:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
