from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
DIRS = ("U", "R", "D", "L")
VEC = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
TIP_OFFSETS = {
    "U": (-2, 0),
    "R": (0, 2),
    "D": (2, 0),
    "L": (0, -2),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
STEP = 6
GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Gear:
    center: tuple[int, int]
    color: int
    teeth: tuple[str, str]
    driver: bool


def _rotate_dir(direction: str, clockwise: bool) -> str:
    idx = DIRS.index(direction)
    return DIRS[(idx + (1 if clockwise else -1)) % 4]


def _rotate_teeth(teeth: tuple[str, str], clockwise: bool) -> tuple[str, str]:
    rotated = sorted(_rotate_dir(direction, clockwise) for direction in teeth)
    return tuple(rotated)


def _tip_pos(center: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = TIP_OFFSETS[direction]
    return center[0] + dr, center[1] + dc


def _detect_gears(grid: np.ndarray) -> list[Gear]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gears: list[Gear] = []

    for r in range(2, h - 2):
        for c in range(2, w - 2):
            ring = [
                int(g[r + dr, c + dc])
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            ]
            if len(set(ring)) != 1 or ring[0] == BG:
                continue

            color = int(ring[0])
            center = int(g[r, c])
            driver = center == YELLOW and color != YELLOW
            if center != color and not driver:
                continue

            teeth = []
            valid = True
            for direction, (dr, dc) in TIP_OFFSETS.items():
                value = int(g[r + dr, c + dc])
                if value == color:
                    teeth.append(direction)
                elif value != BG:
                    valid = False
                    break
            if not valid or len(teeth) != 2:
                continue

            gears.append(Gear(center=(r, c), color=color, teeth=tuple(sorted(teeth)), driver=driver))

    return gears


def _build_graph(grid: np.ndarray, gears: list[Gear]) -> list[list[int]]:
    g = np.asarray(grid, dtype=int)
    graph = [[] for _ in gears]
    for i, gear_a in enumerate(gears):
        for j in range(i + 1, len(gears)):
            gear_b = gears[j]
            meshed = False
            for direction in gear_a.teeth:
                opposite = OPPOSITE[direction]
                if opposite not in gear_b.teeth:
                    continue
                tip_a = _tip_pos(gear_a.center, direction)
                mid = (tip_a[0] + VEC[direction][0], tip_a[1] + VEC[direction][1])
                tip_b = _tip_pos(gear_b.center, opposite)
                expected_b = (
                    tip_a[0] + 2 * VEC[direction][0],
                    tip_a[1] + 2 * VEC[direction][1],
                )
                if tip_b == expected_b and int(g[mid]) == BG:
                    meshed = True
                    break
            if meshed:
                graph[i].append(j)
                graph[j].append(i)
    return graph


def _analyze(grid: np.ndarray) -> tuple[list[Gear], list[list[int]], int]:
    gears = _detect_gears(grid)
    assert gears, "no gears found"
    drivers = [idx for idx, gear in enumerate(gears) if gear.driver]
    assert len(drivers) == 1, "expected exactly one driver gear"
    graph = _build_graph(grid, gears)
    return gears, graph, drivers[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify gears, propagate alternating rotation along the meshing component,
    and redraw the two teeth of each driven gear after a 90-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    gears, graph, driver_idx = _analyze(g)

    depths = {driver_idx: 0}
    queue = deque([driver_idx])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor in depths:
                continue
            depths[neighbor] = depths[node] + 1
            queue.append(neighbor)

    out = g.copy()

    for idx, depth in depths.items():
        gear = gears[idx]
        for direction in gear.teeth:
            rr, cc = _tip_pos(gear.center, direction)
            out[rr, cc] = BG
        if gear.driver:
            out[gear.center] = gear.color

    for idx, depth in depths.items():
        gear = gears[idx]
        clockwise = depth % 2 == 0
        for direction in _rotate_teeth(gear.teeth, clockwise):
            rr, cc = _tip_pos(gear.center, direction)
            out[rr, cc] = gear.color

    return out


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    result = []
    for dr, dc in VEC.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _random_induced_path(rows: int, cols: int, length: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        start = (random.randrange(rows), random.randrange(cols))
        path = [start]
        used = {start}

        while len(path) < length:
            tail = path[-1]
            candidates = []
            for nxt in _neighbors(tail, rows, cols):
                if nxt in used:
                    continue
                if any(_manhattan(nxt, prev) == 1 for prev in path[:-1]):
                    continue
                candidates.append(nxt)
            if not candidates:
                break
            nxt = random.choice(candidates)
            path.append(nxt)
            used.add(nxt)

        if len(path) == length:
            return path
    return None


def _pick_isolates(
    rows: int,
    cols: int,
    blocked: set[tuple[int, int]],
    target: int,
) -> list[tuple[int, int]] | None:
    if target == 0:
        return []

    all_nodes = [(r, c) for r in range(rows) for c in range(cols)]
    base_candidates = [
        cell
        for cell in all_nodes
        if cell not in blocked and all(_manhattan(cell, other) != 1 for other in blocked)
    ]
    if len(base_candidates) < target:
        return None

    for _ in range(200):
        candidates = base_candidates[:]
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for cell in candidates:
            if all(_manhattan(cell, other) != 1 for other in chosen):
                chosen.append(cell)
                if len(chosen) == target:
                    return chosen
    return None


def _dir_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return {
        (-1, 0): "U",
        (1, 0): "D",
        (0, -1): "L",
        (0, 1): "R",
    }[(dr, dc)]


def _lattice_to_center(
    cell: tuple[int, int],
    top_margin: int,
    left_margin: int,
) -> tuple[int, int]:
    return top_margin + 2 + STEP * cell[0], left_margin + 2 + STEP * cell[1]


def _draw_gear(
    grid: np.ndarray,
    center: tuple[int, int],
    color: int,
    teeth: tuple[str, str],
    driver: bool,
) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1 : c + 2] = color
    if driver:
        grid[r, c] = YELLOW
    for direction in teeth:
        rr, cc = _tip_pos(center, direction)
        grid[rr, cc] = color


def _make_attempt() -> np.ndarray | None:
    rows = random.choices([3, 4, 5], weights=[1, 3, 2])[0]
    cols = random.choices([3, 4, 5], weights=[1, 3, 2])[0]
    max_gears = min(len(PALETTE), rows * cols)
    chain_len = random.choices([1, 2, 3, 4, 5], weights=[1, 3, 3, 2, 1])[0]
    chain_len = min(chain_len, max_gears)

    path = _random_induced_path(rows, cols, chain_len)
    if path is None:
        return None

    max_iso = min(3, max_gears - chain_len)
    iso_choices = list(range(max_iso + 1))
    iso_weights = [2] + [2] * max_iso
    isolate_count = random.choices(iso_choices, weights=iso_weights)[0]
    isolates = _pick_isolates(rows, cols, set(path), isolate_count)
    if isolates is None:
        return None

    all_positions = path + isolates
    colors = random.sample(PALETTE, len(all_positions))
    driver_on_path = random.randrange(chain_len)

    teeth_map: dict[tuple[int, int], tuple[str, str]] = {}
    for idx, cell in enumerate(path):
        edge_dirs = []
        if idx > 0:
            edge_dirs.append(_dir_between(cell, path[idx - 1]))
        if idx + 1 < len(path):
            edge_dirs.append(_dir_between(cell, path[idx + 1]))
        edge_dirs = sorted(set(edge_dirs))
        if len(edge_dirs) == 0:
            teeth = tuple(sorted(random.sample(DIRS, 2)))
        elif len(edge_dirs) == 1:
            remaining = [direction for direction in DIRS if direction not in edge_dirs]
            teeth = tuple(sorted(edge_dirs + [random.choice(remaining)]))
        else:
            teeth = tuple(edge_dirs)
        teeth_map[cell] = teeth

    for cell in isolates:
        teeth_map[cell] = tuple(sorted(random.sample(DIRS, 2)))

    top_margin = random.randint(0, 3)
    left_margin = random.randint(0, 3)
    bottom_margin = random.randint(0, 3)
    right_margin = random.randint(0, 3)
    height = STEP * rows - 1 + top_margin + bottom_margin
    width = STEP * cols - 1 + left_margin + right_margin
    grid = np.full((height, width), BG, dtype=int)

    for idx, cell in enumerate(all_positions):
        center = _lattice_to_center(cell, top_margin, left_margin)
        _draw_gear(
            grid,
            center=center,
            color=colors[idx],
            teeth=teeth_map[cell],
            driver=idx == driver_on_path,
        )

    gears, graph, driver_idx = _analyze(grid)
    if len(gears) != len(all_positions):
        return None

    depths = {driver_idx: 0}
    queue = deque([driver_idx])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor in depths:
                continue
            depths[neighbor] = depths[node] + 1
            queue.append(neighbor)

    if len(depths) != chain_len:
        return None

    changed = main(grid)
    if np.array_equal(changed, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _make_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid gear puzzle")
