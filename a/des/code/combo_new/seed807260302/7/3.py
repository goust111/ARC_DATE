from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIGNAL_COLORS = (1, 2, 3, 4, 6, 7, 8)

MIN_SIZE = 8
MAX_SIZE = 24
MIN_PATH_LEN = 6
MAX_PATH_LEN = 22
MAX_GATES = 4
GENERATION_ATTEMPTS = 400
PATH_ATTEMPTS = 120
GATE_SELECTION_ATTEMPTS = 80

Coord = tuple[int, int]


def _neighbors(cell: Coord, shape: tuple[int, int]) -> list[Coord]:
    r, c = cell
    height, width = shape
    out: list[Coord] = []
    for dr, dc in CARDINAL_DIRS:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _build_adjacency(cells: set[Coord], shape: tuple[int, int]) -> dict[Coord, list[Coord]]:
    return {
        cell: [nbr for nbr in _neighbors(cell, shape) if nbr in cells]
        for cell in cells
    }


def _tree_path(adj: dict[Coord, list[Coord]], start: Coord, end: Coord) -> list[Coord]:
    queue: deque[Coord] = deque([start])
    prev: dict[Coord, Coord | None] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == end:
            break
        for nxt in adj[cur]:
            if nxt in prev:
                continue
            prev[nxt] = cur
            queue.append(nxt)
    assert end in prev, "foreground graph must connect source to endpoint"

    path: list[Coord] = []
    cur: Coord | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _candidate_solution(
    g: np.ndarray,
    adj: dict[Coord, list[Coord]],
    source: Coord,
    target: Coord,
) -> tuple[list[Coord], dict[Coord, int]] | None:
    if int(g[source]) in {BLACK, GREY, MAROON}:
        return None
    if int(g[target]) != MAROON:
        return None

    path = _tree_path(adj, source, target)
    if len(path) < 2:
        return None

    path_index = {cell: idx for idx, cell in enumerate(path)}
    for idx, cell in enumerate(path):
        color = int(g[cell])
        if idx == 0:
            if color in {BLACK, GREY, MAROON}:
                return None
        elif idx == len(path) - 1:
            if color != MAROON:
                return None
        elif color != GREY:
            return None

    gate_map: dict[Coord, int] = {}
    for cell, nbrs in adj.items():
        idx = path_index.get(cell)
        if idx is not None:
            off_path = [nbr for nbr in nbrs if nbr not in path_index]
            if len(off_path) > 1:
                return None
            if idx in {0, len(path) - 1} and off_path:
                return None
            if idx in {1, len(path) - 2} and off_path:
                return None
            continue

        if len(nbrs) != 1:
            return None
        color = int(g[cell])
        if color in {BLACK, GREY, MAROON}:
            return None
        attach = nbrs[0]
        attach_idx = path_index.get(attach)
        if attach_idx is None or not (2 <= attach_idx <= len(path) - 3):
            return None
        if int(g[attach]) != GREY:
            return None
        if attach in gate_map:
            return None
        gate_map[attach] = color

    return path, gate_map


def _find_signal_path(g: np.ndarray) -> tuple[Coord, list[Coord], dict[Coord, int]]:
    cells = {tuple(map(int, coord)) for coord in np.argwhere(g != BLACK)}
    assert cells, "expected a non-empty foreground structure"

    adj = _build_adjacency(cells, g.shape)
    seen: set[Coord] = set()
    queue = deque([next(iter(cells))])
    while queue:
        cur = queue.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        queue.extend(adj[cur])
    assert seen == cells, "foreground structure must be connected"

    edge_count = sum(len(nbrs) for nbrs in adj.values()) // 2
    assert edge_count == len(cells) - 1, "foreground structure must be a tree"

    leaves = [cell for cell, nbrs in adj.items() if len(nbrs) == 1]
    maroons = [cell for cell in cells if int(g[cell]) == MAROON]
    assert len(maroons) == 1, "expected exactly one maroon endpoint"
    target = maroons[0]
    assert target in leaves, "maroon endpoint must be a leaf"

    candidates: list[tuple[Coord, list[Coord], dict[Coord, int]]] = []
    for leaf in leaves:
        if leaf == target:
            continue
        result = _candidate_solution(g, adj, leaf, target)
        if result is not None:
            path, gate_map = result
            candidates.append((leaf, path, gate_map))

    assert len(candidates) == 1, "expected a unique source leaf compatible with the gate layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a signal along the grey route, recoloring each path segment and the maroon endpoint after every adjacent gate overwrite.

    Input structure:
    - The background is black.
    - A single one-cell-wide orthogonal route is formed by grey cells, with one colored source endpoint and one maroon endpoint.
    - Additional single colored pixels act as gates: each is attached as a side leaf to exactly one interior grey route cell.

    Rule:
    - Recover the unique source-to-maroon main path and treat every other foreground leaf as a gate.
    - Start with the source color and walk from the source toward the maroon endpoint.
    - Whenever the current grey path cell has an attached gate, replace the carried color with that gate color.
    - Recolor each grey path cell with the carried color after applying any gate at that cell, keep the source and gates unchanged, and replace the maroon endpoint with the final carried color.

    Key insight:
    - The foreground forms a tree whose spine is the real route; the colored side leaves are not part of the route but color-changing events anchored to specific grey cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source, path, gate_map = _find_signal_path(g)
    out = g.copy()
    current_color = int(g[source])

    for cell in path[1:]:
        if int(g[cell]) == GREY:
            gate_color = gate_map.get(cell)
            if gate_color is not None:
                current_color = int(gate_color)
            out[cell] = current_color
        else:
            assert int(g[cell]) == MAROON
            out[cell] = current_color

    return out


def _path_step_options(
    current: Coord,
    path_set: set[Coord],
    shape: tuple[int, int],
    prev_dir: Coord | None,
) -> list[tuple[Coord, Coord]]:
    options: list[tuple[Coord, Coord]] = []
    for dr, dc in CARDINAL_DIRS:
        nxt = (current[0] + dr, current[1] + dc)
        if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1]):
            continue
        if nxt in path_set:
            continue
        touch_count = sum(1 for nbr in _neighbors(nxt, shape) if nbr in path_set)
        if touch_count != 1:
            continue
        options.append(((dr, dc), nxt))

    random.shuffle(options)
    if prev_dir is None:
        return options

    straight = [item for item in options if item[0] == prev_dir]
    turns = [item for item in options if item[0] != prev_dir]
    if straight and turns and random.random() < 0.5:
        return turns + straight
    return straight + turns if straight else turns


def _generate_path(height: int, width: int, length: int) -> list[Coord] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(PATH_ATTEMPTS):
        start = random.choice(cells)
        path = [start]
        path_set = {start}

        def dfs(prev_dir: Coord | None) -> bool:
            if len(path) == length:
                return True
            for step_dir, nxt in _path_step_options(path[-1], path_set, (height, width), prev_dir):
                path.append(nxt)
                path_set.add(nxt)
                if dfs(step_dir):
                    return True
                path.pop()
                path_set.remove(nxt)
            return False

        if dfs(None):
            return path.copy()
    return None


def _gate_slot_options(path: list[Coord], shape: tuple[int, int]) -> list[tuple[int, list[Coord]]]:
    path_set = set(path)
    slots: list[tuple[int, list[Coord]]] = []
    for idx in range(2, len(path) - 2):
        cell = path[idx]
        candidates = []
        for nbr in _neighbors(cell, shape):
            if nbr in path_set:
                continue
            if sum(1 for nn in _neighbors(nbr, shape) if nn in path_set) != 1:
                continue
            candidates.append(nbr)
        if candidates:
            random.shuffle(candidates)
            slots.append((idx, candidates))
    return slots


def _adjacent(a: Coord, b: Coord) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _select_gates(slot_options: list[tuple[int, list[Coord]]], gate_count: int) -> list[tuple[int, Coord]] | None:
    if gate_count == 0:
        return []

    for _ in range(GATE_SELECTION_ATTEMPTS):
        chosen_slots = random.sample(slot_options, gate_count)
        selected: list[tuple[int, Coord]] = []
        used_cells: list[Coord] = []
        ok = True
        for idx, choices in chosen_slots:
            shuffled = choices[:]
            random.shuffle(shuffled)
            gate_cell = None
            for cell in shuffled:
                if any(_adjacent(cell, other) for other in used_cells):
                    continue
                gate_cell = cell
                break
            if gate_cell is None:
                ok = False
                break
            selected.append((idx, gate_cell))
            used_cells.append(gate_cell)
        if ok:
            return selected
    return None


def _turn_count(path: list[Coord]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        a = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        b = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if a != b:
            turns += 1
    return turns


def _render_grid(
    shape: tuple[int, int],
    path: list[Coord],
    source_color: int,
    gates: list[tuple[int, Coord]],
    gate_colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for cell in path[1:-1]:
        grid[cell] = GREY
    grid[path[0]] = int(source_color)
    grid[path[-1]] = MAROON
    for (_idx, cell), color in zip(gates, gate_colors):
        grid[cell] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)
    max_len = min(MAX_PATH_LEN, max(MIN_PATH_LEN, height * width // 5 + 3))
    path_len = random.randint(MIN_PATH_LEN, max_len)
    want_gates = random.random() < 0.75

    path = _generate_path(height, width, path_len)
    if path is None:
        return None

    slot_options = _gate_slot_options(path, (height, width))
    if want_gates and not slot_options:
        return None

    max_gates = min(MAX_GATES, len(slot_options), len(SIGNAL_COLORS) - 1)
    gate_count = 0
    if want_gates and max_gates > 0:
        gate_count = random.randint(1, max_gates)

    gates = _select_gates(slot_options, gate_count)
    if gates is None:
        return None

    source_color = random.choice(SIGNAL_COLORS)
    gate_palette = [color for color in SIGNAL_COLORS if color != source_color]
    gate_colors = random.sample(gate_palette, gate_count)
    grid = _render_grid((height, width), path, source_color, gates, gate_colors)

    if _turn_count(path) == 0 and random.random() < 0.5:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-path puzzle")
