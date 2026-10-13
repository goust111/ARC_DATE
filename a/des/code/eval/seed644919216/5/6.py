from __future__ import annotations

from collections import deque
import random

import numpy as np


ENTRANCE_COLOR = 3
EXIT_COLOR = 2
WALL_COLOR = 5
ALL_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _largest_non_bg_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(values) > 0, "expected at least one non-background color"
    return int(values[np.argmax(counts)])


def _component_mask(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = passable.shape
    out = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque([start])
    out[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if passable[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _bfs_path(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = passable.shape
    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for nr, nc in _neighbors4(r, c, h, w):
            if passable[nr, nc] and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    assert goal in parent, "goal must be reachable from start"

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _degree_map(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros(mask.shape, dtype=int)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc, h, w):
            if mask[nr, nc]:
                deg[rr, cc] += 1
    return deg


def _prune_dead_ends(component: np.ndarray, terminals: set[tuple[int, int]]) -> np.ndarray:
    active = component.copy()
    deg = _degree_map(active)
    q: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(active):
        cell = (int(r), int(c))
        if cell not in terminals and deg[cell] <= 1:
            q.append(cell)

    while q:
        cell = q.popleft()
        r, c = cell
        if not active[r, c] or cell in terminals or deg[r, c] > 1:
            continue
        active[r, c] = False
        for nr, nc in _neighbors4(r, c, active.shape[0], active.shape[1]):
            if active[nr, nc]:
                deg[nr, nc] -= 1
                if (nr, nc) not in terminals and deg[nr, nc] <= 1:
                    q.append((nr, nc))

    return active


def _ordered_path(path_mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    seen = {start}
    while cur != goal:
        candidates = [
            nxt
            for nxt in _neighbors4(cur[0], cur[1], h, w)
            if path_mask[nxt] and nxt != prev
        ]
        assert len(candidates) == 1, "reduced maze must collapse to a single path"
        nxt = candidates[0]
        assert nxt not in seen, "path should not cycle after dead-end pruning"
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt
    return path


def _path_output(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    path: list[tuple[int, int]],
    marker_lookup: dict[tuple[int, int], int],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    out[wall_mask] = WALL_COLOR
    current = ENTRANCE_COLOR
    for pos in path[:-1]:
        if pos in marker_lookup:
            current = marker_lookup[pos]
        out[pos] = current
    out[path[0]] = ENTRANCE_COLOR
    out[path[-1]] = EXIT_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color the unique marker-guided route through the maze."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    wall = _largest_non_bg_color(g, bg)

    entrance_pts = np.argwhere(g == ENTRANCE_COLOR)
    exit_pts = np.argwhere(g == EXIT_COLOR)
    assert entrance_pts.shape == (1, 2), "expected exactly one green entrance"
    assert exit_pts.shape == (1, 2), "expected exactly one red exit"
    start = tuple(map(int, entrance_pts[0]))
    goal = tuple(map(int, exit_pts[0]))

    passable = g != wall
    component = _component_mask(passable, start)
    assert component[goal], "exit must share the maze corridor component"

    markers: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(component):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        if color not in {bg, wall, ENTRANCE_COLOR, EXIT_COLOR}:
            markers[(rr, cc)] = color

    terminals = set(markers) | {start, goal}
    reduced = _prune_dead_ends(component, terminals)
    assert reduced[start] and reduced[goal]
    for pos in terminals:
        assert reduced[pos], "all terminals must survive pruning"

    path = _ordered_path(reduced, start, goal)
    assert set(path) == {tuple(map(int, rc)) for rc in np.argwhere(reduced)}, "reduced component must be a single path"
    for pos in markers:
        assert pos in path, "every marker must lie on the final path"

    return _path_output(g.shape, g == wall, path, markers, bg)


def _carve_tree_maze(cell_rows: int, cell_cols: int, bg: int) -> np.ndarray:
    h = 2 * cell_rows + 1
    w = 2 * cell_cols + 1
    maze = np.full((h, w), WALL_COLOR, dtype=int)
    for r in range(cell_rows):
        for c in range(cell_cols):
            maze[2 * r + 1, 2 * c + 1] = bg

    stack = [(0, 0)]
    visited = {(0, 0)}
    while stack:
        r, c = stack[-1]
        options: list[tuple[int, int, int, int]] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < cell_rows and 0 <= nc < cell_cols and (nr, nc) not in visited:
                options.append((nr, nc, dr, dc))
        if not options:
            stack.pop()
            continue
        nr, nc, dr, dc = random.choice(options)
        maze[2 * r + 1 + dr, 2 * c + 1 + dc] = bg
        visited.add((nr, nc))
        stack.append((nr, nc))
    return maze


def _side_cells(maze: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = maze.shape
    if side == "top":
        return [(1, c) for c in range(1, w, 2)]
    if side == "bottom":
        return [(h - 2, c) for c in range(1, w, 2)]
    if side == "left":
        return [(r, 1) for r in range(1, h, 2)]
    if side == "right":
        return [(r, w - 2) for r in range(1, h, 2)]
    raise AssertionError(side)


def _sample_marker_indices(path: list[tuple[int, int]], passable: np.ndarray) -> list[int] | None:
    deg = _degree_map(passable)
    branch_indices = [i for i, pos in enumerate(path[1:-1], start=1) if deg[pos] > 2]
    last_branch = max(branch_indices, default=0)
    candidates = [i for i in range(1, len(path) - 1) if deg[path[i]] <= 2]
    tail = [i for i in candidates if i > last_branch]
    if len(candidates) < 2 or not tail:
        return None

    max_count = min(5, len(candidates))
    target = random.randint(2, max_count)
    for need in range(target, 1, -1):
        for _ in range(60):
            last_idx = random.choice(tail)
            pool = [i for i in candidates if i != last_idx and abs(i - last_idx) >= 2]
            random.shuffle(pool)
            selected = [last_idx]
            for idx in pool:
                if all(abs(idx - prev) >= 2 for prev in selected):
                    selected.append(idx)
                    if len(selected) == need:
                        return sorted(selected)
    return None


def _design_local_maze(bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    cell_rows = random.randint(4, 7)
    cell_cols = random.randint(4, 7)
    local_in = _carve_tree_maze(cell_rows, cell_cols, bg)
    side_pairs = [("top", "bottom"), ("bottom", "top"), ("left", "right"), ("right", "left")]
    start_side, goal_side = random.choice(side_pairs)

    passable = local_in != WALL_COLOR
    start_options = _side_cells(local_in, start_side)
    goal_options = _side_cells(local_in, goal_side)
    random.shuffle(start_options)
    random.shuffle(goal_options)

    for start in start_options:
        for goal in goal_options:
            path = _bfs_path(passable, start, goal)
            if len(path) < 7:
                continue
            marker_indices = _sample_marker_indices(path, passable)
            if marker_indices is None:
                continue

            available_colors = [c for c in ALL_COLORS if c not in {bg, WALL_COLOR, ENTRANCE_COLOR, EXIT_COLOR}]
            if len(available_colors) < len(marker_indices):
                return None
            marker_colors = random.sample(available_colors, len(marker_indices))
            marker_lookup = {path[idx]: color for idx, color in zip(marker_indices, marker_colors, strict=False)}

            local_in = local_in.copy()
            local_in[start] = ENTRANCE_COLOR
            local_in[goal] = EXIT_COLOR
            for pos, color in marker_lookup.items():
                local_in[pos] = color

            local_out = _path_output(local_in.shape, local_in == WALL_COLOR, path, marker_lookup, bg)
            return local_in, local_out
    return None


def _embed(local: np.ndarray, bg: int, margins: tuple[int, int, int, int]) -> np.ndarray:
    margin_top, margin_bottom, margin_left, margin_right = margins
    out = np.full(
        (
            local.shape[0] + margin_top + margin_bottom,
            local.shape[1] + margin_left + margin_right,
        ),
        bg,
        dtype=int,
    )
    out[margin_top : margin_top + local.shape[0], margin_left : margin_left + local.shape[1]] = local
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        bg = random.choice([c for c in ALL_COLORS if c not in {WALL_COLOR, ENTRANCE_COLOR, EXIT_COLOR}])
        designed = _design_local_maze(bg)
        if designed is None:
            continue
        local_in, local_out = designed
        margins = (
            random.randint(2, 6),
            random.randint(2, 6),
            random.randint(2, 6),
            random.randint(2, 6),
        )
        full_in = _embed(local_in, bg, margins)
        full_out = _embed(local_out, bg, margins)

        if not np.array_equal(main(full_in), full_out):
            continue

        values, counts = np.unique(full_in, return_counts=True)
        bg_count = int(counts[np.where(values == bg)[0][0]])
        if any(bg_count <= int(cnt) for val, cnt in zip(values.tolist(), counts.tolist(), strict=False) if int(val) != bg):
            continue

        return full_in
    raise RuntimeError("failed to generate a valid maze input")
