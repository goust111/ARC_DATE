from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS)}
BRANCH_TARGETS = [
    "prefer_left",
    "prefer_right",
    "fallback_straight_from_left",
    "fallback_straight_from_right",
    "fallback_other_to_left",
    "fallback_other_to_right",
]
_branch_counter = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _opposite(direction: int) -> int:
    return (direction + 2) % 4


def _grey_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "maze must contain grey walls"
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return pts


def _gap_infos(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[dict[str, int | tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    gaps: list[dict[str, int | tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in _perimeter_cells(r0, c0, r1, c1):
        if int(grid[r, c]) == GREY:
            continue
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if r == r0:
            outside = (r - 1, c)
            direction = 2
        elif r == r1:
            outside = (r + 1, c)
            direction = 0
        elif c == c0:
            outside = (r, c - 1)
            direction = 1
        else:
            outside = (r, c + 1)
            direction = 3
        assert _in_bounds(grid.shape, outside[0], outside[1]), "gap marker must be inside the canvas"
        marker = int(grid[outside])
        assert marker not in {BLACK, GREEN, GREY}, "gap must have a colored exterior marker"
        gaps.append({"pos": (r, c), "outside": outside, "direction": direction, "marker": marker})
    assert len(gaps) == 2, "maze must have exactly two boundary gaps"
    assert int(gaps[0]["marker"]) != int(gaps[1]["marker"]), "entrance and exit markers must differ"
    return gaps


def _trace_route(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    start_gap: dict[str, int | tuple[int, int]],
    goal_gap: dict[str, int | tuple[int, int]],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    passable = np.zeros_like(grid, dtype=bool)
    passable[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1] != GREY

    start_color = int(start_gap["marker"])
    exit_color = int(goal_gap["marker"])
    pos = tuple(start_gap["pos"])
    goal = tuple(goal_gap["pos"])
    direction = int(start_gap["direction"])
    path = [pos]
    seen = {(pos, direction)}

    while True:
        if pos == goal:
            return path

        open_dirs = []
        for d, (dr, dc) in enumerate(DIRS):
            nr, nc = pos[0] + dr, pos[1] + dc
            if _in_bounds(grid.shape, nr, nc) and passable[nr, nc]:
                open_dirs.append(d)
        non_back = [d for d in open_dirs if d != _opposite(direction)]
        if not non_back:
            return None

        left = (direction + 3) % 4
        right = (direction + 1) % 4
        value = int(grid[pos])

        next_dir = None
        if value == start_color:
            for cand in (left, direction, right):
                if cand in non_back:
                    next_dir = cand
                    break
        elif value == exit_color:
            for cand in (right, direction, left):
                if cand in non_back:
                    next_dir = cand
                    break
        else:
            if direction in non_back:
                next_dir = direction
            elif len(non_back) == 1:
                next_dir = non_back[0]
            else:
                for cand in (left, right):
                    if cand in non_back:
                        next_dir = cand
                        break

        if next_dir is None:
            return None

        dr, dc = DIRS[next_dir]
        pos = (pos[0] + dr, pos[1] + dc)
        direction = next_dir
        state = (pos, direction)
        if state in seen:
            return None
        seen.add(state)
        path.append(pos)


def _solve_path(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    bbox = _grey_bbox(grid)
    gaps = _gap_infos(grid, bbox)
    attempts = [
        _trace_route(grid, bbox, gaps[0], gaps[1]),
        _trace_route(grid, bbox, gaps[1], gaps[0]),
    ]
    valid = [path for path in attempts if path is not None]
    assert valid, "maze route must be traceable from one of the two gaps"
    if len(valid) == 2:
        assert set(valid[0]) == set(valid[1]), "reversing the route should preserve the traced path"
    return bbox, min(valid, key=len)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the maze from one colored gap marker to the other, turning left at
    markers matching the entrance color and right at markers matching the exit
    color. If the preferred turn is blocked, continue straight; if straight is
    also blocked, take the only remaining non-backtracking direction.
    """
    g = np.asarray(grid, dtype=int)
    bbox, path = _solve_path(g)
    r0, c0, r1, c1 = bbox

    out = g.copy()
    passage = np.zeros_like(g, dtype=bool)
    passage[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1] != GREY
    out[passage] = BLACK

    for r, c in path:
        out[r, c] = GREEN
    return out


def _next_branch_target() -> str:
    global _branch_counter
    target = BRANCH_TARGETS[_branch_counter % len(BRANCH_TARGETS)]
    _branch_counter += 1
    return target


def _touching_boundary_dirs(cell: tuple[int, int], rows: int, cols: int) -> list[int]:
    r, c = cell
    dirs = []
    if r == 0:
        dirs.append(0)
    if c == cols - 1:
        dirs.append(1)
    if r == rows - 1:
        dirs.append(2)
    if c == 0:
        dirs.append(3)
    return dirs


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _generate_maze_graph(rows: int, cols: int, rng: random.Random) -> dict[tuple[int, int], set[tuple[int, int]]]:
    start = (rng.randrange(rows), rng.randrange(cols))
    graph = {(r, c): set() for r in range(rows) for c in range(cols)}
    visited = {start}
    frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for nb in _neighbors(start, rows, cols):
        frontier.append((start, nb))

    while frontier:
        idx = rng.randrange(len(frontier))
        src, dst = frontier.pop(idx)
        if dst in visited:
            continue
        visited.add(dst)
        graph[src].add(dst)
        graph[dst].add(src)
        for nb in _neighbors(dst, rows, cols):
            if nb not in visited:
                frontier.append((dst, nb))
    return graph


def _cell_direction(a: tuple[int, int], b: tuple[int, int]) -> int:
    return DIR_TO_INDEX[(b[0] - a[0], b[1] - a[1])]


def _path_between(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nb in graph[cur]:
            if nb in parent:
                continue
            parent[nb] = cur
            q.append(nb)
    assert goal in parent
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _possible_categories(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    path: list[tuple[int, int]],
    index: int,
) -> list[str]:
    prev_cell = path[index - 1]
    cur = path[index]
    next_cell = path[index + 1]
    incoming = _cell_direction(prev_cell, cur)
    outgoing = _cell_direction(cur, next_cell)
    open_dirs = {_cell_direction(cur, nb) for nb in graph[cur]}
    non_back = open_dirs - {_opposite(incoming)}
    left = (incoming + 3) % 4
    right = (incoming + 1) % 4
    straight = incoming

    possible: list[str] = []
    if len(graph[cur]) >= 3:
        if outgoing == left:
            possible.append("prefer_left")
        elif outgoing == right:
            possible.append("prefer_right")
        elif outgoing == straight:
            if left not in non_back:
                possible.append("fallback_straight_from_left")
            if right not in non_back:
                possible.append("fallback_straight_from_right")

    if outgoing == left and right not in non_back and straight not in non_back:
        possible.append("fallback_other_to_left")
    if outgoing == right and left not in non_back and straight not in non_back:
        possible.append("fallback_other_to_right")
    return possible


def _category_color(category: str, start_color: int, exit_color: int) -> int:
    if category in {"prefer_left", "fallback_straight_from_left", "fallback_other_to_right"}:
        return start_color
    if category in {"prefer_right", "fallback_straight_from_right", "fallback_other_to_left"}:
        return exit_color
    raise ValueError(category)


def _pick_pair_and_markers(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    rows: int,
    cols: int,
    target: str,
    rng: random.Random,
) -> tuple[
    tuple[int, int],
    int,
    tuple[int, int],
    int,
    list[tuple[int, int]],
    tuple[int, int],
]:
    leaves: list[tuple[tuple[int, int], int]] = []
    for cell, nbrs in graph.items():
        if len(nbrs) != 1:
            continue
        boundary_dirs = _touching_boundary_dirs(cell, rows, cols)
        if not boundary_dirs:
            continue
        only_neighbor = next(iter(nbrs))
        preferred_port = _opposite(_cell_direction(cell, only_neighbor))
        if preferred_port in boundary_dirs:
            leaves.append((cell, preferred_port))

    junctions = [cell for cell, nbrs in graph.items() if len(nbrs) >= 3]
    if len(leaves) < 2 or len(junctions) == 0:
        raise ValueError("maze lacks usable leaves or junctions")

    candidates = []
    for i, (start_cell, start_port) in enumerate(leaves):
        for j, (end_cell, end_port) in enumerate(leaves):
            if i == j:
                continue
            path = _path_between(graph, start_cell, end_cell)
            if len(path) < 5:
                continue

            focus_positions = []
            compatible: dict[tuple[int, int], list[str]] = {}
            valid = True
            for idx in range(1, len(path) - 1):
                cell = path[idx]
                possible = _possible_categories(graph, path, idx)
                if len(graph[cell]) >= 3 and not possible:
                    valid = False
                    break
                compatible[cell] = possible
                if target in possible:
                    focus_positions.append(idx)
            if not valid or not focus_positions:
                continue

            marked_count = sum(1 for cell in path[1:-1] if len(graph[cell]) >= 3)
            if target.startswith("fallback_other"):
                marked_count += 1
            off_path_junctions = [cell for cell in junctions if cell not in path[1:-1]]
            if marked_count + len(off_path_junctions) < 2:
                continue

            candidates.append(
                {
                    "start_cell": start_cell,
                    "start_port": start_port,
                    "end_cell": end_cell,
                    "end_port": end_port,
                    "path": path,
                    "compatible": compatible,
                    "focus_positions": focus_positions,
                }
            )

    if not candidates:
        raise ValueError("no boundary pair supports the requested branch target")

    chosen = rng.choice(candidates)
    focus_index = rng.choice(chosen["focus_positions"])

    return (
        chosen["start_cell"],
        chosen["start_port"],
        chosen["end_cell"],
        chosen["end_port"],
        chosen["path"],
        chosen["path"][focus_index],
    )


def _render_grid(
    rows: int,
    cols: int,
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    start_cell: tuple[int, int],
    start_port: int,
    end_cell: tuple[int, int],
    end_port: int,
    markers: dict[tuple[int, int], int],
    start_color: int,
    exit_color: int,
    focus_cell: tuple[int, int],
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int]]:
    maze_h = 2 * rows + 1
    maze_w = 2 * cols + 1
    maze = np.full((maze_h, maze_w), GREY, dtype=int)

    for cell, nbrs in graph.items():
        rr, cc = 2 * cell[0] + 1, 2 * cell[1] + 1
        maze[rr, cc] = BLACK
        for nb in nbrs:
            if nb < cell:
                continue
            dr = nb[0] - cell[0]
            dc = nb[1] - cell[1]
            maze[rr + dr, cc + dc] = BLACK

    for cell, color in markers.items():
        rr, cc = 2 * cell[0] + 1, 2 * cell[1] + 1
        maze[rr, cc] = color

    for cell, port in ((start_cell, start_port), (end_cell, end_port)):
        rr, cc = 2 * cell[0] + 1, 2 * cell[1] + 1
        dr, dc = DIRS[port]
        maze[rr + dr, cc + dc] = BLACK

    pad_top = rng.randint(1, 3)
    pad_bottom = rng.randint(1, 3)
    pad_left = rng.randint(1, 3)
    pad_right = rng.randint(1, 3)
    grid = np.full((maze_h + pad_top + pad_bottom, maze_w + pad_left + pad_right), BLACK, dtype=int)
    grid[pad_top : pad_top + maze_h, pad_left : pad_left + maze_w] = maze

    for cell, port, color in (
        (start_cell, start_port, start_color),
        (end_cell, end_port, exit_color),
    ):
        rr, cc = 2 * cell[0] + 1, 2 * cell[1] + 1
        dr, dc = DIRS[port]
        mr = pad_top + rr + 2 * dr
        mc = pad_left + cc + 2 * dc
        grid[mr, mc] = color

    focus_r = pad_top + 2 * focus_cell[0] + 1
    focus_c = pad_left + 2 * focus_cell[1] + 1
    return grid, (focus_r, focus_c)


def _generate_candidate(rng: random.Random, target: str) -> np.ndarray:
    rows = rng.randint(5, 8)
    cols = rng.randint(5, 8)
    graph = _generate_maze_graph(rows, cols, rng)

    start_cell, start_port, end_cell, end_port, path, focus_cell = _pick_pair_and_markers(graph, rows, cols, target, rng)

    start_color, exit_color = rng.sample([BLUE, RED], 2)
    rebuilt: dict[tuple[int, int], int] = {}
    focus_index = path.index(focus_cell)
    for idx in range(1, len(path) - 1):
        cell = path[idx]
        possible = _possible_categories(graph, path, idx)
        if idx == focus_index:
            rebuilt[cell] = _category_color(target, start_color, exit_color)
        elif len(graph[cell]) >= 3:
            filtered = [cat for cat in possible if not cat.startswith("fallback_other")]
            pick_from = filtered if filtered else possible
            rebuilt[cell] = _category_color(rng.choice(pick_from), start_color, exit_color)

    for cell, nbrs in graph.items():
        if len(nbrs) < 3 or cell in rebuilt:
            continue
        rebuilt[cell] = rng.choice([start_color, exit_color])

    if len(set(rebuilt.values())) < 2:
        spare = [cell for cell, nbrs in graph.items() if len(nbrs) >= 3 and cell not in path[1:-1]]
        if not spare:
            raise ValueError("marker colors collapsed to one value")
        rebuilt[spare[0]] = exit_color if rebuilt[spare[0]] == start_color else start_color

    grid, focus_pixel = _render_grid(
        rows,
        cols,
        graph,
        start_cell,
        start_port,
        end_cell,
        end_port,
        rebuilt,
        start_color,
        exit_color,
        focus_cell,
        rng,
    )
    solved = main(grid)

    assert solved[focus_pixel] == GREEN, "target branch marker must lie on the traced route"
    assert np.count_nonzero(solved == GREEN) >= 8, "path should be visually substantial"
    assert len(set(rebuilt.values())) == 2, "both control colors should appear inside the maze"
    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate a single-grey-wall maze with two colored boundary markers.
    The marked route always satisfies the turning rule implemented by `main()`.
    """
    target = _next_branch_target()
    for _ in range(400):
        rng = random.Random()
        try:
            return _generate_candidate(rng, target)
        except (AssertionError, ValueError):
            continue
    raise RuntimeError(f"failed to generate a valid maze for branch target {target}")
