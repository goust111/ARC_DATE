from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


START_COLOR = 3
END_COLOR = 2
DIRS_RIGHT_FIRST = ((0, 1), (1, 0), (0, -1), (-1, 0))
EDGE_NAMES = ("top", "bottom", "left", "right")
ALL_COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS_RIGHT_FIRST:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _edge_line(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid[0, :], dtype=int)
    if edge == "bottom":
        return np.array(grid[-1, :], dtype=int)
    if edge == "left":
        return np.array(grid[:, 0], dtype=int)
    if edge == "right":
        return np.array(grid[:, -1], dtype=int)
    raise ValueError(f"unknown edge {edge}")


def _detect_gradient_edge(grid: np.ndarray) -> tuple[str, np.ndarray]:
    candidates: list[tuple[str, np.ndarray]] = []
    for edge in EDGE_NAMES:
        line = _edge_line(grid, edge)
        colors = set(map(int, line.tolist()))
        if START_COLOR in colors and END_COLOR in colors and len(colors) >= 3:
            candidates.append((edge, line))

    assert len(candidates) == 1, "expected exactly one edge palette strip"
    edge, strip = candidates[0]
    if int(strip[0]) == END_COLOR and int(strip[-1]) == START_COLOR:
        strip = strip[::-1].copy()
    assert int(strip[0]) == START_COLOR and int(strip[-1]) == END_COLOR
    return edge, strip


def _crop_gradient_edge(grid: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.array(grid[1:, :], dtype=int)
    if edge == "bottom":
        return np.array(grid[:-1, :], dtype=int)
    if edge == "left":
        return np.array(grid[:, 1:], dtype=int)
    if edge == "right":
        return np.array(grid[:, :-1], dtype=int)
    raise ValueError(f"unknown edge {edge}")


def _shortest_path(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = passable.shape
    dist = -np.ones((h, w), dtype=int)
    parent = np.full((h, w, 2), -1, dtype=int)
    q = deque([start])
    dist[start] = 0

    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        for dr, dc in DIRS_RIGHT_FIRST:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                parent[rr, cc] = (r, c)
                q.append((rr, cc))

    er, ec = end
    assert int(dist[er, ec]) >= 0, "start and end must be connected"

    path = [end]
    cur = end
    while cur != start:
        pr, pc = map(int, parent[cur])
        cur = (pr, pc)
        path.append(cur)
    path.reverse()
    return path


def _count_shortest_paths(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int], cap: int = 2
) -> int:
    h, w = passable.shape
    dist = -np.ones((h, w), dtype=int)
    q = deque([start])
    dist[start] = 0

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c, passable.shape):
            if passable[rr, cc] and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))

    if int(dist[end]) < 0:
        return 0

    order = sorted(
        ((int(r), int(c)) for r, c in np.argwhere(dist >= 0)),
        key=lambda rc: int(dist[rc]),
    )
    ways = np.zeros((h, w), dtype=int)
    ways[start] = 1
    for r, c in order[1:]:
        total = 0
        for rr, cc in _neighbors4(r, c, passable.shape):
            if int(dist[rr, cc]) == int(dist[r, c] - 1):
                total += int(ways[rr, cc])
                if total >= cap:
                    total = cap
                    break
        ways[r, c] = total
    return int(ways[end])


def _find_unique_cell(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == int(color))
    assert len(coords) == 1, f"expected exactly one cell of color {color}"
    return tuple(map(int, coords[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the edge palette strip and paint the maze's shortest route with its sampled colors.

    Input structure:
    - One border row or border column is a 1xK color strip that starts green and ends red.
    - After removing that strip, the remaining grid is a wall-bounded maze with one green start pixel and one red end pixel.
    - The maze itself uses exactly one wall color and one passable corridor color.

    Rule:
    - Identify and remove the full edge strip from the canvas.
    - In the remaining maze, find the shortest 4-neighbor path from start to end through corridor cells; when several shortest paths exist, prefer BFS expansion in the order right, down, left, up.
    - Recolor path cell i with strip color floor(i*K/path_length), while keeping the start green and the end red.
    - Leave walls and off-path corridor cells unchanged.

    Key insight: The edge strip is not part of the maze; it is an ordered palette that must be cropped away and sampled by path position rather than matched one-to-one to path length.
    """
    g = np.asarray(grid, dtype=int)
    edge, strip = _detect_gradient_edge(g)
    maze = _crop_gradient_edge(g, edge)

    corners = [int(maze[0, 0]), int(maze[0, -1]), int(maze[-1, 0]), int(maze[-1, -1])]
    wall_color = int(Counter(corners).most_common(1)[0][0])
    start = _find_unique_cell(maze, START_COLOR)
    end = _find_unique_cell(maze, END_COLOR)

    maze_colors = set(map(int, np.unique(maze).tolist()))
    corridor_candidates = maze_colors - {wall_color, START_COLOR, END_COLOR}
    assert len(corridor_candidates) == 1, "expected exactly one corridor color"
    corridor_color = int(next(iter(corridor_candidates)))

    passable = (maze == corridor_color) | (maze == START_COLOR) | (maze == END_COLOR)
    path = _shortest_path(passable, start, end)

    out = maze.copy()
    k = len(strip)
    plen = len(path)
    for i, (r, c) in enumerate(path):
        out[r, c] = int(strip[(i * k) // plen])
    out[start] = START_COLOR
    out[end] = END_COLOR
    return out


def _choose_palette(rng: random.Random) -> tuple[int, int, list[int]]:
    bg, wall = rng.sample([c for c in ALL_COLORS if c not in (START_COLOR, END_COLOR)], 2)
    mids = [c for c in ALL_COLORS if c not in (START_COLOR, END_COLOR, bg, wall)]
    return int(bg), int(wall), mids


def _random_gradient(length: int, bg: int, wall: int, rng: random.Random) -> np.ndarray:
    mids = [c for c in ALL_COLORS if c not in (START_COLOR, END_COLOR, bg, wall)]
    rng.shuffle(mids)
    strip = [START_COLOR] + mids[: length - 2] + [END_COLOR]
    return np.array(strip, dtype=int)


def _attach_strip(maze: np.ndarray, strip: np.ndarray, edge: str) -> np.ndarray:
    if edge == "top":
        return np.vstack([strip[np.newaxis, :], maze])
    if edge == "bottom":
        return np.vstack([maze, strip[np.newaxis, :]])
    if edge == "left":
        return np.hstack([strip[:, np.newaxis], maze])
    if edge == "right":
        return np.hstack([maze, strip[:, np.newaxis]])
    raise ValueError(f"unknown edge {edge}")


def _pick_maze_shape(edge: str, strip_len: int, rng: random.Random) -> tuple[int, int]:
    if edge in {"top", "bottom"}:
        return rng.randint(7, 15), strip_len
    return strip_len, rng.randint(7, 15)


def _search_simple_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    min_len: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = shape
    visited = {start}
    path = [start]
    budget = [0]

    def rec() -> bool:
        budget[0] += 1
        if budget[0] > 25000:
            return False

        cur = path[-1]
        if cur == end:
            return len(path) >= min_len

        raw: list[tuple[int, float, tuple[int, int]]] = []
        for dr, dc in DIRS_RIGHT_FIRST:
            nr = cur[0] + dr
            nc = cur[1] + dc
            nxt = (nr, nc)
            if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                continue
            if nxt in visited:
                continue
            if nxt == end and len(path) + 1 < min_len:
                continue
            if nxt != end and abs(nr - end[0]) + abs(nc - end[1]) == 1 and len(path) + 2 < min_len:
                continue

            adj = sum((rr, cc) in visited for rr, cc in _neighbors4(nr, nc, shape))
            if adj != 1:
                continue

            dist = abs(nr - end[0]) + abs(nc - end[1])
            raw.append((dist, rng.random(), nxt))

        if not raw:
            return False

        if len(path) < int(min_len * 0.7):
            raw.sort(key=lambda item: (-item[0], item[1]))
        else:
            raw.sort(key=lambda item: (item[0], item[1]))

        for _dist, _noise, nxt in raw:
            visited.add(nxt)
            path.append(nxt)
            if rec():
                return True
            path.pop()
            visited.remove(nxt)
        return False

    return path if rec() else None


def _add_dead_end_branches(
    maze: np.ndarray,
    corridor: set[tuple[int, int]],
    wall_color: int,
    corridor_color: int,
    start: tuple[int, int],
    end: tuple[int, int],
    rng: random.Random,
) -> None:
    anchors = [cell for cell in corridor if cell not in {start, end}]
    rng.shuffle(anchors)
    branch_budget = rng.randint(1, max(1, len(corridor) // 4))
    h, w = maze.shape

    for anchor in anchors:
        if branch_budget <= 0:
            break
        cur = anchor
        steps = rng.randint(1, 3)
        carved = 0
        for _ in range(steps):
            options: list[tuple[int, int]] = []
            for dr, dc in DIRS_RIGHT_FIRST:
                nr = cur[0] + dr
                nc = cur[1] + dc
                nxt = (nr, nc)
                if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                    continue
                if int(maze[nxt]) != int(wall_color):
                    continue
                adj = sum((rr, cc) in corridor for rr, cc in _neighbors4(nr, nc, maze.shape))
                if adj == 1:
                    options.append(nxt)
            if not options:
                break
            nxt = rng.choice(options)
            maze[nxt] = int(corridor_color)
            corridor.add(nxt)
            cur = nxt
            carved += 1
        if carved:
            branch_budget -= 1


def _build_unique_maze(
    shape: tuple[int, int],
    corridor_color: int,
    wall_color: int,
    strip_len: int,
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    maze = np.full((h, w), int(wall_color), dtype=int)

    target_gap = max(4, (h + w) // 3)
    for _ in range(60):
        start = (rng.randint(1, h - 2), rng.randint(1, w - 2))
        end = (rng.randint(1, h - 2), rng.randint(1, w - 2))
        if start == end:
            continue
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) < target_gap:
            continue
        min_len = min((h - 2) * (w - 2), max(strip_len + 1, (h + w) // 2))
        path = _search_simple_path(shape, start, end, min_len, rng)
        if path is None:
            continue

        corridor = set(path)
        for r, c in path:
            maze[r, c] = int(corridor_color)
        _add_dead_end_branches(maze, corridor, wall_color, corridor_color, start, end, rng)
        maze[start] = START_COLOR
        maze[end] = END_COLOR
        return maze, start, end
    return None


def _pick_tie_endpoints(
    shape: tuple[int, int], strip_len: int, want_short: bool, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    for _ in range(100):
        if want_short:
            sr = rng.randint(1, max(1, h - 3))
            sc = rng.randint(1, max(1, w - 3))
            max_dr = min(3, h - 2 - sr)
            max_dc = min(3, w - 2 - sc)
            if max_dr < 1 or max_dc < 1:
                continue
            dr = rng.randint(1, max_dr)
            dc = rng.randint(1, max_dc)
            er = sr + dr
            ec = sc + dc
            if dr + dc + 1 >= strip_len:
                continue
        else:
            sr = rng.randint(1, max(1, (h - 1) // 3))
            sc = rng.randint(1, max(1, (w - 1) // 3))
            er = rng.randint(max(sr + 1, h // 2), h - 2)
            ec = rng.randint(max(sc + 1, w // 2), w - 2)
            if (er - sr) + (ec - sc) + 1 <= strip_len:
                continue
        return (sr, sc), (er, ec)
    return None


def _build_tie_maze(
    shape: tuple[int, int],
    corridor_color: int,
    wall_color: int,
    strip_len: int,
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    maze = np.full(shape, int(wall_color), dtype=int)
    maze[1:-1, 1:-1] = int(corridor_color)
    points = _pick_tie_endpoints(shape, strip_len, want_short=rng.random() < 0.5, rng=rng)
    if points is None:
        return None
    start, end = points
    maze[start] = START_COLOR
    maze[end] = END_COLOR
    return maze, start, end


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    edge = rng.choice(EDGE_NAMES)
    strip_len = rng.randint(5, 8)
    shape = _pick_maze_shape(edge, strip_len, rng)
    corridor_color, wall_color, _ = _choose_palette(rng)
    strip = _random_gradient(strip_len, corridor_color, wall_color, rng)

    mode = rng.choice(("unique", "tie"))
    if mode == "unique":
        built = _build_unique_maze(shape, corridor_color, wall_color, strip_len, rng)
    else:
        built = _build_tie_maze(shape, corridor_color, wall_color, strip_len, rng)
    if built is None:
        return None

    maze, start, end = built
    passable = (maze == corridor_color) | (maze == START_COLOR) | (maze == END_COLOR)
    path = _shortest_path(passable, start, end)
    count = _count_shortest_paths(passable, start, end, cap=2)
    if mode == "unique" and count != 1:
        return None
    if mode == "tie":
        if count < 2:
            return None
        sr, sc = start
        if not (sc + 1 < maze.shape[1] and sr + 1 < maze.shape[0]):
            return None
        if int(maze[sr, sc + 1]) != int(corridor_color) or int(maze[sr + 1, sc]) != int(corridor_color):
            return None
        if len(path) < 2 or path[1] != (sr, sc + 1):
            return None

    inp = _attach_strip(maze, strip, edge)
    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != maze.shape:
        return None
    if np.array_equal(out, maze):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid maze after repeated retries")
