from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, neighbors4


START_COLOR = Color.RED
END_COLOR = Color.GREEN
BG_COLOR = Color.BLACK

ROPE_COLOR_CHOICES = (Color.BLUE, Color.YELLOW, Color.PINK, Color.ORANGE)
OBSTACLE_COLOR_CHOICES = (Color.GREY, Color.TEAL, Color.MAROON)

GRID_MIN_SIZE = 22
GRID_MAX_SIZE = 28
BOX_MIN_SIZE = 13
BOX_MAX_SIZE = 17
PATH_MIN_STEPS = 14
PATH_MAX_STEPS = 22
PATH_MIN_TURNS = 6
PATH_MIN_DETOUR = 6
PATH_MIN_DECISIONS = 3
MAZE_MIN_DEAD_ENDS = 4
OVERLONG_EXTRA_MAX = 5
MAX_GENERATION_ATTEMPTS = 500

SIDE_TO_LAST_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _coords_list(coords: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in coords]


def _is_edge_bar(component: dict, shape: tuple[int, int]) -> bool:
    h, w = shape
    area = int(component["area"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    if area < 2:
        return False
    if r0 == r1 and r0 in (0, h - 1):
        return area == c1 - c0 + 1
    if c0 == c1 and c0 in (0, w - 1):
        return area == r1 - r0 + 1
    return False


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    start_pts = _coords_list(np.argwhere(g == START_COLOR))
    end_pts = _coords_list(np.argwhere(g == END_COLOR))
    assert len(start_pts) == 1, "expected exactly one start marker"
    assert len(end_pts) == 1, "expected exactly one end marker"

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    bar_candidates = [
        comp
        for comp in components
        if int(comp["color"]) not in (START_COLOR, END_COLOR) and _is_edge_bar(comp, g.shape)
    ]
    assert len(bar_candidates) == 1, "expected exactly one edge rope bar"
    bar = bar_candidates[0]
    rope_color = int(bar["color"])
    bar_coords = _coords_list(bar["coords"])
    assert int(np.count_nonzero(g == rope_color)) == len(bar_coords), "rope color must appear only on the bar"

    obstacle_mask = (g != bg) & (g != START_COLOR) & (g != END_COLOR) & (g != rope_color)
    passable = (g == bg) | (g == START_COLOR) | (g == END_COLOR)
    assert passable[start_pts[0]] and passable[end_pts[0]]

    return {
        "grid": g,
        "bg": bg,
        "start": start_pts[0],
        "end": end_pts[0],
        "rope_color": rope_color,
        "bar_coords": bar_coords,
        "bar_length": len(bar_coords),
        "obstacle_mask": obstacle_mask,
        "passable": passable,
    }


def _bfs_distances(passable: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque([src])
    dist[src] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(passable.shape, nr, nc):
                continue
            if not passable[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _unique_shortest_path(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    dist_start = _bfs_distances(passable, start)
    total = int(dist_start[end])
    assert total > 0, "start and end must be connected by at least one step"
    dist_end = _bfs_distances(passable, end)

    path = [start]
    cur = start
    while cur != end:
        r, c = cur
        candidates = []
        for nr, nc in neighbors4(r, c):
            if not in_bounds(passable.shape, nr, nc) or not passable[nr, nc]:
                continue
            if dist_start[nr, nc] != dist_start[r, c] + 1:
                continue
            if dist_start[nr, nc] + dist_end[nr, nc] != total:
                continue
            candidates.append((nr, nc))
        assert len(candidates) == 1, "shortest path must be unique"
        cur = candidates[0]
        path.append(cur)
    return path


def _rope_route(
    path: list[tuple[int, int]],
    budget: int,
    passable: np.ndarray,
) -> list[tuple[int, int]]:
    # Start and end are persistent markers, so neither consumes a visible
    # rope pixel.  The fillable corridor is strictly between them.
    route = list(path[1:-1])
    if budget <= len(route):
        return route[:budget]

    dr = path[-1][0] - path[-2][0]
    dc = path[-1][1] - path[-2][1]
    cur = path[-1]
    for _ in range(budget - len(route)):
        cur = (cur[0] + dr, cur[1] + dc)
        assert in_bounds(passable.shape, cur[0], cur[1]), "rope extension must stay inside the grid"
        assert passable[cur], "rope extension must stay on free cells"
        route.append(cur)
    return route


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the edge rope bar with an equally long run laid along the unique shortest route from the start marker.

    Input structure:
    - A background canvas contains one start-marker pixel, one end-marker pixel, and obstacle walls that force a unique 4-neighbor shortest path between them.
    - A separate straight bar of rope color lies flush against one grid edge; its length is the rope budget.

    Rule:
    - Erase the edge bar.
    - Starting from the first path cell after the start marker, place exactly the bar length in background cells along the unique shortest path toward the end marker.
    - The persistent end marker does not consume rope material. If the budget exceeds the number of background corridor cells before it, cross over the marker and continue straight beyond it until all rope pixels are visible.

    Key insight:
    - The border bar is conserved material, not a generic connection cue: first recover the unique geodesic, then truncate it or overshoot it by the bar's exact pixel count.
    """
    scene = _parse_scene(grid)
    path = _unique_shortest_path(scene["passable"], scene["start"], scene["end"])
    route = _rope_route(path, scene["bar_length"], scene["passable"])

    out = np.array(scene["grid"], copy=True)
    for r, c in scene["bar_coords"]:
        out[r, c] = scene["bg"]

    protected = {scene["start"], scene["end"]}
    for r, c in route:
        if (r, c) in protected:
            continue
        assert out[r, c] == scene["bg"], "rope may only occupy free background cells"
        out[r, c] = scene["rope_color"]
    return out


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        prev_step = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_step = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if prev_step != next_step:
            turns += 1
    return turns


def _has_alternate_route(
    passable: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    shortest_path: list[tuple[int, int]],
) -> bool:
    for a, b in zip(shortest_path, shortest_path[1:]):
        banned = frozenset((a, b))
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            if cur == end:
                return True
            for nxt in neighbors4(*cur):
                if not in_bounds(passable.shape, *nxt) or not bool(passable[nxt]):
                    continue
                if nxt in seen or frozenset((cur, nxt)) == banned:
                    continue
                seen.add(nxt)
                q.append(nxt)
    return False


def _count_simple_routes(
    passable: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    limit: int = 4,
) -> int:
    count = 0
    seen = {start}

    def dfs(cur: tuple[int, int]) -> None:
        nonlocal count
        if count >= limit:
            return
        if cur == end:
            count += 1
            return
        choices = [
            nxt
            for nxt in neighbors4(*cur)
            if in_bounds(passable.shape, *nxt) and bool(passable[nxt]) and nxt not in seen
        ]
        random.shuffle(choices)
        for nxt in choices:
            seen.add(nxt)
            dfs(nxt)
            seen.remove(nxt)
            if count >= limit:
                return

    dfs(start)
    return count


def _generate_local_maze(
    box_h: int,
    box_w: int,
    exit_side: str,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Build a looped maze with at least three routes and one unique shortest route."""
    coarse_h = (box_h - 1) // 2
    coarse_w = (box_w - 1) // 2
    nodes = [(r, c) for r in range(coarse_h) for c in range(coarse_w)]
    node_coords = {(r, c): (1 + 2 * r, 1 + 2 * c) for r, c in nodes}

    def coarse_neighbors(node: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = node
        return [
            (nr, nc)
            for nr, nc in neighbors4(r, c)
            if 0 <= nr < coarse_h and 0 <= nc < coarse_w
        ]

    passable = np.zeros((box_h, box_w), dtype=bool)
    root = random.choice(nodes)
    visited = {root}
    stack = [root]
    passable[node_coords[root]] = True
    tree_edges: set[frozenset[tuple[int, int]]] = set()
    while stack:
        cur = stack[-1]
        choices = [nb for nb in coarse_neighbors(cur) if nb not in visited]
        if not choices:
            stack.pop()
            continue
        nxt = random.choice(choices)
        visited.add(nxt)
        stack.append(nxt)
        tree_edges.add(frozenset((cur, nxt)))
        a = node_coords[cur]
        b = node_coords[nxt]
        passable[a] = True
        passable[b] = True
        passable[(a[0] + b[0]) // 2, (a[1] + b[1]) // 2] = True

    loop_edges = []
    for node in nodes:
        for nb in coarse_neighbors(node):
            edge = frozenset((node, nb))
            if nb < node or edge in tree_edges:
                continue
            loop_edges.append((node, nb))
    random.shuffle(loop_edges)
    loop_count = random.randint(3, min(8, len(loop_edges)))
    for a_node, b_node in loop_edges[:loop_count]:
        a = node_coords[a_node]
        b = node_coords[b_node]
        passable[(a[0] + b[0]) // 2, (a[1] + b[1]) // 2] = True

    if exit_side == "top":
        exit_node = (0, random.randrange(coarse_w))
        end = (0, node_coords[exit_node][1])
    elif exit_side == "bottom":
        exit_node = (coarse_h - 1, random.randrange(coarse_w))
        end = (box_h - 1, node_coords[exit_node][1])
    elif exit_side == "left":
        exit_node = (random.randrange(coarse_h), 0)
        end = (node_coords[exit_node][0], 0)
    else:
        exit_node = (random.randrange(coarse_h), coarse_w - 1)
        end = (node_coords[exit_node][0], box_w - 1)
    passable[end] = True

    starts = [node_coords[node] for node in nodes if node != exit_node]
    random.shuffle(starts)
    for start in starts:
        try:
            path = _unique_shortest_path(passable, start, end)
        except AssertionError:
            continue
        steps = len(path) - 1
        if not (PATH_MIN_STEPS <= steps <= PATH_MAX_STEPS):
            continue
        if _turn_count(path) < PATH_MIN_TURNS:
            continue
        manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
        if steps - manhattan < PATH_MIN_DETOUR:
            continue
        decisions = sum(
            sum(
                in_bounds(passable.shape, *nb) and bool(passable[nb])
                for nb in neighbors4(*cell)
            ) >= 3
            for cell in path[1:-1]
        )
        if decisions < PATH_MIN_DECISIONS:
            continue
        dead_ends = sum(
            sum(
                in_bounds(passable.shape, *nb) and bool(passable[nb])
                for nb in neighbors4(*cell)
            ) == 1
            for cell in map(tuple, np.argwhere(passable))
        )
        if dead_ends < MAZE_MIN_DEAD_ENDS:
            continue
        if _count_simple_routes(passable, start, end, limit=5) < 4:
            continue
        return passable, path
    return None


def _max_exit_margin(exit_side: str, h: int, w: int, box_h: int, box_w: int) -> int:
    if exit_side in ("top", "bottom"):
        return h - box_h - 1
    return w - box_w - 1


def _choose_branch(path_steps: int, max_exit_margin: int) -> tuple[str, int, int]:
    mode = random.choice(("short", "exact", "long"))
    if mode == "long" and max_exit_margin < 1:
        mode = random.choice(("short", "exact"))

    if mode == "short":
        return mode, random.randint(2, path_steps - 1), 0
    if mode == "exact":
        return mode, path_steps, 0
    extra = random.randint(1, min(OVERLONG_EXTRA_MAX, max_exit_margin))
    return mode, path_steps + extra, extra


def _choose_box_origin(
    h: int,
    w: int,
    box_h: int,
    box_w: int,
    exit_side: str,
    required_exit_margin: int,
) -> tuple[int, int]:
    if exit_side == "top":
        top = random.randint(required_exit_margin, h - box_h - 1)
        left = random.randint(1, w - box_w - 1)
        return top, left
    if exit_side == "bottom":
        top = random.randint(1, h - box_h - required_exit_margin)
        left = random.randint(1, w - box_w - 1)
        return top, left
    if exit_side == "left":
        top = random.randint(1, h - box_h - 1)
        left = random.randint(required_exit_margin, w - box_w - 1)
        return top, left
    top = random.randint(1, h - box_h - 1)
    left = random.randint(1, w - box_w - required_exit_margin)
    return top, left


def _translate_path(path: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in path]


def _bar_capacity(side: str, h: int, w: int) -> int:
    return w if side in ("top", "bottom") else h


def _random_bar_coords(side: str, length: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        c0 = random.randint(0, w - length)
        return [(0, c) for c in range(c0, c0 + length)]
    if side == "bottom":
        c0 = random.randint(0, w - length)
        return [(h - 1, c) for c in range(c0, c0 + length)]
    if side == "left":
        r0 = random.randint(0, h - length)
        return [(r, 0) for r in range(r0, r0 + length)]
    r0 = random.randint(0, h - length)
    return [(r, w - 1) for r in range(r0, r0 + length)]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    w = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    box_h_choices = [size for size in range(BOX_MIN_SIZE, min(BOX_MAX_SIZE, h - 6) + 1) if size % 2 == 1]
    box_w_choices = [size for size in range(BOX_MIN_SIZE, min(BOX_MAX_SIZE, w - 6) + 1) if size % 2 == 1]
    if not box_h_choices or not box_w_choices:
        return None
    box_h = random.choice(box_h_choices)
    box_w = random.choice(box_w_choices)

    exit_side = random.choice(tuple(SIDE_TO_LAST_DIR))
    maze = _generate_local_maze(box_h, box_w, exit_side)
    if maze is None:
        return None
    local_passable, local_path = maze

    corridor_cells = len(local_path) - 2
    max_exit_margin = _max_exit_margin(exit_side, h, w, box_h, box_w)
    mode, bar_length, extra = _choose_branch(corridor_cells, max_exit_margin)

    required_exit_margin = max(1, extra)
    top, left = _choose_box_origin(h, w, box_h, box_w, exit_side, required_exit_margin)
    path = _translate_path(local_path, top, left)

    rope_color = random.choice(ROPE_COLOR_CHOICES)
    obstacle_color = random.choice([c for c in OBSTACLE_COLOR_CHOICES if c != rope_color])

    bar_side_choices = [
        side
        for side in SIDE_TO_LAST_DIR
        if side != exit_side and _bar_capacity(side, h, w) >= bar_length
    ]
    if not bar_side_choices:
        return None
    bar_side = random.choice(bar_side_choices)
    bar_coords = _random_bar_coords(bar_side, bar_length, h, w)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[top : top + box_h, left : left + box_w] = obstacle_color
    for local_r, local_c in np.argwhere(local_passable):
        r = top + int(local_r)
        c = left + int(local_c)
        grid[r, c] = BG_COLOR

    start = path[0]
    end = path[-1]
    grid[start] = START_COLOR
    grid[end] = END_COLOR
    for r, c in bar_coords:
        if grid[r, c] != BG_COLOR:
            return None
        grid[r, c] = rope_color

    scene = _parse_scene(grid)
    shortest = _unique_shortest_path(scene["passable"], scene["start"], scene["end"])
    if shortest != path:
        return None
    if not _has_alternate_route(scene["passable"], scene["start"], scene["end"], shortest):
        return None
    if _count_simple_routes(scene["passable"], scene["start"], scene["end"], limit=5) < 4:
        return None

    route = _rope_route(shortest, scene["bar_length"], scene["passable"])
    if len(route) != scene["bar_length"] or end in route or start in route:
        return None
    if mode == "short" and not scene["bar_length"] < corridor_cells:
        return None
    if mode == "exact" and scene["bar_length"] != corridor_cells:
        return None
    if mode == "long" and not scene["bar_length"] > corridor_cells:
        return None

    out = main(grid)
    if int(np.count_nonzero(out == rope_color)) != scene["bar_length"]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fixed-rope input")
