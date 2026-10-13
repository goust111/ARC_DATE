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

GRID_MIN_SIZE = 18
GRID_MAX_SIZE = 28
BOX_MIN_SIZE = 7
BOX_MAX_SIZE = 13
PATH_MIN_STEPS = 6
PATH_MAX_STEPS = 14
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
    route = list(path[1:])
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
    - Starting from the first path cell after the start marker, spend exactly the bar length along the unique shortest path toward the end marker.
    - If the budget ends early, stop there; if it reaches or passes the end marker, keep the markers unchanged and continue straight beyond the end in the final travel direction until the full budget is spent.

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


def _is_interior(box_h: int, box_w: int, r: int, c: int) -> bool:
    return 0 < r < box_h - 1 and 0 < c < box_w - 1


def _random_end_cell(box_h: int, box_w: int, exit_side: str) -> tuple[int, int]:
    if exit_side == "top":
        return 0, random.randint(2, box_w - 3)
    if exit_side == "bottom":
        return box_h - 1, random.randint(2, box_w - 3)
    if exit_side == "left":
        return random.randint(2, box_h - 3), 0
    return random.randint(2, box_h - 3), box_w - 1


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        prev_step = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_step = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if prev_step != next_step:
            turns += 1
    return turns


def _can_append_path_cell(
    nxt: tuple[int, int],
    current: tuple[int, int],
    used: set[tuple[int, int]],
    box_h: int,
    box_w: int,
) -> bool:
    if not _is_interior(box_h, box_w, nxt[0], nxt[1]):
        return False
    if nxt in used:
        return False
    for ar, ac in neighbors4(nxt[0], nxt[1]):
        if (ar, ac) in used and (ar, ac) != current:
            return False
    return True


def _dfs_reverse_path(
    path_rev: list[tuple[int, int]],
    used: set[tuple[int, int]],
    remaining_steps: int,
    box_h: int,
    box_w: int,
) -> bool:
    if remaining_steps == 0:
        return True

    current = path_rev[-1]
    choices = []
    for nr, nc in neighbors4(current[0], current[1]):
        nxt = (nr, nc)
        if not _can_append_path_cell(nxt, current, used, box_h, box_w):
            continue
        free_neighbors = sum(
            1
            for rr, cc in neighbors4(nr, nc)
            if _is_interior(box_h, box_w, rr, cc) and (rr, cc) not in used
        )
        choices.append((free_neighbors, random.random(), nxt))

    choices.sort(reverse=True)
    for _score, _tie_break, nxt in choices:
        used.add(nxt)
        path_rev.append(nxt)
        if _dfs_reverse_path(path_rev, used, remaining_steps - 1, box_h, box_w):
            return True
        path_rev.pop()
        used.remove(nxt)
    return False


def _generate_local_path(box_h: int, box_w: int, exit_side: str) -> list[tuple[int, int]] | None:
    last_dir = SIDE_TO_LAST_DIR[exit_side]
    inward = (-last_dir[0], -last_dir[1])
    max_steps = min(PATH_MAX_STEPS, (box_h - 2) * (box_w - 2) - 1)
    if max_steps < PATH_MIN_STEPS:
        return None

    for _ in range(200):
        total_steps = random.randint(PATH_MIN_STEPS, max_steps)
        end = _random_end_cell(box_h, box_w, exit_side)
        first = (end[0] + inward[0], end[1] + inward[1])
        if not _is_interior(box_h, box_w, first[0], first[1]):
            continue

        path_rev = [end, first]
        used = {end, first}
        if not _dfs_reverse_path(path_rev, used, total_steps - 1, box_h, box_w):
            continue

        path = list(reversed(path_rev))
        turns = _turn_count(path)
        if turns == 0 and random.random() < 0.8:
            continue
        return path
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
    box_h = random.randint(BOX_MIN_SIZE, min(BOX_MAX_SIZE, h - 6))
    box_w = random.randint(BOX_MIN_SIZE, min(BOX_MAX_SIZE, w - 6))

    exit_side = random.choice(tuple(SIDE_TO_LAST_DIR))
    local_path = _generate_local_path(box_h, box_w, exit_side)
    if local_path is None:
        return None

    path_steps = len(local_path) - 1
    max_exit_margin = _max_exit_margin(exit_side, h, w, box_h, box_w)
    mode, bar_length, extra = _choose_branch(path_steps, max_exit_margin)

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
    for r, c in path:
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

    route = _rope_route(shortest, scene["bar_length"], scene["passable"])
    if mode == "short" and route[-1] == end:
        return None
    if mode == "exact" and route[-1] != end:
        return None
    if mode == "long" and len(route) <= path_steps:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fixed-rope input")
