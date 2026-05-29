# Combo variant of 88bcf3b4 (index 29)
# combo_spec: D1/2+D5/0+D6/1
# Name: 最短路多障碍串联
#
# Design analysis
# 1. The original task moves one attached segment around one nearby obstacle.
#    This combo keeps the same bar/segment/object vocabulary, but changes the
#    route into a chained shortest-path problem:
#    - D1/2: each wrap is a BFS shortest path rather than a fixed monotone
#      three-phase swing;
#    - D5/0: the segment must continue after the first obstacle and wrap the
#      next one as well;
#    - D6/1: each triple now has 2-3 separate obstacle sprites ahead of the
#      bar, so the route must visibly snake through a multi-obstacle field.
# 2. To keep the rule observable, the generator produces exactly one triple per
#    grid, with 2 or 3 same-colored obstacles placed in front of the bar at
#    alternating lateral offsets. The output route is the shortest simple path
#    that visits, in distance order from the terminal, one designated exit skin
#    cell on each obstacle. For each obstacle, that exit is the farthest walkable
#    skin cell from the current route point, so the path goes to the far side.
# 3. The input segment remains a separate irregular 1-cell-thick stroke attached
#    to the bar terminal. Its length is forced to equal the chained shortest
#    route length, so pixel count is preserved exactly.
# 4. Hidden assumptions removed from the original implementation:
#    - no monotone FL/F/FR regex;
#    - no single-obstacle logic;
#    - no final diagonal extrapolation phase.
# 5. New assumptions, enforced by generate_input():
#    - there is exactly one valid bar/segment pair in each grid;
#    - all non-bar/non-segment components are the chained obstacles for that bar;
#    - there are 2 or 3 obstacles, and the shortest chained route crosses the
#      bar axis and has multiple turns, so the fusion behavior stays visible.

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, paint_coords


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        cols = sorted(set(cs))
        return cols == list(range(min(cols), max(cols) + 1))
    if len(set(cs)) == 1:
        rows = sorted(set(rs))
        return rows == list(range(min(rows), max(rows) + 1))
    return False


def _bar_endpoints(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        c = cs[0]
        return (min(rs), c), (max(rs), c)
    r = rs[0]
    return (r, min(cs)), (r, max(cs))


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _bfs_path4(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _segment_main_path(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)

    def neighbors(p: tuple[int, int]):
        r, c = p
        for q in neighbors4(r, c):
            if q in sset:
                yield q
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            mid1 = (r, c + dc)
            mid2 = (r + dr, c)
            if mid1 in sset or mid2 in sset:
                continue
            yield q

    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    dist: dict[tuple[int, int], int] = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in neighbors(cur):
            if nxt in prev:
                continue
            prev[nxt] = cur
            dist[nxt] = dist[cur] + 1
            q.append(nxt)

    far = max(dist.keys(), key=lambda p: (dist[p], p[0], p[1]))
    path = []
    cur = far
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _segment_degrees(seg_pts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    sset = set(seg_pts)
    deg = {p: 0 for p in sset}
    for r, c in sset:
        for q in neighbors4(r, c):
            if q in sset:
                deg[(r, c)] += 1
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            q = (r + dr, c + dc)
            if q not in sset:
                continue
            mid1 = (r, c + dc)
            mid2 = (r + dr, c)
            if mid1 in sset or mid2 in sset:
                continue
            deg[(r, c)] += 1
    return deg


def _obstacle_skin(obstacle_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    skin = set()
    for r, c in obstacle_set:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in obstacle_set:
                skin.add((nr, nc))
    return skin


def _exit_key(
    p: tuple[int, int], origin: tuple[int, int], obstacle_set: set[tuple[int, int]]
) -> tuple[float, float, int, int]:
    cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
    cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
    away = (cr - origin[0], cc - origin[1])
    r, c = p
    md = abs(r - origin[0]) + abs(c - origin[1])
    dot = (r - origin[0]) * away[0] + (c - origin[1]) * away[1]
    return (md, dot, -r, -c)


def _build_local_walkable(
    blocked: set[tuple[int, int]], anchors: list[tuple[int, int]], margin: int
) -> tuple[np.ndarray, tuple[int, int]]:
    rs = [r for r, _ in blocked] + [r for r, _ in anchors]
    cs = [c for _, c in blocked] + [c for _, c in anchors]
    r0 = min(rs) - margin
    r1 = max(rs) + margin
    c0 = min(cs) - margin
    c1 = max(cs) + margin
    walkable = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in blocked:
        walkable[r - r0, c - c0] = False
    return walkable, (r0, c0)


def _local_to_idx(p: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = offset
    return (p[0] - r0, p[1] - c0)


def _idx_to_local(p: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = offset
    return (p[0] + r0, p[1] + c0)


def _ordered_obstacles(
    obstacle_sets: list[set[tuple[int, int]]], terminal: tuple[int, int]
) -> list[set[tuple[int, int]]]:
    return sorted(
        obstacle_sets,
        key=lambda comp: (
            min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp),
            min(r for r, _ in comp),
            min(c for _, c in comp),
        ),
    )


def _chain_shortest_route(
    start: tuple[int, int],
    terminal: tuple[int, int],
    obstacle_sets: list[set[tuple[int, int]]],
    static_blocked: set[tuple[int, int]],
    margin: int = 10,
) -> list[tuple[int, int]] | None:
    route = [start]
    cur = start
    visited = {start}
    for obstacle_set in _ordered_obstacles(obstacle_sets, terminal):
        segment_blocked = set(static_blocked) | (visited - {cur})
        skin = [
            p
            for p in _obstacle_skin(obstacle_set)
            if p not in segment_blocked
        ]
        if not skin:
            return None
        exit_cell = max(skin, key=lambda p: _exit_key(p, cur, obstacle_set))
        walkable, offset = _build_local_walkable(segment_blocked, [cur, exit_cell, *skin], margin)
        cur_i = _local_to_idx(cur, offset)
        exit_i = _local_to_idx(exit_cell, offset)
        walkable[cur_i] = True
        walkable[exit_i] = True
        segment_i = _bfs_path4(cur_i, exit_i, walkable)
        if segment_i is None:
            return None
        segment = [_idx_to_local(p, offset) for p in segment_i]
        if segment[0] != cur or segment[-1] != exit_cell:
            return None
        route.extend(segment[1:])
        visited.update(segment[1:])
        cur = exit_cell
    return route


def _route_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    return sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])


def _crosses_bar_axis(path: list[tuple[int, int]]) -> bool:
    cols = [c for _, c in path]
    return min(cols) < 0 < max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            for comp in _components(g, col, conn8=False):
                if len(comp) < 2 or not _is_straight_line(comp):
                    continue
                e0, e1 = _bar_endpoints(comp)
                touch0 = _touches_border(e0, h, w)
                touch1 = _touches_border(e1, h, w)
                if touch0 == touch1:
                    continue
                vertical = len({c for _, c in comp}) == 1
                border_ep = e0 if touch0 else e1
                if vertical:
                    if border_ep[0] not in (0, h - 1):
                        continue
                else:
                    if border_ep[1] not in (0, w - 1):
                        continue
                terminal_ep = e1 if touch0 else e0
                tr, tc = terminal_ep
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal_ep

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        if not neigh:
            continue

        best_seg = None
        for cand_col, cand_start in neigh:
            for comp in _components(g, cand_col, conn8=True):
                if cand_start not in comp:
                    continue
                if len(comp) < 2:
                    break
                deg = _segment_degrees(comp)
                if deg.get(cand_start, 0) != 1:
                    break
                path = _segment_main_path(comp, cand_start)
                length = len(path)
                key = (length, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        if best_seg is None:
            continue
        (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        obstacle_sets = []
        for col in colors:
            if col in (bar_col, seg_col):
                continue
            for comp in _components(g, col, conn8=True):
                obstacle_sets.append(set(comp))
        if len(obstacle_sets) < 2:
            continue

        static_blocked = set(bar_comp)
        for comp in obstacle_sets:
            static_blocked |= comp
        route = _chain_shortest_route(seg_start, terminal, obstacle_sets, static_blocked)
        if route is None or len(route) != seg_len:
            continue
        moves.append((seg_comp, seg_col, route))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _, seg_col, route in moves:
        for r, c in route:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN = 26
GRID_SIZE_MAX = 30
BAR_LEN_MIN = 3
BAR_LEN_MAX = 6
BAR_CLEARANCE = 6
SEG_LEN_MIN = 8
SEG_LEN_MAX = 22
OBS_COUNT_CHOICES = [2, 3]
OBS_SIZE_MIN = 2
OBS_SIZE_MAX = 5
OBS_ROW_MIN = 2
OBS_ROW_GAP_MIN = 3
OBS_ROW_GAP_MAX = 5
INPUT_TURN_MIN = 2
INPUT_TURN_MAX = 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 500
TRIPLE_RETRIES = 400
PLACEMENT_PAD = 1
LOCAL_MARGIN = SEG_LEN_MAX + 4
LOCAL_FWD_MAX = SEG_LEN_MAX + 8
LOCAL_SIDE_MAX = 7
INPUT_SIDE_MAX = SEG_LEN_MAX + 3

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _terminal_for(edge: str, n: int, bar_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (bar_len - 1, pos)
    if edge == "bottom":
        return (n - bar_len, pos)
    if edge == "left":
        return (pos, bar_len - 1)
    return (pos, n - bar_len)


def _transform(
    locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int
) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _grow_obstacle_local(
    start: tuple[int, int], max_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    def in_bounds(p: tuple[int, int]) -> bool:
        r, c = p
        return 1 <= r <= LOCAL_FWD_MAX and -LOCAL_SIDE_MAX <= c <= LOCAL_SIDE_MAX

    obstacle = {start}
    for _ in range(max_size - 1):
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden and in_bounds((nr, nc))
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _too_close(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        if (r, c) in b:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _build_input_path(seg_len: int, bar_len: int, side: int) -> list[tuple[int, int]] | None:
    start = (0, side)
    row_min = -(bar_len - 1)
    row_max = 2
    col_min = 1 if side > 0 else -INPUT_SIDE_MAX
    col_max = INPUT_SIDE_MAX if side > 0 else -1
    seen = {start}
    path = [start]

    def in_box(p: tuple[int, int]) -> bool:
        r, c = p
        return row_min <= r <= row_max and col_min <= c <= col_max

    def step_score(p: tuple[int, int]) -> tuple[int, int, int, float]:
        r, c = p
        farther = side * c
        vertical = -r
        branching = sum(1 for q in neighbors4(r, c) if in_box(q) and q not in seen)
        return (farther, vertical, branching, random.random())

    def dfs(turns_left: int, last_dir: tuple[int, int] | None) -> bool:
        if len(path) == seg_len:
            return turns_left <= 0
        cur = path[-1]
        moves = []
        for nxt in neighbors4(cur[0], cur[1]):
            if nxt in seen or not in_box(nxt):
                continue
            dr = nxt[0] - cur[0]
            dc = nxt[1] - cur[1]
            if side * nxt[1] <= 0:
                continue
            if last_dir is not None and (dr, dc) != last_dir:
                next_turns = turns_left - 1
            else:
                next_turns = turns_left
            if next_turns < 0:
                continue
            moves.append((step_score(nxt), nxt, next_turns, (dr, dc)))
        random.shuffle(moves)
        moves.sort(reverse=True)
        for _, nxt, next_turns, step_dir in moves:
            seen.add(nxt)
            path.append(nxt)
            if dfs(next_turns, step_dir):
                return True
            path.pop()
            seen.remove(nxt)
        return False

    turn_target = random.randint(INPUT_TURN_MIN, min(INPUT_TURN_MAX, seg_len - 2))
    if dfs(turn_target, None):
        return path
    return None


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _only_root_touches_bar(seg: list[tuple[int, int]], bar: list[tuple[int, int]]) -> bool:
    bar_set = set(bar)
    root = seg[0]
    touched = set()
    for p in seg:
        if p in bar_set:
            touched.add(p)
        for q in neighbors4(p[0], p[1]):
            if q in bar_set:
                touched.add(p)
    return touched <= {root}


def _make_obstacle_chain(side: int, bar_set: set[tuple[int, int]], seg_start: tuple[int, int]) -> list[list[tuple[int, int]]] | None:
    obstacle_count = random.choice(OBS_COUNT_CHOICES)
    if obstacle_count == 3:
        row_gap_min = 2
        row_gap_max = 3
        size_max = min(OBS_SIZE_MAX, 3)
    else:
        row_gap_min = OBS_ROW_GAP_MIN
        row_gap_max = OBS_ROW_GAP_MAX
        size_max = OBS_SIZE_MAX
    obstacles: list[list[tuple[int, int]]] = []
    forbidden = set(bar_set) | {seg_start}
    row = OBS_ROW_MIN + random.randint(0, 1)
    for idx in range(obstacle_count):
        row += 0 if idx == 0 else random.randint(row_gap_min, row_gap_max)
        if row > LOCAL_FWD_MAX - 1:
            return None
        if idx == 0:
            col_choices = [0, side, 2 * side]
        else:
            swing = -side if idx % 2 == 1 else side
            col_choices = [swing, 2 * swing, 3 * swing]
        random.shuffle(col_choices)
        built = None
        for col in col_choices:
            obstacle = _grow_obstacle_local(
                (row, col),
                random.randint(OBS_SIZE_MIN, size_max),
                forbidden,
            )
            obstacle_set = set(obstacle)
            if any(r <= 0 for r, _ in obstacle_set):
                continue
            if any(_too_close(obstacle_set, set(prev)) for prev in obstacles):
                continue
            if _too_close(obstacle_set, bar_set | {seg_start}):
                continue
            built = obstacle
            break
        if built is None:
            return None
        obstacles.append(built)
        forbidden |= set(built)
    return obstacles


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    for _ in range(TRIPLE_RETRIES):
        bar_len = random.randint(BAR_LEN_MIN, max_bar)
        side = random.choice((-1, 1))
        bar = [(-k, 0) for k in range(bar_len)]
        bar_set = set(bar)
        seg_start = (0, side)

        obstacles = _make_obstacle_chain(side, bar_set, seg_start)
        if obstacles is None:
            continue
        obstacle_sets = [set(obstacle) for obstacle in obstacles]
        static_blocked = set(bar_set)
        for obstacle in obstacle_sets:
            static_blocked |= obstacle
        route = _chain_shortest_route(seg_start, (0, 0), obstacle_sets, static_blocked, margin=LOCAL_MARGIN)
        if route is None:
            continue
        if len(route) < SEG_LEN_MIN or len(route) > SEG_LEN_MAX:
            continue
        if _route_turns(route) < len(obstacles):
            continue
        if not _crosses_bar_axis(route):
            continue
        if not _only_root_touches_bar(route, bar):
            continue

        input_seg = _build_input_path(len(route), bar_len, side)
        if input_seg is None:
            continue
        all_obstacle_cells = [p for obstacle in obstacles for p in obstacle]
        if _is_4adj(input_seg, all_obstacle_cells):
            continue
        if not _only_root_touches_bar(input_seg, bar):
            continue

        union = bar + all_obstacle_cells + input_seg + route
        rs = [r for r, _ in union]
        cs = [c for _, c in union]
        return {
            "bar_len": bar_len,
            "bar": bar,
            "obstacles": obstacles,
            "input_seg": input_seg,
            "output_seg": route,
            "bbox": (min(rs), min(cs), max(rs), max(cs)),
        }
    return None


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _place_triple(triple: dict, edge: str, n: int) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    obstacle_cells = [p for obstacle in triple["obstacles"] for p in obstacle]
    union = triple["bar"] + obstacle_cells + triple["input_seg"] + triple["output_seg"]
    bar_len = triple["bar_len"]
    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        return {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacles": [_transform(obstacle, edge, n, bar_len, pos) for obstacle in triple["obstacles"]],
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "reserve": _expanded_bbox(union_g, n),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)

    triple = _make_triple(n)
    if triple is None:
        return None
    edge = random.choice(list(EDGE_DATA))
    placed = _place_triple(triple, edge, n)
    if placed is None:
        return None

    paint_coords(inp, placed["bar"], bar_color)
    for obstacle in placed["obstacles"]:
        paint_coords(inp, obstacle, obstacle_color)
    paint_coords(inp, placed["input_seg"], seg_color)

    try:
        solved = main(inp)
        if np.array_equal(solved, inp):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
