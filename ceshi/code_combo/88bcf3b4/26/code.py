# Combo variant of 88bcf3b4 (index 26)
# combo_spec: D8/2+D11/0+D13/1
# Name: 尺寸排序共享障碍级联
#
# Design analysis
# ---------------
# 1. Fusion rule:
#    - Keep the original bar + attached irregular segment + static obstacle setting.
#    - D13/1 changes cardinality: the largest obstacle contributes two "slots", so two bars from different edges can
#      legitimately share it and wrap on opposite flanks.
#    - D8/2 applies to ranked slots rather than a strict 1:1 obstacle list: bars are sorted by segment length, obstacle
#      slots are sorted by backing obstacle size, and the two longest bars both map to the largest obstacle.
#    - D11/0 applies after that fixed assignment: bars are processed from the terminal nearest the grid border to the
#      farthest, and each solved rope is painted back immediately so it blocks later ropes.
# 2. Observability:
#    - Every generated sample contains exactly three bars and two obstacles.
#    - The largest obstacle is visibly larger and visibly shared by the two longest ropes; the shortest rope uses the
#      small obstacle.
#    - One of the shared ropes is deliberately closer to the small obstacle, so nearest-obstacle reasoning fails and
#      size ranking is required.
#    - The earlier shared rope occupies cells that the later shared rope would use in a parallel solve, so the cascade
#      remains visible.
# 3. Cleanliness / audited assumptions:
#    - Object discovery is done on the original input only; sequential outputs are used purely as new blockers.
#    - The largest obstacle is the only one allowed to be shared, which keeps the meta rule human-readable.
#    - Colors never encode rank or order; only visible segment length, obstacle size, and edge depth matter.

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


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


def _edge_distance(p: tuple[int, int], h: int, w: int) -> int:
    r, c = p
    return min(r, c, h - 1 - r, w - 1 - c)


def _bfs_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
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


def _bfs_path_with_diag_filter(
    start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray
) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}

    def can_step(r: int, c: int, nr: int, nc: int) -> bool:
        dr = nr - r
        dc = nc - c
        if dr == 0 or dc == 0:
            return True
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if 0 <= mid1[0] < h and 0 <= mid1[1] < w and bool(walkable[mid1]):
            return False
        if 0 <= mid2[0] < h and 0 <= mid2[1] < w and bool(walkable[mid2]):
            return False
        return True

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            if not can_step(r, c, nr, nc):
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


def _greedy_manhattan_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape

    def md(p: tuple[int, int]) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    cur = start
    path = [cur]
    seen = {cur}
    while cur != goal:
        cur_md = md(cur)
        best = None
        for nr, nc in neighbors8(cur[0], cur[1]):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            nmd = md(nxt)
            if nmd >= cur_md:
                continue
            is_diag = 1 if (nr != cur[0] and nc != cur[1]) else 0
            key = (nmd, is_diag, nr, nc)
            if best is None or key < best:
                best = key
        if best is None:
            return None
        _, _, nr, nc = best
        cur = (nr, nc)
        path.append(cur)
        seen.add(cur)
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
            if (r, c + dc) in sset or (r + dr, c) in sset:
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
            if (r, c + dc) in sset or (r + dr, c) in sset:
                continue
            deg[(r, c)] += 1
    return deg


def _bar_frame(comp: list[tuple[int, int]], h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    e0, e1 = _bar_endpoints(comp)
    touch0 = _touches_border(e0, h, w)
    touch1 = _touches_border(e1, h, w)
    assert touch0 != touch1
    border_ep = e0 if touch0 else e1
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        return ((1, 0), (0, 1)) if border_ep[0] == 0 else ((-1, 0), (0, -1))
    return ((0, 1), (-1, 0)) if border_ep[1] == 0 else ((0, -1), (1, 0))


def _discover_scene(g: np.ndarray) -> tuple[list[dict], list[dict], int]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    infos = []

    for col in colors:
        for comp in _components(g, col, conn8=False):
            if len(comp) < 2 or not _is_straight_line(comp):
                continue
            e0, e1 = _bar_endpoints(comp)
            touch0 = _touches_border(e0, h, w)
            touch1 = _touches_border(e1, h, w)
            if touch0 == touch1:
                continue
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            vertical = len(set(cs)) == 1
            border_ep = e0 if touch0 else e1
            if vertical and border_ep[0] not in (0, h - 1):
                continue
            if (not vertical) and border_ep[1] not in (0, w - 1):
                continue

            terminal = e1 if touch0 else e0
            neigh = []
            for nr, nc in neighbors4(*terminal):
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(g[nr, nc])
                    if v != bg and v != col:
                        neigh.append((v, (nr, nc)))
            if not neigh:
                continue

            best_seg = None
            for cand_col, cand_start in neigh:
                for seg_comp in _components(g, cand_col, conn8=True):
                    if cand_start not in seg_comp:
                        continue
                    if len(seg_comp) < 2:
                        break
                    deg = _segment_degrees(seg_comp)
                    if deg.get(cand_start, 0) != 1:
                        break
                    path = _segment_main_path(seg_comp, cand_start)
                    key = (len(path), len(seg_comp), cand_col, cand_start[0], cand_start[1])
                    if best_seg is None or key > best_seg[0]:
                        best_seg = (key, cand_col, cand_start, seg_comp)
                    break
            if best_seg is None:
                continue

            (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg
            fwd, _lat = _bar_frame(comp, h, w)
            infos.append(
                {
                    "bar_col": col,
                    "bar_comp": list(comp),
                    "terminal": terminal,
                    "seg_col": seg_col,
                    "seg_start": seg_start,
                    "seg_comp": list(seg_comp),
                    "seg_len": seg_len,
                    "off": (seg_start[0] - terminal[0], seg_start[1] - terminal[1]),
                    "fwd": fwd,
                    "edge_distance": _edge_distance(terminal, h, w),
                }
            )

    ignored_colors = {info["bar_col"] for info in infos} | {info["seg_col"] for info in infos}
    obstacles = []
    for col in colors:
        if col in ignored_colors:
            continue
        for comp in _components(g, col, conn8=True):
            obstacles.append(
                {
                    "color": col,
                    "comp": list(comp),
                    "set": set(comp),
                    "size": len(comp),
                    "min_r": min(r for r, _ in comp),
                    "min_c": min(c for _, c in comp),
                }
            )

    return infos, obstacles, bg


def _sorted_bar_indices(bar_infos: list[dict]) -> list[int]:
    return sorted(
        range(len(bar_infos)),
        key=lambda i: (
            -bar_infos[i]["seg_len"],
            bar_infos[i]["terminal"][0],
            bar_infos[i]["terminal"][1],
            bar_infos[i]["bar_col"],
            bar_infos[i]["seg_col"],
        ),
    )


def _sorted_obstacles(obstacles: list[dict]) -> list[dict]:
    return sorted(obstacles, key=lambda obs: (-obs["size"], obs["min_r"], obs["min_c"]))


def _assign_size_shared(bar_infos: list[dict], obstacles: list[dict]) -> dict[int, dict]:
    obs_sorted = _sorted_obstacles(obstacles)
    assert len(obs_sorted) >= 2, "generator and solver expect at least two obstacles"
    slots: list[dict] = []
    for obs_rank, obs in enumerate(obs_sorted):
        cap = 2 if obs_rank == 0 else 1
        for _ in range(cap):
            slots.append(obs)
    assert len(slots) >= len(bar_infos), "not enough obstacle slots for bars"
    out = {}
    for rank, bar_idx in enumerate(_sorted_bar_indices(bar_infos)):
        out[bar_idx] = slots[rank]
    return out


def _assign_nearest(bar_infos: list[dict], obstacles: list[dict]) -> dict[int, dict]:
    out = {}
    for idx, info in enumerate(bar_infos):
        tr, tc = info["terminal"]
        best = None
        best_obs = None
        for obs in obstacles:
            dist = min(abs(r - tr) + abs(c - tc) for r, c in obs["comp"])
            key = (dist, obs["min_r"], obs["min_c"], -obs["size"])
            if best is None or key < best:
                best = key
                best_obs = obs
        assert best_obs is not None
        out[idx] = best_obs
    return out


def _route_segment(work: np.ndarray, info: dict, obstacle: dict, bg: int) -> list[tuple[int, int]]:
    h, w = work.shape
    terminal = info["terminal"]
    seg_start = info["seg_start"]
    seg_len = info["seg_len"]
    seg_col = info["seg_col"]
    off = info["off"]
    fwd = info["fwd"]
    obstacle_set = obstacle["set"]
    cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
    cc = sum(c for _, c in obstacle_set) / len(obstacle_set)

    skin = set()
    for r, c in obstacle_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                skin.add((nr, nc))

    def cell_ok(p: tuple[int, int]) -> bool:
        # Sequentially painted ropes must block later ropes even when all segments share one color.
        # The current rope has already been erased from `work`, so only background cells stay traversable.
        return int(work[p]) == bg

    def lane_score(p: tuple[int, int]) -> float:
        return (p[0] - cr) * off[0] + (p[1] - cc) * off[1]

    def fwd_score(p: tuple[int, int]) -> int:
        return (p[0] - terminal[0]) * fwd[0] + (p[1] - terminal[1]) * fwd[1]

    entry_candidates = []
    for p in skin:
        if not cell_ok(p):
            continue
        anchor = (p[0] - off[0], p[1] - off[1])
        if anchor not in obstacle_set:
            continue
        dist = abs(p[0] - terminal[0]) + abs(p[1] - terminal[1])
        entry_candidates.append((dist, -lane_score(p), -fwd_score(p), p[0], p[1], p))
    assert entry_candidates, "no valid side-specific entry for obstacle"
    entry = min(entry_candidates)[-1]

    walkable = work == bg
    walkable[seg_start] = True
    walkable[entry] = True
    approach = _greedy_manhattan_path(seg_start, entry, walkable) or _bfs_path(seg_start, entry, walkable)
    assert approach is not None, f"no approach path from {seg_start} to {entry}"

    lane_skin = [p for p in skin if cell_ok(p) and lane_score(p) >= -0.01]
    if entry not in lane_skin:
        lane_skin.append(entry)
    exit_cell = max(
        lane_skin,
        key=lambda p: (fwd_score(p), lane_score(p), abs(p[0] - terminal[0]) + abs(p[1] - terminal[1]), -p[0], -p[1]),
    )

    skin_walkable = np.zeros((h, w), dtype=bool)
    for r, c in lane_skin:
        skin_walkable[r, c] = True
    skin_walkable[entry] = True
    skin_walkable[exit_cell] = True
    boundary = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]

    route = list(approach)
    if boundary and boundary[0] == entry:
        route.extend(boundary[1:])

    if len(boundary) >= 2:
        dr_ext = boundary[-1][0] - boundary[-2][0]
        dc_ext = boundary[-1][1] - boundary[-2][1]
    elif len(route) >= 2:
        dr_ext = route[-1][0] - route[-2][0]
        dc_ext = route[-1][1] - route[-2][1]
    else:
        dr_ext, dc_ext = fwd
    if dr_ext * fwd[0] + dc_ext * fwd[1] <= 0:
        dr_ext, dc_ext = fwd

    used = set(route)
    cur = route[-1]
    while len(route) < seg_len:
        candidates = [
            (dr_ext, dc_ext),
            fwd,
            (fwd[0] + off[0], fwd[1] + off[1]),
            (fwd[0] - off[0], fwd[1] - off[1]),
        ]
        moved = False
        for dr, dc in candidates:
            if dr == 0 and dc == 0:
                continue
            nxt = (cur[0] + dr, cur[1] + dc)
            nr, nc = nxt
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in used or nxt in obstacle_set:
                continue
            if int(work[nr, nc]) not in (bg, seg_col):
                continue
            route.append(nxt)
            used.add(nxt)
            cur = nxt
            moved = True
            break
        if not moved:
            break

    assert len(route) >= seg_len, "route could not preserve segment length"
    return route[:seg_len]


def _solve_with_assignment(grid: np.ndarray, sequential: bool, assign_mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bar_infos, obstacles, bg = _discover_scene(g)
    assert len(bar_infos) == 3, "combo generator expects exactly three bars"
    assert len(obstacles) >= 2, "combo generator expects at least two obstacles"

    if assign_mode == "size_shared":
        obstacle_map = _assign_size_shared(bar_infos, obstacles)
    elif assign_mode == "nearest":
        obstacle_map = _assign_nearest(bar_infos, obstacles)
    else:
        raise ValueError(assign_mode)

    ordered = sorted(
        range(len(bar_infos)),
        key=lambda i: (
            bar_infos[i]["edge_distance"],
            bar_infos[i]["terminal"][0],
            bar_infos[i]["terminal"][1],
            bar_infos[i]["bar_col"],
            bar_infos[i]["seg_col"],
        ),
    )

    if sequential:
        cur = g.copy()
        for idx in ordered:
            info = bar_infos[idx]
            work = cur.copy()
            for r, c in info["seg_comp"]:
                work[r, c] = bg
            new_pts = _route_segment(work, info, obstacle_map[idx], bg)
            for r, c in info["seg_comp"]:
                cur[r, c] = bg
            for r, c in new_pts:
                if int(cur[r, c]) == bg:
                    cur[r, c] = info["seg_col"]
        return cur

    out = g.copy()
    moves = []
    for idx in ordered:
        info = bar_infos[idx]
        work = g.copy()
        for r, c in info["seg_comp"]:
            work[r, c] = bg
        new_pts = _route_segment(work, info, obstacle_map[idx], bg)
        moves.append((info["seg_comp"], info["seg_col"], new_pts))
    for seg_comp, _seg_col, _new_pts in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _seg_comp, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route the ropes with three coupled rules.

    1. Rank the ropes by input segment length.
    2. Rank obstacle slots by obstacle size, with the largest obstacle contributing two slots so it is shared.
    3. Apply the reroutes from the terminal closest to the border to the farthest; earlier outputs become blockers.
    """
    return _solve_with_assignment(grid, sequential=True, assign_mode="size_shared")


def _main_parallel_size_shared_reference(grid: np.ndarray) -> np.ndarray:
    return _solve_with_assignment(grid, sequential=False, assign_mode="size_shared")


def _main_nearest_sequential_reference(grid: np.ndarray) -> np.ndarray:
    return _solve_with_assignment(grid, sequential=True, assign_mode="nearest")


def _analyze_instance(grid: np.ndarray) -> dict[str, int]:
    bar_infos, obstacles, _bg = _discover_scene(np.asarray(grid, dtype=int))
    size_map = _assign_size_shared(bar_infos, obstacles)
    near_map = _assign_nearest(bar_infos, obstacles)
    obs_sorted = _sorted_obstacles(obstacles)
    largest_key = tuple(sorted(obs_sorted[0]["comp"]))
    assigned_keys = [tuple(sorted(size_map[i]["comp"])) for i in range(len(bar_infos))]
    nearest_keys = [tuple(sorted(near_map[i]["comp"])) for i in range(len(bar_infos))]
    shared_count = Counter(assigned_keys)[largest_key]
    size_mismatch = sum(assigned_keys[i] != nearest_keys[i] for i in range(len(bar_infos)))
    combo = main(grid)
    parallel = _main_parallel_size_shared_reference(grid)
    nearest = _main_nearest_sequential_reference(grid)
    return {
        "shared_count": shared_count,
        "size_mismatch": size_mismatch,
        "cascade_diff": int(np.sum(combo != parallel)),
        "nearest_diff": int(np.sum(combo != nearest)),
    }


_BASE_TEMPLATE = np.array(
    [
        [0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 0, 0, 0],
    ],
    dtype=int,
)


def _apply_symmetry(grid: np.ndarray, sym: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    if sym >= 4:
        out = np.fliplr(out)
    return np.rot90(out, k=sym % 4)


def _recolor_template(grid: np.ndarray) -> np.ndarray:
    palette = sorted(int(v) for v in np.unique(grid))
    mapping = {old: new for old, new in zip(palette, random.sample(range(10), len(palette)))}
    return np.vectorize(lambda v: mapping[int(v)])(grid).astype(int)


def generate_input() -> np.ndarray:
    for _ in range(48):
        grid = _recolor_template(_apply_symmetry(_BASE_TEMPLATE, random.randrange(8)))
        metrics = _analyze_instance(grid)
        if metrics["shared_count"] != 2:
            continue
        if metrics["size_mismatch"] <= 0:
            continue
        if metrics["cascade_diff"] <= 0:
            continue
        if metrics["nearest_diff"] <= 0:
            continue
        return grid
    raise RuntimeError("failed to generate a stable D8/2+D11/0+D13/1 sample")
