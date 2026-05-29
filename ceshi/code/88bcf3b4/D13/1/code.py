# Variant of 88bcf3b4
# Dimension: D13 — 三元组基数关系 (meta)
# Change: 每个 triple 是严格的 1:1:1 关系：一个 bar 对应一个线段对应一个障碍物
#   -> 两个不同边缘的 bar 共享同一个障碍物。
#
# Design analysis
# ---
# variant_spec:
#   scene:
#     - 画面中固定有 2 根 bar，分别连在相对边缘（默认 top / bottom）。
#     - 两根 bar 各自挂着一条 irregular input segment。
#     - 两条线段共享 1 个 obstacle；输出时分别占据 obstacle 的左右两侧皮肤。
#   behavior_trace:
#     original:
#       - 原 solver 对每根 bar 独立找最近 obstacle，再独立重画 segment。
#       - 这隐含了“一个 obstacle 只服务一个 bar”的前提。
#     variant:
#       - 先识别全部 bar+segment，再把它们按最近 obstacle 分组。
#       - 同组内的路线规划带 reserved 占位，后处理的 bar 不能踩前一条已分配的输出路线。
#       - 每条线段使用 obstacle 的哪一侧，不由颜色决定，而由 terminal->seg_start 的可见方向决定。
#   implicit_assumption_audit:
#     - 原版会把“除本 bar/segment 颜色外的任意颜色组件”都当 obstacle 候选；
#       在多 bar 变体里这会误把另一根 bar 当 obstacle，所以先收集所有 bars/segments，再用剩余组件作为 obstacle。
#     - 原版 exit 选择不区分 obstacle 左右侧；共享目标时会把两条线挤到同一出口。
#       变体改成：entry、skin、exit 全都限制在各自侧翼上。
#     - 原版 repaint 只在背景格落笔，重叠会静默丢像素；变体在规划阶段直接禁止输出重叠。
#   generator_legality:
#     - 生成器固定为 2 bars + 1 obstacle，先构造 input，再直接调用 main(inp) 作为 oracle 验证。
#     - obstacle 使用细长竖向 4-connected sprite，可选一格小凸起，保证左右两翼明显且可同时容纳两条线。
#     - 两条 input segments 都朝远离 obstacle 的方向长出，避免遮挡共享目标这个核心线索。
#   observability:
#     - 人眼能直接看到：输入里是 2 条来自不同边缘的绳段，但只有 1 个 obstacle；
#       输出里两条线都围绕同一个 obstacle 重定位，一左一右。

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


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


def _obstacle_skin(obstacle_set: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    skin = set()
    for r, c in obstacle_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                skin.add((nr, nc))
    return skin


def _route_segment(
    g: np.ndarray,
    bg: int,
    terminal: tuple[int, int],
    seg_start: tuple[int, int],
    seg_col: int,
    seg_len: int,
    obstacle_set: set[tuple[int, int]],
    fwd: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = g.shape
    off = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
    assert abs(off[0]) + abs(off[1]) == 1
    skin = _obstacle_skin(obstacle_set, h, w)
    cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
    cc = sum(c for _, c in obstacle_set) / len(obstacle_set)

    def cell_ok(p: tuple[int, int]) -> bool:
        return int(g[p]) in (bg, seg_col)

    def lane_score(p: tuple[int, int]) -> float:
        return (p[0] - cr) * off[0] + (p[1] - cc) * off[1]

    def fwd_score(p: tuple[int, int]) -> int:
        return (p[0] - terminal[0]) * fwd[0] + (p[1] - terminal[1]) * fwd[1]

    entry_candidates = []
    for p in skin:
        if p in reserved or not cell_ok(p):
            continue
        anchor = (p[0] - off[0], p[1] - off[1])
        if anchor not in obstacle_set:
            continue
        dist = abs(p[0] - terminal[0]) + abs(p[1] - terminal[1])
        entry_candidates.append((dist, -lane_score(p), -fwd_score(p), p[0], p[1], p))
    assert entry_candidates, "no side-specific entry available"
    entry = min(entry_candidates)[-1]

    walkable = (g == bg) | (g == seg_col)
    for r, c in reserved:
        walkable[r, c] = False
    walkable[seg_start] = True
    walkable[entry] = True
    approach = _greedy_manhattan_path(seg_start, entry, walkable) or _bfs_path(seg_start, entry, walkable)
    assert approach is not None, f"no approach path from {seg_start} to {entry}"

    lane_skin = []
    for p in skin:
        if not cell_ok(p):
            continue
        if p in reserved and p != entry:
            continue
        if lane_score(p) >= -0.01:
            lane_skin.append(p)
    if entry not in lane_skin:
        lane_skin.append(entry)

    exit_cell = max(
        lane_skin,
        key=lambda p: (fwd_score(p), lane_score(p), abs(p[0] - terminal[0]) + abs(p[1] - terminal[1]), -p[0], -p[1]),
    )
    skin_walkable = np.zeros((h, w), dtype=bool)
    for r, c in lane_skin:
        skin_walkable[r, c] = True
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
        options = [(dr_ext, dc_ext), fwd, (fwd[0] + off[0], fwd[1] + off[1]), (fwd[0] - off[0], fwd[1] - off[1])]
        moved = False
        for dr, dc in options:
            if dr == 0 and dc == 0:
                continue
            nxt = (cur[0] + dr, cur[1] + dc)
            nr, nc = nxt
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in used or nxt in reserved or nxt in obstacle_set:
                continue
            if int(g[nr, nc]) not in (bg, seg_col):
                continue
            route.append(nxt)
            used.add(nxt)
            cur = nxt
            moved = True
            break
        if not moved:
            break

    assert len(route) >= seg_len, "route could not preserve segment length"
    new_pts = route[:seg_len]
    assert not any(p in obstacle_set for p in new_pts)
    assert not any(p in reserved for p in new_pts)
    return new_pts


def _collect_bar_infos(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
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
            if vertical:
                if border_ep[0] not in (0, h - 1):
                    continue
            else:
                if border_ep[1] not in (0, w - 1):
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
                    seg_len = len(path)
                    key = (seg_len, len(seg_comp), cand_col, cand_start[0], cand_start[1])
                    if best_seg is None or key > best_seg[0]:
                        best_seg = (key, cand_col, cand_start, seg_comp)
                    break
            assert best_seg is not None, f"no valid segment for bar terminal {terminal}"
            (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg
            fwd, _lat = _bar_frame(comp, h, w)
            infos.append(
                {
                    "bar_col": col,
                    "bar_comp": comp,
                    "terminal": terminal,
                    "seg_col": seg_col,
                    "seg_start": seg_start,
                    "seg_comp": seg_comp,
                    "seg_len": seg_len,
                    "fwd": fwd,
                }
            )
    return infos


def _obstacle_components(g: np.ndarray, ignored_colors: set[int]) -> list[dict]:
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg} - set(ignored_colors))
    obstacles = []
    for col in colors:
        for comp in _components(g, col, conn8=True):
            obstacles.append(
                {
                    "color": col,
                    "comp": comp,
                    "set": set(comp),
                    "min_r": min(r for r, _ in comp),
                    "min_c": min(c for _, c in comp),
                }
            )
    return obstacles


def _assign_obstacles(bar_infos: list[dict], obstacles: list[dict]) -> dict[int, list[dict]]:
    assert obstacles, "must have at least one obstacle"
    groups: dict[int, list[dict]] = {i: [] for i in range(len(obstacles))}
    for info in bar_infos:
        tr, tc = info["terminal"]
        best = None
        best_idx = None
        for idx, obs in enumerate(obstacles):
            dist = min(abs(r - tr) + abs(c - tc) for r, c in obs["comp"])
            key = (dist, obs["min_r"], obs["min_c"])
            if best is None or key < best:
                best = key
                best_idx = idx
        assert best_idx is not None
        groups[best_idx].append(info)
    return {idx: members for idx, members in groups.items() if members}


def _plan_routes(g: np.ndarray, bar_infos: list[dict], obstacles: list[dict]) -> list[tuple[list[tuple[int, int]], int, list[tuple[int, int]]]]:
    bg = get_bg(g)
    groups = _assign_obstacles(bar_infos, obstacles)
    moves = []
    for obs_idx in sorted(groups):
        obstacle_set = obstacles[obs_idx]["set"]
        reserved: set[tuple[int, int]] = set()
        members = sorted(
            groups[obs_idx],
            key=lambda info: (
                info["seg_start"][0] - info["terminal"][0],
                info["seg_start"][1] - info["terminal"][1],
                info["terminal"][0],
                info["terminal"][1],
            ),
        )
        for info in members:
            new_pts = _route_segment(
                g,
                bg,
                info["terminal"],
                info["seg_start"],
                info["seg_col"],
                info["seg_len"],
                obstacle_set,
                info["fwd"],
                reserved,
            )
            reserved.update(new_pts)
            moves.append((info["seg_comp"], info["seg_col"], new_pts))
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    bar_infos = _collect_bar_infos(g, bg)
    ignored_colors = {info["bar_col"] for info in bar_infos} | {info["seg_col"] for info in bar_infos}
    obstacles = _obstacle_components(g, ignored_colors)
    moves = _plan_routes(g, bar_infos, obstacles)

    for seg_comp, _seg_col, _new_pts in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    painted = set()
    for _seg_comp, seg_col, new_pts in moves:
        for p in new_pts:
            assert p not in painted, "overlapping routes detected for shared obstacle"
            painted.add(p)
            out[p] = seg_col
    return out


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 15, 23
BAR_LEN_MIN, BAR_LEN_MAX = 5, 7
SEG_LEN_MIN, SEG_LEN_MAX = 5, 7
OBS_HEIGHT_MIN, OBS_HEIGHT_MAX = 3, 4
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 1000
INPUT_RETRIES = 160

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
}


def _transform(
    locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int
) -> list[tuple[int, int]]:
    tr = bar_len - 1 if edge == "top" else n - bar_len
    tc = pos
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, side)]
    diag_step = (-1, side)
    orth_step = (-1, 0)
    last_was_diag = None
    turns_left = turns
    while remaining > 0:
        if turns_left <= 0:
            if last_was_diag is None:
                step = random.choice([diag_step, orth_step])
            else:
                step = diag_step if last_was_diag else orth_step
            run = remaining
        else:
            if last_was_diag is None:
                is_diag = random.random() < DIAG_START_PROB
            elif last_was_diag:
                is_diag = random.random() < DIAG_CONTINUE_PROB
            else:
                is_diag = random.random() < DIAG_START_PROB
            step = diag_step if is_diag else orth_step
            if last_was_diag is not None and is_diag != last_was_diag:
                turns_left -= 1
            max_run = remaining - turns_left
            run = random.randint(1, max(1, min(max_run, remaining)))
        last_was_diag = step == diag_step
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
    return coords


def _make_obstacle(n: int, top_row: int, height: int, center_c: int) -> list[tuple[int, int]]:
    return [(top_row + k, center_c) for k in range(height)]


def _valid_cells(coords: list[tuple[int, int]], n: int) -> bool:
    return all(0 <= r < n and 0 <= c < n for r, c in coords)


def _make_input_segment(
    edge: str,
    n: int,
    bar_len: int,
    pos: int,
    local_side: int,
    seg_len: int,
    forbidden: set[tuple[int, int]],
    obstacle_set: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    min_turns = min(INPUT_TURN_MIN, max_turns)
    for _ in range(INPUT_RETRIES):
        turns = random.randint(min_turns, max_turns)
        glob = _transform(_build_input_path(seg_len, turns, local_side), edge, n, bar_len, pos)
        if not _valid_cells(glob, n):
            continue
        if len(set(glob)) != len(glob):
            continue
        if _is_straight_line(glob):
            continue
        if any(r in (0, n - 1) or c in (0, n - 1) for r, c in glob):
            continue
        if any(p in forbidden for p in glob):
            continue
        too_close = False
        for p in glob:
            if p in obstacle_set:
                too_close = True
                break
            for q in neighbors4(*p):
                if q in obstacle_set:
                    too_close = True
                    break
            if too_close:
                break
        if too_close:
            continue
        return glob
    return None


def _shared_obstacle_signature(inp: np.ndarray, out: np.ndarray) -> tuple[str, str] | None:
    bg = get_bg(inp)
    bar_infos = _collect_bar_infos(inp, bg)
    ignored_colors = {info["bar_col"] for info in bar_infos} | {info["seg_col"] for info in bar_infos}
    obstacles = _obstacle_components(inp, ignored_colors)
    groups = _assign_obstacles(bar_infos, obstacles)
    if len(obstacles) != 1 or list(groups.keys()) != [0] or len(groups[0]) != 2:
        return None

    obs = obstacles[0]["comp"]
    obsset = set(obs)
    c0 = min(c for _, c in obs)
    c1 = max(c for _, c in obs)
    sides = []
    for info in sorted(groups[0], key=lambda x: x["terminal"][0]):
        comps = _components(out, info["seg_col"], conn8=True)
        if not comps:
            return None
        comp = min(comps, key=lambda comp_: min(abs(r - info["terminal"][0]) + abs(c - info["terminal"][1]) for r, c in comp_))
        if set(comp) == set(info["seg_comp"]):
            return None
        skin = [p for p in comp if any(q in obsset for q in neighbors4(*p))]
        if not skin:
            return None
        entry = min(skin, key=lambda p: abs(p[0] - info["terminal"][0]) + abs(p[1] - info["terminal"][1]))
        if entry[1] < c0:
            sides.append("L")
        elif entry[1] > c1:
            sides.append("R")
        else:
            return None
    return tuple(sides) if set(sides) == {"L", "R"} else None


def _gen_attempt() -> np.ndarray | None:
    bg, obstacle_color, top_bar_color, top_seg_color, bot_bar_color, bot_seg_color = random.sample(range(10), 6)
    top_bar_len = random.randint(BAR_LEN_MIN, BAR_LEN_MAX)
    bot_bar_len = random.randint(BAR_LEN_MIN, BAR_LEN_MAX)
    obs_h = random.randint(OBS_HEIGHT_MIN, OBS_HEIGHT_MAX)
    top_gap = random.randint(OBS_MD_MIN, OBS_MD_MAX - 1)
    bot_gap = random.randint(OBS_MD_MIN, OBS_MD_MAX - 1)
    n = top_bar_len + top_gap + obs_h + bot_gap + bot_bar_len - 1
    if not (GRID_SIZE_MIN <= n <= GRID_SIZE_MAX):
        return None

    top_row = top_bar_len - 1 + top_gap
    top_seg_len = random.randint(SEG_LEN_MIN, min(SEG_LEN_MAX, top_bar_len))
    bot_seg_len = random.randint(SEG_LEN_MIN, min(SEG_LEN_MAX, bot_bar_len))
    side_margin = max(top_seg_len, bot_seg_len) + 1
    if side_margin > n - side_margin - 1:
        return None
    center_c = random.randint(side_margin, n - side_margin - 1)
    obstacle = _make_obstacle(n, top_row, obs_h, center_c)
    obstacle_set = set(obstacle)

    top_global_side = random.choice([-1, 1])
    bot_global_side = -top_global_side
    top_pos = center_c
    bot_pos = center_c

    top_bar = [(r, center_c) for r in range(top_bar_len)]
    bot_bar = [(r, center_c) for r in range(n - bot_bar_len, n)]
    if obstacle_set & set(top_bar):
        return None
    if obstacle_set & set(bot_bar):
        return None

    top_seg_start = (top_bar_len - 1, center_c + top_global_side)
    bot_seg_start = (n - bot_bar_len, center_c + bot_global_side)
    if not _valid_cells([top_seg_start, bot_seg_start], n):
        return None
    if top_seg_start in obstacle_set or bot_seg_start in obstacle_set:
        return None

    base = np.full((n, n), bg, dtype=int)
    paint_coords(base, obstacle, obstacle_color)
    paint_coords(base, top_bar, top_bar_color)
    paint_coords(base, bot_bar, bot_bar_color)

    top_input = _make_input_segment("top", n, top_bar_len, top_pos, top_global_side, top_seg_len, set(top_bar) | set(bot_bar) | obstacle_set, obstacle_set)
    if top_input is None:
        return None
    bot_input = _make_input_segment(
        "bottom",
        n,
        bot_bar_len,
        bot_pos,
        -bot_global_side,
        bot_seg_len,
        set(top_bar) | set(bot_bar) | obstacle_set | set(top_input),
        obstacle_set,
    )
    if bot_input is None:
        return None

    inp = base.copy()
    paint_coords(inp, top_input, top_seg_color)
    paint_coords(inp, bot_input, bot_seg_color)

    try:
        out = main(inp)
    except Exception:
        return None
    if _shared_obstacle_signature(inp, out) is None:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
