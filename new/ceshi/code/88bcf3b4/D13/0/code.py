# Variant of 88bcf3b4
# Dimension: D13 — 三元组基数关系 (meta)
# Change: 每个 triple 是严格的 1:1:1 关系：一个 bar 对应一个线段对应一个障碍物
#   -> 一个 bar 对应 2-3 个不同颜色的线段；它们都挂在同一个 terminal 上，并分别绕过同一个 obstacle。
#
# Design analysis
# - a) 具体化与行为追踪
#   原题里，一个 bar 的 terminal 只会在 4 邻域里看到一个有效线段端点，solver 于是只挑 “best_seg”。
#   本变体把 terminal 附近的 3 个可用邻格当成挂点：
#     左侧 (0,-1)、前方 (1,0)、右侧 (0,1)。
#   2 条线段时固定用左右两侧；3 条线段时再加前方这一条。
#   它们颜色各不相同，但共享同一个 obstacle。输出时每条线段仍按原任务的“靠近 obstacle -> 沿 skin 走 ->
#   顺最后方向继续延展”的机制独立计算，所以视觉上会看到同一根 bar 甩出多根不同颜色的 rope。
#   小例子：原逻辑在 terminal 看到 left/red 和 right/blue 两个邻格时，只会选较长的那一条，另一条被忽略；
#   现在逻辑会把 red、blue 都登记到 attached_segments，再对它们共享同一个 obstacle 分别 reroute。
# - b) 复杂度校准
#   只保留两个清晰档位：2 条线段 = 左+右；3 条线段 = 左+前+右。避免 “任意选 2/3 个挂点” 带来的支撑不足。
#   triple 数量进一步收紧到 1 个，并给每个对象全局唯一颜色，避免多条线段与多组 triple 交错后匹配歧义；
#   D13 的观察重点因此集中在 “一个 bar -> 多条线段” 本身。
# - c) 隐式假设审查
#   原 main() 中和 D13 直接相关的旧假设有：
#   1. `best_seg`：默认 terminal 只需要选一个线段；变体中必须改成收集全部 attached segments。
#   2. `obstacle_candidates = colors if c not in (bar_col, seg_col)`：只排除一个 seg 颜色；变体中要排除同一 bar 下全部 seg 颜色。
#   3. `inner = (-perp_dir)` 的 apex 断言：它默认 seg_start 永远在 terminal 左/右侧。第三条前向线段会破坏这一点，
#      所以保留单调包裹断言，但去掉这个只对侧向挂点成立的附加断言。
#   4. `if out[r,c] == bg: out[r,c] = seg_col`：原题里一 bar 一线段，这个“只在背景上画”的写法无害；
#      现在若两条 reroute 结果重叠，会静默丢像素。因此 solver 显式断言同一 bar 的新路径两两不重叠，
#      generator 也会拒绝产生重叠样本。
# - d) 放置合法性
#   generator 先在局部坐标中构造单个多线段 triple，再把它放到真实网格边缘。保留 bar/obstacle/input/output 的并集 bbox，
#   保证输入与输出两阶段都不越界、不自相撞。每个候选样本最后都用 `main(input)` 回验。
# - e) 视觉可观察性
#   2 条时左右对称，3 条时左右再加前方一条，2-4 个样例足够让人看出 “同一 bar 可以拖动多条 rope”；
#   规则不依赖颜色值顺序，颜色只是区分 rope 身份。
# - f) 可视化整洁度
#   obstacle 限制在 terminal 前方的小范围中央区域；输入线段固定朝远离 obstacle 的扇区生长。
#   这样画面里不会出现大面积覆盖或缠在一起的多层对象。
# - g) 颜色变化可推断性
#   颜色不做映射。每条线段在输入输出中保持原色，只是位置改变。解题者只需按颜色追踪各条 rope 即可。

import random
import re
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


def _has_2x2(coords: list[tuple[int, int]]) -> bool:
    sset = set(coords)
    for r, c in sset:
        if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= sset:
            return True
    return False


def _bar_forward(bar_comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in bar_comp]
    cs = [c for _, c in bar_comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


def _bar_side(fwd: tuple[int, int]) -> tuple[int, int]:
    if fwd == (1, 0):
        return (0, 1)
    if fwd == (-1, 0):
        return (0, -1)
    if fwd == (0, 1):
        return (-1, 0)
    return (1, 0)


def _to_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    dr = p[0] - terminal[0]
    dc = p[1] - terminal[1]
    return dr * fwd[0] + dc * fwd[1], dr * side[0] + dc * side[1]


def _from_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side: tuple[int, int]
) -> tuple[int, int]:
    return (
        terminal[0] + p[0] * fwd[0] + p[1] * side[0],
        terminal[1] + p[0] * fwd[1] + p[1] * side[1],
    )


def _segment_is_valid(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> bool:
    if len(set(seg_pts)) != len(seg_pts):
        return False
    if _has_2x2(seg_pts):
        return False
    deg = _segment_degrees(seg_pts)
    vals = list(deg.values())
    if any(v > 2 for v in vals):
        return False
    if vals.count(1) != 2:
        return False
    if deg.get(start, 0) != 1:
        return False
    path = _segment_main_path(seg_pts, start)
    return len(path) == len(seg_pts)


def _collect_attached_segments(
    g: np.ndarray, bg: int, bar_col: int, terminal: tuple[int, int]
) -> list[dict[str, object]]:
    h, w = g.shape
    found = []
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    term_neigh = {(nr, nc) for nr, nc in neighbors4(terminal[0], terminal[1]) if 0 <= nr < h and 0 <= nc < w}
    for nr, nc in term_neigh:
        cand_col = int(g[nr, nc])
        if cand_col in (bg, bar_col):
            continue
        for comp in _components(g, cand_col, conn8=True):
            if (nr, nc) not in comp:
                continue
            key = (cand_col, tuple(sorted(comp)))
            if key in seen:
                break
            seen.add(key)
            start = (nr, nc)
            if not _segment_is_valid(comp, start):
                break
            adj_count = sum(1 for p in comp if p in term_neigh)
            if adj_count != 1:
                break
            path = _segment_main_path(comp, start)
            found.append(
                {
                    "color": cand_col,
                    "start": start,
                    "comp": comp,
                    "path": path,
                }
            )
            break
    found.sort(key=lambda seg: (seg["start"][0] - terminal[0], seg["start"][1] - terminal[1], seg["color"]))  # type: ignore[index]
    return found


def _pick_obstacle_for_bar(
    g: np.ndarray,
    colors: list[int],
    bar_col: int,
    segments: list[dict[str, object]],
    terminal: tuple[int, int],
) -> set[tuple[int, int]]:
    excluded = {bar_col} | {int(seg["color"]) for seg in segments}
    best = None
    obstacle_set = None
    for col in colors:
        if col in excluded:
            continue
        for comp in _components(g, col, conn8=True):
            d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
            mr = min(r for r, _ in comp)
            mc = min(c for _, c in comp)
            key = (d, mr, mc)
            if best is None or key < best:
                best = key
                obstacle_set = set(comp)
    assert obstacle_set is not None, f"no obstacle found for bar at {terminal}"
    return obstacle_set


def _validate_wrap_shape(new_pts: list[tuple[int, int]], bar_comp: list[tuple[int, int]]) -> None:
    fwd = _bar_forward(bar_comp)
    step_classes = []
    for j in range(len(new_pts) - 1):
        dr = new_pts[j + 1][0] - new_pts[j][0]
        dc = new_pts[j + 1][1] - new_pts[j][1]
        fwd_comp = dr * fwd[0] + dc * fwd[1]
        perp_comp = dr * fwd[1] + dc * (-fwd[0])
        if abs(dr) + abs(dc) == 1:
            step_classes.append("F" if fwd_comp > 0 else "X")
        elif fwd_comp > 0 and perp_comp > 0:
            step_classes.append("FR")
        elif fwd_comp > 0 and perp_comp < 0:
            step_classes.append("FL")
        else:
            step_classes.append("X")
    s = "".join(step_classes)
    assert re.fullmatch(r"(FL)*(F)*(FR)*", s) or re.fullmatch(r"(FR)*(F)*(FL)*", s), (
        f"output segment not monotone wrap: {' '.join(step_classes)}"
    )


def _segment_branch_local(local_path: list[tuple[int, int]]) -> int:
    for _r, c in local_path[1:]:
        if c < 0:
            return -1
        if c > 0:
            return 1
    return 1


def _plan_local_route(
    start_local: tuple[int, int],
    length: int,
    obstacle_locals: set[tuple[int, int]],
    lane: int,
) -> list[tuple[int, int]]:
    rs = [r for r, _ in obstacle_locals]
    r0 = min(rs)
    r1 = max(rs)
    sr, sc = start_local
    lane_sign = -1 if lane < sc else 1 if lane > sc else (-1 if lane < 0 else 1)
    diag_in = abs(lane - sc)
    straight_in = r0 - sr - diag_in
    assert straight_in >= 0, f"obstacle too close for start {start_local} and lane {lane}"

    route = [start_local]
    cur = start_local
    for _ in range(diag_in):
        cur = (cur[0] + 1, cur[1] + lane_sign)
        route.append(cur)
    for _ in range(straight_in):
        cur = (cur[0] + 1, cur[1])
        route.append(cur)
    for _ in range(max(0, r1 - r0)):
        cur = (cur[0] + 1, cur[1])
        route.append(cur)

    assert len(route) <= length, f"base route too long for start {start_local}"
    exit_dc = -lane_sign
    while len(route) < length:
        cur = (cur[0] + 1, cur[1] + exit_dc)
        route.append(cur)
    return route


def _reroute_segment(
    g: np.ndarray,
    bg: int,
    terminal: tuple[int, int],
    bar_comp: list[tuple[int, int]],
    seg: dict[str, object],
    obstacle_set: set[tuple[int, int]],
    lane: int,
) -> list[tuple[int, int]]:
    seg_col = int(seg["color"])
    seg_start = seg["start"]
    seg_path = seg["path"]
    fwd = _bar_forward(bar_comp)
    side = _bar_side(fwd)
    local_start = _to_local(seg_start, terminal, fwd, side)
    local_path = [_to_local(p, terminal, fwd, side) for p in seg_path]
    local_obstacle = {_to_local(p, terminal, fwd, side) for p in obstacle_set}
    local_route = _plan_local_route(local_start, len(local_path), local_obstacle, lane)
    new_pts = [_from_local(p, terminal, fwd, side) for p in local_route]
    h, w = g.shape
    bar_set = set(bar_comp)
    assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), f"reroute exits grid for segment at {seg_start}"
    assert not any(p in bar_set for p in new_pts), f"reroute crosses bar for segment at {seg_start}"
    assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
    _validate_wrap_shape(new_pts, bar_comp)
    return new_pts


def _bar_candidates(g: np.ndarray, bg: int, colors: list[int]) -> list[tuple[int, list[tuple[int, int]], tuple[int, int]]]:
    h, w = g.shape
    bars = []
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
            terminal_ep = e1 if touch0 else e0
            tr, tc = terminal_ep
            if not any(
                0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                for nr, nc in neighbors4(tr, tc)
            ):
                continue
            bars.append((col, comp, terminal_ep))
    bars.sort(key=lambda x: (-len(x[1]), x[0], x[2]))
    return bars


def _bar_segment_counts(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    counts = []
    for bar_col, _bar_comp, terminal in _bar_candidates(g, bg, colors):
        counts.append(len(_collect_attached_segments(g, bg, bar_col, terminal)))
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    moves: list[tuple[list[tuple[int, int]], int, list[tuple[int, int]]]] = []
    for bar_col, bar_comp, terminal in _bar_candidates(g, bg, colors):
        segments = _collect_attached_segments(g, bg, bar_col, terminal)
        assert segments, f"bar terminal {terminal} must have at least one valid segment"
        obstacle_set = _pick_obstacle_for_bar(g, colors, bar_col, segments, terminal)
        fwd = _bar_forward(bar_comp)
        side = _bar_side(fwd)
        local_obstacle = {_to_local(p, terminal, fwd, side) for p in obstacle_set}
        c0 = min(c for _r, c in local_obstacle)
        c1 = max(c for _r, c in local_obstacle)

        front_branch = 0
        for seg in segments:
            local_start = _to_local(seg["start"], terminal, fwd, side)
            if local_start == (1, 0):
                local_path = [_to_local(p, terminal, fwd, side) for p in seg["path"]]
                front_branch = _segment_branch_local(local_path)
                break

        occupied_new: set[tuple[int, int]] = set()
        for seg in segments:
            local_start = _to_local(seg["start"], terminal, fwd, side)
            if local_start == (0, -1):
                lane = c0 - 2 if front_branch < 0 else c0 - 1
            elif local_start == (0, 1):
                lane = c1 + 2 if front_branch > 0 else c1 + 1
            else:
                assert local_start == (1, 0), f"unexpected attachment slot {local_start}"
                lane = c0 - 1 if front_branch < 0 else c1 + 1
            new_pts = _reroute_segment(g, bg, terminal, bar_comp, seg, obstacle_set, lane)
            overlap = occupied_new & set(new_pts)
            assert not overlap, f"segments from one bar overlap after reroute: {sorted(overlap)}"
            occupied_new |= set(new_pts)
            moves.append((seg["comp"], int(seg["color"]), new_pts))

    for seg_comp, _seg_col, _new_pts in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _seg_comp, seg_col, new_pts in moves:
        for r, c in new_pts:
            assert int(out[r, c]) == bg, f"unexpected paint collision at {(r, c)}"
            out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 10, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
SEG_LEN_MIN, SEG_LEN_MAX = 5, 8
OBS_SIZE_MIN, OBS_SIZE_MAX = 2, 6
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 4
PLACEMENT_PAD = 2
MAX_RETRIES = 400
TRIPLE_LOCAL_RETRIES = 180
LOCAL_CANVAS = 27

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}

ATTACH_LAYOUTS = {
    2: [("left", (0, -1)), ("right", (0, 1))],
    3: [("left", (0, -1)), ("front", (1, 0)), ("right", (0, 1))],
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


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, side)]
    diag_step = (-1, side)
    orth_step = (-1, 0)
    last_was_diag = None
    turns_left = turns
    while remaining > 0:
        if turns_left <= 0:
            step = diag_step if last_was_diag else orth_step
            if last_was_diag is None:
                step = random.choice([diag_step, orth_step])
            run = remaining
        else:
            if last_was_diag is None:
                is_diag = random.random() < 0.5
            elif last_was_diag:
                is_diag = random.random() < 0.35
            else:
                is_diag = random.random() < 0.5
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


def _build_front_input_path(seg_len: int, turns: int, branch_side: int) -> list[tuple[int, int]]:
    coords = [(1, 0), (2, 0), (2, branch_side), (1, 2 * branch_side)]
    remaining = seg_len - 4
    diag_step = (-1, branch_side)
    orth_step = (-1, 0)
    last_was_diag = None
    turns_left = max(0, turns - 1)
    while remaining > 0:
        if turns_left <= 0:
            step = diag_step if last_was_diag else orth_step
            if last_was_diag is None:
                step = random.choice([diag_step, orth_step])
            run = remaining
        else:
            if last_was_diag is None:
                is_diag = random.random() < 0.6
            elif last_was_diag:
                is_diag = random.random() < 0.45
            else:
                is_diag = random.random() < 0.6
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


def _is_4adj(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    bset = set(coords_b)
    for r, c in coords_a:
        if (r, c) in bset:
            return True
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in bset:
                return True
    return False


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _grow_obstacle(
    start: tuple[int, int],
    max_size: int,
    forbidden: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    obstacle = {start}
    for _ in range(max_size - 1):
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden and (nr, nc) in allowed
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _make_local_obstacle(seg_count: int, bar: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    bar_set = set(bar)
    obs_dist = random.randint(max(4 if seg_count == 3 else 3, OBS_MD_MIN), OBS_MD_MAX)
    start = (obs_dist, random.choice([-1, 0, 1]))
    side_span = 2 if seg_count == 2 else 1
    allowed = {(r, c) for r in range(2, obs_dist + 4) for c in range(-side_span, side_span + 1)}
    forbidden = bar_set | {(0, -1), (0, 1), (1, 0)}
    max_size = OBS_SIZE_MAX if seg_count == 2 else min(4, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(start, random.randint(OBS_SIZE_MIN, max_size), forbidden, allowed)
    rs = [r for r, _ in obstacle]
    cs = [c for _, c in obstacle]
    if max(abs(c) for c in cs) > side_span:
        return None
    max_h = 4 if seg_count == 2 else 2
    if max(rs) - min(rs) > max_h or max(cs) - min(cs) > 2:
        return None
    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None
    return obstacle


def _paint_local_triple(
    bar_len: int,
    bar: list[tuple[int, int]],
    obstacle: list[tuple[int, int]],
    input_segments: list[dict[str, object]],
) -> tuple[np.ndarray, tuple[int, int], list[int]]:
    n = LOCAL_CANVAS
    pos = n // 2
    bg = 0
    bar_color = 1
    obstacle_color = 2
    seg_colors = list(range(3, 3 + len(input_segments)))
    inp = np.full((n, n), bg, dtype=int)
    paint_coords(inp, _transform(bar, "top", n, bar_len, pos), bar_color)
    paint_coords(inp, _transform(obstacle, "top", n, bar_len, pos), obstacle_color)
    for seg, seg_col in zip(input_segments, seg_colors):
        paint_coords(inp, _transform(seg["coords"], "top", n, bar_len, pos), seg_col)
    return inp, _terminal_for("top", n, bar_len, pos), seg_colors


def _extract_local_output_segments(
    out: np.ndarray, terminal: tuple[int, int], seg_colors: list[int], input_segments: list[dict[str, object]]
) -> list[dict[str, object]] | None:
    used: set[tuple[int, int]] = set()
    output_segments = []
    for seg_col, inp_seg in zip(seg_colors, input_segments):
        comps = _components(out, seg_col, conn8=True)
        if len(comps) != 1:
            return None
        comp = comps[0]
        if len(comp) != len(inp_seg["coords"]):
            return None
        local = [(r - terminal[0], c - terminal[1]) for r, c in comp]
        if _has_2x2(local):
            return None
        overlap = used & set(local)
        if overlap:
            return None
        used |= set(local)
        output_segments.append({"coords": local, "color": seg_col})
    return output_segments


def _make_triple(seg_count: int) -> dict | None:
    for _ in range(TRIPLE_LOCAL_RETRIES):
        bar_len = random.randint(BAR_LEN_MIN, BAR_LEN_MAX)
        bar = [(-k, 0) for k in range(bar_len)]
        obstacle = _make_local_obstacle(seg_count, bar)
        if obstacle is None:
            continue

        r0 = min(r for r, _ in obstacle)
        r1 = max(r for r, _ in obstacle)
        along = r1 - r0
        front_branch = random.choice([-1, 1]) if seg_count == 3 else 0
        min_len = {
            "left": 1 + r0 + along,
            "right": 1 + r0 + along,
            "front": r0 + along,
        }

        input_segments = []
        occupied = set(bar) | set(obstacle)
        ok = True
        for name, start in ATTACH_LAYOUTS[seg_count]:
            need = max(SEG_LEN_MIN, min_len[name])
            if need > SEG_LEN_MAX:
                ok = False
                break
            seg_len = random.randint(need, SEG_LEN_MAX)
            turns = random.randint(INPUT_TURN_MIN, min(INPUT_TURN_MAX, seg_len - 2))
            if name == "front":
                coords = _build_front_input_path(seg_len, turns, front_branch)
            else:
                coords = _build_input_path(seg_len, turns, -1 if name == "left" else 1)
            if start != coords[0]:
                ok = False
                break
            if not _segment_is_valid(coords, start):
                ok = False
                break
            if any(p in occupied for p in coords):
                ok = False
                break
            if _is_4adj(coords, obstacle):
                ok = False
                break
            input_segments.append({"name": name, "start": start, "coords": coords})
            occupied |= set(coords)
        if not ok:
            continue

        inp, terminal, seg_colors = _paint_local_triple(bar_len, bar, obstacle, input_segments)
        try:
            out = main(inp)
        except Exception:
            continue
        if _bar_segment_counts(inp) != [seg_count]:
            continue
        output_segments = _extract_local_output_segments(out, terminal, seg_colors, input_segments)
        if output_segments is None:
            continue

        out_obstacle = _components(out, 2, conn8=True)
        expected_obstacle = _transform(obstacle, "top", LOCAL_CANVAS, bar_len, LOCAL_CANVAS // 2)
        if len(out_obstacle) != 1 or set(out_obstacle[0]) != set(expected_obstacle):
            continue

        union = list(bar) + list(obstacle)
        for seg in input_segments:
            union.extend(seg["coords"])
        for seg in output_segments:
            union.extend(seg["coords"])
        return {
            "bar_len": bar_len,
            "bar": bar,
            "obstacle": obstacle,
            "input_segments": input_segments,
            "output_segments": output_segments,
            "union": union,
        }
    return None


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    bar_len = triple["bar_len"]
    union = triple["union"]
    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_segments": [
                {"coords": _transform(seg["coords"], edge, n, bar_len, pos)}
                for seg in triple["input_segments"]
            ],
            "output_segments": [
                {"coords": _transform(seg["coords"], edge, n, bar_len, pos)}
                for seg in triple["output_segments"]
            ],
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt(forced_seg_count: int | None = None) -> np.ndarray | None:
    seg_count = forced_seg_count if forced_seg_count is not None else random.choice([2, 3])
    triple = _make_triple(seg_count)
    if triple is None:
        return None

    bg = random.randrange(10)
    palette = random.sample([c for c in range(10) if c != bg], seg_count + 2)
    bar_color = palette[0]
    obstacle_color = palette[1]
    seg_colors = palette[2:]

    full_in = np.full((LOCAL_CANVAS, LOCAL_CANVAS), bg, dtype=int)
    full_out = np.full((LOCAL_CANVAS, LOCAL_CANVAS), bg, dtype=int)
    pos = LOCAL_CANVAS // 2
    bar_len = triple["bar_len"]
    bar_abs = _transform(triple["bar"], "top", LOCAL_CANVAS, bar_len, pos)
    obstacle_abs = _transform(triple["obstacle"], "top", LOCAL_CANVAS, bar_len, pos)
    paint_coords(full_in, bar_abs, bar_color)
    paint_coords(full_in, obstacle_abs, obstacle_color)
    paint_coords(full_out, bar_abs, bar_color)
    paint_coords(full_out, obstacle_abs, obstacle_color)
    for seg, seg_col in zip(triple["input_segments"], seg_colors):
        paint_coords(full_in, _transform(seg["coords"], "top", LOCAL_CANVAS, bar_len, pos), seg_col)
    for seg, seg_col in zip(triple["output_segments"], seg_colors):
        paint_coords(full_out, _transform(seg["coords"], "top", LOCAL_CANVAS, bar_len, pos), seg_col)

    union_abs = _transform(triple["union"], "top", LOCAL_CANVAS, bar_len, pos)
    rs = [r for r, _ in union_abs]
    cs = [c for _, c in union_abs]
    r1 = max(rs)
    c0 = min(cs)
    c1 = max(cs)
    n_min = max(GRID_SIZE_MIN, r1 + 1, c1 - c0 + 1)
    n_max = min(GRID_SIZE_MAX, LOCAL_CANVAS)
    if n_min > n_max:
        return None
    n = random.randint(n_min, n_max)
    left_lo = max(0, c1 - n + 1)
    left_hi = min(c0, LOCAL_CANVAS - n)
    if left_lo > left_hi:
        return None
    left = random.randint(left_lo, left_hi)
    inp = full_in[:n, left : left + n].copy()
    out = full_out[:n, left : left + n].copy()

    k = random.randrange(4)
    inp = np.rot90(inp, k).copy()
    out = np.rot90(out, k).copy()

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    counts = _bar_segment_counts(inp)
    if counts != [seg_count]:
        return None
    return inp


def generate_input() -> np.ndarray:
    target_seg_count = 3 if random.random() < 0.2 else 2
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(target_seg_count)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
