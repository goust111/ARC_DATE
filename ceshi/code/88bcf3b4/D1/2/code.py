# Variant of 88bcf3b4
# Dimension: D1 — 路径绕行模式 (path)
# Change: 线段从 bar 末端出发，沿曼哈顿最短路接近障碍物，沿障碍物边界行走，最后以 45 度对角方向延伸退出；三段单调相
#         （approach-alongside-exit） -> 最短路绕行：线段从 bar 末端出发，沿障碍物外侧的 BFS 最短路径到达障碍物对侧
#         的指定出口点（离 terminal 曼哈顿距离最远的 skin cell），允许任意转向。
#
# Design analysis
# ----------------
# spec:
#   core_change:
#     old_rule: 输出段被硬编码为三相单调包裹，solver 先找靠近障碍物的入口，再沿 skin 走，最后按最后一步方向延长补足长度。
#     new_rule: 输出段直接等于一条 4 邻域 BFS 最短路；起点仍是 bar terminal 旁的 attached segment endpoint，终点是
#               最近障碍物组件外侧 skin 中，离 terminal 曼哈顿距离最远的那个可走格。
#   concrete_behavior:
#     input:
#       - bar 仍垂直于边界，terminal 在内部。
#       - segment 仍是 terminal 一侧接上的 1-cell-thick irregular stroke，作为“待重定位绳子”。
#       - obstacle 仍是不规则小精灵，保持不动。
#     output:
#       - segment 脱离原输入形状，改画成从 seg_start 到 farthest-skin exit 的最短路。
#       - 路径只允许 4 邻域步，因此视觉上会出现正交折线/回折，而不是原题的斜向 V + 直行 + 斜向离开。
#       - 像素数保持不变，因此生成器必须构造“最短路长度 == 输入段长度”的实例。
#   traced_example:
#     - 原逻辑在 terminal 附近先贪心/最短地接近 obstacle，再把整个 route 强制拆成 approach / along-side / diagonal-exit。
#     - 新逻辑不再分相。假设 seg_start 在 terminal 右侧、obstacle 横跨 bar 前方，solver 先枚举该 obstacle 的 skin，
#       选出最远的 skin 格作为 exit，再在 obstacle 外部 free space 上做 BFS，得到一条可能“右 -> 下 -> 左 -> 下 -> 右”
#       的最短路。旧代码在这里会失败，因为它只接受 FL*F*FR* / FR*F*FL* 这种单调模式，并且要求边界段长度至少 2 才能外延。
#   complexity_calibration:
#     - 为了让 D1 的变化肉眼可见，generator 会偏向生成会跨过 bar 轴线、且至少有 2 次转向的 shortest path。
#     - 仍然只保留 1 个 obstacle / 1 个 segment / 1 个 bar 的局部关系，避免自由度过多导致识别歧义。
#   implicit_assumption_review:
#     - 旧 `_greedy_manhattan_path`、`pick_p_near`、`boundary_path`、`dr_ext/dc_ext`、三相 regex 断言、
#       V apex 紧贴断言，都依赖“输出是三相单调包裹”。这些前提在变体中不成立，必须整段移除。
#     - bar / segment / obstacle 的检测逻辑不依赖三相模式，仍然可复用。
#     - 新 solver 新增的假设是：输入实例满足“seg_start 到 farthest-skin exit 的 BFS 最短路长度恰好等于原 segment 长度”，
#       且该路不需要穿过 bar 或 obstacle。generator 会显式构造并验证这一点。
#   placement_legality:
#     - triple 仍用 output-first：先随机 obstacle，再计算 shortest-path output；只有当 route 长度在 5..9、放置不越界、
#       与 input stroke / 其他 triple 不冲突，并且 `main(input) == output` 时才接受。
#     - obstacle 放在 terminal 前方的局部框内，bar 占据 terminal 轴线，route 只在自由区域内寻路。
#   visual_observability:
#     - 新样例会稳定出现正交多折 shortest path；旧题常见的对角相位将消失。
#     - generator 要求 route 至少 2 turns 且跨过 bar 轴线，因此不会长期退化为简单直线或原题式“单侧滑过”。
#   cleanliness:
#     - 多 triple 仍通过 expanded bbox 预留空隙，避免不同局部互相遮挡。
#     - 颜色规则不变：bar / segment / obstacle 三色结构足以让人类看出“哪一段在移动、哪一块是固定障碍物”。

from __future__ import annotations

from collections import deque
import random

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


def _bfs_shortest_count4(
    start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray, cap: int = 3
) -> tuple[int | None, int]:
    h, w = walkable.shape
    q = deque([start])
    dist = {start: 0}
    count = {start: 1}
    while q:
        cur = q.popleft()
        d = dist[cur]
        if goal in dist and d >= dist[goal]:
            continue
        for nr, nc in neighbors4(cur[0], cur[1]):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            if nxt not in dist:
                dist[nxt] = d + 1
                count[nxt] = count[cur]
                q.append(nxt)
            elif dist[nxt] == d + 1:
                count[nxt] = min(cap, count[nxt] + count[cur])
    if goal not in dist:
        return None, 0
    return int(dist[goal]), int(min(cap, count[goal]))


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


def _obstacle_skin(obstacle_set: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    skin = set()
    for r, c in obstacle_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                skin.add((nr, nc))
    return skin


def _exit_key(
    p: tuple[int, int], terminal: tuple[int, int], obstacle_set: set[tuple[int, int]]
) -> tuple[float, float, int, int]:
    cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
    cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
    away = (cr - terminal[0], cc - terminal[1])
    r, c = p
    md = abs(r - terminal[0]) + abs(c - terminal[1])
    dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
    return (md, dot, -r, -c)


def _route_turns(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    dirs = [(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1)]
    return sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])


def _path_crosses_axis(path: list[tuple[int, int]]) -> bool:
    cols = [c for _, c in path]
    return min(cols) < 0 < max(cols)


def _build_local_walkable(
    blocked: set[tuple[int, int]], anchor_pts: list[tuple[int, int]], margin: int
) -> tuple[np.ndarray, tuple[int, int]]:
    rs = [r for r, _ in blocked] + [r for r, _ in anchor_pts]
    cs = [c for _, c in blocked] + [c for _, c in anchor_pts]
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


def _shortest_route_to_farthest_skin(
    start: tuple[int, int],
    terminal: tuple[int, int],
    obstacle_set: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    margin: int = 12,
) -> tuple[list[tuple[int, int]] | None, tuple[int, int] | None, int]:
    skin = {
        p
        for p in {
            q
            for cell in obstacle_set
            for q in neighbors4(cell[0], cell[1])
        }
        if p not in obstacle_set and p not in blocked
    }
    if not skin:
        return None, None, 0
    walkable, offset = _build_local_walkable(blocked, [start, terminal, *skin], margin)
    start_i = _local_to_idx(start, offset)

    best_exit = max(skin, key=lambda p: _exit_key(p, terminal, obstacle_set))
    exit_i = _local_to_idx(best_exit, offset)
    walkable[start_i] = True
    walkable[exit_i] = True
    route_i = _bfs_path4(start_i, exit_i, walkable)
    if route_i is None:
        return None, None, 0
    route = [_idx_to_local(p, offset) for p in route_i]
    dist, count = _bfs_shortest_count4(start_i, exit_i, walkable, cap=3)
    if dist is None:
        return None, None, 0
    return route, best_exit, count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each irregular segment around its nearest obstacle using a 4-neighbor shortest path.

    Detection of bars / attached segments / nearest obstacle is unchanged from the original task.
    The only changed behavior is the output route:
    - start at the attached segment endpoint next to the bar terminal,
    - choose the obstacle's walkable skin cell with maximum Manhattan distance from the terminal,
    - draw the BFS shortest path to that exit while staying outside the obstacle and bar.
    """
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
                yield col, comp, terminal_ep

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

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
                L = len(path)
                key = (L, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (L, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
        assert obstacle_candidates, "must have at least one obstacle color distinct from bar and segment"
        best = None
        obstacle_set = None
        for col in obstacle_candidates:
            for comp in _components(g, col, conn8=True):
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best is None or key < best:
                    best = key
                    obstacle_set = set(comp)
        assert obstacle_set is not None, "must find at least one obstacle component"

        walkable = (g == bg) | (g == seg_col)
        walkable[seg_start] = True
        skin = _obstacle_skin(obstacle_set, h, w)
        skin_candidates = [p for p in skin if bool(walkable[p])]
        assert skin_candidates, "obstacle must expose a walkable skin cell"
        exit_cell = max(skin_candidates, key=lambda p: _exit_key(p, terminal, obstacle_set))
        walkable[exit_cell] = True
        route = _bfs_path4(seg_start, exit_cell, walkable)
        assert route is not None, f"no shortest path from {seg_start} to {exit_cell}"
        assert route[0] == seg_start and route[-1] == exit_cell
        assert len(route) == L, f"segment length mismatch: input={L}, route={len(route)}"
        assert not any(p in obstacle_set for p in route), "output segment passes through obstacle"
        moves.append((seg_comp, seg_col, route))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
OBS_SIZE_MIN, OBS_SIZE_MAX = 2, 7
OBS_MD_MIN, OBS_MD_MAX = 2, 6
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 300
TRIPLE_RETRIES = 250
PLACEMENT_PAD = 1
LOCAL_MARGIN = SEG_LEN_MAX + 4
LOCAL_FWD_MAX = SEG_LEN_MAX + OBS_MD_MAX + 2
LOCAL_SIDE_MAX = SEG_LEN_MAX + 3
MIN_OUTPUT_TURNS = 1
OBS_START_CHOICES = [
    (r, c)
    for r in range(1, OBS_MD_MAX + 1)
    for c in [-2, -1, 0, 1, 2]
    if OBS_MD_MIN <= abs(r) + abs(c) <= OBS_MD_MAX
]

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
    def in_local_bounds(p: tuple[int, int]) -> bool:
        r, c = p
        return 1 <= r <= LOCAL_FWD_MAX and -LOCAL_SIDE_MAX <= c <= LOCAL_SIDE_MAX

    obstacle = {start}
    for _ in range(max_size - 1):
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden and in_local_bounds((nr, nc))
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


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

        start_r, start_c = random.choice(OBS_START_CHOICES)

        obstacle = _grow_obstacle_local(
            (start_r, start_c),
            random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX),
            forbidden=bar_set | {seg_start},
        )
        obstacle_set = set(obstacle)
        nearest_md = min(abs(r) + abs(c) for r, c in obstacle_set)
        if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
            continue
        if any(r <= 0 for r, _ in obstacle_set):
            continue

        route, exit_cell, n_shortest = _shortest_route_to_farthest_skin(
            seg_start,
            (0, 0),
            obstacle_set,
            blocked=obstacle_set | bar_set,
            margin=LOCAL_MARGIN,
        )
        if route is None or exit_cell is None:
            continue
        seg_len = len(route)
        if not (SEG_LEN_MIN <= seg_len <= SEG_LEN_MAX):
            continue
        if _route_turns(route) < MIN_OUTPUT_TURNS:
            continue
        if not _only_root_touches_bar(route, bar):
            continue

        max_turns = min(INPUT_TURN_MAX, seg_len - 2)
        if max_turns < INPUT_TURN_MIN:
            continue
        turns = random.randint(INPUT_TURN_MIN, max_turns)
        input_seg = _build_input_path(seg_len, turns, side)
        if _is_4adj(input_seg, obstacle):
            continue
        if not _only_root_touches_bar(input_seg, bar):
            continue

        union = bar + obstacle + input_seg + route
        rs = [r for r, _ in union]
        cs = [c for _, c in union]
        return {
            "bar_len": bar_len,
            "bar": bar,
            "obstacle": obstacle,
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


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
    bar_len = triple["bar_len"]
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
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 11:
        max_triples = 1
    elif n < 15:
        max_triples = 2
    else:
        max_triples = 3
    triple_count = random.randint(1, max_triples)

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    for edge in random.sample(list(EDGE_DATA), triple_count):
        triple = _make_triple(n)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["obstacle"], obstacle_color)
        paint_coords(out, placed["output_seg"], seg_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    if len(placed_triples) > 1:
        for i, placed in enumerate(placed_triples):
            tr, tc = placed["terminal"]
            own = min(abs(r - tr) + abs(c - tc) for r, c in placed["obstacle"])
            other = min(
                abs(r - tr) + abs(c - tc)
                for j, other_placed in enumerate(placed_triples)
                if j != i
                for r, c in other_placed["obstacle"]
            )
            if own >= other:
                return None

    try:
        if not np.array_equal(main(inp), out):
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
