# Variant of 88bcf3b4
# Dimension: D15 — 线段连接位置 (src)
# Change: 线段的一个端点通过 4-邻接连接到 bar 的内部末端（terminal），且连接方向垂直于 bar 方向
#         → 线段改为连接到 bar 的内部中段连接点（attach），bar 继续越过连接点向网格内部延伸。
#
# Design analysis
# spec:
#   preserved_core:
#     - bar 仍然是垂直于边界的直线，且恰有一个端点在边界上。
#     - segment 仍然是一条 1-cell-thick 的不规则 8-connected 路径，像绳索一样绕最近障碍物重定位。
#     - obstacle 仍然保持不动，segment 总像素数保持不变。
#   D15_change:
#     - 原题把 bar 的内部端点 terminal 当作 segment 的锚点；solver 直接检查 terminal 的 4 邻域即可找到 segment。
#     - 变体中锚点改成 bar 内部的 attach 点。bar 会继续穿过 attach，向网格内部多出 2-3 个像素的“尾巴”。
#     - 人类要先观察“真正与 segment 端点 4 邻接的是 bar 的哪一个内部像素”，不能再把 bar 的末端当作默认起点。
#   concrete_behavior:
#     - 输入里，segment 的一个端点仍然垂直贴着 bar，但贴着的是 bar 中段。
#     - 输出里，segment 仍从该 attach 点旁边出发，向前接近 obstacle、贴着 obstacle 外皮走，再从另一侧斜向离开。
#     - bar 的 interior terminal 仍然存在，但不再是 segment 的根部。
#   small_trace:
#     - 原逻辑：vertical bar 顶端接边界，terminal 在较深处；solver 在 terminal 旁找到 seg_start，再以 terminal 为 rope source。
#     - 新逻辑：同样的 bar 现在在 attach 之后还有尾巴；solver 必须在 bar 的内部像素中找到那个与 seg_start 4 邻接且 seg_start 是路径端点的位置，
#       再以 attach 为 rope source。若继续使用 terminal，路径会从错误的行/列出发，输出会整体错位。
#   complexity_calibration:
#     - 只增加一个可见自由度：attach 深度固定在 bar 中央 1-2 个候选格之一，不引入更多类别分支。
#     - 为了让变化肉眼可见，bar 长度下限提高到 5，保证 attach 两侧都留有可见 bar 像素，不会退化成“几乎还是端点连接”。
#   implicit_assumption_audit_for_original_main:
#     - 原 bar_candidates 假设“唯一相关的非背景邻居发生在 terminal 周围”。该前提失效，必须扫描整根 bar 的内部像素。
#     - 原 obstacle 距离、入口路径、单调性判断都以 terminal 为参考点。变体中这些参考点都必须统一改成 attach。
#     - 原生成器把局部坐标原点设在 terminal，并让 bar 只朝边界方向展开。变体中若不改原点，输出段和 obstacle 的布局会与新锚点不一致。
#   placement_legality:
#     - generator 以 attach 为局部原点。
#     - bar 使用局部坐标区间 [-attach_depth, tail_len]，其中 attach_depth 取 bar 中央位置，tail_len >= 2。
#     - obstacle 需要远离 bar 尾巴，否则 bar 其他像素会额外接触 obstacle，既难看也会给 bar 匹配带来噪音。
#     - 因此 phase1 至少 2 步，且 obstacle 生长时避开 bar 及其 4-neighborhood halo。
#   visual_observability:
#     - 同一组样例中，bar 明显会“穿过”连接点继续延伸；这是 D15 的主要视觉证据。
#     - 连接点保持垂直贴边，颜色逻辑不变，解题者仍能从结构而不是颜色数值推理规则。
#   cleanliness:
#     - segment 除根部外不会再贴着 bar 尾巴平行前进；phase1 的第一步直接把 lateral offset 从 1 扩到 2，避免视觉叠靠。
#     - obstacle 与 bar 留出至少 1 格缓冲，避免 bar/obstacle/segment 三者在同一区域混杂。
#   generator_non_degeneracy_check:
#     - 每个样本都保证 attach 不是端点，且 terminal 与 seg_start 不 4 邻接。
#     - 额外统计 attach 后 bar 尾长 tail_len 以及“原 solver 在这些输入上失败或给出不同输出”的比例，防止样本退化回原题。
#   new_main_assumptions_to_guard:
#     - 真正的 segment 根像素是某个 bar 内部像素的 4 邻接点，且该根像素是 segment 组件的度为 1 的端点。
#     - obstacle 不与 bar 4 邻接；否则 bar 内部会出现与 obstacle 的无关接触点。
#     - 同一 bar 只会有一个合法 attach/segment 组合；generator 通过空间留白和 halo 保证这一点。

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
    assert goal in prev, "BFS-with-diag-filter must reach goal"
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


def _bar_forward(border_ep: tuple[int, int], vertical: bool, h: int, w: int) -> tuple[int, int]:
    if vertical:
        return (1, 0) if border_ep[0] == 0 else (-1, 0)
    return (0, 1) if border_ep[1] == 0 else (0, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant D15: the rope-like segment is attached to an interior point on the bar,
    not to the bar terminal. Re-route the segment from that attach point around the
    nearest obstacle while preserving segment length.
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
                terminal_ep = e1 if touch0 else e0
                if vertical:
                    if border_ep[0] not in (0, h - 1):
                        continue
                else:
                    if border_ep[1] not in (0, w - 1):
                        continue
                fwd = _bar_forward(border_ep, vertical, h, w)
                bar_set = set(comp)
                attach_options = []
                for br, bc in comp:
                    prev_cell = (br - fwd[0], bc - fwd[1])
                    next_cell = (br + fwd[0], bc + fwd[1])
                    if prev_cell not in bar_set or next_cell not in bar_set:
                        continue
                    for nr, nc in neighbors4(br, bc):
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        v = int(g[nr, nc])
                        if v != bg and v != col:
                            attach_options.append(((br, bc), (nr, nc)))
                if attach_options:
                    yield col, comp, border_ep, terminal_ep, fwd, attach_options

    moves = []
    for bar_col, bar_comp, border_ep, terminal_ep, fwd, attach_options in sorted(
        list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2], x[3])
    ):
        bar_set = set(bar_comp)
        best_seg = None
        for attach, cand_start in attach_options:
            dr0 = cand_start[0] - attach[0]
            dc0 = cand_start[1] - attach[1]
            if abs(dr0) + abs(dc0) != 1:
                continue
            if dr0 * fwd[0] + dc0 * fwd[1] != 0:
                continue
            cand_col = int(g[cand_start])
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
                key = (L, len(comp), cand_col, attach[0], attach[1], cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, attach, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment attached to bar with terminal {terminal_ep}"
        (L, _, _, _, _, _, _), attach, seg_col, seg_start, seg_comp = best_seg

        obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
        assert obstacle_candidates, "must have at least one obstacle color distinct from bar and segment"

        best = None
        obstacle_set = None
        for col in obstacle_candidates:
            for comp in _components(g, col, conn8=True):
                d = min(abs(r - attach[0]) + abs(c - attach[1]) for r, c in comp)
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best is None or key < best:
                    best = key
                    obstacle_set = set(comp)
        assert obstacle_set is not None, "must find at least one obstacle component"

        off_r = seg_start[0] - attach[0]
        off_c = seg_start[1] - attach[1]

        walkable = (g == bg) | (g == seg_col)
        walkable[attach] = True

        def pick_p_near() -> tuple[int, int]:
            good = []
            for pr, pc in obstacle_set:
                er, ec = pr + off_r, pc + off_c
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if (er, ec) in obstacle_set:
                    continue
                if int(g[er, ec]) not in (bg, seg_col):
                    continue
                d = abs(pr - attach[0]) + abs(pc - attach[1])
                good.append((d, pr, pc))
            assert good, "no obstacle pixel has a valid translated endpoint outside obstacle"
            _, pr, pc = min(good)
            return (pr, pc)

        p_near = pick_p_near()
        walkable[p_near] = True
        path_a_to_obs = _greedy_manhattan_path(attach, p_near, walkable) or _bfs_path(attach, p_near, walkable)
        assert path_a_to_obs is not None, f"no path from attach {attach} to obstacle {p_near}"
        shifted = [(r + off_r, c + off_c) for r, c in path_a_to_obs]
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted path must stay within grid"
        assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
        cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
        away = (cr - attach[0], cc - attach[1])

        def skin_ok(p: tuple[int, int]) -> bool:
            r, c = p
            return int(g[r, c]) == bg or int(g[r, c]) == seg_col

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]

        def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
            r, c = p
            md = abs(r - attach[0]) + abs(c - attach[1])
            dot = (r - attach[0]) * away[0] + (c - attach[1]) * away[1]
            return (md, dot, -r, -c)

        exit_cell = max(skin_candidates, key=exit_key)

        skin_walkable = np.zeros((h, w), dtype=bool)
        for r, c in skin:
            if skin_ok((r, c)):
                skin_walkable[r, c] = True
        skin_walkable[entry] = True
        skin_walkable[exit_cell] = True
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]

        assert len(boundary_path) >= 2, "boundary path must have at least 2 cells to derive extension direction"
        dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
        dc_ext = boundary_path[-1][1] - boundary_path[-2][1]

        route = list(shifted)
        if boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        cur = route[-1]
        while len(route) < L:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != bg and int(g[nr, nc]) != seg_col:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        new_pts = route[:L]
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

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

        first_step = step_classes[0] if step_classes else ""
        if first_step in ("FL", "FR"):
            p1_end = 0
            for j2, sc in enumerate(step_classes):
                if sc == first_step:
                    p1_end = j2 + 1
                else:
                    break
            apex = new_pts[p1_end]
            perp_dir = (seg_start[0] - attach[0], seg_start[1] - attach[1])
            inner = (-perp_dir[0], -perp_dir[1])
            apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
            assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"

        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


# --------------- generator (variant D15) ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 9, 27
BAR_LEN_MIN, BAR_LEN_MAX = 5, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 3, 8
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 400
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _attach_depth_options(bar_len: int) -> list[int]:
    center = (bar_len - 1) / 2
    return [d for d in range(1, bar_len - 1) if abs(d - center) <= 0.5]


def _attach_for(edge: str, n: int, pos: int, attach_depth: int) -> tuple[int, int]:
    if edge == "top":
        return (attach_depth, pos)
    if edge == "bottom":
        return (n - 1 - attach_depth, pos)
    if edge == "left":
        return (pos, attach_depth)
    return (pos, n - 1 - attach_depth)


def _transform(
    locals_: list[tuple[int, int]], edge: str, n: int, pos: int, attach_depth: int
) -> list[tuple[int, int]]:
    ar, ac = _attach_for(edge, n, pos, attach_depth)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(ar + lr * fr + lc * sr, ac + lr * fc + lc * sc) for lr, lc in locals_]


def _halo4(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(coords)
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            out.add((nr, nc))
    return out


def _grow_obstacle(
    start: tuple[int, int], max_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = {start}
    for _ in range(max_size - 1):
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _build_input_path(seg_len: int, turns: int, side: int) -> list[tuple[int, int]]:
    """
    Build a simple irregular stroke whose only bar-adjacent cell is the root.
    The first move immediately increases lateral offset, so the rest of the path
    stays visually detached from the bar tail.
    """
    remaining = seg_len - 1
    coords = [(0, side)]
    steps = [(1, side), (1, 0), (0, side)]
    last_kind = None
    turns_left = turns
    while remaining > 0:
        candidates = [0, 2] if abs(coords[-1][1]) == 1 else [0, 1, 2]
        if turns_left <= 0 and last_kind is not None:
            kind = last_kind
        else:
            if last_kind is None:
                kind = random.choice(candidates)
            else:
                if random.random() < DIAG_CONTINUE_PROB:
                    kind = last_kind
                else:
                    alt = [k for k in candidates if k != last_kind]
                    kind = random.choice(alt) if alt else last_kind
                if kind != last_kind:
                    turns_left -= 1
        max_run = remaining if turns_left <= 0 else max(1, remaining - turns_left)
        run = random.randint(1, max_run)
        step = steps[kind]
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + step[0], pc + step[1]))
        remaining -= run
        last_kind = kind
    return coords


def _build_output_path(seg_len: int, side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, side)]
    phase_steps = [(1, side), (1, 0), (1, -side)]
    for phase_len, (dr, dc) in zip([p1, p2, p3], phase_steps):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
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


def _bar_touch_count(pt: tuple[int, int], bar_set: set[tuple[int, int]]) -> int:
    count = 0
    for nr, nc in neighbors4(pt[0], pt[1]):
        if (nr, nc) in bar_set:
            count += 1
    return count


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    attach_depth = random.choice(_attach_depth_options(bar_len))
    tail_len = bar_len - 1 - attach_depth
    max_seg = min(SEG_LEN_MAX, n - attach_depth)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side = random.choice((-1, 1))

    bar = [(k, 0) for k in range(-attach_depth, tail_len + 1)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(2, min(remaining - 1, PHASE_MAX))
    remaining -= p1
    p2 = random.randint(0, min(remaining, PHASE_MAX))
    p3 = remaining - p2

    output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(output_seg)

    root = output_seg[0]
    if _bar_touch_count(root, bar_set) != 1:
        return None
    if any(_bar_touch_count(p, bar_set) > 0 for p in output_seg[1:]):
        return None

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obstacle_seed_forbidden = seg_set | _halo4(bar_set)
    if apex_inner in obstacle_seed_forbidden:
        return None
    obs_size = random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(apex_inner, obs_size, obstacle_seed_forbidden)
    if _is_4adj(obstacle, list(bar_set)):
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side)
    if _is_4adj(input_seg, obstacle):
        return None
    if _bar_touch_count(input_seg[0], bar_set) != 1:
        return None
    if any(_bar_touch_count(p, bar_set) > 0 for p in input_seg[1:]):
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
        "attach_depth": attach_depth,
        "tail_len": tail_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


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
    attach_depth = triple["attach_depth"]

    for pos in positions:
        union_g = _transform(union, edge, n, pos, attach_depth)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        placed = {
            "bar": _transform(triple["bar"], edge, n, pos, attach_depth),
            "obstacle": _transform(triple["obstacle"], edge, n, pos, attach_depth),
            "input_seg": _transform(triple["input_seg"], edge, n, pos, attach_depth),
            "output_seg": _transform(triple["output_seg"], edge, n, pos, attach_depth),
            "attach": _transform([(0, 0)], edge, n, pos, attach_depth)[0],
            "terminal": _transform([(triple["tail_len"], 0)], edge, n, pos, attach_depth)[0],
            "reserve": (r0, c0, r1, c1),
            "tail_len": triple["tail_len"],
        }
        return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 13:
        max_triples = 1
    elif n < 18:
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

        root = placed["input_seg"][0]
        tr = placed["terminal"]
        if abs(root[0] - tr[0]) + abs(root[1] - tr[1]) == 1:
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
            ar, ac = placed["attach"]
            own = min(abs(r - ar) + abs(c - ac) for r, c in placed["obstacle"])
            other = min(
                abs(r - ar) + abs(c - ac)
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
