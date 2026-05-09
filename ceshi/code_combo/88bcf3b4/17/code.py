# Combo variant of 88bcf3b4 (index 17)
# combo_spec: D12/0+D15/0
# Name: L形bar中点连接
#
# Design analysis
# spec:
#   preserved_core:
#     - 仍然是 1-3 组 "bar + segment + obstacle" 三元组；segment 是细的 8-connected simple path，
#       obstacle 保持不动，segment 输出时像绳索一样绕最近 obstacle 并保持像素数。
#     - 输出路径的宏观单调性仍是相对边界法向的 3 段式：靠近 obstacle 的斜向 approach、沿 obstacle 前进、
#       再从另一侧斜向 exit。
#   fused_change:
#     - D12 把原来的直 bar 改成 L 形 bar：一条贴边法向的主干加一条较短的侧向短臂。
#     - D15 再把连接点从 bar 末端改到 bar 中间；在这个融合里，“中间”具体化为 L 的拐点 elbow。
#     - 因而 solver 不能再通过“边界那头直线的内端 terminal”定位 segment，只能先识别 L 形结构，
#       再把拐点当作 attach。
#   concrete_behavior:
#     - 输入中，每个 bar 是单色 L：一条较长主干从边界伸入网格，拐点处再横出 2-3 格短臂。
#     - segment 的根像素与拐点 4 邻接，但在短臂的反侧；也就是说，短臂和 segment 分居拐点两边。
#     - 输出中，segment 仍从该拐点旁边重新出发，靠近最近 obstacle、贴 obstacle 外皮走，再斜向离开。
#   small_trace:
#     - 原题：顶部竖 bar 的 terminal 在更深处；solver 看 terminal 的 4 邻域就能找到 segment 根，再以
#       terminal 为 rope source。
#     - 融合后：顶部对象先要识别成 "主干向下 + 拐点右短臂" 的 L，真正的 attach 是拐点本身；segment 根在
#       拐点左边，而不是任何末端。若仍把主干末端或短臂末端当锚点，整条输出 route 都会错位。
#   complexity_calibration:
#     - 只引入一个清晰可见的新几何自由度：短臂左右/上下分侧，以及主干长度 4-7、短臂长度 2-3。
#     - 不再额外引入多种 attach 类型；所有样本都固定为“拐点连接”，这样人眼能在 2-4 个例子里稳定归纳。
#     - 主干显著长于短臂，避免 bar 退化成“几乎对称的小角块”，保证谁是主干、谁是短臂一眼可见。
#   implicit_assumption_audit_for_original_main:
#     - 原 `_is_straight_line` / `_bar_endpoints` 假设 bar 是单条直线；这里完全失效，必须改为检测 corner-L。
#     - 原 solver 把 terminal 作为路径起点、障碍距离参考点、单调性参考点；融合后这些参考点统一改成 elbow attach。
#     - 原 generator 以直 bar 的内端为局部原点；融合后若不改，segment 与 obstacle 会围绕错误位置布置。
#   placement_legality:
#     - generator 以 elbow attach 为局部原点；主干坐标是 [-stem_len, 0]，短臂坐标是 [1, arm_len] 的侧向射线。
#     - obstacle 生长时避开 bar 的 4-neighborhood halo，避免 bar/obstacle 无关接触污染 attach 检测。
#     - input/output segment 除根像素外都禁止再碰 bar，避免 segment 沿短臂贴行造成视觉噪音与匹配歧义。
#   visual_observability:
#     - 人眼会同时看到两件变化：bar 不再是直线而是 L，且 rope 并不接在任何 bar 末端，而是接在转角处。
#     - 短臂和 segment 根分列拐点两侧，这给出强结构线索，不依赖颜色数值或隐式排序。
#   cleanliness:
#     - segment 第一段会立刻向前/向外离开 elbow，避免和短臂平贴；obstacle 则固定落在输出 V 凹侧且远离 bar halo。
#     - 各三元组通过扩张 bbox 分区放置，减少多组对象间的互扰。
#   generator_non_degeneracy_check:
#     - 每个样本都保证 bar 组件是非直线的真正 L，且 segment 根只接触 elbow 一个 bar 像素。
#     - 若把 bar 简化回直线，或把 attach 错认成短臂末端，生成样本上的几何统计会立刻失真，因此后续验证会统计
#       stem_len / arm_len / root_side，并检查 bar 无一是直线组件。
#   new_main_assumptions_to_guard:
#     - 某个 bar 组件必须恰好是 corner-L，且其两条臂中只有主干那一端触边；generator 保证这一点。
#     - attach 的非 bar 邻居里，只有一个合法 segment 根，它是对应 segment 组件的度 1 端点；generator 保证。
#     - obstacle 不与 bar 4 邻接，且每个 attach 最近的 obstacle 属于本三元组；generator 通过 halo 和最近距检查保证。

import random
import re
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _step_toward(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr != 0:
        dr //= abs(dr)
    if dc != 0:
        dc //= abs(dc)
    return dr, dc


def _analyze_l_bar(comp: list[tuple[int, int]], h: int, w: int) -> dict | None:
    pts = set(comp)
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1 or len(set(cs)) == 1:
        return None
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    found = []
    for er, ec in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
        horiz_end = (er, c1 if ec == c0 else c0)
        vert_end = (r1 if er == r0 else r0, ec)
        expected = {(er, c) for c in range(min(ec, horiz_end[1]), max(ec, horiz_end[1]) + 1)}
        expected |= {(r, ec) for r in range(min(er, vert_end[0]), max(er, vert_end[0]) + 1)}
        if pts != expected:
            continue
        for stem_far, arm_end in ((vert_end, horiz_end), (horiz_end, vert_end)):
            if not _touches_border(stem_far, h, w):
                continue
            if _touches_border(arm_end, h, w):
                continue
            fwd = _step_toward(stem_far, (er, ec))
            arm_dir = _step_toward((er, ec), arm_end)
            stem_len = abs(stem_far[0] - er) + abs(stem_far[1] - ec)
            arm_len = abs(arm_end[0] - er) + abs(arm_end[1] - ec)
            if stem_len < 2 or arm_len < 1:
                continue
            found.append(
                {
                    "attach": (er, ec),
                    "border_ep": stem_far,
                    "arm_end": arm_end,
                    "fwd": fwd,
                    "arm_dir": arm_dir,
                    "stem_len": stem_len,
                    "arm_len": arm_len,
                }
            )
    if len(found) != 1:
        return None
    return found[0]


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


def _bar_touch_count(pt: tuple[int, int], bar_set: set[tuple[int, int]]) -> int:
    return sum((nr, nc) in bar_set for nr, nc in neighbors4(pt[0], pt[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each rope-like segment from the elbow of an L-shaped bar so it wraps
    around the nearest obstacle, while keeping the segment length unchanged.

    Input structure:
    - Non-background objects come in triples: an L-shaped bar whose long stem is perpendicular to one grid edge, an irregular 1-cell-thick segment, and a nearby obstacle sprite.
    - The segment is not attached to a bar endpoint. Its endpoint sits 4-neigh from the L corner, on the side opposite the bar's short arm.

    Rule:
    - For each valid L bar, first recover its elbow attach point from shape, not from any terminal endpoint.
    - Detach the segment from that elbow-side root and route it toward the nearest obstacle, along the obstacle skin, then out diagonally on the far side.
    - The obstacle and bar stay fixed; the segment preserves its original pixel count and must not pass through the obstacle.

    Key insight:
    - The source anchor is encoded geometrically by the L corner. The bar's visible endpoints are distractions; the true rope origin is the unique elbow where the stem meets the short arm.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            for comp in _components(g, col, conn8=False):
                info = _analyze_l_bar(comp, h, w)
                if info is None:
                    continue
                attach = info["attach"]
                arm_dir = info["arm_dir"]
                fwd = info["fwd"]
                seg_neighbors = []
                for nr, nc in neighbors4(attach[0], attach[1]):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in set(comp):
                        continue
                    v = int(g[nr, nc])
                    if v == bg or v == col:
                        continue
                    dr0 = nr - attach[0]
                    dc0 = nc - attach[1]
                    if dr0 * fwd[0] + dc0 * fwd[1] != 0:
                        continue
                    if (dr0, dc0) == arm_dir:
                        continue
                    seg_neighbors.append((nr, nc))
                if seg_neighbors:
                    yield col, comp, info, seg_neighbors

    moves = []
    for bar_col, bar_comp, bar_info, seg_neighbors in sorted(
        list(bar_candidates()),
        key=lambda x: (-x[2]["stem_len"], -x[2]["arm_len"], x[0], x[2]["attach"]),
    ):
        attach = bar_info["attach"]
        fwd = bar_info["fwd"]
        arm_dir = bar_info["arm_dir"]
        best_seg = None
        for cand_start in seg_neighbors:
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
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment attached to L-bar at {attach}"
        (L, _, _, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

        off_r = seg_start[0] - attach[0]
        off_c = seg_start[1] - attach[1]
        assert (off_r, off_c) != arm_dir, "segment root must be opposite the short arm"

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
            return pr, pc

        p_near = pick_p_near()
        walkable[p_near] = True
        path_a_to_obs = _greedy_manhattan_path(attach, p_near, walkable) or _bfs_path(attach, p_near, walkable)
        assert path_a_to_obs is not None, f"no path from attach {attach} to obstacle {p_near}"
        shifted = [(r + off_r, c + off_c) for r, c in path_a_to_obs]
        assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted path must stay within grid"
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

        if len(boundary_path) >= 2:
            dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
            dc_ext = boundary_path[-1][1] - boundary_path[-2][1]
        else:
            dr_ext = fwd[0] - off_r
            dc_ext = fwd[1] - off_c

        route = list(shifted)
        if boundary_path and boundary_path[0] == entry:
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
        assert len(new_pts) == L, "output route must preserve segment length"
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
            inner = (-off_r, -off_c)
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


# --------------- generator (combo D12/0 + D15/0) ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 10, 27
STEM_LEN_MIN, STEM_LEN_MAX = 4, 7
ARM_LEN_MIN, ARM_LEN_MAX = 2, 3
STEM_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 3, 8
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
DIAG_CONTINUE_PROB = 0.35
DIAG_START_PROB = 0.5
MAX_RETRIES = 500
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _attach_for(edge: str, n: int, pos: int, stem_len: int) -> tuple[int, int]:
    if edge == "top":
        return stem_len, pos
    if edge == "bottom":
        return n - 1 - stem_len, pos
    if edge == "left":
        return pos, stem_len
    return pos, n - 1 - stem_len


def _transform(
    locals_: list[tuple[int, int]], edge: str, n: int, pos: int, stem_len: int
) -> list[tuple[int, int]]:
    ar, ac = _attach_for(edge, n, pos, stem_len)
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


def _make_triple(n: int) -> dict | None:
    max_stem = min(STEM_LEN_MAX, n - STEM_CLEARANCE)
    if max_stem < STEM_LEN_MIN:
        return None

    stem_len = random.randint(STEM_LEN_MIN, max_stem)
    arm_len = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    max_seg = min(SEG_LEN_MAX, n - stem_len)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)

    seg_side = random.choice((-1, 1))
    arm_side = -seg_side
    stem = [(k, 0) for k in range(-stem_len, 1)]
    arm = [(0, arm_side * k) for k in range(1, arm_len + 1)]
    bar = stem + arm
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(2, min(remaining - 1, PHASE_MAX))
    remaining -= p1
    p2 = random.randint(0, min(remaining, PHASE_MAX))
    p3 = remaining - p2

    output_seg = _build_output_path(seg_len, seg_side, p1, p2, p3)
    seg_set = set(output_seg)
    root = output_seg[0]
    if _bar_touch_count(root, bar_set) != 1:
        return None
    if any(_bar_touch_count(p, bar_set) > 0 for p in output_seg[1:]):
        return None

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - seg_side)
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
    input_seg = _build_input_path(seg_len, turns, seg_side)
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
        "stem_len": stem_len,
        "arm_len": arm_len,
        "seg_side": seg_side,
        "arm_side": arm_side,
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
    stem_len = triple["stem_len"]

    for pos in positions:
        union_g = _transform(union, edge, n, pos, stem_len)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        placed = {
            "bar": _transform(triple["bar"], edge, n, pos, stem_len),
            "obstacle": _transform(triple["obstacle"], edge, n, pos, stem_len),
            "input_seg": _transform(triple["input_seg"], edge, n, pos, stem_len),
            "output_seg": _transform(triple["output_seg"], edge, n, pos, stem_len),
            "attach": _transform([(0, 0)], edge, n, pos, stem_len)[0],
            "reserve": (r0, c0, r1, c1),
            "stem_len": triple["stem_len"],
            "arm_len": triple["arm_len"],
            "seg_side": triple["seg_side"],
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
