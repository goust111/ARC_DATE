# Combo variant of 88bcf3b4 (index 21)
# combo_spec: D4/3+D12/0
# Name: U形线段L形bar
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题里，每个 triple 由“贴边直 bar + terminal 旁单端细线 + nearby obstacle”组成；solver 把 terminal 当作唯一根，
#      再把细线甩到 obstacle 周围。
#    - 本融合版同时改两处：
#      1. bar 不再是单一直线，而是一个贴边的 L：主干仍垂直于边界，内端拐角再接一个短臂。
#      2. 输入线段不再只有一个端点贴 bar，而是一条细 U 形桥，两端分别贴在 L 的拐角前方一格和短臂末端前方一格。
#    - 具体小例子：若某个 top-edge L bar 的拐角在 `(4,5)`、短臂向右到 `(4,7)`，则输入 U 端点在 `(5,5)` 与 `(5,7)`。
#      如果 obstacle 靠右，solver 先认出这是一条双端 simple path，再比较两端到 obstacle 的距离：
#      右端更近，所以右端保留，左端脱离；输出就从 `(5,7)` 出发按原题的绕障逻辑重建一条同长度路径。
#    - 与原题的差异不是“先认 L、再认 U”两件孤立小改动，而是 attachment 几何已经改变：
#      solver 必须先恢复 L bar 的朝向与两个合法附着点，才能判断哪一端移动。
#
# b) 复杂度校准
#    - 保留原题的 1-3 个 triple、四个边缘朝向、随机 obstacle sprite 与输出绕障路径，避免退化成死模板。
#    - 同时控制新自由度，避免谜题变脏：
#      - U 线段只使用 5 个长度档位里的少量规范 U 模板；
#      - L bar 的短臂长度只用 2 或 3，与 U 宽度一致；
#      - “哪一端固定”只在 `elbow` / `tip` 两类中切换。
#    - 这样人类能在 2-4 个样例里稳定看到：
#      “bar 是 L，不是直杆” + “segment 两端都接 bar” + “靠障碍更近的一端留下，另一端被甩走”。
#
# c) 隐式假设审查
#    - 原 solver 的 `_is_straight_line(comp)` / `_bar_endpoints(comp)` 假设 bar 是单条直线，这里完全失效，
#      必须改成显式识别 L bar 的 border cell、拐角、短臂方向和 tip。
#    - 原 solver 假设 terminal 邻域里只有一个 segment 根端点；这里改成“同一 simple-path 组件的两个端点都贴 bar”，
#      所以 segment 检测要从“一个 endpoint”改成“两个 endpoints + 各自唯一 bar anchor”。
#    - 原 solver 的 `seg_start - terminal` 是 lateral offset；这里 attachment offset 改为 bar 的 `forward`，
#      因为两个端点都放在各自 anchor 的前方一格。绕障平移仍可沿用，但 anchor 不再是统一 terminal，而是保留端对应的 anchor。
#    - 原 solver 用 terminal 选最近 obstacle。融合后更稳妥的前提是：
#      obstacle 对两个 attached endpoints 的最小距离在本 triple 内唯一最小；generator 显式保证这一点。
#
# d) 放置合法性
#    - 生成器继续使用 output-first 风格：先造 L bar、合法 output route 和 obstacle，再反推输入 U。
#    - reserve 区域覆盖 bar + obstacle + input U + output route 的完整并集，而不是只覆盖输入对象，
#      这样不同 triple 不会共享未来绕行走廊。
#    - `_place_triple()` 只接受完全在界内、expanded bbox 不冲突的位置；多 triple 时还额外检查“自有 obstacle 最近”。
#
# e) 视觉可观察性
#    - 输入中可见的新规律是非常直接的：bar 从直杆变成 L，线段从单挂变成拱桥式 U。
#    - 输出中只保留 obstacle 更近的一端继续贴 bar，因此几组样例足够支持“近端留下、远端移动”的归纳，
#      不依赖颜色值、组件编号或隐藏排序。
#    - 固定端在 `elbow` 与 `tip` 两类之间都会出现，避免解题者误以为总是固定某一侧。
#
# f) 可视化整洁度
#    - U 模板全部是 1-cell-thick 的简单 4 连通桥，bar 与 obstacle 不直接接触，triple 间保留 1 格 padding。
#    - obstacle 生长点放在 output route 的 V apex 内侧，但通过 `p1 >= 2` 与显式过滤，避免它贴到输入 U 上造成看图混乱。
#
# g) 变化的可推断性
#    - 颜色角色不变：bar / segment / obstacle 三色各司其职，规则完全由可见几何关系决定。
#    - 量化上，本融合版要求每个有效 triple 同时满足：
#      - `bar_arm_len > 0`，即 bar 真的是 L；
#      - `attached_endpoint_count == 2`，即 segment 真的是双端 U；
#      - `dist_gap = d_far - d_near >= 1`，即“哪一端移动”是可见且非 tie 的。
#
# h) 新 solver 假设与 generator 保证
#    - solver 假设 L bar 由一条贴边主干和一条拐角短臂组成，且 bar 只在一个边界侧触边；generator 严格按此模板构造。
#    - solver 假设 segment 是 simple path，且两个 endpoints 分别等于 `corner+fwd` 与 `tip+fwd`；generator 直接用规范 U 模板保证。
#    - solver 假设固定端与移动端到 obstacle 的距离不相等；generator 在局部坐标下先筛掉 tie，再做全局 nearest-obstacle 校验。

from __future__ import annotations

import random
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _bar_forward_dir(edge: str) -> tuple[int, int]:
    if edge == "top":
        return (1, 0)
    if edge == "bottom":
        return (-1, 0)
    if edge == "left":
        return (0, 1)
    return (0, -1)


def _segment_neighbors(p: tuple[int, int], sset: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = p
    out = []
    for q in neighbors4(r, c):
        if q in sset:
            out.append(q)
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        q = (r + dr, c + dc)
        if q not in sset:
            continue
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if mid1 in sset or mid2 in sset:
            continue
        out.append(q)
    return out


def _segment_degrees(seg_pts: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    sset = set(seg_pts)
    return {p: len(_segment_neighbors(p, sset)) for p in sset}


def _segment_path_from_endpoint(seg_pts: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    sset = set(seg_pts)
    path = [start]
    prev = None
    cur = start
    seen = {start}
    while True:
        nxts = [q for q in _segment_neighbors(cur, sset) if q != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "segment must be a simple path"
        nxt = nxts[0]
        assert nxt not in seen, "segment path must not self-intersect"
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt
    assert len(path) == len(seg_pts), "endpoint traversal must cover the whole segment"
    return path


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


def _analyze_l_bar(comp: list[tuple[int, int]], h: int, w: int) -> dict | None:
    pts = set(comp)
    border_pts = [p for p in comp if _touches_border(p, h, w)]
    if len(border_pts) != 1:
        return None
    br, bc = border_pts[0]
    on_edges = []
    if br == 0:
        on_edges.append("top")
    if br == h - 1:
        on_edges.append("bottom")
    if bc == 0:
        on_edges.append("left")
    if bc == w - 1:
        on_edges.append("right")
    if len(on_edges) != 1:
        return None
    edge = on_edges[0]

    if edge in ("top", "bottom"):
        trunk_c = bc
        trunk_rows = sorted(r for r, c in comp if c == trunk_c)
        if edge == "top":
            if trunk_rows != list(range(0, max(trunk_rows) + 1)):
                return None
            corner_r = max(trunk_rows)
        else:
            if trunk_rows != list(range(min(trunk_rows), h)):
                return None
            corner_r = min(trunk_rows)
        arm_cols = sorted(c for r, c in comp if r == corner_r)
        if trunk_c not in arm_cols or arm_cols != list(range(min(arm_cols), max(arm_cols) + 1)):
            return None
        tip_c = min(arm_cols) if min(arm_cols) != trunk_c else max(arm_cols)
        if tip_c == trunk_c:
            return None
        trunk = {(r, trunk_c) for r in trunk_rows}
        arm = {(corner_r, c) for c in arm_cols}
        if pts != trunk | arm:
            return None
        side_sign = 1 if tip_c > trunk_c else -1
        corner = (corner_r, trunk_c)
        tip = (corner_r, tip_c)
    else:
        trunk_r = br
        trunk_cols = sorted(c for r, c in comp if r == trunk_r)
        if edge == "left":
            if trunk_cols != list(range(0, max(trunk_cols) + 1)):
                return None
            corner_c = max(trunk_cols)
        else:
            if trunk_cols != list(range(min(trunk_cols), w)):
                return None
            corner_c = min(trunk_cols)
        arm_rows = sorted(r for r, c in comp if c == corner_c)
        if trunk_r not in arm_rows or arm_rows != list(range(min(arm_rows), max(arm_rows) + 1)):
            return None
        tip_r = min(arm_rows) if min(arm_rows) != trunk_r else max(arm_rows)
        if tip_r == trunk_r:
            return None
        trunk = {(trunk_r, c) for c in trunk_cols}
        arm = {(r, corner_c) for r in arm_rows}
        if pts != trunk | arm:
            return None
        side_sign = 1 if tip_r > trunk_r else -1
        corner = (trunk_r, corner_c)
        tip = (tip_r, corner_c)

    return {
        "edge": edge,
        "fwd": _bar_forward_dir(edge),
        "corner": corner,
        "tip": tip,
        "bar_set": pts,
        "arm_len": abs(tip[0] - corner[0]) + abs(tip[1] - corner[1]),
        "trunk_len": len(trunk_rows) if edge in ("top", "bottom") else len(trunk_cols),
        "side_sign": side_sign,
    }


def _classify_wrap_steps(route: list[tuple[int, int]], fwd: tuple[int, int]) -> list[str]:
    step_classes = []
    for j in range(len(route) - 1):
        dr = route[j + 1][0] - route[j][0]
        dc = route[j + 1][1] - route[j][1]
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
    return step_classes


def _canonical_phase_split(seg_len: int) -> tuple[int, int, int]:
    phase_map = {
        5: (2, 1, 1),
        6: (2, 1, 2),
        7: (3, 1, 2),
        8: (3, 2, 2),
        9: (3, 2, 3),
    }
    return phase_map[seg_len]


def _canonical_route(
    keep_anchor: tuple[int, int],
    keep_name: str,
    seg_len: int,
    fwd: tuple[int, int],
    side_vec: tuple[int, int],
) -> list[tuple[int, int]]:
    p1, p2, p3 = _canonical_phase_split(seg_len)
    root_sign = -1 if keep_name == "elbow" else 1
    steps = [
        (fwd[0] + root_sign * side_vec[0], fwd[1] + root_sign * side_vec[1]),
        fwd,
        (fwd[0] - root_sign * side_vec[0], fwd[1] - root_sign * side_vec[1]),
    ]
    route = [(keep_anchor[0] + fwd[0], keep_anchor[1] + fwd[1])]
    for phase_len, (dr, dc) in zip((p1, p2, p3), steps):
        for _ in range(phase_len):
            pr, pc = route[-1]
            route.append((pr + dr, pc + dc))
    assert len(route) == seg_len
    return route


def _collect_moves(g: np.ndarray) -> tuple[int, list[dict]]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

    moves = []
    for bar_col in colors:
        for comp in _components(g, bar_col, conn8=False):
            if len(comp) < 4:
                continue
            bar = _analyze_l_bar(comp, h, w)
            if bar is None:
                continue

            fwd = bar["fwd"]
            anchors = {
                "elbow": bar["corner"],
                "tip": bar["tip"],
            }
            expected_eps = {
                name: (anchor[0] + fwd[0], anchor[1] + fwd[1]) for name, anchor in anchors.items()
            }

            candidate_seg_colors = {
                int(g[r, c])
                for p in expected_eps.values()
                for r, c in [p]
                if 0 <= r < h and 0 <= c < w and int(g[r, c]) not in (bg, bar_col)
            }
            best_seg = None
            for seg_col in sorted(candidate_seg_colors):
                for seg_comp in _components(g, seg_col, conn8=True):
                    seg_set = set(seg_comp)
                    if not all(p in seg_set for p in expected_eps.values()):
                        continue
                    deg = _segment_degrees(seg_comp)
                    endpoints = sorted([p for p, d in deg.items() if d == 1])
                    if len(endpoints) != 2 or sorted(expected_eps.values()) != endpoints:
                        continue
                    path = _segment_path_from_endpoint(seg_comp, endpoints[0])
                    if path[-1] != endpoints[1]:
                        continue
                    key = (len(path), len(seg_comp), seg_col)
                    if best_seg is None or key > best_seg[0]:
                        best_seg = (key, seg_col, seg_comp, endpoints, path)
            assert best_seg is not None, f"no valid U-segment for L bar at {bar['corner']}"
            (_, _, _), seg_col, seg_comp, endpoints, path = best_seg

            obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
            assert obstacle_candidates, "must have at least one obstacle color distinct from bar and segment"
            obstacle_set = None
            best_obs_key = None
            for obs_col in obstacle_candidates:
                for obs_comp in _components(g, obs_col, conn8=True):
                    d = min(abs(r - ep[0]) + abs(c - ep[1]) for ep in endpoints for r, c in obs_comp)
                    key = (d, min(r for r, _ in obs_comp), min(c for _, c in obs_comp))
                    if best_obs_key is None or key < best_obs_key:
                        best_obs_key = key
                        obstacle_set = set(obs_comp)
            assert obstacle_set is not None, "must find obstacle for L bar triple"

            def md_to_obstacle(p: tuple[int, int]) -> int:
                return min(abs(r - p[0]) + abs(c - p[1]) for r, c in obstacle_set)

            d_elbow = md_to_obstacle(expected_eps["elbow"])
            d_tip = md_to_obstacle(expected_eps["tip"])
            assert d_elbow != d_tip, "the two attached endpoints must not tie in obstacle distance"
            keep_name = "elbow" if d_elbow < d_tip else "tip"
            keep_anchor = anchors[keep_name]
            keep_ep = expected_eps[keep_name]
            move_ep = expected_eps["tip" if keep_name == "elbow" else "elbow"]
            off_r = keep_ep[0] - keep_anchor[0]
            off_c = keep_ep[1] - keep_anchor[1]
            assert (off_r, off_c) == fwd, "kept endpoint must be one forward step from its bar anchor"
            seg_len = len(path)
            side_vec = (
                (bar["tip"][0] - bar["corner"][0]) // bar["arm_len"],
                (bar["tip"][1] - bar["corner"][1]) // bar["arm_len"],
            )
            new_pts = _canonical_route(keep_anchor, keep_name, seg_len, fwd, side_vec)
            step_classes = _classify_wrap_steps(new_pts, fwd)
            assert all(0 <= r < h and 0 <= c < w for r, c in new_pts), "rerouted segment must stay in bounds"
            assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"
            assert not any(p in bar["bar_set"] for p in new_pts), "output segment must not overlap the bar"
            s = "".join(step_classes)
            assert re.fullmatch(r"(FL)*(F)*(FR)*", s) or re.fullmatch(r"(FR)*(F)*(FL)*", s), (
                f"output segment not monotone wrap: {step_classes}"
            )
            assert any(
                any(q in obstacle_set for q in neighbors4(r, c))
                for r, c in new_pts
            ), "rerouted segment must actually touch obstacle skin"

            moves.append(
                {
                    "seg_comp": seg_comp,
                    "seg_col": seg_col,
                    "new_pts": new_pts,
                    "bar_col": bar_col,
                    "bar": bar,
                    "keep_name": keep_name,
                    "keep_ep": keep_ep,
                    "move_ep": move_ep,
                    "dist_gap": abs(d_elbow - d_tip),
                }
            )
    return bg, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route one side of each U-shaped bridge after identifying which attachment point of an L-shaped border bar lies closer to the nearby obstacle.

    Input structure:
    - Background plus 1-3 triples. Each triple contains an L-shaped bar touching exactly one grid edge, a 1-cell-thick U-shaped segment whose two endpoints attach to the bar's inner elbow-side and short-arm-side attachment cells, and a nearby obstacle sprite of a third color.

    Rule:
    - For each L bar, recover its inward direction, elbow, and short-arm tip, then find the U segment whose two endpoints sit one forward step away from those two anchors.
    - Compare the two attached endpoints against the nearest obstacle: the closer endpoint stays attached, while the farther endpoint detaches.
    - Clear the whole U segment and rebuild a same-length path from the kept endpoint toward the obstacle, along obstacle skin, and out on the far side, preserving the obstacle and the L bar.

    Key insight:
    - The crucial inference chain is fused: the bar shape now encodes two distinct legal anchors, and the U segment makes both candidates explicit. The solver must first decode the L geometry, then use obstacle proximity to choose which endpoint remains fixed before applying the original wrap-around-obstacle motion.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, moves = _collect_moves(g)

    for move in moves:
        for r, c in move["seg_comp"]:
            out[r, c] = bg
    for move in moves:
        for r, c in move["new_pts"]:
            if int(out[r, c]) == bg:
                out[r, c] = move["seg_col"]
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
TRUNK_LEN_MIN, TRUNK_LEN_MAX = 3, 7
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
ARM_LEN_BY_PARITY = {1: 2, 0: 3}
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
MAX_RETRIES = 500
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, 1)},
    "left": {"fwd": (0, 1), "side": (1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _corner_for(edge: str, n: int, trunk_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (trunk_len - 1, pos)
    if edge == "bottom":
        return (n - trunk_len, pos)
    if edge == "left":
        return (pos, trunk_len - 1)
    return (pos, n - trunk_len)


def _transform(
    locals_: list[tuple[int, int]], edge: str, n: int, trunk_len: int, pos: int, arm_sign: int
) -> list[tuple[int, int]]:
    cr, cc = _corner_for(edge, n, trunk_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr0, sc0 = EDGE_DATA[edge]["side"]
    sr, sc = sr0 * arm_sign, sc0 * arm_sign
    return [(cr + lr * fr + lc * sr, cc + lr * fc + lc * sc) for lr, lc in locals_]


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


def _build_input_u_path(seg_len: int, arm_len: int) -> list[tuple[int, int]]:
    depth = (seg_len - arm_len + 1) // 2
    coords = [(r, 0) for r in range(1, depth + 1)]
    coords.extend((depth, c) for c in range(1, arm_len + 1))
    coords.extend((r, arm_len) for r in range(depth - 1, 0, -1))
    return coords


def _build_output_path(seg_len: int, anchor_b: int, root_sign: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(1, anchor_b)]
    phase_steps = [(1, root_sign), (1, 0), (1, -root_sign)]
    for phase_len, (dr, dc) in zip([p1, p2, p3], phase_steps):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
    assert len(coords) == seg_len
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


def _expand_with_neighbors(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(coords)
    for r, c in list(coords):
        for nr, nc in neighbors4(r, c):
            out.add((nr, nc))
    return out


def _make_triple(n: int) -> dict | None:
    seg_len = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
    arm_len = ARM_LEN_BY_PARITY[seg_len % 2]
    p1, p2, p3 = _canonical_phase_split(seg_len)

    trunk_max = min(TRUNK_LEN_MAX, n - 2)
    if trunk_max < TRUNK_LEN_MIN:
        return None
    trunk_len = random.randint(TRUNK_LEN_MIN, trunk_max)
    arm_sign = random.choice((-1, 1))
    fixed_kind = random.choice(("elbow", "tip"))
    anchor_b = 0 if fixed_kind == "elbow" else arm_len
    root_sign = -1 if fixed_kind == "elbow" else 1

    bar = [(-k, 0) for k in range(trunk_len)] + [(0, c) for c in range(1, arm_len + 1)]
    input_seg = _build_input_u_path(seg_len, arm_len)
    output_seg = _build_output_path(seg_len, anchor_b, root_sign, p1, p2, p3)

    bar_set = set(bar)
    input_set = set(input_seg)
    output_set = set(output_seg)
    if len(input_set) != seg_len or len(output_set) != seg_len:
        return None

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - root_sign)
    if apex_inner in bar_set or apex_inner in input_set or apex_inner in output_set:
        return None
    forbidden = bar_set | output_set | _expand_with_neighbors(input_set)
    obstacle = _grow_obstacle(apex_inner, random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX), forbidden)
    if _is_4adj(input_seg, obstacle):
        return None

    fixed_ep = (1, anchor_b)
    moving_ep = (1, arm_len if fixed_kind == "elbow" else 0)
    d_fixed = min(abs(r - fixed_ep[0]) + abs(c - fixed_ep[1]) for r, c in obstacle)
    d_moving = min(abs(r - moving_ep[0]) + abs(c - moving_ep[1]) for r, c in obstacle)
    if not (d_fixed < d_moving):
        return None

    fixed_anchor = (0, anchor_b)
    nearest_md = min(abs(r - fixed_anchor[0]) + abs(c - fixed_anchor[1]) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    deg = _segment_degrees(input_seg)
    if sorted(deg.values()) != [1, 1] + [2] * (seg_len - 2):
        return None

    union = list(bar_set | input_set | output_set | set(obstacle))
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "trunk_len": trunk_len,
        "arm_len": arm_len,
        "arm_sign": arm_sign,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "fixed_ep": fixed_ep,
        "moving_ep": moving_ep,
        "fixed_kind": fixed_kind,
        "dist_gap": d_moving - d_fixed,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_seg"]
    trunk_len = triple["trunk_len"]
    arm_sign = triple["arm_sign"]

    for pos in positions:
        union_g = _transform(union, edge, n, trunk_len, pos, arm_sign)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        placed = {
            "bar": _transform(triple["bar"], edge, n, trunk_len, pos, arm_sign),
            "obstacle": _transform(triple["obstacle"], edge, n, trunk_len, pos, arm_sign),
            "input_seg": _transform(triple["input_seg"], edge, n, trunk_len, pos, arm_sign),
            "output_seg": _transform(triple["output_seg"], edge, n, trunk_len, pos, arm_sign),
            "fixed_ep": _transform([triple["fixed_ep"]], edge, n, trunk_len, pos, arm_sign)[0],
            "moving_ep": _transform([triple["moving_ep"]], edge, n, trunk_len, pos, arm_sign)[0],
            "fixed_kind": triple["fixed_kind"],
            "dist_gap": triple["dist_gap"],
            "reserve": (r0, c0, r1, c1),
        }
        return placed
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
            own = min(
                abs(r - placed["fixed_ep"][0]) + abs(c - placed["fixed_ep"][1])
                for r, c in placed["obstacle"]
            )
            other = min(
                abs(r - ep[0]) + abs(c - ep[1])
                for j, other_placed in enumerate(placed_triples)
                if j != i
                for ep in (placed["fixed_ep"], placed["moving_ep"])
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
