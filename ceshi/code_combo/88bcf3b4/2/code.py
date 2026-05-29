# Combo variant of 88bcf3b4 (index 2)
# combo_spec: D3/1+D9/0
# Name: 虚线L形绕行
#
# Design analysis
# a) 具体化与行为追踪
#    - 基础对象仍是原题的三元组：贴边直 bar、terminal 旁边的细 segment、以及 terminal 前方的 obstacle。
#    - 原题在 solver 里先恢复完整 route：segment 贴着 obstacle 走一段，再沿末端对角方向继续拉长，最终整条 route 都上色。
#    - 本融合版把这两处一起改掉：完整 route 的几何改成 A/F/O 三相位，
#      也就是“斜向靠近 + 贴边前进 + 沿根侧正交退出”的 L 形；着色层则只保留 route 的第 1、3、5... 格。
#    - 具体小例子：若某个 top-edge bar 的根侧在 terminal 右边，原题可能输出
#      (0,1)->(1,2)->(2,2)->(3,1)->(4,0) 这种 V 形实线；
#      现在完整 route 改成 (0,1)->(1,2)->(2,2)->(2,3)->(2,4)，可见输出只涂第 0/2/4 格，得到一个虚线 L。
#    - 差异因此不是“先 L 再虚线”两件孤立的事，而是 solver 必须先恢复完整 L 形 corridor，之后才能在正确的几何顺序上做隔格采样。
#
# b) 复杂度校准
#    - 保留原题 1-3 个 triple、四个边缘朝向、随机 obstacle 形状与输入 irregular stroke，避免简化成单一模板。
#    - 同时限制新自由度：输出仍固定为单一的 A/F/O 单调相位，虚线规则固定为从 attachment 开始保留奇数位。
#    - generator 强制三相位都为正长度，并用完整 route 做 reserve，占住虚线空档，避免别的 triple 插进来造成歧义。
#
# c) 隐式假设审查
#    - 原 solver 依赖 boundary_path 的最后一步来决定“继续对角退出”；融合后这个前提失效，必须改成沿根侧 side 方向正交延伸。
#    - 原 solver 的单调性断言是 FL*F*FR* / FR*F*FL*；融合后应改成 A*F*O*，否则会把合法 L 形错判掉。
#    - 原 solver 默认 repaint 整条 new_pts；融合后若不显式抽样，就会退化回 D9 单变体而不是组合。
#    - 原题 generator 把 obstacle 从 V apex 内侧单点长出；融合后若继续这样，solver 往往能很快离开 obstacle，
#      使 F/O 相位不明显，所以生成时改为沿 F 相位内侧先铺一条基础脊柱。
#
# d) 放置合法性
#    - 继续采用 output-first：先造合法完整 output_route，再从中抽出 dashed output_seg，最后反推输入 irregular segment。
#    - `_place_triple` 与 reserve 仍基于 bar + obstacle + input_seg + output_route，而不是可见虚线路径，
#      从而保证 triple 间不重叠、不越界，也不会共享虚线空格。
#
# e) 视觉可观察性
#    - 人类从样例里应能同时看到两件事：segment 不再对角离开 obstacle，而是拐成横/竖尾巴；并且这条尾巴与前段一起都呈隔格断续。
#    - 三相位都至少 1 步，因此不会出现“只有虚线、看不出 L”或“只有 L、看不出虚线”的退化样本。
#
# f) 可视化整洁度
#    - obstacle 被约束在 output_route 内侧，且 reserve 覆盖完整 corridor，减少不同对象在同一区域交叠。
#    - 输出只使用 bar/segment/obstacle 三类颜色，不叠加额外标记层，虚线空位保持背景，结构清楚。
#
# g) 变化的可推断性
#    - 颜色角色完全沿用原题：segment 仍用输入 segment 的颜色，bar 与 obstacle 保持不变。
#    - 规则不依赖颜色编号或隐藏排序，只依赖可见几何顺序与隔格采样。
#
# h) 融合行为量化
#    - 每个 triple 的完整 route 长度记为 L，可见段长度必须严格等于 ceil(L/2)。
#    - route 的步型必须含有至少一个 O 步，说明“正交退出”真实发生；同时可见像素之间至少出现一个曼哈顿距离 2 的同色断点，说明“虚线”真实发生。
#
# i) 新 solver 假设与 generator 保证
#    - solver 假设 segment attachment 与 bar forward 正交；generator 总把输入 segment 的起点放在 terminal 根侧一格。
#    - solver 假设最近 obstacle 唯一且其 skin 上存在可达前向 exit；generator 通过最近距离筛选和 obstacle_base 构造保证这一点。
#    - solver 假设 route 的完整 corridor 不被别的 triple 占用；generator 用 output_route 做 reserve 来满足。

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


def _bar_forward_dir(comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


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


def _bfs_path4(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
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


def _dashed_points(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [path[i] for i in range(0, len(path), 2)]


def _classify_l_steps(route: list[tuple[int, int]], fwd: tuple[int, int], side: tuple[int, int]) -> list[str]:
    step_classes = []
    for j in range(len(route) - 1):
        dr = route[j + 1][0] - route[j][0]
        dc = route[j + 1][1] - route[j][1]
        fwd_comp = dr * fwd[0] + dc * fwd[1]
        side_comp = dr * side[0] + dc * side[1]
        man = abs(dr) + abs(dc)
        if fwd_comp == 1 and side_comp == 1 and man == 2:
            step_classes.append("A")
        elif fwd_comp == 1 and side_comp == 0 and man == 1:
            step_classes.append("F")
        elif fwd_comp == 0 and side_comp == 1 and man == 1:
            step_classes.append("O")
        else:
            step_classes.append("X")
    return step_classes


def _has_visible_orth_exit(
    route: list[tuple[int, int]], step_classes: list[str], dashed: list[tuple[int, int]] | None = None
) -> bool:
    dashed_set = set(_dashed_points(route) if dashed is None else dashed)
    for idx in range(1, len(route)):
        if step_classes[idx - 1] == "O" and route[idx] in dashed_set:
            return True
    return False


def _collect_moves(g: np.ndarray) -> tuple[int, list[dict]]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})

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
                seg_len = len(path)
                key = (seg_len, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid segment for bar at {terminal}"
        (seg_len, _, _, _, _), seg_col, seg_start, seg_comp = best_seg

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

        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]
        side = (off_r, off_c)
        fwd = _bar_forward_dir(bar_comp)
        assert abs(off_r) + abs(off_c) == 1, "segment attachment must be orthogonally adjacent to terminal"
        assert off_r * fwd[0] + off_c * fwd[1] == 0, "segment attachment must be orthogonal to bar forward direction"

        walkable = (g == bg) | (g == seg_col)
        walkable[terminal] = True

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
                d = abs(pr - terminal[0]) + abs(pc - terminal[1])
                good.append((d, pr, pc))
            assert good, "no obstacle pixel has a valid translated endpoint outside obstacle"
            _, pr, pc = min(good)
            return (pr, pc)

        p_near = pick_p_near()
        walkable[p_near] = True
        path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
        assert path_t_to_obs is not None, f"no path from terminal {terminal} to obstacle {p_near}"
        shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted path must stay within grid"
        assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        def skin_ok(p: tuple[int, int]) -> bool:
            r, c = p
            return int(g[r, c]) == bg or int(g[r, c]) == seg_col

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]
        skin_walkable = np.zeros((h, w), dtype=bool)
        for r, c in skin:
            if skin_ok((r, c)):
                skin_walkable[r, c] = True
        skin_walkable[entry] = True

        def exit_key(p: tuple[int, int]) -> tuple[int, int, int, int, int]:
            r, c = p
            forward = (r - terminal[0]) * fwd[0] + (c - terminal[1]) * fwd[1]
            lateral = (r - terminal[0]) * side[0] + (c - terminal[1]) * side[1]
            return (forward, lateral, -abs(lateral), -r, -c)

        reachable = []
        for p in skin_candidates:
            path = _bfs_path4(entry, p, skin_walkable)
            if path is not None:
                reachable.append((exit_key(p), p, path))
        assert reachable, "no reachable obstacle skin cells"
        _, _, boundary_path = max(reachable, key=lambda item: item[0])

        route = list(shifted)
        if boundary_path and boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        cur = route[-1]
        while len(route) < seg_len:
            nr, nc = cur[0] + side[0], cur[1] + side[1]
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != bg and int(g[nr, nc]) != seg_col:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        full_route = route[:seg_len]
        dashed_pts = _dashed_points(full_route)
        step_classes = _classify_l_steps(full_route, fwd, side)
        assert not any(p in obstacle_set for p in full_route), "output segment passes through obstacle"
        assert re.fullmatch(r"A*F*O*", "".join(step_classes or [""])), f"output segment not monotone L-wrap: {step_classes}"
        assert len(dashed_pts) == (len(full_route) + 1) // 2, "visible dashed length must be ceil(route_len/2)"
        moves.append(
            {
                "seg_comp": seg_comp,
                "seg_col": seg_col,
                "route": full_route,
                "dashed": dashed_pts,
                "terminal": terminal,
                "fwd": fwd,
                "side": side,
                "step_classes": step_classes,
            }
        )
    return bg, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each terminal-attached irregular segment around the nearest obstacle with an L-shaped wrap, then paint only every other cell of that rerouted path.

    Input structure:
    - Background plus 1-3 triples: a straight bar perpendicular to a grid edge, a 1-cell-thick irregular segment touching the bar's terminal from the side, and a nearby obstacle sprite of a third color.

    Rule:
    - For each triple, identify the segment attached to the bar terminal and detach it from its original location.
    - Translate the segment root toward the nearest obstacle, trace along reachable obstacle skin cells in the bar's forward direction, then continue by extending orthogonally along the segment's root side, producing an A/F/O-style L path rather than the original diagonal exit.
    - Keep the full rerouted corridor only as an internal route; the visible output paints just route cells 1, 3, 5, ... in order, so the segment becomes a dashed L-shaped wrap with ceil(L/2) painted cells when the full route has length L.

    Key insight:
    - Solving the puzzle requires combining two cues at once: recover the hidden full L-shaped rope geometry around the obstacle, then apply a deterministic odd-index sampling rule to that route instead of painting it solid.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, moves = _collect_moves(g)

    for move in moves:
        for r, c in move["seg_comp"]:
            out[r, c] = bg

    for move in moves:
        for r, c in move["dashed"]:
            if int(out[r, c]) == bg:
                out[r, c] = move["seg_col"]
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
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


def _grow_obstacle(
    base: list[tuple[int, int]], max_size: int, forbidden: set[tuple[int, int]], side_sign: int
) -> list[tuple[int, int]]:
    obstacle = set(base)
    start_r = min(r for r, _ in base)
    end_r = max(r for r, _ in base)
    anchor_c = base[0][1]
    while len(obstacle) < max_size:
        cands = []
        for r, c in list(obstacle):
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in obstacle or (nr, nc) in forbidden:
                    continue
                if not (start_r <= nr <= end_r):
                    continue
                if (nc - anchor_c) * side_sign > 0:
                    continue
                if abs(nc - anchor_c) > 2:
                    continue
                cands.append((nr, nc))
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _build_input_path(seg_len: int, turns: int, side_sign: int) -> list[tuple[int, int]]:
    remaining = seg_len - 1
    coords = [(0, side_sign)]
    diag_step = (-1, side_sign)
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


def _build_output_route(seg_len: int, side_sign: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(0, side_sign)]
    phase_steps = [(1, side_sign), (1, 0), (0, side_sign)]
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


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side_sign = random.choice((-1, 1))
    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1_max = min(PHASE_MAX, remaining - 2)
    if p1_max < 1:
        return None
    p1 = random.randint(1, p1_max)
    remaining -= p1
    p2_max = min(PHASE_MAX, remaining - 1)
    if p2_max < 1:
        return None
    p2 = random.randint(1, p2_max)
    p3 = remaining - p2
    if p3 < 1:
        return None

    output_route = _build_output_route(seg_len, side_sign, p1, p2, p3)
    output_seg = _dashed_points(output_route)
    output_steps = _classify_l_steps(output_route, (1, 0), (0, side_sign))
    if not re.fullmatch(r"A*F*O+", "".join(output_steps)):
        return None
    if not _has_visible_orth_exit(output_route, output_steps, output_seg):
        return None
    seg_set = set(output_route)

    apex = output_route[p1]
    obstacle_base = [(apex[0] + k, apex[1] - side_sign) for k in range(p2 + 1)]
    if any(p in seg_set or p in bar_set for p in obstacle_base):
        return None

    obs_size_min = max(OBS_SIZE_MIN, len(obstacle_base))
    if obs_size_min > OBS_SIZE_MAX:
        return None
    obs_size = random.randint(obs_size_min, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(obstacle_base, obs_size, seg_set | bar_set, side_sign)

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    if max_turns < INPUT_TURN_MIN:
        return None
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, turns, side_sign)
    if _is_4adj(input_seg, obstacle):
        return None

    union = bar + obstacle + input_seg + output_route
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_route": output_route,
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
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_route"]
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
            "output_route": _transform(triple["output_route"], edge, n, bar_len, pos),
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
        pred = main(inp)
        if not np.array_equal(pred, out):
            return None
        _, moves = _collect_moves(inp)
        if len(moves) != len(placed_triples):
            return None
        if any("O" not in move["step_classes"] for move in moves):
            return None
        if any(not _has_visible_orth_exit(move["route"], move["step_classes"], move["dashed"]) for move in moves):
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
