# Combo variant of 88bcf3b4 (index 12)
# combo_spec: D9/0+D15/1
# Name: 对角入正交出不对称路径
#
# Design analysis
# a) 具体化与行为追踪
#    - 保留原题三元组：贴边直 bar、与其内侧 terminal 相连的细 segment、terminal 前方附近的 obstacle。
#    - 融合后唯一的连接根像素不再是 terminal 的 4 邻格 `(0, side)`，而是 terminal 的对角 8 邻格 `(1, side)`；
#      也就是 segment 从 bar 斜着“挂出”。
#    - 输出主路径改成 A/F/O：根像素后先继续做若干步对角前进 `A=(1,side)`，再沿 bar 前进方向直走 `F=(1,0)`，
#      最后不再对角离开，而是做纯横/纯竖的正交尾巴 `O=(0,side)`。
#    - 例：top-edge bar、segment 在右侧时，局部坐标从原题常见的 `(0,1)->(1,2)*->(2,2)*->(3,1)*`
#      变成 `(1,1)->(2,2)*->(3,2)*->(3,3)*`。入口由“terminal 旁边贴出”变成“从 terminal 右下角斜出”，
#      出口由对角转成水平/垂直，因此整体从近似 V 形变成明显不对称的折线。
# b) 复杂度校准
#    - 不删除原题的 bar 检测、segment 主路径提取、最近 obstacle 匹配、多 triple 放置。
#    - 为了让融合效果肉眼可见，generator 强制三相位都存在：对角接入已由根像素保证，另外 A/F/O 三段长度都至少 1。
#    - obstacle 沿 F 相位内侧先铺一条基础脊柱，再受限生长少量附肢；这样样本不会退化成“几乎没有沿障碍行进”的短路情况。
# c) 隐式假设审查
#    - 原 solver 假设 segment 根像素在 terminal 的 4 邻域；融合后必须改为只看 terminal 的对角 8 邻域。
#    - 原 D9 solver 直接把 `off` 当成 lateral 方向使用；融合后 `off = fwd + side_vec`，必须显式拆出单位侧向 `side_vec`，
#      否则正交退出会被误判成继续沿对角方向延伸。
#    - 原题 / D9 的 `pick_p_near`、shifted route、障碍物皮肤 entry 都依赖“根偏移是固定向量”；这个前提仍成立，
#      只是偏移从正交单位向量变成对角向量。
#    - 原 D9 的单调性校验 A*F*O* 仍可用，但对 A 的定义要改成 `fwd + side_vec` 的对角步，而非“任意含前进分量的斜步”。
# d) 放置合法性
#    - generator 采用 output-first：先在 local 坐标里造 bar、输出段、obstacle，再反推出输入段，并用 `main(inp)` 做 oracle。
#    - 根像素固定在 `(1, side)`，保证它与 terminal 仅角接；其余 input/output 像素都被约束为不再触碰 bar。
#    - `_place_triple` 仍用 input+output+obstacle 的并集 bbox 加 padding 预留，避免多 triple 干扰。
# e) 视觉可观察性
#    - 每个样本都会同时出现两个可见证据：segment 从 terminal 的角上斜着接出；输出末尾是纯水平/纯竖的尾巴。
#    - 三相位都为正长度，所以 2-4 个样例就能看出“斜入 + 贴边直走 + 正交甩尾”的不对称路径。
# f) 可视化整洁度
#    - obstacle 只在输出段内侧生长，不跨到 O 相位外侧；bar 与 obstacle 保持缓冲，避免三种对象在一小块区域缠成一团。
#    - 输入段使用单调向外的斜/横组合，除了根像素之外不会回贴 bar，也不会制造 2x2 粗块。
# g) 变化的可推断性
#    - 颜色角色不变：bar / segment / obstacle 三色，背景仍是最高频色。融合规则完全由几何关系决定，不依赖颜色值。
# h) 新 solver 假设
#    - 对每根 bar，terminal 的对角邻域中只有一个真正的 segment 端点候选；generator 通过 obstacle 距离与留白保证这一点。
#    - 输出期望满足 A*F*O* 单调相位，且末段至少有一个 O 步；generator 明确按此构造并用 oracle 验证。

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


def _diag_neighbors_of_terminal(p: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = p
    out = []
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-route each irregular segment that is diagonally hung from a bar terminal so it wraps a nearby obstacle and leaves with an orthogonal tail.

    Input structure:
    - Background plus three functional colors per triple: a straight bar perpendicular to a border, a 1-cell-thick irregular segment, and a nearby obstacle sprite.
    - Exactly one segment endpoint is attached to the bar's interior terminal by diagonal 8-neighbor contact rather than side-by-side 4-neighbor contact.

    Rule:
    - Keep the bar and obstacle fixed.
    - Detach the segment from its original irregular shape and rebuild it from the same diagonal root: first approach the nearest obstacle diagonally in the bar-forward direction, then trace reachable obstacle skin cells, then extend with a pure orthogonal tail parallel to the segment's side offset.
    - Preserve the segment pixel count and never paint through the obstacle.

    Key insight:
    - The root offset now encodes two directions at once: one forward step away from the border and one lateral step to the segment side. Use that diagonal offset to find the correct rope source, but once the rope exits the obstacle skin, ignore the diagonal and continue only along the lateral orthogonal direction.
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
                    int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in _diag_neighbors_of_terminal((tr, tc), h, w)
                ):
                    continue
                yield col, comp, terminal_ep

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        neigh = []
        for nr, nc in _diag_neighbors_of_terminal(terminal, h, w):
            v = int(g[nr, nc])
            if v != bg and v != bar_col:
                neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a diagonal non-bg/non-bar neighbor"

        best_seg = None
        fwd = _bar_forward_dir(bar_comp)
        for cand_col, cand_start in neigh:
            off_r = cand_start[0] - terminal[0]
            off_c = cand_start[1] - terminal[1]
            side = (off_r - fwd[0], off_c - fwd[1])
            if max(abs(off_r), abs(off_c)) != 1 or abs(off_r) + abs(off_c) != 2:
                continue
            if abs(side[0]) + abs(side[1]) != 1:
                continue
            if side[0] * fwd[0] + side[1] * fwd[1] != 0:
                continue
            if (fwd[0] + side[0], fwd[1] + side[1]) != (off_r, off_c):
                continue
            for comp in _components(g, cand_col, conn8=True):
                if cand_start not in comp:
                    continue
                if len(comp) < 2:
                    break
                deg = _segment_degrees(comp)
                if deg.get(cand_start, 0) != 1:
                    break
                path = _segment_main_path(comp, cand_start)
                l_seg = len(path)
                key = (l_seg, len(comp), cand_col, cand_start[0], cand_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, side, cand_col, cand_start, comp)
                break
        assert best_seg is not None, f"no valid diagonally attached segment for bar at {terminal}"
        (seg_len, _, _, _, _), side, seg_col, seg_start, seg_comp = best_seg
        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]

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

        def exit_key(p: tuple[int, int]) -> tuple[int, int, int, int]:
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
        _, exit_cell, boundary_path = max(reachable, key=lambda item: item[0])
        skin_walkable[exit_cell] = True

        route = list(shifted)
        if boundary_path and boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        dr_ext, dc_ext = side
        cur = route[-1]
        while len(route) < seg_len:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != bg and int(g[nr, nc]) != seg_col:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        new_pts = route[:seg_len]
        assert len(new_pts) == seg_len, "output segment must preserve length"
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

        step_classes = []
        for j in range(len(new_pts) - 1):
            dr = new_pts[j + 1][0] - new_pts[j][0]
            dc = new_pts[j + 1][1] - new_pts[j][1]
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
        assert re.fullmatch(r"A*F*O*", "".join(step_classes or [""])), f"output segment not monotone A/F/O: {step_classes}"
        assert "O" in step_classes, "output segment must have an orthogonal exit tail"
        assert (seg_start[0] - terminal[0], seg_start[1] - terminal[1]) == (fwd[0] + side[0], fwd[1] + side[1]), (
            "segment root must be diagonally forward from terminal"
        )

        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 9, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 5
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 3, 7
INPUT_TURN_MIN, INPUT_TURN_MAX = 2, 5
RUN_CONTINUE_PROB = 0.35
MAX_RETRIES = 500
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
    base: list[tuple[int, int]], max_size: int, forbidden: set[tuple[int, int]], side: int
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
                if (nc - anchor_c) * side > 0:
                    continue
                if abs(nc - anchor_c) > 2:
                    continue
                cands.append((nr, nc))
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _build_input_path(seg_len: int, side: int, turns: int) -> list[tuple[int, int]]:
    """
    Build an 8-connected stroke whose only bar-touching cell is the diagonal root `(1, side)`.
    Every later step keeps moving outward in the side direction, preventing accidental extra bar contacts.
    """
    steps = [(-1, side), (0, side), (1, side)]
    coords = [(1, side)]
    remaining = seg_len - 1
    last_kind = None
    turns_left = turns
    while remaining > 0:
        if last_kind is None:
            kind = random.randrange(len(steps))
        else:
            if turns_left <= 0 or random.random() < RUN_CONTINUE_PROB:
                kind = last_kind
            else:
                options = [k for k in range(len(steps)) if k != last_kind]
                kind = random.choice(options)
                turns_left -= 1
        max_run = remaining if turns_left <= 0 else max(1, remaining - turns_left)
        run = random.randint(1, max_run)
        dr, dc = steps[kind]
        for _ in range(run):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
        remaining -= run
        last_kind = kind
    return coords


def _build_output_path(seg_len: int, side: int, p1: int, p2: int, p3: int) -> list[tuple[int, int]]:
    coords = [(1, side)]
    phase_steps = [(1, side), (1, 0), (0, side)]
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


def _bar_touch_count(pt: tuple[int, int], bar_set: set[tuple[int, int]]) -> int:
    count = 0
    for nr, nc in neighbors4(pt[0], pt[1]):
        if (nr, nc) in bar_set:
            count += 1
    return count


def _bar_diag_touch_count(pt: tuple[int, int], bar_set: set[tuple[int, int]]) -> int:
    count = 0
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        if (pt[0] + dr, pt[1] + dc) in bar_set:
            count += 1
    return count


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side = random.choice((-1, 1))
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

    output_seg = _build_output_path(seg_len, side, p1, p2, p3)
    seg_set = set(output_seg)

    root = output_seg[0]
    if root != (1, side):
        return None
    if _bar_diag_touch_count(root, bar_set) != 1:
        return None
    if _bar_touch_count(root, bar_set) != 0:
        return None
    if any(_bar_touch_count(p, bar_set) > 0 or _bar_diag_touch_count(p, bar_set) > 0 for p in output_seg[1:]):
        return None

    apex = output_seg[p1]
    obstacle_base = [(apex[0] + k, apex[1] - side) for k in range(p2 + 1)]
    if any(p in seg_set or p in bar_set for p in obstacle_base):
        return None

    obs_size_min = max(OBS_SIZE_MIN, len(obstacle_base))
    if obs_size_min > OBS_SIZE_MAX:
        return None
    obs_size = random.randint(obs_size_min, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(obstacle_base, obs_size, seg_set | bar_set, side)
    if _is_4adj(obstacle, list(bar_set)):
        return None
    if any(_bar_diag_touch_count(p, bar_set) > 0 for p in obstacle):
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    max_turns = min(INPUT_TURN_MAX, seg_len - 2)
    if max_turns < INPUT_TURN_MIN:
        return None
    turns = random.randint(INPUT_TURN_MIN, max_turns)
    input_seg = _build_input_path(seg_len, side, turns)
    if _is_4adj(input_seg, obstacle):
        return None
    if _bar_diag_touch_count(input_seg[0], bar_set) != 1 or _bar_touch_count(input_seg[0], bar_set) != 0:
        return None
    if any(_bar_touch_count(p, bar_set) > 0 or _bar_diag_touch_count(p, bar_set) > 0 for p in input_seg[1:]):
        return None
    if len(set(input_seg)) != len(input_seg):
        return None

    union = bar + obstacle + input_seg + output_seg
    rs = [r for r, _ in union]
    cs = [c for _, c in union]

    return {
        "bar_len": bar_len,
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
    if n < 12:
        max_triples = 1
    elif n < 17:
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

        tr, tc = placed["terminal"]
        root = placed["input_seg"][0]
        if max(abs(root[0] - tr), abs(root[1] - tc)) != 1 or abs(root[0] - tr) + abs(root[1] - tc) != 2:
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
