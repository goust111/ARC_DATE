# Variant of 88bcf3b4
# Dimension: D4 — 源对象结构 (src)
# Change: 源对象是 8-连通的不规则线段（5-9 像素），一端通过 4-邻接连接到 bar 末端，沿远离边界方向延伸，包含 2-5... → 源为双端线段
#
# Design analysis for this D4 variant
# -----------------------------------
# spec_variant:
#   kept_from_original:
#     - 网格仍是 7-27 的方阵，1-3 组 bar+segment+obstacle。
#     - bar 仍是贴边的直线，长度 3-7，输出仍然按“靠近障碍 -> 沿障碍皮肤 -> 对角退出”重定位。
#     - obstacle 仍是不规则 4-连通小精灵，保持不动。
#     - segment 总像素数仍保持 5-9。
#   changed_core:
#     - 输入中的 segment 不再是单端挂在 terminal 旁边，而是一个 8-连通、1-cell-thick 的简单 U 形路径。
#     - U 形路径的两个端点都与同一个 bar terminal 4-邻接，分别位于 terminal 两侧，因此人眼可直接看到“双端都接 bar”。
#     - 输出时仅有更远离 obstacle 的那一端脱离并重新绕行；更靠近 obstacle 的端点保持与 terminal 相连。
#   concrete_behavior_trace:
#     - 原题：terminal 旁只有一个 segment 端点，solver 直接把它视为整条 rope 的根。
#     - 本变体：terminal 左右各有一个端点，同属一个简单路径组件。solver 先找到最近 obstacle，再比较两个端点到该 obstacle 的曼哈顿距离。
#     - 较近端点 = 保留端；较远端点 = 移动端。输出路径从保留端出发，沿原题绕障规则生成一条同长度的新路径。
#     - 例：top-edge bar 的 terminal 在 (4,5)，segment 端点在 (4,4)/(4,6)，obstacle 靠右，则右端保留，左端脱离；输出看起来像“右臂固定，左臂甩到障碍另一侧”。
#   complexity_calibration:
#     - 端点只有两个，且都显式贴着 terminal；这让“哪一端移动”可由 obstacle 相对位置决定，不引入多分支歧义。
#     - 不允许 tie：generator 保证两个端点到 obstacle 的距离严格不同，否则人类无法从 2-3 个样例推出选择规则。
#     - U 形模板只用少量长度档位（5-9），结构重复度高，视觉上足够稳定，不会退化成任意曲线猜谜。
#   implicit_assumptions_audit:
#     - 原 solver 的 `deg[cand_start] == 1` 假设“terminal 邻接像素就是唯一根端点”；本变体不成立，必须改成“组件有两个端点，且这两个端点都在 terminal 邻域里”。
#     - 原 solver 用 `seg_start - terminal` 作为唯一 lateral offset；本变体仍可保留，但 `seg_start` 必须先由 obstacle 决定为“保留端”。
#     - 原 generator 默认输入线段不接近 obstacle；本变体仍保留这一点，否则“哪一端更近”会因为整条 U 太贴近 obstacle 而不明显。
#     - 原 solver 的 obstacle 选择按“离 terminal 最近”进行；generator 保证每组 triple 的自有 obstacle 对该 terminal 严格最近，因此无需改动匹配策略。
#   placement_legality:
#     - placement 仍按 local->global 变换，并用 input/output/obstacle/bar 的并集 bbox 做 reserved 区域，避免跨 triple 混淆。
#     - U 形模板两端固定在 terminal 两侧，forward 深度最多 4；输出路径最大深度仍由 seg_len 控制，故沿用原 bar_len / grid_size 约束即可。
#   visual_observability:
#     - 变体的核心可见特征是：输入里同色线段两端都贴 bar，输出里只有靠 obstacle 的那一侧还贴 bar，另一端被甩开。
#     - 左右两种“移动端”都会生成，解题者能从少量例子归纳“远的一端移动”而非固定左/右。
#   neatness:
#     - segment 仍然是单像素细线；obstacle 与 segment 在输入中不接触；triple 间保持 pad，避免局部堆叠难辨。
#   color_reasoning:
#     - 颜色语义不变：bar / segment / obstacle 三色结构性分工明确，不依赖颜色数值做任何排序。
#   new_main_assumptions:
#     - segment 组件是简单路径（全部度 <= 2，且恰有两个度 1 端点）。
#     - 两个端点都在同一 terminal 的 4 邻域里，并且各自距 obstacle 不等。
#     - 保留端与 terminal 的偏移仍是单位 4-邻接偏移，generator 显式保证这一点。

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: the source segment is U-shaped with both endpoints attached to the bar terminal's sides.
    The endpoint closer to the nearest obstacle stays attached; the farther endpoint detaches and is rerouted
    around the obstacle using the original wrap logic.
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
        terminal_neigh = []
        terminal_neigh_set = set()
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    terminal_neigh.append((v, (nr, nc)))
                    terminal_neigh_set.add((nr, nc))
        assert terminal_neigh, f"bar terminal {terminal} must have non-bg/non-bar neighbors"

        best_seg = None
        for cand_col, _ in terminal_neigh:
            for comp in _components(g, cand_col, conn8=True):
                attached = [p for p in terminal_neigh_set if p in comp]
                if len(attached) < 2:
                    continue
                deg = _segment_degrees(comp)
                endpoints = sorted([p for p, d in deg.items() if d == 1])
                if len(endpoints) != 2:
                    continue
                attached_eps = sorted([p for p in endpoints if p in terminal_neigh_set])
                if len(attached_eps) != 2:
                    continue
                path = _segment_path_from_endpoint(comp, attached_eps[0])
                if path[-1] != attached_eps[1]:
                    continue
                key = (len(path), len(comp), cand_col, attached_eps[0], attached_eps[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, attached_eps, comp, path)
        assert best_seg is not None, f"no valid U-segment for bar at {terminal}"
        (L, _, _, _, _), seg_col, attached_eps, seg_comp, _ = best_seg

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

        def md_to_obstacle(p: tuple[int, int]) -> int:
            return min(abs(r - p[0]) + abs(c - p[1]) for r, c in obstacle_set)

        d0 = md_to_obstacle(attached_eps[0])
        d1 = md_to_obstacle(attached_eps[1])
        assert d0 != d1, "the two attached endpoints must not tie in obstacle distance"
        seg_start = attached_eps[0] if d0 < d1 else attached_eps[1]
        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]
        assert abs(off_r) + abs(off_c) == 1, "kept endpoint must be 4-adjacent to the terminal"

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
        assert shifted and shifted[0] == seg_start, "shifted path must start at the kept endpoint"
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted path must stay in bounds"
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
        cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
        away = (cr - terminal[0], cc - terminal[1])

        def skin_ok(p: tuple[int, int]) -> bool:
            r, c = p
            return int(g[r, c]) == bg or int(g[r, c]) == seg_col

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]

        def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
            r, c = p
            md = abs(r - terminal[0]) + abs(c - terminal[1])
            dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
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
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

        rs_bar = [r for r, _ in bar_comp]
        cs_bar = [c for _, c in bar_comp]
        if len(set(cs_bar)) == 1:
            fwd = (1, 0) if min(rs_bar) == 0 else (-1, 0)
        else:
            fwd = (0, 1) if min(cs_bar) == 0 else (0, -1)
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
        import re

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
            perp_dir = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
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


# --------------- generator ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
OBS_SIZE_MIN, OBS_SIZE_MAX = 1, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
MAX_RETRIES = 400
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}

U_TEMPLATES = {
    5: [(0, 1), (1, 1), (2, 0), (1, -1), (0, -1)],
    6: [(0, 1), (1, 1), (2, 1), (2, 0), (1, -1), (0, -1)],
    7: [(0, 1), (1, 1), (2, 1), (3, 0), (2, -1), (1, -1), (0, -1)],
    8: [(0, 1), (1, 1), (2, 1), (3, 1), (3, 0), (2, -1), (1, -1), (0, -1)],
    9: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 0), (3, -1), (2, -1), (1, -1), (0, -1)],
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


def _build_input_u_path(seg_len: int, side: int) -> list[tuple[int, int]]:
    base = U_TEMPLATES[seg_len]
    return [(r, c * side) for r, c in base]


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
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    fixed_side = random.choice((-1, 1))

    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)

    remaining = seg_len - 1
    p1 = random.randint(1, min(PHASE_MAX, remaining - 1))
    remaining -= p1
    p2 = random.randint(0, min(PHASE_MAX, remaining))
    p3 = remaining - p2
    output_seg = _build_output_path(seg_len, fixed_side, p1, p2, p3)
    seg_set = set(output_seg)

    apex = output_seg[p1]
    apex_inner = (apex[0], apex[1] - fixed_side)
    if apex_inner in seg_set or apex_inner in bar_set:
        return None

    obs_size = random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(apex_inner, obs_size, seg_set | bar_set)

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    input_seg = _build_input_u_path(seg_len, fixed_side)
    input_set = set(input_seg)
    if len(input_set) != seg_len:
        return None
    if _is_4adj(input_seg, obstacle):
        return None

    fixed_end = input_seg[0]
    moving_end = input_seg[-1]
    d_fixed = min(abs(r - fixed_end[0]) + abs(c - fixed_end[1]) for r, c in obstacle)
    d_moving = min(abs(r - moving_end[0]) + abs(c - moving_end[1]) for r, c in obstacle)
    if not (d_fixed < d_moving):
        return None

    deg = _segment_degrees(input_seg)
    if sorted(deg.values()) != [1, 1] + [2] * (seg_len - 2):
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
        "fixed_side": fixed_side,
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
        placed = {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
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
