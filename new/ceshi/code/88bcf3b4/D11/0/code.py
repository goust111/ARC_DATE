# Variant of 88bcf3b4
# Dimension: D11 — 多步处理顺序 (multi_step)
# Change: 当存在多个 triple 时，所有绕行独立并行计算，互不影响。... → 按距离边缘排序处理
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     original: 先在原图上为每个 triple 独立算 reroute，最后统一清空并重画，所以 triple 之间没有因果关系。
#     new: triple 按 bar terminal 到最近边缘的距离从近到远处理；前一步输出线段立即写回网格，
#       后一步的 walkability 必须把它视作新的阻塞物。
#   concrete_pixel_behavior:
#     - bar 越短，terminal 越靠近边缘，也就越早处理。
#     - 早处理的 rope 先落位；晚处理的 rope 仍绕自己的静态 obstacle sprite，但它的通道可能被前一步 rope 挤占，
#       所以后续路径会缩短、改道或提前停止，产生级联。
#   behavior_trace:
#     - 原逻辑：A 和 B 都只看原始输入，B 的 reroute 与 A 输出无关。
#     - 新逻辑：先算 A，再把 A 的输出留在当前网格；再算 B，B 的目标障碍仍是静态 sprite，
#       但路径规划会被 A 的输出阻挡，因此结果可与并行版不同。
#   complexity_calibration:
#     - 保留原题的 3 对象结构：bar / irregular segment / obstacle sprite。
#     - 不引入新颜色语义，不按颜色值排序；排序只依赖可见的几何深度。
#     - 生成器固定使用一个已验证的 2-triple D11 基模板，再通过旋转/镜像/重着色扩展，
#       保证级联稳定可见而不是依赖低命中率随机搜索。
#   implicit_assumptions_review:
#     - 原 main 中“先收集全部 moves 再统一 repaint”依赖 triple 独立；D11 下必须边算边写回。
#     - 原 main 中 walkable 允许整张图的 segment 颜色通行；D11 下这会错误穿过前一步 rope，
#       所以改成只允许当前 work grid 的背景通行。
#     - 原 bar/segment 识别若在当前网格上做，会把前一步输出 rope 误认成后续 triple 的邻接 segment；
#       因此 triple 识别固定基于原始输入，只把 reroute 放到顺序阶段。
#     - 静态 obstacle 目标仍从原始输入中抽取，前一步 rope 只作为阻塞物，不作为新的目标 sprite。
#   placement_legality:
#     - 基模板中的两个 triple 在输入中互不重叠，静态 obstacle 对各自 terminal 的最近关系明确。
#     - 但它们的独立输出会在同一区域竞争，正好触发顺序依赖。
#   visual_observability:
#     - 两个 bar 深度不同，处理顺序由几何位置直接可见。
#     - 同一输入在输出里会出现“后一个 rope 被先一个 rope 挡住”的效果，能肉眼观察到不是单纯并行重排。

from __future__ import annotations

import random
import re
from collections import deque

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


def _classify_steps(new_pts: list[tuple[int, int]], bar_comp: list[tuple[int, int]]) -> list[str]:
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
    return step_classes


def _find_triples(g: np.ndarray) -> tuple[list[dict], list[set[tuple[int, int]]], int]:
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    triples = []

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
        triples.append(
            {
                "bar_col": bar_col,
                "bar_comp": list(bar_comp),
                "terminal": terminal,
                "seg_col": seg_col,
                "seg_start": seg_start,
                "seg_comp": list(seg_comp),
                "off": (seg_start[0] - terminal[0], seg_start[1] - terminal[1]),
                "L": L,
                "edge_distance": _edge_distance(terminal, h, w),
            }
        )

    occ = g != bg
    for triple in triples:
        for p in triple["bar_comp"]:
            occ[p] = False
        for p in triple["seg_comp"]:
            occ[p] = False
    obstacle_components = [{(int(r), int(c)) for r, c in comp} for comp in components_mask(occ, connectivity=8)]
    assert obstacle_components, "must find at least one static obstacle component"
    return triples, obstacle_components, bg


def _route_triple(
    work: np.ndarray,
    triple: dict,
    obstacle_components: list[set[tuple[int, int]]],
    bg: int,
) -> list[tuple[int, int]]:
    h, w = work.shape
    terminal = triple["terminal"]
    seg_start = triple["seg_start"]
    off_r, off_c = triple["off"]
    L = triple["L"]

    best = None
    obstacle_set = None
    for comp in obstacle_components:
        d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
        mr = min(r for r, _ in comp)
        mc = min(c for _, c in comp)
        key = (d, mr, mc)
        if best is None or key < best:
            best = key
            obstacle_set = comp
    assert obstacle_set is not None

    walkable = work == bg
    walkable[terminal] = True

    def pick_p_near() -> tuple[int, int]:
        good = []
        for pr, pc in obstacle_set:
            er, ec = pr + off_r, pc + off_c
            if not (0 <= er < h and 0 <= ec < w):
                continue
            if (er, ec) in obstacle_set:
                continue
            if int(work[er, ec]) != bg:
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
    away = (cr - terminal[0], cc - terminal[1])

    def skin_ok(p: tuple[int, int]) -> bool:
        r, c = p
        return int(work[r, c]) == bg

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
    if boundary_path[0] == entry:
        route.extend(boundary_path[1:])

    cur = route[-1]
    while len(route) < L:
        nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(work[nr, nc]) != bg:
            break
        route.append((nr, nc))
        cur = (nr, nc)

    new_pts = route[:L]
    assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

    step_classes = _classify_steps(new_pts, triple["bar_comp"])
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
        perp_dir = (seg_start[0] - terminal[0], seg_start[1] - terminal[1])
        inner = (-perp_dir[0], -perp_dir[1])
        apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
        assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"
    return new_pts


def _solve_with_mode(grid: np.ndarray, sequential: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    triples, static_obstacles, bg = _find_triples(g)
    ordered = sorted(triples, key=lambda t: (t["edge_distance"], t["terminal"][0], t["terminal"][1], t["bar_col"]))

    if sequential:
        cur = g.copy()
        for triple in ordered:
            work = cur.copy()
            for r, c in triple["seg_comp"]:
                work[r, c] = bg
            new_pts = _route_triple(work, triple, static_obstacles, bg)
            for r, c in triple["seg_comp"]:
                cur[r, c] = bg
            for r, c in new_pts:
                if int(cur[r, c]) == bg:
                    cur[r, c] = triple["seg_col"]
        return cur

    out = g.copy()
    moves = []
    for triple in ordered:
        work = g.copy()
        for r, c in triple["seg_comp"]:
            work[r, c] = bg
        new_pts = _route_triple(work, triple, static_obstacles, bg)
        moves.append((triple["seg_comp"], triple["seg_col"], new_pts))
    for seg_comp, _seg_col, _new_pts in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _seg_comp, seg_col, new_pts in moves:
        for r, c in new_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_mode(grid, sequential=True)


def _main_parallel_reference(grid: np.ndarray) -> np.ndarray:
    return _solve_with_mode(grid, sequential=False)


_BASE_TEMPLATE_D11 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=int,
)


def _sample_base_template() -> np.ndarray:
    grid = _BASE_TEMPLATE_D11.copy()
    grid = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        grid = np.fliplr(grid)
    return grid


def _recolor_template(grid: np.ndarray) -> np.ndarray:
    old_bg, old_bar, old_seg, old_obs = 0, 1, 2, 3
    new_bg, new_bar, new_seg, new_obs = random.sample(range(10), 4)
    mapping = {old_bg: new_bg, old_bar: new_bar, old_seg: new_seg, old_obs: new_obs}
    return np.vectorize(lambda v: mapping[int(v)])(grid).astype(int)


def generate_input() -> np.ndarray:
    grid = _recolor_template(_sample_base_template())
    assert not np.array_equal(main(grid), _main_parallel_reference(grid)), "template must preserve D11 effect"
    return grid
