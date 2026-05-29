# Combo variant of a47bf94d (index 27)
# combo_spec: D1/0+D2/0+D8/2+D19/0
# Name: 最短路径+反射交叉+旋转节点+多标记编码

from __future__ import annotations

import heapq
import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 原题依赖显式的 8/5/9 管道组件；这里把中部改成“迷宫走廊”，走廊本身是背景 0，墙是统一的 2 色，因此 solver 不再顺着现成管道追踪，
#      而是要在可通行空间里做最短路搜索，这落实 D1/0。
#    - 具体对象：
#      1. 顶/底两排仍是按颜色配对的 3x3 token 位置；
#      2. 输入源 token 仍保留两类：solid 3x3 或 X；
#      3. 对侧目标位固定放同色 diamond 锚点；
#      4. 输出中路径被画成 8，但输入里没有任何 8 预制线。
#    - 中部走廊来自原题的换轨网络几何，但颜色线条被删掉，只保留可通行的空隙，因此“是否连通、哪条更短”变成空间搜索问题。
#    - 交叉/拐弯节点改成两类可见标记：
#      1. 只有 1 个对角 5 标记的中心格：表示穿过该点时保持当前方向直行；
#      2. 有 3 个对角 5 标记的中心格：表示穿过该点时像镜面一样 90 度反射转向。
#         主导模板是缺 NW 的 slash 与缺 NE 的 backslash；局部标记叠加时也可能出现它们的镜像缺角，但仍只编码这两种反射手性。
#    - color-9 不再是原题里的宽接合体，也不再承担交叉反射；这里把它们改成可穿越的“旋转门”矩形块，路径从一侧进入、从对侧出来，
#      同时把远端要补出的 hook 形旋转 90 度。宽>高顺时针，高>宽逆时针，这落实 D8/2。
#    - 具体小例子：一条颜色从顶部 solid 出发，沿背景走廊向下；在第一个 3-marker 节点从竖向被反射到右侧，
#      中间穿过一个 2x3 的 9 号宽矩形，因此远端 hook 顺时针转一次；随后它穿过一个 1-marker 十字点保持水平直行，
#      再在另一端 3-marker 节点反射向下到达底部 diamond。输出时，起点 solid 变 diamond，终点 diamond 变成旋转后的 hook，
#      中间背景最短路被涂成 8。原题对应环节则是：沿现成 8 管线走、在 5 标记直穿、在 9 块里通行但不改形状。
#
# b) 复杂度校准
#    - 若只做“迷宫最短路”，样例会退化成普通配对路径题；因此必须让最短路同时经过 1-marker 直行点、3-marker 反射点和至少一个 9 旋转门。
#    - 3-marker 同时编码“要转弯”和左右手性；generator 以缺 NW / 缺 NE 为主模板，必要时允许其镜像缺角自然出现，
#      但语义仍压缩成 slash / backslash 两类，使“数量决定直/转，缺角决定转向手性”同时可见。
#    - 若继续沿用原题的 X / diamond 作为远端完成形状，则 90 度旋转不可见；因此把远端补全形状改成有方向的 5-cell hook，
#      使 D8 的旋转门真正影响输出，而本地 solid->diamond 的原核心仍保留。
#
# c) 隐式假设审查（对应原 main 的关键前提）
#    - 原 `track_mask = g != bg` 假设所有非背景都是管道部件。现在不成立：2 是墙，5 是对角标记，9 是特殊门，token 盒内部还要整体屏蔽。
#    - 原 `_trace_wire(...)` 假设局部贪心沿组件前进即可。现在不成立：必须做带方向状态的最短路，因为 1-marker/3-marker 对“下一个方向”施加约束。
#    - 原 `9-rect pass-through` 假设 9 的角色只影响路由。现在不成立：9 还会累计远端 hook 的旋转状态。
#    - 原终态 X / diamond 都是旋转对称的。现在不成立：为了让 D8 可见，远端形状改成有朝向的 hook，solver 需要在路由状态里携带旋转模 4。
#    - 新 solver 的核心假设：
#      1. 每个活动颜色恰有一个源 token（solid 或 X）和一个对侧 diamond 锚点；
#      2. 所有 9 组件都是独立非方形矩形门，且只暴露一对相对面；
#      3. 5 标记只作为节点的对角编码出现，不会散落在别处；
#      4. 迷宫是单像素宽走廊，因此除特殊节点外不会出现大量等长分支。
#
# d) 放置合法性
#    - generator 先用原题的 permutation/crossover 思路生成一张“单像素宽走廊网络”，再把它嵌在 2 色墙里，避免开放平面造成歧义。
#    - token 仍放在离边缘 2-3 行的位置，3x3 完整落盘；走廊端点与 token 中心保持原题同样的 2 格关系。
#    - 9 旋转门只放在纯直线段上，不放在 marker 节点或 token 附近；放置前检查周围都是墙，保证它只暴露一对相对面。
#    - 5 标记只放在对角墙格，不覆盖走廊本身，因此不阻塞最短路。
#
# e) 视觉可观察性
#    - 输入没有任何 8 号线，输出新增 8 号路径，D1 直接可见。
#    - 1-marker 与 3-marker 数量差异非常显眼，且输出中的路径会在 1-marker 处“穿过不拐”，在 3-marker 处“入一侧出另一侧”，
#      这使 D19 + D2 的组合可见。
#    - 9 矩形门保留在输出中，路径在门两侧衔接但不覆盖门本身；终点 hook 的朝向变化与经过的宽/高 9 门对应，因此 D8 可见。
#
# f) 可视化整洁度
#    - 颜色职责被强约束：
#      0=走廊背景，2=墙，5=交叉编码，8=输出路径，9=旋转门，其他少量颜色只给 token。
#    - 走廊一像素宽，marker 只占对角，9 门是孤立粗块；三类结构分层明确，不会在同一格叠加。
#
# g) 变化的可推断性
#    - 颜色本身不参与排序或 tie-break；颜色只告诉解题者“哪个源 token 对应哪个锚点”。
#    - 真正控制输出的是空间结构：最短路、marker 数量、3-marker 缺角手性、9 门长宽比。它们都能直接从像素观察到。
#
# h) 退化防护
#    - generator 会拒绝没有 1-marker 直行点、没有 3-marker 转向点、没有 9 门穿越、或所有路径净旋转都为 0 的样本，
#      避免退化成普通最短路或普通配对题。
#
# i) 新隐式假设审查
#    - `main()` 现在假设 hook 远端锚点的 3x3 盒是空闲的；generator 用“每色恰一源一锚”的构造保证。
#    - `main()` 现在假设不同 9 门互不相连；generator 通过只在纯直线段上单独放门并保持间距保证。
#    - `main()` 现在假设路径不会需要在普通走廊中心无规则拐弯；generator 的单像素宽网络保证了拐弯只发生在几何角点或 3-marker 节点。


BG = 0
WALL = 2
MARK = 5
PATH = 8
ROT = 9

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
_DIR_TO_IDX = {d: i for i, d in enumerate(_DIRS)}
_DIR2FACE = {(-1, 0): "S", (0, 1): "W", (1, 0): "N", (0, -1): "E"}
_FACE2DIR = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_SLASH_REFLECT = {0: 3, 1: 2, 2: 1, 3: 0}       # enter S/W/N/E -> exit W/S/E/N
_BACKSLASH_REFLECT = {0: 1, 1: 0, 2: 3, 3: 2}   # enter S/W/N/E -> exit E/N/W/S

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
_HOOK_BASE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)]  # top bar + left leg


def _rot_offsets(offsets: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    k %= 4
    out = list(offsets)
    for _ in range(k):
        out = [(c, -r) for r, c in out]  # 90 deg clockwise
    return out


def _stamp_token(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[cr + dr, cc + dc] = color


def _clear_box(grid: np.ndarray, center: tuple[int, int], bg: int) -> None:
    cr, cc = center
    grid[cr - 1:cr + 2, cc - 1:cc + 2] = bg


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            isolated = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                isolated = False
                                break
                    if not isolated:
                        break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in _X_OFF:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 1])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr, dc in _DIAMOND_OFF:
                rr, cc = r0 + 1 + dr, c0 + 1 + dc
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _by_color(found: dict[tuple[int, int], tuple[int, tuple[int, int]]]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for center, (col, _) in found.items():
        assert col not in out
        out[col] = center
    return out


def _build_token_block_mask(
    g: np.ndarray,
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    diamonds: dict[tuple[int, int], tuple[int, tuple[int, int]]],
) -> np.ndarray:
    mask = np.zeros_like(g, dtype=bool)
    for found in (solids, xs, diamonds):
        for (cr, cc), _ in found.items():
            mask[cr - 1:cr + 2, cc - 1:cc + 2] = True
    return mask


def _find_crossing_rules(g: np.ndarray, bg: int, blocked: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    h, w = g.shape
    rules: dict[tuple[int, int], dict[str, object]] = {}
    diags = [(-1, -1, "NW"), (-1, 1, "NE"), (1, -1, "SW"), (1, 1, "SE")]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != bg or blocked[r, c]:
                continue
            present = [name for dr, dc, name in diags if int(g[r + dr, c + dc]) == MARK]
            if len(present) == 1:
                rules[(r, c)] = {"kind": "straight"}
            elif len(present) == 3:
                missing = ({name for _, _, name in diags} - set(present)).pop()
                reflect = _SLASH_REFLECT if missing in ("NW", "SE") else _BACKSLASH_REFLECT
                rules[(r, c)] = {"kind": "turn", "reflect": reflect, "missing": missing}
    return rules


def _find_rotation_rects(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    for comp in components_mask(g == ROT, connectivity=4):
        cells = {(int(r), int(c)) for r, c in comp}
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh <= 1 or ww <= 1 or hh == ww:
            continue
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and (nr, nc) not in cells and int(g[nr, nc]) == bg:
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        if set(faces) not in ({"N", "S"}, {"E", "W"}):
            continue
        delta = 1 if ww > hh else -1
        travel_cost = (hh + 1) if set(faces) == {"N", "S"} else (ww + 1)
        rects.append(
            {
                "cells": cells,
                "bbox": (r0, c0, r1, c1),
                "faces": faces,
                "delta": delta,
                "travel_cost": travel_cost,
            }
        )
    return rects


def _pick_exit_face_point(cur: tuple[int, int], exit_face: str, pts: list[tuple[int, int]]) -> tuple[int, int]:
    r, c = cur
    if exit_face in ("N", "S"):
        return min(pts, key=lambda p: abs(p[1] - c))
    return min(pts, key=lambda p: abs(p[0] - r))


def _endpoint_pair(src: tuple[int, int], dst: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if src[0] < dst[0]:
        return (src[0] + 2, src[1]), (dst[0] - 2, dst[1])
    return (src[0] - 2, src[1]), (dst[0] + 2, dst[1])


def _route_shortest(
    g: np.ndarray,
    bg: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: np.ndarray,
    crossing_rules: dict[tuple[int, int], dict[str, object]],
    rects: list[dict[str, object]],
) -> dict[str, object]:
    h, w = g.shape
    passable = (g == bg) & (~blocked)
    passable[start] = True
    passable[goal] = True

    rect_by_cell: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rects):
        for cell in rect["cells"]:
            rect_by_cell[cell] = idx

    start_state = (start[0], start[1], 4, 0)  # r, c, dir_idx(4=None), rot_mod4
    pq: list[tuple[int, int, tuple[int, int, int, int]]] = []
    counter = 0
    heapq.heappush(pq, (0, counter, start_state))
    dist = {start_state: 0}
    prev: dict[tuple[int, int, int, int], tuple[tuple[int, int, int, int] | None, int | None]] = {start_state: (None, None)}

    goal_state: tuple[int, int, int, int] | None = None

    while pq:
        cost, _, state = heapq.heappop(pq)
        if cost != dist.get(state):
            continue
        r, c, dir_idx, rot = state
        if (r, c) == goal:
            goal_state = state
            break

        if (r, c) in crossing_rules and dir_idx != 4:
            rule = crossing_rules[(r, c)]
            if rule["kind"] == "straight":
                cand_dirs = [dir_idx]
            else:
                cand_dirs = [int(rule["reflect"][dir_idx])]
        else:
            cand_dirs = [0, 1, 2, 3]

        for nd_idx in cand_dirs:
            dr, dc = _DIRS[nd_idx]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue

            if passable[nr, nc]:
                nstate = (nr, nc, nd_idx, rot)
                ncost = cost + 1
                if ncost < dist.get(nstate, 10**9):
                    dist[nstate] = ncost
                    prev[nstate] = (state, None)
                    counter += 1
                    heapq.heappush(pq, (ncost, counter, nstate))
                continue

            rect_idx = rect_by_cell.get((nr, nc))
            if rect_idx is None:
                continue
            rect = rects[rect_idx]
            entry_face = _DIR2FACE[(dr, dc)]
            if entry_face not in rect["faces"]:
                continue
            exit_face = _OPP[entry_face]
            if exit_face not in rect["faces"]:
                continue
            exit_cell = _pick_exit_face_point((r, c), exit_face, rect["faces"][exit_face])
            nrot = (rot + int(rect["delta"])) % 4
            nstate = (exit_cell[0], exit_cell[1], nd_idx, nrot)
            ncost = cost + int(rect["travel_cost"])
            if ncost < dist.get(nstate, 10**9):
                dist[nstate] = ncost
                prev[nstate] = (state, rect_idx)
                counter += 1
                heapq.heappush(pq, (ncost, counter, nstate))

    assert goal_state is not None

    states_rev = []
    rect_hits_rev: list[int] = []
    cur = goal_state
    while cur is not None:
        states_rev.append(cur)
        pstate, rect_idx = prev[cur]
        if rect_idx is not None:
            rect_hits_rev.append(rect_idx)
        cur = pstate
    states = list(reversed(states_rev))
    rect_hits = list(reversed(rect_hits_rev))
    path_cells = [(int(r), int(c)) for r, c, _, _ in states]

    straight_hits = 0
    turn_hits = 0
    for r, c, _, _ in states[1:]:
        info = crossing_rules.get((int(r), int(c)))
        if info is None:
            continue
        if info["kind"] == "straight":
            straight_hits += 1
        else:
            turn_hits += 1

    return {
        "path": path_cells,
        "rotation": int(goal_state[3]),
        "rect_hits": rect_hits,
        "straight_hits": straight_hits,
        "turn_hits": turn_hits,
        "cost": int(dist[goal_state]),
    }


def _analyze_routes(
    g: np.ndarray,
    bg: int,
    solids: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    xs: dict[tuple[int, int], tuple[int, tuple[int, int]]],
    diamonds: dict[tuple[int, int], tuple[int, tuple[int, int]]],
) -> tuple[dict[int, dict[str, object]], dict[tuple[int, int], dict[str, object]], list[dict[str, object]]]:
    blocked = _build_token_block_mask(g, solids, xs, diamonds)
    crossing_rules = _find_crossing_rules(g, bg, blocked)
    rects = _find_rotation_rects(g, bg)

    solid_by_color = _by_color(solids)
    x_by_color = _by_color(xs)
    diamond_by_color = _by_color(diamonds)
    source_colors = sorted(set(solid_by_color) | set(x_by_color))
    assert source_colors

    routes: dict[int, dict[str, object]] = {}
    for col in source_colors:
        assert not (col in solid_by_color and col in x_by_color)
        assert col in diamond_by_color
        src = solid_by_color[col] if col in solid_by_color else x_by_color[col]
        dst = diamond_by_color[col]
        start, goal = _endpoint_pair(src, dst)
        route = _route_shortest(g, bg, start, goal, blocked, crossing_rules, rects)
        route["source_center"] = src
        route["target_center"] = dst
        route["source_kind"] = "solid" if col in solid_by_color else "x"
        routes[col] = route

    return routes, crossing_rules, rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route paired tokens through a shortest-path maze whose crossings reflect and whose maroon gates rotate the delivered shape.

    Input structure:
    - The board is mostly walls and background corridors rather than prebuilt cyan pipes.
    - Each active color appears exactly twice: one source token (either a solid 3x3 block or an X) and one opposite-side diamond anchor.
    - Some corridor centers carry diagonal 5 markers, and some isolated non-square 9 rectangles sit directly on straight corridor segments.

    Rule:
    - Find the shortest route from each source-side endpoint to its same-colored anchor endpoint through background corridors.
    - A solid source collapses in place to a diamond; an X source stays an X.
    - At a 1-marker node the route must continue straight; at a 3-marker node it reflects 90 degrees according to the missing diagonal corner.
    - Crossing a wide 9 rectangle rotates the delivered hook clockwise; crossing a tall 9 rectangle rotates it counter-clockwise.
    - Draw the chosen route in color 8 and replace the far diamond anchor with the appropriately rotated hook.

    Key insight: The color pairing is still local, but endpoint pairing now depends on shortest-path search plus two stateful routing rules: 5-marker crossings constrain direction, and 9 gates accumulate hook rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)
    routes, _, _ = _analyze_routes(g, bg, solids, xs, diamonds)

    out = g.copy()
    for col, route in routes.items():
        src = route["source_center"]
        dst = route["target_center"]
        if route["source_kind"] == "solid":
            _clear_box(out, src, bg)
            _stamp_token(out, src, col, _DIAMOND_OFF)

        for r, c in route["path"]:
            if int(out[r, c]) == bg:
                out[r, c] = PATH

        _clear_box(out, dst, bg)
        hook = _rot_offsets(_HOOK_BASE, int(route["rotation"]))
        _stamp_token(out, dst, col, hook)

    return out


# --------------- generator ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
RECT_COUNT_WEIGHTS = [15, 45, 30, 10]  # 0..3 attempted
RECT_MARGIN_GAP = 1
GEN_MAX_RETRIES = 500


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    n = len(perm)
    visited = [False] * n
    crossovers = []
    for start in range(n):
        if visited[start] or perm[start] == start:
            visited[start] = True
            continue
        cycle = []
        i = start
        while not visited[i]:
            visited[i] = True
            cycle.append(i)
            i = perm[i]
        if len(cycle) < 2:
            continue
        crossovers.append((cycle[0], -1))
        for k in range(len(cycle) - 1, 0, -1):
            crossovers.append((cycle[k], cycle[(k + 1) % len(cycle)]))
        crossovers.append((-1, cycle[1]))
    return crossovers


def _swap_permutation(n: int) -> list[int]:
    pairs = [(i, j) for i in range(n) for j in range(i + 2, n)]
    i, j = random.choice(pairs)
    perm = list(range(n))
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def _carve_box(grid: np.ndarray, center: tuple[int, int]) -> None:
    cr, cc = center
    grid[cr - 1:cr + 2, cc - 1:cc + 2] = BG


def _place_straight_marker(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    if grid[r - 1, c - 1] == WALL:
        grid[r - 1, c - 1] = MARK


def _place_turn_marker(grid: np.ndarray, center: tuple[int, int], orientation: str) -> None:
    r, c = center
    diag_cells = {
        "NW": (r - 1, c - 1),
        "NE": (r - 1, c + 1),
        "SW": (r + 1, c - 1),
        "SE": (r + 1, c + 1),
    }
    missing = "NW" if orientation == "slash" else "NE"
    for name, (rr, cc) in diag_cells.items():
        if name == missing:
            continue
        if grid[rr, cc] == WALL:
            grid[rr, cc] = MARK


def _build_corridor_network(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None, int, int]:
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None
    col_seg_start: dict[int, int] = {}
    straight_marker_count = 0
    turn_marker_count = 0

    for j in range(n):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = BG

    for xover_idx, (src_slot, dst_slot) in enumerate(crossovers):
        xrow = xover_rows[xover_idx]
        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]

        active_cols = {ch_cols[j] for j in range(n) if slot_wire[j] is not None}
        if bypass_wire is not None:
            active_cols.add(bypass_col)

        for c in active_cols:
            if c in col_seg_start:
                start = col_seg_start[c]
                grid[start:xrow + 1, c] = BG

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min:c_max + 1] = BG

        if src_col != dst_col:
            orientation = "slash" if dst_col > src_col else "backslash"
            _place_turn_marker(grid, (xrow, src_col), orientation)
            _place_turn_marker(grid, (xrow, dst_col), orientation)
            turn_marker_count += 2

        for c in active_cols:
            if c == src_col or c == dst_col:
                continue
            if c_min < c < c_max:
                _place_straight_marker(grid, (xrow, c))
                straight_marker_count += 1

        new_seg_start = {c: xrow for c in active_cols if c != src_col}
        new_seg_start[dst_col] = xrow
        col_seg_start = new_seg_start

        if src_slot == -1:
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

    for j in range(n):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c]
                grid[start:bot_ep_row + 1, c] = BG

    return slot_wire, bypass_wire, straight_marker_count, turn_marker_count


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    r0, c0, r1, c1 = a
    rr0, cc0, rr1, cc1 = b
    return r0 <= rr1 + gap and rr0 <= r1 + gap and c0 <= cc1 + gap and cc0 <= c1 + gap


def _find_rotation_candidates(grid: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    h, w = grid.shape
    cands: list[tuple[str, tuple[int, int, int, int]]] = []

    for r in range(1, h - 3):
        for c in range(1, w - 2):
            bbox = (r, c, r + 2, c + 1)  # tall 3x2
            if (
                np.all(grid[r:r + 3, c] == BG)
                and np.all(grid[r:r + 3, c + 1] == WALL)
                and grid[r - 1, c] == BG
                and grid[r + 3, c] == BG
                and grid[r - 1, c + 1] != BG
                and grid[r + 3, c + 1] != BG
                and not np.any(grid[r:r + 3, c - 1] == BG)
                and not np.any(grid[r:r + 3, c + 2] == BG)
            ):
                cands.append(("tall", bbox))
            if (
                np.all(grid[r:r + 3, c] == WALL)
                and np.all(grid[r:r + 3, c + 1] == BG)
                and grid[r - 1, c + 1] == BG
                and grid[r + 3, c + 1] == BG
                and grid[r - 1, c] != BG
                and grid[r + 3, c] != BG
                and not np.any(grid[r:r + 3, c - 1] == BG)
                and not np.any(grid[r:r + 3, c + 2] == BG)
            ):
                cands.append(("tall", bbox))

    for r in range(1, h - 2):
        for c in range(1, w - 3):
            bbox = (r, c, r + 1, c + 2)  # wide 2x3
            if (
                np.all(grid[r, c:c + 3] == BG)
                and np.all(grid[r + 1, c:c + 3] == WALL)
                and grid[r, c - 1] == BG
                and grid[r, c + 3] == BG
                and grid[r + 1, c - 1] != BG
                and grid[r + 1, c + 3] != BG
                and not np.any(grid[r - 1, c:c + 3] == BG)
                and not np.any(grid[r + 2, c:c + 3] == BG)
            ):
                cands.append(("wide", bbox))
            if (
                np.all(grid[r, c:c + 3] == WALL)
                and np.all(grid[r + 1, c:c + 3] == BG)
                and grid[r + 1, c - 1] == BG
                and grid[r + 1, c + 3] == BG
                and grid[r, c - 1] != BG
                and grid[r, c + 3] != BG
                and not np.any(grid[r - 1, c:c + 3] == BG)
                and not np.any(grid[r + 2, c:c + 3] == BG)
            ):
                cands.append(("wide", bbox))

    return cands


def _place_rotation_rects(
    grid: np.ndarray,
    target_count: int,
    required_kinds: list[str],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    cands = _find_rotation_candidates(grid)
    random.shuffle(cands)
    placed: list[tuple[str, tuple[int, int, int, int]]] = []

    def try_place(pool: list[tuple[str, tuple[int, int, int, int]]]) -> bool:
        random.shuffle(pool)
        for kind, bbox in pool:
            if any(_bbox_overlaps(bbox, prev_bbox, RECT_MARGIN_GAP) for _, prev_bbox in placed):
                continue
            r0, c0, r1, c1 = bbox
            grid[r0:r1 + 1, c0:c1 + 1] = ROT
            placed.append((kind, bbox))
            return True
        return False

    for kind in required_kinds[:target_count]:
        if not try_place([cand for cand in cands if cand[0] == kind]):
            return []

    for kind, bbox in cands:
        if len(placed) >= target_count:
            break
        if any(_bbox_overlaps(bbox, prev_bbox, RECT_MARGIN_GAP) for _, prev_bbox in placed):
            continue
        r0, c0, r1, c1 = bbox
        grid[r0:r1 + 1, c0:c1 + 1] = ROT
        placed.append((kind, bbox))
    return placed


def _build_specs(
    n: int,
    ch_cols: list[int],
    perm: list[int],
    colors: list[int],
    top_row: int,
    bot_row: int,
    x_channels: set[int],
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for i in range(n):
        top_center = (top_row, ch_cols[i])
        bot_center = (bot_row, ch_cols[perm[i]])
        source_on_top = random.choice([True, False])
        source_center = top_center if source_on_top else bot_center
        target_center = bot_center if source_on_top else top_center
        specs.append(
            {
                "color": colors[i],
                "top_center": top_center,
                "bot_center": bot_center,
                "source_center": source_center,
                "target_center": target_center,
                "source_kind": "x" if i in x_channels else "solid",
            }
        )
    return specs


def _make_input(base_grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    inp = base_grid.copy()
    for spec in specs:
        src = spec["source_center"]
        dst = spec["target_center"]
        color = int(spec["color"])
        _carve_box(inp, src)
        _carve_box(inp, dst)
        if spec["source_kind"] == "solid":
            _stamp_token(inp, src, color, _SOLID_OFF)
        else:
            _stamp_token(inp, src, color, _X_OFF)
        _stamp_token(inp, dst, color, _DIAMOND_OFF)
    return inp


def _expected_from_specs(inp: np.ndarray, specs: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, int]]:
    bg = BG
    solids = _find_solid_3x3(inp, bg)
    xs = _find_x_3x3(inp, bg)
    diamonds = _find_diamond_3x3(inp, bg)
    routes, _, rects = _analyze_routes(inp, bg, solids, xs, diamonds)

    out = inp.copy()
    total_straight = 0
    total_turn = 0
    total_rect = 0
    nonzero_rot = 0
    cw_hits = 0
    ccw_hits = 0

    for spec in specs:
        color = int(spec["color"])
        route = routes[color]
        src = tuple(spec["source_center"])
        dst = tuple(spec["target_center"])
        if spec["source_kind"] == "solid":
            _clear_box(out, src, bg)
            _stamp_token(out, src, color, _DIAMOND_OFF)

        for r, c in route["path"]:
            if int(out[r, c]) == bg:
                out[r, c] = PATH

        _clear_box(out, dst, bg)
        _stamp_token(out, dst, color, _rot_offsets(_HOOK_BASE, int(route["rotation"])))

        total_straight += int(route["straight_hits"])
        total_turn += int(route["turn_hits"])
        total_rect += len(route["rect_hits"])
        nonzero_rot += int(route["rotation"] != 0)
        for rect_idx in route["rect_hits"]:
            if int(rects[rect_idx]["delta"]) > 0:
                cw_hits += 1
            else:
                ccw_hits += 1

    stats = {
        "straight_hits": total_straight,
        "turn_hits": total_turn,
        "rect_hits": total_rect,
        "nonzero_rot_paths": nonzero_rot,
        "rect_count": len(rects),
        "cw_hits": cw_hits,
        "ccw_hits": ccw_hits,
    }
    return out, stats


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_row + EP_TOKEN_GAP
    colors = random.sample(TOKEN_COLORS_POOL, n)
    perm = _swap_permutation(n)
    crossovers = _decompose_to_crossovers(perm)
    if not crossovers:
        return None

    min_wire_rows = len(crossovers) * 3 + 2
    h_min = top_ep_row + min_wire_rows + top_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])
    bot_row = h - top_row - 1
    bot_ep_row = bot_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(3, avail // (len(crossovers) + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + 3 * s for s in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        return None

    base = np.full((h, w), WALL, dtype=int)
    slot_wire, bypass_wire, straight_marker_count, turn_marker_count = _build_corridor_network(
        base, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None
    if straight_marker_count <= 0 or turn_marker_count <= 0:
        return None

    target_rects = random.choices([1, 2, 3], weights=RECT_COUNT_WEIGHTS[1:])[0]
    required_rect_kinds = [random.choice(["wide", "tall"])]
    if target_rects >= 2 and random.random() < 0.35:
        required_rect_kinds = ["wide", "tall"]
    placed_rects = _place_rotation_rects(base, target_rects, required_rect_kinds)
    if not placed_rects:
        return None

    x_count = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(x_count, n)))
    specs = _build_specs(n, ch_cols, perm, colors, top_row, bot_row, x_channels)
    inp = _make_input(base, specs)

    try:
        expected, stats = _expected_from_specs(inp, specs)
    except Exception:
        return None

    if stats["straight_hits"] <= 0 or stats["turn_hits"] <= 0:
        return None
    if stats["rect_hits"] <= 0 or stats["nonzero_rot_paths"] <= 0:
        return None
    if "wide" in required_rect_kinds and stats["cw_hits"] <= 0:
        return None
    if "tall" in required_rect_kinds and stats["ccw_hits"] <= 0:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
