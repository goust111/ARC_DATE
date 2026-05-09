# Combo variant of 142ca369 (index 12)
# combo_spec: D15/0+D16/1
# Name: 颜色对决定反射+反射墙与穿透墙

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 保留原题对象：3 像素 L 形箭头、单色墙、对角线射线。
#    - D15 先把墙分成两类：
#      * 1 像素墙 = reflect wall，基础行为是反射并继承墙色。
#      * 3 像素轴对齐直线墙 = pass wall，基础行为是穿透并继承墙色。
#    - D16 再引入颜色配对覆写：
#      * same-color 只会把 reflect wall 改成 pass。
#      * diff-color 只会把 pass wall 改成 reflect。
#    - 因而 solver 不能只看颜色，也不能只看墙尺寸；必须先识别墙型，再结合
#      当前射线颜色与墙色判断这次接触是“保持基础行为”还是“覆写基础行为”。
#    - 具体例子：箭头色 2，方向 (+1,+1)，当前走到 (8,8)。
#      * 若 (8,9) 是 1 像素色 2 墙：基础应反射，但 same-color 把它改成穿透，
#        射线继续走到 (9,9)，路径不拐弯。
#      * 若 (8,9) 是 3 像素竖线色 5 墙：基础应穿透，但 diff-color 把它改成反射，
#        当前格 (8,8) 变成 5，随后改向走到 (9,7)。
#      * 若 (8,9) 是 1 像素色 5 墙：保持基础反射。
#      * 若 (8,9) 是 3 像素竖线色 2 墙：保持基础穿透。
#    - 这四种组合共同构成融合判定表；生成器会强制它们全部出现。
#
# b) 复杂度校准
#    - 只保留两种墙型与 same/diff 两种颜色关系，总共 4 种局部情况，复杂度可控。
#    - 每个样例强制出现四个象限：reflect+same、reflect+diff、pass+same、pass+diff。
#      这样不会退化成“只靠墙尺寸”或“只靠颜色”的单维规则。
#    - 仍以 sparse 布局为主，箭头数和事件数限制在中等范围，避免整张图过密。
#
# c) 隐式假设审查
#    - 原 solver 的 `wall_cells` 把所有墙视为同类，这在融合后失效；必须按 4 连通块
#      分类并记住每个墙像素对应的墙型与颜色。
#    - 原 side-touch 代码无条件翻转轴向，这依赖“所有墙都反射”；融合后只能在
#      effective behavior 为 reflect 时翻转。
#    - 原 corner-hit 代码默认墙不可穿透，这在 reflect+same 与 pass+same 情况下失效；
#      这些情况都必须允许射线进入墙格继续前进。
#    - 新 solver 对输入有两个新假设：
#      * 非箭头连通块只会是 1 像素或 3 像素轴对齐线段。
#      * 每个计划接触点不会同时遇到两个独立侧墙。
#      生成器显式保证这两点，并用 oracle 验证。
#
# d) 放置合法性
#    - 箭头、墙、不同箭头的路径都不重叠，也不越界。
#    - 生成器只规划 side-touch 事件；这样 3 像素 pass wall 不需要与射线路径共格，
#      视觉更清楚，也更容易控制合法性。
#    - 生成路径时会同时避开已有墙的 diagonal hit 与 side-touch 触发位置，防止出现
#      未计划事件。
#
# e) 视觉可观察性
#    - 人类能在样例里直接看到：
#      * 有些单像素墙会把射线“放过去”，但前提是它和射线同色。
#      * 有些 3 像素墙反而会把射线“弹回来”，前提是它和射线异色。
#      * 同时也存在 reflect+diff 与 pass+same 作为对照组。
#    - 因为墙型差异来自尺寸、颜色关系来自是否同色，这两条线索都是肉眼可见的。
#
# f) 可视化整洁度
#    - 保持原题 18-20 的网格尺度和稀疏对象密度。
#    - 每支箭的事件数有限，且路径间互相避让，避免局部区域堆叠过多颜色。
#
# g) 变化的可推断性
#    - 调色板限制在 3-5 种活动颜色，颜色重复出现，使 same/diff 关系可比较。
#    - 生成器强制四类事件都出现，并统计 override 事件数量，防止样例退化。


Coord = tuple[int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_arrow_pts(pts: list[Coord]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[Coord]) -> tuple[tuple[int, int], Coord] | None:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None
    mr, mc = miss[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _classify_wall_component(pts: list[Coord]) -> str:
    if len(pts) == 1:
        return "reflect"
    if len(pts) == 3:
        rs = {r for r, _c in pts}
        cs = {c for _r, c in pts}
        if len(rs) == 1 or len(cs) == 1:
            return "pass"
    return "reflect"


def _detect_scene(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, list[Coord]]], set[Coord], dict[Coord, str], dict[Coord, int]]:
    arrow_cells: set[Coord] = set()
    arrow_sigs: set[tuple[Coord, ...]] = set()
    arrows: list[tuple[int, list[Coord]]] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                sig = tuple(sorted(pts))
                arrows.append((int(col), pts))
                arrow_sigs.add(sig)
                arrow_cells |= set(pts)

    wall_kind: dict[Coord, str] = {}
    wall_color: dict[Coord, int] = {}
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if tuple(sorted(pts)) in arrow_sigs:
                continue
            kind = _classify_wall_component(pts)
            for p in pts:
                wall_kind[p] = kind
                wall_color[p] = int(col)

    return arrows, arrow_cells, wall_kind, wall_color


def _effective_action(kind: str, same_color: bool) -> str:
    if kind == "reflect":
        return "pass" if same_color else "reflect"
    return "reflect" if not same_color else "pass"


def _record_event(
    stats: dict[str, int], kind: str, same_color: bool, action: str, mode: str
) -> None:
    pair = "same" if same_color else "diff"
    key = f"{kind}_{pair}_{action}"
    stats[key] += 1
    stats[f"{mode}_events"] += 1
    stats["same_events" if same_color else "diff_events"] += 1
    stats["actual_reflections" if action == "reflect" else "actual_passes"] += 1
    if (kind == "reflect" and same_color) or (kind == "pass" and not same_color):
        stats["override_events"] += 1


def _trace(grid: np.ndarray, collect_stats: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells, wall_kind, wall_color = _detect_scene(g, bg)
    wall_cells = set(wall_kind)
    out = g.copy()
    stats = {
        "reflect_same_pass": 0,
        "reflect_diff_reflect": 0,
        "pass_same_pass": 0,
        "pass_diff_reflect": 0,
        "override_events": 0,
        "corner_events": 0,
        "side_events": 0,
        "same_events": 0,
        "diff_events": 0,
        "actual_reflections": 0,
        "actual_passes": 0,
    }

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen: set[tuple[Coord, int, int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in arrow_cells:
                break

            if (nr, nc) in wall_cells:
                wcol = int(wall_color[(nr, nc)])
                kind = wall_kind[(nr, nc)]
                same = int(cur_col) == int(wcol)
                action = _effective_action(kind, same)
                _record_event(stats, kind, same, action, "corner")
                if action == "reflect":
                    if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                        out[int(cur[0]), int(cur[1])] = int(wcol)
                    cur_col = int(wcol)
                    dr, dc = -int(dr), -int(dc)
                    continue
                cur = (int(nr), int(nc))
                cur_col = int(wcol)
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(g[nr, nc]) != int(bg):
                continue

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(wall_color[(nr + 1, nc)])
                kind = wall_kind[(nr + 1, nc)]
                same = int(cur_col) == int(wcol)
                action = _effective_action(kind, same)
                _record_event(stats, kind, same, action, "side")
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if action == "reflect":
                    dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(wall_color[(nr - 1, nc)])
                kind = wall_kind[(nr - 1, nc)]
                same = int(cur_col) == int(wcol)
                action = _effective_action(kind, same)
                _record_event(stats, kind, same, action, "side")
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if action == "reflect":
                    dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(wall_color[(nr, nc + 1)])
                kind = wall_kind[(nr, nc + 1)]
                same = int(cur_col) == int(wcol)
                action = _effective_action(kind, same)
                _record_event(stats, kind, same, action, "side")
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if action == "reflect":
                    dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(wall_color[(nr, nc - 1)])
                kind = wall_kind[(nr, nc - 1)]
                same = int(cur_col) == int(wcol)
                action = _effective_action(kind, same)
                _record_event(stats, kind, same, action, "side")
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if action == "reflect":
                    dc = -int(dc)

    if collect_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _trace(grid, collect_stats=False)


def _make_arrow(start: Coord, dr: int, dc: int) -> list[Coord]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [(int(r), int(c)) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _diag_walk(start: Coord, dr: int, dc: int, steps: int) -> list[Coord]:
    sr, sc = start
    return [(int(sr + dr * i), int(sc + dc * i)) for i in range(1, steps + 1)]


def _to_edge(start: Coord, dr: int, dc: int, H: int, W: int) -> list[Coord]:
    r, c = start
    pts: list[Coord] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < H and 0 <= c < W):
            return pts
        pts.append((int(r), int(c)))


def _apply_symmetry(arr: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(arr, int(rot_k))
    if flip_lr:
        out = np.fliplr(out)
    return np.array(out, dtype=int)

def _gen_attempt() -> np.ndarray | None:
    H = W = 20
    palette = _rng.sample(list(range(1, 10)), k=6)
    a_col, b_col, c_col, d_col, bw_col, dw_col = [int(x) for x in palette]

    grid = np.zeros((H, W), dtype=int)
    start_a = (2, 2)
    start_b = (1, 15)
    start_c = (16, 1)
    start_d = (16, 15)

    arrow_a = _make_arrow(start_a, 1, 1)
    arrow_b = _make_arrow(start_b, 1, -1)
    arrow_c = _make_arrow(start_c, -1, 1)
    arrow_d = _make_arrow(start_d, -1, -1)

    contact_a = _diag_walk(start_a, 1, 1, 3)[-1]
    contact_b = _diag_walk(start_b, 1, -1, 3)[-1]
    contact_c = _diag_walk(start_c, -1, 1, 3)[-1]
    contact_d = _diag_walk(start_d, -1, -1, 3)[-1]

    wall_a = [(int(contact_a[0] + 1), int(contact_a[1]))]
    wall_b = [(int(contact_b[0] + 1), int(contact_b[1]))]
    wall_c = [
        (int(contact_c[0] - 1), int(contact_c[1] - 1)),
        (int(contact_c[0] - 1), int(contact_c[1])),
        (int(contact_c[0] - 1), int(contact_c[1] + 1)),
    ]
    wall_d = [
        (int(contact_d[0] - 1), int(contact_d[1] - 1)),
        (int(contact_d[0] - 1), int(contact_d[1])),
        (int(contact_d[0] - 1), int(contact_d[1] + 1)),
    ]

    all_input = set(arrow_a) | set(arrow_b) | set(arrow_c) | set(arrow_d) | set(wall_a) | set(wall_b) | set(wall_c) | set(wall_d)
    expected_input_count = len(arrow_a) + len(arrow_b) + len(arrow_c) + len(arrow_d) + len(wall_a) + len(wall_b) + len(wall_c) + len(wall_d)
    if len(all_input) != expected_input_count:
        return None

    for r, c in arrow_a:
        grid[r, c] = int(a_col)
    for r, c in arrow_b:
        grid[r, c] = int(b_col)
    for r, c in arrow_c:
        grid[r, c] = int(c_col)
    for r, c in arrow_d:
        grid[r, c] = int(d_col)

    for r, c in wall_a:
        grid[r, c] = int(a_col)
    for r, c in wall_b:
        grid[r, c] = int(bw_col)
    for r, c in wall_c:
        grid[r, c] = int(c_col)
    for r, c in wall_d:
        grid[r, c] = int(dw_col)

    rot_k = _rng.randint(0, 3)
    flip_lr = bool(_rng.randint(0, 1))
    grid_t = _apply_symmetry(grid, rot_k, flip_lr)

    _out, stats = _trace(grid_t, collect_stats=True)
    if stats["reflect_same_pass"] < 1:
        return None
    if stats["reflect_diff_reflect"] < 1:
        return None
    if stats["pass_same_pass"] < 1:
        return None
    if stats["pass_diff_reflect"] < 1:
        return None
    if stats["override_events"] < 2:
        return None
    return grid_t


def generate_input() -> np.ndarray:
    for _ in range(700):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 700 attempts")


def _variant_stats(grid: np.ndarray) -> dict[str, int]:
    _out, stats = _trace(grid, collect_stats=True)
    return stats
