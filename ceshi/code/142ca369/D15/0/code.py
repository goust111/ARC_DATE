# Variant of 142ca369
# Dimension: D15 — 墙壁功能分化 (obstacle)
# Change: 所有墙壁功能相同——都触发侧面反射并传递颜色... → 反射墙与穿透墙

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the D15 variant
# -----------------------------------
# a) 具体化与行为追踪
#    - 箭头、背景、对角线射线都保持原题结构不变。
#    - 新规则只改“墙”的功能，并把墙的尺寸固定成语义线索：
#      1 像素墙 = 反射墙；3 像素轴对齐直线墙 = 穿透墙。
#    - 反射墙沿用原题侧碰规则：射线走到某个对角线格子后，若其前进方向同轴相邻处有反射墙，
#      就把当前射线颜色改成墙色，并在该格子处翻转对应轴向。
#    - 穿透墙仍然在同样的“侧碰位置”起作用，但不翻转方向；射线只继承墙色，然后继续沿原对角线前进。
#    - 小例子：箭头颜色 2，方向 (+1,+1)。当射线走到 (8,8) 时，若 (8,9) 是 1 像素墙，
#      则 (8,8) 变成墙色并转向 (+1,-1)；若 (8,9) 是 3 像素竖线墙，则 (8,8) 变成墙色，
#      下一步仍走到 (9,9)。差异就在“同样发生染色，但只有单像素墙会拐弯”。
# b) 复杂度校准
#    - 只保留两类墙，不引入额外颜色映射表或更多尺寸档位，避免规则过碎。
#    - 每个样例都强制同时出现两类墙，避免新维度偶然缺席而退化回原题。
#    - 穿透墙固定为 3 像素直线，视觉上比单像素墙更醒目，求解者可以直接把“尺寸”当判别线索。
# c) 原 solver 中依赖旧属性的不变性
#    - 原代码把所有非箭头像素都塞进一个 `wall_cells` 集合；这隐含“所有墙等价”。
#      变体后必须先按连通块分类，否则 side-touch 时无法决定是翻转还是直行。
#    - 原 side-touch 代码对 `dr`/`dc` 无条件翻转；这依赖“任何墙都反射”。
#      变体后只能在 1 像素墙上翻转，3 像素墙只能改色不翻转。
#    - 原生成器把 1 像素墙和 3 像素墙都当成同一种 bounce 的不同外观。
#      变体后尺寸不再只是外观，生成器必须显式规划 reflect / pass 两类事件。
# d) 放置合法性
#    - 箭头和不同墙组件在输入中仍保持不重叠。
#    - 穿透墙允许射线在后续一步穿过它的一个像素；这是变体的必要适配，且 solver / generator 都显式处理了这种重叠。
#    - 生成器给两支箭和两堵墙使用互不相同的颜色，避免同色 4 连通并块。
# e) 视觉可观察性
#    - 反射墙附近，射线会拐弯且变色。
#    - 穿透墙附近，射线不会拐弯，但会在接触点之后持续变成墙色。
#    - 这两种现象在 2-4 个样例中足以形成稳定对比。
# f) 可视化整洁度
#    - 网格大小、箭头大小、墙形态、对角线路径都沿用原题，减少额外视觉负担。
#    - 限制箭头数和事件数，避免单个样例中过多折返造成混乱。
# g) 颜色变化可推断性
#    - 墙色始终是射线新颜色来源，且生成器尽量避免墙色等于当前射线色，让“染色”可见。
#    - 颜色变化仍由“接触到哪一堵墙”决定，没有新的隐藏颜色映射。
# h) 量化变体行为
#    - 核心指标：样例中 3 像素穿透墙数量、1 像素反射墙数量、以及“同一输入上新旧 solver 输出是否不同”。
#      只要穿透墙存在且常导致新旧输出不同，就说明变体确实发生。
# i) 新 solver 的隐式假设与生成器保证
#    - 假设 1：1 像素非箭头块就是反射墙；3 像素轴对齐直线块就是穿透墙。
#      生成器只产出这两种墙组件。
#    - 假设 2：箭头与墙不会同色 4 连通。
#      生成器直接为两支箭和两堵墙分配互异颜色，保证连通块可被正确分类。
#    - 假设 3：每个接触点至多触发一个侧墙。
#      生成器只在单轴放墙，并用 oracle 检查 `main(grid)` 是否等于规划输出。


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


def _detect_scene(g: np.ndarray, bg: int):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - 1-pixel walls reflect exactly as in the original task.
    - 3-pixel axis-aligned walls are permeable: they recolor the ray but do not flip it.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells, wall_kind, wall_color = _detect_scene(g, bg)
    wall_cells = set(wall_kind)
    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen = set()

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
                if wall_kind[(nr, nc)] == "pass":
                    cur = (int(nr), int(nc))
                    cur_col = int(wcol)
                    continue
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(g[nr, nc]) != int(bg):
                continue

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(wall_color[(nr + 1, nc)])
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if wall_kind[(nr + 1, nc)] == "reflect":
                    dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(wall_color[(nr - 1, nc)])
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if wall_kind[(nr - 1, nc)] == "reflect":
                    dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(wall_color[(nr, nc + 1)])
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if wall_kind[(nr, nc + 1)] == "reflect":
                    dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(wall_color[(nr, nc - 1)])
                out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                if wall_kind[(nr, nc - 1)] == "reflect":
                    dc = -int(dc)

    return out


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
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow1_col, arrow2_col, wall1_col, wall2_col = [int(x) for x in palette[:4]]

    for _ in range(160):
        grid = np.zeros((H, W), dtype=int)
        out = np.zeros((H, W), dtype=int)

        start1 = (_rng.randint(2, 3), _rng.randint(2, 4))
        seg1 = _rng.randint(4, 6)
        arrow1 = _make_arrow(start1, 1, 1)
        ray1a = _diag_walk(start1, 1, 1, seg1)
        contact1 = ray1a[-1]
        wall1 = [(int(contact1[0]), int(contact1[1] + 1))]
        ray1b = _to_edge(contact1, 1, -1, H, W)

        start2 = (_rng.randint(15, 17), _rng.randint(2, 3))
        seg2 = _rng.randint(3, 5)
        arrow2 = _make_arrow(start2, -1, 1)
        ray2a = _diag_walk(start2, -1, 1, seg2)
        contact2 = ray2a[-1]
        wall2 = [
            (int(contact2[0] - 1), int(contact2[1] - 1)),
            (int(contact2[0] - 1), int(contact2[1])),
            (int(contact2[0] - 1), int(contact2[1] + 1)),
        ]
        ray2b = _to_edge(contact2, -1, 1, H, W)

        all_input = set(arrow1) | set(arrow2) | set(wall1) | set(wall2)
        if len(all_input) != len(arrow1) + len(arrow2) + len(wall1) + len(wall2):
            continue

        ray1_set = set(ray1a) | set(ray1b)
        ray2_allowed_wall = set(wall2)
        ray2_set = set(ray2a) | set(ray2b)
        if ray1_set & all_input:
            continue
        if (ray2_set - ray2_allowed_wall) & (set(arrow1) | set(arrow2) | set(wall1)):
            continue
        if ray1_set & (ray2_set - ray2_allowed_wall):
            continue
        if len(ray1_set) != len(ray1a) + len(ray1b):
            continue
        if len(ray2_set) != len(ray2a) + len(ray2b):
            continue

        for r, c in arrow1:
            grid[r, c] = out[r, c] = int(arrow1_col)
        for r, c in arrow2:
            grid[r, c] = out[r, c] = int(arrow2_col)
        for r, c in wall1:
            grid[r, c] = out[r, c] = int(wall1_col)
        for r, c in wall2:
            grid[r, c] = out[r, c] = int(wall2_col)

        for r, c in ray1a[:-1]:
            out[r, c] = int(arrow1_col)
        out[contact1] = int(wall1_col)
        for r, c in ray1b:
            if grid[r, c] == 0:
                out[r, c] = int(wall1_col)

        for r, c in ray2a[:-1]:
            out[r, c] = int(arrow2_col)
        out[contact2] = int(wall2_col)
        for r, c in ray2b:
            if grid[r, c] == 0:
                out[r, c] = int(wall2_col)

        rot_k = _rng.randint(0, 3)
        flip_lr = bool(_rng.randint(0, 1))
        grid_t = _apply_symmetry(grid, rot_k, flip_lr)
        out_t = _apply_symmetry(out, rot_k, flip_lr)

        if not np.array_equal(main(grid_t), out_t):
            continue
        return grid_t

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
