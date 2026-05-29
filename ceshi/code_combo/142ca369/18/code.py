# Combo variant of 142ca369 (index 18)
# combo_spec: D2/1+D15/0+D19/0
# Name: 颜色转向+墙壁功能分化+边缘图例

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 输入分成两层：顶部 2 行是灰色图例带；下面是原题风格的稀疏 playfield。
   - 图例仍然编码“墙颜色 -> 转向方向”：
       上排固定 1 个彩色像素，下排左侧像素表示 CCW，下排右侧像素表示 CW。
   - playfield 里有两类墙：
       1 像素墙 = reflect，触发“按图例转 90 度”；
       3 像素轴对齐直线墙 = pass，只改射线颜色，不改方向。
   - 具体小例子：方向 (1,1) 的射线走到 (r,c)。
       若 (r+1,c) 是 1 像素红墙，且图例说红= CW，则 (r,c) 染成红，方向变成 (1,-1)。
       若 (r+1,c) 是 3 像素红墙，则 (r,c) 也染成红，但下一步仍沿 (1,1) 前进。
     差异是：颜色决定“若反射则怎么拐”，尺寸决定“到底拐不拐”。

b) 复杂度校准
   - 图例只放 2 个规则色，分别对应 CW / CCW，颜色码表很小。
   - 每个样例强制同时使用：
       两种 legend 颜色都作为 reflect 墙出现；
       两种 legend 颜色都作为 pass 墙出现。
   - 这样人类能从少量样例看出：同色墙在 1 像素和 3 像素时行为不同，且 1 像素墙的拐向又由图例决定。

c) 隐式假设审查
   - 原 solver 把所有非背景像素都当物理墙；这里不成立，必须先剥离顶边图例。
   - 原 solver 假设“墙几何位置决定翻哪根轴”；这里不成立，reflect 墙改成读图例决定 CW/CCW。
   - D15 又破坏了“所有墙都反射”的假设；所以 tracer 不能只存一个 wall set，必须按连通块分类。
   - 顶层逻辑现在依赖三个条件同时成立：
       颜色映射来自图例；
       反射/穿透来自墙组件尺寸；
       每个接触点最多只有一个有效侧墙。
     generator 会显式保证这些条件。

d) 放置合法性
   - 顶部灰色只用于 legend，playfield 不使用灰色，避免解析串区。
   - 箭头、reflect 墙、pass 墙在输入里互不重叠。
   - pass 墙允许被射线在后续一步踏入 1 个像素，这是变体语义的一部分；solver 显式支持。
   - generator 只生成 1 像素或 3 像素直线墙，不生成别的连通块形状。

e) 视觉可观察性
   - 图例是一个独立顶部带，显眼且不会被误认成障碍物。
   - 1 像素墙附近会出现“变色 + 转向”；3 像素墙附近会出现“变色 + 直行”。
   - 同一 legend 颜色在 reflect/pass 两种尺寸上都会复用，因此“颜色”和“尺寸”两个维度都可见。

f) 可视化整洁度
   - 生成器固定为两支箭，分别占左半区和右半区，避免输出路径互相缠绕。
   - 每支箭都是“先 reflect，再 pass”，路径层次稳定，便于肉眼跟踪。
   - 墙只放在射线路径相邻 1 格处，仍保留原题家族的稀疏观感。

g) 变化的可推断性
   - 只有 2 个 legend 颜色，且每个样例都重复使用。
   - 颜色变化仍然直接来自所接触的墙色，没有隐藏排序或数值依赖。
   - 为了防退化，generator 还要求：
       变体输出必须不同于“原始镜面反射 solver”；
       变体输出必须不同于“所有墙都按图例转向”的退化 solver。
     这分别量化 D2/D19 和 D15 是否真的起作用。

h) main() 的新隐式假设与 generator 保证
   - 假设 1：legend 是顶部连续 2 行灰色带。generator 保证。
   - 假设 2：playfield 中 1 像素墙=reflect，3 像素直线墙=pass。generator 只产出这两种。
   - 假设 3：每个 reflect 墙颜色都能在 legend 中找到方向。generator 保证所有墙色都来自 legend。
   - 假设 4：每个接触点最多一个侧墙。generator 用受控布局 + 事件回放校验保证。
   - 假设 5：箭头路径彼此不穿越。generator 用事件追踪拒绝交叉样例。
"""


Coord = tuple[int, int]

LEGEND_BG = 5
LEGEND_H = 2
PLAY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rot_cw(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), -int(dr)


def _rot_ccw(dr: int, dc: int) -> tuple[int, int]:
    return -int(dc), int(dr)


def _apply_turn(dr: int, dc: int, turn: str) -> tuple[int, int]:
    return _rot_cw(dr, dc) if turn == "cw" else _rot_ccw(dr, dc)


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


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


def _parse_legend(grid: np.ndarray) -> tuple[int, dict[int, str]]:
    h, w = grid.shape
    legend_h = 0
    while legend_h < h and np.any(grid[legend_h] == LEGEND_BG):
        legend_h += 1
    if legend_h == 0:
        return 0, {}

    legend = grid[:legend_h, :]
    sep_cols = [c for c in range(w) if np.all(legend[:, c] == LEGEND_BG)]
    turn_by_color: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:]):
        interior = legend[:, a + 1 : b]
        if interior.shape[0] != LEGEND_H or interior.shape[1] < 2:
            continue
        colors = set(map(int, np.unique(interior)))
        colors.discard(0)
        colors.discard(LEGEND_BG)
        if len(colors) != 1:
            continue
        col = int(next(iter(colors)))
        pts = [(int(r), int(c)) for r, c in np.argwhere(interior == col)]
        if len(pts) != 2:
            continue
        lower = [c for r, c in pts if r == LEGEND_H - 1]
        if len(lower) != 1:
            continue
        turn_by_color[col] = "ccw" if lower[0] <= (interior.shape[1] - 1) // 2 else "cw"
    return legend_h, turn_by_color


def _detect_scene(core: np.ndarray, bg: int):
    arrow_cells: set[Coord] = set()
    arrow_sigs: set[tuple[Coord, ...]] = set()
    arrows: list[tuple[int, list[Coord]]] = []

    for col in colors_present(core, exclude={bg}):
        for coords in components_mask(core == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                sig = tuple(sorted(pts))
                arrows.append((int(col), pts))
                arrow_sigs.add(sig)
                arrow_cells |= set(pts)

    wall_kind: dict[Coord, str] = {}
    wall_color: dict[Coord, int] = {}
    wall_comp: dict[Coord, int] = {}
    component_info: dict[int, dict[str, int | str]] = {}
    comp_id = 0

    for col in colors_present(core, exclude={bg}):
        for coords in components_mask(core == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if tuple(sorted(pts)) in arrow_sigs:
                continue
            kind = _classify_wall_component(pts)
            component_info[comp_id] = {"kind": kind, "color": int(col), "size": len(pts)}
            for p in pts:
                wall_kind[p] = kind
                wall_color[p] = int(col)
                wall_comp[p] = comp_id
            comp_id += 1

    return arrows, arrow_cells, wall_kind, wall_color, wall_comp, component_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Top edge legend maps wall colors to CW / CCW turns.
    - 1-pixel walls are reflective: they recolor the ray and rotate it by the legend rule.
    - 3-pixel line walls are permeable: they recolor the ray but keep the same direction.
    """
    g = np.array(grid, dtype=int)
    legend_h, turn_by_color = _parse_legend(g)
    if legend_h >= g.shape[0]:
        return g.copy()

    core = g[legend_h:, :].copy()
    h, w = core.shape
    bg = int(get_bg(core))

    arrows, arrow_cells, wall_kind, wall_color, _wall_comp, _component_info = _detect_scene(core, bg)
    wall_cells = set(wall_kind)
    out = core.copy()

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
            if not _in_bounds(h, w, nr, nc):
                break

            if (nr, nc) in arrow_cells:
                break

            if (nr, nc) in wall_cells:
                kind = wall_kind[(nr, nc)]
                wcol = int(wall_color[(nr, nc)])
                if kind == "pass":
                    cur = (int(nr), int(nc))
                    cur_col = int(wcol)
                    continue
                turn = turn_by_color.get(wcol)
                if turn is None:
                    break
                if int(core[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = _apply_turn(int(dr), int(dc), turn)
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(core[nr, nc]) != int(bg):
                continue

            touches: list[Coord] = []
            if (nr + int(dr), nc) in wall_cells:
                touches.append((int(nr + dr), int(nc)))
            if (nr, nc + int(dc)) in wall_cells:
                touches.append((int(nr), int(nc + dc)))
            if len(touches) > 1:
                break
            if not touches:
                continue

            wr, wc = touches[0]
            kind = wall_kind[(wr, wc)]
            wcol = int(wall_color[(wr, wc)])
            out[nr, nc] = int(wcol)
            cur_col = int(wcol)
            if kind == "reflect":
                turn = turn_by_color.get(wcol)
                if turn is None:
                    break
                dr, dc = _apply_turn(int(dr), int(dc), turn)

    result = g.copy()
    result[legend_h:, :] = out
    return result


def _main_original_core(core: np.ndarray) -> np.ndarray:
    g = np.array(core, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[Coord] = set()
    arrows: list[tuple[int, list[Coord]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    out = g.copy()

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
            if not _in_bounds(h, w, nr, nc):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def _main_turn_all(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    legend_h, turn_by_color = _parse_legend(g)
    if legend_h >= g.shape[0]:
        return g.copy()

    core = g[legend_h:, :].copy()
    h, w = core.shape
    bg = int(get_bg(core))

    arrows, arrow_cells, wall_kind, wall_color, _wall_comp, _component_info = _detect_scene(core, bg)
    wall_cells = set(wall_kind)
    out = core.copy()

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
            if not _in_bounds(h, w, nr, nc):
                break

            if (nr, nc) in arrow_cells:
                break

            if (nr, nc) in wall_cells:
                wcol = int(wall_color[(nr, nc)])
                turn = turn_by_color.get(wcol)
                if turn is None:
                    break
                if int(core[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = _apply_turn(int(dr), int(dc), turn)
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(core[nr, nc]) != int(bg):
                continue

            touches: list[Coord] = []
            if (nr + int(dr), nc) in wall_cells:
                touches.append((int(nr + dr), int(nc)))
            if (nr, nc + int(dc)) in wall_cells:
                touches.append((int(nr), int(nc + dc)))
            if len(touches) > 1:
                break
            if not touches:
                continue

            wr, wc = touches[0]
            wcol = int(wall_color[(wr, wc)])
            turn = turn_by_color.get(wcol)
            if turn is None:
                break
            out[nr, nc] = int(wcol)
            cur_col = int(wcol)
            dr, dc = _apply_turn(int(dr), int(dc), turn)

    result = g.copy()
    result[legend_h:, :] = out
    return result


def _build_legend(width: int, turn_by_color: dict[int, str], order: list[int]) -> np.ndarray:
    legend = np.full((LEGEND_H, width), LEGEND_BG, dtype=int)
    for i, col in enumerate(order):
        start = 1 + 4 * i
        legend[0, start + 1] = int(col)
        legend[1, start + (0 if turn_by_color[int(col)] == "ccw" else 2)] = int(col)
    return legend


def _make_arrow(start: Coord, dr: int, dc: int) -> list[Coord]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [(int(r), int(c)) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _original_turn(dr: int, dc: int, side: str) -> tuple[int, int]:
    return (-int(dr), int(dc)) if side == "row" else (int(dr), -int(dc))


def _mismatch_sides(dr: int, dc: int, turn: str) -> list[str]:
    target = _apply_turn(int(dr), int(dc), turn)
    return [side for side in ("row", "col") if _original_turn(int(dr), int(dc), side) != target]


def _wall_cells(kind: str, side: str, r: int, c: int, dr: int, dc: int, h: int, w: int) -> list[Coord] | None:
    if side == "row":
        wr, wc = int(r + dr), int(c)
        if not _in_bounds(h, w, wr, wc):
            return None
        if kind == "reflect":
            return [(wr, wc)]
        cells = [(wr, wc + d) for d in (-1, 0, 1)]
    else:
        wr, wc = int(r), int(c + dc)
        if not _in_bounds(h, w, wr, wc):
            return None
        if kind == "reflect":
            return [(wr, wc)]
        cells = [(wr + d, wc) for d in (-1, 0, 1)]

    if all(_in_bounds(h, w, rr, cc) for rr, cc in cells):
        return [(int(rr), int(cc)) for rr, cc in cells]
    return None


def _trace_events(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    legend_h, turn_by_color = _parse_legend(g)
    core = g[legend_h:, :].copy()
    h, w = core.shape
    bg = int(get_bg(core))

    arrows, arrow_cells, wall_kind, wall_color, wall_comp, component_info = _detect_scene(core, bg)
    wall_cells = set(wall_kind)

    path_owner: dict[Coord, int] = {}
    arrow_sequences: list[list[int]] = []
    events: list[dict[str, object]] = []
    ambiguous = False
    overlap = False

    for arrow_idx, (start_col, pts) in enumerate(arrows):
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen: set[tuple[Coord, int, int, int]] = set()
        seq: list[int] = []

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not _in_bounds(h, w, nr, nc):
                break

            if (nr, nc) in arrow_cells:
                break

            if (nr, nc) in wall_cells:
                comp_id = int(wall_comp[(nr, nc)])
                kind = str(wall_kind[(nr, nc)])
                wcol = int(wall_color[(nr, nc)])
                events.append(
                    {
                        "arrow": arrow_idx,
                        "mode": "front",
                        "kind": kind,
                        "color": wcol,
                        "component": comp_id,
                    }
                )
                if not seq or seq[-1] != comp_id:
                    seq.append(comp_id)
                if kind == "pass":
                    cur = (int(nr), int(nc))
                    cur_col = int(wcol)
                    owner = path_owner.get(cur)
                    if owner is not None and owner != arrow_idx:
                        overlap = True
                    path_owner[cur] = arrow_idx
                    continue
                turn = turn_by_color.get(wcol)
                if turn is None:
                    ambiguous = True
                    break
                cur_col = int(wcol)
                dr, dc = _apply_turn(int(dr), int(dc), turn)
                continue

            cur = (int(nr), int(nc))
            owner = path_owner.get(cur)
            if owner is not None and owner != arrow_idx:
                overlap = True
            path_owner[cur] = arrow_idx

            if int(core[nr, nc]) != int(bg):
                continue

            touches: list[Coord] = []
            if (nr + int(dr), nc) in wall_cells:
                touches.append((int(nr + dr), int(nc)))
            if (nr, nc + int(dc)) in wall_cells:
                touches.append((int(nr), int(nc + dc)))
            if len(touches) > 1:
                ambiguous = True
                break
            if not touches:
                continue

            wr, wc = touches[0]
            comp_id = int(wall_comp[(wr, wc)])
            kind = str(wall_kind[(wr, wc)])
            wcol = int(wall_color[(wr, wc)])
            events.append(
                {
                    "arrow": arrow_idx,
                    "mode": "side",
                    "kind": kind,
                    "color": wcol,
                    "component": comp_id,
                }
            )
            if not seq or seq[-1] != comp_id:
                seq.append(comp_id)
            cur_col = int(wcol)
            if kind == "reflect":
                turn = turn_by_color.get(wcol)
                if turn is None:
                    ambiguous = True
                    break
                dr, dc = _apply_turn(int(dr), int(dc), turn)

        arrow_sequences.append(seq)

    return {
        "events": events,
        "arrow_sequences": arrow_sequences,
        "component_info": component_info,
        "ambiguous": ambiguous,
        "path_overlap": overlap,
    }


def _gen_attempt() -> np.ndarray | None:
    total_h, w = _rng.randint(18, 20), _rng.randint(18, 20)
    h = total_h - LEGEND_H
    if h < 16:
        return None

    palette = PLAY_COLORS[:]
    _rng.shuffle(palette)
    cw_color, ccw_color, arrow_a_color, arrow_b_color = [int(x) for x in palette[:4]]
    turn_by_color = {int(cw_color): "cw", int(ccw_color): "ccw"}
    legend_order = [int(cw_color), int(ccw_color)]
    _rng.shuffle(legend_order)

    for _ in range(250):
        core = np.zeros((h, w), dtype=int)
        occupied: set[Coord] = set()

        start_a = (_rng.randint(1, 2), _rng.randint(3, 4))
        start_b = (_rng.randint(1, 2), _rng.randint(w - 5, w - 4))
        arrow_a = _make_arrow(start_a, 1, 1)
        arrow_b = _make_arrow(start_b, 1, -1)
        if any(not _in_bounds(h, w, r, c) for r, c in arrow_a + arrow_b):
            continue
        if set(arrow_a) & set(arrow_b):
            continue

        max_a1 = min(5, h - start_a[0] - 6, w - start_a[1] - 6)
        max_b1 = min(5, h - start_b[0] - 6, start_b[1] - 2)
        if max_a1 < 3 or max_b1 < 3:
            continue
        a1 = _rng.randint(3, max_a1)
        b1 = _rng.randint(3, max_b1)
        pa1 = (int(start_a[0] + a1), int(start_a[1] + a1))
        pb1 = (int(start_b[0] + b1), int(start_b[1] - b1))

        reflect_a = _wall_cells("reflect", "row", pa1[0], pa1[1], 1, 1, h, w)
        reflect_b = _wall_cells("reflect", "row", pb1[0], pb1[1], 1, -1, h, w)
        if reflect_a is None or reflect_b is None:
            continue

        max_a2 = min(5, h - pa1[0] - 4, pa1[1] - 2)
        max_b2 = min(5, h - pb1[0] - 4, w - pb1[1] - 3)
        if max_a2 < 2 or max_b2 < 2:
            continue
        a2 = _rng.randint(2, max_a2)
        b2 = _rng.randint(2, max_b2)
        pa2 = (int(pa1[0] + a2), int(pa1[1] - a2))
        pb2 = (int(pb1[0] + b2), int(pb1[1] + b2))

        pass_a_options = []
        pass_b_options = []
        for side in ("row", "col"):
            wall = _wall_cells("pass", side, pa2[0], pa2[1], 1, -1, h, w)
            if wall is not None:
                pass_a_options.append(wall)
            wall = _wall_cells("pass", side, pb2[0], pb2[1], 1, 1, h, w)
            if wall is not None:
                pass_b_options.append(wall)
        if not pass_a_options or not pass_b_options:
            continue
        pass_a = _rng.choice(pass_a_options)
        pass_b = _rng.choice(pass_b_options)

        objects = [
            (arrow_a, arrow_a_color),
            (arrow_b, arrow_b_color),
            (reflect_a, cw_color),
            (reflect_b, ccw_color),
            (pass_a, ccw_color),
            (pass_b, cw_color),
        ]

        ok = True
        for cells, color in objects:
            for r, c in cells:
                if (r, c) in occupied:
                    ok = False
                    break
            if not ok:
                break
            for r, c in cells:
                core[r, c] = int(color)
                occupied.add((int(r), int(c)))
        if not ok:
            continue

        legend = _build_legend(w, turn_by_color, legend_order)
        full = np.vstack([legend, core])
        trace = _trace_events(full)
        if bool(trace["ambiguous"]) or bool(trace["path_overlap"]):
            continue

        sequences = trace["arrow_sequences"]
        if len(sequences) != 2 or any(len(seq) != 2 for seq in sequences):
            continue

        component_info = trace["component_info"]
        seq_kinds = [[str(component_info[cid]["kind"]) for cid in seq] for seq in sequences]
        if seq_kinds != [["reflect", "pass"], ["reflect", "pass"]]:
            continue

        reflect_colors = {
            int(component_info[cid]["color"])
            for seq in sequences
            for cid in seq
            if str(component_info[cid]["kind"]) == "reflect"
        }
        pass_colors = {
            int(component_info[cid]["color"])
            for seq in sequences
            for cid in seq
            if str(component_info[cid]["kind"]) == "pass"
        }
        if reflect_colors != {int(cw_color), int(ccw_color)}:
            continue
        if pass_colors != {int(cw_color), int(ccw_color)}:
            continue

        variant_out = main(full)
        turn_all_out = _main_turn_all(full)
        orig_out = _main_original_core(core)

        if np.array_equal(variant_out[LEGEND_H:, :], turn_all_out[LEGEND_H:, :]):
            continue
        if np.array_equal(variant_out[LEGEND_H:, :], orig_out):
            continue
        return full

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
