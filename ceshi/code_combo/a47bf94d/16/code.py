# Combo variant of a47bf94d (index 16)
# combo_spec: D3/0+D13/1
# Name: 连续轨迹涂色+按长度排序处理
#
# Design analysis
# a) Concrete behavior and trace
#    - 保留原题对象: 端点外侧 2 格处的 3x3 token、由 8/5/9 组成的正交管道、以及 9-rect 宽通道块。
#    - D3/0 让“传输”变可见: token 一旦沿管道移动，经过的 8、5、以及 9-rect 内实际穿越条带都会改成
#      token 自身颜色。
#    - D13/1 让“谁先走”变关键: 所有 token 先按自己在原始未占线网络中的完整路径长度从短到长排序，
#      再逐个执行。已被更早 token 染色的管道不再算作可走管道，所以后续 token 只能在剩余的 5/8/9 上前进。
#    - 具体小例子:
#      1. 红色 solid 的全长是 9，蓝色 solid 的全长是 17，二者共享末端竖直干线。
#      2. 先处理红色: 红色在原位塌成 diamond，并把那段共享干线染红，远端补一个红色 X。
#      3. 再处理蓝色: 蓝色仍从自己的端点出发，但走到那段已变红的干线前就停住，因为红线不再是可用管道。
#         于是蓝色只留下被截断的蓝色 trail，不再补到原本的远端。
#    - 与原题的差异集中在“route application”阶段: token 检测、solid->diamond、X 发送 diamond、9-rect
#      的穿越几何都保留；新增的是 trail 着色和“先染先占”的顺序约束。
# b) Complexity calibration
#    - 不增加新的 token 类型，也不引入隐藏的颜色映射。唯一新自由度是“路径长短决定先后”，而长度可直接
#      从输出里彩色 trail 的相对长度观察。
#    - 生成器强制保留 4-5 个通道、非恒等置换和至少一个真实顺序冲突，避免样本退化成“只是 D3 连续涂色”。
#    - 还要求至少两个不同路径长度档位，并保证有多于一个成功路由，否则人眼看不到“短的先占线，长的被影响”。
# c) Implicit-assumption audit of original main()
#    - 原 `_trace_wire()` 只需终点；D3/0 下必须返回完整 trail，否则无法连续着色。
#    - 原 solver 默认每个 token 都在完整网络上独立路由，route 顺序无关。D13/1 打破此假设，因此 solver
#      需要维护每个 component 的 `available` 管道集合，并在每一步后删除已占用 trail。
#    - 原版进入 9-rect 时可直接“跳”到出口；连续 trail 下必须显式补出矩形内部条带，而且在占线版本里还要
#      检查条带上的 9 是否仍可用，否则后续 token 会在矩形入口处被截断。
#    - 原版只要 trace 到某个 dead end 就会按当前方向在外侧 2 格放图形；融合后这个前提不成立，因为 dead end
#      可能是前一个 token 造成的人为堵点。这里改成: 只有真正抵达另一个原始 endpoint 时才允许补远端图形。
# d) Placement legality
#    - 仍沿用原布局约束: token 中心与 endpoint 相距 2 格，因此 trail 着色不会覆盖 token 自身。
#    - 9-rect 只嵌在已有合法管道上；占线后仅改变“后续可走性”，不会制造越界或重叠对象。
#    - 生成器继续用 oracle 过滤，并额外检查成功路由数、被截断/改道数、长度档位数和与 D3-only 基线的差异。
# e) Visual observability
#    - 解题者在 2-4 个样例里可以同时看到:
#      1. 所有成功传输都会留下同色连续 trail；
#      2. 明显更短的 trail 先完整到达；
#      3. 更长 token 的 trail 会在先前已染色区域前终止，或改到别的仍可达 endpoint。
#    - 顺序依据不是颜色值，而是输出里直接可见的几何长度，因此是人类可观察的。
# f) Visual cleanliness
#    - 占线只发生在原本就属于管道的像素上，不额外添加符号层。
#    - 生成器要求至少两个 route 成功，避免输出只剩一条线；也限制总是全堵死的极端样本。
#    - 共享区域的覆盖只发生在少量相交管道上，整体仍保留清晰的“同色 token -> 同色 trail -> 远端补块”链路。
# g) Color inference
#    - 没有新的颜色映射。trail 颜色永远等于源 token 颜色；变化规律来自结构顺序，而不是颜色数值。
# h) New solver assumptions after fusion
#    - 每个 token 的“原始完整路径长度”在输入解析时可确定；生成器保持原题那种单一合法 endpoint 配置。
#    - 若某 token 的起始 endpoint 已被更短 token 占掉，它不会再移动；生成器接受这种结果，并用指标保证
#      这类情况只是“部分出现”，不是整题退化。
#    - `_trace_wire()` 仍假设 crossing 的直行优先和 9-rect 的对面穿越规则成立；生成器只放原题合法网络。
# i) Generator-side guarantees and anti-degeneracy checks
#    - 每个候选样本都同时计算:
#      1. 融合版输出: 连续 trail + 长度排序 + 占线；
#      2. D3-only 基线: 连续 trail 但不占线。
#    - 只有当 `combo_out` 与 `d3_out` 在同一输入上明显不同，且至少有一个后续 token 被截断或改道时才接受。
#    - 这些指标验证的是融合核心行为本身，而不是单纯 `main(input)` 不报错。

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

_DIAMOND = _DIAMOND_OFF
_X = _X_OFF
_SOLID = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if not maximal:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if not isolated:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 diamonds used as passive hints."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            cols = {int(sub[rr, cc]) for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]}
            if len(cols) != 1:
                continue
            col = cols.pop()
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _DIAMOND_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if not isolated:
                continue
            found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    comps = [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]
    return sorted(comps, key=lambda pts: min(pts))


def _find_9rects(g: np.ndarray) -> list[dict]:
    """Find color-9 rectangular blocks and their bounding boxes."""
    mask9 = g == 9
    if not mask9.any():
        return []
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = set((int(r), int(c)) for r, c in comp9)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        rects.append({"cells": cells, "bbox": (min(rs), min(cs), max(rs), max(cs))})
    return rects


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive axis-aligned segment between two cells."""
    if a == b:
        return [a]
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise AssertionError("9-rect trail must stay axis-aligned")


def _available_rect_faces(rect: dict, comp: set[tuple[int, int]], h: int, w: int) -> dict[str, list[tuple[int, int]]]:
    """Return current available outside-neighbor cells per face for a 9-rect."""
    cells = rect["cells"]
    r0, c0, r1, c1 = rect["bbox"]
    faces: dict[str, list[tuple[int, int]]] = {}
    for r, c in cells:
        if (r, c) not in comp:
            continue
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in comp and (nr, nc) not in cells:
                face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                faces.setdefault(face, []).append((nr, nc))
    return faces


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rect_lookup: dict[tuple[int, int], dict],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    """
    Greedy wire following with explicit trail capture.

    On a 9-rectangle, keep the original pass-through rule, but only if the
    required strip inside the rectangle is still available in `comp`.
    """
    if start not in comp:
        return start, dir0, []

    cur, d, prev = start, dir0, None
    trail = [start]
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped unexpectedly")
        visited.add(state)

        rect = rect_lookup.get(cur)
        if rect is not None:
            faces = _available_rect_faces(rect, comp, h, w)
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            if opp in faces:
                exit_face = opp
            else:
                others = [face for face in faces if face != entry_face]
                if len(others) != 1:
                    return cur, d, trail
                exit_face = others[0]
            pts = faces[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(pts, key=lambda p: (abs(p[1] - ec), abs(p[0] - er), p))
            else:
                exit_cell = min(pts, key=lambda p: (abs(p[0] - er), abs(p[1] - ec), p))
            exit_dir = _FACE2DIR[exit_face]
            inside_end = (exit_cell[0] - exit_dir[0], exit_cell[1] - exit_dir[1])
            rect_strip = _line_cells(cur, inside_end)
            if any(cell not in comp for cell in rect_strip):
                return cur, d, trail
            trail.extend(rect_strip[1:])
            trail.append(exit_cell)
            cur, d, prev = exit_cell, exit_dir, None
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, trail

        rev = (-d[0], -d[1])
        nonrev = [opt for opt in opts if (opt[1], opt[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [opt for opt in opts2 if (opt[1], opt[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [opt for opt in opts2 if opt[1] * d[0] + opt[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2], x[0]))[0]
            else:
                p, dr, dc = sorted(opts2, key=lambda x: (x[1], x[2], x[0]))[0]

        if p == prev:
            return cur, d, trail

        prev = cur
        cur = p
        d = (dr, dc)
        trail.append(cur)

    raise AssertionError("wire trace exceeded step limit")


def _paint_trail(out: np.ndarray, trail: list[tuple[int, int]], color: int, protected: set[tuple[int, int]]) -> None:
    for r, c in trail:
        if (r, c) not in protected:
            out[r, c] = color


def _stamp_shape(out: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    h, w = out.shape
    cr, cc = center
    for dr, dc in offsets:
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < h and 0 <= cc2 < w:
            out[rr, cc2] = color


def _prepare(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)

    track_mask = g != bg
    protected: set[tuple[int, int]] = set()
    for (_cr, _cc), (_col, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
        for rr in range(r0, r0 + 3):
            for cc in range(c0, c0 + 3):
                protected.add((rr, cc))
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False
                protected.add((rr, cc2))
    for cr, cc in diamonds.keys():
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False
                protected.add((rr, cc2))

    comps = _components(track_mask)
    rects = _find_9rects(g)
    rect_lookup: dict[tuple[int, int], dict] = {}
    for rect in rects:
        for cell in rect["cells"]:
            rect_lookup[cell] = rect

    comp_infos = []
    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

        comp_colors = {int(g[r, c]) for r, c in comp}
        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = sorted(p for p in comp if deg[p] <= 1)
        endpoint_set = set(endpoints)
        jobs = []
        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue
                if cen in solids:
                    col, _ = solids[cen]
                    kind = "solid"
                elif cen in xs:
                    col, _ = xs[cen]
                    kind = "x"
                else:
                    continue
                if col in comp_colors:
                    continue
                dir_in = (-odr, -odc)
                full_end, full_dir, full_trail = _trace_wire(comp, ep, dir_in, h, w, rect_lookup)
                jobs.append(
                    {
                        "center": cen,
                        "color": int(col),
                        "start": ep,
                        "dir_in": dir_in,
                        "kind": kind,
                        "full_end": full_end,
                        "full_dir": full_dir,
                        "full_trail": list(full_trail),
                        "full_len": len(full_trail),
                        "full_reached_far": bool(full_trail) and full_end in endpoint_set and full_end != ep,
                    }
                )
        comp_infos.append({"comp": comp, "endpoints": endpoint_set, "jobs": jobs})

    return {
        "grid": g,
        "h": h,
        "w": w,
        "bg": int(bg),
        "solids": solids,
        "xs": xs,
        "protected": protected,
        "rect_lookup": rect_lookup,
        "comp_infos": comp_infos,
    }


def _run_prepared(prep: dict, occupy_paths: bool, sort_by_length: bool) -> tuple[np.ndarray, list[dict]]:
    g = prep["grid"]
    h, w = prep["h"], prep["w"]
    out = g.copy()

    for (_cr, _cc), (col, (r0, c0)) in sorted(prep["solids"].items()):
        out[r0 : r0 + 3, c0 : c0 + 3] = prep["bg"]
        _stamp_shape(out, (r0 + 1, c0 + 1), col, _DIAMOND_OFF)

    route_stats: list[dict] = []
    for info in prep["comp_infos"]:
        comp = set(info["comp"])
        endpoints = info["endpoints"]
        jobs = list(info["jobs"])
        if sort_by_length:
            jobs.sort(key=lambda job: (job["full_len"], job["center"], job["color"], job["start"], job["kind"]))
        else:
            jobs.sort(key=lambda job: (job["center"], job["color"], job["start"], job["kind"]))

        available = set(comp)
        for job in jobs:
            start = job["start"]
            if occupy_paths:
                end, dd, trail = _trace_wire(available, start, job["dir_in"], h, w, prep["rect_lookup"])
            else:
                end, dd, trail = job["full_end"], job["full_dir"], list(job["full_trail"])

            if trail:
                _paint_trail(out, trail, job["color"], prep["protected"])
                if occupy_paths:
                    for cell in trail:
                        available.discard(cell)

            reached_far = bool(trail) and end in endpoints and end != start
            if reached_far:
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                if job["kind"] == "solid":
                    _stamp_shape(out, dest, job["color"], _X)
                else:
                    _stamp_shape(out, dest, job["color"], _DIAMOND)

            route_stats.append(
                {
                    "center": job["center"],
                    "color": job["color"],
                    "kind": job["kind"],
                    "start": start,
                    "full_len": job["full_len"],
                    "full_end": job["full_end"],
                    "full_reached_far": job["full_reached_far"],
                    "actual_len": len(trail),
                    "actual_end": end,
                    "reached_far": reached_far,
                }
            )

    return out, route_stats


def _solve_with_meta(grid: np.ndarray, occupy_paths: bool = True, sort_by_length: bool = True) -> tuple[np.ndarray, list[dict]]:
    prep = _prepare(grid)
    return _run_prepared(prep, occupy_paths=occupy_paths, sort_by_length=sort_by_length)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit endpoint tokens through the pipe network while showing the route and respecting earlier occupiers.

    Input structure:
    - Background cells surround one or more orthogonal wire components made of 8/5/9, plus 3x3 tokens near pipe endpoints.
    - Active tokens are maximal solid 3x3 blocks or isolated X patterns; any pre-existing diamonds are passive hints only.

    Rule:
    - Every solid token collapses in place into a diamond, while an X token keeps its local X shape.
    - Each token is assigned the full path length it would have on the untouched pipe network. Tokens are then processed from
      shorter path to longer path.
    - When a token moves, every pipe pixel on its actually traversed route is recolored to that token's color. Those recolored
      pixels are now occupied, so later tokens may be forced to stop early or to exit through a different still-open endpoint.
    - A complementary remote shape is added only if the token truly reaches another original endpoint on the remaining pipe.

    Key insight:
    - The output is no longer just "source shape changes + far-end completion". The colored trail is both a visualization and
      a resource claim: once a short token paints a segment, that segment disappears as pipe for longer tokens, so path length
      directly determines processing order and later routing outcomes.
    """
    out, _stats = _solve_with_meta(grid, occupy_paths=True, sort_by_length=True)
    return out


# --------------- generator (combo_trails_length_order) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_PARALLEL_PROB = 0.75
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [65, 25, 10]
FAR_DIAMOND_PROB = 0.33

MIN_TRAIL_PIXELS = 18
MIN_TRAIL_COLORS = 2
MIN_LENGTH_VARIANTS = 2
MIN_LENGTH_SPREAD = 4
MIN_ROUTE_EFFECTS = 1
MIN_SUCCESSFUL_ROUTES = 2
MIN_COMBO_DIFF_PIXELS = 8
GEN_MAX_RETRIES = 500


def _decompose_to_crossovers(perm):
    """Decompose permutation into sequential crossover moves using bypass for cycles."""
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


def _stamp_token(grid, center_r, center_c, color, offsets):
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects):
    """Attempt to place one color-9 rectangular block. Returns (r0, c0, r1, c1) or None."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and n >= 2:
        mode = "parallel"
    elif target_wires >= 2 and xover_rows:
        mode = "crossing"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, n - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, n - 1)
        c_lo = c_hi = ch_cols[j0]
        margin_l, margin_r = random.randint(0, 2), random.randint(0, 2)
        if margin_l + margin_r < 1:
            if random.random() < 0.5:
                margin_l = 1
            else:
                margin_r = 1
    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r

    if rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    rect_h = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4))
    if mode == "crossing" and xover_rows:
        xr = random.choice(xover_rows)
        r_start = xr - random.randint(0, rect_h - 1)
    else:
        r_start = random.randint(top_ep_row + 2, bot_ep_row - rect_h - 1)
    rect_r0, rect_r1 = r_start, r_start + rect_h - 1

    wire_cols = sorted(c for c in ch_cols if rect_c0 <= c <= rect_c1)
    if not wire_cols:
        return None
    for c in wire_cols:
        if not np.all(grid[rect_r0 : rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
            return None

    saved = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1].copy()
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9

    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(rect_c0, rect_c1 + 1):
        if rect_r0 - 1 >= 0 and grid[rect_r0 - 1, c] in (5, 8):
            n_cols.add(c)
        if rect_r1 + 1 < h and grid[rect_r1 + 1, c] in (5, 8):
            s_cols.add(c)
    for r in range(rect_r0, rect_r1 + 1):
        if rect_c0 - 1 >= 0 and grid[r, rect_c0 - 1] in (5, 8):
            w_rows.add(r)
        if rect_c1 + 1 < w and grid[r, rect_c1 + 1] in (5, 8):
            e_rows.add(r)
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows)

    if total_wires < 1 or total_wires > 3:
        grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _build_wire_network(grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows):
    """Draw the wire network onto grid. Returns (slot_wire, bypass_wire) after all crossovers."""
    slot_wire = list(range(n))
    bypass_wire = None

    col_seg_start = {}
    for j in range(n):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = 8

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
                start = col_seg_start[c] + 1
                if start <= xrow - 1:
                    grid[start:xrow, c] = 8

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min : c_max + 1] = 8

        for c in active_cols:
            grid[xrow, c] = 8

        for c in active_cols:
            if c == src_col or c == dst_col:
                continue
            if c_min < c < c_max:
                if c - 1 >= c_min:
                    grid[xrow, c - 1] = 5
                if c + 1 <= c_max:
                    grid[xrow, c + 1] = 5

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
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row):
    """Derive the input grid from the wire-only grid by placing endpoint tokens."""
    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            if random.choice([True, False]):
                near, far = top_center, bot_center
            else:
                near, far = bot_center, top_center
            _stamp_token(inp, near[0], near[1], col, _X)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID)
    return inp


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    combo_out, combo_stats = _solve_with_meta(inp, occupy_paths=True, sort_by_length=True)
    d3_out, _d3_stats = _solve_with_meta(inp, occupy_paths=False, sort_by_length=False)

    wire_mask = np.isin(inp, [5, 8, 9])
    recolored = wire_mask & (combo_out != inp)
    painted_colors = {int(combo_out[r, c]) for r, c in np.argwhere(recolored)}
    lengths = [int(stat["full_len"]) for stat in combo_stats]
    blocked = sum(1 for stat in combo_stats if stat["full_reached_far"] and not stat["reached_far"])
    rerouted = sum(1 for stat in combo_stats if stat["reached_far"] and stat["actual_end"] != stat["full_end"])
    shortened = sum(1 for stat in combo_stats if stat["actual_len"] < stat["full_len"])
    successful = sum(1 for stat in combo_stats if stat["reached_far"])
    return {
        "trail_pixels": int(np.count_nonzero(recolored)),
        "trail_colors": len(painted_colors),
        "length_variants": len(set(lengths)),
        "length_spread": max(lengths) - min(lengths) if lengths else 0,
        "blocked_routes": int(blocked),
        "rerouted_routes": int(rerouted),
        "shortened_routes": int(shortened),
        "successful_routes": int(successful),
        "combo_diff_pixels": int(np.count_nonzero(combo_out != d3_out)),
    }


def _gen_attempt():
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            break

    crossovers = _decompose_to_crossovers(perm)
    n_xovers = len(crossovers)
    if n_xovers == 0:
        return None

    min_wire_rows = n_xovers * 2 + 2
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (n_xovers + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if result is not None:
            placed_rects.append(result)

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        out = main(inp)
    except Exception:
        return None

    if not isinstance(out, np.ndarray) or out.shape != inp.shape:
        return None

    metrics = _combo_metrics(inp)
    if metrics["trail_pixels"] < MIN_TRAIL_PIXELS:
        return None
    if metrics["trail_colors"] < MIN_TRAIL_COLORS:
        return None
    if metrics["length_variants"] < MIN_LENGTH_VARIANTS:
        return None
    if metrics["length_spread"] < MIN_LENGTH_SPREAD:
        return None
    if metrics["shortened_routes"] < MIN_ROUTE_EFFECTS:
        return None
    if metrics["blocked_routes"] + metrics["rerouted_routes"] < MIN_ROUTE_EFFECTS:
        return None
    if metrics["successful_routes"] < MIN_SUCCESSFUL_ROUTES:
        return None
    if metrics["combo_diff_pixels"] < MIN_COMBO_DIFF_PIXELS:
        return None
    return inp


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
