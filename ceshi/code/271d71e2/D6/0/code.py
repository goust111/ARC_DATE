# Variant of 271d71e2
# Dimension: D6 — 容器与目标的基数关系 (meta)
# Change: 1:1 — 每个容器恰好对应一条目标线，颜色相同（均为maroon/9）... → 多容器共享一条目标线 (many:1)
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     original: 每个容器独立朝自己的 9 目标线移动，最近同形同色目标就是它的终点。
#     variant: 同一车道上的 2-4 个容器共享一条 9 目标线。所有容器仍沿头部方向移动、
#       仍按灰色像素数充能，但它们不能重叠到同一目标线上；离目标更近的容器先占据前排，
#       后面的容器依次贴在前一个容器后面，形成 convoy / queue。
#   concrete_grid_form:
#     - 颜色、边框、内部橙/灰填充、头线样式均保持原题不变。
#     - 整组容器共用同一方向、同一车道、同一头线长度，因此可以共享同一条 9 目标线。
#     - 垂直运动时所有容器共享同一宽度；水平运动时共享同一高度。
#     - 输出中前排容器的头线贴到目标线上；后排容器的头线贴到前排容器的远侧边，形成连续堆叠。
#   behavior_trace:
#     example_upward:
#       original_logic:
#         - 两个向上的容器若看到同一条 9 线，旧 solver 会让它们都把 head 对齐到那条 9 线，
#           因为它对每个容器独立地选“最近同形目标”，完全不考虑别的容器已占位。
#         - 结果会发生重叠覆盖，这正是旧 1:1 假设“恰好无害”的地方。
#       variant_logic:
#         - 先按 head 到共享目标的距离排序。
#         - 最近的容器先移动到目标线处。
#         - 第二个容器的可停靠 slot 不再是目标线，而是“第一个容器后方紧贴的位置”。
#         - 它的实际步数 = min(到该 slot 的距离, 原目标距离, 灰色像素数)。
#       visible_difference:
#         - 输入里只有一条 9 目标线，但有多个容器朝它移动。
#         - 输出里出现明显的排队堆叠，而不是多个容器各自贴到不同终点。
#   complexity_calibration:
#     - 不引入新的颜色、对象类别或多种优先级规则；优先级只由“谁离共享目标更近”决定。
#     - 为避免同一目标线与不同容器长度不匹配，生成器约束同组容器的 head span 相同。
#     - 仍保留容器内部橙/灰渐进充能，使任务看起来仍然属于原题家族，而不是全新题。
#   old_solver_assumptions_that_break:
#     - `for d in comp_by_color[hc]: ... dests.append(d)`：旧代码把所有同色同形组件都当作可能目的地，
#       many:1 下会把“别的容器的 head 线”错认成自己的目标，因此现在只允许“不邻接任何黑边”的组件做独立目标。
#     - `best_dest = nearest aligned dest`：旧代码默认每个容器各自对齐到同一个终点也不会冲突；
#       many:1 下必须先分组，再顺序分配 slot。
#     - 旧代码一边遍历一边擦除/粘贴容器，在共享车道情况下处理顺序会影响结果；
#       新 solver 先计算所有 move plan，再统一擦除和重绘。
#   placement_legality:
#     - 生成器使用 output-first：先构造最终队列，再把每个容器按各自步数反推回输入位置。
#     - 输入中相邻容器的步数严格递增至少 2，因此不同输入对象之间至少隔 1 个背景格，不相互接触。
#     - 输出中允许队列式贴靠；这是 many:1 规则本身的一部分。
#   observability:
#     - 每个样例都至少有 2 个移动容器共享 1 条目标线。
#     - 方向在上下左右间变化，但同一张图只使用一个方向和一条共享目标线，视觉上很整洁。
#     - 人类可直接从“单目标 + 多容器 + 输出排队”推断新规则，不依赖颜色数值或隐藏排序。
#   generator_guarantees_for_new_solver:
#     - 同组容器方向一致、head shape 一致、共享同一目标线、到目标距离严格递增。
#     - 每个容器灰色像素数 >= 它需要移动到分配 slot 的步数，因此 many:1 行为确实由排队规则触发。

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _box_tl_from_head(head_tl: tuple[int, int], outer_h: int, outer_w: int, head_side: int) -> tuple[int, int]:
    hr, hc = head_tl
    if head_side == 0:
        return hr + 1, hc
    if head_side == 1:
        return hr - outer_h, hc
    if head_side == 2:
        return hr, hc + 1
    return hr, hc - outer_w


def _draw_container(
    grid: np.ndarray,
    box_tl: tuple[int, int],
    outer_h: int,
    outer_w: int,
    head_side: int,
    inner_h: int,
    inner_w: int,
    orange_count: int,
):
    cr, cc = box_tl
    grid[cr, cc : cc + outer_w] = BORDER
    grid[cr + outer_h - 1, cc : cc + outer_w] = BORDER
    grid[cr : cr + outer_h, cc] = BORDER
    grid[cr : cr + outer_h, cc + outer_w - 1] = BORDER

    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    fill_order = _interior_fill_order(inner_h, inner_w, head_side)
    for idx in range(int(orange_count)):
        rr, cc2 = fill_order[idx]
        interior[rr, cc2] = ORANGE
    grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior


def _draw_head(grid: np.ndarray, head_tl: tuple[int, int], span: int, head_side: int):
    hr, hc = head_tl
    if head_side in (0, 1):
        grid[hr, hc : hc + span] = HEAD_COLOR
    else:
        grid[hr : hr + span, hc] = HEAD_COLOR


def _next_slot_tl(direction: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if direction == (-1, 0):
        return (r1 + 1, c0)
    if direction == (1, 0):
        return (r0 - 1, c0)
    if direction == (0, -1):
        return (r0, c1 + 1)
    return (r0, c0 - 1)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_head_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            adj_border = False
            for r, c in pts:
                for dr, dc in DIR4:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
                        adj_border = True
                        break
                if adj_border:
                    break
            comp_by_color[int(col)].append(
                {
                    "pts": pts,
                    "shape": shape,
                    "tl": (int(r0), int(c0)),
                    "adj_border": adj_border,
                }
            )

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == int(hc)}
        comp0 = None
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

        border_pts = list(_rect_border_coords(r0, c0, r1, c1))
        best_d = None
        dist = {}
        for pr, pc in comp0["pts"]:
            d = min(abs(int(pr - br)) + abs(int(pc - bc)) for br, bc in border_pts)
            dist[(int(pr), int(pc))] = int(d)
            if best_d is None or d < best_d:
                best_d = int(d)
        if best_d is None:
            return None
        head_pts = {p for p, d in dist.items() if int(d) == int(best_d)}

        coords = np.array(list(head_pts), dtype=int)
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        if hc1 < c0:
            direction = (0, -1)
        elif hc0 > c1:
            direction = (0, 1)
        elif hr1 < r0:
            direction = (-1, 0)
        elif hr0 > r1:
            direction = (1, 0)
        else:
            direction = (0, 0)

        r00, c00 = int(hr0), int(hc0)
        head_shape = {(int(r - r00), int(c - c00)) for r, c in head_pts}
        return {
            "color": hc,
            "head_pts": head_pts,
            "head_shape": head_shape,
            "head_tl": (r00, c00),
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
        }

    infos = []
    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
        if head_info is None or head_info["dir"] == (0, 0):
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape or d["pts"] == comp_pts or d["adj_border"]:
                continue
            dests.append(d)

        rem = comp_pts - head_pts
        if rem:
            mask_rem = np.zeros((H, W), dtype=bool)
            for rr, cc in rem:
                mask_rem[int(rr), int(cc)] = True
            labr, nr = label(mask_rem, structure=STRUCT4)
            for k in range(1, int(nr) + 1):
                pts = {(int(r), int(c)) for r, c in np.argwhere(labr == int(k))}
                if not pts:
                    continue
                adj = False
                for r, c in pts:
                    for dr, dc in DIR4:
                        rr, cc = int(r + dr), int(c + dc)
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == BORDER:
                            adj = True
                            break
                    if adj:
                        break
                if adj:
                    continue
                rs = [r for r, _c in pts]
                cs = [c for _r, c in pts]
                r00, c00 = int(min(rs)), int(min(cs))
                shp = {(int(r - r00), int(c - c00)) for r, c in pts}
                if shp != head_shape:
                    continue
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00), "adj_border": False})

        best = None
        best_dest = None
        for d in dests:
            tr0 = int(d["tl"][0] - head_tl[0])
            tc0 = int(d["tl"][1] - head_tl[1])
            if direction[0] != 0 and tc0 != 0:
                continue
            if direction[1] != 0 and tr0 != 0:
                continue
            if direction[0] == 1 and tr0 <= 0:
                continue
            if direction[0] == -1 and tr0 >= 0:
                continue
            if direction[1] == 1 and tc0 <= 0:
                continue
            if direction[1] == -1 and tc0 >= 0:
                continue
            dist = abs(tr0) + abs(tc0)
            if best is None or dist < best:
                best = dist
                best_dest = d
        if best_dest is None:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        infos.append(
            {
                "box": (r0, c0, r1, c1),
                "head_pts": head_pts,
                "head_tl": head_tl,
                "head_shape": head_shape,
                "dir": direction,
                "color": hc,
                "dest_tl": tuple(best_dest["tl"]),
                "dest_dist": int(best),
                "grey_coords": grey_coords,
            }
        )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for info in infos:
        shape_key = tuple(sorted(info["head_shape"]))
        groups[(info["color"], info["dir"], shape_key, info["dest_tl"])].append(info)

    move_plans = []
    for group in groups.values():
        group.sort(key=lambda info: (info["dest_dist"], info["head_tl"][0], info["head_tl"][1]))
        slot_tl = tuple(group[0]["dest_tl"])
        for info in group:
            r0, c0, r1, c1 = info["box"]
            head_tl = info["head_tl"]
            direction = info["dir"]
            grey_coords = list(info["grey_coords"])

            required = abs(int(slot_tl[0] - head_tl[0])) + abs(int(slot_tl[1] - head_tl[1]))
            steps = min(int(required), int(info["dest_dist"]), int(len(grey_coords)))
            dr, dc = direction
            tr = int(dr * steps)
            tc = int(dc * steps)

            if direction == (0, 1):
                order = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
            elif direction == (0, -1):
                order = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
            elif direction == (1, 0):
                order = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
            else:
                order = sorted(grey_coords, key=lambda p: (p[0], p[1]))
            to_fill = set(order[:steps])

            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            for rr, cc in to_fill:
                pr = int(rr - r0)
                pc = int(cc - c0)
                if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                    patch[pr, pc] = ORANGE

            new_box = (int(r0 + tr), int(c0 + tc), int(r1 + tr), int(c1 + tc))
            move_plans.append(
                {
                    "box": (r0, c0, r1, c1),
                    "patch": patch,
                    "new_box": new_box,
                    "head_pts": set(info["head_pts"]),
                    "color": int(info["color"]),
                    "tr": tr,
                    "tc": tc,
                }
            )
            slot_tl = _next_slot_tl(direction, new_box)

    out = g.copy()
    for plan in move_plans:
        r0, c0, r1, c1 = plan["box"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in plan["head_pts"]:
            if int(out[int(r), int(c)]) == int(plan["color"]):
                out[int(r), int(c)] = int(bg)

    for plan in move_plans:
        nr0, nc0, _nr1, _nc1 = plan["new_box"]
        paste(out, plan["patch"], nr0, nc0, clip=True)

    for plan in move_plans:
        for r, c in plan["head_pts"]:
            rr = int(r + plan["tr"])
            cc = int(c + plan["tc"])
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(plan["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.choice([2, 3, 3, 4])
    head_side = random.randint(0, 3)

    common_span = random.randint(4, 5 if n_containers == 4 else 6)
    fixed_inner = common_span - 2

    steps = []
    step0 = 1 if n_containers == 4 else random.randint(1, 2)
    steps.append(step0)
    while len(steps) < n_containers:
        step_gap = 2 if n_containers == 4 else random.randint(2, 3)
        steps.append(steps[-1] + step_gap)

    specs = []
    for step in steps:
        if head_side in (0, 1):
            inner_w = fixed_inner
            inner_h_cap = 5 if n_containers == 4 else 7
            inner_h_min = max(1, (step + inner_w - 1) // inner_w)
            if inner_h_min > inner_h_cap:
                return None
            inner_h_hi = min(inner_h_cap, inner_h_min + (1 if n_containers == 4 else 3))
            inner_h = random.randint(inner_h_min, inner_h_hi)
        else:
            inner_h = fixed_inner
            inner_w_cap = 4 if n_containers >= 3 else 5
            inner_w_min = max(1, (step + inner_h - 1) // inner_h)
            if inner_w_min > inner_w_cap:
                return None
            inner_w_hi = min(inner_w_cap, inner_w_min + (1 if n_containers == 4 else 2))
            inner_w = random.randint(inner_w_min, inner_w_hi)

        inner_area = inner_h * inner_w
        grey = random.randint(step, inner_area)
        initial_orange = inner_area - grey
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "step": int(step),
                "grey": int(grey),
                "initial_orange": int(initial_orange),
            }
        )

    final_head_tl = []
    slot = (0, 0)
    direction = SIDE_TO_DIR[head_side]
    for spec in specs:
        final_head_tl.append(slot)
        box_tl = _box_tl_from_head(slot, spec["outer_h"], spec["outer_w"], head_side)
        box = (
            int(box_tl[0]),
            int(box_tl[1]),
            int(box_tl[0] + spec["outer_h"] - 1),
            int(box_tl[1] + spec["outer_w"] - 1),
        )
        slot = _next_slot_tl(direction, box)

    for spec, head_tl in zip(specs, final_head_tl):
        dr, dc = direction
        spec["final_head_tl"] = tuple(head_tl)
        spec["input_head_tl"] = (int(head_tl[0] - dr * spec["step"]), int(head_tl[1] - dc * spec["step"]))

    dest_tl = final_head_tl[0]
    span = common_span

    min_r = 10**9
    min_c = 10**9
    max_r = -10**9
    max_c = -10**9

    def include_line(head_tl_: tuple[int, int]):
        nonlocal min_r, min_c, max_r, max_c
        hr, hc = head_tl_
        if head_side in (0, 1):
            min_r = min(min_r, hr)
            max_r = max(max_r, hr)
            min_c = min(min_c, hc)
            max_c = max(max_c, hc + span - 1)
        else:
            min_r = min(min_r, hr)
            max_r = max(max_r, hr + span - 1)
            min_c = min(min_c, hc)
            max_c = max(max_c, hc)

    include_line(dest_tl)
    for spec in specs:
        include_line(spec["input_head_tl"])
        br, bc = _box_tl_from_head(spec["input_head_tl"], spec["outer_h"], spec["outer_w"], head_side)
        min_r = min(min_r, br)
        max_r = max(max_r, br + spec["outer_h"] - 1)
        min_c = min(min_c, bc)
        max_c = max(max_c, bc + spec["outer_w"] - 1)

    obj_h = max_r - min_r + 1
    obj_w = max_c - min_c + 1
    if obj_h > 26 or obj_w > 26:
        return None

    grid_h = random.randint(max(8, obj_h), min(26, obj_h + 4))
    grid_w = random.randint(max(8, obj_w), min(26, obj_w + 4))
    shift_r = random.randint(0, grid_h - obj_h) - min_r
    shift_c = random.randint(0, grid_w - obj_w) - min_c

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    shifted_dest = (int(dest_tl[0] + shift_r), int(dest_tl[1] + shift_c))
    _draw_head(expected, shifted_dest, span, head_side)
    _draw_head(grid, shifted_dest, span, head_side)

    for spec in specs:
        input_head = (int(spec["input_head_tl"][0] + shift_r), int(spec["input_head_tl"][1] + shift_c))
        final_head = (int(spec["final_head_tl"][0] + shift_r), int(spec["final_head_tl"][1] + shift_c))

        input_box = _box_tl_from_head(input_head, spec["outer_h"], spec["outer_w"], head_side)
        final_box = _box_tl_from_head(final_head, spec["outer_h"], spec["outer_w"], head_side)

        _draw_container(
            grid,
            input_box,
            spec["outer_h"],
            spec["outer_w"],
            head_side,
            spec["inner_h"],
            spec["inner_w"],
            spec["initial_orange"],
        )
        _draw_head(grid, input_head, span, head_side)

        _draw_container(
            expected,
            final_box,
            spec["outer_h"],
            spec["outer_w"],
            head_side,
            spec["inner_h"],
            spec["inner_w"],
            spec["initial_orange"] + spec["step"],
        )
        _draw_head(expected, final_head, span, head_side)

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
