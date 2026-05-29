# Variant of 271d71e2
# Dimension: D1 — 路径方向类型 (path)
# Change: straight_ray — 容器沿头部指示的四个基本方向（上/下/左/右）直线平移... → L形两段路径
#
# Design analysis
# ===============
# variant_spec:
#   source_task_note:
#     - 提示引用 e87109e9 作为 turning_rules 灵感来源.
#     - 当前工作区可读到 `/ARC-data/.../e87109e9.json`, 但不存在可读取的 solver `.py`,
#       因此无法直接复用其代码实现; 下述设计仅依据本题描述完成.
#   core_change:
#     - 原题每个容器只有 1 条同色目标线, 头线沿单轴直线移动到那条线.
#     - 本变体改成 2 条同色目标线, 都与头线同形:
#       1) 第一条线和头线在第一段轴上对齐, 表示转折点;
#       2) 第二条线再从转折点沿正交轴偏移, 表示终点.
#     - 路径总长度 = 第一段距离 + 第二段距离.
#     - 实际移动步数 = min(路径总长度, 灰色像素数).
#     - 灰色像素仍然只按“头侧向内”的原顺序变橙, 只是现在这些步数会先消耗第一段,
#       剩余步数再消耗第二段.
#   concrete_pixel_geometry:
#     - 头线仍紧贴容器某一边, 颜色 9, 长度等于对应边长度.
#     - 两条目标线也使用颜色 9, 与头线同方向同长度.
#     - 若头在线容器上边/下边, 第一段必为竖直(沿头朝外), 第二段为水平.
#     - 若头在线容器左边/右边, 第一段必为水平(沿头朝外), 第二段为竖直.
#     - 因而“先水平后竖直 / 先竖直后水平”由头所在边决定; 第二段的正负方向由
#       第二条目标线相对第一条目标线的位置决定.
#   behavior_trace_example:
#     - 原题: 右侧头线, 右方只有 1 条目标线, gap=4, grey=3 => 向右走 3 步.
#     - 变体: 右侧头线, 先在右方 4 格处有转折线, 再在该线正上方 7 格处有终点线.
#       路径总长 11.
#       * 若 grey=2: 只走第一段前 2 步, 输出仍停在转折前.
#       * 若 grey>=11: 先走满 4 步到转折, 再向上走 7 步到终点.
#     - 差异是 solver 不能再用“head 与单个 dest 的曼哈顿距离”; 必须识别有序线对,
#       先分配给第一段, 再分配给第二段.
#   complexity_calibration:
#     - 保留原来的容器、颜色、灰橙计步规则, 只新增一个同形目标线和转折逻辑.
#     - 为避免视觉上看不出变体, 生成器保证至少有一个 mover 完整走完 L 路径, 同时
#       常常再有一个 mover 在转弯前耗尽灰格, 让“总步数按两段分配”可观察.
#     - 不生成“第二段走到一半”的训练样例: 这种情况下最终头线会与转折线部分重叠,
#       对人眼不整洁; 这不是 D1 的核心, 所以生成器主动避开.
#   implicit_assumption_audit_main:
#     - 原 solver 依赖“唯一目标线与头线共轴”. 相关代码是:
#       * 仅接受与 head_tl 单轴对齐的 dest;
#       * 步数直接取 head->dest 曼哈顿距离;
#       * 平移向量 = direction * steps.
#     - 在本变体中这些前提不成立:
#       * 需要先找第一段目标线 turn, 再找第二段目标线 final;
#       * 总距离 = dist(turn) + dist(turn->final);
#       * 平移向量分段累积, 不能再一次性 direction * steps.
#     - 其余容器检测、背景推断、灰格充能顺序仍可复用.
#   placement_legality:
#     - 生成器按“整个输入对象 + 两条目标线 + 完整移动后容器/头线”的外接矩形保留
#       装配区, 装配区之间再留 1 格背景, 从而保证:
#       * 输入对象不重叠;
#       * 输出状态彼此也不重叠;
#       * L 路径经过的空间属于本装配区, solver 的匹配不会串组.
#     - 同时强制各容器的头线形状(方向+长度)互异, 减少跨容器误配.
#   visual_observability:
#     - 同一容器会看到 3 条同色同形线: 邻边的是头线, 远处同轴的是转折线,
#       再正交偏移的是终点线. 多个样例下, 人类可直接看出“先到同轴线, 再拐弯到另一线”.
#     - 颜色变化仍然只有灰->橙一种, 结构线索来自位置和路径, 不依赖颜色数值排序.
#   cleanliness:
#     - 为保持转折线与终点线分离, 第二段长度至少比头线长度大 1.
#     - 第一段长度至少 3; “转弯前耗尽”时只允许走到 first_dist-2 以内, 避免最终头线
#       与转折线连成一体.

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

# head_side -> outward first-segment direction
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _head_side_from_dir(direction: tuple[int, int]) -> int:
    for side, vec in SIDE_TO_DIR.items():
        if vec == direction:
            return side
    raise ValueError(direction)


def _interior_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if head_side == 3:  # right
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == 2:  # left
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == 1:  # bottom
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:  # top
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _head_tl_for_container(r0: int, c0: int, r1: int, c1: int, head_side: int) -> tuple[int, int]:
    if head_side == 0:
        return (r0 - 1, c0)
    if head_side == 1:
        return (r1 + 1, c0)
    if head_side == 2:
        return (r0, c0 - 1)
    return (r0, c1 + 1)


def _line_bbox_from_tl(tl: tuple[int, int], head_side: int, outer_h: int, outer_w: int) -> tuple[int, int, int, int]:
    r, c = tl
    if head_side in (0, 1):
        return (r, c, r, c + outer_w - 1)
    return (r, c, r + outer_h - 1, c)


def _translate_along_l(
    first_dir: tuple[int, int],
    second_dir: tuple[int, int],
    first_dist: int,
    steps: int,
) -> tuple[int, int]:
    if steps <= first_dist:
        return (first_dir[0] * steps, first_dir[1] * steps)
    rem = steps - first_dist
    return (
        first_dir[0] * first_dist + second_dir[0] * rem,
        first_dir[1] * first_dist + second_dir[1] * rem,
    )


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != BORDER), key=lambda c: int(cnt[int(c)])))

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

    out = g.copy()
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
            comp_by_color[int(col)].append({"pts": pts, "shape": shape, "tl": (int(r0), int(c0))})

    def pick_head_for_container(box_: tuple[int, int, int, int]):
        r0, c0, r1, c1 = box_
        head_seeds = []
        for r, c in _rect_border_coords(r0, c0, r1, c1):
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    val = int(g[rr, cc])
                    if val not in (int(bg), BORDER, GREY, ORANGE):
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

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container(cont["box"])
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        first_dir = head_info["dir"]
        if first_dir == (0, 0):
            continue

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape:
                continue
            if d["pts"] == comp_pts:
                continue
            dests.append(d)
        if len(dests) < 2:
            continue

        best_key = None
        best_path = None
        for turn in dests:
            dr1 = int(turn["tl"][0] - head_tl[0])
            dc1 = int(turn["tl"][1] - head_tl[1])
            if first_dir[0] != 0:
                if dc1 != 0 or dr1 * first_dir[0] <= 0:
                    continue
                first_dist = abs(dr1)
                second_axis_vertical = False
            else:
                if dr1 != 0 or dc1 * first_dir[1] <= 0:
                    continue
                first_dist = abs(dc1)
                second_axis_vertical = True

            for final in dests:
                if final is turn:
                    continue
                dr2 = int(final["tl"][0] - turn["tl"][0])
                dc2 = int(final["tl"][1] - turn["tl"][1])
                if second_axis_vertical:
                    if dc2 != 0 or dr2 == 0:
                        continue
                    second_dir = (1 if dr2 > 0 else -1, 0)
                    second_dist = abs(dr2)
                else:
                    if dr2 != 0 or dc2 == 0:
                        continue
                    second_dir = (0, 1 if dc2 > 0 else -1)
                    second_dist = abs(dc2)

                total_dist = int(first_dist + second_dist)
                key = (total_dist, first_dist, second_dist, abs(dr2) + abs(dc2))
                if best_key is None or key < best_key:
                    best_key = key
                    best_path = (turn, final, first_dist, second_dir, second_dist, total_dist)

        if best_path is None:
            continue

        _turn, _final, first_dist, second_dir, _second_dist, total_dist = best_path
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(total_dist), int(len(grey_coords)))
        tr, tc = _translate_along_l(first_dir, second_dir, first_dist, steps)

        if first_dir == (0, 1):
            ordered_greys = sorted(grey_coords, key=lambda p: (-p[1], p[0]))
        elif first_dir == (0, -1):
            ordered_greys = sorted(grey_coords, key=lambda p: (p[1], -p[0]))
        elif first_dir == (1, 0):
            ordered_greys = sorted(grey_coords, key=lambda p: (-p[0], -p[1]))
        else:
            ordered_greys = sorted(grey_coords, key=lambda p: (p[0], p[1]))
        to_fill = set(ordered_greys[:steps])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(hc)

    return out


def _build_spec(mode: str, used_shape_keys: set[tuple[str, int]]) -> dict | None:
    for _ in range(300):
        inner_h = random.randint(2, 6)
        inner_w = random.randint(2, 4)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        head_side = random.randint(0, 3)
        shape_key = ("h", outer_w) if head_side in (0, 1) else ("v", outer_h)
        if shape_key in used_shape_keys:
            continue

        first_dir = SIDE_TO_DIR[head_side]
        if first_dir[0] != 0:
            second_dir = (0, random.choice([-1, 1]))
            second_min = outer_w + 1
        else:
            second_dir = (random.choice([-1, 1]), 0)
            second_min = outer_h + 1

        inner_area = inner_h * inner_w
        if mode == "full":
            if inner_area < second_min + 3:
                continue
            first_max = min(5, inner_area - second_min)
            if first_max < 3:
                continue
            first_dist = random.randint(3, first_max)
            second_max = min(second_min + 3, inner_area - first_dist)
            if second_max < second_min:
                continue
            second_dist = random.randint(second_min, second_max)
            total_dist = first_dist + second_dist
            grey = random.randint(total_dist, inner_area)
        elif mode == "pre_turn":
            first_dist = random.randint(3, 5)
            second_dist = random.randint(second_min, second_min + 3)
            grey_cap = min(inner_area, first_dist - 2)
            if grey_cap < 1:
                continue
            grey = random.randint(1, grey_cap)
            total_dist = first_dist + second_dist
        else:  # decoy
            first_dist = random.randint(3, 5)
            second_dist = random.randint(second_min, second_min + 3)
            grey = 0
            total_dist = first_dist + second_dist

        initial_orange = inner_area - grey
        used_shape_keys.add(shape_key)
        return {
            "mode": mode,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "first_dir": first_dir,
            "second_dir": second_dir,
            "first_dist": first_dist,
            "second_dist": second_dist,
            "total_dist": total_dist,
            "grey": grey,
            "initial_orange": initial_orange,
        }
    return None


def _decorate_spec_geometry(spec: dict) -> dict:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    fd = spec["first_dir"]
    sd = spec["second_dir"]
    d1 = spec["first_dist"]
    d2 = spec["second_dist"]

    cont_box = (0, 0, oh - 1, ow - 1)
    head_tl = _head_tl_for_container(*cont_box, hs)
    turn_tl = (head_tl[0] + fd[0] * d1, head_tl[1] + fd[1] * d1)
    final_tl = (turn_tl[0] + sd[0] * d2, turn_tl[1] + sd[1] * d2)
    final_disp = (fd[0] * d1 + sd[0] * d2, fd[1] * d1 + sd[1] * d2)
    moved_cont_box = (final_disp[0], final_disp[1], final_disp[0] + oh - 1, final_disp[1] + ow - 1)

    boxes = [
        cont_box,
        _line_bbox_from_tl(head_tl, hs, oh, ow),
        _line_bbox_from_tl(turn_tl, hs, oh, ow),
        _line_bbox_from_tl(final_tl, hs, oh, ow),
        moved_cont_box,
        _line_bbox_from_tl((head_tl[0] + final_disp[0], head_tl[1] + final_disp[1]), hs, oh, ow),
    ]
    min_r = min(b[0] for b in boxes)
    min_c = min(b[1] for b in boxes)
    max_r = max(b[2] for b in boxes)
    max_c = max(b[3] for b in boxes)
    shift_r = -min_r
    shift_c = -min_c

    return {
        **spec,
        "assembly_h": max_r - min_r + 1,
        "assembly_w": max_c - min_c + 1,
        "cont_r": shift_r,
        "cont_c": shift_c,
        "head_tl": (head_tl[0] + shift_r, head_tl[1] + shift_c),
        "turn_tl": (turn_tl[0] + shift_r, turn_tl[1] + shift_c),
        "final_tl": (final_tl[0] + shift_r, final_tl[1] + shift_c),
    }


def _draw_line(grid: np.ndarray, tl: tuple[int, int], head_side: int, outer_h: int, outer_w: int, color: int):
    r, c = tl
    if head_side in (0, 1):
        grid[r, c : c + outer_w] = color
    else:
        grid[r : r + outer_h, c] = color


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out = []
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def _pack_rows(specs: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    n = len(specs)
    order = list(range(n))
    random.shuffle(order)
    ordered = [specs[i] for i in order]
    gap = 1

    fits = []
    for row_count in range(1, n + 1):
        for comp in _compositions(n, row_count):
            rows = []
            idx = 0
            for size in comp:
                rows.append(ordered[idx : idx + size])
                idx += size
            row_heights = [max(s["assembly_h"] for s in row) for row in rows]
            row_widths = [sum(s["assembly_w"] for s in row) + gap * (len(row) - 1) for row in rows]
            need_h = sum(row_heights) + gap * (len(rows) - 1)
            need_w = max(row_widths)
            if need_h + 2 > 26 or need_w + 2 > 26:
                continue
            fits.append((need_h, need_w, rows, row_heights, row_widths))
    if not fits:
        return None

    need_h, need_w, rows, row_heights, row_widths = min(fits, key=lambda x: (x[0] * x[1], x[0] + x[1]))
    grid_h = random.randint(need_h + 2, 26)
    grid_w = random.randint(need_w + 2, 26)
    top_margin = random.randint(0, grid_h - need_h)

    placements_by_id: dict[int, tuple[int, int]] = {}
    r = top_margin
    for row, row_h, row_w in zip(rows, row_heights, row_widths):
        left_margin = random.randint(0, grid_w - row_w)
        c = left_margin
        for spec in row:
            placements_by_id[id(spec)] = (r, c)
            c += spec["assembly_w"] + gap
        r += row_h + gap

    placements = [placements_by_id[id(specs[i])] for i in range(n)]
    return grid_h, grid_w, placements


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    # L-path assemblies reserve both the input cluster and the full bent travel envelope,
    # so 2-3 containers are the practical range that still leaves clean 1-cell gaps.
    n_containers = random.randint(2, 3)
    n_decoys = 0 if n_containers == 2 else random.choice([0, 1])
    n_movers = n_containers - n_decoys
    if n_movers < 2:
        n_movers = 2
        n_decoys = n_containers - n_movers

    modes = ["full", "pre_turn"]
    while len(modes) < n_movers:
        modes.append(random.choice(["full", "pre_turn"]))
    modes.extend(["decoy"] * n_decoys)
    random.shuffle(modes)

    specs = []
    used_shape_keys: set[tuple[str, int]] = set()
    for mode in modes:
        spec = _build_spec(mode, used_shape_keys)
        if spec is None:
            return None
        specs.append(_decorate_spec_geometry(spec))

    packed = _pack_rows(specs)
    if packed is None:
        return None
    grid_h, grid_w, placements = packed

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    placements_info = []

    for spec, (pr, pc) in zip(specs, placements):
        oh, ow = spec["outer_h"], spec["outer_w"]
        ih, iw = spec["inner_h"], spec["inner_w"]
        hs = spec["head_side"]
        fd = spec["first_dir"]
        sd = spec["second_dir"]
        d1 = spec["first_dist"]
        d2 = spec["second_dist"]
        total_dist = spec["total_dist"]
        grey = spec["grey"]
        initial_orange = spec["initial_orange"]

        cr = pr + spec["cont_r"]
        cc = pc + spec["cont_c"]
        head_tl = (pr + spec["head_tl"][0], pc + spec["head_tl"][1])
        turn_tl = (pr + spec["turn_tl"][0], pc + spec["turn_tl"][1])
        final_tl = (pr + spec["final_tl"][0], pc + spec["final_tl"][1])

        patch = np.full((oh, ow), BORDER, dtype=int)
        interior = np.full((ih, iw), GREY, dtype=int)
        for idx, (rr, cc2) in enumerate(_interior_fill_order(ih, iw, hs)):
            if idx < initial_orange:
                interior[rr, cc2] = ORANGE
        patch[1:-1, 1:-1] = interior
        grid[cr : cr + oh, cc : cc + ow] = patch
        _draw_line(grid, head_tl, hs, oh, ow, HEAD_COLOR)
        _draw_line(grid, turn_tl, hs, oh, ow, HEAD_COLOR)
        _draw_line(grid, final_tl, hs, oh, ow, HEAD_COLOR)
        placements_info.append(
            {
                "spec": spec,
                "cr": cr,
                "cc": cc,
                "head_tl": head_tl,
                "turn_tl": turn_tl,
                "final_tl": final_tl,
            }
        )

    expected = grid.copy()
    for info in placements_info:
        spec = info["spec"]
        oh, ow = spec["outer_h"], spec["outer_w"]
        ih, iw = spec["inner_h"], spec["inner_w"]
        hs = spec["head_side"]
        fd = spec["first_dir"]
        sd = spec["second_dir"]
        d1 = spec["first_dist"]
        total_dist = spec["total_dist"]
        grey = spec["grey"]
        initial_orange = spec["initial_orange"]
        cr = info["cr"]
        cc = info["cc"]
        head_tl = info["head_tl"]

        steps = min(total_dist, grey)
        if steps == 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = BG
        _draw_line(expected, head_tl, hs, oh, ow, BG)

        tr, tc = _translate_along_l(fd, sd, d1, steps)
        moved_patch = np.full((oh, ow), BORDER, dtype=int)
        moved_interior = np.full((ih, iw), GREY, dtype=int)
        for idx, (rr, cc2) in enumerate(_interior_fill_order(ih, iw, hs)):
            if idx < initial_orange + steps:
                moved_interior[rr, cc2] = ORANGE
        moved_patch[1:-1, 1:-1] = moved_interior

        nr = cr + tr
        nc = cc + tc
        expected[nr : nr + oh, nc : nc + ow] = moved_patch
        moved_head_tl = (head_tl[0] + tr, head_tl[1] + tc)
        _draw_line(expected, moved_head_tl, hs, oh, ow, HEAD_COLOR)

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None
    return grid, expected, specs


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("Failed to generate valid after 500 attempts")
