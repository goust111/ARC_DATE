# Variant of 7ed72f31
# Dimension: D8 — 多标记组的处理顺序 (multi_step)
# Change: 各标记组独立处理，无特定顺序要求，结果直接叠加... → 从外到内依次处理

from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg, label_mask


# Design analysis for the D8 variant
# ==================================
# a) Concrete behavior
#    原题里每个红色标记组都独立生成一批“待补上的镜像像素”，solver 按连通块扫描顺序
#    直接写入 out，因此重叠时谁最后被扫描到谁就赢，但题面没有显式顺序。
#    本变体把“写入顺序”显式化为：按标记组到整张网格中心的距离，从外到内依次处理。
#    距离更近的组更晚落笔，因此在重叠镜像区会覆盖更外层组的颜色。
#    这里把“标记组的位置”定义为该组红色标记像素的几何中心：
#    - 水平/垂直红线：整条红线的中心
#    - 单个红点：该点本身
#    距离使用到网格中心 ((H-1)/2, (W-1)/2) 的平方欧氏距离。
#
#    一个具体追踪例子：
#    - 组 A：较靠外的竖直红线，镜像后会在局部区域 P 画出颜色 7。
#    - 组 B：较靠内的水平红线，镜像后也会在同一区域 P 画出颜色 4。
#    原逻辑：如果扫描顺序恰好先 B 后 A，则 P 最终会变成 7；若先 A 后 B，则会变成 4。
#    新逻辑：无论输入里哪个组排在前面，都必须先处理更外的 A，再处理更内的 B，因此 P 最终为 4。
#
# b) Complexity calibration
#    为了让 D8 真正可见，生成器强制制造至少一对会把镜像结果落到同一片区域的标记组，
#    且两组颜色不同。这样“后处理可覆盖先处理”一定能在输出里留下痕迹。
#    同时不把自由度放得过多：重叠主对固定为一横一竖两个组，其他组只作不重叠补充。
#
# c) Implicit-assumption audit of the original solver
#    原 solver 中与 D8 直接相关的隐式前提是：
#    - `out = g.copy()` 后在两个大循环里直接写 `out[...] = v`
#      依赖前提：各组互不影响，或影响无关紧要。
#      在 D8 里这个前提失效，因为覆盖次序决定最终颜色。
#    - “混合组件先处理，孤立红点后处理”
#      原题无序叠加时这只是实现细节；D8 里如果保留这个两阶段顺序，就会错误地把
#      组件类型引入优先级。因此这里必须先抽取所有组的候选镜像像素，再统一排序落笔。
#    - 原代码用连通块扫描顺序作为默认写入顺序；在变体里这只能用于稳定 tie-break，
#      不能再决定最终覆盖关系。
#
# d) Placement legality
#    生成器保证：
#    - 输入中的各组原件彼此不重叠，且不同组之间没有 4 邻接，避免 solver 把它们并成同一组件。
#    - 所有镜像像素都在网格内。
#    - 镜像结果不落到任何输入原件/红标记上；只有设计好的重叠主对会共享部分镜像目标。
#
# e) Visual observability
#    观察者能看到：局部一块区域被两个不同方向的镜像规则同时指向，但最后只保留更靠近
#    网格中心的那组颜色。由于生成器还强制让“扫描顺序”和“距离顺序”相反，变体不会退化
#    成原始 solver 的偶然行为。
#
# f) Visual cleanliness
#    只允许一对主组发生镜像重叠；其余 filler 组的镜像区互不重叠，避免整张图到处覆盖。
#    重叠区域只发生在背景上，不压住原始精灵或红标记。
#
# g) Color inference
#    不引入额外的颜色映射。每组仍保持“原色 = 镜像色”，颜色只用于区分不同组并让覆盖结果可见。
#
# h) Generator validation metrics
#    生成器除 `main(inp)` 不报错外，还要求：
#    - 至少存在 1 个被两个以上标记组共同绘制的输出坐标
#    - 新顺序版输出与旧版（扫描顺序）输出至少有 1 个像素不同
#    - 各组到中心的距离没有近似平局，避免排序规则不明显
#
# i) New solver assumptions and generator support
#    新 solver 假设：
#    - 每个标记组都可被原规则唯一识别（生成器用 4-邻接隔离输入组件来保证）
#    - 距离排序在样例中可稳定比较（生成器避免距离接近/相等）
#    - 重叠主要发生在镜像副本之间，而不是镜像压到原件上（生成器显式检查）


MARK = 2
PALETTE = [0, 1, 3, 4, 5, 6, 7, 8, 9]


def _inb(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(H) and 0 <= int(c) < int(W)


def _refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, p)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _marker_distance(marker_coords: list[tuple[int, int]], H: int, W: int) -> float:
    mr = sum(int(r) for r, _c in marker_coords) / len(marker_coords)
    mc = sum(int(c) for _r, c in marker_coords) / len(marker_coords)
    cr = (int(H) - 1) / 2.0
    cc = (int(W) - 1) / 2.0
    return float((mr - cr) ** 2 + (mc - cc) ** 2)


def _extract_groups(g: np.ndarray) -> tuple[int, list[dict]]:
    g = np.array(g, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    lab, n = label_mask(g != int(bg), connectivity=4)
    comps = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )

    mixed_groups: list[dict] = []
    isolated_groups: list[dict] = []
    discovery = 0

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(vals):
            continue

        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == int(MARK)]
        payload = {int(v) for v in vals if int(v) != int(MARK)}

        if payload:
            rs = {int(r) for r, _c in reds}
            cs = {int(c) for _r, c in reds}
            if len(reds) == 1:
                axis = ("point", (int(reds[0][0]), int(reds[0][1])))
            elif len(rs) == 1:
                axis = ("row", int(next(iter(rs))))
            elif len(cs) == 1:
                axis = ("col", int(next(iter(cs))))
            else:
                continue

            r0, c0, r1, c1 = map(int, comp["bbox"])

            if axis[0] == "row":

                def eligible(r: int, c: int, v: int) -> bool:
                    return int(v) in payload and int(c0) <= int(c) <= int(c1)

            elif axis[0] == "col":

                def eligible(r: int, c: int, v: int) -> bool:
                    return int(v) in payload and int(r0) <= int(r) <= int(r1)

            else:
                rr0 = max(0, int(r0 - 1))
                rr1 = min(int(H - 1), int(r1 + 1))
                cc0 = max(0, int(c0 - 1))
                cc1 = min(int(W - 1), int(c1 + 1))

                def eligible(r: int, c: int, v: int) -> bool:
                    return int(v) in payload and int(rr0) <= int(r) <= int(rr1) and int(cc0) <= int(c) <= int(cc1)

            paint: list[tuple[int, int, int]] = []
            for r in range(int(H)):
                for c in range(int(W)):
                    v = int(g[r, c])
                    if v == int(bg) or v == int(MARK):
                        continue
                    if not eligible(int(r), int(c), int(v)):
                        continue
                    if axis[0] == "row":
                        rr, cc = _refl_row(int(axis[1]), int(r), int(c))
                    elif axis[0] == "col":
                        rr, cc = _refl_col(int(axis[1]), int(r), int(c))
                    else:
                        rr, cc = _refl_point(axis[1], int(r), int(c))
                    if _inb(H, W, int(rr), int(cc)):
                        paint.append((int(rr), int(cc), int(v)))

            if paint:
                mixed_groups.append(
                    {
                        "phase": 0,
                        "discovery": discovery,
                        "marker_coords": reds,
                        "marker_center": (
                            sum(r for r, _c in reds) / len(reds),
                            sum(c for _r, c in reds) / len(reds),
                        ),
                        "distance": _marker_distance(reds, H, W),
                        "paint": paint,
                    }
                )
                discovery += 1

        elif len(coords) == 1 and int(vals[0]) == int(MARK):
            pr, pc = map(int, coords[0])
            neigh_colors: set[int] = set()
            neigh_pixels: list[tuple[int, int, int]] = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = int(pr + dr), int(pc + dc)
                    if not _inb(H, W, rr, cc):
                        continue
                    v = int(g[rr, cc])
                    if v == int(bg) or v == int(MARK):
                        continue
                    neigh_colors.add(int(v))
                    neigh_pixels.append((int(rr), int(cc), int(v)))

            if not neigh_colors:
                continue

            paint: list[tuple[int, int, int]] = []
            for v in neigh_colors:
                labv, _nv = label_mask(g == int(v), connectivity=8)
                labs = {
                    int(labv[int(rr), int(cc)])
                    for rr, cc, vv in neigh_pixels
                    if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
                }
                for lbl in labs:
                    for r, c in np.argwhere(labv == int(lbl)):
                        rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                        if _inb(H, W, int(rr), int(cc)):
                            paint.append((int(rr), int(cc), int(v)))

            if paint:
                isolated_groups.append(
                    {
                        "phase": 1,
                        "discovery": discovery,
                        "marker_coords": [(int(pr), int(pc))],
                        "marker_center": (float(pr), float(pc)),
                        "distance": _marker_distance([(int(pr), int(pc))], H, W),
                        "paint": paint,
                    }
                )
                discovery += 1

    return bg, mixed_groups + isolated_groups


def _solve(grid: np.ndarray, *, ordered: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    _bg, groups = _extract_groups(g)
    out = g.copy()

    if ordered:
        groups = sorted(
            groups,
            key=lambda grp: (
                -float(grp["distance"]),
                float(grp["marker_center"][0]),
                float(grp["marker_center"][1]),
                int(grp["phase"]),
                int(grp["discovery"]),
            ),
        )
    else:
        groups = sorted(groups, key=lambda grp: (int(grp["phase"]), int(grp["discovery"])))

    for grp in groups:
        for r, c, v in grp["paint"]:
            out[int(r), int(c)] = int(v)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, ordered=True)


def _shift(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _group_from_pattern(pattern: dict, color: int, dr: int, dc: int) -> dict:
    reds = _shift(pattern["reds"], dr, dc)
    payload = _shift(pattern["payload"], dr, dc)
    paint = _shift(pattern["paint"], dr, dc)
    return {
        "reds": reds,
        "payload": payload,
        "paint": paint,
        "color": int(color),
    }


def _group_input_cells(group: dict) -> set[tuple[int, int]]:
    return set(group["reds"]) | set(group["payload"])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _can_place_input(group: dict, H: int, W: int, occupied_input: set[tuple[int, int]], reserved_paint: set[tuple[int, int]]) -> bool:
    cells = _group_input_cells(group)
    for r, c in cells:
        if not _inb(H, W, r, c):
            return False
        if (int(r), int(c)) in occupied_input or (int(r), int(c)) in reserved_paint:
            return False
        for rr, cc in _neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in occupied_input:
                return False
    return True


def _can_place_paint(group: dict, H: int, W: int, occupied_input: set[tuple[int, int]], occupied_paint: set[tuple[int, int]]) -> bool:
    for r, c in group["paint"]:
        if not _inb(H, W, r, c):
            return False
        if (int(r), int(c)) in occupied_input or (int(r), int(c)) in occupied_paint:
            return False
    return True


PAIR_NEAR = {
    "name": "pair_near_row",
    "reds": [(3, 4), (3, 5), (3, 6)],
    "payload": [(0, 5), (1, 4), (1, 5), (2, 4)],
    "paint": [(6, 5), (5, 4), (5, 5), (4, 4)],
}

PAIR_FAR = {
    "name": "pair_far_col",
    "reds": [(5, 3), (6, 3), (7, 3)],
    "payload": [(5, 2), (6, 1), (6, 2), (7, 0)],
    "paint": [(5, 4), (6, 5), (6, 4), (7, 6)],
}

FILLER_PATTERNS = [
    {
        "name": "row_top",
        "reds": [(3, 1), (3, 2), (3, 3)],
        "payload": [(0, 2), (1, 1), (1, 2), (2, 1)],
        "paint": [(6, 2), (5, 1), (5, 2), (4, 1)],
    },
    {
        "name": "col_left",
        "reds": [(1, 3), (2, 3), (3, 3)],
        "payload": [(1, 0), (2, 1), (2, 2), (3, 1)],
        "paint": [(1, 6), (2, 5), (2, 4), (3, 5)],
    },
    {
        "name": "point_mixed",
        "reds": [(2, 2)],
        "payload": [(1, 2), (2, 3), (3, 3)],
        "paint": [(3, 2), (2, 1), (1, 1)],
    },
    {
        "name": "isolated_point",
        "reds": [(2, 2)],
        "payload": [(0, 0), (0, 1), (1, 1)],
        "paint": [(4, 4), (4, 3), (3, 3)],
    },
]


def _paint_input(grid: np.ndarray, group: dict) -> None:
    color = int(group["color"])
    for r, c in group["reds"]:
        grid[int(r), int(c)] = int(MARK)
    for r, c in group["payload"]:
        grid[int(r), int(c)] = int(color)


def _variant_metrics(grid: np.ndarray) -> dict:
    ordered = main(grid)
    legacy = _solve(grid, ordered=False)
    _bg, groups = _extract_groups(grid)
    paint_count: dict[tuple[int, int], int] = {}
    for grp in groups:
        seen = set()
        for r, c, _v in grp["paint"]:
            key = (int(r), int(c))
            if key in seen:
                continue
            seen.add(key)
            paint_count[key] = paint_count.get(key, 0) + 1
    dists = [float(grp["distance"]) for grp in groups]
    dists_sorted = sorted(dists)
    min_gap = min((b - a) for a, b in zip(dists_sorted, dists_sorted[1:])) if len(dists_sorted) >= 2 else 999.0
    return {
        "ordered": ordered,
        "legacy": legacy,
        "overlap_cells": sum(1 for v in paint_count.values() if int(v) >= 2),
        "changed_vs_legacy": int(np.count_nonzero(ordered != legacy)),
        "group_count": len(groups),
        "min_dist_gap": float(min_gap),
    }


def generate_input() -> np.ndarray:
    for _attempt in range(400):
        H = random.randint(22, 26)
        W = random.randint(22, 26)
        bg = random.choice([c for c in PALETTE if int(c) != int(MARK)])

        num_groups = random.randint(3, 5)
        usable_colors = [c for c in PALETTE if int(c) not in {int(bg), int(MARK)}]
        random.shuffle(usable_colors)
        colors = usable_colors[:num_groups]

        grid = np.full((H, W), int(bg), dtype=int)
        groups: list[dict] = []
        occupied_input: set[tuple[int, int]] = set()
        occupied_paint: set[tuple[int, int]] = set()

        pair_placed = False
        for _pair_try in range(120):
            dr = random.randint(2, max(2, H - 11))
            dc = random.randint(1, max(1, W - 9))
            near = _group_from_pattern(PAIR_NEAR, colors[0], dr, dc)
            far = _group_from_pattern(PAIR_FAR, colors[1], dr, dc)

            near_input = _group_input_cells(near)
            far_input = _group_input_cells(far)
            pair_input = near_input | far_input
            pair_paint = set(near["paint"]) | set(far["paint"])
            overlap = set(near["paint"]) & set(far["paint"])
            if len(overlap) < 2:
                continue
            if any(cell in pair_paint for cell in pair_input):
                continue
            if not _can_place_input(near, H, W, set(), pair_paint):
                continue
            if not _can_place_input(far, H, W, near_input, pair_paint):
                continue
            if not all(_inb(H, W, r, c) for r, c in pair_paint):
                continue

            near_dist = _marker_distance(near["reds"], H, W)
            far_dist = _marker_distance(far["reds"], H, W)
            if not (far_dist > near_dist + 1.0):
                continue

            groups.extend([near, far])
            occupied_input |= pair_input
            occupied_paint |= pair_paint
            pair_placed = True
            break

        if not pair_placed:
            continue

        filler_ok = True
        for idx in range(2, num_groups):
            random.shuffle(FILLER_PATTERNS)
            placed = False
            for pattern in FILLER_PATTERNS:
                max_r = max(r for cells in (pattern["reds"], pattern["payload"], pattern["paint"]) for r, _c in cells)
                max_c = max(c for cells in (pattern["reds"], pattern["payload"], pattern["paint"]) for _r, c in cells)
                for _place_try in range(120):
                    dr = random.randint(0, H - max_r - 1)
                    dc = random.randint(0, W - max_c - 1)
                    grp = _group_from_pattern(pattern, colors[idx], dr, dc)
                    if not _can_place_input(grp, H, W, occupied_input, occupied_paint):
                        continue
                    if not _can_place_paint(grp, H, W, occupied_input, occupied_paint):
                        continue
                    groups.append(grp)
                    occupied_input |= _group_input_cells(grp)
                    occupied_paint |= set(grp["paint"])
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                filler_ok = False
                break

        if not filler_ok:
            continue

        for grp in groups:
            _paint_input(grid, grp)

        metrics = _variant_metrics(grid)
        if metrics["group_count"] != num_groups:
            continue
        if metrics["overlap_cells"] < 1:
            continue
        if metrics["changed_vs_legacy"] < 1:
            continue
        if metrics["min_dist_gap"] <= 0.25:
            continue

        expected = metrics["ordered"]
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid D8 variant sample")
