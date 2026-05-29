# Combo variant of 7ed72f31 (index 14)
# combo_spec: D5/2+D8/0
# Name: 碰撞消除有序处理
#
# Design analysis
# ---------------
# a) Concretization and behavior trace
#    原题把每个红标记组的镜像像素直接叠加到输出上，默认“谁写到哪里都只是覆盖”。
#    这次融合后必须改成真正的顺序过程：
#    1. 先从输入里抽取所有可反射的标记组（横红线、竖红线、混合单红点、孤立红点）。
#    2. 按红标记几何中心到整张网格中心的平方欧氏距离，从远到近排序。
#    3. 依次处理每个组的镜像像素；每个像素落笔时：
#       - 目标是背景：写入镜像颜色。
#       - 目标已是任意非背景像素：当前像素与旧像素一起消失，该格改回背景。
#    关键是“已有非背景像素”不只包含输入原件，也包含前面较远组已经画上的镜像。
#
#    一个具体追踪：
#    - 组 A（较远，竖线）和组 B（较近，横线）都会把镜像打到同两个格子 X/Y。
#    - X/Y 在输入里先放了阻挡色像素。
#    - 先处理 A：A 命中 X/Y 时先把阻挡像素消掉，X/Y 变背景。
#    - 后处理 B：B 再命中 X/Y 时，因为此时已经是背景，所以 B 的颜色留在 X/Y。
#    - 若顺序反过来，则最后留下的是 A 的颜色。
#    因此这里不是“静态碰撞检测”，而是“顺序性消除链”：先到者负责清空，后到者决定幸存颜色。
#
# b) Complexity calibration
#    只做单个偶发碰撞会过于隐蔽，所以生成器固定放两类阻挡点：
#    - 共享阻挡点：被主对的两个组共同命中，用来体现“顺序决定最后是谁留下颜色”；
#    - 单命中阻挡点：只被一个组命中，用来体现“碰撞后会留下背景空洞，而不是简单覆盖”。
#    这样每张图里都会同时出现：
#    - D5：镜像撞到已有像素会消掉两者，形成可见空洞；
#    - D8：谁先撞掉共享阻挡、谁后补上颜色，受距中心排序控制。
#    其余 1-3 个 filler 组保持原题风格，但禁止再制造额外重叠，避免视觉噪声。
#
# c) Implicit-assumption audit of the original main
#    - 原代码的 `out[rr, cc] = v` 假设覆盖永远合法；融合后失效，必须改成读取当前 out 的动态状态。
#    - 原代码把“混合组件”和“孤立红点”分两阶段处理；在 D8 中这只能作为发现顺序，不能再决定优先级。
#      所以这里先统一抽组，再统一排序。
#    - 单独的 D5 版本把所有 hits 先收集再一次性消解；那会丢失本融合里最重要的顺序效应。
#      本版本必须逐组逐像素执行，才能让“先清空、后补上”成立。
#    - 原 solver 允许 bbox 扫描同色杂点；因此生成器继续保证每组 payload 颜色唯一，阻挡色也不复用。
#
# d) Placement legality
#    - 各组输入原件彼此不重叠，也不 4 邻接，避免 solver 把多组并成一个组件。
#    - 所有镜像目标都在网格内。
#    - 只有主组允许共享镜像目标；filler 组的镜像区与任何已有镜像区都不重叠。
#    - 阻挡像素只放在主组共享目标里、且不 4 邻接任何输入原件，避免它们在输入中并入某个反射组件。
#
# e) Visual observability
#    观察者能直接看到两种证据：
#    - 共享阻挡点：输入里两枚阻挡像素正位于两个红标记组都“应该补出来”的镜像区域，
#      输出里它们统一变成更靠中心那组的颜色，体现顺序性。
#    - 单命中阻挡点：输入里还有若干阻挡像素只挡住一个镜像，输出里这些位置直接变回背景，
#      明确体现“碰撞即双消”，而不是普通覆盖。
#
# f) Cleanliness
#    重叠只集中在两格主目标上；其他组不共享镜像区，也不让镜像压到别人的原件/红标记。
#    这样局部差异明显，但整张图仍保持原题的清爽镜像结构。
#
# g) New solver assumptions and generator support
#    - 假设：排序比较稳定。生成器要求主对的距离差至少 2.0，并检查所有组的最小距离间隔。
#    - 假设：顺序差异真的改变结果。生成器同时计算“按发现顺序的碰撞版输出”，要求与正式输出
#      至少在两个主阻挡格上不同。
#    - 假设：阻挡格会被至少两个组命中。生成器固定使用共享目标作为阻挡位置来保证这一点.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


MARK = 2
PALETTE = [0, 1, 3, 4, 5, 6, 7, 8, 9]


def _inb(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(H) and 0 <= int(c) < int(W)


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    cr, cc = map(int, center)
    return int(2 * cr - r), int(2 * cc - c)


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


def _build_components(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    lab, n = label_mask(g != int(bg), connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        rs = [int(r) for r, _c in coords]
        cs = [int(c) for _r, c in coords]
        comps.append(
            {
                "coords": coords,
                "vals": vals,
                "bbox": (int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))),
            }
        )
    return comps


def _extract_groups(g: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.array(g, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    comps = _build_components(g, int(bg))
    groups: list[dict[str, object]] = []
    discovery = 0

    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if int(MARK) not in set(map(int, vals)):
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
                    v = int(g[int(r), int(c)])
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
                groups.append(
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
            continue

        if len(coords) != 1 or int(vals[0]) != int(MARK):
            continue

        pr, pc = map(int, coords[0])
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(H, W, int(rr), int(cc)):
                    continue
                v = int(g[int(rr), int(cc)])
                if v == int(bg) or v == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        if not neigh_colors:
            continue

        paint = []
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
            groups.append(
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

    return bg, groups


def _sort_groups(groups: list[dict[str, object]], *, ordered: bool) -> list[dict[str, object]]:
    if ordered:
        return sorted(
            groups,
            key=lambda grp: (
                -float(grp["distance"]),
                float(grp["marker_center"][0]),
                float(grp["marker_center"][1]),
                int(grp["phase"]),
                int(grp["discovery"]),
            ),
        )
    return sorted(groups, key=lambda grp: (int(grp["phase"]), int(grp["discovery"])))


def _apply_groups(g: np.ndarray, bg: int, groups: list[dict[str, object]], *, collision: bool) -> np.ndarray:
    out = np.array(g, dtype=int, copy=True)
    for grp in groups:
        for r, c, v in grp["paint"]:
            if not collision:
                out[int(r), int(c)] = int(v)
            elif int(out[int(r), int(c)]) != int(bg):
                out[int(r), int(c)] = int(bg)
            else:
                out[int(r), int(c)] = int(v)
    return out


def _solve(grid: np.ndarray, *, ordered: bool, collision: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, groups = _extract_groups(g)
    return _apply_groups(g, int(bg), _sort_groups(groups, ordered=ordered), collision=collision)


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, ordered=True, collision=True)


def _shift(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _group_from_pattern(pattern: dict[str, object], color: int, dr: int, dc: int) -> dict[str, object]:
    reds = _shift(pattern["reds"], dr, dc)
    payload = _shift(pattern["payload"], dr, dc)
    paint = _shift(pattern["paint"], dr, dc)
    return {
        "name": pattern["name"],
        "kind": pattern["kind"],
        "reds": reds,
        "payload": payload,
        "paint": paint,
        "color": int(color),
    }


def _group_input_cells(group: dict[str, object]) -> set[tuple[int, int]]:
    return set(group["reds"]) | set(group["payload"])


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _can_place_input(
    group: dict[str, object],
    H: int,
    W: int,
    occupied_input: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    for r, c in _group_input_cells(group):
        if not _inb(H, W, int(r), int(c)):
            return False
        if (int(r), int(c)) in occupied_input or (int(r), int(c)) in reserved:
            return False
        for rr, cc in _neighbors4((int(r), int(c))):
            if (int(rr), int(cc)) in occupied_input:
                return False
    return True


def _can_place_paint(
    group: dict[str, object],
    H: int,
    W: int,
    occupied_input: set[tuple[int, int]],
    occupied_paint: set[tuple[int, int]],
) -> bool:
    for r, c in group["paint"]:
        if not _inb(H, W, int(r), int(c)):
            return False
        if (int(r), int(c)) in occupied_input or (int(r), int(c)) in occupied_paint:
            return False
    return True


def _paint_input(grid: np.ndarray, group: dict[str, object]) -> None:
    color = int(group["color"])
    for r, c in group["reds"]:
        grid[int(r), int(c)] = int(MARK)
    for r, c in group["payload"]:
        grid[int(r), int(c)] = int(color)


def _distance_gap(groups: list[dict[str, object]], H: int, W: int) -> float:
    dists = [_marker_distance(list(map(tuple, grp["marker_coords"])), H, W) for grp in groups]
    dists = sorted(float(d) for d in dists)
    if len(dists) < 2:
        return 999.0
    return min(b - a for a, b in zip(dists, dists[1:]))


MAIN_NEAR_ROW = {
    "name": "main_near_row",
    "kind": "row",
    "reds": [(4, 5), (4, 6), (4, 7)],
    "payload": [(3, 5), (3, 6), (2, 6), (2, 7)],
    "paint": [(5, 5), (5, 6), (6, 6), (6, 7)],
}

MAIN_FAR_COL = {
    "name": "main_far_col",
    "kind": "col",
    "reds": [(5, 4), (6, 4), (7, 4)],
    "payload": [(5, 3), (6, 3), (6, 2), (6, 1)],
    "paint": [(5, 5), (6, 5), (6, 6), (6, 7)],
}

MAIN_BLOCKERS = [(6, 6), (6, 7)]

FILLER_PATTERNS = [
    {
        "name": "row_top",
        "kind": "row",
        "reds": [(3, 1), (3, 2), (3, 3)],
        "payload": [(0, 2), (1, 1), (1, 2), (2, 1)],
        "paint": [(6, 2), (5, 1), (5, 2), (4, 1)],
    },
    {
        "name": "col_left",
        "kind": "col",
        "reds": [(1, 3), (2, 3), (3, 3)],
        "payload": [(1, 0), (2, 1), (2, 2), (3, 1)],
        "paint": [(1, 6), (2, 5), (2, 4), (3, 5)],
    },
    {
        "name": "point_mixed",
        "kind": "point",
        "reds": [(2, 2)],
        "payload": [(1, 2), (2, 3), (3, 3)],
        "paint": [(3, 2), (2, 1), (1, 1)],
    },
    {
        "name": "row_long",
        "kind": "row",
        "reds": [(4, 1), (4, 2), (4, 3), (4, 4)],
        "payload": [(1, 2), (2, 1), (2, 2), (3, 2)],
        "paint": [(7, 2), (6, 1), (6, 2), (5, 2)],
    },
]


def _blocker_safe(
    cell: tuple[int, int],
    input_cells: set[tuple[int, int]],
    chosen: set[tuple[int, int]],
) -> bool:
    r, c = map(int, cell)
    if (int(r), int(c)) in input_cells or (int(r), int(c)) in chosen:
        return False
    for rr, cc in _neighbors4((int(r), int(c))):
        if (int(rr), int(cc)) in input_cells or (int(rr), int(cc)) in chosen:
            return False
    return True


def _pick_solo_blockers(
    groups: list[dict[str, object]],
    shared_blockers: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    input_cells: set[tuple[int, int]] = set()
    paint_counts: dict[tuple[int, int], int] = {}
    for grp in groups:
        input_cells |= _group_input_cells(grp)
        for cell in grp["paint"]:
            key = (int(cell[0]), int(cell[1]))
            paint_counts[key] = paint_counts.get(key, 0) + 1

    candidates_by_group: list[list[tuple[int, int]]] = []
    shared_set = {(int(r), int(c)) for r, c in shared_blockers}
    for grp in groups:
        uniq: list[tuple[int, int]] = []
        for r, c in grp["paint"]:
            cell = (int(r), int(c))
            if cell in shared_set:
                continue
            if paint_counts.get(cell, 0) != 1:
                continue
            if _blocker_safe(cell, input_cells, shared_set):
                uniq.append(cell)
        if uniq:
            candidates_by_group.append(sorted(set(uniq)))

    rng.shuffle(candidates_by_group)
    chosen: set[tuple[int, int]] = set(shared_set)
    solo: list[tuple[int, int]] = []

    for group_cands in candidates_by_group:
        rng.shuffle(group_cands)
        for cell in group_cands:
            if _blocker_safe(cell, input_cells, chosen):
                chosen.add(cell)
                solo.append(cell)
                break
        if len(solo) >= 2:
            break

    if len(solo) >= 2:
        return solo[:2]

    flat = [cell for group_cands in candidates_by_group for cell in group_cands]
    rng.shuffle(flat)
    for cell in flat:
        if len(solo) >= 2:
            break
        if cell in solo:
            continue
        if _blocker_safe(cell, input_cells, chosen):
            chosen.add(cell)
            solo.append(cell)
    return solo


def _combo_metrics(
    grid: np.ndarray,
    shared_blockers: list[tuple[int, int]],
    solo_blockers: list[tuple[int, int]],
    near_color: int,
    far_color: int,
) -> dict[str, object]:
    ordered = main(grid)
    legacy = _solve(grid, ordered=False, collision=True)
    overlay = _solve(grid, ordered=True, collision=False)
    H, W = grid.shape
    _bg, groups = _extract_groups(grid)
    swing_cells = 0
    near_wins = 0
    far_wins_legacy = 0
    for r, c in shared_blockers:
        if int(ordered[int(r), int(c)]) != int(legacy[int(r), int(c)]):
            swing_cells += 1
        if int(ordered[int(r), int(c)]) == int(near_color):
            near_wins += 1
        if int(legacy[int(r), int(c)]) == int(far_color):
            far_wins_legacy += 1
    collision_holes = 0
    for r, c in solo_blockers:
        if int(ordered[int(r), int(c)]) != int(overlay[int(r), int(c)]) and int(ordered[int(r), int(c)]) == int(get_bg(grid)):
            collision_holes += 1
    return {
        "ordered": ordered,
        "legacy": legacy,
        "overlay": overlay,
        "group_count": len(groups),
        "changed_vs_legacy": int(np.count_nonzero(ordered != legacy)),
        "changed_vs_overlay": int(np.count_nonzero(ordered != overlay)),
        "swing_cells": int(swing_cells),
        "ordered_near_wins": int(near_wins),
        "legacy_far_wins": int(far_wins_legacy),
        "collision_holes": int(collision_holes),
        "min_dist_gap": float(_distance_gap(groups, H, W)),
    }


def generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    rng = rng or random.Random()

    for _attempt in range(500):
        H = rng.randint(22, 26)
        W = rng.randint(22, 26)
        bg = rng.choice([c for c in PALETTE if int(c) != int(MARK)])

        num_groups = rng.randint(3, 5)
        usable = [c for c in PALETTE if int(c) not in {int(bg), int(MARK)}]
        rng.shuffle(usable)
        group_colors = usable[:num_groups]
        blocker_color = int(usable[num_groups])

        pair_ready = False
        groups: list[dict[str, object]] = []
        shared_blockers_abs: list[tuple[int, int]] = []
        occupied_input: set[tuple[int, int]] = set()
        occupied_paint: set[tuple[int, int]] = set()

        for _pair_try in range(160):
            dr = rng.randint(0, H - 8)
            dc = rng.randint(0, W - 8)
            if int(dr - dc) < 2:
                continue

            near = _group_from_pattern(MAIN_NEAR_ROW, int(group_colors[0]), int(dr), int(dc))
            far = _group_from_pattern(MAIN_FAR_COL, int(group_colors[1]), int(dr), int(dc))
            blockers = _shift(MAIN_BLOCKERS, int(dr), int(dc))

            pair_input = _group_input_cells(near) | _group_input_cells(far)
            pair_paint = set(near["paint"]) | set(far["paint"])
            if not _can_place_input(near, H, W, set(), pair_paint):
                continue
            if not _can_place_input(far, H, W, _group_input_cells(near), pair_paint):
                continue
            if any((int(r), int(c)) in pair_input for r, c in blockers):
                continue
            if any(not _inb(H, W, int(r), int(c)) for r, c in pair_paint):
                continue
            if any(not _inb(H, W, int(r), int(c)) for r, c in blockers):
                continue
            if any(any((int(rr), int(cc)) in pair_input for rr, cc in _neighbors4((int(r), int(c)))) for r, c in blockers):
                continue

            near_dist = _marker_distance(list(map(tuple, near["reds"])), H, W)
            far_dist = _marker_distance(list(map(tuple, far["reds"])), H, W)
            if not (float(far_dist) > float(near_dist) + 2.0):
                continue

            groups = [near, far]
            shared_blockers_abs = blockers
            occupied_input = set(pair_input)
            occupied_paint = set(pair_paint)
            pair_ready = True
            break

        if not pair_ready:
            continue

        filler_ok = True
        for idx in range(2, num_groups):
            placed = False
            for pattern in rng.sample(FILLER_PATTERNS, k=len(FILLER_PATTERNS)):
                max_r = max(r for cells in (pattern["reds"], pattern["payload"], pattern["paint"]) for r, _c in cells)
                max_c = max(c for cells in (pattern["reds"], pattern["payload"], pattern["paint"]) for _r, c in cells)
                for _place_try in range(120):
                    dr = rng.randint(0, H - max_r - 1)
                    dc = rng.randint(0, W - max_c - 1)
                    grp = _group_from_pattern(pattern, int(group_colors[idx]), int(dr), int(dc))
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

        solo_blockers_abs = _pick_solo_blockers(groups, shared_blockers_abs, rng)
        if len(solo_blockers_abs) < 2:
            continue

        inp = np.full((H, W), int(bg), dtype=int)
        for grp in groups:
            _paint_input(inp, grp)
        all_blockers_abs = list(shared_blockers_abs) + list(solo_blockers_abs)
        for r, c in all_blockers_abs:
            inp[int(r), int(c)] = int(blocker_color)

        metrics = _combo_metrics(inp, shared_blockers_abs, solo_blockers_abs, int(group_colors[0]), int(group_colors[1]))
        if metrics["group_count"] != num_groups:
            continue
        if metrics["swing_cells"] < len(shared_blockers_abs):
            continue
        if metrics["ordered_near_wins"] < len(shared_blockers_abs):
            continue
        if metrics["legacy_far_wins"] < len(shared_blockers_abs):
            continue
        if metrics["collision_holes"] < len(solo_blockers_abs):
            continue
        if metrics["changed_vs_legacy"] < len(shared_blockers_abs):
            continue
        if metrics["changed_vs_overlay"] < len(solo_blockers_abs):
            continue
        if metrics["min_dist_gap"] <= 0.5:
            continue

        out = metrics["ordered"]
        meta = {
            "bg": int(bg),
            "grid_shape": (int(H), int(W)),
            "group_count": int(num_groups),
            "group_types": [str(grp["kind"]) for grp in groups],
            "group_colors": [int(grp["color"]) for grp in groups],
            "blocker_color": int(blocker_color),
            "shared_blockers": [(int(r), int(c)) for r, c in shared_blockers_abs],
            "solo_blockers": [(int(r), int(c)) for r, c in solo_blockers_abs],
            "swing_cells": int(metrics["swing_cells"]),
            "collision_holes": int(metrics["collision_holes"]),
            "changed_vs_legacy": int(metrics["changed_vs_legacy"]),
            "changed_vs_overlay": int(metrics["changed_vs_overlay"]),
            "min_dist_gap": float(metrics["min_dist_gap"]),
        }
        return inp, out, meta

    raise RuntimeError("failed to generate a valid combo sample")


def generate_input() -> np.ndarray:
    inp, _out, _meta = generate_case()
    return inp
