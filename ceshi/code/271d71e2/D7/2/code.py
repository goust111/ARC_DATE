# Variant of 271d71e2
# Dimension: D7 — 干扰项（decoy）设计 (obstacle)
# Change: 0-1个全橙色容器作为干扰项，不移动（灰色像素数=0，步数=0）... → 障碍物阻挡路径
#
# Design analysis
# ===============
# variant_spec:
#   kept_from_base:
#     - 背景仍为 6，容器仍是黑框矩形，内部仍只有 5/7。
#     - head/dest 仍是颜色 9 的整条线，沿 head 指向 dest 的轴向平移。
#     - 灰色转橙色的顺序完全保留，仍从 head 朝向的那一侧开始。
#   d7_change:
#     obstacle:
#       color: 8
#       shape: 2x2 单色小方块
#       placement: 放在 head 与 dest 之间、且位于容器/头部的扫掠走廊内
#     movement:
#       planned_steps = min(head_dest_distance, grey_count)
#       actual_steps = planned_steps 再额外受 obstacle 碰撞约束
#       collision_rule: 若容器矩形或 head 在第 t 步会与 obstacle 重叠，则只能停在 t-1
#       orange_gain = actual_steps
#   concrete_trace:
#     - 原题：若 head 到 dest 的距离为 6，灰格数为 5，则移动 5 格并把前 5 个灰格变橙。
#     - 变体：若同样是距离 6、灰格数 5，但第 3 步前方有障碍，则第 4 步会撞上，
#       所以只能移动 3 格，只把前 3 个灰格变橙，dest 保持在更远处不被碰到。
#   complexity_calibration:
#     - 障碍固定为颜色 8，避免把“识别障碍”变成额外颜色映射题。
#     - 障碍固定为 2x2，小而明显；不使用过多形态，避免规则自由度过高。
#     - 每个样本至少 1 个容器被障碍真正截停；若有多个容器，则保留至少 1 个无障碍对照。
#   implicit_assumption_audit:
#     - 原 solver 里 `steps = min(dist, len(grey_coords))` 默认路径中只有背景；现在失效。
#       新 solver 必须在这个位置额外检查平移中的碰撞。
#     - 原 solver 把所有非 bg/0/5/7 颜色都当 head/dest 候选；现在加入颜色 8 后，
#       必须把 8 从 head 颜色候选中排除。
#     - 原 solver 的擦除/粘贴逻辑默认最终落点安全；新 generator 必须保证 obstacle
#       不会放在最终合法落点上，只会阻挡“再往前一步”的位置。
#   placement_legality:
#     - obstacle 永远不接触容器初始位置，不接触 dest，不越界。
#     - 各 assembly 仍按带 1 格背景边距的矩形整体放置，避免不同对象互相污染。
#     - obstacle 只放在本容器的通道里，因此最近的同形 dest 仍唯一可配对。
#   visual_observability:
#     - 人能直接看到：同色 head/dest 之间多了一个 8 色小方块，且输出中容器停在它前面。
#     - 由于 dest 还在更远处，解题者能观察到“不是灰格不够，也不是目标线变短，而是被挡住了”。
#   cleanliness:
#     - 障碍不与 head/dest/边框重叠，且颜色 8 在本题其他对象中不复用，视觉层次清楚。

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
OBSTACLE = 8
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
    """Return local interior coords in the exact solver fill order."""
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


def _collision_limited_steps(
    box_: tuple[int, int, int, int],
    head_pts: set[tuple[int, int]],
    direction: tuple[int, int],
    max_steps: int,
    obstacle_mask: np.ndarray,
) -> int:
    """Largest safe step count before the moving container/head would overlap an obstacle."""
    if max_steps <= 0 or not obstacle_mask.any():
        return int(max_steps)

    H, W = obstacle_mask.shape
    r0, c0, r1, c1 = box_
    moving_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    moving_cells.extend(list(head_pts))
    dr, dc = direction

    for step in range(1, int(max_steps) + 1):
        tr = int(dr * step)
        tc = int(dc * step)
        for r, c in moving_cells:
            rr = int(r + tr)
            cc = int(c + tc)
            if not (0 <= rr < H and 0 <= cc < W):
                return step - 1
            if bool(obstacle_mask[rr, cc]):
                return step - 1
    return int(max_steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging gray cells to orange.

    Variant rule:
    - Planned steps still equal min(head-to-dest distance, gray interior count).
    - Teal obstacles (8) placed in the path can reduce that movement further.
    - The container stops one step before the first collision with an obstacle.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))
    obstacle_mask = g == int(OBSTACLE)

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
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE, OBSTACLE})]

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
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE), int(OBSTACLE)):
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
        box_ = cont["box"]
        r0, c0, r1, c1 = box_
        head_info = pick_head_for_container(box_)
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        head_shape = head_info["head_shape"]
        head_tl = head_info["head_tl"]
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        if direction == (0, 0):
            continue

        dests = []
        for d in comp_by_color.get(hc, []):
            if d["shape"] != head_shape:
                continue
            if d["pts"] == comp_pts:
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
                        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(BORDER):
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
                dests.append({"pts": pts, "shape": shp, "tl": (r00, c00)})

        if not dests:
            continue

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

        tr0 = int(best_dest["tl"][0] - head_tl[0])
        tc0 = int(best_dest["tl"][1] - head_tl[1])
        planned_steps = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        planned_steps = min(int(planned_steps), int(len(grey_coords)))
        steps = _collision_limited_steps(box_, head_pts, direction, planned_steps, obstacle_mask)

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

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _choose_specs() -> list[dict] | None:
    n_containers = random.randint(2, 4)
    obstacle_count = random.randint(1, n_containers)
    if n_containers > 1 and obstacle_count == n_containers:
        obstacle_count = n_containers - 1
    obstacle_ids = set(random.sample(range(n_containers), obstacle_count))

    specs = []
    for i in range(n_containers):
        has_obstacle = i in obstacle_ids
        head_side = random.randint(0, 3)

        if has_obstacle:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            if inner_h * inner_w < 2:
                return None
            dist = random.randint(4, 11)
            max_stop = min(inner_h * inner_w - 1, dist - 3)
            if max_stop < 1:
                return None
            steps = random.randint(1, max_stop)
            grey = random.randint(steps + 1, inner_h * inner_w)
            initial_orange = inner_h * inner_w - grey
        else:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            dist = random.randint(1, 11)
            grey = random.randint(1, inner_h * inner_w)
            initial_orange = inner_h * inner_w - grey
            steps = min(grey, dist)

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "dist": dist,
                "gap": dist - 1,
                "steps": steps,
                "grey": grey,
                "initial_orange": initial_orange,
                "has_obstacle": has_obstacle,
            }
        )
    return specs


def _obstacle_cells(spec: dict, cr: int, cc: int) -> list[tuple[int, int]]:
    if not spec["has_obstacle"]:
        return []

    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    steps = spec["steps"]
    orth_offset = spec["obstacle_offset"]

    if hs == 0:
        head_r = cr - 1
        near_r = head_r - (steps + 1)
        return [(near_r - 1, cc + orth_offset + dc) for dc in range(2)] + [
            (near_r, cc + orth_offset + dc) for dc in range(2)
        ]
    if hs == 1:
        head_r = cr + oh
        near_r = head_r + (steps + 1)
        return [(near_r, cc + orth_offset + dc) for dc in range(2)] + [
            (near_r + 1, cc + orth_offset + dc) for dc in range(2)
        ]
    if hs == 2:
        head_c = cc - 1
        near_c = head_c - (steps + 1)
        return [(cr + orth_offset + dr, near_c - 1) for dr in range(2)] + [
            (cr + orth_offset + dr, near_c) for dr in range(2)
        ]
    head_c = cc + ow
    near_c = head_c + (steps + 1)
    return [(cr + orth_offset + dr, near_c) for dr in range(2)] + [
        (cr + orth_offset + dr, near_c + 1) for dr in range(2)
    ]


def _assembly_bounds(spec: dict) -> tuple[int, int, int, int]:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    gap = spec["gap"]

    if hs == 0:
        total_h, total_w = 1 + gap + 1 + oh, ow
        cont_r, cont_c = gap + 2, 0
    elif hs == 1:
        total_h, total_w = oh + 1 + gap + 1, ow
        cont_r, cont_c = 0, 0
    elif hs == 2:
        total_h, total_w = oh, 1 + gap + 1 + ow
        cont_r, cont_c = 0, gap + 2
    else:
        total_h, total_w = oh, ow + 1 + gap + 1
        cont_r, cont_c = 0, 0
    return total_h, total_w, cont_r, cont_c


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_specs()
    if specs is None:
        return None

    assemblies = []
    for spec in specs:
        total_h, total_w, cont_r, cont_c = _assembly_bounds(spec)
        spec = {**spec, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c}
        if spec["has_obstacle"]:
            orth_span = spec["outer_w"] if spec["head_side"] in (0, 1) else spec["outer_h"]
            if orth_span < 2:
                return None
            spec["obstacle_offset"] = random.randint(0, orth_span - 2)
        assemblies.append(spec)

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(9, int(total_area ** 0.5) + 3)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(9, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(9, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for a in assemblies:
        th, tw = a["total_h"], a["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            cr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            cr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if not occupied[cr0:cr1, cc0:cc1].any():
                occupied[r0 : r0 + th, c0 : c0 + tw] = True
                placements.append((r0, c0))
                placed = True
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 1 - a["dist"], cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + a["dist"], cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 1 - a["dist"]] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + a["dist"]] = HEAD_COLOR

        for r, c in _obstacle_cells(a, cr, cc):
            if not (0 <= r < grid_h and 0 <= c < grid_w):
                return None
            if int(grid[r, c]) != int(BG):
                return None
            grid[r, c] = OBSTACLE

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        steps = a["steps"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        dr, dc = SIDE_TO_DIR[hs]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        expected[cr : cr + oh, cc : cc + ow] = BG
        if hs == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif hs == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif hs == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

        nr = cr + dr * steps
        nc = cc + dc * steps
        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange + steps):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr : nr + oh, nc + ow] = HEAD_COLOR

    if not np.array_equal(output, expected):
        return None

    if not any(a["has_obstacle"] for a in assemblies):
        return None
    if not any(a["has_obstacle"] and a["steps"] < min(a["grey"], a["dist"]) for a in assemblies):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid obstacle variant after 400 attempts")
