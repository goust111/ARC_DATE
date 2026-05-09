# Variant of 271d71e2
# Dimension: D5 — 灰色像素着色顺序 (multi_step)
# Change: ordered_processing — 灰色像素按从头侧向内的行/列顺序逐个变为橙色... → 随机但可推断的顺序（按距离排序）
#
# Design analysis
# ===============
# source_task review:
# - 参考任务 dd6b8c4b 的关键机制不是“扫描顺序”，而是“先找参考区，再按曼哈顿距离排序，
#   平局再用固定坐标顺序打破”。这里沿用这个复杂度级别，只替换本题 D5 着色顺序。
#
# variant concrete rule:
# - 黑框容器、头部线、目标线、平移步数规则全部保持原题。
# - 唯一核心变化：每一步把哪个 GREY(5) 变成 ORANGE(7)，不再按 head-facing side 的
#   行/列顺序，而是按“该灰格到目标线中点的曼哈顿距离”升序决定。
# - 目标线是整条 1-cell 粗的 9 线，中点在人眼可见的位置；若线长为偶数，中点是半格，
#   用 doubled coordinates 计算距离，避免浮点。
# - 平局用稳定坐标顺序打破：solver 中用绝对 (row, col)，generator 中等价地用局部 (row, col)。
#
# behavior trace on a small right-moving example:
# - 假设 inner = 3x3，头在右边，目标线也在更右边。
# - 原题顺序：先右上，再右中，再右下，再中上...（按列从右向左、列内从上到下）。
# - 本变体顺序：先右中，再右上/右下，再中中...，因为这些格到目标线中点的距离最小。
# - 因而输出里新增橙色会围绕“朝向目标的中轴”展开，而不是整列整行推进。
#
# complexity calibration:
# - 若横向移动但 inner_h=1/2，或纵向移动但 inner_w=1/2，距离排序可能退化得不明显。
# - 为了让 2-4 个样例就能看出差异，generator 对 mover 的“垂直于运动方向”的尺寸优先选
#   奇数且 >=3，这样中轴优先的效果会直接出现在第一步或前几步。
# - 仍保留 2-4 个容器、0-1 个全橙 decoy、gap capped / grey capped 的混合。
#
# implicit-assumption audit against original main():
# - container detection 只依赖 interior 属于 {5,7}，不依赖橙色是否连续，所以仍成立。
# - head/destination detection、方向判定、最近目标匹配都与 D5 无关，保留。
# - 原代码唯一依赖旧 D5 的位置是 `order = sorted(grey_coords, key=...)`；这里必须改成
#   距离到目标线中点排序，否则 solver 会在新输入上把错误的灰格涂橙。
# - 原 generator 用 `_interior_fill_order()` 同时生成输入已有橙色与输出新增橙色；若只改 solver
#   不改 generator，样例会退化成“旧输入 + 新输出”，人类难以从状态前缀推断规则。
#   因此 generator 里的已有橙色也必须改成同一距离序列的前缀。
#
# placement legality:
# - 仍沿用原题 assembly 放置：dest + gap + head + container 共线且相邻区域之间至少留 1 格背景。
# - 目标线中点排序只影响内部颜色，不增加新几何对象，因此不会引入新的重叠模式。
# - solver 新假设：已有 ORANGE 必须是同一距离序列的前缀，这样“对剩余 GREY 再排序后取前 steps 个”
#   才等价于继续执行多步过程。generator 明确按此前缀构造，保证这一点。
#
# visual observability / cleanliness:
# - 参考点使用可见的目标线中点，而不是颜色编号等不可见信息。
# - 颜色种类不变，布局不变；唯一视觉变化是橙色在容器内部呈现“向目标中轴聚拢”的层次，
#   比原题的整行/整列推进更不规则但仍可从几组样例归纳。

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

# head on top -> container moves up; head on bottom -> moves down; etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _midpoint2_from_pts(pts: set[tuple[int, int]]) -> tuple[int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    return int(min(rs) + max(rs)), int(min(cs) + max(cs))


def _distance_fill_order_abs(
    cells: list[tuple[int, int]], ref2r: int, ref2c: int
) -> list[tuple[int, int]]:
    return sorted(
        cells,
        key=lambda p: (
            abs(2 * int(p[0]) - int(ref2r)) + abs(2 * int(p[1]) - int(ref2c)),
            int(p[0]),
            int(p[1]),
        ),
    )


def _distance_fill_order_local(inner_h: int, inner_w: int, head_side: int, gap: int = 0) -> list[tuple[int, int]]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if head_side == 0:
        ref2r, ref2c = -2 * (gap + 2), outer_w - 1
    elif head_side == 1:
        ref2r, ref2c = 2 * (outer_h + gap + 1), outer_w - 1
    elif head_side == 2:
        ref2r, ref2c = outer_h - 1, -2 * (gap + 2)
    else:
        ref2r, ref2c = outer_h - 1, 2 * (outer_w + gap + 1)

    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    return sorted(
        cells,
        key=lambda p: (
            abs(2 * (p[0] + 1) - ref2r) + abs(2 * (p[1] + 1) - ref2c),
            p[0],
            p[1],
        ),
    )


def _legacy_fill_order_local(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
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


def _spec_exposes_variant(inner_h: int, inner_w: int, head_side: int, initial_orange: int, steps: int) -> bool:
    if steps <= 0:
        return False
    dist_order = _distance_fill_order_local(inner_h, inner_w, head_side, gap=0)
    legacy_order = _legacy_fill_order_local(inner_h, inner_w, head_side)
    for prefix_len in (initial_orange, initial_orange + steps):
        if 0 < prefix_len < inner_h * inner_w:
            if set(dist_order[:prefix_len]) != set(legacy_order[:prefix_len]):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging gray cells to orange.

    Variant D5:
    - Translation remains capped by min(head_to_dest_distance, number_of_gray_cells).
    - The charged gray cells are chosen by distance to the destination-line midpoint,
      not by a simple scan from the head-facing side.
    """
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

    out = g.copy()
    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in all_head_colors:
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, _r1, _c1 = bbox_coords(coords)
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
                    if val not in (int(bg), int(BORDER), int(GREY), int(ORANGE)):
                        head_seeds.append((rr, cc, val))
        if not head_seeds:
            return None

        hc = int(Counter(val for _r, _c, val in head_seeds).most_common(1)[0][0])
        seed_pts = {(r, c) for r, c, val in head_seeds if int(val) == int(hc)}
        cands = comp_by_color.get(hc, [])
        comp0 = None
        for comp in cands:
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
        steps = abs(tr0) + abs(tc0)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(steps), int(len(grey_coords)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        ref2r, ref2c = _midpoint2_from_pts(best_dest["pts"])
        order = _distance_fill_order_abs(grey_coords, ref2r, ref2c)
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


def _make_container_spec(is_decoy: bool, force_visible: bool) -> dict:
    head_side = random.randint(0, 3)
    if not is_decoy and head_side in (0, 1):
        inner_h = random.choice([3, 5, 7])
        inner_w = random.randint(1, 5)
    elif not is_decoy:
        inner_h = random.randint(1, 8)
        inner_w = random.choice([3, 5])
    else:
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)

    inner_area = inner_h * inner_w
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if is_decoy:
        grey = 0
        initial_orange = inner_area
        gap = random.randint(0, 6)
        steps = 0
    else:
        if force_visible:
            max_gap = min(4, inner_area - 2)
            if max_gap < 0:
                max_gap = 0
            gap = random.randint(0, max_gap)
            h2d = gap + 1
            grey = random.randint(min(inner_area, h2d + 1), inner_area)
        else:
            gap = random.randint(0, 10)
            h2d = gap + 1
            grey = random.randint(1, inner_area)
        initial_orange = inner_area - grey
        steps = min(grey, h2d)

    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": gap,
        "steps": steps,
        "initial_orange": initial_orange,
        "grey": grey,
        "head_len": outer_w if head_side in (0, 1) else outer_h,
        "is_decoy": is_decoy,
    }


def _build_specs() -> list[dict]:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    visible_slot = random.randrange(n_movers)
    for i in range(n_containers):
        is_decoy = i >= n_movers
        spec = _make_container_spec(is_decoy=is_decoy, force_visible=(i == visible_slot))
        specs.append(spec)

    if not any(
        _spec_exposes_variant(s["inner_h"], s["inner_w"], s["head_side"], s["initial_orange"], s["steps"])
        for s in specs
        if not s["is_decoy"]
    ):
        return []
    return specs


def _assembly_spec(spec: dict) -> dict:
    oh, ow = spec["outer_h"], spec["outer_w"]
    hs = spec["head_side"]
    gap = spec["gap"]
    if hs == 0:
        total_h = 1 + gap + 1 + oh
        total_w = ow
        cont_r = gap + 2
        cont_c = 0
    elif hs == 1:
        total_h = oh + 1 + gap + 1
        total_w = ow
        cont_r = 0
        cont_c = 0
    elif hs == 2:
        total_h = oh
        total_w = 1 + gap + 1 + ow
        cont_r = 0
        cont_c = gap + 2
    else:
        total_h = oh
        total_w = ow + 1 + gap + 1
        cont_r = 0
        cont_c = 0
    return {
        **spec,
        "total_h": total_h,
        "total_w": total_w,
        "cont_r": cont_r,
        "cont_c": cont_c,
    }


def _place_assemblies(assemblies: list[dict], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []
    for a in assemblies:
        th, tw = a["total_h"], a["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(300):
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
    return placements


def _line_starts(assembly: dict, cr: int, cc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    oh, ow = assembly["outer_h"], assembly["outer_w"]
    hs = assembly["head_side"]
    gap = assembly["gap"]
    if hs == 0:
        head_start = (cr - 1, cc)
        dest_start = (cr - 2 - gap, cc)
    elif hs == 1:
        head_start = (cr + oh, cc)
        dest_start = (cr + oh + 1 + gap, cc)
    elif hs == 2:
        head_start = (cr, cc - 1)
        dest_start = (cr, cc - 2 - gap)
    else:
        head_start = (cr, cc + ow)
        dest_start = (cr, cc + ow + 1 + gap)
    return head_start, dest_start


def _gen_attempt() -> np.ndarray | None:
    specs = _build_specs()
    if not specs:
        return None

    assemblies = [_assembly_spec(s) for s in specs]
    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area**0.5) + 2)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

    lo_h = max(8, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))
    placements = _place_assemblies(assemblies, grid_h, grid_w)
    if placements is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        hs = a["head_side"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _distance_fill_order_local(inner_h, inner_w, hs, gap=a["gap"])
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(a["initial_orange"]):
            rr, cc2 = fill_order[idx_cell]
            interior[rr, cc2] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        head_start, dest_start = _line_starts(a, cr, cc)
        if hs in (0, 1):
            grid[head_start[0], head_start[1] : head_start[1] + ow] = HEAD_COLOR
            grid[dest_start[0], dest_start[1] : dest_start[1] + ow] = HEAD_COLOR
        else:
            grid[head_start[0] : head_start[0] + oh, head_start[1]] = HEAD_COLOR
            grid[dest_start[0] : dest_start[0] + oh, dest_start[1]] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        hs = a["head_side"]
        steps = a["steps"]
        dr, dc = SIDE_TO_DIR[hs]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]
        if steps == 0:
            continue

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
        fill_order = _distance_fill_order_local(inner_h, inner_w, hs, gap=a["gap"])
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(a["initial_orange"] + steps):
            rr, cc2 = fill_order[idx_cell]
            interior[rr, cc2] = ORANGE
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
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
