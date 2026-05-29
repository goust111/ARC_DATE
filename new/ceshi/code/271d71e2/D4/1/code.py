from __future__ import annotations

# Variant of 271d71e2
# Dimension: D4 — 源对象结构 (src)
# Change: 黑色边框矩形容器，内部为灰色/橙色填充，附带同色头部线段指示方向... → 多层嵌套容器
#
# Design analysis
# ===============
# variant_spec:
#   overview:
#     - 每个源对象不再是“黑框 + 灰/橙内核”，而是“外黑框 + 内彩框 + 灰/橙内核”三层结构。
#     - 头部线段与目标线段仍在容器外侧，仍然同色，仍然决定平移方向与目标位置。
#     - 新规则：只有当“内彩框颜色 == 头/目标线颜色”时，容器才执行原题的移动/充能。
#       若颜色不匹配，则即使存在灰色像素、目标也在同轴线上，容器也完全不动。
#   concrete_geometry:
#     - 外框颜色固定为 0，厚度 1。
#     - 内框颜色取少量显眼颜色之一 {1,2,3,4}，厚度 1。
#     - 最内部矩形仍由 5/7 构成，按头部朝向从近侧向远侧排序。
#     - 因为多了一层内框，最小总尺寸从 3x3 变为 5x5，总尺寸 = (inner_h+4) x (inner_w+4)。
#   behavior_trace:
#     - 原题中，solver 从灰/橙连通块 bbox 外扩 1 格就能遇到黑框。
#     - 本变体里，灰/橙连通块 bbox 外扩 1 格遇到的是“彩色内框”，再外扩 2 格才是黑框。
#     - 因此检测逻辑必须改为：先找到灰/橙内核，再验证外扩 1 层是统一颜色的内框，
#       外扩 2 层是纯黑外框；之后再检查该内框颜色是否与头/目标线颜色一致。
#     - 例子：若一个向右的容器 head=2、dest=2、内框=2，则 steps=min(灰数, head到dest距离)；
#       若 head=2、dest=2、内框=3，则原题会移动，但变体要求 steps=0。
#   complexity_calibration:
#     - 为避免退化，generator 强制生成至少一个“可移动匹配容器”和至少一个
#       “有灰、有目标但因颜色不匹配而静止”的容器。
#     - 头/目标颜色只使用 4 种低复杂度颜色，且每个样例中容器数最多 4 个，避免匹配歧义。
#   implicit_assumptions_review:
#     - 原代码 `for pts in components_mask((g==GREY)|(g==FILL))` 仍然可用，但 bbox 外扩尺度必须从 1 改 2。
#     - 原代码 `pick_head_for_container` 从黑框四邻域收集任意非背景色；在本变体中这会把“内彩框”误识别为 head。
#       所以改成只检查外黑框四侧“外侧一格”的整条线。
#     - 原代码将所有非 bg/0/5/7 颜色都当作 head/dest 候选；现在这些颜色还包含内框颜色，
#       因此后续 dest 匹配必须继续要求“形状等于 head 线段形状”，从而排除空心矩形内框。
#     - 原代码对所有容器都尝试移动；本变体新增前提“inner_color == head_color”，
#       若不满足则必须直接跳过移动，且保持 head/dest 不变。
#   placement_legality:
#     - 生成器仍按“container + head + gap + dest”的总装配 bbox 放置对象，装配之间至少隔 1 个背景格。
#     - 输出中的移动发生在装配 bbox 内部，因此输入不重叠即可推出输出也不重叠。
#     - 所有 blocker 也保留正灰数和正 head-dest 距离，确保静止确实来自颜色条件而非其它因素。
#   visual_observability:
#     - 人眼可以直接看到：有些容器内框颜色和外部线段同色，于是移动并把灰色变橙；
#       有些容器线段颜色与内框不同，即使外部几何关系相同也不移动。
#     - 这个差异在 2-4 个样例中可重复出现，且不依赖颜色编号大小规律，只依赖可见“颜色是否一致”。
#   cleanliness:
#     - 三层结构仍为规则矩形，外部标记仍是一条整线，不引入局部叠加噪声。
#     - 颜色类别被限制在少数固定值，避免大量无提示的颜色映射。

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
OUTER_BORDER = 0
GREY = 5
ORANGE = 7
LINE_COLORS = (1, 2, 3, 4)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
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


def _component_map_by_color(g: np.ndarray, bg: int) -> dict[int, list[dict]]:
    comp_by_color: dict[int, list[dict]] = defaultdict(list)
    for col in colors_present(g, exclude={bg, OUTER_BORDER, GREY, ORANGE}):
        labc, nc = label(g == int(col), structure=STRUCT4)
        for k in range(1, int(nc) + 1):
            coords = np.argwhere(labc == int(k))
            if coords.size == 0:
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            pts = {(int(r), int(c)) for r, c in coords}
            shape = {(int(r - r0), int(c - c0)) for r, c in pts}
            comp_by_color[int(col)].append(
                {"pts": pts, "shape": shape, "tl": (int(r0), int(c0))}
            )
    return comp_by_color


def _find_containers(g: np.ndarray, bg: int) -> list[dict]:
    H, W = g.shape
    containers = []
    seen = set()
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 2), int(ic0 - 2), int(ir1 + 2), int(ic1 + 2)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        key = (r0, c0, r1, c1)
        if key in seen:
            continue
        seen.add(key)

        inner_ring_r0, inner_ring_c0 = r0 + 1, c0 + 1
        inner_ring_r1, inner_ring_c1 = r1 - 1, c1 - 1
        core = g[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1]
        if core.size == 0:
            continue
        if not bool(np.isin(core, [GREY, ORANGE]).all()):
            continue
        if any(int(g[r, c]) != OUTER_BORDER for r, c in _rect_border_coords(r0, c0, r1, c1)):
            continue

        inner_ring_vals = [int(g[r, c]) for r, c in _rect_border_coords(inner_ring_r0, inner_ring_c0, inner_ring_r1, inner_ring_c1)]
        inner_colors = set(inner_ring_vals)
        if len(inner_colors) != 1:
            continue
        inner_color = int(next(iter(inner_colors)))
        if inner_color in {bg, OUTER_BORDER, GREY, ORANGE}:
            continue

        containers.append(
            {
                "box": (r0, c0, r1, c1),
                "inner_color": inner_color,
                "inner_shape": (core.shape[0], core.shape[1]),
            }
        )
    return containers


def _pick_head_for_container(
    g: np.ndarray, box_: tuple[int, int, int, int], bg: int, comp_by_color: dict[int, list[dict]]
) -> dict | None:
    H, W = g.shape
    r0, c0, r1, c1 = box_
    side_specs = [
        (0, (-1, 0), [(r0 - 1, c) for c in range(c0, c1 + 1)]),
        (1, (1, 0), [(r1 + 1, c) for c in range(c0, c1 + 1)]),
        (2, (0, -1), [(r, c0 - 1) for r in range(r0, r1 + 1)]),
        (3, (0, 1), [(r, c1 + 1) for r in range(r0, r1 + 1)]),
    ]

    candidates = []
    for side, direction, coords in side_specs:
        if any(not (0 <= r < H and 0 <= c < W) for r, c in coords):
            continue
        vals = [int(g[r, c]) for r, c in coords]
        if len(set(vals)) != 1:
            continue
        col = int(vals[0])
        if col in {bg, OUTER_BORDER, GREY, ORANGE}:
            continue
        head_pts = {(int(r), int(c)) for r, c in coords}
        comp0 = None
        for comp in comp_by_color.get(col, []):
            if head_pts <= comp["pts"]:
                comp0 = comp
                break
        if comp0 is None:
            continue
        hr0, hc0, hr1, hc1 = bbox_coords(np.array(list(head_pts), dtype=int))
        head_shape = {(int(r - hr0), int(c - hc0)) for r, c in head_pts}
        candidates.append(
            {
                "color": col,
                "head_pts": head_pts,
                "head_shape": head_shape,
                "head_tl": (int(hr0), int(hc0)),
                "comp_pts": set(comp0["pts"]),
                "dir": direction,
                "side": side,
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _dest_candidates_from_head(
    g: np.ndarray,
    box_: tuple[int, int, int, int],
    head_info: dict,
    comp_by_color: dict[int, list[dict]],
) -> list[dict]:
    H, W = g.shape
    r0, c0, r1, c1 = box_
    hc = int(head_info["color"])
    head_shape = head_info["head_shape"]
    head_pts = set(head_info["head_pts"])
    comp_pts = set(head_info["comp_pts"])

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
            adj_outer = False
            for r, c in pts:
                for dr, dc in DIR4:
                    rr, cc = int(r + dr), int(c + dc)
                    if rr == r0 or rr == r1 or cc == c0 or cc == c1:
                        adj_outer = True
                        break
                if adj_outer:
                    break
            if adj_outer:
                continue
            rs = [r for r, _c in pts]
            cs = [c for _r, c in pts]
            r00, c00 = int(min(rs)), int(min(cs))
            shp = {(int(r - r00), int(c - c00)) for r, c in pts}
            if shp != head_shape:
                continue
            dests.append({"pts": pts, "shape": shp, "tl": (r00, c00)})
    return dests


def _best_destination(head_info: dict, dests: list[dict]) -> dict | None:
    head_tl = head_info["head_tl"]
    direction = head_info["dir"]
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
    return best_dest


def analyze_grid(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(OUTER_BORDER)), key=lambda c: int(cnt[int(c)])))
    containers = _find_containers(g, bg)
    comp_by_color = _component_map_by_color(g, bg)
    info = []
    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = _pick_head_for_container(g, cont["box"], bg, comp_by_color)
        if head_info is None:
            continue
        dests = _dest_candidates_from_head(g, cont["box"], head_info, comp_by_color)
        best_dest = _best_destination(head_info, dests)
        if best_dest is None:
            continue
        tr0 = int(best_dest["tl"][0] - head_info["head_tl"][0])
        tc0 = int(best_dest["tl"][1] - head_info["head_tl"][1])
        head_dist = abs(tr0) + abs(tc0)
        core = g[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1]
        grey_count = int(np.sum(core == GREY))
        match = int(cont["inner_color"]) == int(head_info["color"])
        old_steps = min(int(head_dist), int(grey_count))
        new_steps = old_steps if match else 0
        info.append(
            {
                "box": cont["box"],
                "inner_color": int(cont["inner_color"]),
                "head_color": int(head_info["color"]),
                "match": bool(match),
                "head_side": int(head_info["side"]),
                "head_dist": int(head_dist),
                "grey_count": int(grey_count),
                "old_steps": int(old_steps),
                "new_steps": int(new_steps),
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only the nested containers whose inner colored frame matches the head/destination color.

    Input structure:
    - bg = 6. Containers have an outer black frame (0), a one-cell-thick inner colored frame,
      and a grey/orange interior.
    - A head line sits adjacent to one outer edge; a same-colored destination line lies on the same axis.

    Rule:
    - If inner-frame color == head/destination color, apply the original motion:
      steps = min(head-to-dest distance, grey interior count), move toward destination,
      and convert that many grey cells to orange from the head-facing side inward.
    - If the colors do not match, the container is locked and remains unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(OUTER_BORDER)), key=lambda c: int(cnt[int(c)])))
    containers = _find_containers(g, bg)
    assert containers

    out = g.copy()
    comp_by_color = _component_map_by_color(g, bg)

    for cont in containers:
        box_ = cont["box"]
        r0, c0, r1, c1 = box_

        head_info = _pick_head_for_container(g, box_, bg, comp_by_color)
        if head_info is None:
            continue
        if int(cont["inner_color"]) != int(head_info["color"]):
            continue

        dests = _dest_candidates_from_head(g, box_, head_info, comp_by_color)
        best_dest = _best_destination(head_info, dests)
        if best_dest is None:
            continue

        tr0 = int(best_dest["tl"][0] - head_info["head_tl"][0])
        tc0 = int(best_dest["tl"][1] - head_info["head_tl"][1])
        steps = abs(tr0) + abs(tc0)

        core = g[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1]
        grey_coords = [
            (int(r0 + 2 + rr), int(c0 + 2 + cc))
            for rr in range(core.shape[0])
            for cc in range(core.shape[1])
            if int(core[rr, cc]) == GREY
        ]
        steps = min(int(steps), int(len(grey_coords)))
        if steps <= 0:
            continue

        direction = head_info["dir"]
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
        for r, c in head_info["head_pts"]:
            if int(out[int(r), int(c)]) == int(head_info["color"]):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if 0 <= pr < patch.shape[0] and 0 <= pc < patch.shape[1] and int(patch[pr, pc]) == GREY:
                patch[pr, pc] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_info["head_pts"]:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(head_info["color"])

    return out


def _build_container_patch(
    inner_h: int,
    inner_w: int,
    inner_color: int,
    head_side: int,
    orange_count: int,
) -> np.ndarray:
    oh, ow = inner_h + 4, inner_w + 4
    patch = np.full((oh, ow), BG, dtype=int)
    patch[0, :] = OUTER_BORDER
    patch[-1, :] = OUTER_BORDER
    patch[:, 0] = OUTER_BORDER
    patch[:, -1] = OUTER_BORDER
    patch[1, 1:-1] = inner_color
    patch[-2, 1:-1] = inner_color
    patch[1:-1, 1] = inner_color
    patch[1:-1, -2] = inner_color

    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    fill_order = _fill_order(inner_h, inner_w, head_side)
    for idx in range(int(orange_count)):
        rr, cc = fill_order[idx]
        interior[rr, cc] = ORANGE
    patch[2:-2, 2:-2] = interior
    return patch


def _draw_head_line(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, color: int) -> None:
    if head_side == 0:
        grid[cr - 1, cc : cc + ow] = color
    elif head_side == 1:
        grid[cr + oh, cc : cc + ow] = color
    elif head_side == 2:
        grid[cr : cr + oh, cc - 1] = color
    else:
        grid[cr : cr + oh, cc + ow] = color


def _draw_dest_line(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, gap: int, color: int) -> None:
    if head_side == 0:
        grid[cr - 2 - gap, cc : cc + ow] = color
    elif head_side == 1:
        grid[cr + oh + 1 + gap, cc : cc + ow] = color
    elif head_side == 2:
        grid[cr : cr + oh, cc - 2 - gap] = color
    else:
        grid[cr : cr + oh, cc + ow + 1 + gap] = color


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_blockers = random.randint(1, n_containers - 1)
    blocker_ids = set(random.sample(range(n_containers), n_blockers))
    line_colors = random.sample(LINE_COLORS, n_containers)

    specs = []
    for i in range(n_containers):
        head_side = random.randint(0, 3)
        inner_h = random.randint(1, 6)
        inner_w = random.randint(1, 4)
        inner_area = inner_h * inner_w
        grey = random.randint(1, inner_area)
        initial_orange = inner_area - grey
        gap = random.randint(0, 6)
        line_color = int(line_colors[i])
        is_blocker = i in blocker_ids
        if is_blocker:
            mismatch_choices = [c for c in LINE_COLORS if int(c) != line_color]
            inner_color = int(random.choice(mismatch_choices))
            steps = 0
        else:
            inner_color = line_color
            steps = min(int(grey), int(gap + 1))

        oh, ow = inner_h + 4, inner_w + 4
        if head_side == 0:
            total_h, total_w = 1 + gap + 1 + oh, ow
            cont_r, cont_c = gap + 2, 0
        elif head_side == 1:
            total_h, total_w = oh + 1 + gap + 1, ow
            cont_r, cont_c = 0, 0
        elif head_side == 2:
            total_h, total_w = oh, 1 + gap + 1 + ow
            cont_r, cont_c = 0, gap + 2
        else:
            total_h, total_w = oh, ow + 1 + gap + 1
            cont_r, cont_c = 0, 0

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "inner_area": inner_area,
                "head_side": head_side,
                "grey": grey,
                "initial_orange": initial_orange,
                "gap": gap,
                "line_color": line_color,
                "inner_color": inner_color,
                "steps": steps,
                "is_blocker": is_blocker,
                "outer_h": oh,
                "outer_w": ow,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
            }
        )

    total_area = sum(s["total_h"] * s["total_w"] for s in specs)
    min_side = max(10, int(total_area ** 0.5) + 3)
    max_side = min(26, min_side + 8)
    max_asm_h = max(s["total_h"] for s in specs)
    max_asm_w = max(s["total_w"] for s in specs)

    lo_h = max(10, max_asm_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(10, max_asm_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements = []

    for s in specs:
        th, tw = s["total_h"], s["total_w"]
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
    expected = grid.copy()

    for s, (pr, pc) in zip(specs, placements):
        cr = pr + s["cont_r"]
        cc = pc + s["cont_c"]
        oh, ow = s["outer_h"], s["outer_w"]
        head_side = s["head_side"]
        patch = _build_container_patch(
            s["inner_h"], s["inner_w"], s["inner_color"], head_side, s["initial_orange"]
        )
        paste(grid, patch, cr, cc)
        _draw_head_line(grid, cr, cc, oh, ow, head_side, s["line_color"])
        _draw_dest_line(grid, cr, cc, oh, ow, head_side, s["gap"], s["line_color"])

    expected[:] = grid
    for s, (pr, pc) in zip(specs, placements):
        if s["steps"] <= 0:
            continue
        cr = pr + s["cont_r"]
        cc = pc + s["cont_c"]
        oh, ow = s["outer_h"], s["outer_w"]
        head_side = s["head_side"]
        dr, dc = SIDE_TO_DIR[head_side]

        expected[cr : cr + oh, cc : cc + ow] = BG
        if head_side == 0:
            expected[cr - 1, cc : cc + ow] = BG
        elif head_side == 1:
            expected[cr + oh, cc : cc + ow] = BG
        elif head_side == 2:
            expected[cr : cr + oh, cc - 1] = BG
        else:
            expected[cr : cr + oh, cc + ow] = BG

        nr = cr + dr * s["steps"]
        nc = cc + dc * s["steps"]
        moved_patch = _build_container_patch(
            s["inner_h"],
            s["inner_w"],
            s["inner_color"],
            head_side,
            s["initial_orange"] + s["steps"],
        )
        paste(expected, moved_patch, nr, nc)
        _draw_head_line(expected, nr, nc, oh, ow, head_side, s["line_color"])

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None

    analysis = analyze_grid(grid)
    if len(analysis) != n_containers:
        return None
    movers = sum(1 for x in analysis if x["match"] and x["new_steps"] > 0)
    blockers = sum(1 for x in analysis if (not x["match"]) and x["old_steps"] > 0 and x["new_steps"] == 0)
    if movers < 1 or blockers < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid variant after 400 attempts")
