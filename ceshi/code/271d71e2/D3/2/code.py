# Variant of 271d71e2
# Dimension: D3 — 轨迹着色模式 (trajectory)
# Change: recolor — 灰色像素从头侧向内逐个变为橙色(7)，连续着色... → 从远侧向头侧着色（反向填充）

"""
Design analysis for the D3 variant
==================================

variant_spec:
  kept_invariant:
    - 背景仍为 6，容器仍为黑框矩形，内部只含 5/7。
    - 头部与目标仍是同色、同形、共轴的细线标记，颜色仍固定为 9。
    - 平移方向、平移步数、步数上限、同色匹配、连通分量拆分逻辑都保持原题不变。
    - decoy 仍允许存在：当灰格数为 0 时容器不移动。
  changed_core:
    - 原题把将被“充能”的灰格按“靠头侧优先”的顺序排序。
    - 本变体改为按“远离头侧优先”的顺序排序。
    - 为了让变化在样例中可见，generator 也把输入中的初始橙色区域放在远侧，
      这样 output 会表现为“橙色从容器深处朝头部生长”。

a_concretization_and_trace:
  object_layout:
    - 若头在右侧，原题的填充前沿从最右列向左推进；本变体改成从最左列向右推进。
    - 若头在上侧，原题从最上行向下推进；本变体改成从最下行向上推进。
    - 其他两个方向同理，只反转“离头的深度轴”，不改变同一行/列内部的扫描风格。
  concrete_trace:
    - 例：头在右，内区为 2x3，输入灰橙分布按本变体是 [7 5 5; 7 5 5]，
      即左列先橙，右侧更靠头的列仍灰。
    - 若 steps=2：
      原题会把最右列的两个灰格变橙，得到 [7 5 7; 7 5 7]。
      本变体会把中列的两个灰格变橙，得到 [7 7 5; 7 7 5]。
    - 差异出现在“选择哪一批灰格着色”这一环；容器整体平移距离不变。

b_complexity_calibration:
  - 仅反转一个排序维度，不引入新对象、新颜色、新匹配关系。
  - 为避免退化成肉眼不可分辨的情况，generator 强制至少一个 mover 满足：
    inner_area > 1 且 steps < grey。
    这保证同一输入在原题/变体下会选中不同的灰格。
  - 仍保留 2-4 个容器和 0-1 个 decoy，样例内有重复支撑但不会太拥挤。

c_implicit_assumption_audit:
  solver_segments:
    - 容器检测代码只要求内部是 5/7 的矩形区域，不依赖 7 必须贴着头侧。
      这一前提在变体中仍成立。
    - head/destination 匹配只依赖同色、同形、共轴和相对位置，也不依赖填充方向。
    - steps=min(distance, grey_count) 只依赖灰格数量，不依赖灰格空间分布。
    - 真正依赖旧属性的是“choose which greys to fill in order from head side inward”这段排序；
      这里必须改成“from far side toward head side”。
    - patch 复制后只把选中的灰格改成橙，保留已有橙格。因为 generator 改成远侧预填充，
      这里仍然安全，不会错误覆盖已存在的橙色轨迹。

d_placement_legality:
  - assembly 结构仍是 dest + gap + head + container 的同轴布局。
  - 生成时使用 1-cell 背景缓冲区，避免不同 assembly 在输入中相邻或重叠。
  - 由于 steps <= gap + 1，container+head 向目标移动后仍停留在 assembly 预留范围内。
  - 所有头、目标和容器都保持整行/整列对齐，因此 solver 的共轴筛选仍可稳定命中。

e_visual_observability:
  - 输入就能看到橙色预填充在远侧，而非头侧。
  - 输出中橙色继续向头方向推进，与原题相反，2-3 组样例即可观察到模式一致性。
  - generator 额外拒绝“新旧规则输出完全一致”的样本，防止退化。

f_visual_cleanliness:
  - 仍然只使用 magenta 背景、黑框容器、灰/橙内部和 maroon 标记，图面层次简单。
  - assembly 之间保留背景间隔，避免移动后互相贴边导致难以分辨。

g_color_change_inference:
  - 唯一颜色变化仍是 5 -> 7，且变化与“距离头部的深度次序”强绑定。
  - 没有新增颜色映射，也不使用颜色数值做隐藏排序。

post_change_solver_assumptions_checked:
  - 每个 mover 至少有一个可匹配的 head 和同形 dest：generator 保持原约束。
  - 反向填充仍假设内部灰格在“远侧到头侧”的序列上可线性排序：矩形内区保证这一点。
  - patch 移动后不会越界或撞上其他对象：assembly 放置和背景缓冲保证这一点。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

# Direction mapping: head_side -> (dr, dc) movement direction
# head on top -> container moves up; head on bottom -> moves down, etc.
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
DIR_TO_SIDE = {v: k for k, v in SIDE_TO_DIR.items()}


def _ordered_cells(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    head_side: int | None = None,
    direction: tuple[int, int] | None = None,
    from_far_side: bool,
) -> list[tuple[int, int]]:
    """
    Order inner cells along the motion depth axis.

    The variant only reverses which side starts charging. Within a single row/column
    strip we keep the original tie-breaker so the change is isolated to D3.
    """
    if head_side is None:
        assert direction is not None
        head_side = DIR_TO_SIDE[direction]

    ordered = list(cells)
    if head_side == 3:  # head on right, move right
        key = (lambda p: (p[1], p[0])) if from_far_side else (lambda p: (-p[1], p[0]))
    elif head_side == 2:  # head on left, move left
        key = (lambda p: (-p[1], -p[0])) if from_far_side else (lambda p: (p[1], -p[0]))
    elif head_side == 1:  # head on bottom, move down
        key = (lambda p: (p[0], -p[1])) if from_far_side else (lambda p: (-p[0], -p[1]))
    else:  # head on top, move up
        key = (lambda p: (-p[0], p[1])) if from_far_side else (lambda p: (p[0], p[1]))
    ordered.sort(key=key)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging gray cells to orange.

    Variant change:
    - Containers still move by min(head-to-dest distance, grey count).
    - But the charged grey cells are selected from the far side toward the head side.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # True background is dominant non-border color (border can dominate in some cases).
    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

    # Detect containers as rectangles enclosing GREY/ORANGE with a BORDER frame.
    containers = []
    for pts in components_mask((g == GREY) | (g == ORANGE), connectivity=4):
        if pts.size == 0:
            continue
        ir0, ic0, ir1, ic1 = bbox_coords(pts)
        r0, c0, r1, c1 = int(ir0 - 1), int(ic0 - 1), int(ir1 + 1), int(ic1 + 1)
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            continue
        if any(int(g[r, c]) != BORDER for r, c in rect_border_coords(r0, c0, r1, c1)):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (r0, c0, r1, c1)})

    assert containers

    # head + destination are in some non-bg, non-border color (not GREY/ORANGE)
    out = g.copy()

    all_head_colors = [c for c in colors_present(g, exclude={bg, BORDER, GREY, ORANGE})]

    # Precompute components for each head color.
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
        for r, c in rect_border_coords(r0, c0, r1, c1):
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

        # Split head vs destination even if same-color pixels touch:
        # head pixels are those with minimal manhattan distance to this container border.
        border_pts = list(rect_border_coords(r0, c0, r1, c1))
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

        # Add split-from-same-component destinations when dest touches head by color.
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

        # Variant change: fill from the far side toward the head side.
        order = _ordered_cells(grey_coords, direction=direction, from_far_side=True)
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


def _build_input_interior(inner_h: int, inner_w: int, head_side: int, initial_orange: int) -> np.ndarray:
    """Initial orange region is placed on the far side so the trajectory is visually reversed."""
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    order = _ordered_cells(
        [(r, c) for r in range(inner_h) for c in range(inner_w)],
        head_side=head_side,
        from_far_side=True,
    )
    for r, c in order[:initial_orange]:
        interior[r, c] = ORANGE
    return interior


def _manual_expected(
    grid: np.ndarray,
    assemblies: list[dict],
    placements: list[tuple[int, int]],
    *,
    from_far_side: bool,
) -> np.ndarray:
    """Mirror the solver exactly on the already-built input grid, changing only the charge order."""
    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
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

        patch = grid[cr : cr + oh, cc : cc + ow].copy()
        inner = patch[1:-1, 1:-1].copy()
        grey_cells = [
            (r, c)
            for r in range(inner.shape[0])
            for c in range(inner.shape[1])
            if int(inner[r, c]) == GREY
        ]
        order = _ordered_cells(grey_cells, head_side=hs, from_far_side=from_far_side)
        for r, c in order[:steps]:
            inner[r, c] = ORANGE
        patch[1:-1, 1:-1] = inner
        expected[nr : nr + oh, nc : nc + ow] = patch

        if hs == 0:
            expected[nr - 1, nc : nc + ow] = HEAD_COLOR
        elif hs == 1:
            expected[nr + oh, nc : nc + ow] = HEAD_COLOR
        elif hs == 2:
            expected[nr : nr + oh, nc - 1] = HEAD_COLOR
        else:
            expected[nr : nr + oh, nc + ow] = HEAD_COLOR

    return expected


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])  # 0 or 1, slight bias toward 0
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers
    forced_observable_idx = random.randrange(n_movers)

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)

        if i == forced_observable_idx:
            while True:
                inner_h = random.randint(1, 8)
                inner_w = random.randint(1, 5)
                inner_area = inner_h * inner_w
                if inner_area > 1:
                    break
            grey = random.randint(2, inner_area)
            max_gap = min(10, grey - 2)
            gap = random.randint(0, max_gap)
            steps = gap + 1  # guaranteed < grey
            initial_orange = inner_area - grey
        else:
            inner_h = random.randint(1, 8)
            inner_w = random.randint(1, 5)
            inner_area = inner_h * inner_w
            if is_decoy:
                initial_orange = inner_area
                grey = 0
                gap = random.randint(0, 6)
                steps = 0
            else:
                grey = random.randint(1, inner_area)
                initial_orange = inner_area - grey
                gap = random.randint(0, 10)
                steps = min(grey, gap + 1)

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        head_len = outer_w if head_side in (0, 1) else outer_h

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "gap": gap,
                "steps": steps,
                "initial_orange": initial_orange,
                "grey": grey,
                "head_len": head_len,
                "is_decoy": is_decoy,
            }
        )

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]

        if hs == 0:  # top: dest above head above container
            total_h = 1 + gap + 1 + oh
            total_w = ow
            cont_r = gap + 2
            cont_c = 0
        elif hs == 1:  # bottom: container above head above dest
            total_h = oh + 1 + gap + 1
            total_w = ow
            cont_r = 0
            cont_c = 0
        elif hs == 2:  # left: dest left of head left of container
            total_h = oh
            total_w = 1 + gap + 1 + ow
            cont_r = 0
            cont_c = gap + 2
        else:  # right: container left of head left of dest
            total_h = oh
            total_w = ow + 1 + gap + 1
            cont_r = 0
            cont_c = 0

        assemblies.append(
            {
                **s,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
            }
        )

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(8, int(total_area ** 0.5) + 2)
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

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        gap = a["gap"]
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = _build_input_interior(inner_h, inner_w, hs, initial_orange)
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 2 - gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + gap, cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 2 - gap] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + 1 + gap] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _manual_expected(grid, assemblies, placements, from_far_side=True)
    if not np.array_equal(output, expected):
        return None

    old_expected = _manual_expected(grid, assemblies, placements, from_far_side=False)
    if np.array_equal(expected, old_expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
