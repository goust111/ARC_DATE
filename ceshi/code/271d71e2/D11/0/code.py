# Variant of 271d71e2
# Dimension: D11 — 头部与目标线的形状关系 (meta)
# Change: 头部和目标线形状完全相同（同长度的水平或垂直线段），颜色相同（maroon/9）... → 目标线更长或更短

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


# Design analysis for the D11 variant
# ----------------------------------
# variant_spec:
#   visual:
#     - 容器、内部灰/橙充能规则、头部颜色和移动方向都保持原题不变。
#     - 唯一核心变化是：目标线仍与头部同色、同朝向、处在同一移动轴上，
#       但长度不再等于头部长度，而是固定比头部更长或更短。
#     - 为了让“不是靠形状匹配”这件事肉眼可见，生成器强制所有目标线长度 != 头部长度。
#     - 为了仍然可配对，头部与目标线采用“中心对齐”：
#       水平线共享同一个列中心，垂直线共享同一个行中心。
#   concrete_trace:
#     - 原逻辑：solver 先抽出贴着容器的 maroon 线当 head，再在同色组件里找
#       `shape == head_shape` 的另一条线作为 dest，然后用两条线左上角差值算距离。
#     - 新逻辑：head 提取方式不变；dest 改为“同色、同向、中心对齐、位于 head 前方的最近线段”。
#       距离不再看左上角，而只看移动轴上的行差/列差。
#     - 小例子：若容器顶边宽 5，则头部是长度 5 的水平线；目标线可以是长度 3 或 7，
#       但其中心列与头部中心列一致。容器仍然向上移动 `gap+1` 步，直到头部那一行
#       与目标线所在行重合；若灰格数不足，则仍按原题提前停下。
#   complexity_check:
#     - 没有引入新颜色或多种配对规则，只增加一条“中心对齐”线索，复杂度接近原题。
#     - 目标长度只在少量档位里变化（head_len ±2/±4，保持同奇偶性），差异明显但不杂乱。
#     - 每个样例中会重复出现多个“同色但不同长度”的头/目标配对，人类可以从少量例子归纳。
#   implicit_assumptions_review:
#     - 原 `d["shape"] != head_shape` 这一过滤依赖“目标和头完全同形”，在变体中失效，必须删除。
#     - 原 `tc0 == 0` / `tr0 == 0` 假设两条线左上角在正交轴上完全重合；
#       目标长度改变后，即便中心对齐，左上角也会偏移，因此距离必须改成“同中心 + 轴向差”。
#     - 原生成器把目标线宽高直接设成容器边长；现在必须额外记录 `dest_len`，并调整装配盒尺寸。
#     - 新 solver 新增了一个隐式前提：head 与 dest 必须是 1 格厚的实心直线，且中心严格对齐。
#       生成器会显式保证这一点。
#   placement_legality:
#     - 生成器仍使用单个 assembly 放置每个对象组，并在 assembly 外围保留至少 1 格背景间隔。
#     - assembly 的横/纵跨度改成 `max(head_len, dest_len)`，这样较长的目标线也不会越界。
#     - 容器移动最多 `gap+1` 步，最终位置仍留在 assembly 内部，因此输入与输出都合法。
#   observability_and_cleanliness:
#     - 视觉上，头部总是整条贴边线；目标线在同一轴上但明显更长或更短，变化可直接看见。
#     - 中心对齐让“哪条线对应哪只容器”仍然清楚，不需要额外标记，也不会在同一区域叠层过多。
#     - 只有 maroon 线长度变化，灰→橙充能顺序保持原题，避免同时引入多种独立变化。


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


def _line_info_from_pts(pts: set[tuple[int, int]]) -> dict | None:
    if not pts:
        return None
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h == 1 and len(pts) == w:
        return {
            "pts": set(pts),
            "tl": (int(r0), int(c0)),
            "axis": "h",
            "line_coord": int(r0),
            "center2": int(2 * c0 + w - 1),
            "length": int(w),
        }
    if w == 1 and len(pts) == h:
        return {
            "pts": set(pts),
            "tl": (int(r0), int(c0)),
            "axis": "v",
            "line_coord": int(c0),
            "center2": int(2 * r0 + h - 1),
            "length": int(h),
        }
    return None


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


def _choose_dest_len(head_len: int) -> int:
    choices = []
    for delta in (2, 4):
        if head_len - delta >= 1:
            choices.append(head_len - delta)
        if head_len + delta <= 12:
            choices.append(head_len + delta)
    assert choices
    return int(random.choice(choices))


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
            pts = {(int(r), int(c)) for r, c in coords}
            r0, c0, _r1, _c1 = bbox_coords(coords)
            comp_by_color[int(col)].append(
                {
                    "pts": pts,
                    "tl": (int(r0), int(c0)),
                    "line": _line_info_from_pts(pts),
                }
            )

    def pick_head_for_container(box_: tuple[int, int, int, int]) -> dict | None:
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
        head_line = _line_info_from_pts(head_pts)
        if head_line is None:
            return None

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

        return {
            "color": hc,
            "head_pts": head_pts,
            "head_tl": head_line["tl"],
            "comp_pts": set(comp0["pts"]),
            "dir": direction,
            "axis": head_line["axis"],
            "center2": head_line["center2"],
            "line_coord": head_line["line_coord"],
        }

    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = pick_head_for_container((r0, c0, r1, c1))
        if head_info is None:
            continue

        hc = int(head_info["color"])
        head_pts = set(head_info["head_pts"])
        comp_pts = set(head_info["comp_pts"])
        direction = head_info["dir"]
        head_axis = head_info["axis"]
        head_center2 = int(head_info["center2"])
        head_line_coord = int(head_info["line_coord"])
        if direction == (0, 0):
            continue

        dests = []
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] == comp_pts:
                continue
            info = comp["line"]
            if info is None or info["axis"] != head_axis:
                continue
            dests.append(info)

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
                adj_border = False
                for rr, cc in pts:
                    for dr, dc in DIR4:
                        rrr, ccc = int(rr + dr), int(cc + dc)
                        if 0 <= rrr < H and 0 <= ccc < W and int(g[rrr, ccc]) == BORDER:
                            adj_border = True
                            break
                    if adj_border:
                        break
                if adj_border:
                    continue
                info = _line_info_from_pts(pts)
                if info is None or info["axis"] != head_axis:
                    continue
                dests.append(info)

        best_dist = None
        best_dest = None
        for d in dests:
            if int(d["center2"]) != head_center2:
                continue
            delta = int(d["line_coord"] - head_line_coord)
            if direction[0] == 1 and delta <= 0:
                continue
            if direction[0] == -1 and delta >= 0:
                continue
            if direction[1] == 1 and delta <= 0:
                continue
            if direction[1] == -1 and delta >= 0:
                continue
            dist = abs(delta)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dest = d
        if best_dest is None:
            continue

        steps = int(best_dist)
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(steps, int(len(grey_coords)))

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
        for rr, cc in head_pts:
            if int(out[int(rr), int(cc)]) == hc:
                out[int(rr), int(cc)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill:
            pr = int(rr - r0)
            pc = int(cc - c0)
            if int(patch[pr, pc]) == GREY:
                patch[pr, pc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for rr, cc in head_pts:
            rrr = int(rr + tr)
            ccc = int(cc + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = hc

    return out


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        head_len = outer_w if head_side in (0, 1) else outer_h
        dest_len = _choose_dest_len(head_len)

        if is_decoy:
            grey = 0
            initial_orange = inner_area
            gap = random.randint(0, 6)
            steps = 0
        else:
            grey = random.randint(1, inner_area)
            initial_orange = inner_area - grey
            gap = random.randint(0, 10)
            steps = min(grey, gap + 1)

        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "head_side": head_side,
                "head_len": head_len,
                "dest_len": dest_len,
                "gap": gap,
                "steps": steps,
                "grey": grey,
                "initial_orange": initial_orange,
                "is_decoy": is_decoy,
            }
        )

    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]
        dest_len = s["dest_len"]

        if hs in (0, 1):
            span_w = max(ow, dest_len)
            cont_c = (span_w - ow) // 2
            dest_c = (span_w - dest_len) // 2
            if hs == 0:
                total_h = 1 + gap + 1 + oh
                total_w = span_w
                cont_r = gap + 2
                dest_r = 0
            else:
                total_h = oh + 1 + gap + 1
                total_w = span_w
                cont_r = 0
                dest_r = oh + 1 + gap
            assemblies.append(
                {
                    **s,
                    "total_h": total_h,
                    "total_w": total_w,
                    "cont_r": cont_r,
                    "cont_c": cont_c,
                    "dest_r": dest_r,
                    "dest_c": dest_c,
                }
            )
        else:
            span_h = max(oh, dest_len)
            cont_r = (span_h - oh) // 2
            dest_r = (span_h - dest_len) // 2
            if hs == 2:
                total_h = span_h
                total_w = 1 + gap + 1 + ow
                cont_c = gap + 2
                dest_c = 0
            else:
                total_h = span_h
                total_w = ow + 1 + gap + 1
                cont_c = 0
                dest_c = ow + 1 + gap
            assemblies.append(
                {
                    **s,
                    "total_h": total_h,
                    "total_w": total_w,
                    "cont_r": cont_r,
                    "cont_c": cont_c,
                    "dest_r": dest_r,
                    "dest_c": dest_c,
                }
            )

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
        inner_h, inner_w = a["inner_h"], a["inner_w"]
        initial_orange = a["initial_orange"]
        dest_len = a["dest_len"]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]
        dr0 = pr + a["dest_r"]
        dc0 = pc + a["dest_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            rr, cc2 = fill_order[idx_cell]
            interior[rr, cc2] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[dr0, dc0 : dc0 + dest_len] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[dr0, dc0 : dc0 + dest_len] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[dr0 : dr0 + dest_len, dc0] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[dr0 : dr0 + dest_len, dc0] = HEAD_COLOR

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
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")
