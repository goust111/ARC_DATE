# Variant of 271d71e2
# Dimension: D12 — 多容器处理顺序 (multi_step)
# Change: 各容器独立处理，互不影响（路径不交叉，无顺序依赖）... → 容器按颜色分组同步移动

from __future__ import annotations

# Design analysis for the D12 variant
# -----------------------------------
# spec:
#   preserved_structure:
#     - 背景仍为 6，容器仍是 0 边框矩形，内部仍只含 5/7，头和目标仍是与边等长的 9 线段。
#     - 每个容器仍沿“头 -> 目标”的轴向平移，内部仍按靠头的一侧开始把灰色逐格充成橙色。
#   core_change:
#     - 原题对每个容器单独算 steps = min(头到目标距离, 灰色数)。
#     - 变体先按“内部橙色占总内部面积的比例”分组。
#     - 同一比例组内，所有容器共享同一个同步步数：
#         group_steps = min(该组所有容器各自的单体可移动步数)
#     - 然后组内每个容器都移动 group_steps，并各自把前 group_steps 个灰格充橙。
#   concrete_small_trace:
#     - 例：两个 1/2 橙色比例的容器 A、B。
#     - A: 个体可动 2 步；B: 个体可动 5 步。
#     - 原题输出：A 走 2，B 走 5。
#     - 本变体输出：A、B 同组同步，只都走 min(2,5)=2 步。
#     - 差异：B 明显被“拖慢”，但充橙顺序、目标方向、目标位置都不变，只是步数改成组同步。
#   complexity_calibration:
#     - 分组键只用少量可见比例（如 1/2、1/3、2/3），避免不可见或过多类别。
#     - generator 固定生成两组，每组两个容器；两组步数不同，且至少一组内存在“追随者”被组内最小步数限制。
#     - 这样 2-4 个示例中人类可直接看到：相同比例的两个容器总是走一样远，不同比例的组可能走不同远。
#   implicit_assumption_audit:
#     - 原 solver 中 `steps = min(distance, len(grey_coords))` 后立即对单容器落盘，隐含“步数只依赖当前容器”。
#       该前提在变体中失效，必须先收集所有容器计划，再按比例分组统一改写 steps。
#     - 原 solver 的头检测、方向检测、目标匹配、充橙顺序都只依赖单容器局部结构，这些前提仍成立，可直接复用。
#     - 原 solver 顺序落盘之所以安全，是因为 generator 保证路径互不重叠。
#       变体仍保持这条生成约束，所以同步只影响步数，不引入覆盖竞争。
#   placement_legality:
#     - 每个容器仍占据一个独立 assembly：`dest + gap + head + container` 的整条走廊。
#     - 同步后的最终位置仍位于原走廊内部，因为组步数不会超过任一成员自己的单体可动步数。
#     - generator 继续要求 assemblies 之间至少留 1 格背景，从而保证输入、输出和移动路径都不相撞。
#   observability:
#     - 分组依据是内部 5/7 的可见比例，而不是颜色编号、扫描顺序等不可见因素。
#     - 两个同组容器允许大小不同，但橙色占比相同；这样解题者需要数格子/看比例，而不是误以为按尺寸分组。
#   cleanliness:
#     - 不叠加额外标记，不改变原题图素种类，视觉上仍是整洁的“容器 + 头线 + 目标线”布局。
#   new_solver_assumptions:
#     - 输入中至少有一对容器共享比例；generator 强制两组各两个容器。
#     - 同组共享步数使用组内最小个体可动步数；generator 保证这不会导致越界或穿过其他对象。

from collections import Counter, defaultdict
from math import gcd
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
RATIO_POOL = [(1, 2), (1, 3), (2, 3), (1, 4), (3, 4)]


def _ratio_key(orange_count: int, total_count: int) -> tuple[int, int]:
    g = gcd(int(orange_count), int(total_count))
    return (int(orange_count) // g, int(total_count) // g)


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


def _direction_to_side(direction: tuple[int, int]) -> int:
    mapping = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    return mapping[direction]


def _extract_container_plans(grid: np.ndarray) -> tuple[int, list[dict]]:
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
        containers.append({"box": (r0, c0, r1, c1), "inner": inner.copy()})

    assert containers

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

    plans = []
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
            if d["shape"] != head_shape or d["pts"] == comp_pts:
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

        inner = cont["inner"]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        orange_count = int(np.count_nonzero(inner == ORANGE))
        total_count = int(inner.size)
        max_steps = min(int(best), int(len(grey_coords)))
        head_side = _direction_to_side(direction)
        order_local = _interior_fill_order(inner.shape[0], inner.shape[1], head_side)
        plans.append(
            {
                "box": (r0, c0, r1, c1),
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                "head_color": hc,
                "head_pts": head_pts,
                "direction": direction,
                "individual_steps": max_steps,
                "ratio_key": _ratio_key(orange_count, total_count),
                "grey_count": len(grey_coords),
                "orange_count": orange_count,
                "fill_order_local": order_local,
            }
        )

    return bg, plans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Same geometry as 271d71e2, but containers with the same interior orange proportion
    move synchronously: each ratio-group uses the minimum individual move capacity in that group.
    """
    g = np.array(grid, dtype=int)
    bg, plans = _extract_container_plans(g)
    out = g.copy()

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for plan in plans:
        grouped[plan["ratio_key"]].append(plan)

    group_steps = {
        ratio: min(int(p["individual_steps"]) for p in group_plans)
        for ratio, group_plans in grouped.items()
    }

    for plan in plans:
        r0, c0, r1, c1 = plan["box"]
        steps = int(group_steps[plan["ratio_key"]])
        dr, dc = plan["direction"]
        tr = int(dr * steps)
        tc = int(dc * steps)

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in plan["head_pts"]:
            if int(out[int(r), int(c)]) == int(plan["head_color"]):
                out[int(r), int(c)] = int(bg)

        patch = plan["patch"].copy()
        start = int(plan["orange_count"])
        for rr, cc in plan["fill_order_local"][start : start + steps]:
            if int(patch[1 + rr, 1 + cc]) == GREY:
                patch[1 + rr, 1 + cc] = ORANGE
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in plan["head_pts"]:
            nr = int(r + tr)
            nc = int(c + tc)
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = int(plan["head_color"])

    return out


def _pick_inner_dims(ratio: tuple[int, int], min_grey: int, avoid: tuple[int, int] | None = None):
    num, den = ratio
    options = []
    for inner_h in range(1, 9):
        for inner_w in range(1, 6):
            area = inner_h * inner_w
            if area % den != 0:
                continue
            orange = area * num // den
            grey = area - orange
            if orange <= 0 or grey < min_grey:
                continue
            if avoid is not None and (inner_h, inner_w) == avoid:
                continue
            options.append((inner_h, inner_w, orange, grey))
    if not options:
        return None
    return random.choice(options)


def _build_assemblies(specs: list[dict]) -> list[dict]:
    assemblies = []
    for s in specs:
        oh, ow = s["outer_h"], s["outer_w"]
        hs = s["head_side"]
        gap = s["gap"]
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
        assemblies.append({**s, "total_h": total_h, "total_w": total_w, "cont_r": cont_r, "cont_c": cont_c})
    return assemblies


def _draw_input(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> None:
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = np.full((a["inner_h"], a["inner_w"]), GREY, dtype=int)
        fill_order = _interior_fill_order(a["inner_h"], a["inner_w"], hs)
        for idx_cell in range(a["initial_orange"]):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        grid[cr + 1 : cr + 1 + a["inner_h"], cc + 1 : cc + 1 + a["inner_w"]] = interior

        if hs == 0:
            grid[cr - 1, cc : cc + ow] = HEAD_COLOR
            grid[cr - 2 - a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 1:
            grid[cr + oh, cc : cc + ow] = HEAD_COLOR
            grid[cr + oh + 1 + a["gap"], cc : cc + ow] = HEAD_COLOR
        elif hs == 2:
            grid[cr : cr + oh, cc - 1] = HEAD_COLOR
            grid[cr : cr + oh, cc - 2 - a["gap"]] = HEAD_COLOR
        else:
            grid[cr : cr + oh, cc + ow] = HEAD_COLOR
            grid[cr : cr + oh, cc + ow + 1 + a["gap"]] = HEAD_COLOR


def _manual_expected(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        steps = int(a["group_steps"])
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]
        dr, dc = SIDE_TO_DIR[hs]

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

        fill_order = _interior_fill_order(a["inner_h"], a["inner_w"], hs)
        interior = np.full((a["inner_h"], a["inner_w"]), GREY, dtype=int)
        for idx_cell in range(a["initial_orange"] + steps):
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
    return expected


def _gen_attempt() -> np.ndarray | None:
    ratios = random.sample(RATIO_POOL, 2)
    group_targets = random.sample([1, 2, 3, 4], 2)

    specs = []
    for group_id, (ratio, group_step) in enumerate(zip(ratios, group_targets)):
        follower_extra = random.randint(1, 3)
        lead_dims = _pick_inner_dims(ratio, min_grey=group_step)
        if lead_dims is None:
            return None
        follow_dims = _pick_inner_dims(ratio, min_grey=group_step + follower_extra, avoid=lead_dims[:2])
        if follow_dims is None:
            follow_dims = _pick_inner_dims(ratio, min_grey=group_step + follower_extra)
        if follow_dims is None:
            return None

        for member_idx, dims in enumerate((lead_dims, follow_dims)):
            inner_h, inner_w, orange, grey = dims
            head_side = random.randint(0, 3)
            outer_h = inner_h + 2
            outer_w = inner_w + 2

            if member_idx == 0:
                distance = group_step
            else:
                upper = min(8, grey)
                if upper <= group_step:
                    return None
                distance = random.randint(group_step + 1, upper)

            gap = distance - 1
            individual_steps = min(grey, distance)
            if member_idx == 0 and individual_steps != group_step:
                return None
            if member_idx == 1 and individual_steps <= group_step:
                return None

            specs.append(
                {
                    "group_id": group_id,
                    "ratio_key": ratio,
                    "group_steps": group_step,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "head_side": head_side,
                    "gap": gap,
                    "distance": distance,
                    "initial_orange": orange,
                    "grey": grey,
                    "individual_steps": individual_steps,
                }
            )

    random.shuffle(specs)
    assemblies = _build_assemblies(specs)

    total_area = sum(a["total_h"] * a["total_w"] for a in assemblies)
    min_side = max(10, int(total_area**0.5) + 3)
    max_side = min(26, min_side + 8)
    max_asm_h = max(a["total_h"] for a in assemblies)
    max_asm_w = max(a["total_w"] for a in assemblies)

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
    _draw_input(grid, assemblies, placements)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _manual_expected(grid, assemblies, placements)
    if not np.array_equal(output, expected):
        return None

    # Reject degenerate cases where the new grouped rule accidentally matches the old per-container rule.
    if all(int(a["individual_steps"]) == int(a["group_steps"]) for a in assemblies):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid grouped instance after 500 attempts")
