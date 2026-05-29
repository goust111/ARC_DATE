# Variant of 271d71e2
# Dimension: D2 — 移动距离编码方式 (dst)
# Change: distance_encoding — 步数 = min(头到目标距离, 灰色像素数)，两个约束取较小值... → 纯灰色数量驱动
#
# Design analysis for this variant
# a) 具体化与行为追踪
# - 视觉对象不变：洋红背景(6)、黑框容器(0)、内部灰/橙(5/7)、贴着容器一侧的酒红色头线(9)、
#   沿同轴方向更远处的同色目标线(9)。
# - 唯一核心变化是“容器平移步数”的来源。原题先找头线方向，再用
#   `steps = min(head_to_dest_distance, grey_count)`；本变体改为
#   `steps = grey_count`，目标线只负责给出朝向，不再截断位移。
# - 小例子：若头在线右侧，目标线距头 2 格，而容器内有 5 个灰格。
#   原题右移 2 格，只把最右侧 2 个灰格变橙；本变体右移 5 格，并把 5 个灰格全变橙。
#   若目标线更远，例如距头 7 格而灰格 3 个，两题都右移 3 格；所以 generator 必须经常生成
#   `grey_count > head_to_dest_distance` 的样例，才能让新规则在样例里可见。
# - 由于每移动一步就消耗 1 个灰格，而步数现在恰好等于灰格数，所以所有“会移动”的容器在输出中
#   都会变成全橙内部；目标线可能被越过，也可能仍在前方。
#
# b) 复杂度校准
# - 不改头/目标的配对机制，不引入新颜色或新形状，避免把题从“距离编码变化”扩展成别的维度。
# - 但 generator 不能长期退化为 `grey_count == head_to_dest_distance`。因此每个样例至少保留一个
#   “越过目标线”的 mover；若有多个 mover，则再混入“停在线前”或“正好到线”的情况。
# - 自由度仍受控：对象数维持 2-4，方向仍是四向，目标仍是与头同形的整条线。
#
# c) 隐式假设审查（与 main() 相关）
# - 容器检测逻辑只依赖黑框+灰橙内部，不依赖旧的距离约束，保持成立。
# - `pick_head_for_container()` 仍假设容器旁边只有一个真正贴边的头线；generator 必须继续保证这点。
# - 目标匹配代码仍会从同色同形且位于前方的候选里取最近者。虽然目标不再限制步数，但它仍必须是
#   “最近的合法方向标记”；否则 solver 会朝错误的线解释方向。
# - 原 solver 里 `steps = min(distance, grey_count)` 的“distance 截断”此前顺带保证了不会越过目标。
#   现在去掉这个约束后，`paste(..., clip=True)` 若遇到越界会静默裁剪，这是新风险。因此 generator
#   必须为完整的 `grey_count` 位移预留边界空间，确保 clip 实际上不会触发。
# - 现在最终容器可能覆盖目标线。这在逻辑上是允许的：目标只在输入中承担指示方向的作用，输出里若被
#   移动后的容器或头覆盖，不再需要保留可见性。
#
# d) 放置合法性
# - 为每个对象预先计算“输入对象 + 目标线 + 输出对象”的并集包围盒，并用该包围盒做随机摆放。
#   这样可同时保证输入和输出都不越界、不同对象最终不会互相覆盖。
# - 仍保留 1 格背景缓冲，减少头线/目标线与别的对象发生意外接触。
# - 生成后继续用手写 expected 与 `main(input)` 比较，若有任何歧义配对或放置问题，直接丢弃重试。
#
# e) 视觉可观察性
# - 变体最关键的可观察证据是“有的容器越过目标线仍继续前进”。因此生成器强制至少一个 overrun mover。
# - 若同一题有多个 mover，则尽量混入非 overrun 案例，帮助人类看出“目标线只给方向，不给终点”。
#
# f) 可视化整洁度
# - 不叠加新标记，不改变配色。对象仍是一条线 + 一个矩形容器的简单组合。
# - 通过为整个输入/输出并集留出独占区域，避免不同对象的路径互相穿插导致画面杂乱。
#
# g) 颜色变化可推断性
# - 颜色角色完全继承原题：灰=待消耗，橙=已充能，酒红=头/目标，黑=边框，洋红=背景。
# - 没有新增颜色映射；唯一变化是灰格全数决定位移，因此移动容器最终总是全橙，这和“越线继续前进”
#   共同构成稳定、可见的结构线索。
#
# Solver/generator assumptions introduced by the new rule
# - 每个可移动容器都必须存在一个位于其运动方向前方、且与头线同形的目标线。
# - 该目标线只需最近且同轴，不需要覆盖最终停点。
# - 生成器必须保证 `grey_count` 步完整位移后，容器和头都仍在网格内。

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

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
MAX_STEPS = 8
MAX_GAP = 7


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers, charging gray cells to orange.

    Variant rule:
    - The destination marker still indicates direction.
    - Movement distance is driven purely by the number of gray interior cells.
    - steps = gray_count, not min(distance_to_target, gray_count).
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    cnt = Counter(map(int, g.flatten()))
    bg = int(max((c for c in cnt.keys() if int(c) != int(BORDER)), key=lambda c: int(cnt[int(c)])))

    def rect_border_coords(r0: int, c0: int, r1: int, c1: int):
        for c in range(c0, c1 + 1):
            yield (r0, c)
            yield (r1, c)
        for r in range(r0 + 1, r1):
            yield (r, c0)
            yield (r, c1)

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
        comp0 = None
        for comp in comp_by_color.get(hc, []):
            if comp["pts"] & seed_pts:
                comp0 = comp
                break
        if comp0 is None:
            return None

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
        steps = int(len(grey_coords))
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


def _relative_union_bbox(outer_h: int, outer_w: int, head_side: int, gap: int, steps: int) -> tuple[int, int, int, int]:
    dr, dc = SIDE_TO_DIR[head_side]
    boxes = [(0, 0, outer_h - 1, outer_w - 1), (dr * steps, dc * steps, dr * steps + outer_h - 1, dc * steps + outer_w - 1)]

    if head_side == 0:
        boxes.extend([(-1, 0, -1, outer_w - 1), (-2 - gap, 0, -2 - gap, outer_w - 1), (-1 - steps, 0, -1 - steps, outer_w - 1)])
    elif head_side == 1:
        boxes.extend([(outer_h, 0, outer_h, outer_w - 1), (outer_h + 1 + gap, 0, outer_h + 1 + gap, outer_w - 1), (outer_h + steps, 0, outer_h + steps, outer_w - 1)])
    elif head_side == 2:
        boxes.extend([(0, -1, outer_h - 1, -1), (0, -2 - gap, outer_h - 1, -2 - gap), (0, -1 - steps, outer_h - 1, -1 - steps)])
    else:
        boxes.extend([(0, outer_w, outer_h - 1, outer_w), (0, outer_w + 1 + gap, outer_h - 1, outer_w + 1 + gap), (0, outer_w + steps, outer_h - 1, outer_w + steps)])

    r0 = min(b[0] for b in boxes)
    c0 = min(b[1] for b in boxes)
    r1 = max(b[2] for b in boxes)
    c1 = max(b[3] for b in boxes)
    return int(r0), int(c0), int(r1), int(c1)


def _choose_relation_roles(n_movers: int) -> list[str]:
    roles = ["overrun"]
    if n_movers >= 2:
        roles.append(random.choice(["underrun", "equal"]))
    while len(roles) < n_movers:
        roles.append(random.choice(["overrun", "underrun", "equal"]))
    random.shuffle(roles)
    return roles


def _sample_mover_spec(relation: str) -> dict | None:
    for _ in range(40):
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        inner_area = inner_h * inner_w
        step_cap = min(inner_area, MAX_STEPS)
        if relation == "overrun" and step_cap < 2:
            continue

        head_side = random.randint(0, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if relation == "overrun":
            grey = random.randint(2, step_cap)
            gap = random.randint(0, min(MAX_GAP, grey - 2))
        elif relation == "equal":
            grey = random.randint(1, step_cap)
            gap = grey - 1
        else:
            grey = random.randint(1, min(step_cap, MAX_GAP))
            gap = random.randint(grey, MAX_GAP)

        initial_orange = inner_area - grey
        bbox = _relative_union_bbox(outer_h, outer_w, head_side, gap, grey)
        total_h = bbox[2] - bbox[0] + 1
        total_w = bbox[3] - bbox[1] + 1

        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "steps": grey,
            "initial_orange": initial_orange,
            "grey": grey,
            "is_decoy": False,
            "relation": relation,
            "bbox": bbox,
            "total_h": total_h,
            "total_w": total_w,
            "cont_r": -bbox[0],
            "cont_c": -bbox[1],
        }
    return None


def _sample_decoy_spec() -> dict:
    inner_h = random.randint(1, 8)
    inner_w = random.randint(1, 5)
    inner_area = inner_h * inner_w
    head_side = random.randint(0, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    gap = random.randint(0, MAX_GAP)
    bbox = _relative_union_bbox(outer_h, outer_w, head_side, gap, 0)
    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "head_side": head_side,
        "gap": gap,
        "steps": 0,
        "initial_orange": inner_area,
        "grey": 0,
        "is_decoy": True,
        "relation": "decoy",
        "bbox": bbox,
        "total_h": bbox[2] - bbox[0] + 1,
        "total_w": bbox[3] - bbox[1] + 1,
        "cont_r": -bbox[0],
        "cont_c": -bbox[1],
    }


def _draw_head_or_dest(grid: np.ndarray, cr: int, cc: int, outer_h: int, outer_w: int, head_side: int, *, dest_gap: int | None = None) -> None:
    if dest_gap is None:
        if head_side == 0:
            grid[cr - 1, cc : cc + outer_w] = HEAD_COLOR
        elif head_side == 1:
            grid[cr + outer_h, cc : cc + outer_w] = HEAD_COLOR
        elif head_side == 2:
            grid[cr : cr + outer_h, cc - 1] = HEAD_COLOR
        else:
            grid[cr : cr + outer_h, cc + outer_w] = HEAD_COLOR
    else:
        if head_side == 0:
            grid[cr - 2 - dest_gap, cc : cc + outer_w] = HEAD_COLOR
        elif head_side == 1:
            grid[cr + outer_h + 1 + dest_gap, cc : cc + outer_w] = HEAD_COLOR
        elif head_side == 2:
            grid[cr : cr + outer_h, cc - 2 - dest_gap] = HEAD_COLOR
        else:
            grid[cr : cr + outer_h, cc + outer_w + 1 + dest_gap] = HEAD_COLOR


def _expected_from_specs(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
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

        if steps == 0:
            continue

        expected[cr : cr + oh, cc : cc + ow] = BG
        _draw_head_or_dest(expected, cr, cc, oh, ow, hs)

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
        _draw_head_or_dest(expected, nr, nc, oh, ow, hs)

    return expected


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    roles = _choose_relation_roles(n_movers)
    assemblies = []
    for relation in roles:
        spec = _sample_mover_spec(relation)
        if spec is None:
            return None
        assemblies.append(spec)
    for _ in range(n_decoys):
        assemblies.append(_sample_decoy_spec())
    random.shuffle(assemblies)

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
    placements: list[tuple[int, int]] = []
    for a in assemblies:
        th, tw = a["total_h"], a["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(400):
            r0 = random.randint(0, grid_h - th)
            c0 = random.randint(0, grid_w - tw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + th + 1)
            cc1 = min(grid_w, c0 + tw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
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

        fill_order = _interior_fill_order(inner_h, inner_w, hs)
        interior = np.full((inner_h, inner_w), GREY, dtype=int)
        for idx_cell in range(initial_orange):
            r, c = fill_order[idx_cell]
            interior[r, c] = ORANGE
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        _draw_head_or_dest(grid, cr, cc, oh, ow, hs)
        _draw_head_or_dest(grid, cr, cc, oh, ow, hs, dest_gap=gap)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_from_specs(grid, assemblies, placements)
    if not np.array_equal(output, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
