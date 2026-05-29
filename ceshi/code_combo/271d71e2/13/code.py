# Combo variant of 271d71e2 (index 13)
# combo_spec: D2/0+D3/2
# Name: 纯距离驱动反向填充
#
# Design analysis
# a) 具体化与行为追踪
# - 视觉对象沿用原题：6 背景，0 黑框矩形容器，内部只含 5/7，9 同时画贴边头线与同形目标线。
# - 融合后有两处改动，但它们要在同一条因果链里协同：
#   1) D2/0：容器总是移动到目标线位置，步数 = 头线到目标线的纯几何距离，不再被灰格数截断。
#   2) D3/2：剩余灰格的“被充能方向”改成从远离头部的一侧向头部推进，因此输入里的初始橙区也应放在远侧。
# - 小例子：头在右，内区 2x3，输入按融合规则可为
#     [7 5 5
#      7 5 5]
#   即左列先橙、右列仍灰。若目标距头 4 格、灰格只有 4 个：
#   原题步数 = min(4, 4) = 4，若输入按原题构造，橙区会从右往左长。
#   融合题仍走 4 格，但输入橙区的“生长起点”在左侧，表现为容器整体向右移动时，橙色前沿概念上朝左反方向回推。
# - 再看 distance > grey 的情形：若距离 5、灰格 2，原题只走 2 格；融合题直接走满 5 格并把剩余 2 个灰格全部转橙。
#   因而最终输出的关键证据主要来自 D2；D3 体现在输入橙区必须稳定出现在远侧，而不是头侧。
#
# b) 复杂度校准
# - 保持 2-4 个容器、四个可能朝向、同一条 9 色线既作头又作目标，不引入新颜色或额外标记。
# - 为避免退化为单纯 D2：
#   1) 每个样本至少有一个 overshoot mover（distance > grey），保证位置与原题不同。
#   2) 若容器数 >= 2，至少有一个 underfill mover（distance < grey），保证“原题会残灰而本题全橙”。
#   3) 至少有一个 mover 的输入内区满足“远侧预充橙块可见”：深度轴长度 > 1，且 0 < initial_orange < inner_area。
# - 这样既不会让匹配关系变复杂，也能让 2-3 组样例内重复出现同一种远侧预充模式。
#
# c) 隐式假设审查（main 逐段相关前提）
# - 容器检测只要求 5/7 内区外包完整黑框，不要求橙色必须贴头侧；融合后这一前提仍成立。
# - 头/目标匹配依赖“同色、同形、共轴、头是离容器最近的一段”，与填充方向无关；生成器继续保证这一点。
# - 原题的关键假设 `steps = min(distance, grey_count)` 在 D2/0 下失效，必须改成纯距离 `steps = distance`。
# - 原题还假设“只填前 steps 个灰格”。融合后所有灰格都会转橙，因此 solver 不再依赖灰格数量与顺序来决定终态。
# - D3/2 在融合后不再改变终态集合，但仍改变“输入应长什么样”：若生成器继续把橙区放在头侧，任务会退化成普通 D2。
#   因此 D3/2 的责任主要落在 generator 的输入构造与退化检查上。
#
# d) 放置合法性
# - 输入 assembly 仍采用原题 corridor：dest + gap + head + container。
# - 因为移动步数就是 head-to-dest distance，输出中的头线恰好覆盖到目标线，容器也完全落在同一 assembly bbox 内。
# - 生成时给每个 assembly 外围留 1 格背景缓冲，避免输入或输出阶段与其它 assembly 接触。
#
# e) 视觉可观察性
# - 通过 overshoot 样本，人类能直接看到“灰格数不再限制移动距离”。
# - 通过 underfill 样本，人类能看到“即使距离较短，最终也会全橙”，排除原题的逐步限额。
# - 通过远侧预充输入，人类能看到橙区起点稳定出现在容器深处而非头侧，支持“反向填充”的叙事。
# - 这里 D3 的可观察性主要来自输入分布，而非最终输出局部形态；这是由 D2 的“全部转橙”决定的。
#
# f) 可视化整洁度
# - 仍然只有矩形容器和直线标记两类对象，层次简单。
# - 远侧预充只改变 5/7 的空间分布，不新增覆盖层，也不会让对象互相遮挡。
#
# g) 变化的可推断性
# - 颜色语义不变：5 -> 7 是唯一颜色变化，9 只承担方向/匹配提示。
# - 变体需要学习的新规律只有两条：移动看纯距离；输入里的橙区总是在远侧起步。
# - 生成器会拒绝“融合输出与原题输出完全相同”的输入，并统计至少一个“远侧预充可见”的 mover，防止样本退化。
#
# post_change_solver_assumptions_checked:
# - 每个容器都必须有唯一可匹配的同色目标线；generator 继续使用单色单轴 assembly 保证这一点。
# - solver 现在允许 grey_count < distance，因此 generator 不能再依赖“灰格耗尽会提前停下”防止越界；
#   assembly bbox 已按 full-distance movement 预留完整走廊。
# - 由于 D3 的终态信息被 D2 吞没，generator 必须显式制造 far-side prefill 证据；代码中已强制至少一个此类 mover。

from __future__ import annotations

from collections import Counter, defaultdict
import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
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
DIR_TO_SIDE = {v: k for k, v in SIDE_TO_DIR.items()}
MAX_GAP = 10

LAST_GENERATED_META: dict | None = None


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int):
    for c in range(c0, c1 + 1):
        yield (r0, c)
        yield (r1, c)
    for r in range(r0 + 1, r1):
        yield (r, c0)
        yield (r, c1)


def _ordered_cells(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    head_side: int | None = None,
    direction: tuple[int, int] | None = None,
    from_far_side: bool,
) -> list[tuple[int, int]]:
    if head_side is None:
        assert direction is not None
        head_side = DIR_TO_SIDE[direction]

    ordered = list(cells)
    if head_side == 3:  # head on right
        key = (lambda p: (p[1], p[0])) if from_far_side else (lambda p: (-p[1], p[0]))
    elif head_side == 2:  # head on left
        key = (lambda p: (-p[1], -p[0])) if from_far_side else (lambda p: (p[1], -p[0]))
    elif head_side == 1:  # head on bottom
        key = (lambda p: (p[0], -p[1])) if from_far_side else (lambda p: (-p[0], -p[1]))
    else:  # head on top
        key = (lambda p: (-p[0], p[1])) if from_far_side else (lambda p: (p[0], p[1]))
    ordered.sort(key=key)
    return ordered


def _draw_head(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, color: int) -> None:
    if head_side == 0:
        grid[cr - 1, cc : cc + ow] = color
    elif head_side == 1:
        grid[cr + oh, cc : cc + ow] = color
    elif head_side == 2:
        grid[cr : cr + oh, cc - 1] = color
    else:
        grid[cr : cr + oh, cc + ow] = color


def _draw_dest(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int, gap: int, color: int) -> None:
    if head_side == 0:
        grid[cr - 2 - gap, cc : cc + ow] = color
    elif head_side == 1:
        grid[cr + oh + 1 + gap, cc : cc + ow] = color
    elif head_side == 2:
        grid[cr : cr + oh, cc - 2 - gap] = color
    else:
        grid[cr : cr + oh, cc + ow + 1 + gap] = color


def _erase_head(grid: np.ndarray, cr: int, cc: int, oh: int, ow: int, head_side: int) -> None:
    _draw_head(grid, cr, cc, oh, ow, head_side, BG)


def _depth_len(inner_h: int, inner_w: int, head_side: int) -> int:
    return int(inner_h if head_side in (0, 1) else inner_w)


def _build_input_interior(inner_h: int, inner_w: int, head_side: int, initial_orange: int) -> np.ndarray:
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    order = _ordered_cells(
        [(r, c) for r in range(inner_h) for c in range(inner_w)],
        head_side=head_side,
        from_far_side=True,
    )
    for r, c in order[:initial_orange]:
        interior[r, c] = ORANGE
    return interior


def _input_prefill_is_visibly_reversed(inner_h: int, inner_w: int, head_side: int, initial_orange: int) -> bool:
    if not (0 < initial_orange < inner_h * inner_w):
        return False
    far = _build_input_interior(inner_h, inner_w, head_side, initial_orange)
    near = np.full((inner_h, inner_w), GREY, dtype=int)
    near_order = _ordered_cells(
        [(r, c) for r in range(inner_h) for c in range(inner_w)],
        head_side=head_side,
        from_far_side=False,
    )
    for r, c in near_order[:initial_orange]:
        near[r, c] = ORANGE
    return not np.array_equal(far, near)


def _expected_under_rule(
    grid: np.ndarray,
    assemblies: list[dict],
    placements: list[tuple[int, int]],
    *,
    full_distance: bool,
    from_far_side: bool,
    fill_all_remaining: bool,
) -> np.ndarray:
    expected = grid.copy()
    for a, (pr, pc) in zip(assemblies, placements):
        oh, ow = a["outer_h"], a["outer_w"]
        hs = a["head_side"]
        distance = a["distance"]
        dr, dc = SIDE_TO_DIR[hs]

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        patch = grid[cr : cr + oh, cc : cc + ow].copy()
        inner = patch[1:-1, 1:-1].copy()
        grey_cells = [(r, c) for r in range(inner.shape[0]) for c in range(inner.shape[1]) if int(inner[r, c]) == GREY]

        steps = int(distance if full_distance else min(distance, len(grey_cells)))
        fill_count = int(len(grey_cells) if fill_all_remaining else min(steps, len(grey_cells)))
        order = _ordered_cells(grey_cells, head_side=hs, from_far_side=from_far_side)
        for r, c in order[:fill_count]:
            inner[r, c] = ORANGE
        patch[1:-1, 1:-1] = inner

        expected[cr : cr + oh, cc : cc + ow] = BG
        _erase_head(expected, cr, cc, oh, ow, hs)

        nr = cr + dr * steps
        nc = cc + dc * steps
        expected[nr : nr + oh, nc : nc + ow] = patch
        _draw_head(expected, nr, nc, oh, ow, hs, HEAD_COLOR)

    return expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move bordered containers toward same-colored destination markers.

    Combo rule:
    - Move by the full head-to-destination distance.
    - Convert every remaining grey interior cell to orange.
    - The conceptual charge order is from the far side toward the head side.
      In the final state this only affects how valid inputs are generated, because all greys are consumed.
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

        tr0 = int(best_dest["tl"][0] - head_tl[0])
        tc0 = int(best_dest["tl"][1] - head_tl[1])
        steps = abs(tr0) + abs(tc0)
        tr = int(direction[0] * steps)
        tc = int(direction[1] * steps)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_coords = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in _ordered_cells(grey_coords, direction=direction, from_far_side=True):
            patch[int(rr - r0), int(cc - c0)] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _mode_sequence(n_containers: int) -> list[str]:
    modes = ["overshoot"]
    if n_containers >= 2:
        modes.append("underfill")
    while len(modes) < n_containers:
        modes.append(random.choices(["overshoot", "underfill", "balanced"], weights=[4, 4, 1])[0])
    random.shuffle(modes)
    return modes


def _sample_spec(mode: str, *, require_reverse_prefill: bool) -> dict | None:
    for _ in range(200):
        inner_h = random.randint(1, 8)
        inner_w = random.randint(1, 5)
        area = inner_h * inner_w
        head_side = random.randint(0, 3)

        if require_reverse_prefill and _depth_len(inner_h, inner_w, head_side) < 2:
            continue

        if mode == "overshoot":
            grey_max = area - 1 if require_reverse_prefill else area
            if grey_max < 1:
                continue
            grey = random.randint(1, min(grey_max, 4))
            min_gap = grey
            if min_gap > MAX_GAP:
                continue
            gap = random.randint(min_gap, MAX_GAP)
        elif mode == "underfill":
            grey_max = area - 1 if require_reverse_prefill else area
            if grey_max < 2:
                continue
            max_gap = min(MAX_GAP, grey_max - 2)
            if max_gap < 0:
                continue
            gap = random.randint(0, max_gap)
            distance = gap + 1
            grey_min = distance + 1
            if grey_min > grey_max:
                continue
            grey = random.randint(grey_min, grey_max)
        else:
            grey_max = area - 1 if require_reverse_prefill else area
            if grey_max < 1:
                continue
            max_dist = min(MAX_GAP + 1, grey_max)
            if max_dist < 1:
                continue
            distance = random.randint(1, max_dist)
            gap = distance - 1
            grey = distance

        initial_orange = area - grey
        if require_reverse_prefill and not _input_prefill_is_visibly_reversed(inner_h, inner_w, head_side, initial_orange):
            continue

        outer_h = inner_h + 2
        outer_w = inner_w + 2
        return {
            "mode": mode,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": outer_h,
            "outer_w": outer_w,
            "head_side": head_side,
            "gap": gap,
            "distance": gap + 1,
            "grey": grey,
            "initial_orange": initial_orange,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    global LAST_GENERATED_META

    n_containers = random.randint(2, 4)
    modes = _mode_sequence(n_containers)
    forced_reverse_idx = random.randrange(n_containers)

    specs = []
    for idx, mode in enumerate(modes):
        spec = _sample_spec(mode, require_reverse_prefill=(idx == forced_reverse_idx))
        if spec is None:
            return None
        specs.append(spec)

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

        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        interior = _build_input_interior(inner_h, inner_w, hs, a["initial_orange"])
        grid[cr + 1 : cr + 1 + inner_h, cc + 1 : cc + 1 + inner_w] = interior

        _draw_head(grid, cr, cc, oh, ow, hs, HEAD_COLOR)
        _draw_dest(grid, cr, cc, oh, ow, hs, a["gap"], HEAD_COLOR)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_under_rule(
        grid,
        assemblies,
        placements,
        full_distance=True,
        from_far_side=True,
        fill_all_remaining=True,
    )
    if not np.array_equal(output, expected):
        return None

    base_expected = _expected_under_rule(
        grid,
        assemblies,
        placements,
        full_distance=False,
        from_far_side=False,
        fill_all_remaining=False,
    )
    if np.array_equal(expected, base_expected):
        return None

    overshoot_count = sum(1 for a in assemblies if a["distance"] > a["grey"])
    underfill_count = sum(1 for a in assemblies if a["distance"] < a["grey"])
    reverse_visible_count = sum(
        1
        for a in assemblies
        if _input_prefill_is_visibly_reversed(a["inner_h"], a["inner_w"], a["head_side"], a["initial_orange"])
    )
    if overshoot_count < 1:
        return None
    if len(assemblies) >= 2 and underfill_count < 1:
        return None
    if reverse_visible_count < 1:
        return None

    LAST_GENERATED_META = {
        "overshoot_count": overshoot_count,
        "underfill_count": underfill_count,
        "reverse_visible_count": reverse_visible_count,
        "distance_gt_grey": [a["distance"] > a["grey"] for a in assemblies],
        "distance_lt_grey": [a["distance"] < a["grey"] for a in assemblies],
        "prefill_reversed": [
            _input_prefill_is_visibly_reversed(a["inner_h"], a["inner_w"], a["head_side"], a["initial_orange"])
            for a in assemblies
        ],
        "fusion_differs_from_base": True,
    }
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
