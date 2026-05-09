# Combo variant of 271d71e2 (index 4)
# combo_spec: D3/0+D5/3
# Name: 螺旋顺序间隔着色

from __future__ import annotations

"""
design_analysis:
  fused_rule:
    keep:
      - 黑框容器、头线、同色目标线的检测方式不变
      - 平移方向仍由头线相对容器的位置决定
      - 步数仍是 steps = min(head_to_dest_distance, grey_count)
    change:
      - D5/3: “处理哪些灰格”的顺序改为 interior 矩形的固定顺时针螺旋序，
        从外圈向内圈，不再依赖 head_side
      - D3/0: 在这条螺旋序上，只对前 steps 个待处理灰格中的第 1/3/5/... 个着色；
        第 2/4/6/... 个虽然被“经过”，但保持灰色

  concrete_trace:
    - 设一个 4x4 interior，螺旋序为
      [(0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(3,3),(3,2),(3,1),(3,0),(2,0),(1,0),(1,1),(1,2),(2,2),(2,1)]
    - 若输入里前 5 个螺旋格已是橙色，剩余格为灰，且步数 steps=4：
      - 原题会按“头侧条带序”连续把 4 个灰格都染橙
      - 仅 D5/3 会把螺旋剩余序中的前 4 个灰格全染橙
      - 仅 D3/0 会按原条带序，但只染第 1/3 个
      - 本融合版会在螺旋剩余序的前 4 个灰格里，只染第 1/3 个
    - 因此输出同时出现两类可见差异：
      1. 新增橙格落在“绕边走”的螺旋位置，而不是 head_side 条纹
      2. 被经过的区域不是连续全橙，而是灰橙交替

  complexity_calibration:
    - 如果 mover 的 interior 太小，或 steps 太短，螺旋序与原条带序可能局部重合，
      间隔规则也可能只染 1 格，肉眼难以区分
    - 因此生成器对 mover 采用至少 3x3 的 interior，并要求至少一个 mover 的融合输出
      同时不同于原题、D3 单变体、D5 单变体
    - decoy 仍允许全橙不动容器，保留原题“有的容器不移动”的干扰，但不增加新对象类别

  implicit_assumptions_review:
    - 容器检测依赖 interior 是完整矩形且只含 5/7；融合后仍成立，因为只是改变 5->7 的子集
    - 头线/目标线匹配依赖同色、同形、同轴；融合后未改动这些对象，仍成立
    - 原 solver 中唯一真正依赖“处理顺序”的地方，是对 grey_coords 的排序和取前 steps 个；
      融合后必须同时替换排序规则与取样规则
    - 新 solver 对输入增加了一个生成侧假设：已有橙格应是某个螺旋前缀。
      否则“剩余灰格按螺旋排序”仍可运行，但题目样例对人类会不整洁、难归纳。
      生成器会显式保证这一点
    - steps 仍由 grey_count 决定，而不是由最终新增橙格数量决定。
      这意味着当 steps=grey_count 时，容器可停止移动但 interior 里仍保留部分灰格；
      这是融合规则的刻意效果，不是 bug

  placement_legality:
    - 仍沿用原题的 assembly 布局：dest + gap + head + container 在同一条轴线上
    - 该布局天然容纳输入位置与平移后的输出位置，且对象之间保留至少 1 格背景隔离
    - 融合规则只改 interior 着色，不扩大 bbox，不新增额外对象，因此不会额外引入重叠风险

  visual_observability:
    - 输入中的已有橙格就是螺旋前缀，先给出“顺时针外圈向内”的视觉提示
    - 输出中的新增橙格不是把这条前缀继续连满，而是隔一个染一个；
      这样两三个样例就能同时支撑“螺旋顺序”和“奇偶位着色”
    - 生成器拒绝所有退化样本：如果融合输出与原题或任一单变体输出完全一样，就不接受

  cleanliness_and_inference:
    - 颜色系统仍只有 0/5/6/7/9 五色，变化仅发生在容器内部
    - 螺旋序只依赖可见几何位置，不依赖颜色编号、坐标和等人类不可见元信息
    - 头线、目标线、容器边框互不叠加，网格结构保持清晰
"""

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[4]
for candidate in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))

from arc_utils import STRUCT4, bbox_coords, colors_present, components_mask, paste


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


def _spiral_fill_order(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    """Clockwise spiral over a rectangle, starting at the top-left corner."""
    order: list[tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = inner_h - 1, inner_w - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            order.append((top, c))
        for r in range(top + 1, bottom + 1):
            order.append((r, right))
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                order.append((bottom, c))
        if left < right:
            for r in range(bottom - 1, top, -1):
                order.append((r, left))
        top += 1
        bottom -= 1
        left += 1
        right -= 1
    return order


def _linear_fill_order(inner_h: int, inner_w: int, head_side: int) -> list[tuple[int, int]]:
    """Original 271d71e2 stripe order, kept for generator anti-degeneration checks."""
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


def _simulate_fill(
    initial_orange: set[tuple[int, int]],
    order: list[tuple[int, int]],
    steps: int,
    alternating: bool,
) -> set[tuple[int, int]]:
    remaining = [cell for cell in order if cell not in initial_orange]
    chosen = remaining[:steps:2] if alternating else remaining[:steps]
    return set(initial_orange) | set(chosen)


def _spec_behavior_metrics(specs: list[dict]) -> dict[str, int]:
    metrics = {
        "movers": 0,
        "diff_vs_original": 0,
        "diff_vs_d3": 0,
        "diff_vs_d5": 0,
        "distinctive_movers": 0,
        "strong_distinctive_movers": 0,
    }
    for spec in specs:
        if spec["is_decoy"] or spec["steps"] <= 0:
            continue
        metrics["movers"] += 1
        spiral = _spiral_fill_order(spec["inner_h"], spec["inner_w"])
        linear = _linear_fill_order(spec["inner_h"], spec["inner_w"], spec["head_side"])
        initial = set(spiral[: spec["initial_orange"]])
        fused = _simulate_fill(initial, spiral, spec["steps"], alternating=True)
        original = _simulate_fill(initial, linear, spec["steps"], alternating=False)
        d3_only = _simulate_fill(initial, linear, spec["steps"], alternating=True)
        d5_only = _simulate_fill(initial, spiral, spec["steps"], alternating=False)

        diff_orig = fused != original
        diff_d3 = fused != d3_only
        diff_d5 = fused != d5_only
        if diff_orig:
            metrics["diff_vs_original"] += 1
        if diff_d3:
            metrics["diff_vs_d3"] += 1
        if diff_d5:
            metrics["diff_vs_d5"] += 1
        if diff_orig and diff_d3 and diff_d5:
            metrics["distinctive_movers"] += 1
            if (
                spec["inner_h"] >= 3
                and spec["inner_w"] >= 3
                and spec["steps"] >= 4
                and len(fused) < spec["inner_h"] * spec["inner_w"]
            ):
                metrics["strong_distinctive_movers"] += 1
    return metrics


def _specs_show_fusion(specs: list[dict]) -> bool:
    metrics = _spec_behavior_metrics(specs)
    return (
        metrics["movers"] >= 1
        and metrics["diff_vs_original"] >= 1
        and metrics["diff_vs_d3"] >= 1
        and metrics["diff_vs_d5"] >= 1
        and metrics["strong_distinctive_movers"] >= 1
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each container toward its same-colored destination line.

    Fused variant:
    - movement steps are unchanged from 271d71e2
    - remaining grey cells are ordered by a fixed clockwise spiral over the interior rectangle
    - among the first `steps` cells of that spiral-grey sequence, only odd 1-based positions
      turn orange; even positions stay grey
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
        head_info = pick_head_for_container((r0, c0, r1, c1))
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
        grey_local = [
            (rr, cc)
            for rr in range(inner.shape[0])
            for cc in range(inner.shape[1])
            if int(inner[rr, cc]) == GREY
        ]
        steps = min(int(steps), int(len(grey_local)))

        dr, dc = direction
        tr = int(dr * steps)
        tc = int(dc * steps)

        spiral_index = {cell: idx for idx, cell in enumerate(_spiral_fill_order(inner.shape[0], inner.shape[1]))}
        ordered_greys = sorted(grey_local, key=lambda p: spiral_index[p])
        to_fill_local = set(ordered_greys[:steps:2])

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in head_pts:
            if int(out[int(r), int(c)]) == int(hc):
                out[int(r), int(c)] = int(bg)

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        for rr, cc in to_fill_local:
            patch[1 + int(rr), 1 + int(cc)] = int(ORANGE)
        paste(out, patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in head_pts:
            rrr = int(r + tr)
            ccc = int(c + tc)
            if 0 <= rrr < H and 0 <= ccc < W:
                out[rrr, ccc] = int(hc)

    return out


def _build_expected_output(grid: np.ndarray, assemblies: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
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

        fill_order = _spiral_fill_order(a["inner_h"], a["inner_w"])
        initial = set(fill_order[: a["initial_orange"]])
        final = _simulate_fill(initial, fill_order, steps, alternating=True)
        interior = np.full((a["inner_h"], a["inner_w"]), GREY, dtype=int)
        for r, c in final:
            interior[int(r), int(c)] = ORANGE

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


def _gen_attempt(return_meta: bool = False):
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        head_side = random.randint(0, 3)

        if is_decoy:
            inner_h = random.randint(1, 6)
            inner_w = random.randint(1, 5)
        else:
            inner_h = random.randint(3, 8)
            inner_w = random.randint(3, 5)

        inner_area = inner_h * inner_w
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if is_decoy:
            initial_orange = inner_area
            grey = 0
            gap = random.randint(0, 6)
            steps = 0
            cap_mode = "decoy"
        else:
            if inner_area < 4:
                return None
            mode = random.choice(["distance", "distance", "grey"])
            if mode == "distance":
                max_steps = min(7, inner_area - 2)
                if max_steps < 3:
                    return None
                steps = random.randint(3, max_steps)
                grey = random.randint(steps + 1, inner_area - 1)
                initial_orange = inner_area - grey
                gap = steps - 1
            else:
                max_steps = min(6, inner_area - 1)
                if max_steps < 3:
                    return None
                steps = random.randint(3, max_steps)
                grey = steps
                initial_orange = inner_area - grey
                h2d = random.randint(steps + 1, min(10, steps + 4))
                gap = h2d - 1
            cap_mode = mode

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
                "is_decoy": is_decoy,
                "cap_mode": cap_mode,
            }
        )

    if not _specs_show_fusion(specs):
        return None

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
        cr = pr + a["cont_r"]
        cc = pc + a["cont_c"]

        grid[cr, cc : cc + ow] = BORDER
        grid[cr + oh - 1, cc : cc + ow] = BORDER
        grid[cr : cr + oh, cc] = BORDER
        grid[cr : cr + oh, cc + ow - 1] = BORDER

        fill_order = _spiral_fill_order(a["inner_h"], a["inner_w"])
        interior = np.full((a["inner_h"], a["inner_w"]), GREY, dtype=int)
        for idx_cell in range(a["initial_orange"]):
            r, c = fill_order[idx_cell]
            interior[int(r), int(c)] = ORANGE
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

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _build_expected_output(grid, assemblies, placements)
    if not np.array_equal(output, expected):
        return None

    if return_meta:
        return grid, {"specs": specs, "assemblies": assemblies, "placements": placements}
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 600 attempts")
