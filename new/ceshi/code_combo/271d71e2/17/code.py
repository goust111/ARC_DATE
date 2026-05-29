# Combo variant of 271d71e2 (index 17)
# combo_spec: D4/0+D6/1
# Name: 无边框容器多目标选择

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for the combo variant
# a) 具体化与行为追踪
#    这个融合版同时改两处：
#    1. D4/0 去掉黑框，容器本体直接就是一个由 5/7 构成的实心矩形。
#    2. D6/1 把单目标改成多目标，head 颜色决定该选哪条线。
#    具体小例子：一个 3x4 的 5/7 矩形，右侧紧贴一条蓝色(1) head；在更右方还会有两条候选线：
#    近处一条红色(2)，远处一条蓝色(1)。原题逻辑会先试图在 5/7 外围找黑框，这里会直接失败；
#    即使人为补成“无边框但仍按同色找目标”，也会错误地选远处蓝线。融合后应当：
#    - 先把这块完整的 5/7 矩形识别为容器；
#    - 在四侧寻找唯一一条与整条边等长、紧邻容器的 marker 线作为 head；
#    - 再沿该方向扫描所有候选线，依据颜色交换规则 1<->2 选择正确目标；
#    - 步数仍是 min(目标距离, 灰像素数)，并且从 head 侧向内把灰格依次染成橙色。
#    差异不只是“去框”或“多线”各改各的相加：solver 必须先从 5/7 块恢复容器范围，
#    才知道该去哪一侧读 head；只有读出 head 颜色后，才知道多条前方线里哪条才是真目标。
# b) 复杂度校准
#    无边框后若允许 1x1，容器会退化成单点，不利于肉眼识别，因此生成器把容器尺寸限制为至少 2x2。
#    多目标选择若颜色种类过多会让规则不可归纳，因此只保留两种 head/目标颜色 1 和 2，
#    且规则固定为 1 -> 2、2 -> 1。每个 mover 恰好看到两条候选线：较近的正确异色线，
#    和更远的同色 decoy。这样多目标变化清晰，但自由度仍受控。
# c) 隐式假设审查
#    原 main() 中失效的前提：
#    - “5/7 连通块外扩 1 格后应是黑框” 在本变体中恒不成立，必须改成直接识别 5/7 实心矩形。
#    - “从黑框外收集 head seed” 也失效，因为 head 现在直接贴着 5/7 容器。
#    - “按 head 同色搜索目标” 失效，因为正确目标是异色，不是同色。
#    - “是否贴着 BORDER 可区分 head/dest” 也失效，因为全图没有黑框。
#    新 solver 的显式前提改成：
#    - 容器是完整的 GREY/ORANGE 实心矩形；
#    - 恰有一侧紧邻一条完整 marker 线作为 head；
#    - 前方候选线都与 head 平行且等长；
#    - 颜色映射固定为 1<->2。
# d) 放置合法性
#    生成器按 assembly 放置：每个 assembly 只含一个容器、一条 head、两条前方候选线。
#    assembly 的 bbox 会预留到最远 decoy 线，因此输入和输出都留在同一保留区域内；
#    assembly 外再留 1 格背景 moat，避免不同容器或线段贴边、越界或互相干扰。
# e) 视觉可观察性
#    人类在样例里能直接看到两点：
#    - 没有任何黑框，只有 5/7 矩形本体，因此需要从色块形状本身识别容器；
#    - 每个 mover 前面至少有两条线，且 head 为 1 时去找 2，head 为 2 时去找 1。
#    生成器尽量让 mover 中同时出现两种 head 颜色，并保证至少有一个 mover 属于
#    “grey > 正确目标距离”的信息型样本，这样如果错误地按同色远目标求步数，输出会不同，
#    多目标选择不会退化成看不出来的假变化。
# f) 可视化整洁度
#    每个 assembly 都是一条直线方向上的局部结构：容器、head、近目标、远 decoy 共线排列，
#    没有交叉路径，也不引入额外花哨标记。这样无边框变化和颜色匹配变化都能被肉眼稳定读出。
# g) 变化的可推断性
#    本融合版只引入一个固定的双向颜色映射：1<->2。候选线颜色种类与 head 颜色种类相同，
#    不再额外增加更多语义颜色。配合“近处异色、远处同色”的重复模式，解题者可从少量样例
#    归纳出“颜色决定选哪条线”，而不是误以为只是“永远选最近”或“永远选同色”。
# h) main() 的新隐式假设与生成器保障
#    main() 现在依赖这些输入结构：
#    - 5/7 连通块必须是实心矩形；
#    - 每个容器四侧只有一侧紧邻完整 head 线；
#    - 候选目标线只出现在 head 正前方，且与容器边等长；
#    - 正确异色目标一定比同色 decoy 更近；
#    - 至少一个 mover 满足 grey > 正确目标距离，使颜色选择真正影响输出。
#    generate_input() 会逐项构造并验证这些条件，因此 solver 的假设不是悬空的。


BG = 6
GREY = 5
ORANGE = 7
HEAD_COLORS = (1, 2)
HEAD_TO_TARGET = {1: 2, 2: 1}

TOP = 0
BOTTOM = 1
LEFT = 2
RIGHT = 3

SIDE_TO_DIR = {
    TOP: (-1, 0),
    BOTTOM: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


def _fill_order(height: int, width: int, head_side: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(height) for c in range(width)]
    if head_side == RIGHT:
        cells.sort(key=lambda p: (-p[1], p[0]))
    elif head_side == LEFT:
        cells.sort(key=lambda p: (p[1], -p[0]))
    elif head_side == BOTTOM:
        cells.sort(key=lambda p: (-p[0], -p[1]))
    else:
        cells.sort(key=lambda p: (p[0], p[1]))
    return cells


def _side_line(box: tuple[int, int, int, int], side: int, offset: int) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    d = int(offset)
    if side == TOP:
        rr = r0 - d
        if rr < 0:
            return None
        return [(rr, c) for c in range(c0, c1 + 1)]
    if side == BOTTOM:
        rr = r1 + d
        return [(rr, c) for c in range(c0, c1 + 1)]
    if side == LEFT:
        cc = c0 - d
        if cc < 0:
            return None
        return [(r, cc) for r in range(r0, r1 + 1)]
    cc = c1 + d
    return [(r, cc) for r in range(r0, r1 + 1)]


def _line_top_left(coords: list[tuple[int, int]]) -> tuple[int, int]:
    arr = np.array(coords, dtype=int)
    r0, c0, _r1, _c1 = bbox_coords(arr)
    return int(r0), int(c0)


def _detect_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    mask = (grid == GREY) | (grid == ORANGE)
    for pts in components_mask(mask, connectivity=4):
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if pts.shape[0] != patch.size:
            continue
        if not bool(np.isin(patch, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (int(r0), int(c0), int(r1), int(c1))})
    return containers


def _pick_head_and_targets(grid: np.ndarray, bg: int, box: tuple[int, int, int, int]) -> dict | None:
    H, W = grid.shape
    head_candidates = []
    for side in (TOP, BOTTOM, LEFT, RIGHT):
        line = _side_line(box, side, 1)
        if line is None:
            continue
        if any(not (0 <= r < H and 0 <= c < W) for r, c in line):
            continue
        vals = [int(grid[r, c]) for r, c in line]
        uniq = set(vals)
        if len(uniq) != 1:
            continue
        color = vals[0]
        if color not in HEAD_COLORS:
            continue
        head_candidates.append((side, color, line))
    if len(head_candidates) != 1:
        return None

    side, head_color, head_line = head_candidates[0]
    targets = []
    max_offset = max(H, W)
    for offset in range(2, max_offset + 1):
        line = _side_line(box, side, offset)
        if line is None:
            break
        if any(not (0 <= r < H and 0 <= c < W) for r, c in line):
            break
        vals = [int(grid[r, c]) for r, c in line]
        uniq = set(vals)
        if uniq == {int(bg)}:
            continue
        if len(uniq) == 1 and next(iter(uniq)) in HEAD_COLORS:
            color = int(vals[0])
            targets.append(
                {
                    "color": color,
                    "distance": int(offset - 1),
                    "pts": set(line),
                    "tl": _line_top_left(line),
                }
            )
            if len(targets) == 2:
                break
            continue
        break

    return {
        "side": int(side),
        "color": int(head_color),
        "dir": SIDE_TO_DIR[int(side)],
        "head_pts": set(head_line),
        "head_tl": _line_top_left(head_line),
        "targets": targets,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move borderless grey/orange containers toward the color-selected target line.

    Variant rule:
    - Containers are solid GREY/ORANGE rectangles with no black border.
    - The head is a full-edge line in color 1 or 2.
    - Two candidate lines appear ahead; the correct target is the nearest line
      whose color is the opposite of the head color (1 -> 2, 2 -> 1).
    - Steps = min(distance to chosen target, grey cell count).
    - Grey cells become orange from the head-facing side inward.
    """
    g = np.array(grid, dtype=int)
    cnt = Counter(map(int, g.flatten()))
    bg = int(cnt.most_common(1)[0][0])

    containers = _detect_containers(g)
    assert containers

    out = g.copy()
    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        info = _pick_head_and_targets(g, bg, cont["box"])
        if info is None:
            continue

        side = int(info["side"])
        head_color = int(info["color"])
        target_color = int(HEAD_TO_TARGET[head_color])
        matching = [t for t in info["targets"] if int(t["color"]) == target_color]
        if not matching:
            continue
        best = min(matching, key=lambda item: int(item["distance"]))

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        grey_local = [
            (rr, cc)
            for rr in range(patch.shape[0])
            for cc in range(patch.shape[1])
            if int(patch[rr, cc]) == GREY
        ]
        steps = min(int(best["distance"]), int(len(grey_local)))
        dr, dc = SIDE_TO_DIR[side]
        tr = int(dr * steps)
        tc = int(dc * steps)

        fill_order = _fill_order(patch.shape[0], patch.shape[1], side)
        to_fill = []
        for rr, cc in fill_order:
            if int(patch[rr, cc]) == GREY:
                to_fill.append((rr, cc))
                if len(to_fill) == steps:
                    break

        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        for r, c in info["head_pts"]:
            if int(out[r, c]) == head_color:
                out[r, c] = int(bg)

        moved_patch = patch.copy()
        for rr, cc in to_fill:
            moved_patch[rr, cc] = int(ORANGE)
        paste(out, moved_patch, int(r0 + tr), int(c0 + tc), clip=True)

        for r, c in info["head_pts"]:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = head_color

    return out


def _assembly_geometry(cont_h: int, cont_w: int, head_side: int, farthest_dist: int) -> tuple[int, int, int, int]:
    if head_side == TOP:
        return cont_h + farthest_dist + 1, cont_w, farthest_dist + 1, 0
    if head_side == BOTTOM:
        return cont_h + farthest_dist + 1, cont_w, 0, 0
    if head_side == LEFT:
        return cont_h, cont_w + farthest_dist + 1, 0, farthest_dist + 1
    return cont_h, cont_w + farthest_dist + 1, 0, 0


def _line_cells(cr: int, cc: int, cont_h: int, cont_w: int, head_side: int, dist_from_head: int | None) -> list[tuple[int, int]]:
    if dist_from_head is None:
        if head_side == TOP:
            return [(cr - 1, c) for c in range(cc, cc + cont_w)]
        if head_side == BOTTOM:
            return [(cr + cont_h, c) for c in range(cc, cc + cont_w)]
        if head_side == LEFT:
            return [(r, cc - 1) for r in range(cr, cr + cont_h)]
        return [(r, cc + cont_w) for r in range(cr, cr + cont_h)]

    d = int(dist_from_head)
    if head_side == TOP:
        return [(cr - 1 - d, c) for c in range(cc, cc + cont_w)]
    if head_side == BOTTOM:
        return [(cr + cont_h + d, c) for c in range(cc, cc + cont_w)]
    if head_side == LEFT:
        return [(r, cc - 1 - d) for r in range(cr, cr + cont_h)]
    return [(r, cc + cont_w + d) for r in range(cr, cr + cont_h)]


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _build_expected(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for item in placements:
        steps = int(item["steps"])
        if steps == 0:
            continue

        cr, cc = item["container_rc"]
        cont_h = int(item["cont_h"])
        cont_w = int(item["cont_w"])
        head_side = int(item["head_side"])
        dr, dc = SIDE_TO_DIR[head_side]

        expected[cr : cr + cont_h, cc : cc + cont_w] = int(BG)
        _paint_cells(expected, item["head_coords"], BG)

        nr = int(cr + dr * steps)
        nc = int(cc + dc * steps)
        moved = item["container_patch"].copy()
        fill_order = _fill_order(cont_h, cont_w, head_side)
        painted = 0
        for rr, cc0 in fill_order:
            if int(moved[rr, cc0]) == GREY:
                moved[rr, cc0] = int(ORANGE)
                painted += 1
                if painted == steps:
                    break
        paste(expected, moved, nr, nc)
        moved_head = [(int(r + dr * steps), int(c + dc * steps)) for r, c in item["head_coords"]]
        _paint_cells(expected, moved_head, int(item["head_color"]))
    return expected


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = 0 if n_containers == 2 else random.choice([0, 1])
    n_movers = n_containers - n_decoys
    if n_movers < 2:
        return None

    mover_colors = [HEAD_COLORS[i % 2] for i in range(n_movers)]
    random.shuffle(mover_colors)

    specs: list[dict] = []
    informative_movers = 0
    for idx in range(n_containers):
        is_decoy = idx >= n_movers
        head_color = int(mover_colors[idx]) if not is_decoy else int(random.choice(HEAD_COLORS))
        target_color = int(HEAD_TO_TARGET[head_color])
        head_side = random.randint(0, 3)
        cont_h = random.randint(2, 8)
        cont_w = random.randint(2, 6)
        area = cont_h * cont_w

        max_match_dist = min(6, area - 1)
        if max_match_dist < 1:
            return None
        match_dist = random.randint(1, max_match_dist)
        same_dist = int(match_dist + random.randint(2, 5))

        if is_decoy:
            grey = 0
            initial_orange = area
            steps = 0
        else:
            if area >= match_dist + 1 and random.random() < 0.75:
                grey = random.randint(match_dist + 1, area)
                steps = match_dist
                informative_movers += 1
            else:
                if random.random() < 0.5:
                    grey = random.randint(match_dist, area)
                    steps = match_dist
                    if grey > match_dist:
                        informative_movers += 1
                else:
                    grey = random.randint(1, match_dist)
                    steps = grey
            initial_orange = area - grey

        total_h, total_w, cont_r, cont_c = _assembly_geometry(cont_h, cont_w, head_side, same_dist)
        if total_h > 26 or total_w > 26:
            return None

        specs.append(
            {
                "cont_h": cont_h,
                "cont_w": cont_w,
                "head_side": head_side,
                "head_color": head_color,
                "target_color": target_color,
                "match_dist": match_dist,
                "same_dist": same_dist,
                "grey": grey,
                "initial_orange": initial_orange,
                "steps": steps,
                "is_decoy": is_decoy,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
            }
        )

    if informative_movers == 0:
        return None

    total_area = sum(s["total_h"] * s["total_w"] for s in specs)
    min_side = max(8, int(total_area ** 0.5) + 2)
    max_side = min(26, min_side + 8)
    max_h = max(s["total_h"] for s in specs)
    max_w = max(s["total_w"] for s in specs)

    lo_h = max(8, max_h + 2)
    hi_h = max(lo_h, min(26, max_side + 4))
    lo_w = max(8, max_w + 2)
    hi_w = max(lo_w, min(26, max_side + 4))
    if lo_h > 26 or lo_w > 26:
        return None

    grid_h = random.randint(lo_h, min(26, hi_h))
    grid_w = random.randint(lo_w, min(26, hi_w))

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for spec in specs:
        th, tw = int(spec["total_h"]), int(spec["total_w"])
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
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0 : r0 + th, c0 : c0 + tw] = True
            placements.append((r0, c0))
            placed = True
            break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    build_records: list[dict] = []
    for spec, (pr, pc) in zip(specs, placements):
        cont_h = int(spec["cont_h"])
        cont_w = int(spec["cont_w"])
        head_side = int(spec["head_side"])
        head_color = int(spec["head_color"])
        target_color = int(spec["target_color"])
        match_dist = int(spec["match_dist"])
        same_dist = int(spec["same_dist"])

        cr = int(pr + spec["cont_r"])
        cc = int(pc + spec["cont_c"])

        fill_order = _fill_order(cont_h, cont_w, head_side)
        container = np.full((cont_h, cont_w), GREY, dtype=int)
        for idx in range(int(spec["initial_orange"])):
            rr, cc0 = fill_order[idx]
            container[rr, cc0] = int(ORANGE)
        paste(grid, container, cr, cc)

        head_coords = _line_cells(cr, cc, cont_h, cont_w, head_side, None)
        match_coords = _line_cells(cr, cc, cont_h, cont_w, head_side, match_dist)
        same_coords = _line_cells(cr, cc, cont_h, cont_w, head_side, same_dist)
        _paint_cells(grid, head_coords, head_color)
        _paint_cells(grid, match_coords, target_color)
        _paint_cells(grid, same_coords, head_color)

        build_records.append(
            {
                **spec,
                "container_rc": (cr, cc),
                "container_patch": container,
                "head_coords": head_coords,
                "match_coords": match_coords,
                "same_coords": same_coords,
            }
        )

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _build_expected(grid, build_records)
    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 400 attempts")


def fusion_behavior_score(grid: np.ndarray) -> int:
    """
    Diagnostic metric for this combo: count containers where choosing the
    opposite-color target changes the realized step count vs a same-color rule.

    A score > 0 means the generated sample genuinely exercises the D6/1 part of
    the fusion instead of collapsing to an output that would also arise from a
    same-color target selection rule.
    """
    g = np.array(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    score = 0
    for cont in _detect_containers(g):
        r0, c0, r1, c1 = cont["box"]
        info = _pick_head_and_targets(g, bg, cont["box"])
        if info is None:
            continue
        head_color = int(info["color"])
        target_color = int(HEAD_TO_TARGET[head_color])
        true_targets = [t for t in info["targets"] if int(t["color"]) == target_color]
        wrong_targets = [t for t in info["targets"] if int(t["color"]) == head_color]
        if not true_targets or not wrong_targets:
            continue
        grey = int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == GREY))
        true_steps = min(grey, int(min(true_targets, key=lambda t: int(t["distance"]))["distance"]))
        wrong_steps = min(grey, int(min(wrong_targets, key=lambda t: int(t["distance"]))["distance"]))
        if true_steps != wrong_steps:
            score += 1
    return score
