# Variant of 271d71e2
# Dimension: D4 — 源对象结构 (src)
# Change: 黑色边框矩形容器，内部为灰色/橙色填充，附带同色头部线段指示方向... → 无边框容器

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for the D4 variant
# a) 具体化与行为追踪
#    原题容器是「黑框 + 内部灰/橙填充」，solver 先找灰/橙连通块，再向外扩 1 格验证黑框。
#    这个变体删掉黑框后，输入中的容器直接是一个由 5/7 组成的实心矩形，head 仍是贴着某一边的 9 线段，
#    dest 仍是沿同轴方向更远处的一条同形 9 线段。于是 solver 的关键改动是：
#    - 原逻辑：灰/橙块 bbox -> 向外找黑框 -> 再从黑框外找 head
#    - 新逻辑：灰/橙块 bbox 本身就是容器 -> 直接检查四侧紧邻的整条 9 线找 head
#    小例子（head 在上侧）：
#    - 原题：容器 bbox 是 5/7 内部，真正外轮廓还要再扩 1 格黑框；head 在黑框上方 1 行。
#    - 变体：容器 bbox 直接就是 5/7 矩形；head 在该矩形上方 1 行。
#    - 差异：head 到容器的参考边从「黑框外边」改成「灰/橙矩形边」。
# b) 复杂度校准
#    去黑框后若允许 1x1 容器，视觉上会退化成单点，方向/范围都不明显。因此生成器把容器尺寸限制为
#    至少 2x2，保留 2-4 个容器、0-1 个全橙 decoy、gap 0-10 等原题自由度，避免任务过于简单。
# c) 隐式假设审查
#    原 main 中失效的关键前提：
#    - `rect_border_coords` 与向外扩 1 格验证 BORDER：变体下恒不成立，必须删掉。
#    - 从黑框边附近收集 head seed：现在 head 贴的是灰/橙块，不是黑框。
#    - 用“与 BORDER 相邻”来区分 head/dest 的残余同色组件：也不再可用。
#    新 solver 改成两个可见前提：
#    - 容器是完整的 5/7 实心矩形。
#    - head 是与某一整条边等长、紧邻容器外侧的整条同色线；dest 在同轴更外侧，之间只有背景。
# d) 放置合法性
#    生成器按每个 assembly 的总包围盒放置：container + head + gap + dest 全都包含在一个长条 bbox 里，
#    并在 assembly 外再保留 1 格背景缓冲。这样可同时保证输入和输出都不越界、不重叠。
# e) 视觉可观察性
#    去黑框后，解题者必须从 5/7 矩形本身认出“容器范围”，这正是 D4 的可见变化。
#    为了让这个变化肉眼可见，容器最小边长不低于 2，而且保留从 head 侧向内逐格充能的橙色结构。
# f) 可视化整洁度
#    同一 assembly 内只有一块 5/7 矩形和两条 9 线；assembly 之间至少隔 1 格背景，避免线段互相贴边。
# g) 颜色变化可推断性
#    颜色体系不变：背景 6，容器内部 5/7，方向标记 9。变化只在“有没有黑框”，不会引入新的颜色映射负担。
#    人类可以从多组样例稳定看到：黑框消失了，但移动规则和橙化顺序保持原题一致。


BG = 6
GREY = 5
ORANGE = 7
HEAD_COLOR = 9

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
    """Order cells from the head-facing side inward, matching the original task."""
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
    """Cells on the line parallel to the chosen container side, offset outward by `offset`."""
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
    """Borderless containers are exactly 4-connected GREY/ORANGE rectangles."""
    containers: list[dict] = []
    mask = (grid == GREY) | (grid == ORANGE)
    for pts in components_mask(mask, connectivity=4):
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if patch.size == 0:
            continue
        if pts.shape[0] != patch.size:
            continue
        if not bool(np.isin(patch, [GREY, ORANGE]).all()):
            continue
        containers.append({"box": (int(r0), int(c0), int(r1), int(c1))})
    return containers


def _pick_head_and_dest(grid: np.ndarray, bg: int, box: tuple[int, int, int, int]) -> dict | None:
    """Find the unique adjacent head line and the first matching destination line further outward."""
    H, W = grid.shape
    candidates = []
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
        if color in (int(bg), GREY, ORANGE):
            continue
        candidates.append((side, color, line))
    if len(candidates) != 1:
        return None

    side, color, head_line = candidates[0]
    max_offset = max(H, W)
    dest_line = None
    dest_offset = None
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
        if uniq == {int(color)}:
            dest_line = line
            dest_offset = offset
        break
    if dest_line is None or dest_offset is None:
        return None

    return {
        "side": side,
        "color": int(color),
        "dir": SIDE_TO_DIR[side],
        "head_pts": set(head_line),
        "head_tl": _line_top_left(head_line),
        "dest_tl": _line_top_left(dest_line),
        "distance": int(dest_offset - 1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move borderless grey/orange containers toward same-colored destination lines, charging grey cells to orange.

    Input structure:
    - bg = 6 (magenta).
    - Each container is a solid GREY/ORANGE rectangle with no black border.
    - A head line (9) is flush against one side; an identical destination line lies further along the same axis.

    Rule:
    - Steps = min(head-to-dest distance, grey cell count).
    - The whole container and the head translate by that many cells toward the destination.
    - Grey cells convert to orange from the head-facing side inward, one per step.
    """
    g = np.array(grid, dtype=int)
    cnt = Counter(map(int, g.flatten()))
    bg = int(cnt.most_common(1)[0][0])

    containers = _detect_containers(g)
    assert containers

    out = g.copy()
    for cont in containers:
        r0, c0, r1, c1 = cont["box"]
        head_info = _pick_head_and_dest(g, bg, cont["box"])
        if head_info is None:
            continue

        side = int(head_info["side"])
        dr, dc = head_info["dir"]
        head_pts = set(head_info["head_pts"])
        distance = int(head_info["distance"])

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        grey_local = [(rr, cc) for rr in range(patch.shape[0]) for cc in range(patch.shape[1]) if int(patch[rr, cc]) == GREY]
        steps = min(distance, len(grey_local))
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
        for r, c in head_pts:
            if int(out[r, c]) == int(head_info["color"]):
                out[r, c] = int(bg)

        moved_patch = patch.copy()
        for rr, cc in to_fill:
            moved_patch[rr, cc] = ORANGE
        paste(out, moved_patch, r0 + tr, c0 + tc, clip=True)

        for r, c in head_pts:
            rr = int(r + tr)
            cc = int(c + tc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = int(head_info["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.randint(2, 4)
    n_decoys = random.choice([0, 0, 1])
    n_movers = max(1, n_containers - n_decoys)
    n_decoys = n_containers - n_movers

    specs = []
    for i in range(n_containers):
        is_decoy = i >= n_movers
        cont_h = random.randint(2, 8)
        cont_w = random.randint(2, 6)
        area = cont_h * cont_w
        head_side = random.randint(0, 3)

        if is_decoy:
            grey = 0
            initial_orange = area
            gap = random.randint(0, 6)
            steps = 0
        else:
            grey = random.randint(1, area)
            initial_orange = area - grey
            gap = random.randint(0, 10)
            steps = min(grey, gap + 1)

        if head_side in (TOP, BOTTOM):
            total_h = cont_h + gap + 2
            total_w = cont_w
            cont_r = gap + 2 if head_side == TOP else 0
            cont_c = 0
        else:
            total_h = cont_h
            total_w = cont_w + gap + 2
            cont_r = 0
            cont_c = gap + 2 if head_side == LEFT else 0

        specs.append(
            {
                "cont_h": cont_h,
                "cont_w": cont_w,
                "head_side": head_side,
                "gap": gap,
                "steps": steps,
                "initial_orange": initial_orange,
                "grey": grey,
                "is_decoy": is_decoy,
                "total_h": total_h,
                "total_w": total_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
            }
        )

    total_area = sum(s["total_h"] * s["total_w"] for s in specs)
    max_asm_h = max(s["total_h"] for s in specs)
    max_asm_w = max(s["total_w"] for s in specs)
    min_side = max(8, int(total_area**0.5) + 2)
    max_side = min(26, min_side + 8)

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
    for spec in specs:
        th, tw = spec["total_h"], spec["total_w"]
        if th > grid_h or tw > grid_w:
            return None
        placed = False
        for _ in range(300):
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
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    for spec, (pr, pc) in zip(specs, placements):
        cont_h = spec["cont_h"]
        cont_w = spec["cont_w"]
        head_side = spec["head_side"]
        gap = spec["gap"]
        steps = spec["steps"]
        initial_orange = spec["initial_orange"]
        dr, dc = SIDE_TO_DIR[head_side]

        cr = pr + spec["cont_r"]
        cc = pc + spec["cont_c"]

        fill_order = _fill_order(cont_h, cont_w, head_side)
        container = np.full((cont_h, cont_w), GREY, dtype=int)
        for idx in range(initial_orange):
            rr, cc0 = fill_order[idx]
            container[rr, cc0] = ORANGE
        paste(grid, container, cr, cc)

        if head_side == TOP:
            head_coords = [(cr - 1, c) for c in range(cc, cc + cont_w)]
            dest_coords = [(cr - 2 - gap, c) for c in range(cc, cc + cont_w)]
        elif head_side == BOTTOM:
            head_coords = [(cr + cont_h, c) for c in range(cc, cc + cont_w)]
            dest_coords = [(cr + cont_h + 1 + gap, c) for c in range(cc, cc + cont_w)]
        elif head_side == LEFT:
            head_coords = [(r, cc - 1) for r in range(cr, cr + cont_h)]
            dest_coords = [(r, cc - 2 - gap) for r in range(cr, cr + cont_h)]
        else:
            head_coords = [(r, cc + cont_w) for r in range(cr, cr + cont_h)]
            dest_coords = [(r, cc + cont_w + 1 + gap) for r in range(cr, cr + cont_h)]

        for r, c in head_coords + dest_coords:
            grid[r, c] = HEAD_COLOR

        nr = cr + dr * steps
        nc = cc + dc * steps
        moved = container.copy()
        for idx in range(initial_orange, initial_orange + steps):
            rr, cc0 = fill_order[idx]
            moved[rr, cc0] = ORANGE
        paste(expected, moved, nr, nc)

        moved_head = [(r + dr * steps, c + dc * steps) for r, c in head_coords]
        for r, c in moved_head + dest_coords:
            expected[r, c] = HEAD_COLOR

    try:
        output = main(grid)
    except Exception:
        return None
    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid after 300 attempts")
