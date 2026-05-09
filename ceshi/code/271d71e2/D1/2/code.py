# Variant of 271d71e2
# Dimension: D1 — 路径方向类型 (path)
# Change: straight_ray — 容器沿头部指示的四个基本方向（上/下/左/右）直线平移... → 方向由内部图案决定

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, paste


# -----------------------------------------------------------------------------
# Design analysis for the D1 variant
#
# spec_variant:
#   unchanged_core:
#     - 黑色边框矩形容器，内部只含灰(5)/橙(7)
#     - 容器沿四联通直线平移
#     - 平移步数仍受“目标距离”和“灰像素数量”共同限制
#     - 每平移 1 格，就有 1 个灰像素按固定次序变橙
#   changed_core:
#     - 原题通过容器外侧相邻的 maroon 头线确定方向
#     - 这个变体改为：方向只由容器内部的橙色箭头模板决定
#     - 为了让内部方向信息真正参与决策，外部不再只给单个目标线，
#       而是在同一轴的两侧各放一个候选 maroon 目标线；轴由目标线对给出，
#       具体选左/右或上/下必须看内部箭头
#   concrete_visuals:
#     - mover 容器内部至少 3x3，初始橙色永远是一个 4-cell 箭头：
#         up    :  (0,m), (1,m-1), (1,m), (1,m+1)
#         down  :  (h-1,m), (h-2,m-1), (h-2,m), (h-2,m+1)
#         left  :  (mr,0), (mr-1,1), (mr,1), (mr+1,1)
#         right :  (mr,w-1), (mr-1,w-2), (mr,w-2), (mr+1,w-2)
#     - 目标线仍是 9，但现在总是成对出现，位于相反两侧，同轴对齐容器整条边
#     - 容器只朝箭头指向的一侧移动；另一侧目标线是显式 decoy
#   behavior_trace_example:
#     - 原题某个容器若右侧贴着 head、再往右有 dest，则 solver 直接判定向右
#     - 变体里同一个容器左右两侧都放 dest 候选线，外部不再告诉你选哪边
#     - solver 必须先读出内部箭头，例如 right，再只取右侧候选线，随后
#       steps = min(右侧可移动距离, 灰格数)，并从右侧向内继续给灰格充橙
#   complexity_calibration:
#     - 没把方向做成任意图案，而是固定 4 种箭头模板，复杂度与 ARC 训练样例可观察性匹配
#     - 没引入多颜色映射；方向只靠橙色相对灰色/边框的位置
#     - 候选目标线只做成 opposite pair，而不是四面全放，避免视觉噪声过大
#   implicit_assumption_audit_against_original_main:
#     - 原 solver 的 pick_head_for_container 依赖“边框外最近的非背景组件就是 head”，
#       这个前提在本变体不成立，因为两侧都有目标线，且没有相邻 head；因此整段必须移除
#     - 原 solver 用 head bbox 相对 container bbox 推出方向；这里改为从 interior orange
#       模板推方向，因此 grey-fill 排序也要改成“按箭头所指边 inward”
#     - 原 solver 通过匹配同色 head/dest shape 来找目标；这里外部目标永远是整边直线，
#       且容器各自只需要找自己四侧射线上的最近整边直线即可
#     - 原 solver 在擦除原位置时还会擦 head；这里无 head，只擦容器框体
#   placement_legality:
#     - generator 为每个容器构造一个 assembly bbox：左/右或上/下两个目标线 + 中间容器
#     - 该 assembly 已经覆盖输入位置和所有可能输出位置；assembly 之间留 1 格背景，
#       所以输入与输出都不会互相碰撞
#     - 候选线与容器整边严格对齐，solver 才能用“沿四条射线找最近整边 9 线”的方式匹配
#   visual_observability:
#     - 外部 always 有两个候选目标，不能只看外部就决定方向
#     - 箭头模板固定且醒目，2-3 个样例足以归纳“箭头朝哪边就往哪边走”
#   cleanliness:
#     - 颜色仍只有 0/5/6/7/9 五种
#     - 每个 assembly 只含一个容器和一对目标线，没有同一区域叠层
#   generator_guarantees_for_new_main:
#     - mover 的 interior orange 恰好等于 1 个箭头模板，因此方向检测唯一
#     - stationary decoy 允许全橙；solver 先检查 grey_count==0 并跳过，不会误判方向
#     - 目标线都不是相邻 head，而是距离容器至少 1 格背景，因此 old solver 不会退化为原逻辑
# -----------------------------------------------------------------------------

BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
MARKER = 9

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _arrow_template(inner_h: int, inner_w: int, direction: str) -> list[tuple[int, int]]:
    mid_r = inner_h // 2
    mid_c = inner_w // 2
    if direction == "U":
        return [(0, mid_c), (1, mid_c - 1), (1, mid_c), (1, mid_c + 1)]
    if direction == "D":
        return [
            (inner_h - 1, mid_c),
            (inner_h - 2, mid_c - 1),
            (inner_h - 2, mid_c),
            (inner_h - 2, mid_c + 1),
        ]
    if direction == "L":
        return [(mid_r, 0), (mid_r - 1, 1), (mid_r, 1), (mid_r + 1, 1)]
    if direction == "R":
        return [
            (mid_r, inner_w - 1),
            (mid_r - 1, inner_w - 2),
            (mid_r, inner_w - 2),
            (mid_r + 1, inner_w - 2),
        ]
    raise ValueError(direction)


def _fill_order(inner_h: int, inner_w: int, direction: str) -> list[tuple[int, int]]:
    base = _arrow_template(inner_h, inner_w, direction)
    base_set = set(base)
    mid_r = inner_h // 2
    mid_c = inner_w // 2
    remaining = []
    for r in range(inner_h):
        for c in range(inner_w):
            if (r, c) in base_set:
                continue
            if direction == "U":
                key = (r, abs(c - mid_c), c)
            elif direction == "D":
                key = (inner_h - 1 - r, abs(c - mid_c), c)
            elif direction == "L":
                key = (c, abs(r - mid_r), r)
            else:
                key = (inner_w - 1 - c, abs(r - mid_r), r)
            remaining.append((key, (r, c)))
    remaining.sort(key=lambda item: item[0])
    return base + [cell for _key, cell in remaining]


def _infer_direction(inner: np.ndarray) -> str | None:
    orange = {(int(r), int(c)) for r, c in np.argwhere(inner == ORANGE)}
    if not orange:
        return None
    matches = []
    inner_h, inner_w = inner.shape
    for direction in DIRS:
        tpl = set(_arrow_template(inner_h, inner_w, direction))
        if tpl.issubset(orange):
            matches.append(direction)
    if len(matches) == 1:
        return matches[0]
    return None


def _find_side_marker_distance(
    g: np.ndarray, box: tuple[int, int, int, int], direction: str, bg: int,
) -> int | None:
    r0, c0, r1, c1 = box
    H, W = g.shape
    if direction == "R":
        for off in range(1, W - c1):
            col = c1 + off
            vals = g[r0 : r1 + 1, col]
            if np.all(vals == bg):
                continue
            if np.all(vals == MARKER):
                if r0 > 0 and int(g[r0 - 1, col]) == MARKER:
                    return None
                if r1 + 1 < H and int(g[r1 + 1, col]) == MARKER:
                    return None
                return off - 1
            return None
    if direction == "L":
        for off in range(1, c0 + 1):
            col = c0 - off
            vals = g[r0 : r1 + 1, col]
            if np.all(vals == bg):
                continue
            if np.all(vals == MARKER):
                if r0 > 0 and int(g[r0 - 1, col]) == MARKER:
                    return None
                if r1 + 1 < H and int(g[r1 + 1, col]) == MARKER:
                    return None
                return off - 1
            return None
    if direction == "D":
        for off in range(1, H - r1):
            row = r1 + off
            vals = g[row, c0 : c1 + 1]
            if np.all(vals == bg):
                continue
            if np.all(vals == MARKER):
                if c0 > 0 and int(g[row, c0 - 1]) == MARKER:
                    return None
                if c1 + 1 < W and int(g[row, c1 + 1]) == MARKER:
                    return None
                return off - 1
            return None
    if direction == "U":
        for off in range(1, r0 + 1):
            row = r0 - off
            vals = g[row, c0 : c1 + 1]
            if np.all(vals == bg):
                continue
            if np.all(vals == MARKER):
                if c0 > 0 and int(g[row, c0 - 1]) == MARKER:
                    return None
                if c1 + 1 < W and int(g[row, c1 + 1]) == MARKER:
                    return None
                return off - 1
            return None
    return None


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
        if not np.all(g[r0, c0 : c1 + 1] == BORDER):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == BORDER):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == BORDER):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == BORDER):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        containers.append((r0, c0, r1, c1))

    assert containers

    out = g.copy()

    for r0, c0, r1, c1 in containers:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_count = int(np.sum(inner == GREY))
        if grey_count == 0:
            continue

        direction = _infer_direction(inner)
        if direction is None:
            continue

        marker_dist = _find_side_marker_distance(g, (r0, c0, r1, c1), direction, bg)
        if marker_dist is None:
            continue
        steps = min(marker_dist, grey_count)
        if steps <= 0:
            continue

        fill_order = _fill_order(inner.shape[0], inner.shape[1], direction)
        orange_count = int(np.sum(inner == ORANGE))
        target_orange = min(len(fill_order), orange_count + steps)

        new_inner = np.full(inner.shape, GREY, dtype=int)
        for idx, (rr, cc) in enumerate(fill_order):
            if idx >= target_orange:
                break
            new_inner[rr, cc] = ORANGE

        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BORDER, dtype=int)
        patch[1:-1, 1:-1] = new_inner

        dr, dc = DIRS[direction]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
        paste(out, patch, int(r0 + dr * steps), int(c0 + dc * steps), clip=True)

    return out


def _build_interior(inner_h: int, inner_w: int, direction: str, stationary: bool) -> np.ndarray:
    if stationary:
        return np.full((inner_h, inner_w), ORANGE, dtype=int)
    interior = np.full((inner_h, inner_w), GREY, dtype=int)
    for r, c in _arrow_template(inner_h, inner_w, direction):
        interior[r, c] = ORANGE
    return interior


def _draw_assembly(
    grid: np.ndarray,
    pr: int,
    pc: int,
    spec: dict,
) -> None:
    cr = pr + spec["cont_r"]
    cc = pc + spec["cont_c"]
    oh = spec["outer_h"]
    ow = spec["outer_w"]
    axis = spec["axis"]

    grid[cr, cc : cc + ow] = BORDER
    grid[cr + oh - 1, cc : cc + ow] = BORDER
    grid[cr : cr + oh, cc] = BORDER
    grid[cr : cr + oh, cc + ow - 1] = BORDER
    grid[cr + 1 : cr + oh - 1, cc + 1 : cc + ow - 1] = spec["interior"]

    if axis == "H":
        grid[cr : cr + oh, pc] = MARKER
        grid[cr : cr + oh, pc + spec["total_w"] - 1] = MARKER
    else:
        grid[pr, cc : cc + ow] = MARKER
        grid[pr + spec["total_h"] - 1, cc : cc + ow] = MARKER


def _expected_output(input_grid: np.ndarray, specs: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    expected = np.array(input_grid, copy=True)
    for spec, (pr, pc) in zip(specs, placements):
        if spec["steps"] <= 0:
            continue
        cr = pr + spec["cont_r"]
        cc = pc + spec["cont_c"]
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        dr, dc = DIRS[spec["direction"]]

        expected[cr : cr + oh, cc : cc + ow] = BG

        order = _fill_order(spec["inner_h"], spec["inner_w"], spec["direction"])
        interior = np.full((spec["inner_h"], spec["inner_w"]), GREY, dtype=int)
        for idx, (r, c) in enumerate(order):
            if idx >= spec["orange_after"]:
                break
            interior[r, c] = ORANGE

        patch = np.full((oh, ow), BORDER, dtype=int)
        patch[1:-1, 1:-1] = interior
        nr = cr + dr * spec["steps"]
        nc = cc + dc * spec["steps"]
        expected[nr : nr + oh, nc : nc + ow] = patch
    return expected


def _gen_attempt() -> np.ndarray | None:
    n_containers = random.choice([2, 2, 3, 3])
    n_decoys = 0 if n_containers == 2 else random.choice([0, 1])
    mover_count = n_containers - n_decoys
    if mover_count < 1:
        return None

    directions = [random.choice(list(DIRS.keys())) for _ in range(n_containers)]
    if mover_count >= 2 and len(set(directions[:mover_count])) == 1:
        directions[1] = random.choice([d for d in DIRS if d != directions[0]])

    specs: list[dict] = []
    for i in range(n_containers):
        stationary = i >= mover_count
        direction = directions[i]
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        axis = "H" if direction in ("L", "R") else "V"

        if axis == "H":
            left_steps_cap = random.randint(1, 5)
            right_steps_cap = random.randint(1, 5)
            total_h = outer_h
            total_w = 1 + left_steps_cap + outer_w + right_steps_cap + 1
            cont_r = 0
            cont_c = 1 + left_steps_cap
        else:
            up_steps_cap = random.randint(1, 5)
            down_steps_cap = random.randint(1, 5)
            total_h = 1 + up_steps_cap + outer_h + down_steps_cap + 1
            total_w = outer_w
            cont_r = 1 + up_steps_cap
            cont_c = 0

        interior = _build_interior(inner_h, inner_w, direction, stationary)
        grey_count = int(np.sum(interior == GREY))
        if stationary:
            steps = 0
        else:
            if direction == "L":
                steps_cap = left_steps_cap
            elif direction == "R":
                steps_cap = right_steps_cap
            elif direction == "U":
                steps_cap = up_steps_cap
            else:
                steps_cap = down_steps_cap
            steps = min(steps_cap, grey_count)
            if steps <= 0:
                return None

        orange_after = int(np.sum(interior == ORANGE)) + steps
        specs.append(
            {
                "direction": direction,
                "axis": axis,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "cont_r": cont_r,
                "cont_c": cont_c,
                "total_h": total_h,
                "total_w": total_w,
                "interior": interior,
                "steps": steps,
                "orange_after": orange_after,
            }
        )

    total_area = sum(spec["total_h"] * spec["total_w"] for spec in specs)
    max_h = max(spec["total_h"] for spec in specs)
    max_w = max(spec["total_w"] for spec in specs)
    min_side = max(10, int(total_area ** 0.5) + 3)
    grid_h = random.randint(max_h + 2, min(26, max(max_h + 2, min_side + 4)))
    grid_w = random.randint(max_w + 2, min(26, max(max_w + 2, min_side + 4)))
    if grid_h > 26 or grid_w > 26:
        return None

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for spec in specs:
        th = spec["total_h"]
        tw = spec["total_w"]
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
    for spec, (pr, pc) in zip(specs, placements):
        _draw_assembly(grid, pr, pc, spec)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(grid, specs, placements)
    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid variant after 400 attempts")
