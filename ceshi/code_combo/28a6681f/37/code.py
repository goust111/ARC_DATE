# Combo variant of 28a6681f (index 37)
# combo_spec: D4/1+D11/1+D12/0+D16/0
# Name: 多色水+分批处理+蒸发+壁面摩擦

from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
ROUGH = 3
SMOOTH = 4
GUIDE = 5

GRID_SHAPE = (10, 10)
WATER_BATCHES = (BLUE, RED)
STATIC_COLORS = (ROUGH, SMOOTH, GUIDE)
ALLOWED_COLORS = {BG, *WATER_BATCHES, *STATIC_COLORS}

_TEMPLATE_ROWS = (
    ("5533300000", "0033300400", "0033304440"),
    ("5533300000", "0033304000", "0033304440"),
    ("5533000400", "0033300400", "0033344440"),
    ("5533300000", "0033304040", "0033344440"),
    ("5533000400", "0033304400", "0033344400"),
)


# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 网格仍固定为 10x10，但现在可动水固定用两种颜色：蓝色 1 和红色 2。静态结构颜色固定分工：
#   3=高摩擦粗糙壁，4=低摩擦光滑壁，5=源侧托架/导台。
# - 生成器只保留一个清晰子家族：左侧或右侧贴边水库，红水总在蓝水上方；下方是粗糙台阶接光滑盆地。
# - 原任务里，所有水像素同时参与迭代，且质量守恒；滑到哪里只是由空位和最近坠口决定。
# - 融合后要分四层推理：
#   1. 先只处理蓝水，红水暂时视为“未来批次”，不参与碰撞；
#   2. 蓝水不是整片同时更新，而是把贴边水库当作 supply reservoir，按几何顺序逐滴放出；
#   3. 每个水滴的垂直下落免费，但每次对角/水平移动都会消耗一个同色后续水滴，所以远路会减员；
#   4. 蓝水全部落稳后才开始红水，且蓝水已经是实墙，会改变红水路径和蒸发量。
# - 具体小例子：左边 3 列红水压在 4 列蓝水之上。蓝水先落到粗糙台边，部分像素因为粗糙支撑而停在高处，
#   只有少量蓝水穿过光滑区域进入右侧深盆，但这些穿行步骤会蒸发掉若干后续蓝水。然后红水开始下落；
#   如果按原任务，红水也能直接进入深盆。融合后，深盆入口已被蓝水抬高/堵住，红水只能更早转向或更快蒸发。
#
# b) 复杂度校准
# - 水色只用两种，壁面功能只用粗糙/光滑两档，避免把任务变成“太多颜色角色需要硬记”。
# - 生成器复用少量底部模板，但每个样例里粗糙壁和光滑壁都会重复出现，解题者能从单个样例里看到两种壁面的对比。
# - 水源自由度只保留 4 个：左右镜像、蓝水宽度、红水宽度、蓝/红各占 2 或 3 行。这样变化够多，但不会导致路径分叉失控。
#
# c) 对原 main() 隐式假设的审查
# - `water = 1` 失效：现在必须跟踪两种水，并且批次顺序固定为 `BLUE -> RED`。
# - “所有水同时存在于工作网格”失效：当前批次之外的水要先从工作网格移除，否则红水会错误地提前阻挡蓝水。
# - “每个输入水像素最终都会落成一个输出水像素”失效：非垂直步数会吃掉后续同色 supply，因此只能保证 `output_count <= input_count`。
# - “支撑面颜色只做视觉区分”失效：水平滑动前必须查看 `g[r+1, c]` 是否为 `ROUGH`。
# - 原版输出检查里“左右两侧都不能是背景”失效：粗糙台边允许水停在侧向开口旁，所以改成更强的局部重放稳定性检查。
# - 新增冲突点：蒸发需要一个离散像素语义。这里采用“当前水滴落位，但它经历的每一个非垂直步都会删掉一个同色后续水滴”。
#   这保证总损失量精确等于非垂直步数，同时仍能让少量水到达远处深盆。
#
# d) 放置合法性
# - 所有模板都只在底部 3 行画静态结构，且整体锚定到底边，避免与上方源水重叠。
# - 水源永远贴住单侧边框，红在上、蓝在下，形成一个连续支撑柱：每个水像素下方不是同色/异色水，就是 5/3/4 托架。
# - 镜像操作在整张 10x10 上做，合法性会完整保留：仍然只接触一个边界，不会越界，也不会产生双侧开口水像素。
#
# e) 视觉可观察性
# - 2-4 组样例里，人类能直接看到四个重复信号：
#   1. 蓝色总是先落位并占深位/入口位；
#   2. 蓝色经常卡在粗糙壁上方，而不是像原题那样继续横滑；
#   3. 到达更远光滑盆地的水量明显减少，说明横移/斜移会蒸发；
#   4. 红色终态紧贴或被蓝色顶住，说明后批次受前批次阻挡。
# - 颜色角色不是隐藏排序：蓝=第一批、红=第二批、3=粗糙、4=光滑，这些角色在每个样例里都重复出现。
#
# f) 可视化整洁度
# - 图面分成上下两层：上半是整齐的贴边双色水库，下半是 3 行厚的盆地/台阶。
# - 同一区域不叠加过多编码：水的颜色只表示批次身份，墙的颜色只表示摩擦类型，不再引入额外标记色。
# - 生成器要求输出里蓝红至少有一次正交接触，并要求粗糙停留与远处盆地同时出现，避免只看到一堆杂色静止块。
#
# g) 变化的可推断性
# - 这里只有两种动态颜色和两种功能壁色，映射很少而且有重复支撑，不需要记复杂条件表。
# - 批次顺序不是靠数字大小暗示，而是靠样例中持续出现的“蓝先占位、红后受阻”现象推断。
# - 蒸发也不是隐藏计数：远路样例总是比无蒸发反事实少水，且低摩擦模板会稳定产生更多减员。
#
# 9) 新 solver 假设与 generator 保证
# - 假设：所有源水接触且只接触一个边界。保证：`generate_input()` 只生成单侧贴边水库。
# - 假设：同一批次 supply 的几何顺序可由“下方优先、再按朝盆地方向优先”唯一确定。保证：水库始终是规则矩形贴边堆叠。
# - 假设：当一个水滴开始下落时，它的起点单元仍为空。保证：模板只保留单调排水结构，候选样例还会经过 `main()` 验证。
# - 假设：对角/最近坠口选择唯一。保证：模板只保留一个明确的缺口/深盆方向，歧义候选会被 solver assert 拒绝。


def _grid_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


def _water_mask(arr: np.ndarray) -> np.ndarray:
    return np.isin(arr, WATER_BATCHES)


def _water_points(arr: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(_water_mask(arr))]


def _source_side(grid: np.ndarray) -> str:
    left = bool(np.isin(grid[:, 0], WATER_BATCHES).any())
    right = bool(np.isin(grid[:, -1], WATER_BATCHES).any())
    assert left ^ right, "source water must touch exactly one vertical edge"
    return "left" if left else "right"


def _ordered_source_cells(grid: np.ndarray, color: int, side: str) -> list[tuple[int, int]]:
    points = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
    if side == "left":
        return sorted(points, key=lambda rc: (-int(rc[0]), -int(rc[1])))
    return sorted(points, key=lambda rc: (-int(rc[0]), int(rc[1])))


def _horizontal_run(grid: np.ndarray, rr: int, cc: int) -> tuple[int, int, list[int]]:
    h, w = grid.shape
    assert rr + 1 < h
    left = int(cc)
    while left - 1 >= 0 and int(grid[rr, left - 1]) == int(BG):
        left -= 1
    right = int(cc)
    while right + 1 < w and int(grid[rr, right + 1]) == int(BG):
        right += 1
    drops = [int(x) for x in range(int(left), int(right) + 1) if int(grid[rr + 1, int(x)]) == int(BG)]
    return int(left), int(right), drops


def _trace_unit(
    grid: np.ndarray,
    start: tuple[int, int],
    *,
    friction_enabled: bool,
) -> tuple[tuple[int, int], int, dict[str, int]]:
    h, w = grid.shape
    rr, cc = map(int, start)
    nonvertical_steps = 0
    rough_blocks = 0

    for _ in range(h * w):
        if rr + 1 < h and int(grid[rr + 1, cc]) == int(BG):
            rr += 1
            continue

        opts: list[tuple[int, int]] = []
        if (
            rr + 1 < h
            and cc - 1 >= 0
            and int(grid[rr, cc - 1]) == int(BG)
            and int(grid[rr + 1, cc - 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc - 1)))
        if (
            rr + 1 < h
            and cc + 1 < w
            and int(grid[rr, cc + 1]) == int(BG)
            and int(grid[rr + 1, cc + 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc + 1)))
        if opts:
            assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
            rr, cc = opts[0]
            nonvertical_steps += 1
            continue

        if rr + 1 < h and int(grid[rr + 1, cc]) != int(BG):
            _left, _right, drops = _horizontal_run(grid, int(rr), int(cc))
            support_color = int(grid[rr + 1, cc])
            if friction_enabled and support_color == int(ROUGH):
                if drops:
                    rough_blocks += 1
                break
            if not drops:
                break
            min_dist = min(abs(int(x - cc)) for x in drops)
            nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
            assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
            best = int(nearest[0])
            assert best != int(cc)
            step = -1 if best < int(cc) else 1
            nx = int(cc + step)
            assert int(grid[rr, nx]) == int(BG)
            cc = nx
            nonvertical_steps += 1
            continue

        break
    else:
        raise AssertionError("water unit did not settle")

    return (int(rr), int(cc)), int(nonvertical_steps), {"rough_blocks": int(rough_blocks)}


def _simulate_batches(
    grid: np.ndarray,
    *,
    friction_enabled: bool,
    evaporation_enabled: bool,
    blue_blocks_red: bool,
    collect_metrics: bool,
) -> tuple[np.ndarray, dict[str, int]] | np.ndarray:
    src = np.array(grid, dtype=int).copy()
    side = _source_side(src)
    settled = np.where(np.isin(src, STATIC_COLORS), src, int(BG))

    metrics = {
        "rough_blocks": 0,
        "nonvertical_steps": 0,
        "initial_blue": int(np.count_nonzero(src == int(BLUE))),
        "initial_red": int(np.count_nonzero(src == int(RED))),
        "surviving_blue": 0,
        "surviving_red": 0,
    }

    for color in WATER_BATCHES:
        queue = _ordered_source_cells(src, int(color), side)
        if color == int(RED) and not blue_blocks_red:
            settled = np.where(settled == int(BLUE), int(BG), settled)

        while queue:
            start = queue.pop(0)
            assert int(settled[start]) == int(BG), "source cell must stay open while a batch is pouring"
            end, cost, unit_metrics = _trace_unit(settled, start, friction_enabled=friction_enabled)
            assert int(settled[end]) == int(BG), "settled water cannot overwrite an occupied cell"
            settled[end] = int(color)

            if color == int(BLUE):
                metrics["surviving_blue"] += 1
            else:
                metrics["surviving_red"] += 1

            if collect_metrics:
                metrics["rough_blocks"] += int(unit_metrics["rough_blocks"])
                metrics["nonvertical_steps"] += int(cost)

            if evaporation_enabled:
                for _ in range(int(cost)):
                    if queue:
                        queue.pop()

    if not collect_metrics:
        return settled
    return settled, metrics


def _cross_color_contacts(grid: np.ndarray) -> int:
    contacts = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) not in WATER_BATCHES:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                rr = int(r + dr)
                cc = int(c + dc)
                if rr >= grid.shape[0] or cc >= grid.shape[1]:
                    continue
                if int(grid[r, c]) != int(grid[rr, cc]) and int(grid[rr, cc]) in WATER_BATCHES:
                    contacts += 1
    return int(contacts)


def _water_on_support(grid: np.ndarray, *, water_color: int, support_color: int) -> int:
    count = 0
    for r, c in np.argwhere(grid == int(water_color)):
        rr = int(r)
        cc = int(c)
        if rr + 1 < grid.shape[0] and int(grid[rr + 1, cc]) == int(support_color):
            count += 1
    return int(count)


def _moved_cells(inp: np.ndarray, out: np.ndarray) -> int:
    moved = 0
    for color in WATER_BATCHES:
        in_pts = {(int(r), int(c)) for r, c in np.argwhere(inp == int(color))}
        out_pts = {(int(r), int(c)) for r, c in np.argwhere(out == int(color))}
        moved += len(in_pts.symmetric_difference(out_pts))
    return int(moved)


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    combo, combo_metrics = _simulate_batches(
        grid,
        friction_enabled=True,
        evaporation_enabled=True,
        blue_blocks_red=True,
        collect_metrics=True,
    )
    no_friction = _simulate_batches(
        grid,
        friction_enabled=False,
        evaporation_enabled=True,
        blue_blocks_red=True,
        collect_metrics=False,
    )
    no_evaporation = _simulate_batches(
        grid,
        friction_enabled=True,
        evaporation_enabled=False,
        blue_blocks_red=True,
        collect_metrics=False,
    )
    red_without_blue_block = _simulate_batches(
        grid,
        friction_enabled=True,
        evaporation_enabled=True,
        blue_blocks_red=False,
        collect_metrics=False,
    )

    return {
        "rough_blocks": int(combo_metrics["rough_blocks"]),
        "nonvertical_steps": int(combo_metrics["nonvertical_steps"]),
        "friction_diff_cells": int(np.count_nonzero(combo != no_friction)),
        "evaporation_diff_cells": int(np.count_nonzero(combo != no_evaporation)),
        "red_obstacle_diff": int(np.count_nonzero((combo == int(RED)) != (red_without_blue_block == int(RED)))),
        "surviving_blue": int(combo_metrics["surviving_blue"]),
        "surviving_red": int(combo_metrics["surviving_red"]),
        "blue_loss": int(combo_metrics["initial_blue"] - combo_metrics["surviving_blue"]),
        "red_loss": int(combo_metrics["initial_red"] - combo_metrics["surviving_red"]),
        "cross_contacts": int(_cross_color_contacts(combo)),
        "blue_on_rough": int(_water_on_support(combo, water_color=BLUE, support_color=ROUGH)),
        "moved_cells": int(_moved_cells(np.array(grid, dtype=int), combo)),
    }


def _validate_input(grid: np.ndarray) -> None:
    assert grid.shape == GRID_SHAPE, "grid must be 10x10"
    vals = set(map(int, np.unique(grid)))
    assert vals <= ALLOWED_COLORS, f"unexpected colors: {sorted(vals - ALLOWED_COLORS)}"
    assert np.any(grid == int(BLUE)), "input must contain blue water"
    assert np.any(grid == int(RED)), "input must contain red water"
    assert np.any(grid == int(ROUGH)), "input must contain rough walls"
    assert np.any(grid == int(SMOOTH)), "input must contain smooth walls"
    _source_side(grid)

    for r, c in _water_points(grid):
        rr = int(r)
        cc = int(c)
        assert rr == grid.shape[0] - 1 or int(grid[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(grid[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < grid.shape[1] and int(grid[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"


def _validate_output(inp: np.ndarray, out: np.ndarray) -> None:
    assert out.shape == inp.shape
    vals = set(map(int, np.unique(out)))
    assert vals <= ALLOWED_COLORS, f"unexpected output colors: {sorted(vals - ALLOWED_COLORS)}"

    inp_static = np.where(np.isin(inp, STATIC_COLORS), inp, int(BG))
    out_static = np.where(np.isin(out, STATIC_COLORS), out, int(BG))
    assert np.array_equal(inp_static, out_static), "static structure must remain unchanged"

    for color in WATER_BATCHES:
        assert int(np.count_nonzero(out == int(color))) <= int(np.count_nonzero(inp == int(color))), "water cannot increase"

    replay = np.array(out, dtype=int).copy()
    for r, c in _water_points(replay):
        rr = int(r)
        cc = int(c)
        color = int(replay[rr, cc])
        replay[rr, cc] = int(BG)
        end, _cost, _metrics = _trace_unit(replay, (int(rr), int(cc)), friction_enabled=True)
        replay[rr, cc] = int(color)
        assert tuple(map(int, end)) == (int(rr), int(cc)), "output water is not locally stable"


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate two colored water batches settling into linked basins where wall color controls friction and sideways travel burns off later supply.

    Input structure:
    - Background is 0. Blue water uses color 1 and red water uses color 2.
    - Static terrain uses fixed roles: color 3 is a rough support that cancels horizontal spill, color 4 is a smooth spill surface, and color 5 is a source-side ledge/guide.
    - The source water is a single edge-packed reservoir with red above blue, and the lower rows form bottom-anchored adjacent shelves and basins.

    Rule:
    - Process the colors in batch order: first settle every blue supply unit, then settle every red one on top of the unchanged walls plus the already-settled blue water.
    - Within one color batch, treat the edge reservoir as a queue of individual water units released from lower cells first toward the basin side.
    - A released unit may fall vertically for free; every diagonal descent or horizontal slide counts as one evaporative cost.
    - That cost removes the same number of later units from the current color's remaining queue, so long smooth detours reduce how much water survives to settle.
    - Horizontal motion is allowed only on smooth supports; if the current support is rough, the unit stops instead of sliding.

    Key insight: The answer depends on composing three visible controls at once: blue settles before red, rough walls pin water early and save mass, while smooth longer routes reach deeper cells but consume later supply, changing both counts and red's eventual path through the blue-shaped obstacle field.
    """
    src = np.array(grid, dtype=int).copy()
    if check_input:
        _validate_input(src)

    out = _simulate_batches(
        src,
        friction_enabled=True,
        evaporation_enabled=True,
        blue_blocks_red=True,
        collect_metrics=False,
    )

    if check_output:
        _validate_output(src, out)

    return out


def _build_candidate(rng: random.Random) -> np.ndarray:
    rows = ["0000000000"] * 7 + list(rng.choice(_TEMPLATE_ROWS))
    grid = _grid_from_rows(rows)

    blue_width = rng.choice([4, 5])
    red_width = rng.choice([2, 3, 4])
    blue_rows_n = rng.choice([2, 3])
    red_rows_n = rng.choice([2, 3])

    blue_rows = list(range(7 - int(blue_rows_n), 7))
    red_start = int(7 - blue_rows_n - red_rows_n)
    red_rows = list(range(int(red_start), int(7 - blue_rows_n)))
    assert red_rows and blue_rows

    for r in blue_rows:
        grid[int(r), : int(blue_width)] = int(BLUE)
    for r in red_rows:
        grid[int(r), : int(red_width)] = int(RED)

    if bool(rng.randrange(2)):
        grid = np.fliplr(grid).copy()

    return grid


def generate_input(rng: random.Random | None = None) -> np.ndarray:
    rng = random.Random() if rng is None else rng
    for _ in range(400):
        candidate = _build_candidate(rng)
        try:
            out = main(candidate, check_input=True, check_output=True)
            stats = variant_metrics(candidate)
        except AssertionError:
            continue

        if stats["rough_blocks"] < 1:
            continue
        if stats["nonvertical_steps"] < 8:
            continue
        if stats["friction_diff_cells"] < 4:
            continue
        if stats["evaporation_diff_cells"] < 8:
            continue
        if stats["red_obstacle_diff"] < 4:
            continue
        if stats["surviving_blue"] < 4 or stats["surviving_red"] < 2:
            continue
        if stats["blue_loss"] < 1 or stats["red_loss"] < 2:
            continue
        if stats["cross_contacts"] < 1:
            continue
        if stats["blue_on_rough"] < 1:
            continue
        if stats["moved_cells"] < 6:
            continue
        if int(np.count_nonzero(out == int(RED))) < 2:
            continue
        return candidate

    raise AssertionError("failed to generate a visible multi-batch evaporating-friction sample")
