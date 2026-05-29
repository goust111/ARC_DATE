# Variant of 28a6681f
# Dimension: D7 — 盆地结构形状 (obstacle)
# Change: 2-3个底部锚定的结构：U形、L形、阶梯形或单竖壁，相邻盆地壁高不等且可相互连通允许溢流... → 嵌套盆地（盆中盆）

from __future__ import annotations

import random

import numpy as np


# Design analysis
# a) 具体化与行为追踪
#    - 网格仍固定为 10x10，背景为 0，水为 1，障碍使用 2 种不同颜色。
#    - 原任务强调“相邻盆地之间的侧向溢流”；这个变体把障碍改成“外层大 U 形盆 + 内层小 U 形盆”。
#    - 外盆始终落在底边，内盆完全放在外盆内部，且内盆底比外盆底高 1 格，因此会出现两层可见容积：
#      1) 内盆腔体
#      2) 外盆与内盆之间的环形侧槽
#    - 输入水源不再依赖导流件。为了把 D7 的变化放到视觉中心，直接在内盆开口上方放一个 2 列、竖向堆叠的受支撑水柱。
#      这个水柱本身满足原 solver 的输入断言：每格都由下方水格或内盆底支撑，而且每格左右至少有一侧不是背景。
#    - 具体跟踪一个样例：
#      外盆左右墙在 c=1,8，内盆左右墙在 c=3,6，内盆墙顶在 r=5，输入水量为 14。
#      原题的“相邻盆地”思路会把额外水看成从一个盆溢到旁边的另一个盆。
#      这里同样的局部水流规则会先把 6 格水塞满内盆腔体 (r=5..7, c=4..5)，随后剩余 8 格被迫绕到两侧环槽
#      (r=5..8, c=2 与 c=7)，形成“先内后外”的分级填充。差异不在移动规则，而在障碍拓扑。
#
# b) 复杂度校准
#    - 如果内盆太宽，样例会退化成“只是一个更宽的普通盆”；如果内盆太窄，视觉上难以看出是嵌套结构。
#    - 这里固定内盆腔宽为 2、内盆底宽为 4、外盆总宽为 8。这样既能清楚看见内外两层，又不会引入太多自由度。
#    - 只保留两个合法水量档位：10 或 14。它们分别对应“填到内盆边缘并已溢入外环”的两个稳定层级，
#      能重复支持同一条规律；不采样半层水量，避免出现肉眼难解释的边缘状态。
#
# c) 隐式假设审查（围绕 main 中与变体相关的代码）
#    - 输入检查 `input water must be supported` 依赖“每个输入水格下方非背景”。生成器用竖直水柱并让其底部落在内盆底上，
#      所以前提仍成立。
#    - 输入检查 `input water cannot have bg on both horizontal sides` 在原题中约束侧向孤立水格。这里用宽 2 的水柱，
#      每格至少有一个水平水邻居，因此前提仍成立。
#    - `bottom_colors` / guide 检查原本会对顶部斜坡导流件做合法性断言。嵌套盆地全由水平/垂直墙构成，
#      对角相邻计数不会达到 guide 的阈值，因此不会被误判成导流件。
#    - `settle()` 的下落、对角滑动、水平找最近落点逻辑完全保留；变体只改变障碍形状，不改局部物理。
#    - 最关键的新隐式约束来自原 solver 的输出检查：最终每个水格左、右、下都必须非背景。
#      这意味着“刚越过内盆边缘但还没铺满外盆整层”的半层状态会被判为非法。
#      生成器因此只使用累计满层水量 10 / 14，严格避开非法半层。
#
# d) 放置合法性
#    - 外盆左右墙固定相距 7 列，始终完全落在网格内；外盆整体可左右平移 0/1/2 列。
#    - 内盆始终放在外盆内部，左右墙比外盆各缩进 2 列，因此不会与外盆重叠，也不会越界。
#    - 外盆底固定在 r=9，内盆底固定在 r=8；内盆墙顶只取 r=5 或 r=6，保证既看得出嵌套，又给输入水柱留出高度。
#    - 水柱只落在内盆腔体两列中，绝不覆盖障碍。
#
# e) 视觉可观察性
#    - 人类观察 2-4 个样例时，应直接看到：一个大盆里面还有一个小盆，输出里小盆先被填满，水再出现在外环。
#    - 为了让这一点稳定可见，这里不混入 guide、颜色排序、镜像规则等额外变化，只保留左右平移、墙高和颜色变化。
#    - 水量只取两个档位，样例间会反复出现“内盆已满且外环有水”的共同模式，便于归纳。
#
# f) 可视化整洁度
#    - 所有元素都处在同一静态场景：外盆、内盆、水源。
#    - 不让水源跨越多个不相干区域，也不让额外障碍散落到上半区，避免画面同时出现太多结构层次。
#    - 外盆和内盆用不同颜色，能快速分辨“包裹关系”。
#
# g) 颜色变化可推断性
#    - 颜色只用于区分两层静态障碍，不参与任何排序、匹配或变换逻辑。
#    - 水永远保持颜色 1，背景永远保持 0，因此解题者不需要推断隐藏的颜色映射。


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
    remaining = set(points)
    if use_diagonal:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _find_layout(grid: np.ndarray) -> dict[str, int]:
    solids = grid > 1
    assert solids.shape == (10, 10)
    bottom_cols = [int(c) for c in np.where(solids[9])[0]]
    assert bottom_cols, "missing outer basin base"
    outer_left = min(bottom_cols)
    outer_right = max(bottom_cols)

    inner_cols = [int(c) for c in np.where(solids[8])[0] if outer_left < int(c) < outer_right]
    assert inner_cols, "missing inner basin base"
    inner_left = min(inner_cols)
    inner_right = max(inner_cols)

    left_wall_rows = [int(r) for r in np.where(solids[:, inner_left])[0] if int(r) < 9]
    right_wall_rows = [int(r) for r in np.where(solids[:, inner_right])[0] if int(r) < 9]
    assert left_wall_rows and right_wall_rows, "missing inner basin walls"
    inner_top = min(min(left_wall_rows), min(right_wall_rows))

    outer_left_rows = [int(r) for r in np.where(solids[:, outer_left])[0]]
    outer_right_rows = [int(r) for r in np.where(solids[:, outer_right])[0]]
    outer_top = min(min(outer_left_rows), min(outer_right_rows))

    return {
        "outer_left": int(outer_left),
        "outer_right": int(outer_right),
        "outer_top": int(outer_top),
        "inner_left": int(inner_left),
        "inner_right": int(inner_right),
        "inner_top": int(inner_top),
    }


def _nested_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    layout = _find_layout(inp)
    inner_left = layout["inner_left"]
    inner_right = layout["inner_right"]
    inner_top = layout["inner_top"]
    outer_left = layout["outer_left"]
    outer_right = layout["outer_right"]

    inner_cells = [
        (r, c)
        for r in range(inner_top, 8)
        for c in range(inner_left + 1, inner_right)
    ]
    annulus_cells = [
        (r, c)
        for r in range(inner_top, 9)
        for c in range(outer_left + 1, outer_right)
        if c < inner_left or c > inner_right
    ]
    water_total = int(np.sum(out == 1))
    inner_fill = sum(1 for r, c in inner_cells if int(out[r, c]) == 1)
    annulus_fill = sum(1 for r, c in annulus_cells if int(out[r, c]) == 1)
    inner_capacity = len(inner_cells)
    annulus_capacity = len(annulus_cells)
    moved = int(np.any(inp != out))
    return {
        "water_total": water_total,
        "inner_fill": int(inner_fill),
        "annulus_fill": int(annulus_fill),
        "inner_capacity": int(inner_capacity),
        "annulus_capacity": int(annulus_capacity),
        "moved": moved,
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling into nested basins where a smaller cup sits inside a larger outer basin.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors draw static walls: one large bottom-anchored basin plus one smaller basin nested inside it.
    - Water begins as a supported source column above the inner opening, so some water is already stacked but not yet fully settled.

    Rule:
    - A water pixel first falls straight downward through background.
    - If the cell below is blocked, it may slide diagonally only through an open side opening.
    - If it is resting on support, it moves horizontally toward the nearest drop column and continues falling from there.
    - Repeating that local motion fills the inner basin first and only then spills water into the outer ring.

    Key insight:
    - The motion law is unchanged from the original task; the new challenge is to read the nested obstacle topology so the staged inner-to-outer fill can be predicted.
    """
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {bg, water} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                continue

            opts = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(bg)
                and int(g[rr + 1, cc - 1]) == int(bg)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(bg)
                and int(g[rr + 1, cc + 1]) == int(bg)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(bg):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(bg):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(bg)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(water):
                continue
            g[int(r), int(c)] = int(bg)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(water)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(bg)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(bg)

    return g


def _build_nested_basin(
    outer_left: int,
    outer_top: int,
    inner_top: int,
    outer_color: int,
    inner_color: int,
) -> np.ndarray:
    grid = np.zeros((10, 10), dtype=int)
    outer_right = outer_left + 7
    inner_left = outer_left + 2
    inner_right = inner_left + 3

    for r in range(outer_top, 10):
        grid[r, outer_left] = int(outer_color)
        grid[r, outer_right] = int(outer_color)
    for c in range(outer_left, outer_right + 1):
        grid[9, c] = int(outer_color)

    for r in range(inner_top, 9):
        grid[r, inner_left] = int(inner_color)
        grid[r, inner_right] = int(inner_color)
    for c in range(inner_left, inner_right + 1):
        grid[8, c] = int(inner_color)

    return grid


def _place_source_water(grid: np.ndarray, inner_top: int) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    layout = _find_layout(g)
    inner_left = layout["inner_left"]
    inner_right = layout["inner_right"]

    water_count = 2 + 4 * (8 - inner_top)
    height = water_count // 2
    start_row = 8 - height
    assert height in (5, 7)
    assert start_row >= 0

    for r in range(start_row, 8):
        for c in range(inner_left + 1, inner_right):
            assert int(g[r, c]) == 0
            g[r, c] = 1
    return g


def generate_input() -> np.ndarray:
    solid_colors = [2, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(200):
        inner_top = random.choice([5, 6])
        outer_top = random.randint(2, inner_top - 2)
        outer_left = random.choice([0, 1, 2])
        outer_color, inner_color = random.sample(solid_colors, 2)

        base = _build_nested_basin(
            outer_left=outer_left,
            outer_top=outer_top,
            inner_top=inner_top,
            outer_color=outer_color,
            inner_color=inner_color,
        )
        inp = _place_source_water(base, inner_top=inner_top)
        out = main(inp)
        metrics = _nested_metrics(inp, out)

        if metrics["moved"] != 1:
            continue
        if metrics["inner_fill"] != metrics["inner_capacity"]:
            continue
        if metrics["annulus_fill"] <= 0:
            continue
        if metrics["water_total"] <= metrics["inner_capacity"]:
            continue

        return inp

    raise AssertionError("failed to generate a nested-basin sample")
