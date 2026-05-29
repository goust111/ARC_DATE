# Variant of 28a6681f
# Dimension: D3 — 水平滑动规则 (path)
# Change: 支撑在非背景像素上时，水像素在当前行的连续背景段内找到最近的可落点（下方为背景的列），然后逐步水平移动过去... → 带惯性的滑动

from __future__ import annotations

import random

import numpy as np

# Design analysis for the D3 variant
# spec:
#   concretization:
#     - 保留 10x10、bg=0、water=1、其余颜色为静态障碍。
#     - 变体的关键场景不是“刚完成对角下落就立刻左右二选一”，因为原题的对角下落会天然把反方向的相邻格堵死；
#       真正会产生新旧分歧的是：水先对角下落获得方向，再竖直落入下一层平台，之后旧规则会按“最近落点”选边，
#       新规则则必须沿既有方向继续水平滑动。
#     - 生成器因此固定造一个两阶段路径：边缘水柱 -> 唯一对角入口 -> 竖直井道 -> 左右都有落点的平台 -> 双盆地。
#   behavior_trace:
#     - 以右侧进水为例，水先从 (3,7) 对角到 (4,6)，获得“向左”惯性。
#     - 之后它竖直落到平台格 (5,6)。平台右盆地的落井更近，但那要求向右回头；旧规则会去右盆地。
#     - 新规则保留“向左”，只能一路滑到更远的左侧落井，再进入左盆地。解题者必须追踪来源方向才能预测终点。
#   complexity_calibration:
#     - 只保留两个镜像分支：从左进水得到向右惯性，或从右进水得到向左惯性。
#     - 颜色只用 2 种固体色，避免颜色承担隐藏语义；方向完全由可见几何决定。
#     - 水量限制在 3-4 个，使差异明显但不把平台上层塞得过满。
#   implicit_assumptions_review:
#     - 原 solver 的“最近落点唯一”断言不再适用；惯性版本只看当前方向的下一格，因此删除该依赖。
#     - 原 solver 不需要粒子历史；变体必须在 settle() 内维护 momentum，并且该状态要穿过后续的竖直下落。
#     - 原 solver 的逐步重算可能在不同平台上换边；变体禁止换边，只有新的对角下落才能重置方向。
#     - 原 solver 的输出校验把“左右都非背景”当成稳定性的静态代理；变体中历史惯性不可从最终静态图直接读出，
#       因此改为检查输出在新求解器下是幂等固定点。
#   placement_legality:
#     - 源水柱贴着边墙放置，保证每个输入水格都被下方水格/实墙支撑，且不会出现左右两边同时为背景。
#     - 结构全部在 10x10 内显式写死，再通过水平镜像得到另一分支，避免越界和重叠。
#     - 生成器先用新 solver 正向注水得到稳定输出，再回填成支持良好的输入，并再次验证 main(input)==output。
#   visual_observability:
#     - 同一输入在旧规则下会优先落入“更近”的盆地，在新规则下会落入“惯性方向”的盆地，最终积水区域整体左右对调。
#     - 镜像样本会同时出现“向左惯性”和“向右惯性”，人类可以从 2-3 个例子中看到这是方向跟踪，而不是固定选左/右。
#   cleanliness:
#     - 水、边墙、双盆地、平台各自占据明确区域，没有额外装饰物覆盖同一区域。
#   new_solver_assumptions:
#     - 每次对角下落都必须唯一；生成器用边墙和单入口保证这一点。
#     - 平台至少在惯性方向上留出一段可滑动背景；否则水会立刻“撞墙停住”。模板已显式满足。
#     - 为了让变体可见，样本必须让新旧规则输出不同；生成器用 legacy 对照求解器筛掉退化样本。


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


def _solve(
    grid: np.ndarray,
    *,
    inertial_slide: bool,
    check_input: bool = True,
    check_output: bool = True,
) -> np.ndarray:
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
        momentum = 0
        for _ in range(H * W * 4):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
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
                nr, nc = opts[0]
                momentum = -1 if int(nc) < int(cc) else 1
                rr, cc = int(nr), int(nc)
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                if inertial_slide:
                    if momentum != 0:
                        nx = int(cc + momentum)
                        if 0 <= nx < W and int(g[rr, nx]) == int(bg):
                            cc = nx
                            continue
                else:
                    L = int(cc)
                    while L - 1 >= 0 and int(g[rr, L - 1]) == int(bg):
                        L -= 1
                    R = int(cc)
                    while R + 1 < W and int(g[rr, R + 1]) == int(bg):
                        R += 1
                    drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
                    if drops:
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
        stabilized = _solve(
            g,
            inertial_slide=inertial_slide,
            check_input=False,
            check_output=False,
        )
        assert np.array_equal(stabilized, g), "output is not a fixed point of the solver"

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate gravity-driven water settling into bottom basins, with horizontal motion controlled by carried inertia.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors are fixed walls, shelves, shafts, and basin rims anchored near the lower half of the grid.
    - Water begins as a supported edge source that must enter the basin area through a unique diagonal opening.

    Rule:
    - A water pixel first falls straight down through background cells whenever possible.
    - If the cell below is blocked, it may move through the unique open diagonal gap, and that diagonal step sets a left/right inertia direction.
    - While resting on support, the pixel may only slide one cell at a time in its stored inertia direction; it keeps that direction across later vertical drops, and it cannot reverse unless a new diagonal step resets the direction.
    - The process repeats for every water pixel until the whole configuration stops changing.

    Key insight: The final basin is determined not by the nearest later drop, but by the last diagonal entry direction that the water carries through subsequent shelves.
    """
    return _solve(
        grid,
        inertial_slide=True,
        check_input=check_input,
        check_output=check_output,
    )


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, inertial_slide=False, check_input=True, check_output=False)


def _build_base_structure(left_color: int, right_color: int) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)

    g[:5, 8] = int(right_color)
    g[4, 7] = int(right_color)
    g[5, 2] = int(left_color)
    g[5, 9] = int(right_color)

    g[6, 2] = int(left_color)
    g[6, 4:8] = int(right_color)
    g[6, 9] = int(right_color)

    g[7, 2] = int(left_color)
    g[7, 6] = int(right_color)
    g[7, 9] = int(right_color)

    g[8, 2] = int(left_color)
    g[8, 6] = int(right_color)
    g[8, 9] = int(right_color)

    g[9, 2:6] = int(left_color)
    g[9, 6:10] = int(right_color)

    return g


def _make_case(direction: int, left_color: int, right_color: int, water_count: int) -> tuple[np.ndarray, np.ndarray]:
    assert direction in (-1, 1)
    assert 3 <= water_count <= 4

    structure = _build_base_structure(int(left_color), int(right_color))
    source_col = 7
    if direction == 1:
        structure = np.fliplr(structure)
        source_col = structure.shape[1] - 1 - source_col

    output = structure.copy()
    for _ in range(water_count):
        assert int(output[0, source_col]) == 0
        output[0, source_col] = 1
        output = main(output, check_input=False, check_output=False)

    inp = structure.copy()
    source_rows = list(range(4 - water_count, 4))
    for r in source_rows:
        inp[r, source_col] = 1

    expected = main(inp)
    assert np.array_equal(expected, output), "output-first reconstruction mismatch"
    assert not np.array_equal(expected, _legacy_main(inp)), "degenerate sample matches legacy rule"
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(64):
        colors = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 2)
        direction = random.choice([-1, 1])
        water_count = random.choice([3, 4])
        inp, _out = _make_case(direction, colors[0], colors[1], water_count)
        return inp
    raise AssertionError("failed to generate a valid inertial-slide sample")
