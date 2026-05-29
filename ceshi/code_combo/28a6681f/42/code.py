# Combo variant of 28a6681f (index 42)
# combo_spec: D10/0+D18/1
# Name: 水量增殖+分裂消歧

from __future__ import annotations

import random

import numpy as np

# Design analysis
# a) 具体化与行为追踪:
#    - 保留 10x10、bg=0、water=1、其余颜色为静态结构的骨架，但把上部导轨改成单边锯齿，
#      中部放一个中央立柱，底部放左右两个浅盆。
#    - 典型轨迹: 左侧输入水先沿导轨经历 3 次“唯一对角下滑”，例如
#      `(1,0)->(2,1)->(3,2)->(4,3)`；每次对角起跳点都会留下一个静态蓝像素。
#    - 随后主水滴会垂直落到中央分叉点，如 `(6,3)`；这里正下方被立柱挡住，而左下和右下都开放，
#      因此原题会 assert 的“双可行对角”在本变体中改为分裂：同时生成左支和右支两个活动水滴。
#    - 两个子水滴分别落入左右浅盆。于是输出同时包含: 导轨上的逐级残迹、分叉点残迹、以及两个盆中的终点水。
#      相比原题“水量守恒且对角无歧义”，这里在唯一对角时会留痕，在双对角时还会继续一分为二，二者叠加。
# b) 复杂度校准:
#    - 生成器只使用两类紧凑模板: 3 级导轨或 4 级导轨，再允许左右镜像。
#    - 每个样例都至少出现 1 次分叉和多次留痕，重复证据足够；同时模板数量不多，避免样例间规则漂移。
#    - 不再额外引入水平歧义，保证观察重点集中在“对角留痕 + 对角分裂”这两个联动机制上。
# c) 隐式假设审查:
#    - 原 solver 的外层“反复扫描所有水像素直到稳定”依赖水量守恒；现在新生成的蓝像素如果继续被当作活动源，
#      语义会膨胀。这里改成: 只有输入源和分裂产生的活动分支会继续运动；对角留下的蓝像素是静态残迹。
#    - 原 `settle()` 依赖 `len(opts)==1`。D18/1 破坏了这个前提，因此改为:
#      `len(opts)==2` 时留下当前格的水，并向两个对角目标各派生一条活动分支。
#    - 原输出检查要求左右下都非背景；导轨残迹通常左右仍有背景，因此新版本只检查“下方受支撑”。
#    - 分裂后的两个分支是按固定顺序处理的，因此若两支会争抢同一空间，结果会依赖处理次序。
#      生成器显式构造左右独立浅盆，保证这条新假设成立。
# d) 放置合法性:
#    - 模板全部在 10x10 内硬编码，镜像后仍在界内。
#    - 导轨颜色只出现在上部，不接触底行，因此继承的“edge-entering guide”检查仍然合法。
#    - 水源只放在导轨入口处，输入时都满足“下方受支撑、两侧不同时为空”的原始前置条件。
# e) 视觉可观察性:
#    - 人眼能直接看到输入只有一小段边缘水，输出却同时出现“导轨蓝色残迹”和“左右双盆各有终点水”，
#      这比单纯的留痕或单纯的分裂都更强烈。
#    - 若只应用 D10/0，中央只会选一侧盆；若只应用 D18/1，导轨上不会留下连续残迹。
#      生成器会显式排除这两种退化情形。
# f) 可视化整洁度:
#    - 结构层次固定为“上导轨 / 中央立柱 / 下双盆”，空间分区清晰。
#    - 颜色只用 3 种结构色以内，蓝水与结构对比强，不会出现同一区域多种标记互相遮挡。
# g) 变化的可推断性:
#    - 不存在颜色映射规则。颜色只区分静态结构段，不参与任何优先级、排序或计数。
#    - 解题者只需根据可见几何关系判断哪里会留痕、哪里会一分为二。
# 9) 新隐式假设与生成器保障:
#    - `_simulate()` 仍假设水平最近可落点唯一；生成器不制造水平并列落点。
#    - `_simulate()` 允许多个活动分支，但默认它们不会在狭窄通道内彼此抢占；生成器用分叉后左右分盆来保证。
#    - `_simulate()` 要求分叉点下方是静态支撑，且左右对角目标为空；模板在构造时固定满足这些空间关系。

BG = 0
WATER = 1


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


def _build_template(guide_len: int) -> np.ndarray:
    assert guide_len in (3, 4)
    g = np.zeros((10, 10), dtype=int)

    for i in range(guide_len):
        g[2 + i, i] = 3
        g[3 + i, i] = 3

    if guide_len == 3:
        g[7:10, 3] = 2
        g[7:9, 1] = 2
        g[7:9, 5] = 4
        g[9, 1:6] = 2
    else:
        g[8:10, 4] = 2
        g[8, 2] = 2
        g[8, 6] = 4
        g[9, 2:7] = 2

    g[1, 0] = WATER
    return g


def _colorize_structure(template: np.ndarray, rng: random.Random) -> np.ndarray:
    palette = [2, 3, 4, 5, 6, 7, 8, 9]
    guide_color, left_color, right_color = rng.sample(palette, 3)
    out = np.array(template, dtype=int).copy()
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if int(out[r, c]) == 2:
                out[r, c] = int(left_color)
            elif int(out[r, c]) == 4:
                out[r, c] = int(right_color)
            elif int(out[r, c]) == 3:
                out[r, c] = int(guide_color)
    return out


def _simulate(
    grid: np.ndarray,
    *,
    check_input: bool,
    check_output: bool,
    leave_trails: bool,
    split_on_ambiguous_diag: bool,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    stats = {
        "input_water": int(np.sum(g == int(WATER))),
        "output_water": 0,
        "diag_events": 0,
        "single_diag_events": 0,
        "split_events": 0,
        "branches_spawned": 0,
        "trail_cells_created": 0,
        "settled_cells_created": 0,
    }

    def spawn_branch(start_r: int, start_c: int) -> None:
        stack = [(int(start_r), int(start_c))]
        while stack:
            rr, cc = stack.pop()
            stats["branches_spawned"] += 1
            for _ in range(H * W):
                if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                    rr += 1
                    continue

                opts: list[tuple[int, int]] = []
                if (
                    rr + 1 < H
                    and cc - 1 >= 0
                    and int(g[rr, cc - 1]) == int(BG)
                    and int(g[rr + 1, cc - 1]) == int(BG)
                ):
                    opts.append((int(rr + 1), int(cc - 1)))
                if (
                    rr + 1 < H
                    and cc + 1 < W
                    and int(g[rr, cc + 1]) == int(BG)
                    and int(g[rr + 1, cc + 1]) == int(BG)
                ):
                    opts.append((int(rr + 1), int(cc + 1)))
                if opts:
                    stats["diag_events"] += len(opts) if len(opts) == 2 and split_on_ambiguous_diag else 1
                    if len(opts) == 2 and split_on_ambiguous_diag:
                        stats["split_events"] += 1
                    else:
                        stats["single_diag_events"] += 1

                    if leave_trails and int(g[rr, cc]) == int(BG):
                        g[rr, cc] = int(WATER)
                        stats["trail_cells_created"] += 1

                    if len(opts) == 2 and split_on_ambiguous_diag:
                        # Children are processed left-to-right; generated inputs keep the two exits disjoint.
                        stack.extend(reversed(opts))
                        break

                    next_rr, next_cc = opts[0] if len(opts) == 1 else min(opts, key=lambda rc: int(rc[1]))
                    rr, cc = int(next_rr), int(next_cc)
                    continue

                if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
                    left = int(cc)
                    while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                        left -= 1
                    right = int(cc)
                    while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                        right += 1

                    drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                    if not drops:
                        break

                    min_dist = min(abs(int(x - cc)) for x in drops)
                    nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                    assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                    best = nearest[0]
                    if best == int(cc):
                        break
                    step = -1 if best < int(cc) else 1
                    nx = int(cc + step)
                    assert int(g[rr, nx]) == int(BG)
                    cc = nx
                    continue

                break
            else:
                raise AssertionError("water branch did not settle")

            if int(g[rr, cc]) == int(BG):
                g[rr, cc] = int(WATER)
                stats["settled_cells_created"] += 1

    initial_sources = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
    for r, c in initial_sources:
        if int(g[r, c]) != int(WATER):
            continue
        g[r, c] = int(BG)
        spawn_branch(int(r), int(c))

    stats["output_water"] = int(np.sum(g == int(WATER)))

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "output water must be supported"

    if return_stats:
        return g, stats
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate branching water that both leaves diagonal residue and duplicates at forked slides.

    Input structure:
    - `0` is background and `1` is water; the other colors are fixed solids forming an edge-entering guide,
      a central blocking pillar, and one or two shallow bottom basins.
    - Source water starts as a small supported edge-adjacent run or single droplet at the guide entrance.
    - The solids never move; only the water behavior changes in this combo variant.

    Rule:
    - Water still falls straight down through background and can still slide horizontally toward a unique nearest drop.
    - Whenever a moving water branch takes a diagonal step, it leaves a water pixel behind at the takeoff cell.
    - If both diagonal directions are open at once, the moving branch does not choose one side: it splits into two
      active branches, one per diagonal exit, and both continue independently.

    Key insight: The output combines two distinct growth mechanisms. Ordinary diagonal turns create a visible trail,
    while ambiguous diagonal forks create extra moving branches, so a single source can paint a staircase residue and
    also end up filling multiple basins.
    """
    result = _simulate(
        grid,
        check_input=check_input,
        check_output=check_output,
        leave_trails=True,
        split_on_ambiguous_diag=True,
        return_stats=False,
    )
    assert isinstance(result, np.ndarray)
    return result


def generate_input() -> np.ndarray:
    rng = random.Random()
    templates = [_build_template(3), _build_template(4)]

    for _ in range(200):
        grid = np.array(rng.choice(templates), dtype=int).copy()
        if rng.random() < 0.5:
            grid = np.fliplr(grid)
        grid = _colorize_structure(grid, rng)

        combo, combo_stats = _simulate(
            grid,
            check_input=True,
            check_output=True,
            leave_trails=True,
            split_on_ambiguous_diag=True,
            return_stats=True,
        )
        assert isinstance(combo, np.ndarray)
        assert isinstance(combo_stats, dict)

        split_only = _simulate(
            grid,
            check_input=True,
            check_output=True,
            leave_trails=False,
            split_on_ambiguous_diag=True,
            return_stats=False,
        )
        trails_only = _simulate(
            grid,
            check_input=True,
            check_output=True,
            leave_trails=True,
            split_on_ambiguous_diag=False,
            return_stats=False,
        )

        if combo_stats["split_events"] < 1:
            continue
        if combo_stats["trail_cells_created"] < 3:
            continue
        if combo_stats["output_water"] <= combo_stats["input_water"]:
            continue
        if np.array_equal(combo, split_only):
            continue
        if np.array_equal(combo, trails_only):
            continue
        return grid

    raise RuntimeError("failed to generate a valid combo sample")
