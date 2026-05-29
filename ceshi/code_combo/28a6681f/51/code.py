# Combo variant of 28a6681f (index 51)
# combo_spec: D12/2+D15/2
# Name: 单轮处理+两侧均可为背景

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 画面仍是 10x10、0=背景、1=水、其余颜色=静态盆壁/台阶。底部仍保留“相邻盆地链”风格，
#   但上方水源改成竖直水柱或细长小团，明确允许某些源水格左右两侧都直接是背景。
# - 典型模板里，列 6 上有 3 个竖直水格，最下方旁边再带 1 个肩部水格，下面是一个会把水导入
#   左下盆地的台阶结构。输入时像 `(3,6)`、`(4,6)` 这种源水都满足“下方有支撑，但左右都是空白”。
# - 原题做法：按底到顶处理完这一轮后，会再开下一轮；因此最上面的那格水会在下方水已经腾空后继续下落，
#   最终也补进底部盆地。
# - 融合后做法：每个输入水像素只在这一轮里 settle 一次。于是先处理的低位水已经进入盆地，但高位水
#   在它自己的 turn 里只来得及落到中间台面；因为没有第二轮，它会残留在较高位置。
# - 差异是可见的：输出里一方面能看见“开放两侧”的源水布局，另一方面能看见仍悬在中层的残余水，
#   而不是全部灌到最低盆地。
#
# b) 复杂度校准
# - 不去追求任意随机地形，而是固定使用 5 个手工筛选过的“单轮会留下残余”的模板家族。
#   这些模板都共享相同视觉语法：上部细水柱 + 中层台面/竖壁 + 下部相连浅盆。
# - 每个模板只再做颜色重映射，避免自由度过大导致某些样例退化成原题的多轮稳定结果。
# - 模板里的单轮残余通常是 1 格；水总量只有 3-4 格，因此这个差异在人眼上已经足够明显。
#
# c) 对原 main() 的隐式假设审查
# - 原 `check_input` 的“左右不同时为背景”前提直接失效，必须删除；但“下方有支撑”仍保留，
#   否则输入会退化成任意空中散点，和原任务的注水语义差太远。
# - 原外层 `for _pass in range(H * W)` 依赖“重复多轮直到收敛”；D12/2 要求删掉这层循环，
#   只保留一轮按坐标顺序处理。
# - 原 `check_output` 要求每个水格左/下/右都被堵住。单轮处理后，某些水格会因为缺少第二轮而在
#   侧向仍然暴露，所以这个断言不再成立。这里改成只检查“下方仍受支撑”，并由生成器额外比较
#   单轮输出与迭代基线输出，保证样例确实体现单轮残余。
# - 原 `settle()` 对双对角开放写了 `assert len(opts) == 1`。D15/2 允许更自由的开阔源水后，
#   这类局面更容易出现。为了让 solver 定义闭合，这里改成固定选择更左的对角格；生成器本身并不
#   依赖这个 tie-break 才能成立，但 solver 不能再因为合法输入而崩溃。
# - 水平滑动的“最近落点唯一”假设仍保留；模板在入库前已筛过，不制造水平并列落点。
#
# d) 放置合法性
# - 所有模板都是显式 10x10 位图，只做颜色重映射，不做几何缩放或旋转，因此不会越界。
# - 源水与静态结构在模板里已经分层摆好；重映射只改颜色，不改占位关系，不会引入重叠。
# - 每个模板都满足至少 1 个输入水格“左右都是背景且下方有支撑”，并且每个模板都已经通过
#   单轮 solver 与迭代基线 solver 的对照验证。
#
# e) 视觉可观察性
# - 人类能直接看到：输入上方有瘦高水柱，而不是原题常见的贴墙侧仓；这体现 D15/2。
# - 输出里总会有至少 1 格水没有沉到最低盆地，而是停在更高的台面/通道；与“如果继续多轮它还会下去”
#   的直觉形成对比，这体现 D12/2。
# - 这两个特征都来自可见几何，不依赖颜色编号大小或隐藏排序键。
#
# f) 可视化整洁度
# - 语义层次固定为“上部源水 / 中部导流墙 / 下部盆地”，避免多个标记系统叠加。
# - 只使用水色加 2 个结构色，视觉上仍接近原题，不会因为过多附加编码而杂乱。
#
# g) 变化的可推断性
# - 颜色不参与规则推断，只区分不同静态结构区域。
# - 观察者只需要归纳两个几何事实：源水可以是左右敞开的受支撑细柱；输出不是完全稳定终态，
#   而是“做完一轮 settle 后就停”。
#
# 9) 新隐式假设与生成器保障
# - 新 solver 假设单轮输出至少仍然保持竖向支撑；模板家族都满足这一点。
# - 新 generator 假设“单轮输出”和“原始迭代输出”不同；它会显式比较两者，若相同就丢弃。
# - 新 generator 还要求至少 1 个 `open_side_source`，避免样例虽然通过新 solver、但在视觉上退化回原题。


BG = 0
WATER = 1
H = 10
W = 10
COLOR_POOL = (2, 3, 4, 5, 6, 7, 8, 9)


BASE_TEMPLATES: tuple[tuple[str, ...], ...] = (
    (
        "0000000000",
        "0000000000",
        "0000010000",
        "0000010000",
        "0000011000",
        "0000086666",
        "0000080000",
        "0000080080",
        "0000080880",
        "8888886880",
    ),
    (
        "0000000000",
        "0000000000",
        "0000000000",
        "0000001000",
        "0000001000",
        "0040001100",
        "0040055555",
        "0040000400",
        "0040000400",
        "5445554440",
    ),
    (
        "0000000000",
        "0000000000",
        "0000010000",
        "0000117777",
        "0002220000",
        "0000000000",
        "0007000000",
        "0007000000",
        "0007000200",
        "7777722220",
    ),
    (
        "0000000000",
        "0000000000",
        "0000000000",
        "0000100000",
        "0001188000",
        "0008800000",
        "0008700000",
        "0008700000",
        "0078700000",
        "8878787780",
    ),
    (
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000100",
        "0000000110",
        "0004003333",
        "0004000400",
        "0004000400",
        "0044000400",
        "4444333430",
    ),
)


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=int)


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


def _count_open_side_sources(g: np.ndarray) -> int:
    count = 0
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
        supported = rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
        if left_bg and right_bg and supported:
            count += 1
    return int(count)


def _check_optional_guides(g: np.ndarray) -> None:
    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _simulate(
    grid: np.ndarray,
    *,
    single_pass: bool,
    check_input: bool,
    check_output: bool,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    stats = {
        "input_water": int(np.sum(g == int(WATER))),
        "open_side_sources": _count_open_side_sources(g),
        "ambiguous_diagonal_events": 0,
        "moved_pixels": 0,
        "output_water": 0,
    }

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        _check_optional_guides(g)

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
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
                if len(opts) > 1:
                    stats["ambiguous_diagonal_events"] += 1
                rr, cc = min(opts, key=lambda rc: (int(rc[1]), int(rc[0])))
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
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    def run_one_pass() -> bool:
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(int(r), int(c))
            if (int(rr), int(cc)) != (int(r), int(c)):
                moved = True
                stats["moved_pixels"] += 1
            g[rr, cc] = int(WATER)
        return bool(moved)

    if single_pass:
        run_one_pass()
    else:
        for _pass in range(H * W):
            if not run_one_pass():
                break
        else:
            raise AssertionError("water simulation did not converge")

    stats["output_water"] = int(np.sum(g == int(WATER)))

    if check_output:
        assert stats["output_water"] == stats["input_water"], "water count must be conserved"
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "output water must stay supported"

    if return_stats:
        return g, stats
    return g


def _iterative_reference(grid: np.ndarray) -> np.ndarray:
    return _simulate(
        grid,
        single_pass=False,
        check_input=True,
        check_output=False,
        return_stats=False,
    )


def _recolor_template(base: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    labels = sorted(int(x) for x in set(map(int, out.flatten())) if int(x) > int(WATER))
    colors = rng.sample(list(COLOR_POOL), len(labels))
    mapping = {label: color for label, color in zip(labels, colors)}
    for src, dst in mapping.items():
        out[out == int(src)] = int(dst)
    return out


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    single, single_stats = _simulate(
        inp,
        single_pass=True,
        check_input=True,
        check_output=True,
        return_stats=True,
    )
    iterative = _iterative_reference(inp)
    single_only = int(np.sum((single == int(WATER)) & (iterative != int(WATER))))
    iterative_only = int(np.sum((iterative == int(WATER)) & (single != int(WATER))))
    return {
        "open_side_sources": int(single_stats["open_side_sources"]),
        "single_only_cells": int(single_only),
        "iterative_only_cells": int(iterative_only),
        "ambiguous_diagonal_events": int(single_stats["ambiguous_diagonal_events"]),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate one-pass water settling into bottom basins, allowing supported source water to sit with both horizontal sides open.

    Input structure:
    - `0` is background and `1` is water; the other colors are rigid basin walls, shelves, and dividers.
    - Water still starts in supported cells, but unlike the base task it may appear as a skinny column or small clump whose left and right neighbors are both background.
    - The static structures are unchanged by the transformation.

    Rule:
    - Each water pixel falls straight down through background, can slip diagonally through open side cells, and can slide horizontally toward a unique reachable drop exactly as before.
    - If both diagonals are available at once, the move is resolved deterministically toward the left so the freer source layouts remain solvable.
    - The crucial change is global scheduling: scan the current water pixels once from bottom to top, settle each of them once, and stop immediately without any outer convergence loop.

    Key insight:
    - The output is not the fully stabilized basin fill from the original task; it is the intermediate state after exactly one sweep, so some water can remain stranded on a higher shelf even though another sweep would let it descend farther.
    """
    return _simulate(
        grid,
        single_pass=True,
        check_input=check_input,
        check_output=check_output,
        return_stats=False,
    )


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        base = _rows_to_grid(rng.choice(BASE_TEMPLATES))
        candidate = _recolor_template(base, rng)
        try:
            metrics = _behavior_metrics(candidate)
        except AssertionError:
            continue
        if metrics["open_side_sources"] < 1:
            continue
        if metrics["single_only_cells"] < 1 or metrics["iterative_only_cells"] < 1:
            continue
        return candidate
    raise AssertionError("failed to generate a non-degenerate D12/2+D15/2 sample")
