# Variant of 28a6681f
# Dimension: D10 — 变换规则 (transform)
# Change: 结构像素不变，水像素守恒（数量不变），水从不稳定的输入位置流到稳定的输出位置... → 水量增殖

from __future__ import annotations

import random
import numpy as np

# Design analysis
# a) 具体化与行为追踪:
#    - 仍然使用 10x10 网格、黑底(0)、蓝水(1)、其余颜色作为静态障碍。
#    - 保留原任务的“边缘导轨 + 底部盆地”视觉骨架，但把上方导轨做成 4-connected 的锯齿楼梯。
#    - 水滴从导轨起点出发时，会连续执行若干次唯一的对角滑动；每执行一次对角滑动，
#      就在“对角起点”留下一个额外蓝像素，主水滴继续往下运动并最终落入底部盆地。
#    - 小例子: 原规则里 `(1,0)->(2,1)->(3,2)` 最终只留下终点水；新规则里会额外保留
#      `(1,0)` 和 `(2,1)` 两个蓝点，因此输出会出现沿楼梯的蓝色残迹，且总水量增加。
# b) 复杂度校准:
#    - 不引入多种复制条件，唯一的增殖触发器就是“发生对角滑动”。
#    - 每个样例只放 1-2 条边缘导轨，每条导轨长度 2-4，保证视觉上能看出重复的复制模式，
#      但不会因为对象过多而变乱。
# c) 隐式假设审查:
#    - 原 solver 依赖“水量守恒”，因此多轮反复处理整张图是安全的；新规则下若把新生成的水继续
#      当成可移动源，会导致规则语义膨胀甚至循环。这里改成: 只有输入中的原始水像素是移动源，
#      对角复制留下的像素是静态残迹。
#    - 原输出检查要求左右下都非背景，这对新残迹过强；新版本只要求每个输出水像素下方受支撑。
# d) 放置合法性:
#    - 导轨固定贴左/右边界，长度受限，行列范围始终在 10x10 内。
#    - 盆地只占底部 4 行，与上方导轨分离；生成器先放结构再放水，并用 solver 自检。
# e) 视觉可观察性:
#    - 输入里蓝水很少，输出里会出现“沿斜坡逐级留下蓝点”的稳定图案；这正是人类可见的规则差异。
#    - 同时用不对称盆地承接最终落水，让输出既有残迹也有汇聚结果，避免退化成简单复制。
# f) 可视化整洁度:
#    - 结构只有上方导轨和下方盆地两层，不做重叠标记。
#    - 颜色数控制在 3 种以内（蓝水 + 2 个结构色），保证残迹最醒目。
# g) 颜色变化可推断性:
#    - 变体不依赖颜色映射；结构颜色仅用于区分障碍分段，不参与任何隐藏排序或条件判断。


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


def _simulate(
    grid: np.ndarray,
    *,
    check_input: bool,
    check_output: bool,
    split_on_diagonal: bool,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
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

    def settle(r: int, c: int) -> tuple[int, int, list[tuple[int, int]]]:
        rr = int(r)
        cc = int(c)
        leaves: list[tuple[int, int]] = []
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
                if split_on_diagonal:
                    leaves.append((int(rr), int(cc)))
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                L = int(cc)
                while L - 1 >= 0 and int(g[rr, L - 1]) == int(bg):
                    L -= 1
                R = int(cc)
                while R + 1 < W and int(g[rr, R + 1]) == int(bg):
                    R += 1

                drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
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
        return int(rr), int(cc), leaves

    initial_sources = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
    split_events = 0
    moved = False
    created = 0

    for r, c in initial_sources:
        if int(g[r, c]) != int(water):
            continue
        g[r, c] = int(bg)
        rr, cc, leaves = settle(int(r), int(c))
        moved = bool(moved or leaves or ((int(rr), int(cc)) != (int(r), int(c))))
        g[rr, cc] = int(water)
        for lr, lc in leaves:
            if int(g[lr, lc]) == int(bg):
                g[lr, lc] = int(water)
                created += 1
        split_events += len(leaves)

    if check_output:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "output water must be supported"

    stats = {
        "input_water": len(initial_sources),
        "output_water": int(np.sum(g == int(water))),
        "split_events": int(split_events),
        "created_water": int(created),
        "moved": int(bool(moved)),
    }
    if return_stats:
        return g, stats
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate branching water that slides into basins while duplicating on diagonal turns.

    Input structure:
    - bg = 0 and water = 1, with the other colors forming fixed basin walls and edge-entering guide rails.
    - Source water is initially supported on ledges or guide starts, typically near the left or right border.
    - The static solids keep the original basin geometry; only the water behavior changes in this variant.

    Rule:
    - Water still falls straight down through background, then uses the same unique diagonal and horizontal slides as before.
    - Whenever a moving water pixel takes a diagonal step, it leaves behind an extra water pixel at the diagonal takeoff cell.
    - The moving pixel continues until it settles, while the copied trail pixels remain where they were created.

    Key insight: The output is not count-preserving anymore; each diagonal turn acts like a splitter, so the final blue pattern is the settled basin water plus a visible staircase trail of extra pixels.
    """
    return _simulate(
        grid,
        check_input=check_input,
        check_output=check_output,
        split_on_diagonal=True,
        return_stats=False,
    )


def _paint_left_guide(g: np.ndarray, color: int, length: int) -> None:
    for i in range(length):
        g[2 + i, i] = int(color)
        g[3 + i, i] = int(color)


def _paint_right_guide(g: np.ndarray, color: int, length: int) -> None:
    for i in range(length):
        g[2 + i, 9 - i] = int(color)
        g[3 + i, 9 - i] = int(color)


def _paint_basin(g: np.ndarray, left_color: int, right_color: int, rng: random.Random) -> None:
    g[9, 2:5] = int(left_color)
    g[9, 5:8] = int(right_color)
    g[6:9, 2] = int(left_color)
    g[6:9, 7] = int(right_color)
    if rng.random() < 0.6:
        g[8, 5] = int(right_color)


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [2, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(400):
        g = np.zeros((10, 10), dtype=int)
        left_color, right_color, guide_color = rng.sample(colors, 3)
        _paint_basin(g, left_color, right_color, rng)

        guide_sides = ["left", "right"] if rng.random() < 0.7 else [rng.choice(["left", "right"])]
        split_target = 0

        for side in guide_sides:
            max_len = 4 if len(guide_sides) == 1 else 3
            length = rng.randint(2, max_len)
            split_target += length
            if side == "left":
                _paint_left_guide(g, guide_color, length)
                g[1, 0] = 1
            else:
                _paint_right_guide(g, guide_color, length)
                g[1, 9] = 1

        if rng.random() < 0.35:
            side = rng.choice(guide_sides)
            if side == "left" and g[0, 0] == 0:
                g[0, 0] = 1
            elif side == "right" and g[0, 9] == 0:
                g[0, 9] = 1

        try:
            out, stats = _simulate(
                g,
                check_input=True,
                check_output=True,
                split_on_diagonal=True,
                return_stats=True,
            )
            base = _simulate(
                g,
                check_input=True,
                check_output=True,
                split_on_diagonal=False,
                return_stats=False,
            )
        except AssertionError:
            continue

        if stats["split_events"] < split_target:
            continue
        if stats["output_water"] <= stats["input_water"]:
            continue
        if np.array_equal(out, base):
            continue
        return g

    raise RuntimeError("failed to generate a valid D10 variant sample")
