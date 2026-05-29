# Combo variant of 28a6681f (index 22)
# combo_spec: D20/2+D21/0
# Name: 输出仅显示水+guide消失
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题：输入里能看到 guide、盆壁和源水；输出仍保留这些结构，只是水换到了稳定位置。
#    - 融合后：仍然要先在完整结构上按原物理模拟，但最终返回值把所有非水像素全部擦成背景。
#      因而输出是同尺寸 10x10 的“纯水掩码”：只有背景 0 和水 1，所有盆壁与 guide 都消失。
#    - 具体小例子（基于本生成器的 funnel 模板）：
#      * 输入上半部有一个从左右边缘汇向中部的 guide 漏斗，5 格水源停在其上；
#      * 下半部有两级相邻盆地；
#      * 原题完整终态会同时显示 guide、盆壁和 5 格 settled water；
#      * 本变体先得到同样的完整终态，再把全部固体结构擦掉，只留下例如
#        {(7,1), (7,2), (7,4), (8,1), (8,4)} 这样的纯水坐标集合；
#      * 若把 guide 预先删掉再跑同一输入，水会走向不同盆地，或直接触发双对角歧义。
#        所以 D20/2 不是被 D21/0 覆盖掉，而是变成“guide 只在推理阶段存在、在验证图里完全不可见”。
# b) 复杂度校准
#    - 家族刻意收窄为 2 个手工筛过的 guide+basin 模板，每个模板再配 2 个不同水量、可选镜像、
#      随机换色。这样 guide 对流路的影响会重复出现，且 output-only 的视觉差异足够稳定。
#    - 自由度控制在少量离散档位：模板 2 种、源水量 2 档、左右镜像 2 档、结构配色若干。
#      既避免退化成单一死模板，也避免随机地形导致流路歧义。
# c) 隐式假设审查（逐段对应原 main）
#    - 原 `check_input` 对输入水的两个前提仍成立：每格水必须有下方支撑，且左右不能同时是背景。
#      生成器只使用经验证的 source layouts，因此保留这些断言是安全的。
#    - 原 guide 校验块默认“guide 位于上 7 行、颜色不出现在底行、并且从边缘进入”。融合后这一点
#      仍成立，因为生成器强制所有样本都带一个这种 guide，并给它分配独立于盆壁的颜色。
#    - 原 `settle()` 与外层多轮收敛逻辑只依赖占用关系，不依赖输出是否保留结构，因此无需改变。
#    - 真正失效的是原 `check_output` 的解释：返回网格里已经没有任何支撑结构，若继续对返回值检查
#      “左/下/右非背景”会错误失败。解决方式是：先在完整 settled 场景上做稳定性断言，再单独把
#      结构整体擦除。
# d) 放置合法性
#    - 模板是显式 10x10 位图；镜像只做左右翻转，不做缩放旋转，因此不会越界。
#    - guide、盆壁和源水占位在模板里已分层排好，换色不会改变几何关系，不会产生重叠。
#    - 每个 source 变体都已人工验证满足输入合法性；生成器还会二次运行 `main()` 复核。
# e) 视觉可观察性
#    - 人类从 2-4 组样例里能直接看到：输入总含有 guide 和盆地，但输出完全看不到任何结构，
#      只剩最终水形。
#    - 同时，因为 guide 删除后会改变流路，解题者不能把 guide 当成装饰；必须先用输入里的结构
#      在脑中模拟，再拿纯水输出核对位置。
# f) 可视化整洁度
#    - 输入画面只有三层语义：上部 guide+源水、中下部盆地、输出中的纯水掩码。
#    - 输出没有任何额外标记或颜色映射，避免多层编码挤在同一区域造成杂乱。
# g) 变化的可推断性
#    - 没有颜色值相关的隐藏规则。颜色只区分不同固体组件；最终一律全部擦除。
#    - 可归纳的变化规律只有两步：先按原水流物理找到最终水位，再把所有非水像素清空。
# 8) 融合行为量化指标
#    - `solid_pixels_out`: 输出中非 0/1 像素数，目标恒为 0，用于验证 D21/0。
#    - `guide_effect`: 与“删掉 guide 后再模拟”的结果相比，最终水掩码的对称差像素数；若删 guide
#      会导致模拟失败，则记为整板大小 100。生成器要求该值至少为 4，用于验证 D20/2 的 guide
#      确实参与了水路控制，而不是仅仅被顺手擦掉。
#    - `moved_pixels`: 完整模拟期间发生位置变化的水像素次数，要求至少为 2，避免退化成静止水。
# 9) 新 main() 的隐式假设与生成器保障
#    - 假设：guide 若存在，满足“边缘进入、4 连通、上半区、颜色独立”。保障：模板固定此结构，
#      `recolor` 也始终给 guide 独立颜色。
#    - 假设：返回前擦除结构不会抹掉最终水。保障：guide 全在上半部，盆壁与最终水不重叠；完整
#      settled 场景中的水永远只落在原背景格上。
#    - 假设：删掉 guide 做对照时，哪怕输入合法性被破坏，也不会影响正题 solver 的定义。保障：
#      对照只用于指标统计，调用时关闭输入校验；真实 `main()` 仍只对完整合法输入负责。

from __future__ import annotations

import random

import numpy as np


BG = 0
WATER = 1
H = 10
W = 10
GUIDE_LABEL = 9
COLOR_POOL = (2, 3, 4, 5, 6, 7, 8)


BASE_TEMPLATES = (
    {
        "name": "funnel",
        "rows": (
            "0000000000",
            "9900000099",
            "0990000990",
            "0099000900",
            "0009009900",
            "0000099000",
            "0000040000",
            "2004040000",
            "2024040000",
            "2224440000",
        ),
        "sources": (
            ((1, 2), (1, 3), (2, 3)),
            ((1, 2), (1, 3), (2, 3), (0, 2), (0, 3)),
        ),
    },
    {
        "name": "slope",
        "rows": (
            "0000000000",
            "9900000000",
            "0990000000",
            "0099900005",
            "0000999005",
            "0000000505",
            "3000000005",
            "3300005505",
            "3330055505",
            "3333555505",
        ),
        "sources": (
            ((0, 0), (0, 1), (1, 2), (1, 3), (2, 3)),
            ((0, 0), (0, 1), (1, 2), (1, 3), (2, 3), (0, 2), (0, 3)),
        ),
    },
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
    return int(count)


def _detect_guide_mask(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    mask = np.zeros_like(g, dtype=bool)
    bottom_colors = set(map(int, g[H - 1]))
    guide_colors = sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors)
    for color in guide_colors:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"
            for r, c in pts:
                mask[int(r), int(c)] = True
    return mask


def _settle(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
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
            assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
            rr, cc = opts[0]
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


def _simulate_full(
    grid: np.ndarray,
    check_input: bool = True,
    check_output: bool = True,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    assert g.shape == (H, W), "grid must be 10x10"

    input_water = int(np.sum(g == int(WATER)))
    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"
        _detect_guide_mask(g)

    moved_pixels = 0
    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = _settle(g, int(r), int(c))
            if (int(rr), int(cc)) != (int(r), int(c)):
                moved = True
                moved_pixels += 1
            g[int(rr), int(cc)] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        assert int(np.sum(g == int(WATER))) == input_water, "water count must be conserved"
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    if return_stats:
        return g, {
            "input_water": int(input_water),
            "output_water": int(np.sum(g == int(WATER))),
            "moved_pixels": int(moved_pixels),
        }
    return g


def _water_only_view(full_grid: np.ndarray) -> np.ndarray:
    return np.where(np.array(full_grid, dtype=int) == int(WATER), int(WATER), int(BG))


def _mirror_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(W - 1 - c)) for r, c in cells)


def _recolor_solids(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    solid_labels = sorted(int(v) for v in set(map(int, out.flatten())) if int(v) > int(WATER))
    other_labels = [label for label in solid_labels if label != int(GUIDE_LABEL)]
    guide_color = int(rng.choice(COLOR_POOL))
    available = [int(c) for c in COLOR_POOL if int(c) != int(guide_color)]
    other_colors = rng.sample(available, k=len(other_labels))
    mapping = {int(GUIDE_LABEL): int(guide_color)}
    mapping.update({int(label): int(color) for label, color in zip(other_labels, other_colors)})
    for src, dst in mapping.items():
        out[out == int(src)] = int(dst)
    return out


def _build_candidate(rng: random.Random) -> np.ndarray:
    template = rng.choice(BASE_TEMPLATES)
    grid = _rows_to_grid(template["rows"])
    sources = tuple(tuple(cell) for cell in rng.choice(template["sources"]))
    if rng.random() < 0.5:
        grid = np.fliplr(grid).copy()
        sources = _mirror_cells(sources)
    grid = _recolor_solids(grid, rng)
    for r, c in sources:
        assert int(grid[int(r), int(c)]) == int(BG), "source water overlapped solids"
        grid[int(r), int(c)] = int(WATER)
    return grid


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    full, stats = _simulate_full(inp, check_input=True, check_output=True, return_stats=True)
    out = _water_only_view(full)
    guide_mask = _detect_guide_mask(inp)
    no_guide = np.array(inp, dtype=int).copy()
    no_guide[np.array(guide_mask, dtype=bool)] = int(BG)
    try:
        no_guide_full = _simulate_full(no_guide, check_input=False, check_output=False, return_stats=False)
        guide_effect = int(np.count_nonzero((_water_only_view(no_guide_full) != out)))
    except Exception:
        guide_effect = int(H * W)
    return {
        "guide_cells": int(np.count_nonzero(guide_mask)),
        "solid_pixels_out": int(np.count_nonzero(out > int(WATER))),
        "erase_diff": int(np.count_nonzero(full != out)),
        "guide_effect": int(guide_effect),
        "moved_pixels": int(stats["moved_pixels"]),
        "water_only_delta": int(np.count_nonzero((inp == int(WATER)) != (out == int(WATER)))),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate guide-assisted water flow, then return only the final water mask.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors form static basin walls in the lower half and may also form an upper guide that
      enters from the left or right edge to steer the source water toward the basin chain.
    - Water begins in supported source cells near that guide rather than already resting in its final basin.

    Rule:
    - First run the original local water physics on the full scene: fall vertically through background,
      slip diagonally only through a uniquely open side channel, and slide horizontally along a supported
      ledge toward the nearest unique drop until the configuration stabilizes.
    - After the full settled state has been obtained, erase every non-water pixel, including all basin walls
      and any guide pixels, by turning them into background.
    - Return the same-size grid containing only background and the final water positions.

    Key insight: The structures still matter for simulation, but they are only a hidden scaffold for the
    answer; the visible output is a pure water footprint, so one must mentally reconstruct the vanished guide
    and basin geometry from the input before validating the returned water pattern.
    """
    settled = _simulate_full(grid, check_input=check_input, check_output=check_output, return_stats=False)
    out = _water_only_view(settled)
    if check_output:
        assert int(np.sum(out == int(WATER))) == int(np.sum(np.array(grid, dtype=int) == int(WATER)))
        assert int(np.count_nonzero(out > int(WATER))) == 0, "output must contain only background and water"
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        inp = _build_candidate(rng)
        try:
            out = main(inp)
            metrics = _variant_metrics(inp)
        except Exception:
            continue
        if metrics["guide_cells"] < 6:
            continue
        if metrics["solid_pixels_out"] != 0:
            continue
        if metrics["erase_diff"] < 12:
            continue
        if metrics["guide_effect"] < 4:
            continue
        if metrics["moved_pixels"] < 2:
            continue
        if metrics["water_only_delta"] < 2:
            continue
        assert np.array_equal(out, main(inp))
        return inp
    raise RuntimeError("failed to generate a non-degenerate D20/2+D21/0 sample")
