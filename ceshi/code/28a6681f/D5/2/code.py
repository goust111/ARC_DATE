# Variant of 28a6681f
# Dimension: D5 — 水源对象形态 (src)
# Change: 蓝色像素——guide样本使用guide旁的小水柱（3-10像素），无guide样本使用边缘侧的大水库或不规则片状水体（10-25像素） → 控制像素标记的水源量

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D5 variant
# ----------------------------------
# a) 具体化与行为追踪
# - 画面仍是 10x10、背景 0、蓝色水 1，底部盆地/guide 的水物理不改。
# - 唯一核心变化是“输入里不再直接画水”。改为在顶边放 1-2 个控制像素，
#   控制像素所在列就是注水列，颜色固定映射到释放量：
#   6 -> 3 格水，7 -> 5 格水，8 -> 7 格水。
# - solver 的第一步不再是“读取已有水像素并让它们继续流”，而是
#   “读取控制像素 -> 把这些像素清空为背景 -> 按控制量逐滴从该列顶端注水 ->
#   每次注入后用原题同一套重力/对角/水平滑动规则把水沉降到稳定状态”。
# - 具体小例子：
#   原题输入若在右上角直接画 10 个蓝点，它们会沿右侧通道落入两个相邻盆地；
#   现在输入只画两个控制像素，比如顶边第 8 列放 7、第 9 列放 6，
#   表示“第 8 列放 5 滴、第 9 列放 3 滴”。solver 解码后得到同类物理过程，
#   但旧 solver 在同一输入上看不到任何蓝水，只会原样返回，因此新旧行为显著不同。
#
# b) 复杂度校准
# - 参考任务 4a21e3da 的 control pixel 机制本质上很轻：边缘上的专用标记像素先被读取，
#   再驱动后续变换。这里保持同等级复杂度，不引入多步密码或大表查找。
# - 数量编码只允许 3 个离散档位 3/5/7；控制像素总数控制在 1-2 个。
#   这样 2-4 个训练样例里通常会重复出现相同颜色，解题者可从输出水量反推颜色语义。
# - guide 样例保持小水量 3-10；无 guide 样例保持大水量 10-21。
#   这延续原题“guide 小、reservoir 大”的整体视觉分层，只把可见水体换成控制标记。
#
# c) 原 main() 中与变体相关的隐式假设审查
# - 原 `check_input` 假设输入里已经有水，而且每个水像素都被支撑。D5 中输入没有可见水，
#   所以这段检查必须改成“检查控制像素是否合法”，否则会把合法变体输入误判为非法。
# - 原主循环直接对 `np.argwhere(g == water)` 迭代。D5 下这会得到空集，旧代码会退化成原样返回。
#   因此必须在主循环前新增“控制像素 -> 隐藏水像素”的解码阶段。
# - 原 guide 检测会扫描上半区的非背景色。D5 中顶边新增控制色 6/7/8；如果不排除，
#   单像素控制标记会被混入“上半区彩色对象”集合。虽然它们通常不会触发对角 guide 断言，
#   但这是脆弱写法，所以显式把控制色从 guide 候选中排除。
# - 原 `check_output` 对稳定水的左右下约束仍然成立，因为变体只改了水源表示，没有改终止物理。
#
# d) 放置合法性
# - 生成器只用从原题评测样例抽出的 4 个无水结构模板，并允许水平镜像；
#   这样盆地结构、guide 形态、可溢流关系都严格落在原题分布内。
# - 控制像素只放在模板顶边原本是背景的位置；不会与 guide 或实心结构重叠。
# - 每个模板都预先给出安全的注水列候选，避免把控制像素放到会立刻造成双向歧义的位置。
# - 生成器每次采样后都直接调用新 `main()` 验证；若某次控制组合导致源口堵塞或路径断言失败，
#   该尝试会被丢弃。
#
# e) 视觉可观察性
# - 输入和输出里都保留控制像素，因此“这些边缘标记在驱动蓝水”是可见线索，不是一次性消失的提示。
# - 同一控制色跨样例反复出现，且输出蓝水总量/分布随控制组合变化，肉眼可归纳出
#   “先读边缘控制像素，再按数量放水”的两阶段规则。
# - 因为旧题的盆地和 guide 仍然保留，人类能把变化聚焦到 source 形态，而不是重新学习整个任务。
#
# f) 可视化整洁度
# - 控制像素固定在顶边一行，不与下方盆地水面叠加。
# - 结构色仅沿用原题样例里的 2/3/4/5/9，控制色单独用 6/7/8，语义边界清晰。
#
# g) 颜色变化可推断性
# - 没有颜色排序或颜色 id 运算。6/7/8 的作用是固定语义标签，而不是依赖数值大小。
# - 颜色种类仍然很少：0 背景，1 水，2/3/4/5/9 结构，6/7/8 控制。
#
# h) 量化退化检查
# - 退化风险是“虽然加了控制像素，但它们不起作用，输出等同于无水输入”。
# - 因此额外统计指标定义为：
#   1. `decoded_water`: 由控制像素解码出的总注水量。
#   2. `delta_vs_visible_baseline`: 新输出是否不同于“把输入当成没有可见水”的旧物理输出。
#   对本变体，合格样本必须 `decoded_water > 0` 且 `delta_vs_visible_baseline = True`。
#
# i) 新 main() 的输入假设与 generator 保证
# - 假设所有控制像素都在顶边且颜色属于 {6,7,8}；生成器严格这样放置。
# - 假设顶边控制口在每次单滴注水前是空的；生成器用 `main()` 反复验证后才接受样本。
# - 假设输入不含可见水；生成器始终输出纯结构 + 控制像素。


BG = 0
WATER = 1
CONTROL_TO_AMOUNT = {6: 3, 7: 5, 8: 7}
CONTROL_COLORS = tuple(sorted(CONTROL_TO_AMOUNT))
GUIDE_COLOR = 9
H = 10
W = 10


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


def _validate_guides(g: np.ndarray) -> None:
    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - set(CONTROL_COLORS) - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _simulate_visible_water(
    grid: np.ndarray,
    check_input: bool = True,
    check_output: bool = True,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()

    if check_input:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    _validate_guides(g)

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts = []
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
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == W - 1 or int(g[rr, cc + 1]) != int(BG)

    return g


def _decode_controls(grid: np.ndarray, check_input: bool = True) -> list[tuple[int, int, int]]:
    controls: list[tuple[int, int, int]] = []
    for color in CONTROL_COLORS:
        for r, c in np.argwhere(grid == int(color)):
            controls.append((int(r), int(c), int(color)))
    controls.sort(key=lambda x: (x[0], x[1]))

    if check_input:
        assert controls, "input needs at least one control pixel"
        assert int(np.sum(grid == WATER)) == 0, "variant input should not contain visible water"
        for r, c, color in controls:
            assert r == 0, "control pixels must sit on the top edge"
            assert int(grid[r, c]) == int(color)
        for color in CONTROL_COLORS:
            pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
            for r, c in pts:
                assert r == 0, "control colors may only appear on the top edge"

    return controls


def _ordered_controls(controls: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    mean_col = sum(c for _r, c, _color in controls) / len(controls)
    reverse = bool(mean_col >= (W - 1) / 2)
    return sorted(controls, key=lambda x: x[1], reverse=reverse)


def _pour_hidden_water(base_grid: np.ndarray, controls: list[tuple[int, int, int]]) -> np.ndarray:
    g = np.array(base_grid, dtype=int).copy()
    for r, c, _color in controls:
        g[int(r), int(c)] = int(BG)

    for _r, c, color in _ordered_controls(controls):
        amount = int(CONTROL_TO_AMOUNT[int(color)])
        for _ in range(amount):
            assert int(g[0, int(c)]) == int(BG), "control column source mouth became blocked"
            g[0, int(c)] = int(WATER)
            g = _simulate_visible_water(g, check_input=False, check_output=False)

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Decode top-edge control pixels into hidden water pours, then let that water settle through the original basin network.

    Input structure:
    - Background is 0 and settled output water is still color 1.
    - Bottom-anchored colored obstacles form the same basins, steps, and optional edge-entering guide as in the base task.
    - Instead of visible source water, the top edge contains 1-2 control pixels in colors 6/7/8.
    - Each control pixel marks a release column and encodes a fixed batch size of hidden water for that column.

    Rule:
    - Read every control pixel, convert its color into a water amount, and temporarily clear the control cells back to background.
    - For each encoded batch, inject that many water pixels one by one from the marked top-edge column.
    - After every injection, apply the same gravity rule as the original task: fall vertically, use the unique open diagonal when blocked, and slide horizontally toward the nearest unique drop.
    - When all hidden pours have settled, restore the control pixels in place and keep every obstacle pixel unchanged.

    Key insight:
    - The physical water behavior is unchanged; the only missing information is the source itself. The solver must first interpret the edge control markers to reconstruct how much unseen water enters each column.
    """
    g = np.array(grid, dtype=int).copy()
    assert g.shape == (H, W), f"expected 10x10 grid, got {g.shape}"

    controls = _decode_controls(g, check_input=check_input)
    sim = _pour_hidden_water(g, controls)

    if check_output:
        stable = _simulate_visible_water(sim, check_input=False, check_output=True)
        assert np.array_equal(stable, sim), "decoded water must already be fully settled"
        expected_water = sum(int(CONTROL_TO_AMOUNT[color]) for _r, _c, color in controls)
        assert int(np.sum(sim == WATER)) == int(expected_water), "hidden water count must be conserved"

    out = np.array(sim, dtype=int).copy()
    for r, c, color in controls:
        assert int(out[r, c]) == int(BG), "control cell cannot be occupied by settled water"
        out[r, c] = int(color)
    return out


def _template_grid(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=int)


TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "name": "right_reservoir_two_basin",
        "guide": False,
        "control_cols": [8, 9],
        "grid": _template_grid(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 3, 0, 3, 0, 0],
                [0, 0, 0, 0, 0, 3, 0, 3, 0, 0],
                [0, 0, 0, 0, 2, 3, 0, 3, 0, 0],
                [0, 0, 0, 0, 2, 3, 0, 3, 0, 0],
                [0, 2, 2, 2, 2, 3, 3, 3, 0, 0],
            ]
        ),
    },
    {
        "name": "right_reservoir_stacked_basin",
        "guide": False,
        "control_cols": [7, 8, 9],
        "grid": _template_grid(
            [
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 3, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 3, 5, 0, 0, 5],
                [0, 0, 3, 0, 0, 3, 5, 0, 0, 5],
                [0, 0, 3, 0, 0, 3, 5, 0, 0, 5],
                [0, 0, 3, 3, 3, 3, 5, 5, 5, 5],
            ]
        ),
    },
    {
        "name": "double_arm_funnel",
        "guide": True,
        "control_cols": [2, 3],
        "grid": _template_grid(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [9, 9, 0, 0, 0, 0, 0, 0, 9, 9],
                [0, 9, 9, 0, 0, 0, 0, 9, 9, 0],
                [0, 0, 9, 9, 0, 0, 0, 9, 0, 0],
                [0, 0, 0, 9, 0, 0, 9, 9, 0, 0],
                [0, 0, 0, 0, 0, 9, 9, 0, 0, 0],
                [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
                [2, 0, 0, 4, 0, 4, 0, 0, 0, 0],
                [2, 0, 2, 4, 0, 4, 0, 0, 0, 0],
                [2, 2, 2, 4, 4, 4, 0, 0, 0, 0],
            ]
        ),
    },
    {
        "name": "left_guide_to_right_basin",
        "guide": True,
        "control_cols": [0, 1, 2, 3],
        "grid": _template_grid(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [9, 9, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 9, 9, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 9, 9, 9, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 9, 9, 9, 0, 0, 5],
                [0, 0, 0, 0, 0, 0, 0, 5, 0, 5],
                [3, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                [3, 3, 0, 0, 0, 0, 5, 5, 0, 5],
                [3, 3, 3, 0, 0, 5, 5, 5, 0, 5],
                [3, 3, 3, 3, 5, 5, 5, 5, 0, 5],
            ]
        ),
    },
)


def _mirror_template(template: dict[str, object]) -> dict[str, object]:
    mirrored = dict(template)
    mirrored["name"] = f"{template['name']}_mirrored"
    mirrored["grid"] = np.fliplr(np.array(template["grid"], dtype=int))
    mirrored["control_cols"] = [int(W - 1 - c) for c in reversed(list(template["control_cols"]))]
    return mirrored


ALL_TEMPLATES = TEMPLATES + tuple(_mirror_template(template) for template in TEMPLATES)
GUIDE_TEMPLATES = tuple(template for template in ALL_TEMPLATES if bool(template["guide"]))
NO_GUIDE_TEMPLATES = tuple(template for template in ALL_TEMPLATES if not bool(template["guide"]))
NO_GUIDE_TEMPLATE_POOL = (
    NO_GUIDE_TEMPLATES[0],
    NO_GUIDE_TEMPLATES[1],
    NO_GUIDE_TEMPLATES[1],
    NO_GUIDE_TEMPLATES[2],
    NO_GUIDE_TEMPLATES[3],
    NO_GUIDE_TEMPLATES[3],
)
GUIDE_TEMPLATE_POOL = GUIDE_TEMPLATES

SAFE_CONTROL_RECIPES: dict[str, list[tuple[list[int], tuple[int, ...]]]] = {
    "right_reservoir_two_basin": [
        ([0, 1], (6, 8)),
        ([0, 1], (7, 7)),
        ([0, 1], (7, 8)),
        ([0, 1], (8, 6)),
        ([0, 1], (8, 7)),
        ([0, 1], (8, 8)),
    ],
    "right_reservoir_stacked_basin": [
        ([0, 1], (7, 8)),
        ([0, 1], (8, 8)),
        ([1, 2], (7, 8)),
        ([1, 2], (8, 8)),
        ([0, 1], (6, 8)),
        ([1, 2], (6, 8)),
    ],
    "double_arm_funnel": [
        ([0], (6,)),
        ([0], (7,)),
        ([1], (6,)),
        ([1], (7,)),
        ([0, 1], (6, 8)),
        ([0, 1], (7, 7)),
        ([0, 1], (8, 6)),
    ],
    "left_guide_to_right_basin": [
        ([0], (6,)),
        ([0], (7,)),
        ([1], (6,)),
        ([1], (7,)),
        ([2], (6,)),
        ([2], (7,)),
        ([3], (6,)),
        ([3], (7,)),
    ],
}


def _water_component_count(grid: np.ndarray) -> int:
    pts = {(int(r), int(c)) for r, c in np.argwhere(grid == int(WATER))}
    if not pts:
        return 0
    return _component_count(pts, use_diagonal=False)


def _choose_controls(
    template: dict[str, object],
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    base_name = str(template["name"]).removesuffix("_mirrored")
    control_cols = list(map(int, template["control_cols"]))
    indices, colors = rng.choice(SAFE_CONTROL_RECIPES[base_name])
    cols = [int(control_cols[idx]) for idx in indices]
    return [(0, int(c), int(color)) for c, color in zip(cols, colors)]


def _gen_attempt(rng: random.Random, want_guide: bool) -> np.ndarray | None:
    template_pool = GUIDE_TEMPLATE_POOL if want_guide else NO_GUIDE_TEMPLATE_POOL
    template = dict(rng.choice(template_pool))
    grid = np.array(template["grid"], dtype=int).copy()
    controls = _choose_controls(template, rng)

    for r, c, color in controls:
        if int(grid[r, c]) != int(BG):
            return None
        grid[r, c] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    decoded_water = sum(int(CONTROL_TO_AMOUNT[color]) for _r, _c, color in controls)
    if int(np.sum(out == WATER)) != int(decoded_water):
        return None

    visible_baseline = _simulate_visible_water(grid, check_input=False, check_output=False)
    if np.array_equal(visible_baseline, out):
        return None

    if bool(template["guide"]):
        if not (3 <= decoded_water <= 10):
            return None
    else:
        if not (10 <= decoded_water <= 21):
            return None
        want_multibasin = bool(rng.random() < 0.6)
        if want_multibasin and _water_component_count(out) < 2:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    want_guide = bool(rng.random() < 0.4)
    for _ in range(400):
        attempt = _gen_attempt(rng, want_guide=want_guide)
        if attempt is not None:
            return attempt
    raise RuntimeError("failed to generate a valid D5 control-pixel instance")
