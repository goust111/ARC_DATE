# Variant of 28a6681f
# Dimension: D2 — 对角线滑动条件 (path)
# Change: 对角线移动要求侧向格和对角格都为背景——即侧面必须开放才能斜向通过... → 双侧开放时选择较深的一侧

from __future__ import annotations

import random

import numpy as np


# design_analysis:
#   concretization:
#     - 保留 10x10、0 为背景、1 为水、其余颜色为静态障碍的原题视觉语法。
#     - 变体样本都包含一个“中央岔点”：水先被送到一个下方被柱子顶住的位置，
#       然后左下和右下两个对角格都可通过。
#     - 左右两侧不再做成等价结构；一侧通向更深的盆地，另一侧只通向较浅的口袋。
#   behavior_trace:
#     - 原规则在双侧同时可走时会报 ambiguous diagonal moves，因此原生成器会避开这种局面。
#     - 现在刻意制造这个局面：例如水位于 (3,4)，可去 (4,3) 或 (4,5)。
#       左路最终到第 9 行、右路最终只到第 6/7/8 行时，新规则必须选左。
#     - 这要求 solver 先分别预测两条路的最终落点，再决定当前一步，而不是看局部唯一性。
#   complexity_guardrails:
#     - 只保留两种模板：左侧更深、右侧更深。这样人类能从少量样例看出“更深侧优先”，
#       同时避免出现过多互相竞争的自由度。
#     - 每个模板固定使用 5 个水像素，让同一个岔点被重复触发 3 次；这是经过验证后
#       同时满足“效果明显”和“终态仍满足原题稳定性断言”的稳定档位。
#   hidden_assumptions_in_original_main:
#     - 原 `assert len(opts) == 1` 依赖“不会出现双侧同时开放”的旧属性，变体中不再成立。
#     - 原水平滑动仍假设最近可落下列唯一；模板继续保证这一点，不改该部分规则。
#     - 原输入校验禁止初始水像素左右同时是背景；生成器继续遵守，只让双侧开放出现在运动过程中。
#   placement_legality:
#     - 所有结构都手工放在 10x10 范围内，且源水只放在被下方固体或其他水支撑的位置。
#     - 模板中的右侧“支撑水”会先滑走，随后中央水才到达双侧开放点，因此既合法又能触发变体。
#   observability:
#     - 两个模板分别让左盆地或右盆地更深，并随机切换，避免解题者误以为只是固定向某侧流。
#     - 变体判断只依赖可见的盆地深浅，不依赖颜色 id、隐藏计数或不可见排序。
#   cleanliness:
#     - 每个样本只叠加三类元素：底部盆地、左上导流件、源水列；区域分工明确，不会互相遮挡。
#   color_logic:
#     - 颜色只区分结构部件，不参与控制流排序；guide 颜色始终与底部颜色不同以保持上方部件可辨。
#   generator_guarantees:
#     - 生成后立即运行新 solver，并记录所有真实发生的双侧对角选择。
#     - 只有当至少发生一次选择、且每次选择都确实偏向预设更深的一侧时，样本才会被接受。


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


def _diag_options(g: np.ndarray, rr: int, cc: int, bg: int) -> list[tuple[int, int]]:
    H, W = g.shape
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
    return opts


def _horizontal_step_target(g: np.ndarray, rr: int, cc: int, bg: int) -> int | None:
    H, W = g.shape
    if rr + 1 >= H or int(g[rr + 1, cc]) == int(bg):
        return None

    L = int(cc)
    while L - 1 >= 0 and int(g[rr, L - 1]) == int(bg):
        L -= 1
    R = int(cc)
    while R + 1 < W and int(g[rr, R + 1]) == int(bg):
        R += 1

    drops = [int(x) for x in range(int(L), int(R) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
    if not drops:
        return None

    min_dist = min(abs(int(x - cc)) for x in drops)
    nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
    assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
    best = nearest[0]
    assert best != int(cc)
    step = -1 if best < int(cc) else 1
    nx = int(cc + step)
    assert int(g[rr, nx]) == int(bg)
    return int(nx)


def _future_rest(
    g: np.ndarray,
    r: int,
    c: int,
    bg: int,
    cache: dict[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    key = (int(r), int(c))
    if key in cache:
        return cache[key]

    H, W = g.shape
    rr, cc = key
    for _ in range(H * W):
        if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
            rr += 1
            continue

        opts = _diag_options(g, rr, cc, bg)
        if len(opts) == 2:
            left_final = _future_rest(g, *opts[0], bg, cache)
            right_final = _future_rest(g, *opts[1], bg, cache)
            assert left_final[0] != right_final[0], (
                f"equal-depth diagonal branches at {(rr, cc)}: "
                f"{opts[0]}->{left_final}, {opts[1]}->{right_final}"
            )
            rr, cc = opts[0] if left_final[0] > right_final[0] else opts[1]
            continue

        if len(opts) == 1:
            rr, cc = opts[0]
            continue

        nx = _horizontal_step_target(g, rr, cc, bg)
        if nx is not None:
            cc = int(nx)
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")

    cache[key] = (int(rr), int(cc))
    return cache[key]


def _settle(
    g: np.ndarray,
    r: int,
    c: int,
    bg: int,
    decisions: list[dict[str, object]] | None,
) -> tuple[int, int]:
    H, W = g.shape
    rr = int(r)
    cc = int(c)
    cache: dict[tuple[int, int], tuple[int, int]] = {}

    for _ in range(H * W):
        if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
            rr += 1
            continue

        opts = _diag_options(g, rr, cc, bg)
        if len(opts) == 2:
            left_final = _future_rest(g, *opts[0], bg, cache)
            right_final = _future_rest(g, *opts[1], bg, cache)
            assert left_final[0] != right_final[0], (
                f"equal-depth diagonal branches at {(rr, cc)}: "
                f"{opts[0]}->{left_final}, {opts[1]}->{right_final}"
            )
            chosen = opts[0] if left_final[0] > right_final[0] else opts[1]
            if decisions is not None:
                decisions.append(
                    {
                        "at": (int(rr), int(cc)),
                        "left_option": opts[0],
                        "right_option": opts[1],
                        "left_final": left_final,
                        "right_final": right_final,
                        "chosen": chosen,
                        "depth_gap": abs(int(left_final[0] - right_final[0])),
                    }
                )
            rr, cc = chosen
            continue

        if len(opts) == 1:
            rr, cc = opts[0]
            continue

        nx = _horizontal_step_target(g, rr, cc, bg)
        if nx is not None:
            cc = int(nx)
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")

    return int(rr), int(cc)


def _simulate(
    grid: np.ndarray,
    check_input: bool,
    check_output: bool,
    record_decisions: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1
    decisions: list[dict[str, object]] = []

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

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(water):
                continue
            g[int(r), int(c)] = int(bg)
            rr, cc = _settle(g, int(r), int(c), bg, decisions if record_decisions else None)
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

    return g, decisions


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling into linked basins, with lookahead at diagonal forks.

    Input structure:
    - bg = 0 and water = 1.
    - Other colors are rigid basin walls, shelves, or optional upper guides.
    - Water begins in supported source stacks above or beside the basin network.

    Rule:
    - Each water pixel first falls straight down through background cells.
    - If blocked below and exactly one diagonal route is open through an open side cell, it slides there.
    - If both diagonal routes are open, compare where each branch would ultimately come to rest and choose the deeper branch.
    - While supported on a shelf, slide toward the unique nearest drop and keep repeating until nothing moves.

    Key insight: The decisive move is no longer purely local; at a two-sided diagonal fork, water commits to the side whose downstream path ends lower.
    """
    return _simulate(grid, check_input=check_input, check_output=check_output)[0]


def _add_source_water(g: np.ndarray, extra_stack: int) -> None:
    for r, c in [(4, 5), (3, 5), (3, 4)]:
        g[r, c] = 1
    for r in range(3 - int(extra_stack), 3):
        g[int(r), 4] = 1


def _build_template(
    deep_left: bool,
    left_color: int,
    right_color: int,
    guide_color: int,
    extra_stack: int,
) -> np.ndarray:
    g = np.zeros((10, 10), dtype=int)

    # Upper guide that visibly enters from the left edge and supports the source column.
    g[0, 0:4] = int(guide_color)
    g[0:3, 3] = int(guide_color)

    if deep_left:
        # Left side is the deep basin; right side is a raised pocket.
        for r in range(5, 10):
            g[r, 0] = int(left_color)
        for r in range(4, 10):
            g[r, 4] = int(left_color)
        for r in range(6, 10):
            g[r, 9] = int(right_color)
            g[r, 6] = int(right_color)
        g[8, 7:9] = int(right_color)
        g[5, 5:7] = int(right_color)
        g[9, 7:10] = int(right_color)
    else:
        # Left side is the raised pocket; right side is the deeper channel.
        for r in range(5, 10):
            g[r, 0] = int(left_color)
        g[7, 1:4] = int(left_color)
        g[8, 0:4] = int(left_color)
        g[9, 0:4] = int(left_color)
        for r in range(4, 10):
            g[r, 4] = int(right_color)
        g[5, 5] = int(right_color)
        g[9, 5] = int(right_color)
        g[9, 7] = int(right_color)
        for r in range(7, 10):
            g[r, 7] = int(right_color)
        for r in range(6, 10):
            g[r, 9] = int(right_color)

    _add_source_water(g, extra_stack=extra_stack)
    return g


def generate_input() -> np.ndarray:
    for _ in range(64):
        left_color, right_color, guide_color = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)
        deep_left = bool(random.getrandbits(1))
        extra_stack = 2
        g = _build_template(
            deep_left=deep_left,
            left_color=int(left_color),
            right_color=int(right_color),
            guide_color=int(guide_color),
            extra_stack=int(extra_stack),
        )

        out, decisions = _simulate(g, check_input=True, check_output=True, record_decisions=True)
        if np.array_equal(out, g):
            continue
        if not decisions:
            continue

        if deep_left:
            if not all(int(d["left_final"][0]) > int(d["right_final"][0]) for d in decisions):
                continue
            if not all(int(d["chosen"][1]) < int(d["at"][1]) for d in decisions):
                continue
        else:
            if not all(int(d["right_final"][0]) > int(d["left_final"][0]) for d in decisions):
                continue
            if not all(int(d["chosen"][1]) > int(d["at"][1]) for d in decisions):
                continue

        if min(int(d["depth_gap"]) for d in decisions) < 1:
            continue

        return g

    raise AssertionError("failed to generate a non-degenerate D2 sample")
