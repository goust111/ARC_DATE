# Combo variant of 28a6681f (index 20)
# combo_spec: D2/1+D18/1
# Name: 双侧选深+水分裂

from __future__ import annotations

import random

import numpy as np


# design_analysis:
#   concretization:
#     - 保留原题的 10x10、0 背景、1 水、其余颜色为静态障碍的视觉语法。
#     - 每个样例固定放两个“顶在柱子上的源水堆”：一个 fork 的左右下游深度不同，
#       另一个 fork 的左右下游完全等深。两处 fork 都是“下方被挡、左右对角都能走”的同类局面。
#     - 每个 fork 上放两层源水，确保同一规则在同一张图里重复发生，而不是一次性的偶然现象。
#   behavior_trace:
#     - 例：水在 (3,2) 且 (4,2) 为实心，(4,1)/(4,3) 都为空。
#       原题会在这里触发 `ambiguous diagonal moves`；D2/1 会前瞻左右终点深度并选更深侧；
#       本融合版进一步规定：若左路最深终点行号 > 右路，就只走左；若两侧最深终点行号相同，
#       则当前水复制成两个分支，分别从左右对角继续模拟。
#     - 在本模板里，不等深 fork 会稳定落到“深杯”里；等深 fork 会同时填左右对称杯。
#   complexity_guardrails:
#     - 只保留一个模板家族：一个不等深 fork + 一个等深 fork，再加整图镜像和颜色随机。
#     - 深浅差只做成 2 档（深杯到底部、浅杯抬高 2 格），这样肉眼容易看出“先比深度，再决定是否分裂”。
#     - 每个 fork 都放两层源水，因此输出会重复出现“连续选深”和“连续分裂”，证据足够但自由度不失控。
#   hidden_assumptions_in_original_main:
#     - 原 `assert len(opts) == 1` 依赖“不会出现双对角同时开放”，融合后必须替换成深度比较逻辑。
#     - 原 `settle` 只返回一个坐标；等深分裂后会有两个终点，因此必须改成返回终点集合。
#     - 原输入校验禁止水像素左右同时为空；本变体故意把源水直接放在可见 fork 顶上，这个旧约束需要移除。
#     - 原水平滑动仍假设最近落点唯一；融合描述只改双对角分支，本实现保留水平唯一性，并由生成器保证不出现水平并列落点。
#   placement_legality:
#     - 所有障碍都在 10x10 范围内、底部锚定且互不重叠；两个 source stack 只占背景格。
#     - 每个输入水像素都由下方固体或下方水支撑，满足“supported source”这一原题核心物理前提。
#     - 两个 fork 区域横向分区，避免分支互相挤压；镜像只做整图翻转，不改变局部物理。
#   observability:
#     - 解题者能直接看到：一个 fork 左右口袋深浅不同，输出只去更深侧；另一个 fork 两边完全对称，输出同时占两边。
#     - 镜像样例会交换“更深侧在左还是在右”，避免把规则误读成固定偏向某一边。
#     - 规则只依赖几何深度，不依赖颜色编号、处理顺序等不可见因素。
#   cleanliness:
#     - 画面只含两组简洁的柱子+杯状盆地，没有额外标记层；左右区域的功能清晰分离。
#     - 分裂后的输出直接落入细长杯状槽，稳定性和去向都容易检查，不会出现杂乱铺开的水面。
#   color_logic:
#     - 颜色只区分静态结构区域，不编码“深/浅/分裂”语义；同一行为在不同配色下都会重复出现。
#   generator_guarantees:
#     - 每个样例都强制验证：至少发生一次 `deeper_choice_events` 和一次 `equal_depth_splits`。
#     - 还会验证：分裂带来可见水量增加，且不等深 fork 的真实选择方向与模板设定一致。
#   new_solver_assumptions_checked_by_generator:
#     - 深度定义为“按融合规则继续模拟后，可到达的最深终点行号”。
#     - 若等深分支后续重新汇合，solver 会按终点并集去重；模板仍尽量让左右杯分开，避免早汇合掩盖现象。
#     - 由于未改水平 tie 规则，生成器只使用纯下落+对角杯，不制造多列等距水平落口。


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

    left = int(cc)
    while left - 1 >= 0 and int(g[rr, left - 1]) == int(bg):
        left -= 1
    right = int(cc)
    while right + 1 < W and int(g[rr, right + 1]) == int(bg):
        right += 1

    drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(bg)]
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


def _simulate(
    grid: np.ndarray,
    check_input: bool = True,
    check_output: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1

    stats = {
        "input_water": int(np.count_nonzero(g == int(water))),
        "output_water": 0,
        "water_gain": 0,
        "dual_diagonal_events": 0,
        "deeper_choice_events": 0,
        "equal_depth_splits": 0,
        "deeper_choice_left": 0,
        "deeper_choice_right": 0,
        "max_terminals_per_source": 1,
        "passes": 0,
    }

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {bg, water} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> set[tuple[int, int]]:
        depth_cache: dict[tuple[int, int], int] = {}
        terminal_cache: dict[tuple[int, int], set[tuple[int, int]]] = {}

        def future_depth(rr: int, cc: int) -> int:
            key = (int(rr), int(cc))
            if key in depth_cache:
                return int(depth_cache[key])

            cur_r, cur_c = key
            if cur_r + 1 < H and int(g[cur_r + 1, cur_c]) == int(bg):
                depth = future_depth(int(cur_r + 1), int(cur_c))
            else:
                opts = _diag_options(g, int(cur_r), int(cur_c), bg)
                if len(opts) == 2:
                    depth = max(future_depth(*opts[0]), future_depth(*opts[1]))
                elif len(opts) == 1:
                    depth = future_depth(*opts[0])
                else:
                    nx = _horizontal_step_target(g, int(cur_r), int(cur_c), bg)
                    if nx is None:
                        depth = int(cur_r)
                    else:
                        depth = future_depth(int(cur_r), int(nx))

            depth_cache[key] = int(depth)
            return int(depth)

        def terminal_set(rr: int, cc: int) -> set[tuple[int, int]]:
            key = (int(rr), int(cc))
            if key in terminal_cache:
                return set(terminal_cache[key])

            cur_r, cur_c = key
            if cur_r + 1 < H and int(g[cur_r + 1, cur_c]) == int(bg):
                terminals = terminal_set(int(cur_r + 1), int(cur_c))
            else:
                opts = _diag_options(g, int(cur_r), int(cur_c), bg)
                if len(opts) == 2:
                    left_depth = future_depth(*opts[0])
                    right_depth = future_depth(*opts[1])
                    if left_depth > right_depth:
                        terminals = terminal_set(*opts[0])
                    elif right_depth > left_depth:
                        terminals = terminal_set(*opts[1])
                    else:
                        terminals = terminal_set(*opts[0]) | terminal_set(*opts[1])
                elif len(opts) == 1:
                    terminals = terminal_set(*opts[0])
                else:
                    nx = _horizontal_step_target(g, int(cur_r), int(cur_c), bg)
                    if nx is None:
                        terminals = {(int(cur_r), int(cur_c))}
                    else:
                        terminals = terminal_set(int(cur_r), int(nx))

            terminal_cache[key] = set(terminals)
            return set(terminals)

        def walk(rr: int, cc: int, seen: set[tuple[int, int]]) -> set[tuple[int, int]]:
            key = (int(rr), int(cc))
            if key in seen:
                return set()
            seen.add(key)

            cur_r, cur_c = key
            if cur_r + 1 < H and int(g[cur_r + 1, cur_c]) == int(bg):
                return walk(int(cur_r + 1), int(cur_c), seen)

            opts = _diag_options(g, int(cur_r), int(cur_c), bg)
            if len(opts) == 2:
                left_depth = future_depth(*opts[0])
                right_depth = future_depth(*opts[1])
                stats["dual_diagonal_events"] += 1
                if left_depth > right_depth:
                    stats["deeper_choice_events"] += 1
                    stats["deeper_choice_left"] += 1
                    return walk(*opts[0], seen)
                if right_depth > left_depth:
                    stats["deeper_choice_events"] += 1
                    stats["deeper_choice_right"] += 1
                    return walk(*opts[1], seen)
                stats["equal_depth_splits"] += 1
                return walk(*opts[0], seen) | walk(*opts[1], seen)

            if len(opts) == 1:
                return walk(*opts[0], seen)

            nx = _horizontal_step_target(g, int(cur_r), int(cur_c), bg)
            if nx is not None:
                return walk(int(cur_r), int(nx), seen)

            return {(int(cur_r), int(cur_c))}

        terminals = walk(int(r), int(c), set())
        if not terminals:
            terminals = terminal_set(int(r), int(c))
        assert terminals, "water branch did not settle"
        stats["max_terminals_per_source"] = max(stats["max_terminals_per_source"], len(terminals))
        return terminals

    for step in range(3 * H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(water):
                continue
            g[int(r), int(c)] = int(bg)
            terminals = settle(int(r), int(c))
            moved = bool(moved or terminals != {(int(r), int(c))})
            for rr, cc in terminals:
                g[int(rr), int(cc)] = int(water)
        if not moved:
            stats["passes"] = int(step + 1)
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

    stats["output_water"] = int(np.count_nonzero(g == int(water)))
    stats["water_gain"] = int(stats["output_water"] - stats["input_water"])
    return g, stats


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate supported water stacks falling into bottom-anchored basins, with depth-aware diagonal branching.

    Input structure:
    - Background is 0 and water is color 1.
    - The other 2-3 colors are fixed basin walls, pillars, ledges, and optional edge-entering guides.
    - Water starts in supported cells, often perched on top of pillars or above connected basin cups.

    Rule:
    - Water always drops straight down through background first.
    - If downward motion is blocked and exactly one diagonal route is open through an open side cell, it slides there.
    - If both lower diagonals are open, compare the deepest reachable resting row on each side: choose the deeper side when one is lower, but split into both sides when the two downstream depths are equal.
    - Once supported on a shelf, water still slides toward the unique nearest drop and repeats the same priorities until every branch settles.

    Key insight: A two-sided diagonal fork is resolved by downstream geometry, not by local uniqueness alone. The solver must look ahead to compare basin depth, and only ties produce branching water.
    """
    return _simulate(grid, check_input=check_input, check_output=check_output)[0]


def _mirror_case(
    grid: np.ndarray,
    sources: list[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    mirrored = np.fliplr(np.array(grid, dtype=int))
    width = mirrored.shape[1]
    mirrored_sources = [(int(r), int(width - 1 - c)) for r, c in sources]
    return mirrored, mirrored_sources


def _build_structure(
    deep_left: bool,
    choice_color: int,
    shallow_color: int,
    split_color: int,
) -> tuple[np.ndarray, list[tuple[int, int]], bool]:
    g = np.zeros((10, 10), dtype=int)

    # Asymmetric chooser fork around column 2.
    g[4:, 2] = int(choice_color)
    if deep_left:
        g[6:, 0] = int(choice_color)
        g[9, 1] = int(choice_color)
        g[5:, 4] = int(shallow_color)
        g[7, 3] = int(shallow_color)
        choice_moves_left = True
    else:
        g[6:, 4] = int(choice_color)
        g[9, 3] = int(choice_color)
        g[5:, 0] = int(shallow_color)
        g[7, 1] = int(shallow_color)
        choice_moves_left = False

    # Symmetric equal-depth splitter around column 7.
    g[4:, 7] = int(split_color)
    g[6:, 5] = int(split_color)
    g[6:, 9] = int(split_color)
    g[9, 5:10] = int(split_color)

    sources = [(3, 2), (2, 2), (3, 7), (2, 7)]
    return g, sources, bool(choice_moves_left)


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    choice_color, shallow_color, split_color = random.sample([2, 3, 4, 5, 6, 7, 8, 9], 3)
    deep_left = bool(random.getrandbits(1))
    mirrored = bool(random.getrandbits(1))

    structure, sources, choice_moves_left = _build_structure(
        deep_left=bool(deep_left),
        choice_color=int(choice_color),
        shallow_color=int(shallow_color),
        split_color=int(split_color),
    )

    if mirrored:
        structure, sources = _mirror_case(structure, sources)
        choice_moves_left = not bool(choice_moves_left)

    expected = np.array(structure, dtype=int).copy()
    for r, c in sources:
        assert int(expected[int(r), int(c)]) == 0
        expected[int(r), int(c)] = 1
        expected = _simulate(expected, check_input=False, check_output=False)[0]

    inp = np.array(structure, dtype=int).copy()
    for r, c in sources:
        assert int(inp[int(r), int(c)]) == 0
        assert int(inp[int(r) + 1, int(c)]) != 0 or any(
            (int(rr), int(cc)) == (int(r) + 1, int(c)) for rr, cc in sources
        )
        inp[int(r), int(c)] = 1

    solved, stats = _simulate(inp, check_input=True, check_output=True)
    assert np.array_equal(solved, expected), "simultaneous input must match sequential pours"
    assert stats["deeper_choice_events"] > 0, "chooser fork did not trigger"
    assert stats["equal_depth_splits"] > 0, "equal-depth split did not trigger"
    assert stats["water_gain"] >= 2, "splitter must increase visible water count"
    assert stats["max_terminals_per_source"] == 2, "split branch count should stay binary"
    if choice_moves_left:
        assert stats["deeper_choice_left"] > 0 and stats["deeper_choice_right"] == 0
    else:
        assert stats["deeper_choice_right"] > 0 and stats["deeper_choice_left"] == 0

    info = dict(stats)
    info["choice_moves_left"] = int(bool(choice_moves_left))
    return inp, solved, info


def generate_input() -> np.ndarray:
    for _ in range(200):
        try:
            inp, _out, _info = _build_case()
            return inp
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid D2/1+D18/1 sample")
