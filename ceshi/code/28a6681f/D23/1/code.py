# Variant of 28a6681f
# Dimension: D23 — 反弹/弹跳机制 (path)
# Change: 无反弹——水碰到阻挡后尝试对角线或水平滑动，最终停在支撑位置... → 水碰到斜面后改变方向

from __future__ import annotations

import random

import numpy as np

# Design analysis
# ---
# variant_spec:
#   core_change:
#     - 仅修改“被支撑后的路径决策”。
#     - 旧规则里，支撑后只看开放对角和最近落点；新规则里，若脚下是单向阶梯壁，则先按阶梯下坡方向滚动。
#   a_concretization_and_trace:
#     - 可观察对象是两个底部井槽 + 一个边缘导水阶梯。上方 2x2 水源块靠边放置，底部井槽里可带少量已稳定积水。
#     - 关键局部发生在 row=5 的宽平台。平台下方既能通向左井，也能通向右井；旧逻辑会在“最近落点”上产生歧义，
#       新逻辑则读取脚下楼梯的朝向并固定向右或向左滚。
#     - 具体例子：水到达 shelf cell (5,5) 时，下一行可落入的空列同时包含左井口和右井口；旧 solver 的
#       `nearest` 约束会报歧义，而新 solver 从 support cell (6,5) 看到更长的同色右下对角链，
#       因而把水推向右侧井槽。
#   b_complexity_calibration:
#     - 不把“斜面方向”做成多档离散规则，只允许 `left` / `right` 两个方向。
#     - 结构模板只保留两种，经镜像和 0-2 个已稳定积水扩展出多样性，避免样例全同又避免自由度过多。
#   c_hidden_assumptions_in_old_main:
#     - 旧版 `assert len(opts) == 1` 与 `assert len(nearest) == 1` 依赖“局部路径天然唯一”。
#       D23 破坏了这个前提，因为平台上可能同时看见左右两个落点，而斜面朝向才是判定依据。
#     - 旧版水平滑动只会朝最近落点迈一步；这在阶梯 tread 上会忽略“继续沿坡滚过一个受支撑台阶”的情形。
#   d_placement_legality:
#     - 模板坐标是手工验证过的 10x10 布局；guide、divider、井壁、源区、carryover 水互不重叠。
#     - 镜像只对整张网格做左右翻转，因此不会引入越界或连通性破坏。
#   e_visual_observability:
#     - 样例中总会出现“上方导水阶梯 + 中部阶梯支撑 + 两个井槽”的组合，人眼能看到水不是就近下落，
#       而是沿楼梯朝某一侧改道。
#     - 生成器强制 `slope_steps > 0`，并要求新旧逻辑输出不同或旧逻辑直接失败，防止退化回原题。
#   f_visual_cleanliness:
#     - 颜色仅三种结构色：井壁/底座、斜面 divider、上方 guide。水只用蓝色 1。
#     - carryover 水只放进单像素宽井槽内，不会和 guide/斜面重叠。
#   g_color_inference:
#     - 颜色不参与排序、方向或匹配；斜面方向完全由几何朝向决定。
#   new_main_assumptions_checked_against_generator:
#     - `_slope_direction` 假设脚下同色结构在左下/右下对角链长度不相等；模板保证这一点。
#     - 斜面滚动允许“先走受支撑同排一步，再继续下落”；模板里的 stair tread 正是按这个像素关系搭的。
#     - guide 仍需从边缘进入且 4-connected；模板里的 guide 全满足原题约束。


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


def _slope_direction(g: np.ndarray, rr: int, cc: int, bg: int, water: int) -> int:
    H, W = g.shape
    sr = rr + 1
    if sr >= H:
        return 0
    color = int(g[sr, cc])
    if color in (int(bg), int(water)):
        return 0

    def chain(dc: int) -> int:
        k = 0
        while True:
            r = int(sr + k)
            c = int(cc + dc * k)
            if not (0 <= r < H and 0 <= c < W):
                break
            if int(g[r, c]) != color:
                break
            k += 1
        return int(k)

    left = chain(-1)
    right = chain(1)
    if max(left, right) < 2 or left == right:
        return 0
    return 1 if right > left else -1


def _simulate(
    grid: np.ndarray,
    slope_mode: bool,
    check_input: bool,
    check_output: bool,
    track_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, int]]:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = 0
    water = 1
    metrics = {"slope_steps": 0, "diag_steps": 0, "horiz_steps": 0, "falls": 0}

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
        seen: set[tuple[int, int]] = set()
        for _ in range(H * W * 2):
            state = (int(rr), int(cc))
            if state in seen:
                raise AssertionError("water pixel looped on a slope")
            seen.add(state)

            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                metrics["falls"] += 1
                continue

            slope_dir = 0
            if slope_mode and rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                slope_dir = _slope_direction(g, rr, cc, bg, water)

            # Stair tread / ramp behavior: prefer downhill motion before the original rules.
            if slope_dir:
                nr = int(rr + 1)
                nc = int(cc + slope_dir)
                if 0 <= nc < W and nr < H and int(g[rr, nc]) == int(bg) and int(g[nr, nc]) == int(bg):
                    rr, cc = nr, nc
                    metrics["slope_steps"] += 1
                    continue
                if 0 <= nc < W and int(g[rr, nc]) == int(bg):
                    side_supported = rr + 1 < H and int(g[rr + 1, nc]) != int(bg)
                    lower_step = rr + 2 < H and int(g[rr + 2, nc]) != int(bg)
                    if side_supported or lower_step:
                        cc = nc
                        metrics["slope_steps"] += 1
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
                if len(opts) == 1:
                    rr, cc = opts[0]
                    metrics["diag_steps"] += 1
                    continue
                forced = (int(rr + 1), int(cc + slope_dir))
                if slope_dir and forced in opts:
                    rr, cc = forced
                    metrics["slope_steps"] += 1
                    continue
                raise AssertionError(f"ambiguous diagonal moves: {opts}")

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

                if slope_dir:
                    downhill = [int(x) for x in drops if (int(x) - cc) * slope_dir > 0]
                    if downhill:
                        target = min(downhill, key=lambda x: abs(int(x) - cc))
                        step = -1 if target < int(cc) else 1
                        nx = int(cc + step)
                        assert int(g[rr, nx]) == int(bg)
                        cc = nx
                        metrics["slope_steps"] += 1
                        continue

                min_dist = min(abs(int(x) - cc) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x) - cc) == int(min_dist)]
                if len(nearest) != 1:
                    raise AssertionError(f"ambiguous nearest drop columns: {nearest}")
                best = nearest[0]
                if best == int(cc):
                    break
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(bg)
                cc = nx
                metrics["horiz_steps"] += 1
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(water):
                continue
            g[r, c] = int(bg)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(water)
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

    if track_metrics:
        return g, metrics
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Settle blue water into bottom basins, letting staircase walls behave like directed slopes.

    Input structure:
    - Background is 0 and water is color 1.
    - Other colors form bottom-anchored wells plus an optional edge-entering stair guide.
    - Water starts in supported source blocks near an edge; some samples also keep a little already-stable basin water.

    Rule:
    - Water still falls straight down through background cells.
    - Once supported, it may use the old diagonal and nearest-drop sliding rules, but a monotone stair wall underneath can impose a downhill direction.
    - On such a stair, the pixel first tries to roll downhill along the slope, using a downhill diagonal when open or a supported same-row tread when needed.
    - Repeat the per-pixel settling passes until the whole water configuration stops moving.

    Key insight: The stepped wall geometry is now part of the path logic; on ambiguous shelves, the staircase orientation decides which basin the water reaches.
    """
    return _simulate(
        grid=np.array(grid, dtype=int),
        slope_mode=True,
        check_input=check_input,
        check_output=check_output,
        track_metrics=False,
    )


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _simulate(
        grid=np.array(grid, dtype=int),
        slope_mode=False,
        check_input=True,
        check_output=False,
        track_metrics=False,
    )


def _build_structure(template_id: int, colors: tuple[int, int, int]) -> np.ndarray:
    base_color, slope_color, guide_color = map(int, colors)
    g = np.zeros((10, 10), dtype=int)

    for c in (2, 3, 4, 6, 7, 8):
        g[9, c] = base_color
    for r in (6, 7, 8):
        g[r, 2] = base_color
        g[r, 4] = base_color
    for r in (5, 6, 7, 8):
        g[r, 8] = base_color

    if template_id == 0:
        slope_cells = [(6, 5), (7, 5), (7, 6), (8, 6)]
        guide_cells = [(2, 0), (2, 1), (3, 1), (3, 2), (4, 2), (4, 3), (4, 4)]
    elif template_id == 1:
        slope_cells = [(6, 5), (7, 5), (7, 6), (8, 6), (9, 6)]
        guide_cells = [(2, 0), (2, 1), (3, 1), (4, 1), (4, 2), (4, 3), (4, 4)]
    else:
        raise AssertionError(f"unknown template: {template_id}")

    for r, c in slope_cells:
        g[r, c] = slope_color
    for r, c in guide_cells:
        g[r, c] = guide_color
    return g


def _build_candidate(
    template_id: int,
    mirror: bool,
    carry_depth: int,
    colors: tuple[int, int, int],
) -> np.ndarray:
    g = _build_structure(template_id, colors)

    for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)):
        g[r, c] = 1

    if carry_depth >= 1:
        g[8, 3] = 1
    if carry_depth >= 2:
        g[7, 3] = 1

    if mirror:
        g = np.fliplr(g)
    return g


def generate_input() -> np.ndarray:
    """
    Generate a valid D23 variant input.

    The generator is constructive rather than reverse-simulating arbitrary targets:
    it samples one of a few pre-validated slope templates, randomizes colors/mirroring/carryover,
    and then accepts only candidates where:
    - the new solver uses slope motion,
    - the new solver produces a stable output,
    - the legacy non-slope logic gives a different result or fails.
    """
    combos = [(template_id, mirror, carry_depth) for template_id in (0, 1) for mirror in (False, True) for carry_depth in (0, 1, 2)]
    palette = list(range(2, 10))
    for _ in range(200):
        template_id, mirror, carry_depth = random.choice(combos)
        colors = tuple(random.sample(palette, 3))
        candidate = _build_candidate(template_id, mirror, carry_depth, colors)

        out, metrics = _simulate(
            candidate,
            slope_mode=True,
            check_input=True,
            check_output=True,
            track_metrics=True,
        )
        if metrics["slope_steps"] <= 0:
            continue

        same_as_legacy = False
        try:
            legacy_out = _legacy_main(candidate)
            same_as_legacy = bool(np.array_equal(legacy_out, out))
        except Exception:
            same_as_legacy = False
        if same_as_legacy:
            continue

        return candidate
    raise RuntimeError("Failed to generate a non-degenerate D23 sample")
