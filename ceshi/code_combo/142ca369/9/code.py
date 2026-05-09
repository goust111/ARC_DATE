# Combo variant of 142ca369 (index 9)
# combo_spec: D3/0+D4/2
# Name: 周期图案轨迹+颜色循环列表

from __future__ import annotations

import random as _rng
import sys

import numpy as np

try:
    from arc_utils import colors_present, components_mask, get_bg
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import colors_present, components_mask, get_bg


"""
design_analysis:
  fusion_summary:
    keep:
      - 原题的几何对象不变：单色 3 像素 L 形箭头、单色墙、45 度对角线射线、侧碰反射。
      - 反射几何不变：corner hit 反转两个轴；side-touch 只翻转被碰到的轴。
    change:
      - 每个箭头背后增加一条固定水平 1x3 提示条，left->right 给出该箭头的颜色循环列表。
      - 射线颜色不再继承墙色；每次反射后改用提示条中的下一个颜色，超出后循环。
      - 每一段轨迹都不是纯色，而是用长度 4 的 AABB 周期着色：
        A = 箭头自身颜色，B = 当前反射阶段的活动颜色。
      - 每次反射时同时发生两件事：
        1. 颜色循环前进一步，得到新的 B。
        2. AABB 相位重置，下一段重新从 A,A,B,B 开始。

  a_concretization_and_trace:
    objects:
      - 箭头仍是 2x2 bbox 缺一角的 L 形，缺角决定发射方向，直角顶点是起点。
      - 提示条固定放在箭头“后侧”，始终是水平 3 格，所以视觉上像局部图例而不是墙。
      - 墙仍只负责几何反射，不再提供颜色语义。
    small_trace:
      - 设箭头色 A=2，提示条是 [4, 8, 3]。
      - 发射前 B=A=2，所以第一段即使按 AABB 绘制，看起来也是 2,2,2,2...。
      - 第一次反射后，原题会把接触点和后续整段都改成墙色；融合后改成提示色 4。
      - 此时第二段颜色变成 2,2,4,4,2,2,4,4...，接触点也改成 4。
      - 第二次反射后 B=8，第三段变成 2,2,8,8...；第三次反射后 B=3，第四段变成 2,2,3,3...。
      - 差异不是“两条规则并排存在”，而是协同的：颜色循环决定 B，周期图案决定 A/B 如何在每段中交替出现。

  b_complexity_calibration:
    choices:
      - 提示条长度固定为 3，避免颜色顺序自由度过高，同时足以展示顺序和回绕。
      - 轨迹周期固定为 AABB，而不是任意模式；这样人类可以把规律归纳为“箭头色两格、当前循环色两格”。
      - 生成器偏向 2-4 次反射，并要求至少一个反射后的段长 >= 4，从而同时暴露 D3 和 D4 两个变化。
    anti_simplification:
      - 若允许 0 bounce 或仅 1-2 格的反射后段，AABB 与纯色段几乎不可区分。
      - 若提示条太长或颜色可重复，样例数不足时很难推断顺序。

  c_implicit_assumption_audit:
    original_lines_broken:
      - `wall_cells = non-bg minus arrow_cells` 原本默认除箭头外都是墙；加入提示条后必须把 hint cells 排除。
      - `out[nr, nc] = cur_col` 原本安全，因为整段单色；融合后必须显式计算 AABB 相位。
      - `cur_col = wcol` 和 bounce 点改涂墙色都不再成立；bounce 后颜色来自提示条循环。
      - 原 solver 的 seen-state 不含相位，也不含颜色循环位置；融合后这两个量都影响未来输出。
    new_conflict_check:
      - 提示条是多色 1x3，不能被误识别成箭头，也不能被当作墙参与反射。
      - 周期相位和颜色循环长度彼此独立，所以不能把二者混成一个计数器；必须分别维护 `phase` 与 `bounce_idx`。

  d_placement_legality:
    generator_guarantees:
      - 先放箭头和提示条，再规划对角线路径和墙。
      - 所有对象都走 `occupied` 检查，防止重叠。
      - 每个非最终段都预留至少 1 格边界余量，以确保反射墙可以合法放在射线侧边。
      - 每次候选都用 `main(grid)` 做 oracle 校验。

  e_visual_observability:
    why_humans_can_see_it:
      - 提示条紧贴箭头，颜色来源是局部且可配对的。
      - 墙颜色刻意与当次提示色不同，所以“墙只管反射、颜色来自提示条”是肉眼可见的。
      - 反射后段长被拉长到至少 4 格，AABB 条纹会稳定出现，而不是只闪现一次。

  f_visual_cleanliness:
    measures:
      - 提示条固定在箭头后侧，不进入主要轨迹区域。
      - 反射墙继续保持单色；提示条固定多色；箭头仍单色，三类对象角色分离。
      - 箭头数量限制在 1-3，避免 18-20 的画布过于拥挤。

  g_change_inferability:
    structured_color_rule:
      - A 永远是箭头色，来源固定。
      - B 永远是提示条按顺序取出的当前颜色，来源固定。
      - 生成器要求融合输出同时不同于：
        1. 连续单色但仍用提示条循环的版本
        2. AABB 周期但仍用墙色继承的版本
      - 这样可以直接排除退化成单一维度变体的样本。

  h_behavior_metric:
    sample_metrics:
      - `d3_diff_cells`: 融合输出与“仅 D4、无周期图案”输出的像素差异数。
      - `d4_diff_cells`: 融合输出与“仅 D3、仍按墙色继承”输出的像素差异数。
      - `used_cycle_indices`: 每个样本实际使用到的提示条索引集合。
    intent:
      - 要求大多数样本同时有 `d3_diff_cells > 0` 和 `d4_diff_cells > 0`。
      - 若经常只使用提示条第一个颜色，或 AABB 差异为 0，就说明生成器退化。

  i_main_generator_consistency_review:
    solver_assumptions_and_generator_support:
      - solver 假设每个真箭头都能唯一反推出一条完整提示条；generator 保证提示条全在网格内且不重叠。
      - solver 把提示条视为 blocker 而非墙；generator 保证路径不穿过提示条。
      - solver 允许 corner reflection，但 generator 只主动构造 side-touch 反射，避免歧义。
      - solver 的未来状态依赖 `(dir, cur_col, phase, bounce_idx mod 3)`；generator 在规划输出时用同样的状态更新。
"""


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None
    mr, mc = miss[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _hint_positions(pts: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    hint_r = r0 - 1 if int(dr) == 1 else r1 + 1
    hint_cs = [c0 - 1, c0, c1] if int(dc) == 1 else [c0, c1, c1 + 1]
    return [(int(hint_r), int(c)) for c in hint_cs]


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < H and 0 <= int(c) < W


def _periodic_color(start_col: int, cur_col: int, phase: int) -> int:
    return int(start_col if (phase % 4) < 2 else cur_col)


def _detect_arrows_and_cycles(g: np.ndarray, bg: int):
    H, W = g.shape
    arrow_cells: set[tuple[int, int]] = set()
    raw_arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                raw_arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    hint_cells: set[tuple[int, int]] = set()
    arrows = []
    for start_col, pts in raw_arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), start = info
        cycle_cells = _hint_positions(pts, dr, dc)
        cycle: list[int] = []
        ok = True
        for r, c in cycle_cells:
            if not _in_bounds(H, W, r, c):
                ok = False
                break
            if (int(r), int(c)) in arrow_cells or int(g[r, c]) == int(bg):
                ok = False
                break
            cycle.append(int(g[r, c]))
        if ok:
            hint_cells.update((int(r), int(c)) for r, c in cycle_cells)
        else:
            cycle = [int(start_col)]
            cycle_cells = []
        arrows.append(
            {
                "start_col": int(start_col),
                "pts": pts,
                "start": (int(start[0]), int(start[1])),
                "dr": int(dr),
                "dc": int(dc),
                "cycle": cycle,
                "cycle_cells": cycle_cells,
            }
        )
    return arrows, arrow_cells, hint_cells


def _trace(grid: np.ndarray, *, use_periodic: bool, use_cycle: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells, hint_cells = _detect_arrows_and_cycles(g, bg)
    blocker_cells = set(arrow_cells) | set(hint_cells)
    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in blocker_cells
    }

    out = g.copy()

    def next_cycle_color(cycle: list[int], bounce_idx: int) -> tuple[int, int]:
        return int(cycle[int(bounce_idx) % len(cycle)]), int(bounce_idx + 1)

    for arrow in arrows:
        cur = arrow["start"]
        dr = int(arrow["dr"])
        dc = int(arrow["dc"])
        start_col = int(arrow["start_col"])
        cur_col = int(start_col)
        cycle = list(map(int, arrow["cycle"]))
        bounce_idx = 0
        phase = 0
        seen = set()

        while True:
            cycle_state = int(bounce_idx % len(cycle)) if use_cycle else 0
            state = (cur, int(dr), int(dc), int(cur_col), int(phase % 4), cycle_state)
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not _in_bounds(H, W, nr, nc):
                break

            if (nr, nc) in wall_cells:
                if use_cycle:
                    new_col, bounce_idx = next_cycle_color(cycle, bounce_idx)
                else:
                    new_col = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(new_col)
                cur_col = int(new_col)
                dr, dc = -int(dr), -int(dc)
                phase = 0
                continue

            if (nr, nc) in blocker_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                if use_periodic:
                    out[nr, nc] = _periodic_color(int(start_col), int(cur_col), int(phase))
                else:
                    out[nr, nc] = int(cur_col)
            phase = (int(phase) + 1) % 4

            bounced = False
            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                if use_cycle:
                    new_col, bounce_idx = next_cycle_color(cycle, bounce_idx)
                else:
                    new_col = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                dr = -int(dr)
                bounced = True
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                if use_cycle:
                    new_col, bounce_idx = next_cycle_color(cycle, bounce_idx)
                else:
                    new_col = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                dr = -int(dr)
                bounced = True

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                if use_cycle:
                    new_col, bounce_idx = next_cycle_color(cycle, bounce_idx)
                else:
                    new_col = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                dc = -int(dc)
                bounced = True
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                if use_cycle:
                    new_col, bounce_idx = next_cycle_color(cycle, bounce_idx)
                else:
                    new_col = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                dc = -int(dc)
                bounced = True

            if bounced:
                phase = 0

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule for 142ca369.

    - Geometry follows the original diagonal-ray reflection task.
    - A local 3-color hint strip behind each arrow defines the bounce color cycle.
    - Each launch/bounce starts a fresh AABB trajectory pattern:
      A = arrow color, B = current cycle color.
    """
    return _trace(np.array(grid, dtype=int), use_periodic=True, use_cycle=True)


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 3)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = [int(c) for c in palette[:n_arrows]]
    non_arrow_palette = [int(c) for c in palette[n_arrows:]]
    if len(non_arrow_palette) < 5:
        return None

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    total_bounces = 0
    used_cycle_indices: set[int] = set()
    max_post_bounce_len = 0

    for arrow_color in arrow_colors:
        cycle = [int(c) for c in _rng.sample(non_arrow_palette, 3)]
        wall_palette = [int(c) for c in non_arrow_palette if c not in set(cycle)]
        if not wall_palette:
            return None
        n_bounces = _rng.choices([1, 2, 3, 4], weights=[1, 4, 4, 2])[0]

        placed = False
        for _ in range(180):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            sr = _rng.randint(0, H - 1)
            sc = _rng.randint(0, W - 1)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not _in_bounds(H, W, r, c) for r, c in arrow_pts):
                continue

            hint_pts = _hint_positions(arrow_pts, dr, dc)
            if any(not _in_bounds(H, W, r, c) for r, c in hint_pts):
                continue

            shape_cells = {(int(r), int(c)) for r, c in arrow_pts + hint_pts}
            if shape_cells & occupied:
                continue

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc = int(dr), int(dc)
            cur_col = int(arrow_color)
            bounce_idx = 0
            phase = 0
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            segment_lengths: list[int] = []
            used_here: set[int] = set()
            ok = True

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                is_final = seg_idx == n_bounces
                if is_final:
                    if max_steps < 4:
                        ok = False
                        break
                    seg_len = _rng.randint(4, max_steps)
                else:
                    min_len = 2 if seg_idx == 0 else 4
                    hi = min(8, max_steps - 1)
                    if hi < min_len:
                        ok = False
                        break
                    seg_len = _rng.randint(min_len, hi)

                moved = 0
                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not _in_bounds(H, W, nr, nc):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in shape_cells or (nr, nc) in ray_set:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), _periodic_color(int(arrow_color), int(cur_col), int(phase))))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                    phase = (int(phase) + 1) % 4
                    moved += 1
                if not ok:
                    break
                segment_lengths.append(int(moved))

                if is_final:
                    continue

                options: list[tuple[list[tuple[int, int]], str]] = []
                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                    if all(_in_bounds(H, W, r, c) for r, c in w3):
                        options.append((w3, "dr"))
                    options.append(([(wr, wc)], "dr"))
                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                    if all(_in_bounds(H, W, r, c) for r, c in w3v):
                        options.append((w3v, "dc"))
                    options.append(([(wr2, wc2)], "dc"))
                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                options.sort(key=lambda item: len(item[0]), reverse=_rng.random() < 0.6)
                chosen_wall = None
                for wpxs, flip in options:
                    wset = {(int(r), int(c)) for r, c in wpxs}
                    if wset & occupied or wset & shape_cells or wset & ray_set:
                        continue
                    chosen_wall = ([(int(r), int(c)) for r, c in wpxs], flip)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip = chosen_wall
                cycle_idx = int(bounce_idx % len(cycle))
                next_col = int(cycle[cycle_idx])
                wall_choices = [int(c) for c in wall_palette if int(c) != int(next_col)]
                if not wall_choices:
                    ok = False
                    break
                wcol = int(_rng.choice(wall_choices))
                walls.append((wpxs, int(wcol)))

                lr, lc, _old = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(next_col))
                cur_col = int(next_col)
                used_here.add(cycle_idx)
                bounce_idx += 1
                phase = 0
                if flip == "dr":
                    cur_dr = -int(cur_dr)
                else:
                    cur_dc = -int(cur_dc)

            if not ok or len(ray_cells) < 8:
                continue
            if len(segment_lengths) != n_bounces + 1:
                continue
            if max(segment_lengths[1:], default=0) < 4:
                continue
            if sum(segment_lengths[1:]) < 6:
                continue

            new_cells = set(shape_cells) | set(ray_set)
            for wpxs, _wcol in walls:
                new_cells.update((int(r), int(c)) for r, c in wpxs)
            expected_count = len(shape_cells) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(arrow_color)
            for (r, c), hint_col in zip(hint_pts, cycle):
                grid[r, c] = out[r, c] = int(hint_col)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)

            occupied |= new_cells
            total_bounces += int(n_bounces)
            used_cycle_indices |= used_here
            max_post_bounce_len = max(int(max_post_bounce_len), int(max(segment_lengths[1:], default=0)))
            placed = True
            break

        if not placed:
            return None

    solver_out = main(grid)
    if not np.array_equal(solver_out, out):
        return None

    d3_only_out = _trace(grid, use_periodic=False, use_cycle=True)
    d4_only_out = _trace(grid, use_periodic=True, use_cycle=False)
    d3_diff_cells = int(np.count_nonzero(solver_out != d3_only_out))
    d4_diff_cells = int(np.count_nonzero(solver_out != d4_only_out))
    if d3_diff_cells <= 0 or d4_diff_cells <= 0:
        return None
    if total_bounces < 2:
        return None
    if max_post_bounce_len < 4:
        return None
    if not used_cycle_indices:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
