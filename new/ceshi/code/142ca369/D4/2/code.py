# Variant of 142ca369
# Dimension: D4 — 反射时颜色继承规则 (trajectory)
# Change: color_change_on_bounce——射线在每次反射时采用墙壁的颜色，反射点也被重新涂为墙壁颜色... → 颜色循环列表

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
  variant_summary:
    keep:
      - 原题的对象类型不变：L 形箭头、单像素/3像素直线墙、对角线射线、遇墙反射。
      - 反射几何不变：side-touch 仍只翻转一个轴；solver 仍保留 corner reflection 分支。
    change:
      - 墙壁不再决定射线颜色。
      - 每个箭头自带一个局部 3 色循环提示条；第 1/2/3/... 次反射后分别采用提示条第 1/2/3/... 个颜色，超出后循环。
      - 反射点仍会被改涂，但改成循环色，而不是墙色。

  a_concretization_and_trace:
    input_objects:
      - 箭头仍是单色 3 像素 L 形，2x2 bbox 缺一角。
      - 每个箭头背后紧贴放一条 1x3 的多色提示条，顺序按 left->right 读取。
      - 墙仍是单色 1 像素或 3 像素轴对齐线段。
    local_hint_geometry:
      - 若箭头沿 dr=1 发射，则提示条放在箭头 bbox 上方一行；dr=-1 时放在下方一行。
      - 若箭头沿 dc=1 发射，则提示条比 bbox 向左多伸 1 格；dc=-1 时向右多伸 1 格。
      - 这样提示条总在箭头“后侧”，且固定为水平 3 格，视觉上是局部图例而不是墙。
    trace_example:
      - 原题：箭头色 2 发射，碰到颜色 5 的墙后，反射点改成 5，后续段也用 5；再碰到颜色 7 的墙后改成 7。
      - 变体：同样输入若箭头背后提示条是 [4,8,3]，则第一次反射后反射点改成 4、后续段也用 4；第二次反射改成 8；第三次反射改成 3；墙色只影响几何，不影响着色。
      - 差异落点：原 solver 在每个 bounce 分支里读取 wcol 并赋给 cur_col；变体要把这里改成读取箭头提示循环。

  b_complexity_calibration:
    decisions:
      - 循环长度固定为 3，避免自由度过高，同时足以展示“顺序”和“循环回绕”。
      - 每个提示条使用 3 个互不相同的颜色，提升顺序可见性。
      - generator 让每个箭头至少发生 1 次反射，并偏向 2-4 次，使 D4 变化能稳定出现。
    why_not_simpler:
      - 若只用 1 色提示，就退化为“固定替换色”，看不出 cycle。
      - 若仍允许大量 0 bounce 样例，则新旧规则在视觉上经常无差别。

  c_implicit_assumption_audit:
    original_sensitive_lines:
      - `wall_cells = non-bg minus arrow_cells`：原题默认除箭头外都是墙；变体新增提示条后这个假设失效，必须把 hint cells 排除。
      - `cur_col = wcol` in all bounce branches：原题把墙色当成颜色来源；变体中这必须改成 `cur_col = cycle[next_bounce_idx]`。
      - `out[...] = wcol` at bounce points：原题无害，因为目标色和墙色相同；变体里若不改，会把正确的循环色覆盖掉。
      - loop state里包含 `cur_col`：变体颜色由 bounce 次数决定，所以状态还要包含 cycle 位置，否则可能错误截断循环轨迹。
    new_assumptions:
      - 每个箭头都能在其后侧找到完整 3 格提示条。
      - 提示条不与墙/箭头/射线路径重叠。
      - 墙仍是单色段，因此多色 1x3 提示条不会和墙的视觉语义混淆。

  d_placement_legality:
    generator_guarantees:
      - 先放箭头和提示条，再规划射线路径和墙，统一检查 occupied 集合防止重叠。
      - 提示条位置由箭头方向唯一决定，必须全在网格内，否则该尝试丢弃。
      - 墙仍只放在射线路径侧边 1 格处，且不占用提示条/箭头/已有路径。
      - output-first 校验继续保留：最终要求 `main(input) == planned_output`。

  e_visual_observability:
    rationale:
      - 局部多色提示条与该箭头紧邻，解题者能把“这 3 个颜色”和“该箭头反射后的段颜色序列”对应起来。
      - 墙保持单色，但输出颜色常常与墙不同，能明显提示“颜色来源变了”。
      - 2-4 次反射会展示提示条的顺序，有些样例会展示第 4 次反射回到第 1 色。

  f_visual_cleanliness:
    measures:
      - 提示条固定在箭头背后，不和路径主区域混在一起。
      - 墙保持单色，提示条固定多色，视觉角色分离。
      - 箭头数量控制在 1-5，避免 18-20 大小画布中过度拥挤。

  g_color_change_inferability:
    constraints:
      - 每个提示条 3 色互异，顺序由几何位置给出（left->right）。
      - 每次反射选用下一个提示色，墙色刻意避开该次提示色，确保“墙色无关”能被观察到。
      - 不要求不同箭头共享同一循环；线索是局部的、配对明确的。

  h_generator_behavior_metric:
    metric:
      - `diff_bounces`: 生成样本中，按旧规则会取墙色、按新规则会取提示色的反射次数。
      - 目标是多数样本 `diff_bounces > 0`，并统计 bounce 次数分布与 cycle index 使用分布。

  i_main_generator_consistency_review:
    checks:
      - solver 用箭头几何反推提示条坐标；generator 也用同一坐标规则放置提示条。
      - solver 将提示条视为装饰/阻挡而非墙；generator 保证路径不穿过提示条。
      - solver 的 cycle 步进以“每次反射”为单位；generator 规划输出时也在每次反射后推进一次 cycle。
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


def _next_cycle_color(cycle: list[int], bounce_idx: int) -> tuple[int, int]:
    return int(cycle[int(bounce_idx) % len(cycle)]), int(bounce_idx + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from triangular arrows, reflecting off colored wall pixels.

    Variant rule:
    - Reflection geometry is unchanged from 142ca369.
    - Each arrow has a local 3-color hint strip behind it.
    - On each bounce, the ray switches to the next hint color in that arrow's cycle.
    - The wall color never affects the ray color; the bounce point is recolored to the cycle color.
    """
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

    for arrow in arrows:
        cur = arrow["start"]
        dr = int(arrow["dr"])
        dc = int(arrow["dc"])
        cur_col = int(arrow["start_col"])
        cycle = list(map(int, arrow["cycle"]))
        bounce_idx = 0
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), int(bounce_idx % len(cycle)))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                new_col, bounce_idx = _next_cycle_color(cycle, bounce_idx)
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(new_col)
                cur_col = int(new_col)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in blocker_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                new_col, bounce_idx = _next_cycle_color(cycle, bounce_idx)
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                new_col, bounce_idx = _next_cycle_color(cycle, bounce_idx)
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                new_col, bounce_idx = _next_cycle_color(cycle, bounce_idx)
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                new_col, bounce_idx = _next_cycle_color(cycle, bounce_idx)
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                dc = -int(dc)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the D4 color-cycle variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 5)
    palette = list(range(1, 10))

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for _ai in range(n_arrows):
        arrow_color = _rng.choice(palette)
        cycle = _rng.sample([c for c in palette if c != int(arrow_color)], 3)
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 2])[0]

        placed = False
        for _ in range(120):
            sr = _rng.randint(0, H - 1)
            sc = _rng.randint(0, W - 1)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not _in_bounds(H, W, r, c) for r, c in arrow_pts):
                continue

            hint_pts = _hint_positions(arrow_pts, dr, dc)
            if any(not _in_bounds(H, W, r, c) for r, c in hint_pts):
                continue

            shape_cells = set((int(r), int(c)) for r, c in arrow_pts + hint_pts)
            if shape_cells & occupied:
                continue

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc = int(dr), int(dc)
            cur_col = int(arrow_color)
            bounce_idx = 0
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            ok = True

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces:
                    if max_steps < 4:
                        ok = False
                        break
                    seg_len = _rng.randint(3, max(3, max_steps - 1))
                else:
                    seg_len = max_steps

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not _in_bounds(H, W, nr, nc):
                        break
                    if (nr, nc) in occupied or (nr, nc) in shape_cells or (nr, nc) in ray_set:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                if seg_idx >= n_bounces:
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
                prefer_3 = _rng.random() < 0.6
                options.sort(key=lambda o: len(o[0]), reverse=prefer_3)
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
                next_col = int(cycle[bounce_idx % len(cycle)])
                wall_palette = [c for c in palette if c != int(next_col)]
                wcol = int(_rng.choice(wall_palette))
                walls.append((wpxs, wcol))

                if ray_cells:
                    lr, lc, _old = ray_cells[-1]
                    ray_cells[-1] = (int(lr), int(lc), int(next_col))
                cur_col = int(next_col)
                bounce_idx += 1
                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3:
                continue

            new_cells = set(shape_cells) | set(ray_set)
            for wpxs, _wcol in walls:
                new_cells.update((int(r), int(c)) for r, c in wpxs)
            expected_count = len(shape_cells) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(arrow_color)
            for (r, c), col in zip(hint_pts, cycle):
                grid[r, c] = out[r, c] = int(col)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)

            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D4 color-cycle variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
