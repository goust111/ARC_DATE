# Combo variant of 142ca369 (index 29)
# combo_spec: D5/3+D6/1
# Name: 固定步数终止+箭头编码额外信息

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, in_bounds


"""
Design analysis for combo D5/3 + D6/1
=====================================

a) Concrete behavior and trace
------------------------------
- 原题的源对象是 3 像素 L，输出规则是从直角顶点沿缺角反方向发出对角射线，遇墙反射，直到出界。
- 这次融合后，源对象本身改成 4/5 像素 L 变体，但仍保留同一个 3 像素 L 核心：
    start S      = 直角顶点 / 发射起点
    vertical arm = (S-dr, S)
    horiz arm    = (S, S-dc)
    missing      = (S-dr, S-dc) 决定发射方向 (dr,dc)
  允许的额外像素只放在臂端：
    ext_v = (S-2*dr, S)
    ext_h = (S, S-2*dc)
- 融合规格要求“额外像素数量编码最大步数”，因此这里把编码收束成两档：
    4 像素箭头 = 1 个额外像素（可加长竖臂或横臂） -> 固定走 5 步
    5 像素箭头 = 2 个额外像素（双臂都加长）       -> 固定走 8 步
  位置不再携带第二层语义，只保留“长得像加长 L”的视觉变化，真正解码只看额外像素数量。
- 小例子：
    若箭头朝右下，起点是 S=(8,8)。
    4 像素版本可能是 {(8,8),(7,8),(8,7),(6,8)}，solver 读出步数=5；
    5 像素版本是 {(8,8),(7,8),(8,7),(6,8),(8,6)}，solver 读出步数=8。
    之后射线仍按原题方式逐步对角前进、侧碰墙翻单轴、正撞墙翻双轴并继承墙颜色，
    但不再追到边界，而是在真实前进了 5 或 8 个格子后立刻停下。

b) Complexity calibration
-------------------------
- 只保留 2 档步数，避免“形状编码”同时引入太多自由度；4/5 像素的视觉差异也很清楚。
- 4 像素箭头允许两种外观（延长竖臂或横臂），但语义相同。这提供少量形状变化而不增加推理分支。
- 选用 5 步和 8 步而不是 1/2 或过大的数值：足够短，肉眼可数；也足够长，输出明显不同。
- 每个样例生成 3-4 个箭头，并强制同时出现 4 像素与 5 像素两类，使观察者能横向比较。

c) Audit of old assumptions
---------------------------
- 原 `is_arrow_pts()` 假设箭头恒为 3 像素、bbox 恒为 2x2。这里不成立，必须改成模板匹配：
  “3 像素 L 核心 + 允许的臂端延长像素”。
- 原 `arrow_dir_and_start()` 用度数最大点找直角顶点；5 像素双延长时会出现多个度数相同候选。
  这里直接枚举 `(start, dr, dc)` 并检查缺角与额外像素位置，避免歧义。
- 原 solver 的 `while True` 依赖“最终出界”停止。融合后主要停止条件变成 `steps_used == step_limit`。
- 原 `seen` 只记录 `(cur, dr, dc, cur_col)`；现在固定步数下，回到同一状态但已走的步数不同，
  后续行为不同，所以状态必须加入 `steps_used`。
- 原 `wall_cells = non-bg - arrow_cells` 仍然可用，因为这次没有额外 tail 标记；但这要求墙颜色不能和箭头
  在 4 邻接上并成同色大连通块。generator 用箭头色/墙色分池来保证。
- 原 generator 默认箭头占 2x2。现在 4/5 像素箭头最多占 3x3，需要新的边界检查。

d) Placement legality
---------------------
- 所有箭头都由合法模板构造：4 像素时仅一个臂端延长，5 像素时双臂延长。
- 生成器仍采用 output-first：先给每个箭头选定总步数、段长和反射墙，再回填输入。
- 箭头、墙、射线轨迹彼此不重叠；所有坐标都做 in-bounds 检查。
- 最终停点之后的下一格对角位置以及两侧潜在 side-touch 位置都保留为空，避免后续对象把
  “本应继续却因步数耗尽而终止”的视觉证据破坏掉。

e) Visual observability
-----------------------
- 解题者能直接看到：5 像素箭头比 4 像素箭头多一个同色附加格，而且它的输出轨迹稳定更长。
- 终点通常停在网格内部，且前方还有空格可走，因此“不是出界才停”这一点可直接从输入输出对看出来。
- 墙颜色继承仍保留，输出仍像原题那样是彩色折线，所以新旧规则的连续性很强。

f) Visual cleanliness
---------------------
- 没有远距离 marker；所有编码都贴在箭头本体上，局部可读性好。
- 箭头、墙、轨迹之间不叠压；每个箭头颜色唯一，避免同色对象难以配对。
- 反射墙仍只用 1 像素或 3 像素直线段，画面风格与原题一致。

g) Change inferability
----------------------
- 颜色不承担新的抽象映射，仅保留原题“撞哪面墙就继承哪面墙颜色”的结构性提示。
- 新增含义完全来自可见形状大小，因此观察者不需要利用颜色编号这种不可视因素。

h) Non-degeneracy target
------------------------
- 样本必须让“固定步数终止”真实发生，而不是恰好在最后一步就出界。
- 因此 generator 要求每条轨迹的终点都留有一个空的、仍在网格内的前向对角格；旧规则在这里会继续，
  新规则会停止，二者输出必然分叉。

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Solver 假设每个箭头连通块恰好匹配唯一合法模板。Generator 只放合法 4/5 像素模板。
- Solver 假设箭头颜色不与墙颜色重用。Generator 将颜色分池，避免同色墙并入箭头连通块。
- Solver 假设终点前方空格没有被别的对象占用。Generator 把这些 continuation / side-clearance
  位置加入 blocked 集合，后续放置不会侵入。
- Solver 允许 head-on reflection 不消耗步数，side-touch 在落点后生效。Generator 的路径规划与此一致。
"""


SHORT_STEPS = 5
LONG_STEPS = 8


def _steps_from_extra(extra: set[tuple[int, int]]) -> int | None:
    if len(extra) == 1:
        return SHORT_STEPS
    if len(extra) == 2:
        return LONG_STEPS
    return None


def _parse_arrow_component(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    """Return ((dr, dc), start, step_limit) for a legal 4/5-pixel arrow."""
    pts_set = {(int(r), int(c)) for r, c in pts}
    if len(pts_set) not in (4, 5):
        return None

    matches: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for sr, sc in pts_set:
        for dr in (-1, 1):
            for dc in (-1, 1):
                base = {(sr, sc), (sr - dr, sc), (sr, sc - dc)}
                miss = (sr - dr, sc - dc)
                ext_v = (sr - 2 * dr, sc)
                ext_h = (sr, sc - 2 * dc)
                if not base.issubset(pts_set):
                    continue
                if miss in pts_set:
                    continue
                extra = pts_set - base
                if not extra:
                    continue
                if not extra.issubset({ext_v, ext_h}):
                    continue
                step_limit = _steps_from_extra(extra)
                if step_limit is None:
                    continue
                matches.append(((int(dr), int(dc)), (int(sr), int(sc)), int(step_limit)))

    uniq: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    seen = set()
    for match in matches:
        if match not in seen:
            seen.add(match)
            uniq.append(match)
    if len(uniq) != 1:
        return None
    return uniq[0]


def _build_arrow_pts(sr: int, sc: int, dr: int, dc: int, style: str) -> list[tuple[int, int]]:
    pts = {(int(sr), int(sc)), (int(sr - dr), int(sc)), (int(sr), int(sc - dc))}
    if style in {"v", "both"}:
        pts.add((int(sr - 2 * dr), int(sc)))
    if style in {"h", "both"}:
        pts.add((int(sr), int(sc - 2 * dc)))
    return [(int(r), int(c)) for r, c in pts]


def _arrow_bounds(H: int, W: int, dr: int, dc: int, style: str) -> tuple[int, int, int, int]:
    if style in {"v", "both"}:
        r_lo = 2 if dr == 1 else 0
        r_hi = H - 1 if dr == 1 else H - 3
    else:
        r_lo = 1 if dr == 1 else 0
        r_hi = H - 1 if dr == 1 else H - 2

    if style in {"h", "both"}:
        c_lo = 2 if dc == 1 else 0
        c_hi = W - 1 if dc == 1 else W - 3
    else:
        c_lo = 1 if dc == 1 else 0
        c_hi = W - 1 if dc == 1 else W - 2
    return int(r_lo), int(r_hi), int(c_lo), int(c_hi)


def _sample_segment_lengths(total_steps: int, n_bounces: int) -> list[int] | None:
    parts = n_bounces + 1
    base = [2] * parts
    remaining = int(total_steps) - 2 * parts
    if remaining < 0:
        return None
    for _ in range(remaining):
        base[_rng.randrange(parts)] += 1
    _rng.shuffle(base)
    return [int(x) for x in base]


def _trace_plan(
    H: int,
    W: int,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    start_col: int,
    segment_lengths: list[int],
    wall_palette: list[int],
    blocked: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    cur_r, cur_c = int(start_r), int(start_c)
    cur_dr, cur_dc = int(dr), int(dc)
    cur_col = int(start_col)

    ray_cells: list[tuple[int, int, int]] = []
    ray_set: set[tuple[int, int]] = set()
    walls: list[tuple[list[tuple[int, int]], int]] = []
    wall_cells: set[tuple[int, int]] = set()
    local_blocked = set(reserved)

    for seg_idx, seg_len in enumerate(segment_lengths):
        for _ in range(seg_len):
            nr = int(cur_r + cur_dr)
            nc = int(cur_c + cur_dc)
            if not in_bounds((H, W), nr, nc):
                return None
            if (nr, nc) in blocked or (nr, nc) in local_blocked or (nr, nc) in ray_set:
                return None
            ray_cells.append((int(nr), int(nc), int(cur_col)))
            ray_set.add((int(nr), int(nc)))
            cur_r, cur_c = int(nr), int(nc)

        if seg_idx == len(segment_lengths) - 1:
            break

        options: list[tuple[list[tuple[int, int]], str]] = []

        wr, wc = int(cur_r + cur_dr), int(cur_c)
        if 0 <= wr < H:
            single = [(int(wr), int(wc))]
            options.append((single, "dr"))
            wall3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
            if all(0 <= cc < W for _, cc in wall3):
                options.append((wall3, "dr"))

        wr2, wc2 = int(cur_r), int(cur_c + cur_dc)
        if 0 <= wc2 < W:
            single = [(int(wr2), int(wc2))]
            options.append((single, "dc"))
            wall3 = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
            if all(0 <= rr < H for rr, _ in wall3):
                options.append((wall3, "dc"))

        if not options:
            return None

        _rng.shuffle(options)
        options.sort(key=lambda item: len(item[0]), reverse=_rng.random() < 0.55)

        chosen = None
        for wpxs, flip in options:
            if any((r, c) in blocked or (r, c) in local_blocked or (r, c) in ray_set for r, c in wpxs):
                continue
            chosen = (wpxs, flip)
            break
        if chosen is None:
            return None

        wpxs, flip = chosen
        w_choices = [c for c in wall_palette if int(c) != int(cur_col)] or wall_palette
        wcol = int(_rng.choice(w_choices))
        walls.append((wpxs, int(wcol)))
        wall_cells.update((int(r), int(c)) for r, c in wpxs)
        local_blocked.update((int(r), int(c)) for r, c in wpxs)

        lr, lc, _ = ray_cells[-1]
        ray_cells[-1] = (int(lr), int(lc), int(wcol))
        cur_col = int(wcol)
        if flip == "dr":
            cur_dr = -int(cur_dr)
        else:
            cur_dc = -int(cur_dc)

    next_diag = (int(cur_r + cur_dr), int(cur_c + cur_dc))
    if not in_bounds((H, W), next_diag[0], next_diag[1]):
        return None
    if next_diag in blocked or next_diag in local_blocked or next_diag in ray_set:
        return None

    side_checks = [(int(cur_r + cur_dr), int(cur_c)), (int(cur_r), int(cur_c + cur_dc))]
    for pos in side_checks:
        if not in_bounds((H, W), pos[0], pos[1]):
            return None
        if pos in blocked or pos in local_blocked or pos in ray_set:
            return None

    clearance = {next_diag, side_checks[0], side_checks[1]}

    return {
        "ray_cells": ray_cells,
        "ray_set": ray_set,
        "walls": walls,
        "wall_cells": wall_cells,
        "clearance": clearance,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from 4/5-pixel L arrows.

    - The 3-pixel L core still gives the launch direction.
    - Arrow size now encodes the fixed number of diagonal advances:
      4 pixels -> 5 steps, 5 pixels -> 8 steps.
    - Reflection geometry and wall-color inheritance remain the same as the base task.
    - The ray stops after exactly that many advances, even if it could continue in-bounds.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            info = _parse_arrow_component(pts)
            if info is None:
                continue
            (dr, dc), start, step_limit = info
            arrows.append((int(col), (int(dr), int(dc)), (int(start[0]), int(start[1])), int(step_limit)))
            arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, (dr, dc), start, step_limit in arrows:
        cur = (int(start[0]), int(start[1]))
        cur_col = int(start_col)
        steps_used = 0
        seen: set[tuple[tuple[int, int], int, int, int, int]] = set()

        while steps_used < step_limit:
            state = (cur, int(dr), int(dc), int(cur_col), int(steps_used))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            steps_used += 1
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            hit_r = (int(nr + dr), int(nc))
            if hit_r in wall_cells:
                wcol = int(g[hit_r[0], hit_r[1]])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            hit_c = (int(nr), int(nc + dc))
            if hit_c in wall_cells:
                wcol = int(g[hit_c[0], hit_c[1]])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(3, 4)

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = [int(c) for c in palette if c not in arrow_colors]
    if len(wall_palette) < 2:
        return None

    step_limits = [SHORT_STEPS, LONG_STEPS]
    while len(step_limits) < n_arrows:
        step_limits.append(int(_rng.choice([SHORT_STEPS, LONG_STEPS])))
    _rng.shuffle(step_limits)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    for color, step_limit in zip(arrow_colors, step_limits):
        color = int(color)
        step_limit = int(step_limit)
        placed = False

        for _ in range(180):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            if step_limit == SHORT_STEPS:
                style = _rng.choice(["v", "h"])
                bounce_choices = [0, 1]
                bounce_weights = [2, 3]
            else:
                style = "both"
                bounce_choices = [0, 1, 2, 3]
                bounce_weights = [1, 3, 3, 1]

            n_bounces = int(_rng.choices(bounce_choices, weights=bounce_weights)[0])
            segment_lengths = _sample_segment_lengths(step_limit, n_bounces)
            if segment_lengths is None:
                continue

            r_lo, r_hi, c_lo, c_hi = _arrow_bounds(H, W, dr, dc, style)
            initial_need = int(segment_lengths[0] + (1 if n_bounces == 0 else 0))
            if dr == 1:
                r_hi = min(r_hi, H - 1 - initial_need)
            else:
                r_lo = max(r_lo, initial_need)
            if dc == 1:
                c_hi = min(c_hi, W - 1 - initial_need)
            else:
                c_lo = max(c_lo, initial_need)
            if r_lo > r_hi or c_lo > c_hi:
                continue

            sr = _rng.randint(r_lo, r_hi)
            sc = _rng.randint(c_lo, c_hi)
            arrow_pts = _build_arrow_pts(sr, sc, dr, dc, style)
            reserved = set(arrow_pts)
            if any(not in_bounds((H, W), r, c) or (r, c) in blocked for r, c in reserved):
                continue

            plan = _trace_plan(
                H=H,
                W=W,
                start_r=sr,
                start_c=sc,
                dr=dr,
                dc=dc,
                start_col=color,
                segment_lengths=segment_lengths,
                wall_palette=wall_palette,
                blocked=blocked,
                reserved=reserved,
            )
            if plan is None:
                continue

            ray_cells = plan["ray_cells"]  # type: ignore[assignment]
            ray_set = plan["ray_set"]  # type: ignore[assignment]
            walls = plan["walls"]  # type: ignore[assignment]
            wall_cells = plan["wall_cells"]  # type: ignore[assignment]
            clearance = plan["clearance"]  # type: ignore[assignment]

            new_cells = set(reserved) | set(ray_set) | set(wall_cells)
            expected_count = len(reserved) + len(ray_cells) + sum(len(wpxs) for wpxs, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = int(color)
                out[r, c] = int(color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = int(wcol)
                    out[r, c] = int(wcol)

            occupied |= new_cells
            blocked |= new_cells | set(clearance)
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for combo variant D5/3 + D6/1 of 142ca369."""
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
