# Combo variant of 142ca369 (index 20)
# combo_spec: D5/0+D6/0+D14/0
# Name: 单像素源+目标像素终止+小网格

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the combo variant
# -------------------------------------
# concrete_form:
#   - 原题的 3 像素 L 形箭头改成“边框单像素控制点”。控制点本身就是射线源，方向由其所在边和半边位置编码。
#   - 每条射线都有一个内部单像素目标。射线进入目标格时立刻停止，目标色在输出里保持不变。
#   - 为了让 solver 在小网格里无歧义地区分“控制点 / 目标 / 墙”，这里强制墙只用 3 像素轴对齐线段；
#     于是：边框 singleton = 控制点，内部 singleton = 目标，非 singleton = 墙。
#   - 网格缩到 8-12 行列，源数量降到 1-3，墙总数降到 0-2，保证紧凑但仍能看见方向编码与终止行为。
#
# behavior_trace:
#   - 例：10x9 网格里，顶边控制点在 (0,2)。它位于顶边左半，因此方向解码为 (1,1)，第一步进入 (1,3)。
#   - 若射线沿 (1,1) 前进，在 (4,6) 处侧碰到位于 (5,6) 的水平墙，则当前格 (4,6) 继承墙色，dr 翻转成 -1，
#     射线后续沿 (-1,1) 继续。
#   - 若再走到下一段末尾的目标 (2,8)，则射线进入该目标并停止；不会再像原题那样继续飞到出界。
#   - 和原题相比，真正改变的是三个环节：源识别、停止条件、以及为保持这两者可见而收紧的布局规模。
#
# complexity_calibration:
#   - 保留对角线射线与原反射物理；不引入水平/垂直射线，否则会把题目从“读边框位置编码”扩成另一套动力学。
#   - 控制点允许落在整条边上，但生成器偏向非角落，确保“边 + 半边 -> 方向”的规律能被肉眼看出来。
#   - 目标统一使用一种重复颜色，墙使用别的颜色，避免小网格里出现太多无结构的颜色变化。
#
# implicit_assumptions_review:
#   - 原 `is_arrow_pts` / `arrow_dir_and_start` 的 L 形箭头假设全部失效，改成 `_control_dir`。
#   - 原 “所有非源非背景都是墙” 的假设失效，因为现在存在目标 singleton；solver 必须先按连通块面积分类。
#   - D5 和 D6 叠加后最大的冲突是“singleton 既可能是源又可能是目标也可能是墙”。
#     解决方法不是在 solver 里猜，而是在生成器里禁止 singleton 墙，只保留 3 像素墙段。
#   - solver 现在隐含依赖：边框 singleton 一定是源，内部 singleton 一定是目标。生成器会严格满足。
#
# placement_legality:
#   - 控制点必须在边框上，且其解码方向的第一步必须进入网格。
#   - 目标必须在内部，且同色目标之间不能 4 连通，否则会被并成一个多像素组件。
#   - 墙不得触边，并且不得与同色控制点 4 连通，否则控制点组件就不再是 singleton。
#   - 所有路径、墙、控制点、目标都不重叠；最后一段在命中目标后仍保留至少 1 格额外 runway，
#     这样“终止于目标”相对“飞到边界”是真实起作用的，而不是退化情况。
#
# observability_and_cleanliness:
#   - 输入里边框上只有散落的单像素源；内部只有一个重复颜色的目标点若干；墙是少量 3 像素短线段。
#   - 这三类对象在视觉上可直接分层，不需要依赖颜色编号大小等人类不可见因素。
#
# solver_assumption_audit:
#   - `main()` 中每个循环都假设单条射线不会与别的射线共享绘制格；生成器用 output-first 规划并拒绝交叉。
#   - 侧碰反射逻辑假设墙格集合固定，不受输出写入影响；因此 solver 始终从输入 `g` 读取墙色。
#   - 目标命中的判断必须发生在“对角撞墙”之前，否则目标会被误当成阻挡物；代码顺序据此调整。


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H) - 1) or int(c) in (0, int(W) - 1)


def _is_corner(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H) - 1) and int(c) in (0, int(W) - 1)


def _control_dir(H: int, W: int, r: int, c: int) -> tuple[int, int] | None:
    r = int(r)
    c = int(c)
    H = int(H)
    W = int(W)
    if not _is_border(r, c, H, W):
        return None
    if (r, c) == (0, 0):
        return 1, 1
    if (r, c) == (0, W - 1):
        return 1, -1
    if (r, c) == (H - 1, 0):
        return -1, 1
    if (r, c) == (H - 1, W - 1):
        return -1, -1
    if r == 0:
        return 1, 1 if c < (W / 2.0) else -1
    if r == H - 1:
        return -1, 1 if c < (W / 2.0) else -1
    if c == 0:
        return (1 if r < (H / 2.0) else -1), 1
    return (1 if r < (H / 2.0) else -1), -1


def _candidate_sources(H: int, W: int, dr: int, dc: int) -> list[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(W):
        if _control_dir(H, W, 0, c) == (int(dr), int(dc)):
            pts.add((0, int(c)))
        if _control_dir(H, W, H - 1, c) == (int(dr), int(dc)):
            pts.add((H - 1, int(c)))
    for r in range(H):
        if _control_dir(H, W, r, 0) == (int(dr), int(dc)):
            pts.add((int(r), 0))
        if _control_dir(H, W, r, W - 1) == (int(dr), int(dc)):
            pts.add((int(r), W - 1))
    return sorted(pts)


def _pick_source(candidates: list[tuple[int, int]], H: int, W: int) -> tuple[int, int]:
    non_corner = [p for p in candidates if not _is_corner(p[0], p[1], H, W)]
    corner = [p for p in candidates if p not in non_corner]
    if non_corner and (_rng.random() < 0.9 or not corner):
        return _rng.choice(non_corner)
    return _rng.choice(corner if corner else candidates)


def _orth_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _max_diag_steps(H: int, W: int, r: int, c: int, dr: int, dc: int) -> int:
    return int(
        min(
            (H - 1 - int(r)) if int(dr) == 1 else int(r),
            (W - 1 - int(c)) if int(dc) == 1 else int(c),
        )
    )


def _find_objects(
    g: np.ndarray,
    bg: int,
) -> tuple[list[tuple[int, tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    H, W = g.shape
    controls: list[tuple[int, tuple[int, int]]] = []
    target_cells: set[tuple[int, int]] = set()
    wall_cells: set[tuple[int, int]] = set()

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if len(pts) == 1:
                r, c = pts[0]
                if _is_border(r, c, H, W):
                    controls.append((int(col), (int(r), int(c))))
                else:
                    target_cells.add((int(r), int(c)))
            else:
                wall_cells.update((int(r), int(c)) for r, c in pts)

    controls.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    return controls, target_cells, wall_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from border control pixels to interior target pixels.

    Input structure:
    - small bg grid
    - 1-pixel controls on the border
    - 1-pixel interior targets
    - 3-pixel interior walls

    Rule:
    - A control pixel launches one diagonal ray; direction is decoded from edge position.
    - Rays inherit wall color on bounce, using the original corner-hit and side-touch reflection rules.
    - Entering a target cell stops the ray immediately.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    controls, target_cells, wall_cells = _find_objects(g, bg)
    control_cells = {pos for _col, pos in controls}
    out = g.copy()

    for start_col, cur in controls:
        decoded = _control_dir(H, W, cur[0], cur[1])
        if decoded is None:
            continue
        dr, dc = int(decoded[0]), int(decoded[1])
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in target_cells:
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in control_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def _adjacent_source_colors(
    wall_pixels: list[tuple[int, int]],
    source_colors: dict[tuple[int, int], int],
) -> set[int]:
    forbidden: set[int] = set()
    for r, c in wall_pixels:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (int(rr), int(cc)) in source_colors:
                forbidden.add(int(source_colors[(int(rr), int(cc))]))
    return forbidden


def _wall_options(
    H: int,
    W: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[list[tuple[int, int]], str]]:
    options: list[tuple[list[tuple[int, int]], str]] = []

    wr = int(r + dr)
    horiz = [(int(wr), int(c + d)) for d in (-1, 0, 1)]
    if all(
        0 <= rr < H and 0 <= cc < W and not _is_border(rr, cc, H, W) and (int(rr), int(cc)) not in occupied
        for rr, cc in horiz
    ):
        options.append((horiz, "dr"))

    wc = int(c + dc)
    vert = [(int(r + d), int(wc)) for d in (-1, 0, 1)]
    if all(
        0 <= rr < H and 0 <= cc < W and not _is_border(rr, cc, H, W) and (int(rr), int(cc)) not in occupied
        for rr, cc in vert
    ):
        options.append((vert, "dc"))

    return options


def _plan_ray(
    H: int,
    W: int,
    source: tuple[int, int],
    source_color: int,
    dr: int,
    dc: int,
    n_bounces: int,
    occupied: set[tuple[int, int]],
    committed_sources: dict[tuple[int, int], int],
    target_cells: set[tuple[int, int]],
    wall_palette: list[int],
) -> tuple[list[tuple[int, int, int]], list[tuple[list[tuple[int, int]], int]], tuple[int, int], int] | None:
    for _ in range(120):
        cur_r, cur_c = int(source[0]), int(source[1])
        cur_dr, cur_dc = int(dr), int(dc)
        cur_col = int(source_color)
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[tuple[int, int]], int]] = []
        local_occ: set[tuple[int, int]] = {source}
        used_wall_colors: set[int] = set()
        local_sources = dict(committed_sources)
        local_sources[source] = int(source_color)
        ok = True

        for bounce_idx in range(n_bounces):
            max_steps = _max_diag_steps(H, W, cur_r, cur_c, cur_dr, cur_dc)
            if max_steps < 2:
                ok = False
                break

            choices: list[tuple[list[tuple[int, int]], list[tuple[int, int]], str, int]] = []
            seg_cells: list[tuple[int, int]] = []
            for step_len in range(1, max_steps + 1):
                nr = int(cur_r + step_len * cur_dr)
                nc = int(cur_c + step_len * cur_dc)
                cell = (int(nr), int(nc))
                if cell in occupied or cell in local_occ:
                    break
                seg_cells.append(cell)
                if step_len < 2:
                    continue

                bounce_r, bounce_c = seg_cells[-1]
                future_bounces = int(n_bounces - bounce_idx - 1)
                for wall_pixels, flip in _wall_options(
                    H,
                    W,
                    bounce_r,
                    bounce_c,
                    cur_dr,
                    cur_dc,
                    occupied | local_occ,
                ):
                    new_dr = -int(cur_dr) if flip == "dr" else int(cur_dr)
                    new_dc = int(cur_dc) if flip == "dr" else -int(cur_dc)
                    need_room = 3 if future_bounces == 0 else 2
                    if _max_diag_steps(H, W, bounce_r, bounce_c, new_dr, new_dc) < need_room:
                        continue
                    choices.append((list(seg_cells), wall_pixels, str(flip), int(step_len)))

            if not choices:
                ok = False
                break

            seg_cells, wall_pixels, flip, _step_len = _rng.choice(choices)
            for rr, cc in seg_cells:
                ray_cells.append((int(rr), int(cc), int(cur_col)))
                local_occ.add((int(rr), int(cc)))

            forbidden = _adjacent_source_colors(wall_pixels, local_sources)
            color_choices = [
                int(col)
                for col in wall_palette
                if int(col) not in used_wall_colors and int(col) not in forbidden
            ]
            if not color_choices:
                ok = False
                break
            wcol = int(_rng.choice(color_choices))
            used_wall_colors.add(int(wcol))
            walls.append((wall_pixels, int(wcol)))
            local_occ.update((int(r), int(c)) for r, c in wall_pixels)

            lr, lc, _old = ray_cells[-1]
            ray_cells[-1] = (int(lr), int(lc), int(wcol))

            cur_r, cur_c = int(seg_cells[-1][0]), int(seg_cells[-1][1])
            cur_col = int(wcol)
            if flip == "dr":
                cur_dr = -int(cur_dr)
            else:
                cur_dc = -int(cur_dc)

        if not ok:
            continue

        max_steps = _max_diag_steps(H, W, cur_r, cur_c, cur_dr, cur_dc)
        if max_steps < 3:
            continue

        final_choices: list[tuple[list[tuple[int, int]], tuple[int, int], int]] = []
        seg_cells = []
        for step_len in range(1, max_steps):
            nr = int(cur_r + step_len * cur_dr)
            nc = int(cur_c + step_len * cur_dc)
            cell = (int(nr), int(nc))
            if cell in occupied or cell in local_occ:
                break
            seg_cells.append(cell)
            if step_len < 2:
                continue

            target = seg_cells[-1]
            if _is_border(target[0], target[1], H, W):
                continue
            if any(_orth_adjacent(target, other) for other in target_cells):
                continue

            painted = list(seg_cells[:-1])
            if len(ray_cells) + len(painted) < 2:
                continue
            spare_runway = int(max_steps - step_len)
            if spare_runway < 1:
                continue
            final_choices.append((painted, target, spare_runway))

        if not final_choices:
            continue

        painted, target, spare_runway = _rng.choice(final_choices)
        for rr, cc in painted:
            ray_cells.append((int(rr), int(cc), int(cur_col)))
            local_occ.add((int(rr), int(cc)))

        return ray_cells, walls, target, int(spare_runway)

    return None


def _gen_attempt() -> np.ndarray | None:
    """Output-first generator for the D5/0 + D6/0 + D14/0 combo."""
    H, W = _rng.randint(8, 12), _rng.randint(8, 12)
    n_controls = _rng.choices([1, 2, 3], weights=[2, 4, 1])[0]
    total_bounces = _rng.choices([0, 1, 2], weights=[1, 4, 3])[0]

    bounces_per_ray = [0 for _ in range(n_controls)]
    for _ in range(total_bounces):
        idx_choices = [i for i in range(n_controls) if bounces_per_ray[i] < 2]
        if not idx_choices:
            break
        bounces_per_ray[_rng.choice(idx_choices)] += 1

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    target_color = int(palette[0])
    source_colors = [int(c) for c in palette[1 : 1 + n_controls]]
    wall_palette = [int(c) for c in palette[1 + n_controls :] if int(c) != int(target_color)]
    if len(wall_palette) < total_bounces:
        return None

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    committed_sources: dict[tuple[int, int], int] = {}
    target_cells: set[tuple[int, int]] = set()
    non_corner_controls = 0
    target_runways: list[int] = []

    for ai in range(n_controls):
        source_color = int(source_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = int(bounces_per_ray[ai])

        placed = False
        for _ in range(120):
            candidates = [p for p in _candidate_sources(H, W, dr, dc) if p not in occupied]
            if not candidates:
                break
            source = _pick_source(candidates, H, W)

            planned = _plan_ray(
                H,
                W,
                source,
                source_color,
                dr,
                dc,
                n_bounces,
                occupied,
                committed_sources,
                target_cells,
                wall_palette,
            )
            if planned is None:
                continue

            ray_cells, walls, target, spare_runway = planned
            new_cells = {source, target}
            new_cells.update((int(r), int(c)) for r, c, _col in ray_cells)
            for wall_pixels, _wcol in walls:
                new_cells.update((int(r), int(c)) for r, c in wall_pixels)

            expected_count = 2 + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or (new_cells & occupied):
                continue

            grid[int(source[0]), int(source[1])] = out[int(source[0]), int(source[1])] = int(source_color)
            committed_sources[source] = int(source_color)
            if not _is_corner(source[0], source[1], H, W):
                non_corner_controls += 1

            for r, c, col in ray_cells:
                out[int(r), int(c)] = int(col)
            for wall_pixels, wcol in walls:
                for r, c in wall_pixels:
                    grid[int(r), int(c)] = out[int(r), int(c)] = int(wcol)

            grid[int(target[0]), int(target[1])] = out[int(target[0]), int(target[1])] = int(target_color)
            target_cells.add((int(target[0]), int(target[1])))
            target_runways.append(int(spare_runway))

            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if non_corner_controls == 0:
        return None
    if len(target_cells) != n_controls:
        return None
    if not target_runways or min(target_runways) < 1:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for combo variant 142ca369/20."""
    for _ in range(700):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 700 attempts")
