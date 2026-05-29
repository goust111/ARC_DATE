# Variant of 142ca369
# Dimension: D10 — 反射后变换效果 (transform)
# Change: 反射点重新涂色为墙壁颜色（recolor），射线继承墙壁颜色... → 反射时墙壁被擦除

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis
# variant_spec:
#   original_scene:
#     - 输入里有 3 像素 L 形箭头，以及离散的单像素/3 像素直线墙。
#     - 箭头从直角顶点沿对角线发射射线；侧碰墙时翻转一根轴，正撞墙时翻转两根轴。
#     - 原题在反射时做两件事：bounce 点改成墙色，之后的射线也继承墙色。
#   d10_change:
#     - 本变体只改“反射后的变换效果”：射线仍继承墙色，但被触碰的墙像素在输出中变成背景色。
#     - 因此输出里会出现“缺口墙”或“整块消失的单像素墙”，而 bounce 点保留为反射前的射线颜色。
#   concrete_trace:
#     - 设箭头颜色为 2，射线先画到 (6,6)，其右侧相邻墙像素 (6,7) 颜色为 8。
#     - 原题：输出把 (6,6) 改成 8，(6,7) 继续保留 8，后续段也用 8。
#     - 本变体：输出保留 (6,6)=2，把 (6,7) 擦成 0，后续段改用 8。
#     - 所以解题者能从“墙消失了，但转折后的线段换色了”反推出原始墙的位置和颜色。
#   complexity_calibration:
#     - 几何、箭头检测、反射方向规则保持原样，避免把题改成别的维度。
#     - 仍保留 1 像素和 3 像素墙；3 像素墙只擦掉真正被碰到的中心像素，视觉上能看到局部缺口。
#     - generator 强制每个样例至少有 1 次 bounce，避免退化成“没有墙被擦除”的原题近似。
#     - 为了让颜色继承与擦除同时可见，generator 令每次墙色都不同于入射射线颜色。
#   implicit_assumption_review:
#     - 原 solver 的 `out = g.copy()` 默认所有墙最终都保留；本变体必须在命中时显式擦除墙像素。
#     - 原 solver 的 `wall_cells` 是静态集合；本变体里墙会消失，必须改成可变 `live_walls`。
#     - 原 solver 在反射时把 bounce 点直接改成墙色；这正是本维度要替换掉的旧行为，必须删除。
#     - 原 solver 的去环状态只记录 `(cur, dr, dc, cur_col)`，默认墙场不变；本变体中墙会减少，
#       所以状态还要包含剩余墙位置，否则同一局部状态在不同墙配置下会被误判成循环。
#     - 顺序依赖风险：若两个射线共用同一墙像素，先擦除者会影响后擦除者。generator 因此保证
#       每个被触碰的墙像素只服务一次，且后续路径不会依赖已擦除的墙。
#   placement_legality:
#     - 箭头、射线路径、墙像素在输入中仍然不重叠。
#     - generator 额外维护 `wall_set`，防止“后续路径踩到本箭头先前放置的墙”这类原代码只靠 oracle 回退的情况。
#     - 被擦除的墙像素虽然在输出里变成背景，但在输入放置阶段仍算占用，保证题面干净。
#   visual_observability:
#     - 人类可直接对比输入/输出看到：某些墙在输出里缺失，而折线在缺口处转向并换成原墙色。
#     - 只改反射处，不改箭头形状、方向判定和全局布局，因此规律集中、可归纳。
#   cleanliness:
#     - 继续使用稀疏散布的墙和不重叠的对角射线，避免多层覆盖。
#     - 3 像素墙最多留下两端残片，仍能和箭头/射线清晰区分。
#   new_solver_assumptions_checked:
#     - `main()` 假设输入中的非箭头非背景像素都可当墙，这与 generator 构造一致。
#     - `main()` 允许墙被擦除后不再参与后续反射；generator 保证样例不会要求“同一墙像素二次反射”。


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows.

    Variant rule:
    - Reflection geometry is unchanged from 142ca369.
    - On every reflection the ray inherits the touched wall color.
    - Instead of recoloring the bounce point, the touched wall pixel is erased to bg.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    live_walls: dict[tuple[int, int], int] = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    def erase_wall(cell: tuple[int, int]) -> int:
        wcol = int(live_walls.pop(cell))
        out[cell[0], cell[1]] = int(bg)
        return wcol

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue

        (dr, dc), cur = info
        cur_col = int(start_col)
        seen = set()

        while True:
            wall_sig = tuple(sorted(live_walls))
            state = (cur, int(dr), int(dc), int(cur_col), wall_sig)
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in live_walls:
                cur_col = erase_wall((nr, nc))
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in live_walls:
                cur_col = erase_wall((nr + 1, nc))
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in live_walls:
                cur_col = erase_wall((nr - 1, nc))
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in live_walls:
                cur_col = erase_wall((nr, nc + 1))
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in live_walls:
                cur_col = erase_wall((nr, nc - 1))
                dc = -int(dc)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Output-first generator for the D10 erase-on-reflection variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    bounce_counts = [_rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0] for _ in range(n_arrows)]
    if sum(bounce_counts) == 0:
        bounce_counts[_rng.randrange(n_arrows)] = 1

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = int(bounce_counts[ai])

        placed = False
        for _ in range(80):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c = int(sr), int(sc)
            cur_dr, cur_dc = int(dr), int(dc)
            cur_col = int(color)
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int, tuple[int, int]]] = []
            ray_set: set[tuple[int, int]] = set()
            wall_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True
            bounces_left = int(n_bounces)

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces and bounces_left > 0:
                    if max_steps < 4:
                        bounces_left = 0
                        seg_len = int(max_steps)
                    else:
                        seg_len = _rng.randint(3, max(3, int(max_steps) - 1))
                else:
                    seg_len = int(max_steps)

                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set or (nr, nc) in wall_set:
                        ok = False
                        break
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                if seg_idx >= n_bounces or bounces_left <= 0:
                    continue
                bounces_left -= 1

                options: list[tuple[list[tuple[int, int]], str, tuple[int, int]]] = []

                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    touched = (int(wr), int(wc))
                    w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _, c in w3):
                        options.append((w3, "dr", touched))
                    options.append(([(int(wr), int(wc))], "dr", touched))

                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    touched = (int(wr2), int(wc2))
                    w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _ in w3v):
                        options.append((w3v, "dc", touched))
                    options.append(([(int(wr2), int(wc2))], "dc", touched))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                prefer_3 = _rng.random() < 0.6
                options.sort(key=lambda o: len(o[0]), reverse=prefer_3)
                chosen_wall = None
                for wpxs, flip, touched in options:
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set or (r, c) in wall_set for r, c in wpxs):
                        continue
                    chosen_wall = (wpxs, flip, touched)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip, touched = chosen_wall
                wall_palette = [p for p in palette if int(p) != int(cur_col)]
                wcol = int(_rng.choice(wall_palette or palette))
                walls.append((wpxs, wcol, touched))
                wall_set.update(wpxs)
                cur_col = int(wcol)
                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3:
                continue

            new_cells = set(arrow_pts) | ray_set | wall_set
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _col, _touch in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol, touched in walls:
                for r, c in wpxs:
                    grid[r, c] = int(wcol)
                    if (int(r), int(c)) != touched:
                        out[r, c] = int(wcol)

            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None

    erased = int(np.count_nonzero((grid != 0) & (out == 0)))
    if erased == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D10 erase-on-reflection variant."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
