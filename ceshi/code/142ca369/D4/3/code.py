# Variant of 142ca369
# Dimension: D4 — 反射时颜色继承规则 (trajectory)
# Change: color_change_on_bounce——射线在每次反射时采用墙壁的颜色，反射点也被重新涂为墙壁颜色... → 射线颜色传递给墙壁

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for this D4 variant
# -----------------------------------
# variant_spec:
#   core_change:
#     - 原题在反射时执行“墙 -> 射线”的颜色继承：射线颜色切换成墙色，反弹点也改成墙色。
#     - 本变体反转成“射线 -> 墙”的颜色继承：射线保持自己的当前颜色不变，
#       被碰到的墙在输出中改成射线色。为了让视觉效果明确，若墙是 3 像素线段，
#       则整段墙都改色，而不是只改一个接触像素。
#   concrete_io_behavior:
#     - 输入仍然由 L 形箭头 + 单像素/3 像素直线墙组成。
#     - 输出中的对角射线整条保持箭头色，不会在反射后换色。
#     - 输出中的墙会在被某条射线命中时改成该射线的颜色。
#   small_trace:
#     - 例：箭头色 = 2，旁边的竖墙输入色 = 5。
#     - 原题：射线到达反弹点后，该反弹点会被改成 5，后续轨迹也都变成 5。
#     - 本题：反弹点仍保持 2，后续轨迹仍保持 2；那段竖墙在输出里改成 2。
#   complexity_calibration:
#     - 若允许 0 次反弹，变体可能退化成“只是画对角线”，看不出 D4 差异。
#       因此生成器强制整张图至少 1 次反弹，且至少有一段被改色的墙。
#     - 为了让“墙变色”更显眼，生成器偏向放置 3 像素墙，并要求墙输入色 != 命中它的箭头色。
#   main_implicit_assumption_audit:
#     - 箭头检测仍可复用：D4 只改反弹时的颜色传播方向，不改箭头几何。
#     - 原 solver 在 corner/side bounce 时都把 cur_col 改成墙色；这个前提已失效，
#       现在必须保持 ray_color 恒定，否则会退回原题行为。
#     - 原 solver 把反弹点改成墙色；这个前提也失效，现在反弹点应继续保留射线色。
#     - 原 solver 用输入网格 g 的墙色读取 wcol；D4 不再需要把该颜色传给射线，
#       但仍需要用输入网格静态识别哪些像素是墙，避免因为输出改色影响后续几何。
#   placement_legality:
#     - 生成器继续 output-first 规划路径与墙，再回填输入。
#     - 每段墙只服务一次反弹；不同箭头/墙/轨迹不重叠，避免输出依赖处理顺序。
#     - 墙段仍保持单色输入组件，方便 solver 把“整段墙”作为一个 4 连通同色组件改色。
#   visual_observability:
#     - 同一箭头的整条射线保持单色，墙从原色变成箭头色；这与原题的“射线分段换色”
#       形成明确反差，几组样例就能看出来。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from L-shaped arrows.

    D4 variant rule:
    - Rays still reflect off wall pixels using the original geometry.
    - On a bounce, the ray keeps its current color.
    - The touched wall component is recolored to the ray color in the output.
    - Ray stops when it exits the grid or reaches another arrow.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
        if len(pts) != 3:
            return False
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)

    def arrow_dir_and_start(pts: list[tuple[int, int]]):
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

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    arrow_mask = np.zeros_like(g, dtype=bool)
    for r, c in arrow_cells:
        arrow_mask[r, c] = True

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere((g != int(bg)) & (~arrow_mask))
    }
    wall_component_by_cell: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for col in colors_present(g, exclude={bg}):
        wall_mask = (g == int(col)) & (~arrow_mask)
        if not wall_mask.any():
            continue
        for coords in components_mask(wall_mask, connectivity=4):
            comp = tuple((int(r), int(c)) for r, c in coords)
            for cell in comp:
                wall_component_by_cell[cell] = comp

    out = g.copy()

    def recolor_wall_component(cell: tuple[int, int], ray_color: int) -> None:
        for wr, wc in wall_component_by_cell.get(cell, (cell,)):
            out[wr, wc] = int(ray_color)

    for ray_color, pts in arrows:
        info = arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        ray_color = int(ray_color)
        seen: set[tuple[tuple[int, int], int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            # Corner reflection: the ray bounces before entering the wall cell.
            if (nr, nc) in wall_cells:
                if int(out[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(ray_color)
                recolor_wall_component((nr, nc), ray_color)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(ray_color)

            prev_dr, prev_dc = int(dr), int(dc)
            if prev_dr == 1 and (nr + 1, nc) in wall_cells:
                recolor_wall_component((nr + 1, nc), ray_color)
                dr = -prev_dr
            elif prev_dr == -1 and (nr - 1, nc) in wall_cells:
                recolor_wall_component((nr - 1, nc), ray_color)
                dr = -prev_dr

            if prev_dc == 1 and (nr, nc + 1) in wall_cells:
                recolor_wall_component((nr, nc + 1), ray_color)
                dc = -prev_dc
            elif prev_dc == -1 and (nr, nc - 1) in wall_cells:
                recolor_wall_component((nr, nc - 1), ray_color)
                dc = -prev_dc

    return out


def _gen_attempt() -> np.ndarray | None:
    """Generate one valid input for the D4 variant via output-first planning."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    total_bounces = 0
    visible_wall_recolors = 0
    long_walls_recolored = 0

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        if ai == 0:
            n_bounces = _rng.choices([1, 2, 3, 4], weights=[4, 3, 2, 1])[0]
        else:
            n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[1, 4, 3, 2, 1])[0]

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

            cur_r, cur_c, cur_dr, cur_dc = sr, sc, dr, dc
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int, int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True
            bounces_left = n_bounces

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces and bounces_left > 0:
                    if max_steps < 4:
                        ok = False
                        break
                    seg_len = _rng.randint(3, max(3, max_steps - 1))
                else:
                    seg_len = max_steps

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    ray_cells.append((nr, nc, color))
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                if seg_idx >= n_bounces or bounces_left <= 0:
                    continue
                bounces_left -= 1

                options: list[tuple[list[tuple[int, int]], str]] = []
                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _, c in w3):
                        options.append((w3, "dr"))
                    options.append(([(wr, wc)], "dr"))

                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _ in w3v):
                        options.append((w3v, "dc"))
                    options.append(([(wr2, wc2)], "dc"))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                options.sort(key=lambda o: (len(o[0]) != 3, _rng.random()))
                chosen_wall = None
                for wpxs, flip in options:
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                        continue
                    chosen_wall = (wpxs, flip)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip = chosen_wall
                wall_palette = [p for p in palette if p != color]
                if not wall_palette:
                    ok = False
                    break
                wall_in_color = int(_rng.choice(wall_palette))
                walls.append((wpxs, wall_in_color, color))

                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _win, _wout in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _win, _wout in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = color
                out[r, c] = color
            for r, c, col in ray_cells:
                out[r, c] = col
            for wpxs, wall_in_color, wall_out_color in walls:
                for r, c in wpxs:
                    grid[r, c] = wall_in_color
                    out[r, c] = wall_out_color

            occupied |= new_cells
            total_bounces += len(walls)
            visible_wall_recolors += sum(1 for _wpxs, wall_in_color, wall_out_color in walls if wall_in_color != wall_out_color)
            long_walls_recolored += sum(1 for wpxs, _wall_in_color, _wall_out_color in walls if len(wpxs) == 3)
            placed = True
            break

        if not placed:
            return None

    if total_bounces == 0 or visible_wall_recolors == 0:
        return None
    if long_walls_recolored == 0 and total_bounces < 2:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D4 variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
