# Variant of 142ca369
# Dimension: D12 — 每条射线的反射次数 (meta)
# Change: 每条射线0-4次反射（bounces_per_arrow），由沿途墙壁数量决定... → 反射次数由箭头编码

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, neighbors4


"""
design_analysis:
  concrete_variant:
    arrows: |
      保留原题“L 形箭头朝缺角的反方向发射对角线射线”的核心语义，但箭头不再固定为 3 像素。
      新箭头都由同一个 3 像素基底 L 构成，右角点是发射起点；再沿两条臂向后可额外长出 0/1/2 个尾像素。
      因此允许 3/4/5 像素三档：3 像素 = 纯基底，4 像素 = 其中一条臂多 1 格，5 像素 = 两条臂都多 1 格。
    encoding: |
      反射额度只由箭头像素数决定：max_bounces = len(arrow) - 3，所以 3/4/5 像素分别编码 0/1/2 次最大反射。
      墙仍然是单像素或 3 像素直线段，并继续通过侧碰让射线继承墙色。
    flow_example: |
      例如某支 3 像素箭头右下发射，路径旁先后有两个侧墙。
      原逻辑：第一面墙会翻转一个轴并变色，第二面墙还会继续翻转。
      变体逻辑：该箭头的 max_bounces=0，所以两面墙都只作为“看得见但无效”的墙，射线始终保持原方向和原颜色直飞出界。
      再例如 5 像素箭头遇到四面侧墙：前两面仍按原题反射，后两面开始被忽略，路径从第二次反射后改为一条直线。
  complexity_calibration:
    rationale: |
      只保留三档额度，避免把“箭头形状→额度”的映射做得过细；同时每支箭都会至少遇到 1 个超额墙，
      这样“额度耗尽后忽略墙”在输出上是稳定可见的，不会退化成原题的“墙数=反射数”。
  solver_assumption_audit:
    original_dependencies:
      - 原 solver 的 is_arrow_pts 假设箭头恒为 3 像素 2x2 L；变体中不成立，必须改为显式匹配 3/4/5 像素尾巴家族。
      - 原 solver 看到侧墙就无条件反射；变体中必须改为有额度门控，否则 3 像素箭头会被误判为还能反射。
      - 原 generator 把“墙的数量”直接当成“实际反射次数”；变体中不成立，因为输入必须允许超额墙存在而不生效。
    new_assumptions:
      - 生成器只放置 side-touch 墙，避免“额度耗尽后正面撞墙是否穿墙”的歧义；solver 仍保留原正面撞墙分支，但本变体数据不依赖它。
      - 4 像素箭头允许把额外像素放在任一臂上，但 solver 只依据像素总数解码额度，不依赖是哪一条臂更长。
  placement_legality:
    guarantees: |
      生成器对每支箭独立回溯规划完整输出路径，确保箭头、墙、射线三者不重叠、不越界。
      墙色避开相邻同色对象，降低同色连通块意外合并成“假箭头”或“大墙”的风险。
  observability:
    human_cues: |
      同一张图中通常会混有不同大小的箭头，且每支箭都会把自己的额度用满并再经过 1-2 个无效侧墙。
      因此人眼能看到：小箭头更早停止转向/变色，大箭头能多反一次，再之后所有墙都只留在输入里而不改变输出。
  cleanliness:
    visual: |
      仍沿用原题的稀疏布局和单色对象；额外信息只体现在箭头尾巴长度，不引入额外标记层。
  color_inference:
    note: |
      颜色规则不变：只有真实生效的反射才继承墙色。超额墙会故意使用与当前射线不同的颜色，
      从而让“为什么这面墙没有改色”也成为可观察线索。
"""


def _detect_arrow(
    pts: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    """Return ((dr,dc), start, max_bounces) for a 3/4/5-pixel arrow variant."""
    pts_set = {(int(r), int(c)) for r, c in pts}
    if len(pts_set) not in {3, 4, 5}:
        return None

    matches: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for sr, sc in pts_set:
        for dr in (-1, 1):
            for dc in (-1, 1):
                core = {(sr, sc), (sr - dr, sc), (sr, sc - dc)}
                if not core.issubset(pts_set):
                    continue
                allowed = core | {(sr - 2 * dr, sc), (sr, sc - 2 * dc)}
                if pts_set <= allowed:
                    matches.append(((int(dr), int(dc)), (int(sr), int(sc)), len(pts_set) - 3))

    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from monochrome L-shaped arrows with encoded bounce budgets.

    Variant rule:
    - Arrow shape is a 3/4/5-pixel L-family; pixel count encodes max side-reflections.
    - The ray still inherits wall color on real bounces.
    - Once the arrow has spent its encoded bounce budget, later side-touch walls are ignored
      and the ray continues in a straight diagonal line.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            info = _detect_arrow(pts)
            if info is None:
                continue
            direction, start, max_bounces = info
            arrows.append((int(col), start, direction, int(max_bounces)))
            arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, cur, direction, max_bounces in arrows:
        dr, dc = map(int, direction)
        cur = (int(cur[0]), int(cur[1]))
        cur_col = int(start_col)
        bounces_used = 0
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), int(bounces_used))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                if bounces_used >= max_bounces:
                    break
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                bounces_used += 1
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if bounces_used < max_bounces:
                if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                    wcol = int(g[nr + 1, nc])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dr = -int(dr)
                    bounces_used += 1
                elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                    wcol = int(g[nr - 1, nc])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dr = -int(dr)
                    bounces_used += 1

            if bounces_used < max_bounces:
                if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                    wcol = int(g[nr, nc + 1])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dc = -int(dc)
                    bounces_used += 1
                elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                    wcol = int(g[nr, nc - 1])
                    if int(g[nr, nc]) == int(bg):
                        out[nr, nc] = int(wcol)
                    cur_col = int(wcol)
                    dc = -int(dc)
                    bounces_used += 1

    return out


def _make_arrow_points(
    start: tuple[int, int], dr: int, dc: int, extra_row: bool, extra_col: bool
) -> list[tuple[int, int]]:
    sr, sc = map(int, start)
    pts = {(sr, sc), (sr - dr, sc), (sr, sc - dc)}
    if extra_row:
        pts.add((sr - 2 * dr, sc))
    if extra_col:
        pts.add((sr, sc - 2 * dc))
    return [(int(r), int(c)) for r, c in sorted(pts)]


def _wall_options(H: int, W: int, cur_r: int, cur_c: int, dr: int, dc: int):
    opts: list[tuple[list[tuple[int, int]], str]] = []

    wr, wc = cur_r + dr, cur_c
    if 0 <= wr < H:
        opts.append(([(wr, wc)], "dr"))
        w3 = [(wr, wc + d) for d in (-1, 0, 1)]
        if all(0 <= c < W for _, c in w3):
            opts.append((w3, "dr"))

    wr2, wc2 = cur_r, cur_c + dc
    if 0 <= wc2 < W:
        opts.append(([(wr2, wc2)], "dc"))
        w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
        if all(0 <= r < H for r, _ in w3v):
            opts.append((w3v, "dc"))

    _rng.shuffle(opts)
    prefer_long = _rng.random() < 0.6
    opts.sort(key=lambda item: len(item[0]), reverse=prefer_long)
    return opts


def _pick_wall_color(
    cells: list[tuple[int, int]],
    current_color: int,
    grid: np.ndarray,
    local_colors: dict[tuple[int, int], int],
    palette: list[int],
) -> int | None:
    blocked = {int(current_color)}
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in neighbors4(int(r), int(c)):
            if (nr, nc) in cell_set:
                continue
            if (nr, nc) in local_colors:
                blocked.add(int(local_colors[(nr, nc)]))
            elif 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) != 0:
                blocked.add(int(grid[nr, nc]))

    choices = [int(col) for col in palette if int(col) not in blocked]
    if not choices:
        return None
    return int(_rng.choice(choices))


def _simulate_arrow_plan(
    H: int,
    W: int,
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    arrow_pts: list[tuple[int, int]],
    start: tuple[int, int],
    direction: tuple[int, int],
    arrow_color: int,
    max_bounces: int,
    total_wall_hits: int,
    palette: list[int],
) -> tuple[list[tuple[int, int, int]], list[tuple[list[tuple[int, int]], int]]] | None:
    arrow_set = set(arrow_pts)
    local_colors: dict[tuple[int, int], int] = {cell: int(arrow_color) for cell in arrow_pts}

    def search(
        cur_r: int,
        cur_c: int,
        cur_dr: int,
        cur_dc: int,
        cur_col: int,
        hits_done: int,
        bounces_used: int,
        ray_cells: list[tuple[int, int, int]],
        ray_set: set[tuple[int, int]],
        walls: list[tuple[list[tuple[int, int]], int]],
        colors_map: dict[tuple[int, int], int],
    ):
        if hits_done == total_wall_hits:
            tail_cells = list(ray_cells)
            tail_set = set(ray_set)
            rr, cc = int(cur_r), int(cur_c)
            while True:
                nr, nc = rr + cur_dr, cc + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if (nr, nc) in occupied or (nr, nc) in tail_set or (nr, nc) in colors_map:
                    return None
                tail_cells.append((int(nr), int(nc), int(cur_col)))
                tail_set.add((int(nr), int(nc)))
                rr, cc = int(nr), int(nc)
            if len(tail_cells) < 4:
                return None
            return tail_cells, walls

        max_steps = min((H - 1 - cur_r) if cur_dr == 1 else cur_r, (W - 1 - cur_c) if cur_dc == 1 else cur_c)
        if max_steps <= 0:
            return None

        lengths = list(range(1, max_steps + 1))
        _rng.shuffle(lengths)
        lengths.sort(key=lambda n: abs(n - 3))

        active_bounce = bounces_used < max_bounces

        for seg_len in lengths:
            new_ray_cells = list(ray_cells)
            new_ray_set = set(ray_set)
            rr, cc = int(cur_r), int(cur_c)
            ok = True
            for _ in range(seg_len):
                nr, nc = rr + cur_dr, cc + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                if (nr, nc) in occupied or (nr, nc) in new_ray_set or (nr, nc) in colors_map:
                    ok = False
                    break
                new_ray_cells.append((int(nr), int(nc), int(cur_col)))
                new_ray_set.add((int(nr), int(nc)))
                rr, cc = int(nr), int(nc)
            if not ok:
                continue

            for wpxs, flip in _wall_options(H, W, rr, cc, cur_dr, cur_dc):
                if any((r, c) in occupied or (r, c) in new_ray_set or (r, c) in colors_map for r, c in wpxs):
                    continue
                wcol = _pick_wall_color(wpxs, cur_col, grid, colors_map, palette)
                if wcol is None:
                    continue

                next_ray_cells = list(new_ray_cells)
                next_walls = list(walls)
                next_walls.append(([(int(r), int(c)) for r, c in wpxs], int(wcol)))
                next_colors = dict(colors_map)
                for r, c in wpxs:
                    next_colors[(int(r), int(c))] = int(wcol)

                next_dr, next_dc = int(cur_dr), int(cur_dc)
                next_col = int(cur_col)
                next_bounces = int(bounces_used)
                if active_bounce:
                    lr, lc, _ = next_ray_cells[-1]
                    next_ray_cells[-1] = (int(lr), int(lc), int(wcol))
                    next_col = int(wcol)
                    next_bounces += 1
                    if flip == "dr":
                        next_dr = -int(next_dr)
                    else:
                        next_dc = -int(next_dc)

                result = search(
                    rr,
                    cc,
                    next_dr,
                    next_dc,
                    next_col,
                    hits_done + 1,
                    next_bounces,
                    next_ray_cells,
                    new_ray_set,
                    next_walls,
                    next_colors,
                )
                if result is not None:
                    return result

        return None

    return search(
        int(start[0]),
        int(start[1]),
        int(direction[0]),
        int(direction[1]),
        int(arrow_color),
        0,
        0,
        [],
        set(),
        [],
        local_colors,
    )


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input for the D12 variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 5)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    size_cycle = [3, 4, 5]
    _rng.shuffle(size_cycle)
    arrow_sizes = [size_cycle[i] if i < len(size_cycle) else _rng.choice([3, 4, 5]) for i in range(n_arrows)]
    _rng.shuffle(arrow_sizes)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for ai in range(n_arrows):
        arrow_color = int(arrow_colors[ai])
        arrow_size = int(arrow_sizes[ai])
        max_bounces = arrow_size - 3
        extra_hits = _rng.choice([1, 2])
        total_wall_hits = max_bounces + extra_hits

        placed = False
        for _ in range(120):
            dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
            if arrow_size == 3:
                extra_row, extra_col = False, False
            elif arrow_size == 4:
                extra_row, extra_col = _rng.choice([(True, False), (False, True)])
            else:
                extra_row, extra_col = True, True

            sr = _rng.randint(0, H - 1)
            sc = _rng.randint(0, W - 1)
            arrow_pts = _make_arrow_points((sr, sc), int(dr), int(dc), extra_row, extra_col)
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            sim = _simulate_arrow_plan(
                H=H,
                W=W,
                grid=grid,
                occupied=occupied,
                arrow_pts=arrow_pts,
                start=(sr, sc),
                direction=(int(dr), int(dc)),
                arrow_color=arrow_color,
                max_bounces=max_bounces,
                total_wall_hits=total_wall_hits,
                palette=palette,
            )
            if sim is None:
                continue

            ray_cells, walls = sim
            all_new = set(arrow_pts) | {(int(r), int(c)) for r, c, _ in ray_cells}
            for wpxs, _ in walls:
                all_new.update((int(r), int(c)) for r, c in wpxs)

            expected = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(all_new) != expected or all_new & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(arrow_color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)
            occupied |= all_new
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D12 variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
