# Variant of 142ca369
# Dimension: D2 — 反射机制 (path)
# Change: 侧面接触反射（side-touch reflection）——射线在前进方向的邻近轴上检测到墙壁时，
#         翻转对应轴的方向分量；正面碰撞反转两个轴
#         → 逆时针/顺时针转向反射
#
# Design analysis for this variant
# --------------------------------
# a) 具体化与行为追踪
#    - 保留原题对象：背景 0；3 像素 L 形箭头；墙仍然是单像素或 3 像素轴对齐线段。
#    - 核心变化只发生在“墙接触后的路径更新”：
#      原题：侧碰哪一侧，就翻转哪一轴；正面撞墙则两轴同时翻转。
#      变体：不再根据墙的朝向镜面反射，而是读取墙颜色：
#        color 8 => 顺时针转 90 度
#        color 9 => 逆时针转 90 度
#      射线仍沿 4 个对角方向之一移动，因此 90 度转向后仍是对角线，只是“哪一轴被翻转”
#      不再由墙的朝向决定，而是由颜色决定。
#    - 小例子：当前方向 (dr, dc) = (1, 1)。
#      原题若墙在 (r+1, c) 这一侧，则新方向为 (-1, 1)；若墙在 (r, c+1) 这一侧，则新方向为 (1, -1)。
#      变体若接触 color 8 墙，则新方向固定为 (1, -1)；接触 color 9 墙，则固定为 (-1, 1)。
#      因而同样的“下侧墙”在不同颜色下会给出不同结果；同色墙在不同朝向下也能给出相同转向规则。
#
# b) 复杂度校准
#    - 只使用两种规则墙色（8, 9），把自由度控制在人类可从 2-4 个样例中归纳的范围内。
#    - 每个生成样例都要求至少 4 次总转向，并且两种规则色都重复出现，避免“只看见一次，无法归纳”。
#    - 墙仍保留单像素 / 3 像素线段两种外观，保证视觉上仍像原任务家族，而不是引入新对象类型。
#
# c) 对原 solver 旧假设的逐段审查
#    - 原代码里 `if (nr, nc) in wall_cells: dr, dc = -dr, -dc` 依赖“正面撞墙=双轴反转”。该前提已失效，
#      现在改为根据墙色做 CW/CCW 转向。
#    - 原代码里 side-touch 的两段：
#        if wall at (nr+dr, nc): dr = -dr
#        if wall at (nr, nc+dc): dc = -dc
#      依赖“墙的几何位置决定翻哪一轴”。该前提已失效；现在只把这些位置当作“接触检测位置”，
#      真正的转向由墙色决定。
#    - 原代码顺序上允许同一步同时触发两次 side-touch；在镜面模型下这等价于更复杂的反射。
#      变体里若同时读到两个墙色会产生歧义，所以新 generator 保证每个转向点最多只有一个可接触墙；
#      solver 仍保留固定优先顺序（先行方向侧，再列方向侧）作为容错。
#    - 原代码默认墙色可以任意继承。变体里 solver 新增了“规则墙色集合”的假设：墙应使用 8/9。
#      generator 会严格保证这一点。
#
# d) 放置合法性
#    - 仍采用 output-first：先规划每条射线的整条路径和每次转向，再反推出输入中的箭头和墙。
#    - 每次转向点只允许放置一个接触墙；墙与箭头、墙与路径、路径与路径都不能重叠。
#    - 同色墙之间禁止 4 邻接，避免多个同色墙拼成一个 3 像素 L 连通块，被箭头检测误识别。
#    - 生成时显式维护占用集，并在提交前用 solver 回放校验 `main(input) == output`。
#
# e) 视觉可观察性
#    - 仅靠“转 90 度”还不够，因为对角方向上的 90 度转向在代数上总是等价于“翻一根轴”；
#      如果墙恰好放在会产生同样结果的那一侧，视觉上就会退化成原题。
#    - 因此 generator 额外强制：每次墙的放置侧都选成“若按原题镜面规则会走另一条路”的那一侧。
#      这样每一次转向事件都与原题在同一输入上的结果不同，变体真正可见。
#
# f) 可视化整洁度
#    - 维持原题稀疏布局：18-20 方阵、2-4 个箭头、每箭头 1-3 次转向。
#    - 墙只出现在转向点旁边 1 格，整体仍是几条清晰的对角折线路径，不引入额外标记层。
#
# g) 颜色变化可推断性
#    - 颜色映射只发生在墙色 -> 转向方向这一处，且只有两类：
#        8: clockwise
#        9: counterclockwise
#    - 两种颜色都会在每个样例中重复出现，且 generator 让同一种颜色能搭配不同墙朝向，
#      从而突出“决定因素是颜色而不是墙的几何方向”。
#
# 9) main() 的新隐式假设与 generator 对应保证
#    - 假设 1：非背景、非箭头像素都是规则墙，颜色在 {8, 9}。generator 保证。
#    - 假设 2：每个转向点最多只有一个可接触墙。generator 保证，不构造双接触歧义。
#    - 假设 3：同色墙不会拼成一个 3 像素 L 形连通块。generator 用“同色墙不可 4 邻接”保证。
#    - 假设 4：箭头之间、墙之间、路径之间都不共享像素。generator 的占用检查和最终 oracle 检查保证。

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


CW_WALL_COLOR = 8
CCW_WALL_COLOR = 9
RULE_WALL_COLORS = (CW_WALL_COLOR, CCW_WALL_COLOR)


def _rotate_by_color(dr: int, dc: int, wall_color: int) -> tuple[int, int]:
    if int(wall_color) == CW_WALL_COLOR:
        return int(dc), -int(dr)
    return -int(dc), int(dr)


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
    - Rays still move diagonally and inherit wall color at contacts.
    - A wall no longer causes mirror reflection.
    - Wall color 8 turns the direction clockwise by 90 degrees.
    - Wall color 9 turns the direction counterclockwise by 90 degrees.
    - The contact geometry (front hit or side-touch) only tells us which wall was touched,
      not how to reflect.
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

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
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

            # Front contact: turn at the current cell, using the wall color as the turn rule.
            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = _rotate_by_color(int(dr), int(dc), int(wcol))
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            contacts: list[tuple[int, int]] = []
            if (nr + int(dr), nc) in wall_cells:
                contacts.append((nr + int(dr), nc))
            if (nr, nc + int(dc)) in wall_cells:
                contacts.append((nr, nc + int(dc)))

            if contacts:
                wr, wc = contacts[0]
                wcol = int(g[wr, wc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr, dc = _rotate_by_color(int(dr), int(dc), int(wcol))

    return out


def _same_color_clear(
    cells: list[tuple[int, int]],
    color: int,
    global_same_color: set[tuple[int, int]],
    local_same_color: set[tuple[int, int]],
) -> bool:
    occupied = global_same_color | local_same_color
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in occupied:
                return False
    return True


def _mismatch_side(dr: int, dc: int, wall_color: int) -> str:
    turned = _rotate_by_color(dr, dc, wall_color)
    row_reflect = (-int(dr), int(dc))
    col_reflect = (int(dr), -int(dc))
    if turned == row_reflect and turned != col_reflect:
        return "col"
    if turned == col_reflect and turned != row_reflect:
        return "row"
    return "row"


def _pick_wall_cells(
    H: int,
    W: int,
    cur_r: int,
    cur_c: int,
    dr: int,
    dc: int,
    wall_color: int,
    blocked: set[tuple[int, int]],
    global_same_color: set[tuple[int, int]],
    local_same_color: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str] | None:
    preferred_side = _mismatch_side(dr, dc, wall_color)
    options: list[tuple[list[tuple[int, int]], str]] = []

    if preferred_side == "row":
        wr, wc = cur_r + dr, cur_c
        if 0 <= wr < H and 0 <= wc < W:
            options.append(([(wr, wc)], "row"))
            line = [(wr, wc + d) for d in (-1, 0, 1)]
            if all(0 <= c < W for _r, c in line):
                options.append((line, "row"))
    else:
        wr, wc = cur_r, cur_c + dc
        if 0 <= wr < H and 0 <= wc < W:
            options.append(([(wr, wc)], "col"))
            line = [(wr + d, wc) for d in (-1, 0, 1)]
            if all(0 <= r < H for r, _c in line):
                options.append((line, "col"))

    if not options:
        return None

    _rng.shuffle(options)
    options.sort(key=lambda item: len(item[0]), reverse=(_rng.random() < 0.6))
    for cells, side in options:
        if any(cell in blocked for cell in cells):
            continue
        if not _same_color_clear(cells, wall_color, global_same_color, local_same_color):
            continue
        return cells, side
    return None


def _advance_cells(
    cur_r: int,
    cur_c: int,
    dr: int,
    dc: int,
    seg_len: int,
    blocked: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    rr, cc = cur_r, cur_c
    cells: list[tuple[int, int]] = []
    for _ in range(seg_len):
        rr += dr
        cc += dc
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        if (rr, cc) in blocked:
            return None
        cells.append((int(rr), int(cc)))
    return cells


def _max_diagonal_steps(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _plan_arrow(
    H: int,
    W: int,
    color: int,
    n_turns: int,
    occupied: set[tuple[int, int]],
    wall_color_cells_global: dict[int, set[tuple[int, int]]],
    forced_colors: list[int],
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int, int]],
    list[tuple[list[tuple[int, int]], int]],
    int,
] | None:
    for _ in range(140):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        mr, mc = sr - dr, sc - dc
        arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = sr, sc
        cur_dr, cur_dc = dr, dc
        cur_col = color
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[tuple[int, int]], int]] = []
        ray_set: set[tuple[int, int]] = set()
        local_wall_cells: set[tuple[int, int]] = set()
        local_wall_color_cells = {CW_WALL_COLOR: set(), CCW_WALL_COLOR: set()}
        local_forced_used = 0
        ok = True

        for turn_idx in range(n_turns):
            blocked = occupied | set(arrow_pts) | ray_set | local_wall_cells
            max_steps = _max_diagonal_steps(cur_r, cur_c, cur_dr, cur_dc, H, W)
            if max_steps < 2:
                ok = False
                break

            turn_color = (
                forced_colors[local_forced_used]
                if local_forced_used < len(forced_colors)
                else _rng.choice(RULE_WALL_COLORS)
            )
            candidate_lengths = list(range(2, max_steps + 1))
            _rng.shuffle(candidate_lengths)
            candidate_lengths.sort(key=lambda n: abs(n - max_steps // 2))

            committed = False
            for seg_len in candidate_lengths:
                seg_cells = _advance_cells(cur_r, cur_c, cur_dr, cur_dc, seg_len, blocked, H, W)
                if seg_cells is None:
                    continue
                end_r, end_c = seg_cells[-1]
                wall_choice = _pick_wall_cells(
                    H,
                    W,
                    end_r,
                    end_c,
                    cur_dr,
                    cur_dc,
                    turn_color,
                    blocked | set(seg_cells),
                    wall_color_cells_global[turn_color],
                    local_wall_color_cells[turn_color],
                )
                if wall_choice is None:
                    continue

                next_dr, next_dc = _rotate_by_color(cur_dr, cur_dc, turn_color)
                if _max_diagonal_steps(end_r, end_c, next_dr, next_dc, H, W) < 2:
                    continue

                for rr, cc in seg_cells:
                    ray_cells.append((int(rr), int(cc), int(cur_col)))
                    ray_set.add((int(rr), int(cc)))
                wall_cells, _side = wall_choice
                walls.append((wall_cells, int(turn_color)))
                local_wall_cells.update(wall_cells)
                local_wall_color_cells[turn_color].update(wall_cells)
                lr, lc, _old = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(turn_color))
                cur_r, cur_c = end_r, end_c
                cur_dr, cur_dc = next_dr, next_dc
                cur_col = int(turn_color)
                local_forced_used += 1
                committed = True
                break

            if not committed:
                ok = False
                break

        if not ok:
            continue

        blocked = occupied | set(arrow_pts) | ray_set | local_wall_cells
        tail_steps = _max_diagonal_steps(cur_r, cur_c, cur_dr, cur_dc, H, W)
        if tail_steps < 2:
            continue
        tail_cells = _advance_cells(cur_r, cur_c, cur_dr, cur_dc, tail_steps, blocked, H, W)
        if tail_cells is None:
            continue
        for rr, cc in tail_cells:
            ray_cells.append((int(rr), int(cc), int(cur_col)))
            ray_set.add((int(rr), int(cc)))

        if len(ray_cells) < 5:
            continue

        new_cells = set(arrow_pts) | ray_set | local_wall_cells
        expected = len(arrow_pts) + len(ray_set) + len(local_wall_cells)
        if len(new_cells) != expected:
            continue
        if new_cells & occupied:
            continue

        return arrow_pts, ray_cells, walls, local_forced_used

    return None


def _trace_turn_events(grid: np.ndarray) -> list[dict[str, object]]:
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

    wall_cells = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }
    events: list[dict[str, object]] = []

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int, int]] = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)
            nr, nc = int(cur[0] + dr), int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                events.append(
                    {
                        "kind": "front",
                        "incoming": (int(dr), int(dc)),
                        "wall_color": wcol,
                    }
                )
                cur_col = wcol
                dr, dc = _rotate_by_color(int(dr), int(dc), int(wcol))
                continue
            if (nr, nc) in arrow_cells:
                break
            cur = (int(nr), int(nc))
            side = None
            if (nr + int(dr), nc) in wall_cells:
                side = "row"
                wr, wc = nr + int(dr), nc
            elif (nr, nc + int(dc)) in wall_cells:
                side = "col"
                wr, wc = nr, nc + int(dc)
            else:
                continue
            wcol = int(g[wr, wc])
            turned = _rotate_by_color(int(dr), int(dc), int(wcol))
            old_result = (-int(dr), int(dc)) if side == "row" else (int(dr), -int(dc))
            events.append(
                {
                    "kind": "side",
                    "side": side,
                    "incoming": (int(dr), int(dc)),
                    "wall_color": wcol,
                    "matches_old_rule": turned == old_result,
                }
            )
            cur_col = wcol
            dr, dc = turned

    return events


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 4)
    arrow_palette = list(range(1, 8))
    _rng.shuffle(arrow_palette)
    arrow_colors = arrow_palette[:n_arrows]

    target_total_turns = _rng.randint(4, min(8, n_arrows * 3))
    turn_counts = [1] * n_arrows
    for _ in range(target_total_turns - n_arrows):
        candidates = [i for i, cnt in enumerate(turn_counts) if cnt < 3]
        if not candidates:
            break
        turn_counts[_rng.choice(candidates)] += 1

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    wall_color_cells_global = {CW_WALL_COLOR: set(), CCW_WALL_COLOR: set()}
    forced_colors = [CW_WALL_COLOR, CCW_WALL_COLOR, CW_WALL_COLOR, CCW_WALL_COLOR]

    for color, n_turns in zip(arrow_colors, turn_counts):
        planned = _plan_arrow(H, W, color, n_turns, occupied, wall_color_cells_global, forced_colors)
        if planned is None:
            return None
        arrow_pts, ray_cells, walls, forced_used = planned

        for r, c in arrow_pts:
            grid[r, c] = int(color)
            out[r, c] = int(color)
        for r, c, ray_col in ray_cells:
            out[r, c] = int(ray_col)
        for wall_cells, wall_color in walls:
            for r, c in wall_cells:
                grid[r, c] = int(wall_color)
                out[r, c] = int(wall_color)
                wall_color_cells_global[int(wall_color)].add((int(r), int(c)))

        occupied |= set(arrow_pts)
        occupied |= {(int(r), int(c)) for r, c, _col in ray_cells}
        occupied |= {(int(r), int(c)) for wall_cells, _wcol in walls for r, c in wall_cells}
        forced_colors = forced_colors[forced_used:]

    events = _trace_turn_events(grid)
    if len(events) < 4:
        return None
    side_events = [e for e in events if e["kind"] == "side"]
    if not side_events:
        return None
    if any(bool(e["matches_old_rule"]) for e in side_events):
        return None
    used_rule_colors = {int(e["wall_color"]) for e in events}
    if used_rule_colors != {CW_WALL_COLOR, CCW_WALL_COLOR}:
        return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
