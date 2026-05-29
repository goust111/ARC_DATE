# Combo variant of 142ca369 (index 2)
# combo_spec: D2/1+D4/0
# Name: 颜色转向+颜色混合
#
# Design analysis for this combo variant
# --------------------------------------
# a) 具体化与行为追踪
#    - 保留原题物体：背景 0；每个箭头仍是 2x2 外接框里缺一角的 3 像素 L 形；墙仍是 1 像素或 3 像素轴对齐线段。
#    - 融合后的墙色只用两类规则色：
#        8 -> 顺时针转 90 度
#        9 -> 逆时针转 90 度
#      这继承 D2/1 的“颜色决定转向方向”。
#    - 同时引入 D4/0 的“颜色混合”：射线颜色不再继承墙色，而是在一个小循环色集
#        [1, 2, 4]
#      中根据墙色做循环位移：
#        current 经过墙 8 后前进一格：1->2, 2->4, 4->1
#        current 经过墙 9 后后退一格：1->4, 4->2, 2->1
#      因此“新颜色”同时依赖当前射线颜色和墙色，而不是只看墙色。
#    - 具体例子：当前方向 (dr, dc) = (1, 1)，当前颜色 = 1。
#      若接触颜色 8 的墙，则方向变成 (1, -1)，颜色变成 2。
#      若接触颜色 9 的墙，则方向变成 (-1, 1)，颜色变成 4。
#      与原题不同：原题只会做镜面反射且颜色直接变成墙色；这里路径和颜色都由墙色参与控制，
#      但颜色结果又保留了当前射线颜色的记忆。
#
# b) 复杂度校准
#    - 转向规则只保留两档（8/9），人类可以从少量样例中归纳。
#    - 混色只在 3 个轨迹色 [1,2,4] 上循环，映射表大小为 3x2，足够有结构又不会太大。
#    - 每个样例强制出现 5-8 次总接触，并要求两种墙色都重复出现，且同一墙色要搭配至少两种入射色，
#      避免“看上去只是固定替换成某个颜色”。
#
# c) 对原 solver 隐式假设的审查
#    - 原 `if (nr, nc) in wall_cells: dr, dc = -dr, -dc` 假设正面撞墙必定双轴反转。这里失效；
#      现在正面接触也改成“按墙色转向 + 按当前色与墙色混色”。虽然 generator 不构造正面撞墙，
#      solver 仍兼容它。
#    - 原 side-touch 逻辑依赖“接触到行侧墙就翻 dr，接触到列侧墙就翻 dc”。这里失效；
#      接触几何只用来找到是哪一块墙被碰到，真正方向更新交给墙色规则。
#    - 原颜色更新 `cur_col = wall_color` 失效；现在必须先读取当前射线颜色，再做混色。
#    - 原代码允许同一步同时读到两侧墙。镜面模型还能解释，颜色转向模型会产生歧义。
#      因此 generator 保证每个接触点最多只有一个可触达墙；solver 仍保留固定优先级：
#      先看行方向侧，再看列方向侧。
#
# d) 放置合法性
#    - 继续用 output-first：先规划每条射线的颜色序列与折线路径，再反推出输入中的箭头和墙。
#    - 墙、箭头、射线轨迹都不能重叠；同色墙还不能 4 邻接，避免同色墙拼成 3 像素 L 形连通块，
#      被箭头检测误识别。
#    - 每次接触后，转向后的新方向必须至少还能走 2 格，避免刚转向就贴墙或立即出界。
#
# e) 视觉可观察性
#    - 仅做“颜色决定转向”还可能在视觉上退化成原题镜面反射，因此 generator 强制把墙放在
#      “若按原题镜面规则会得到另一条路径”的那一侧。
#    - 仅做“颜色混合”若混色结果等于墙色，也会看起来像原题继承颜色，因此轨迹色集完全避开墙色 8/9，
#      且混色结果总与当前色、墙色都不同。
#    - 这样解题者可以同时观察到：同样几何侧位的墙，因颜色不同而转向不同；同样颜色的墙，面对不同入射色
#      时会得到不同的新轨迹色。
#
# f) 可视化整洁度
#    - 网格仍保持 18-20 方阵、3-4 个箭头、每箭头 1-3 次接触。
#    - 墙只在接触点旁 1 格，轨迹仍是稀疏对角折线，不引入额外标记层。
#    - 轨迹色只用 1/2/4，墙色只用 8/9，箭头也只从 1/2/4 取色，视觉角色分层清楚。
#
# g) 变化的可推断性
#    - 转向映射：8 表示顺时针，9 表示逆时针。
#    - 混色映射：在 [1,2,4] 上，8 表示 +1 循环位移，9 表示 -1 循环位移。
#    - generator 额外过滤样本，要求：
#        * 两种墙色都至少出现 2 次；
#        * 每种墙色都要搭配至少两种入射轨迹色；
#        * 至少有一种入射轨迹色分别碰到过两种墙色；
#      这样 2-4 个样例就能支撑人类归纳“转向规则”和“混色规则”两套映射。
#
# 9) main() 的新隐式假设与 generator 对应保证
#    - 假设 1：非背景、非箭头像素都是规则墙色 8/9。generator 保证。
#    - 假设 2：轨迹颜色处于小循环集 [1,2,4]。generator 保证箭头初色如此，混色也闭包在此集合里。
#    - 假设 3：每个接触点最多只有一个可触达墙。generator 保证。
#    - 假设 4：同色墙之间不会 4 邻接。generator 保证，避免墙被误判成箭头。
#    - 假设 5：所有已规划轨迹都能离开网格，不会与别的箭头或墙共享像素。generator 的占用检查、
#      局部可达检查和最终 oracle 检查共同保证。

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg


CW_WALL_COLOR = 8
CCW_WALL_COLOR = 9
RULE_WALL_COLORS = (CW_WALL_COLOR, CCW_WALL_COLOR)

RAY_CYCLE = (1, 2, 4)
RAY_INDEX = {int(col): idx for idx, col in enumerate(RAY_CYCLE)}


def _rotate_by_color(dr: int, dc: int, wall_color: int) -> tuple[int, int]:
    if int(wall_color) == CW_WALL_COLOR:
        return int(dc), -int(dr)
    if int(wall_color) == CCW_WALL_COLOR:
        return -int(dc), int(dr)
    return int(dr), int(dc)


def _mix_color(ray_color: int, wall_color: int) -> int:
    idx = RAY_INDEX.get(int(ray_color))
    if idx is None:
        return int(ray_color)
    if int(wall_color) == CW_WALL_COLOR:
        return int(RAY_CYCLE[(idx + 1) % len(RAY_CYCLE)])
    if int(wall_color) == CCW_WALL_COLOR:
        return int(RAY_CYCLE[(idx - 1) % len(RAY_CYCLE)])
    return int(ray_color)


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

    Combo rule:
    - Wall color 8 turns the ray clockwise by 90 degrees.
    - Wall color 9 turns the ray counterclockwise by 90 degrees.
    - On every contact, the new ray color is the mix of current ray color and wall color:
        [1,2,4] cycles forward on wall 8 and backward on wall 9.
    - Contact geometry only identifies which wall was touched; it no longer determines reflection.
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

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                mixed = _mix_color(int(cur_col), int(wcol))
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(mixed)
                cur_col = int(mixed)
                dr, dc = _rotate_by_color(int(dr), int(dc), int(wcol))
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            contact: tuple[int, int] | None = None
            if (nr + int(dr), nc) in wall_cells:
                contact = (nr + int(dr), nc)
            elif (nr, nc + int(dc)) in wall_cells:
                contact = (nr, nc + int(dc))

            if contact is None:
                continue

            wr, wc = contact
            wcol = int(g[wr, wc])
            mixed = _mix_color(int(cur_col), int(wcol))
            if int(g[nr, nc]) == int(bg):
                out[nr, nc] = int(mixed)
            cur_col = int(mixed)
            dr, dc = _rotate_by_color(int(dr), int(dc), int(wcol))

    return out


def _same_color_clear(
    cells: list[tuple[int, int]],
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
    turned = _rotate_by_color(int(dr), int(dc), int(wall_color))
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
    preferred_side = _mismatch_side(int(dr), int(dc), int(wall_color))
    options: list[tuple[list[tuple[int, int]], str]] = []

    if preferred_side == "row":
        wr, wc = cur_r + dr, cur_c
        if 0 <= wr < H and 0 <= wc < W:
            options.append(([(int(wr), int(wc))], "row"))
            line = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
            if all(0 <= c < W for _r, c in line):
                options.append((line, "row"))
    else:
        wr, wc = cur_r, cur_c + dc
        if 0 <= wr < H and 0 <= wc < W:
            options.append(([(int(wr), int(wc))], "col"))
            line = [(int(wr + d), int(wc)) for d in (-1, 0, 1)]
            if all(0 <= r < H for r, _c in line):
                options.append((line, "col"))

    if not options:
        return None

    _rng.shuffle(options)
    options.sort(key=lambda item: len(item[0]), reverse=(_rng.random() < 0.6))
    for cells, side in options:
        if any(cell in blocked for cell in cells):
            continue
        if not _same_color_clear(cells, global_same_color, local_same_color):
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
    for _ in range(int(seg_len)):
        rr += int(dr)
        cc += int(dc)
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        if (int(rr), int(cc)) in blocked:
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
    for _ in range(160):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        mr, mc = sr - dr, sc - dc
        arrow_pts = [(int(r), int(c)) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = int(sr), int(sc)
        cur_dr, cur_dc = int(dr), int(dc)
        cur_col = int(color)
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[tuple[int, int]], int]] = []
        ray_set: set[tuple[int, int]] = set()
        local_wall_cells: set[tuple[int, int]] = set()
        local_wall_color_cells = {CW_WALL_COLOR: set(), CCW_WALL_COLOR: set()}
        local_forced_used = 0
        ok = True

        for _turn_idx in range(int(n_turns)):
            blocked = occupied | set(arrow_pts) | ray_set | local_wall_cells
            max_steps = _max_diagonal_steps(cur_r, cur_c, cur_dr, cur_dc, H, W)
            if max_steps < 2:
                ok = False
                break

            turn_color = (
                int(forced_colors[local_forced_used])
                if local_forced_used < len(forced_colors)
                else int(_rng.choice(RULE_WALL_COLORS))
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

                wall_cells, _side = wall_choice
                next_dr, next_dc = _rotate_by_color(cur_dr, cur_dc, turn_color)
                turn_blocked = blocked | set(seg_cells) | set(wall_cells)
                if _advance_cells(end_r, end_c, next_dr, next_dc, 2, turn_blocked, H, W) is None:
                    continue

                for rr, cc in seg_cells:
                    ray_cells.append((int(rr), int(cc), int(cur_col)))
                    ray_set.add((int(rr), int(cc)))
                walls.append((wall_cells, int(turn_color)))
                local_wall_cells.update(wall_cells)
                local_wall_color_cells[turn_color].update(wall_cells)

                mixed = _mix_color(int(cur_col), int(turn_color))
                lr, lc, _old = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(mixed))
                cur_r, cur_c = int(end_r), int(end_c)
                cur_dr, cur_dc = int(next_dr), int(next_dc)
                cur_col = int(mixed)
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


def _trace_events(grid: np.ndarray) -> list[dict[str, object]]:
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
                mixed = _mix_color(int(cur_col), int(wcol))
                turned = _rotate_by_color(int(dr), int(dc), int(wcol))
                events.append(
                    {
                        "kind": "front",
                        "incoming_color": int(cur_col),
                        "wall_color": int(wcol),
                        "mixed_color": int(mixed),
                        "matches_old_rule": turned == (-int(dr), -int(dc)),
                    }
                )
                cur_col = int(mixed)
                dr, dc = turned
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if (nr + int(dr), nc) in wall_cells:
                side = "row"
                wr, wc = nr + int(dr), nc
            elif (nr, nc + int(dc)) in wall_cells:
                side = "col"
                wr, wc = nr, nc + int(dc)
            else:
                continue

            wcol = int(g[wr, wc])
            mixed = _mix_color(int(cur_col), int(wcol))
            turned = _rotate_by_color(int(dr), int(dc), int(wcol))
            old_result = (-int(dr), int(dc)) if side == "row" else (int(dr), -int(dc))
            events.append(
                {
                    "kind": "side",
                    "side": side,
                    "incoming_color": int(cur_col),
                    "wall_color": int(wcol),
                    "mixed_color": int(mixed),
                    "matches_old_rule": turned == old_result,
                }
            )
            cur_col = int(mixed)
            dr, dc = turned

    return events


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    events = _trace_events(grid)
    side_events = [e for e in events if e["kind"] == "side"]
    wall_to_inputs = {
        int(wcol): {int(e["incoming_color"]) for e in events if int(e["wall_color"]) == int(wcol)}
        for wcol in RULE_WALL_COLORS
    }
    input_to_walls: dict[int, set[int]] = {}
    for e in events:
        input_to_walls.setdefault(int(e["incoming_color"]), set()).add(int(e["wall_color"]))

    return {
        "event_count": len(events),
        "side_event_count": len(side_events),
        "matches_original_side_rule": sum(int(bool(e["matches_old_rule"])) for e in side_events),
        "mixed_not_equal_old_or_wall": sum(
            int(int(e["mixed_color"]) != int(e["incoming_color"]) and int(e["mixed_color"]) != int(e["wall_color"]))
            for e in events
        ),
        "wall_use_counts": {
            int(wcol): sum(int(int(e["wall_color"]) == int(wcol)) for e in events) for wcol in RULE_WALL_COLORS
        },
        "wall_to_input_variety": {int(w): len(vals) for w, vals in wall_to_inputs.items()},
        "max_wall_variety_per_input_color": max((len(vals) for vals in input_to_walls.values()), default=0),
        "distinct_mix_triples": len(
            {
                (int(e["incoming_color"]), int(e["wall_color"]), int(e["mixed_color"]))
                for e in events
            }
        ),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(3, 4)
    arrow_colors = list(RAY_CYCLE)
    _rng.shuffle(arrow_colors)
    while len(arrow_colors) < n_arrows:
        arrow_colors.append(int(_rng.choice(RAY_CYCLE)))

    target_total_turns = _rng.randint(5, min(8, n_arrows * 3))
    turn_counts = [1] * n_arrows
    for _ in range(target_total_turns - n_arrows):
        candidates = [i for i, cnt in enumerate(turn_counts) if cnt < 3]
        if not candidates:
            break
        turn_counts[int(_rng.choice(candidates))] += 1

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    wall_color_cells_global = {CW_WALL_COLOR: set(), CCW_WALL_COLOR: set()}
    forced_colors = [CW_WALL_COLOR, CCW_WALL_COLOR, CW_WALL_COLOR, CCW_WALL_COLOR, CW_WALL_COLOR, CCW_WALL_COLOR]

    for color, n_turns in zip(arrow_colors, turn_counts):
        planned = _plan_arrow(H, W, int(color), int(n_turns), occupied, wall_color_cells_global, forced_colors)
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

    metrics = _behavior_metrics(grid)
    if int(metrics["event_count"]) < 5:
        return None
    if int(metrics["side_event_count"]) != int(metrics["event_count"]):
        return None
    if int(metrics["matches_original_side_rule"]) != 0:
        return None
    if int(metrics["mixed_not_equal_old_or_wall"]) != int(metrics["event_count"]):
        return None
    if int(metrics["distinct_mix_triples"]) < 4:
        return None
    if any(int(metrics["wall_use_counts"][w]) < 2 for w in RULE_WALL_COLORS):
        return None
    if any(int(metrics["wall_to_input_variety"][w]) < 2 for w in RULE_WALL_COLORS):
        return None
    if int(metrics["max_wall_variety_per_input_color"]) < 2:
        return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
