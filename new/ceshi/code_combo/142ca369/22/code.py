# Combo variant of 142ca369 (index 22)
# combo_spec: D11/1+D20/0
# Name: 成对箭头+射线碰撞偏转
#
# Design analysis
# ---
# fusion_spec:
#   a) concretization_and_trace:
#     scene:
#       - 输入仍是 18-20 的黑底网格，元素只有 L 形箭头和轴对齐墙。
#       - 与原题不同，同色恰好两枚箭头构成一对；这就是“谁和谁相向”的唯一可见编码。
#       - 一对箭头先沿同一条中心对称折线路径朝中间逼近，路径中仍可发生原题那种 side-touch 墙反射，
#         并在转折格继承墙色。
#       - 当两条对向对角射线在同一步进入同一格时，不再相消，而是在该格继续偏转。
#     concrete_example: |
#       设左上同色箭头以 ↘ 接近中心，右下同色箭头以 ↖ 接近同一点。原题若忽略配对，
#       两条射线会继续沿各自对角线穿过中心并一路出界；D11/1 单变体则会在中心停止并留空。
#       本融合版改为：中心格被涂色一次，然后
#         ↘ 变成 →，↖ 变成 ←；
#       如果碰撞前最后一段是 ↗ / ↙，则改为
#         ↗ 变成 ↑，↙ 变成 ↓。
#       也就是：碰撞前保留原题的对角反射物理；碰撞时把“对角推进”切换为“正交散开”。
#     why_this_is_45deg: |
#       对角方向相对水平/垂直恰好相差 45°。因此把 ↘/↖ 映射为 →/←、把 ↗/↙ 映射为 ↑/↓，
#       是最小且可视的“45 度偏转”实现。
#   b) complexity_calibration:
#     not_too_simple:
#       - 若只做成对箭头而无后续尾迹，规则会退化为 D11/1 的“中点停止”。
#       - 若只做碰撞偏转而不保留墙色继承，则又会退化成一眼看穿的“连线后分叉”。
#       - 因此生成器强制至少一对样本在碰撞前出现 1-2 次镜像墙反射，让解题者同时看到：
#         成对关系、墙致转折、墙色继承、碰撞后正交散开。
#     not_too_complex:
#       - 不引入跨颜色配对、优先级、速度差或多次碰撞。
#       - 每对恰好一次碰撞，碰撞后的规则只有两档：横向散开或纵向散开。
#   c) implicit_assumption_audit:
#     original_line_by_line_breakage:
#       - 原 `for start_col, pts in arrows:` 假设每条射线能独立顺序落笔；融合后不成立，
#         因为同一步进入同一格时需要联合结算碰撞。
#       - 原 `if out[nr, nc] == bg` 假设“已画过的轨迹只影响可见性，不影响物理”；融合后不成立，
#         因为碰撞语义取决于前沿位置而不是 `out` 上是否已着色。
#       - 原代码把箭头只当 blocker；融合后同色箭头还编码了配对关系，因此必须先按颜色分组。
#       - 原任务完全是对角运动；融合后若继续假设 `(dr, dc)` 两轴都非零，就无法表示碰撞后的尾迹。
#     new_assumptions_introduced:
#       - 配对规则：同色且仅两枚箭头构成一对。若输入不满足，solver 回退为原题式单箭头追踪。
#       - 正交尾迹规则：碰撞后的直线尾迹不再与墙反射发生组合；生成器保证碰撞后路径上没有墙或箭头。
#       - 每对至多一次真实碰撞；生成器通过中心对称构造保证这一点。
#   d) placement_legality:
#     generator_guarantees:
#       - 箭头像素、墙像素、碰撞前对角路径、碰撞后正交尾迹在全局 occupied 集上统一判重。
#       - 两半对角路径仅允许在碰撞中心 1 格重合。
#       - 反射墙只放在碰撞前的转折处，并且左右两半镜像同构，保证两条射线同步抵达碰撞格。
#       - 碰撞后的正交尾迹全部在空白走廊中延伸到边界，不穿过任何输入对象或别的 pair 的输出区域。
#   e) visual_observability:
#     human_inference:
#       - 输入直接暴露“同色两箭头是一对”。
#       - 输出同时出现两种显著结构：
#         1. 从两端补向中间的对角线/折线；
#         2. 从中心向左右或上下散开的直线尾迹。
#       - 这种“先对角汇聚、后正交散开”的形态与原题的“单向对角延伸到边界”差异非常大，
#         人眼在 2-3 个样例中即可归纳。
#   f) cleanliness:
#     - 每对占据一个独立区域，墙只贴在对应 pair 的折线转角一侧。
#     - 颜色变化只来自墙色继承；碰撞本身不引入新颜色，因此不会把注意力分散到无结构色映射上。
#   g) inferable_change:
#     - 颜色种类保持紧凑：pair 颜色 1-3 种，墙色从剩余调色板中选，且同一对的镜像墙共用颜色。
#     - 碰撞后尾迹沿用碰撞时当前颜色，所以解题者能看见“颜色跟着墙走，方向在中心变”这两个因素
#       是可分离的：颜色由墙决定，几何由碰撞决定。

from __future__ import annotations

import random as _rng
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import colors_present, components_mask, get_bg


Coord = tuple[int, int]


class RayState:
    def __init__(self, cur: Coord, dr: int, dc: int, color: int, mode: str):
        self.cur = cur
        self.dr = int(dr)
        self.dc = int(dc)
        self.color = int(color)
        self.mode = str(mode)
        self.alive = True


def _is_arrow_pts(pts: list[Coord]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[Coord]) -> tuple[tuple[int, int], Coord] | None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
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


def _build_arrow_pts(start: Coord, fire_dir: tuple[int, int]) -> list[Coord]:
    sr, sc = start
    dr, dc = fire_dir
    mr, mc = sr - dr, sc - dc
    pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
    return [(int(r), int(c)) for r, c in pts]


def _detect_arrows(g: np.ndarray, bg: int):
    arrows: list[tuple[int, list[Coord]]] = []
    arrows_by_color: dict[int, list[list[Coord]]] = {}
    arrow_cells: set[Coord] = set()
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            arrows.append((int(col), pts))
            arrows_by_color.setdefault(int(col), []).append(pts)
            arrow_cells |= set(pts)
    return arrows, arrows_by_color, arrow_cells


def _wall_cells(g: np.ndarray, arrow_cells: set[Coord], bg: int) -> set[Coord]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }


def _deflect_dir(diag_dir: tuple[int, int]) -> tuple[int, int]:
    dr, dc = diag_dir
    if dr == 0 or dc == 0:
        return int(dr), int(dc)
    if dr == dc:
        return 0, int(dc)
    return int(dr), 0


def _propose_diag_step(
    ray: RayState,
    g: np.ndarray,
    arrow_cells: set[Coord],
    wall_cells: set[Coord],
    bg: int,
) -> dict | None:
    H, W = g.shape
    cur_r, cur_c = ray.cur
    dr, dc = int(ray.dr), int(ray.dc)
    nr, nc = cur_r + dr, cur_c + dc
    if not (0 <= nr < H and 0 <= nc < W):
        return None

    if (nr, nc) in wall_cells:
        wcol = int(g[nr, nc])
        return {
            "paint_cell": (cur_r, cur_c),
            "paint_color": wcol,
            "next_cur": (cur_r, cur_c),
            "next_dr": -dr,
            "next_dc": -dc,
            "next_color": wcol,
            "step_dir": (dr, dc),
            "mode": "diag",
        }

    if (nr, nc) in arrow_cells:
        return None

    paint_color = int(ray.color)
    next_dr, next_dc, next_color = dr, dc, int(ray.color)

    if dr == 1 and (nr + 1, nc) in wall_cells:
        wcol = int(g[nr + 1, nc])
        paint_color = wcol
        next_color = wcol
        next_dr = -dr
    elif dr == -1 and (nr - 1, nc) in wall_cells:
        wcol = int(g[nr - 1, nc])
        paint_color = wcol
        next_color = wcol
        next_dr = -dr

    if dc == 1 and (nr, nc + 1) in wall_cells:
        wcol = int(g[nr, nc + 1])
        paint_color = wcol
        next_color = wcol
        next_dc = -dc
    elif dc == -1 and (nr, nc - 1) in wall_cells:
        wcol = int(g[nr, nc - 1])
        paint_color = wcol
        next_color = wcol
        next_dc = -dc

    return {
        "paint_cell": (nr, nc),
        "paint_color": int(paint_color),
        "next_cur": (nr, nc),
        "next_dr": int(next_dr),
        "next_dc": int(next_dc),
        "next_color": int(next_color),
        "step_dir": (dr, dc),
        "mode": "diag",
    }


def _propose_ortho_step(
    ray: RayState,
    g: np.ndarray,
    arrow_cells: set[Coord],
    wall_cells: set[Coord],
) -> dict | None:
    H, W = g.shape
    cur_r, cur_c = ray.cur
    nr, nc = cur_r + int(ray.dr), cur_c + int(ray.dc)
    if not (0 <= nr < H and 0 <= nc < W):
        return None
    if (nr, nc) in arrow_cells or (nr, nc) in wall_cells:
        return None
    return {
        "paint_cell": (nr, nc),
        "paint_color": int(ray.color),
        "next_cur": (nr, nc),
        "next_dr": int(ray.dr),
        "next_dc": int(ray.dc),
        "next_color": int(ray.color),
        "step_dir": (int(ray.dr), int(ray.dc)),
        "mode": "ortho",
    }


def _simulate_pair(
    g: np.ndarray,
    color: int,
    pts_a: list[Coord],
    pts_b: list[Coord],
    arrow_cells: set[Coord],
    wall_cells: set[Coord],
    bg: int,
) -> tuple[list[tuple[Coord, int]], dict[str, int]]:
    info_a = _arrow_dir_and_start(pts_a)
    info_b = _arrow_dir_and_start(pts_b)
    if info_a is None or info_b is None:
        return [], {"collision_count": 0, "ortho_tail_cells": 0}

    (dr_a, dc_a), start_a = info_a
    (dr_b, dc_b), start_b = info_b
    rays = [
        RayState(cur=start_a, dr=dr_a, dc=dc_a, color=int(color), mode="diag"),
        RayState(cur=start_b, dr=dr_b, dc=dc_b, color=int(color), mode="diag"),
    ]
    seen = [set(), set()]
    paint_ops: list[tuple[Coord, int]] = []
    collision_count = 0
    ortho_tail_cells = 0

    while any(ray.alive for ray in rays):
        proposals: list[tuple[int, dict]] = []
        for idx, ray in enumerate(rays):
            if not ray.alive:
                continue
            state = (ray.mode, ray.cur, int(ray.dr), int(ray.dc), int(ray.color))
            if state in seen[idx]:
                ray.alive = False
                continue
            seen[idx].add(state)

            if ray.mode == "diag":
                prop = _propose_diag_step(ray, g, arrow_cells, wall_cells, bg)
            else:
                prop = _propose_ortho_step(ray, g, arrow_cells, wall_cells)

            if prop is None:
                ray.alive = False
                continue
            proposals.append((idx, prop))

        if not proposals:
            break

        if (
            len(proposals) == 2
            and rays[0].mode == "diag"
            and rays[1].mode == "diag"
            and proposals[0][1]["paint_cell"] == proposals[1][1]["paint_cell"]
        ):
            cell = proposals[0][1]["paint_cell"]
            col = int(proposals[0][1]["paint_color"])
            paint_ops.append((cell, col))
            collision_count += 1

            for idx, prop in proposals:
                tail_dr, tail_dc = _deflect_dir(prop["step_dir"])
                rays[idx].cur = cell
                rays[idx].dr = int(tail_dr)
                rays[idx].dc = int(tail_dc)
                rays[idx].color = int(prop["paint_color"])
                rays[idx].mode = "ortho"
            continue

        for idx, prop in proposals:
            paint_ops.append((prop["paint_cell"], int(prop["paint_color"])))
            if rays[idx].mode == "ortho":
                ortho_tail_cells += 1
            rays[idx].cur = prop["next_cur"]
            rays[idx].dr = int(prop["next_dr"])
            rays[idx].dc = int(prop["next_dc"])
            rays[idx].color = int(prop["next_color"])
            rays[idx].mode = prop["mode"]

    return paint_ops, {
        "collision_count": int(collision_count),
        "ortho_tail_cells": int(ortho_tail_cells),
    }


def _trace_single_steps(
    g: np.ndarray,
    start_col: int,
    pts: list[Coord],
    arrow_cells: set[Coord],
    wall_cells: set[Coord],
    bg: int,
) -> list[tuple[Coord, int]]:
    info = _arrow_dir_and_start(pts)
    if info is None:
        return []

    H, W = g.shape
    (dr, dc), cur = info
    cur_col = int(start_col)
    seen = set()
    steps: list[tuple[Coord, int]] = []

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
            if steps and int(g[int(cur[0]), int(cur[1])]) == int(bg):
                steps[-1] = (steps[-1][0], int(wcol))
            cur_col = int(wcol)
            dr, dc = -int(dr), -int(dc)
            continue

        if (nr, nc) in arrow_cells:
            break

        cur = (int(nr), int(nc))
        paint_col = int(cur_col)

        if int(dr) == 1 and (nr + 1, nc) in wall_cells:
            wcol = int(g[nr + 1, nc])
            paint_col = int(wcol)
            cur_col = int(wcol)
            dr = -int(dr)
        elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
            wcol = int(g[nr - 1, nc])
            paint_col = int(wcol)
            cur_col = int(wcol)
            dr = -int(dr)

        if int(dc) == 1 and (nr, nc + 1) in wall_cells:
            wcol = int(g[nr, nc + 1])
            paint_col = int(wcol)
            cur_col = int(wcol)
            dc = -int(dc)
        elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
            wcol = int(g[nr, nc - 1])
            paint_col = int(wcol)
            cur_col = int(wcol)
            dc = -int(dc)

        steps.append((cur, int(paint_col)))

    return steps


def _apply_paints(out: np.ndarray, g: np.ndarray, ops: list[tuple[Coord, int]], bg: int) -> None:
    for (r, c), col in ops:
        if int(g[r, c]) == int(bg):
            out[r, c] = int(col)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    arrows, arrows_by_color, arrow_cells = _detect_arrows(g, bg)
    wall_cells = _wall_cells(g, arrow_cells, bg)

    out = g.copy()
    paired_ids: set[int] = set()

    for color, group in arrows_by_color.items():
        if len(group) != 2:
            continue
        pts_a, pts_b = group
        ops, _stats = _simulate_pair(g, int(color), pts_a, pts_b, arrow_cells, wall_cells, bg)
        _apply_paints(out, g, ops, bg)
        paired_ids.add(id(pts_a))
        paired_ids.add(id(pts_b))

    for color, pts in arrows:
        if id(pts) in paired_ids:
            continue
        _apply_paints(out, g, _trace_single_steps(g, int(color), pts, arrow_cells, wall_cells, bg), bg)

    return out


def _reflect(center: Coord, cell: Coord) -> Coord:
    cr, cc = center
    r, c = cell
    return int(2 * cr - r), int(2 * cc - c)


def _dirs_for_cells(cells: list[Coord]) -> list[tuple[int, int]]:
    return [(cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1]) for i in range(len(cells) - 1)]


def _turns_for_path(cells: list[Coord]):
    dirs = _dirs_for_cells(cells)
    turns: list[tuple[Coord, tuple[int, int], str]] = []
    for i in range(1, len(cells) - 1):
        prev_dir = dirs[i - 1]
        next_dir = dirs[i]
        if prev_dir == next_dir:
            continue
        if prev_dir[0] != next_dir[0] and prev_dir[1] != next_dir[1]:
            return None
        flip_axis = "dr" if prev_dir[0] != next_dir[0] else "dc"
        turns.append((cells[i], prev_dir, flip_axis))
    return turns


def _wall_shape(
    turn_cell: Coord,
    incoming_dir: tuple[int, int],
    flip_axis: str,
    size: int,
    H: int,
    W: int,
) -> list[Coord] | None:
    r, c = turn_cell
    dr, dc = incoming_dir
    if flip_axis == "dr":
        wr, wc = r + dr, c
        if not (0 <= wr < H and 0 <= wc < W):
            return None
        if size == 1:
            return [(int(wr), int(wc))]
        cells = [(int(wr), int(wc + off)) for off in (-1, 0, 1)]
        return cells if all(0 <= cc < W for _, cc in cells) else None

    wr, wc = r, c + dc
    if not (0 <= wr < H and 0 <= wc < W):
        return None
    if size == 1:
        return [(int(wr), int(wc))]
    cells = [(int(wr + off), int(wc)) for off in (-1, 0, 1)]
    return cells if all(0 <= rr < H for rr, _ in cells) else None


def _colorize_path(
    cells: list[Coord],
    start_color: int,
    turn_colors: dict[Coord, int],
) -> list[tuple[Coord, int]]:
    cur_col = int(start_color)
    records: list[tuple[Coord, int]] = []
    for cell in cells:
        paint_col = int(turn_colors.get(cell, cur_col))
        records.append((cell, int(paint_col)))
        if cell in turn_colors:
            cur_col = int(turn_colors[cell])
    return records


def _straight_tail(center: Coord, step: tuple[int, int], H: int, W: int) -> list[Coord]:
    dr, dc = step
    r, c = center
    cells: list[Coord] = []
    while True:
        r += int(dr)
        c += int(dc)
        if not (0 <= r < H and 0 <= c < W):
            break
        cells.append((int(r), int(c)))
    return cells


def _gen_pair(
    H: int,
    W: int,
    occupied: set[Coord],
    pair_color: int,
    palette: list[int],
    require_turn: bool,
):
    for _ in range(240):
        center = (_rng.randint(4, H - 5), _rng.randint(4, W - 5))
        if require_turn:
            n_turns = _rng.choice([1, 2])
        else:
            n_turns = _rng.choices([0, 1, 2], weights=[2, 4, 2])[0]
        seg_lengths = [_rng.randint(2, 4) for _ in range(n_turns + 1)]
        cur_dir = _rng.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        current = center
        outward_cells: list[Coord] = []
        used_half: set[Coord] = {center}
        outward_ok = True

        for seg_idx, seg_len in enumerate(seg_lengths):
            for _step in range(seg_len):
                current = (int(current[0] + cur_dir[0]), int(current[1] + cur_dir[1]))
                if not (0 <= current[0] < H and 0 <= current[1] < W):
                    outward_ok = False
                    break
                if current in used_half or current in occupied:
                    outward_ok = False
                    break
                outward_cells.append(current)
                used_half.add(current)
            if not outward_ok:
                break
            if seg_idx < n_turns:
                if _rng.random() < 0.5:
                    cur_dir = (-cur_dir[0], cur_dir[1])
                else:
                    cur_dir = (cur_dir[0], -cur_dir[1])

        if not outward_ok or len(outward_cells) < 3:
            continue

        half_a = list(reversed(outward_cells)) + [center]
        half_b = [_reflect(center, cell) for cell in half_a[:-1]] + [center]
        diag_cells = set(half_a) | set(half_b)
        if len(diag_cells) != len(half_a) + len(half_b) - 1:
            continue
        if (diag_cells - {center}) & occupied:
            continue

        dirs_a = _dirs_for_cells(half_a)
        dirs_b = _dirs_for_cells(half_b)
        if not dirs_a or not dirs_b:
            continue

        incoming_a = dirs_a[-1]
        incoming_b = dirs_b[-1]
        tail_a = _straight_tail(center, _deflect_dir(incoming_a), H, W)
        tail_b = _straight_tail(center, _deflect_dir(incoming_b), H, W)
        if len(tail_a) < 2 or len(tail_b) < 2:
            continue

        tail_cells = set(tail_a) | set(tail_b)
        if len(tail_cells) != len(tail_a) + len(tail_b):
            continue
        if tail_cells & occupied:
            continue
        if tail_cells & (diag_cells - {center}):
            continue

        start_a = (int(half_a[0][0] - dirs_a[0][0]), int(half_a[0][1] - dirs_a[0][1]))
        start_b = (int(half_b[0][0] - dirs_b[0][0]), int(half_b[0][1] - dirs_b[0][1]))
        if not (
            0 <= start_a[0] < H and 0 <= start_a[1] < W and
            0 <= start_b[0] < H and 0 <= start_b[1] < W
        ):
            continue

        arrow_a = _build_arrow_pts(start_a, dirs_a[0])
        arrow_b = _build_arrow_pts(start_b, dirs_b[0])
        arrow_cells = set(arrow_a) | set(arrow_b)
        if len(arrow_cells) != 6:
            continue
        if any(not (0 <= r < H and 0 <= c < W) for r, c in arrow_cells):
            continue
        if arrow_cells & occupied:
            continue
        if arrow_cells & diag_cells:
            continue
        if arrow_cells & tail_cells:
            continue

        turns_a = _turns_for_path(half_a)
        turns_b = _turns_for_path(half_b)
        if turns_a is None or turns_b is None or len(turns_a) != len(turns_b):
            continue

        turn_colors_a: dict[Coord, int] = {}
        turn_colors_b: dict[Coord, int] = {}
        wall_segments: list[tuple[list[Coord], int]] = []
        reserved = set(occupied) | set(arrow_cells) | set(diag_cells) | set(tail_cells)
        walls_ok = True

        for ta, tb in zip(turns_a, turns_b):
            turn_a, in_dir_a, flip_a = ta
            turn_b, in_dir_b, flip_b = tb
            if flip_a != flip_b:
                walls_ok = False
                break

            size_options = [3, 1] if _rng.random() < 0.6 else [1, 3]
            seg_a = seg_b = None
            for size in size_options:
                cand_a = _wall_shape(turn_a, in_dir_a, flip_a, size, H, W)
                cand_b = _wall_shape(turn_b, in_dir_b, flip_b, size, H, W)
                if cand_a is None or cand_b is None:
                    continue
                cells = set(cand_a) | set(cand_b)
                if len(cells) != len(cand_a) + len(cand_b):
                    continue
                if cells & reserved:
                    continue
                seg_a, seg_b = cand_a, cand_b
                break
            if seg_a is None or seg_b is None:
                walls_ok = False
                break

            wall_color_choices = [c for c in palette if c != pair_color] or list(palette)
            wall_color = int(_rng.choice(wall_color_choices))
            turn_colors_a[turn_a] = wall_color
            turn_colors_b[turn_b] = wall_color
            wall_segments.append((seg_a, wall_color))
            wall_segments.append((seg_b, wall_color))
            reserved |= set(seg_a) | set(seg_b)

        if not walls_ok:
            continue

        diag_records_a = _colorize_path(half_a, pair_color, turn_colors_a)
        diag_records_b = _colorize_path(half_b, pair_color, turn_colors_b)
        if diag_records_a[-1][1] != diag_records_b[-1][1]:
            continue

        collision_color = int(diag_records_a[-1][1])
        tail_records_a = [((int(r), int(c)), collision_color) for r, c in tail_a]
        tail_records_b = [((int(r), int(c)), collision_color) for r, c in tail_b]

        pair_cells = set(arrow_cells) | set(diag_cells) | set(tail_cells)
        for wall_cells, _wall_color in wall_segments:
            pair_cells |= set(wall_cells)

        return {
            "arrow_color": int(pair_color),
            "arrows": [arrow_a, arrow_b],
            "diag_records": [diag_records_a, diag_records_b],
            "tail_records": [tail_records_a, tail_records_b],
            "walls": wall_segments,
            "occupied": pair_cells,
            "turn_count": len(turns_a),
            "ortho_tail_cells": len(tail_a) + len(tail_b),
        }

    return None


def _apply_pair_to_grids(grid: np.ndarray, out: np.ndarray, pair_data: dict) -> None:
    color = int(pair_data["arrow_color"])
    for arrow in pair_data["arrows"]:
        for r, c in arrow:
            grid[r, c] = color
            out[r, c] = color

    for wall_cells, wall_color in pair_data["walls"]:
        for r, c in wall_cells:
            grid[r, c] = int(wall_color)
            out[r, c] = int(wall_color)

    for records in pair_data["diag_records"]:
        for (r, c), paint_col in records:
            out[r, c] = int(paint_col)

    for records in pair_data["tail_records"]:
        for (r, c), paint_col in records:
            out[r, c] = int(paint_col)


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _arrows, arrows_by_color, arrow_cells = _detect_arrows(g, bg)
    wall_cells = _wall_cells(g, arrow_cells, bg)

    pair_count = 0
    collision_count = 0
    ortho_tail_cells = 0

    for color, group in arrows_by_color.items():
        if len(group) != 2:
            continue
        pair_count += 1
        _ops, stats = _simulate_pair(g, int(color), group[0], group[1], arrow_cells, wall_cells, bg)
        collision_count += int(stats["collision_count"])
        ortho_tail_cells += int(stats["ortho_tail_cells"])

    return {
        "pair_count": int(pair_count),
        "collision_count": int(collision_count),
        "ortho_tail_cells": int(ortho_tail_cells),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_pairs = _rng.randint(1, 3)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    pair_colors = palette[:n_pairs]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[Coord] = set()

    total_turns = 0
    total_ortho_tail_cells = 0

    for idx, pair_color in enumerate(pair_colors):
        pair_data = _gen_pair(
            H,
            W,
            occupied,
            int(pair_color),
            palette,
            require_turn=(idx == 0),
        )
        if pair_data is None:
            return None
        occupied |= set(pair_data["occupied"])
        total_turns += int(pair_data["turn_count"])
        total_ortho_tail_cells += int(pair_data["ortho_tail_cells"])
        _apply_pair_to_grids(grid, out, pair_data)

    if total_turns < 1 or total_ortho_tail_cells < 6:
        return None

    solved = main(grid)
    if not np.array_equal(solved, out):
        return None

    metrics = _fusion_metrics(grid)
    if metrics["pair_count"] != n_pairs:
        return None
    if metrics["collision_count"] != n_pairs:
        return None
    if metrics["ortho_tail_cells"] < total_ortho_tail_cells:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
