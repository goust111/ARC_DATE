# Combo variant of 142ca369 (index 1)
# combo_spec: D1/0+D2/1
# Name: 直线射线+颜色编码转向
#
# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 保留原题的对象家族：背景 0；单色 3 像素 L 形箭头；单像素或 3 像素直线墙。
#    - 原题里箭头从 L 形缺角信息解码出对角方向 `(dr, dc)`，射线每步都沿 45 度移动。
#      碰到墙后，反射方向由“墙在射线哪一侧”决定。
#    - 融合后，路径几何和转向机制同时改变：
#      1. 射线不再走对角线，而是一次只走一个正交方向。
#      2. 墙不再决定“翻哪一轴”；墙只表示“这里要拐弯”，真正向左还是向右由墙色决定：
#         color 8 -> 顺时针 90 度，color 9 -> 逆时针 90 度。
#    - 由于原始 3 像素 L 形天然编码的是 4 个对角象限，而不是 4 个正交方向，
#      这里给出一个全局固定、可从样例中归纳的解码约定：
#         缺左上角 -> 向右发射
#         缺右上角 -> 向下发射
#         缺右下角 -> 向左发射
#         缺左下角 -> 向上发射
#      这等价于：从右角像素看，朝两个“外侧空方向”里的顺时针那个方向发射。
#    - 小例子：若箭头缺左上角，则右角像素在 2x2 框的右下角，融合版先向右走。
#      若右方正前方墙是 color 8，则在撞墙前一格把当前格改成 8，并转为向下；
#      若是 color 9，则改成 9 并转为向上。原题在同一几何布局中则会继续沿对角线，
#      且“往上还是往下”取决于墙放在侧边哪一侧，而不是取决于墙色。
#
# b) 复杂度校准
#    - 颜色规则只保留两档，8/9，避免颜色到动作的自由度过大。
#    - 每个样例强制出现至少 4 次总转向，且两种规则色都重复出现；
#      否则人类无法从 2-4 个样例稳定归纳“8/9 各代表什么”。
#    - 墙仍保留 1 像素与 3 像素两种外观，但 3 像素墙永远与来向垂直，
#      从而只表达“阻挡点”，不泄露左转/右转信息。
#
# c) 对原 solver 隐式假设的审查
#    - 原代码里 `nr = cur[0] + dr; nc = cur[1] + dc` 假设每一步都同时改行列。
#      融合后失效，必须改为四个正交方向之一。
#    - 原代码里“墙在 (nr+dr,nc) 还是 (nr,nc+dc)”决定翻哪根轴，这个前提已失效。
#      融合后只检查正前方是否是墙；一旦命中，转向完全由墙色给出。
#    - 原代码允许一次 advance 后再做最多两次 side-touch 检查；镜面模型下有意义，
#      但颜色转向下若一次读到两个墙色会歧义。新 generator 保证每个转向点只有
#      一个正前方墙，不构造双重接触。
#    - 原代码默认任意非背景颜色都可能是墙色。融合后 solver 新增假设：
#      非背景、非箭头像素都是规则墙，颜色只会是 8/9。generator 保证这一点。
#
# d) 放置合法性
#    - 仍采用 output-first：先规划每条正交折线路径和每个拐点的颜色，再反推出输入墙。
#    - 箭头、墙、规划出的射线路径三者都不共享格子。
#    - 每个拐点的墙中心放在“当前段终点的正前方一格”；若是 3 像素墙，则沿与来向垂直的轴展开，
#      这样既像“挡板”，又不会覆盖射线终点。
#    - 最终候选必须通过 oracle：`main(input) == planned_output`。
#
# e) 视觉可观察性
#    - 正交长段 + 90 度折弯与原题的对角锯齿有明显视觉差异，D1 的变化可直接观察。
#    - 同一种墙朝向会在不同颜色下导向不同转向结果，能看出“颜色而非几何决定左/右转”。
#    - generator 额外要求：把 8/9 的映射反过来重跑时，输出必须不同；否则该样例在
#      颜色控制变量上退化，不予接受。
#
# f) 可视化整洁度
#    - 网格仍是 18-20 的稀疏布局；箭头数量降到 2-4，避免多条正交折线挤在一起。
#    - 每条射线至少有若干长度为 2+ 的直段，墙只出现在拐点正前方，整体像少量清晰的
#      水平/垂直折线，而不是密集迷宫。
#
# g) 颜色变化可推断性
#    - 颜色变化只有一层语义：墙色 == 转向方向，同时反射后射线继承墙色。
#    - 为了让“颜色决定转向”而不是“某种来向恰好总搭配某种颜色”，generator 要求
#      8 和 9 都至少在横向来向和纵向来向上各出现一次。
#
# 9) main() 的新隐式假设与 generator 对应保证
#    - 假设 1：箭头仍然是单色、孤立的 3 像素 4 连通 L 形。generator 保证。
#    - 假设 2：墙色只会取 8/9。generator 保证。
#    - 假设 3：每个拐点只会遇到一个正前方墙。generator 用 output-first 路径规划保证。
#    - 假设 4：同色墙不会与别的同色墙 4 邻接成新的 3 像素 L 组件，从而误判成箭头。
#      generator 对跨墙的同色 4 邻接做了禁止。
#    - 假设 5：把 8/9 的转向含义互换会改变样例输出。generator 显式检验这一点。

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _cand = _parent / "solvers"
    if _cand.is_dir():
        sys.path.append(str(_cand))
        break

from arc_utils import colors_present, components_mask, get_bg


CW_WALL_COLOR = 8
CCW_WALL_COLOR = 9
RULE_WALL_COLORS = (CW_WALL_COLOR, CCW_WALL_COLOR)

DIR_N = (-1, 0)
DIR_E = (0, 1)
DIR_S = (1, 0)
DIR_W = (0, -1)
CARDINAL_DIRS = (DIR_N, DIR_E, DIR_S, DIR_W)

TURN_MAP = {
    CW_WALL_COLOR: "cw",
    CCW_WALL_COLOR: "ccw",
}
REVERSED_TURN_MAP = {
    CW_WALL_COLOR: "ccw",
    CCW_WALL_COLOR: "cw",
}

# Offsets of the 3 arrow pixels relative to the right-angle start pixel.
ARROW_OFFSETS = {
    DIR_E: [(-1, 0), (0, -1), (0, 0)],   # missing NW
    DIR_S: [(-1, 0), (0, 0), (0, 1)],    # missing NE
    DIR_W: [(0, 0), (0, 1), (1, 0)],     # missing SE
    DIR_N: [(0, -1), (0, 0), (1, 0)],    # missing SW
}


def _rotate(direction: tuple[int, int], turn: str) -> tuple[int, int]:
    dr, dc = direction
    if turn == "cw":
        return int(dc), -int(dr)
    return -int(dc), int(dr)


def _rotate_by_color(
    direction: tuple[int, int],
    wall_color: int,
    turn_map: dict[int, str] = TURN_MAP,
) -> tuple[int, int]:
    return _rotate(direction, turn_map[int(wall_color)])


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < H and 0 <= int(c) < W


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(
    pts: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None

    # Missing-corner -> orthogonal initial direction.
    mr, mc = miss[0]
    if (mr, mc) == (r0, c0):
        direction = DIR_E
    elif (mr, mc) == (r0, c1):
        direction = DIR_S
    elif (mr, mc) == (r1, c1):
        direction = DIR_W
    else:
        direction = DIR_N

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return direction, (int(start[0]), int(start[1]))


def _arrow_points_from_start(sr: int, sc: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    return [(int(sr + dr), int(sc + dc)) for dr, dc in ARROW_OFFSETS[direction]]


def _detect_scene(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, list[tuple[int, int]]]], set[tuple[int, int]], set[tuple[int, int]]]:
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
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return bg, arrows, arrow_cells, wall_cells


def _trace_variant(
    grid: np.ndarray,
    *,
    turn_map: dict[int, str] = TURN_MAP,
    collect_events: bool = False,
):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_cells = _detect_scene(g)
    out = np.array(g, copy=True)
    events: list[dict[str, object]] = []

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        direction, cur = info
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

        while True:
            state = (cur, direction, int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + direction[0])
            nc = int(cur[1] + direction[1])
            if not _in_bounds(H, W, nr, nc):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                new_direction = _rotate_by_color(direction, int(wcol), turn_map)
                if collect_events:
                    events.append(
                        {
                            "incoming": direction,
                            "incoming_axis": "h" if direction[1] != 0 else "v",
                            "wall_color": int(wcol),
                            "turn": turn_map[int(wcol)],
                            "contact": (int(nr), int(nc)),
                            "bounce_from": cur,
                            "outgoing": new_direction,
                        }
                    )
                cur_col = int(wcol)
                direction = new_direction
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

    if collect_events:
        return out, events
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace orthogonal rays from L-shaped arrows.

    Variant rule:
    - Each arrow decodes to one of four cardinal start directions.
    - The ray advances one cell at a time horizontally or vertically.
    - A wall directly ahead does not mirror-reflect by geometry. Instead:
        color 8 -> clockwise 90-degree turn
        color 9 -> counterclockwise 90-degree turn
    - At each turn the current ray cell inherits the wall color.
    - Rays stop when they leave the grid or would step into another arrow.
    """
    return _trace_variant(grid, turn_map=TURN_MAP, collect_events=False)


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


def _steps_to_edge(r: int, c: int, direction: tuple[int, int], H: int, W: int) -> int:
    dr, dc = direction
    if dr == 1:
        return H - 1 - int(r)
    if dr == -1:
        return int(r)
    if dc == 1:
        return W - 1 - int(c)
    return int(c)


def _advance_cells(
    cur_r: int,
    cur_c: int,
    direction: tuple[int, int],
    seg_len: int,
    blocked: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    rr, cc = int(cur_r), int(cur_c)
    cells: list[tuple[int, int]] = []
    for _ in range(seg_len):
        rr += int(dr)
        cc += int(dc)
        if not _in_bounds(H, W, rr, cc) or (rr, cc) in blocked:
            return None
        cells.append((int(rr), int(cc)))
    return cells


def _tail_cells_to_exit(
    cur_r: int,
    cur_c: int,
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = direction
    rr, cc = int(cur_r), int(cur_c)
    cells: list[tuple[int, int]] = []
    while True:
        rr += int(dr)
        cc += int(dc)
        if not _in_bounds(H, W, rr, cc):
            return cells
        if (rr, cc) in blocked:
            return None
        cells.append((int(rr), int(cc)))


def _wall_options(
    center: tuple[int, int],
    incoming: tuple[int, int],
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    r, c = center
    options: list[list[tuple[int, int]]] = [[(int(r), int(c))]]
    if incoming[0] == 0:
        seg = [(int(r - 1), int(c)), (int(r), int(c)), (int(r + 1), int(c))]
    else:
        seg = [(int(r), int(c - 1)), (int(r), int(c)), (int(r), int(c + 1))]
    if all(_in_bounds(H, W, rr, cc) for rr, cc in seg):
        options.append(seg)
    _rng.shuffle(options)
    options.sort(key=len, reverse=(_rng.random() < 0.6))
    return options


def _valid_start_positions(H: int, W: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    offsets = ARROW_OFFSETS[direction]
    min_r = min(dr for dr, _dc in offsets)
    max_r = max(dr for dr, _dc in offsets)
    min_c = min(dc for _dr, dc in offsets)
    max_c = max(dc for _dr, dc in offsets)
    return [
        (int(sr), int(sc))
        for sr in range(-min_r, H - max_r)
        for sc in range(-min_c, W - max_c)
    ]


def _plan_arrow(
    H: int,
    W: int,
    color: int,
    start_direction: tuple[int, int],
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
    start_positions = _valid_start_positions(H, W, start_direction)
    _rng.shuffle(start_positions)

    for sr, sc in start_positions[:160]:
        arrow_pts = _arrow_points_from_start(int(sr), int(sc), start_direction)
        if any((r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = int(sr), int(sc)
        direction = start_direction
        cur_col = int(color)
        ray_cells: list[tuple[int, int, int]] = []
        ray_set: set[tuple[int, int]] = set()
        walls: list[tuple[list[tuple[int, int]], int]] = []
        wall_cells_local: set[tuple[int, int]] = set()
        wall_color_cells_local = {CW_WALL_COLOR: set(), CCW_WALL_COLOR: set()}
        forced_used = 0
        ok = True

        for turn_idx in range(n_turns):
            blocked = occupied | set(arrow_pts) | ray_set | wall_cells_local
            max_steps = _steps_to_edge(cur_r, cur_c, direction, H, W)
            if max_steps < 3:
                ok = False
                break

            wall_color = (
                int(forced_colors[forced_used])
                if forced_used < len(forced_colors)
                else int(_rng.choice(RULE_WALL_COLORS))
            )

            candidate_lengths = list(range(2, max_steps))
            _rng.shuffle(candidate_lengths)
            candidate_lengths.sort(key=lambda n: abs(n - max_steps // 2))
            committed = False

            for seg_len in candidate_lengths:
                seg_cells = _advance_cells(cur_r, cur_c, direction, seg_len, blocked, H, W)
                if seg_cells is None:
                    continue
                end_r, end_c = seg_cells[-1]
                center = (int(end_r + direction[0]), int(end_c + direction[1]))
                if not _in_bounds(H, W, center[0], center[1]):
                    continue

                for wall_cells in _wall_options(center, direction, H, W):
                    if any(cell in blocked or cell in set(seg_cells) for cell in wall_cells):
                        continue
                    if not _same_color_clear(
                        wall_cells,
                        wall_color_cells_global[wall_color],
                        wall_color_cells_local[wall_color],
                    ):
                        continue

                    next_direction = _rotate_by_color(direction, wall_color, TURN_MAP)
                    tail_probe = _tail_cells_to_exit(
                        end_r,
                        end_c,
                        next_direction,
                        blocked | set(seg_cells) | set(wall_cells),
                        H,
                        W,
                    )
                    if tail_probe is None or len(tail_probe) < 2:
                        continue

                    for rr, cc in seg_cells:
                        ray_cells.append((int(rr), int(cc), int(cur_col)))
                        ray_set.add((int(rr), int(cc)))
                    walls.append((wall_cells, int(wall_color)))
                    wall_cells_local.update(wall_cells)
                    wall_color_cells_local[wall_color].update(wall_cells)
                    lr, lc, _old = ray_cells[-1]
                    ray_cells[-1] = (int(lr), int(lc), int(wall_color))
                    cur_r, cur_c = int(end_r), int(end_c)
                    direction = next_direction
                    cur_col = int(wall_color)
                    forced_used += 1
                    committed = True
                    break

                if committed:
                    break

            if not committed:
                ok = False
                break

        if not ok:
            continue

        tail_cells = _tail_cells_to_exit(
            cur_r,
            cur_c,
            direction,
            occupied | set(arrow_pts) | ray_set | wall_cells_local,
            H,
            W,
        )
        if tail_cells is None or len(tail_cells) < 2:
            continue
        for rr, cc in tail_cells:
            ray_cells.append((int(rr), int(cc), int(cur_col)))
            ray_set.add((int(rr), int(cc)))

        if len(ray_cells) < 6:
            continue

        new_cells = set(arrow_pts) | ray_set | wall_cells_local
        expected = len(arrow_pts) + len(ray_set) + len(wall_cells_local)
        if len(new_cells) != expected:
            continue
        if new_cells & occupied:
            continue

        return arrow_pts, ray_cells, walls, forced_used

    return None


def _collect_turn_events(grid: np.ndarray) -> list[dict[str, object]]:
    _out, events = _trace_variant(grid, turn_map=TURN_MAP, collect_events=True)
    return events


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    correct = main(grid)
    reversed_out = _trace_variant(grid, turn_map=REVERSED_TURN_MAP, collect_events=False)
    events = _collect_turn_events(grid)
    diff_cells = int(np.sum(correct != reversed_out))
    per_color_counts = {
        color: sum(1 for e in events if int(e["wall_color"]) == color)
        for color in RULE_WALL_COLORS
    }
    per_color_axes = {
        color: {str(e["incoming_axis"]) for e in events if int(e["wall_color"]) == color}
        for color in RULE_WALL_COLORS
    }
    return {
        "turns": len(events),
        "diff_vs_reversed": diff_cells,
        "counts": per_color_counts,
        "axes": per_color_axes,
    }


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

    start_directions = list(CARDINAL_DIRS)
    _rng.shuffle(start_directions)
    start_directions = start_directions[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    wall_color_cells_global = {CW_WALL_COLOR: set(), CCW_WALL_COLOR: set()}
    forced_colors = [CW_WALL_COLOR, CCW_WALL_COLOR, CW_WALL_COLOR, CCW_WALL_COLOR]

    for color, start_direction, n_turns in zip(arrow_colors, start_directions, turn_counts):
        planned = _plan_arrow(
            H,
            W,
            int(color),
            start_direction,
            n_turns,
            occupied,
            wall_color_cells_global,
            forced_colors,
        )
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
        occupied |= {(int(r), int(c)) for r, c, _ray_col in ray_cells}
        occupied |= {(int(r), int(c)) for wall_cells, _wcol in walls for r, c in wall_cells}
        forced_colors = forced_colors[forced_used:]

    events = _collect_turn_events(grid)
    if len(events) < 4:
        return None
    used_rule_colors = {int(e["wall_color"]) for e in events}
    if used_rule_colors != set(RULE_WALL_COLORS):
        return None
    counts = {color: sum(1 for e in events if int(e["wall_color"]) == color) for color in RULE_WALL_COLORS}
    if any(counts[color] < 2 for color in RULE_WALL_COLORS):
        return None
    axes = {color: {str(e["incoming_axis"]) for e in events if int(e["wall_color"]) == color} for color in RULE_WALL_COLORS}
    if any(axes[color] != {"h", "v"} for color in RULE_WALL_COLORS):
        return None

    solved = main(grid)
    if not np.array_equal(solved, out):
        return None

    reversed_out = _trace_variant(grid, turn_map=REVERSED_TURN_MAP, collect_events=False)
    if np.array_equal(solved, reversed_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 600 attempts")
