# Combo variant of 142ca369 (index 14)
# combo_spec: D1/0+D2/1+D4/0
# Name: 直线射线+颜色转向+颜色混合
#
# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 保留原题对象家族：背景 0；每个箭头仍是 2x2 外接框里缺一角的 3 像素 L 形；
#      墙仍是单像素或 3 像素轴对齐短线段。
#    - 原题用缺角解码一个对角方向 `(dr, dc)`，随后射线沿 45 度逐格前进，墙的几何侧位决定翻哪一轴，
#      反射后颜色直接继承墙色。
#    - 本融合版同时改三处：
#      1. D1/0：射线改为正交路径，只会朝上/右/下/左四个方向之一逐格前进。
#      2. D2/1：墙不再决定镜面反射；正前方遇墙时，墙色 8 表示顺时针转 90 度，墙色 9 表示逆时针转 90 度。
#      3. D4/0：反射后颜色不直接变成墙色，而是在一个小循环色集 [1, 2, 4] 上做“颜色混合”：
#           当前色 + 墙 8  -> 循环前进一步
#           当前色 + 墙 9  -> 循环后退一步
#         也就是 1->2->4->1 与其逆循环。
#    - 为了把原 L 形的“四象限缺角”稳定映射到“四个正交起始方向”，采用全局固定约定：
#         缺左上角 -> 向右发射
#         缺右上角 -> 向下发射
#         缺右下角 -> 向左发射
#         缺左下角 -> 向上发射
#      这保持了“箭头缺角决定发射方向”的视觉语义，只是把对角象限压缩成顺时针排列的四个正交方向。
#    - 小例子：若箭头缺左上角，则直角像素在 2x2 框右下，融合版先向右走。
#      若右侧正前方墙色为 8，则射线不会进入墙格，而是在墙前一格完成事件：
#        几何：向右 -> 向下
#        颜色：若当前色为 1，则混成 2
#      若墙色为 9，则：
#        几何：向右 -> 向上
#        颜色：1 -> 4
#      与原题相比，同一输入里的差异是三重可见的：路径从对角线变成水平/垂直折线；左/右转由颜色而不是
#      墙朝向决定；转折后的颜色由“当前色和墙色共同决定”，而不是直接等于墙色。
#
# b) 复杂度校准
#    - 转向规则只保留两档墙色 8/9，便于从 2-4 个样例归纳。
#    - 混色只保留 3 个轨迹色 [1,2,4]，形成 3x2 的小映射表，信息量足够但不会过度自由。
#    - 每个生成样例都强制至少 5 次总转向，并要求两种墙色都重复出现，且都要在水平来向和垂直来向上
#      各出现一次；否则人类难以确认“颜色决定转向，而不是来向或墙摆放决定转向”。
#    - 箭头数量固定为 3 个，初始色恰好覆盖 [1,2,4]，既能给混色提供足够入射色变化，又不会让场景拥挤。
#
# c) 对原 solver 隐式假设的审查
#    - 原 `nr = cur_r + dr; nc = cur_c + dc` 假设一步同时改行列。融合后失效，必须改成四个正交方向之一。
#    - 原 side-touch 逻辑假设“墙的侧位”决定翻哪一轴。融合后失效；墙只表示“此处触发事件”，真正转向由
#      墙色决定。因此 solver 只检查正前方是否为墙，不再做两侧 side-touch 判定。
#    - 原颜色更新 `cur_col = wall_color` 失效；现在必须先读取“当前轨迹色”和“墙色”，再做混色映射。
#    - 原代码把正面撞墙当作 `(-dr, -dc)` 双轴反转。融合后也失效；正前方撞墙同样按照 8/9 做 90 度转向。
#    - 原代码默认所有非箭头像素都可视为普通墙。这一分类仍成立，但 solver 新增假设：
#        * 墙色只来自规则集 {8,9}
#        * 轨迹色只来自闭包集合 [1,2,4]
#      generator 会显式满足这两点。
#
# d) 放置合法性
#    - 生成器继续使用 output-first：先规划每条正交折线路径与每次转向，再反推出输入中的箭头和墙。
#    - 箭头、墙、路径三者都不共享格子；所有规划出的最终输出路径之间也不重叠。
#    - 每个转向点的墙中心都放在当前段末端的正前方一格；若是 3 像素墙，就沿与来向垂直的轴展开，
#      因而像一个“挡板”，但不会覆盖射线实际经过的格子。
#    - 同色墙之间禁止跨段 4 邻接，避免两个独立墙段拼成 3 像素 L 形同色连通块，被箭头检测误识别。
#    - 候选样例只有在 `main(input)` 精确复现规划输出时才会接受。
#
# e) 视觉可观察性
#    - D1 的变化直接可见：输出路径是长水平/垂直折线，不再是原题的对角线锯齿。
#    - D2 的变化也可见：同样都是“前方挡板”，8 与 9 会让同一来向分别左转/右转；墙的摆放朝向不再编码结果。
#    - D4 的变化通过颜色分层体现：墙只用 8/9，两者都不属于轨迹色循环 [1,2,4]。因此转折后若轨迹继续用
#      1/2/4 延伸，人类能直接排除“直接继承墙色”的解释。
#    - 生成器额外检验两种退化：
#        * 若把 8/9 的转向含义互换，输出必须改变。
#        * 若把混色替换成“直接继承墙色”，输出也必须改变。
#
# f) 可视化整洁度
#    - 保持 18-20 的稀疏方阵；只有 3 个箭头，避免多条正交折线互相压缩。
#    - 每段长度至少为 2，且每次转向后还需至少保留 2 格尾段，确保折线和颜色变化都足够可见。
#    - 颜色角色分层明确：轨迹/箭头只用 [1,2,4]，规则墙只用 [8,9]。
#
# g) 变化的可推断性
#    - 可推断的映射只有两层，而且都有重复证据：
#        * 8/9 -> 顺/逆时针转向
#        * [1,2,4] x {8,9} -> 混色结果
#    - 样例过滤要求：
#        * 每种墙色至少被撞到 2 次
#        * 每种墙色都要搭配至少 2 种不同入射轨迹色
#        * 至少有一种入射轨迹色分别撞到过两种墙色
#      这样解题者既能看到“同墙色面对不同当前色，结果不同”，也能看到“同当前色面对不同墙色，结果不同”。
#
# 9) main() 的新隐式假设与 generator 对应保证
#    - 假设 1：箭头仍是孤立的单色 3 像素 L 形。generator 保证箭头不重叠且初始色互异。
#    - 假设 2：非背景、非箭头像素都是规则墙，颜色在 {8,9}。generator 保证。
#    - 假设 3：轨迹颜色始终在 [1,2,4] 闭包中。generator 让箭头初色来自该集合，混色规则也闭包于此。
#    - 假设 4：每个拐点只会遇到一个正前方墙。generator 的 output-first 规划保证，不构造双重接触。
#    - 假设 5：同色墙不会跨组件 4 邻接。generator 对同色墙位置做检查，避免误识别为箭头。
#    - 假设 6：所有规划路径都会正常离开网格且不撞上别的箭头/墙/已绘路径。generator 用占用检查、
#      局部尾段可达检查和最终 oracle 检查共同保证。

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg


CW_WALL_COLOR = 8
CCW_WALL_COLOR = 9
RULE_WALL_COLORS = (CW_WALL_COLOR, CCW_WALL_COLOR)

RAY_CYCLE = (1, 2, 4)
RAY_INDEX = {int(col): idx for idx, col in enumerate(RAY_CYCLE)}

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

COLOR_MODE_MIX = "mix"
COLOR_MODE_INHERIT = "inherit"

# Offsets of the 3 arrow pixels relative to the degree-2 launch cell.
ARROW_OFFSETS = {
    DIR_E: [(-1, 0), (0, -1), (0, 0)],   # missing NW
    DIR_S: [(-1, 0), (0, 0), (0, 1)],    # missing NE
    DIR_W: [(0, 0), (0, 1), (1, 0)],     # missing SE
    DIR_N: [(0, -1), (0, 0), (1, 0)],    # missing SW
}


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < H and 0 <= int(c) < W


def _rotate(direction: tuple[int, int], turn: str) -> tuple[int, int]:
    dr, dc = direction
    if turn == "cw":
        return int(dc), -int(dr)
    if turn == "ccw":
        return -int(dc), int(dr)
    return int(dr), int(dc)


def _rotate_by_color(
    direction: tuple[int, int],
    wall_color: int,
    turn_map: dict[int, str] = TURN_MAP,
) -> tuple[int, int]:
    return _rotate(direction, turn_map.get(int(wall_color), ""),)


def _next_color(ray_color: int, wall_color: int, color_mode: str) -> int:
    if color_mode == COLOR_MODE_INHERIT:
        return int(wall_color)
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
    color_mode: str = COLOR_MODE_MIX,
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
                new_col = _next_color(int(cur_col), int(wcol), color_mode)
                new_direction = _rotate_by_color(direction, int(wcol), turn_map)
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(new_col)
                if collect_events:
                    events.append(
                        {
                            "incoming": direction,
                            "incoming_axis": "h" if direction[1] != 0 else "v",
                            "incoming_color": int(cur_col),
                            "wall_color": int(wcol),
                            "result_color": int(new_col),
                            "turn": turn_map.get(int(wcol), ""),
                            "contact": (int(nr), int(nc)),
                            "bounce_from": cur,
                            "outgoing": new_direction,
                        }
                    )
                cur_col = int(new_col)
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

    Fused rule:
    - Each arrow decodes to one of four cardinal launch directions.
    - The ray advances one cell at a time horizontally or vertically.
    - A wall directly ahead turns the ray:
        color 8 -> clockwise 90 degrees
        color 9 -> counterclockwise 90 degrees
    - The wall also mixes with the current ray color on cycle [1,2,4]:
        wall 8 -> +1 step on the cycle
        wall 9 -> -1 step on the cycle
    - The cell just before the wall is recolored to the mixed color, and the
      following segment continues in that mixed color.
    - Rays stop when they leave the grid or would step into another arrow.
    """
    return _trace_variant(grid, turn_map=TURN_MAP, color_mode=COLOR_MODE_MIX, collect_events=False)


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
    for _ in range(int(seg_len)):
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

    for sr, sc in start_positions[:180]:
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

        for _turn_idx in range(int(n_turns)):
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

                    new_col = _next_color(int(cur_col), int(wall_color), COLOR_MODE_MIX)
                    lr, lc, _old = ray_cells[-1]
                    ray_cells[-1] = (int(lr), int(lc), int(new_col))
                    cur_r, cur_c = int(end_r), int(end_c)
                    direction = next_direction
                    cur_col = int(new_col)
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


def _collect_events(grid: np.ndarray) -> list[dict[str, object]]:
    _out, events = _trace_variant(
        grid,
        turn_map=TURN_MAP,
        color_mode=COLOR_MODE_MIX,
        collect_events=True,
    )
    return events


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    mixed_out = main(grid)
    reversed_out = _trace_variant(
        grid,
        turn_map=REVERSED_TURN_MAP,
        color_mode=COLOR_MODE_MIX,
        collect_events=False,
    )
    inherit_out = _trace_variant(
        grid,
        turn_map=TURN_MAP,
        color_mode=COLOR_MODE_INHERIT,
        collect_events=False,
    )
    events = _collect_events(grid)

    wall_to_inputs = {
        int(wcol): {int(e["incoming_color"]) for e in events if int(e["wall_color"]) == int(wcol)}
        for wcol in RULE_WALL_COLORS
    }
    input_to_walls: dict[int, set[int]] = {}
    for e in events:
        input_to_walls.setdefault(int(e["incoming_color"]), set()).add(int(e["wall_color"]))

    return {
        "turn_count": len(events),
        "diff_vs_reversed": int(np.sum(mixed_out != reversed_out)),
        "diff_vs_inherit": int(np.sum(mixed_out != inherit_out)),
        "wall_use_counts": {
            int(wcol): sum(int(int(e["wall_color"]) == int(wcol)) for e in events)
            for wcol in RULE_WALL_COLORS
        },
        "axes": {
            int(wcol): {str(e["incoming_axis"]) for e in events if int(e["wall_color"]) == int(wcol)}
            for wcol in RULE_WALL_COLORS
        },
        "wall_to_input_variety": {int(w): len(vals) for w, vals in wall_to_inputs.items()},
        "max_wall_variety_per_input_color": max((len(vals) for vals in input_to_walls.values()), default=0),
        "mixed_not_equal_old_or_wall": sum(
            int(
                int(e["result_color"]) != int(e["incoming_color"])
                and int(e["result_color"]) != int(e["wall_color"])
            )
            for e in events
        ),
        "distinct_mix_triples": len(
            {
                (int(e["incoming_color"]), int(e["wall_color"]), int(e["result_color"]))
                for e in events
            }
        ),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = 3
    arrow_colors = list(RAY_CYCLE)
    _rng.shuffle(arrow_colors)

    target_total_turns = _rng.randint(5, 7)
    turn_counts = [1] * n_arrows
    for _ in range(target_total_turns - n_arrows):
        candidates = [i for i, cnt in enumerate(turn_counts) if cnt < 3]
        if not candidates:
            break
        turn_counts[int(_rng.choice(candidates))] += 1

    start_directions = list(CARDINAL_DIRS)
    _rng.shuffle(start_directions)
    start_directions = start_directions[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    wall_color_cells_global = {CW_WALL_COLOR: set(), CCW_WALL_COLOR: set()}
    forced_colors = [CW_WALL_COLOR, CCW_WALL_COLOR, CW_WALL_COLOR, CCW_WALL_COLOR, CW_WALL_COLOR, CCW_WALL_COLOR]

    for color, start_direction, n_turns in zip(arrow_colors, start_directions, turn_counts):
        planned = _plan_arrow(
            H,
            W,
            int(color),
            start_direction,
            int(n_turns),
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

    metrics = _behavior_metrics(grid)
    if int(metrics["turn_count"]) < 5:
        return None
    if int(metrics["diff_vs_reversed"]) < 4:
        return None
    if int(metrics["diff_vs_inherit"]) < 4:
        return None
    if int(metrics["mixed_not_equal_old_or_wall"]) != int(metrics["turn_count"]):
        return None
    if int(metrics["distinct_mix_triples"]) < 5:
        return None
    if any(int(metrics["wall_use_counts"][w]) < 2 for w in RULE_WALL_COLORS):
        return None
    if any(metrics["axes"][w] != {"h", "v"} for w in RULE_WALL_COLORS):
        return None
    if any(int(metrics["wall_to_input_variety"][w]) < 2 for w in RULE_WALL_COLORS):
        return None
    if int(metrics["max_wall_variety_per_input_color"]) < 2:
        return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(700):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 700 attempts")
