# Combo variant of 142ca369 (index 25)
# combo_spec: D2/0+D7/2
# Name: 镜面反射+对角线墙壁
#
# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 保留原题对象家族：背景 0；每个箭头仍是 2x2 外接框里缺一角的 3 像素 L 形。
#    - 但 D7/2 把“墙”整体改成 45 度短镜面，因此如果仍保留原题的对角线步进，镜面反射会退化成
#      “平行滑过”或“正撞后原路返回”，几乎没有可见的几何信息。为了让“对角镜面”真正承载 D2/0 的
#      入射角=反射角规则，这个融合版把射线段具体化为正交的水平/垂直折线；L 形缺角仍然决定四个发射态，
#      只是四个象限被稳定映射成四个正交方向：
#         缺左上角 -> 向右发射
#         缺右上角 -> 向下发射
#         缺右下角 -> 向左发射
#         缺左下角 -> 向上发射
#    - 墙只允许两种 3 像素对角镜面：
#         "/"  = {(r-1,c+1),(r,c),(r+1,c-1)}
#         "\" = {(r-1,c-1),(r,c),(r+1,c+1)}
#      其中中心像素正好放在射线正前方一格，所以“撞墙”发生在墙前一格；该格被改涂成墙色，之后射线
#      继承墙色并按镜面朝向转 90 度。
#    - 具体小例子：
#         当前射线在 (5,5)，方向向右，前方中心镜面在 (5,6)。
#         若镜面是 "/"，则右 -> 上，所以输出把 (5,5) 改成墙色，并从那里向上继续。
#         若镜面是 "\"，则右 -> 下，所以同一个来向会走成完全不同的折线。
#      这和原题的差异有两层：
#         1. 原题是对角线步进，且靠“邻边是否有墙”决定翻哪根轴。
#         2. 融合后是正交步进，且真正决定转向的是镜面朝向 "/" 或 "\"。
#
# b) 复杂度校准
#    - 每个样例只用两种镜面颜色 {7,8}，但两种颜色都会在两种朝向上重复出现，防止“颜色决定转向”的
#      错误归纳；真正起作用的是镜面几何朝向。
#    - 只保留两种朝向 "/" 与 "\"，是足够小的离散状态空间；解题者可以从 2-4 个样例中归纳
#      “同样是水平入射，/ 和 \ 会分别把光路折到上/下；垂直入射时也对应转到右/左”。
#    - 每个生成样例固定 3 个箭头、总反射 5 次，其中前 4 次强制覆盖：
#         (/,7), (\,7), (/,8), (\,8)
#      且这四次由水平来向和垂直来向各出现一次。这样既展示“颜色无关”，又展示“镜面朝向 + 入射轴”
#      才共同决定出射方向。
#
# c) 对原 solver 隐式假设的审查
#    - 原 `nr = cur + (dr,dc)` 假设一条射线永远按对角线前进。融合后不成立，必须改成四个正交方向之一。
#    - 原 side-touch 分支假设：只要检测到 `(r+dr,c)` 或 `(r,c+dc)` 的邻边墙，就翻对应轴。这依赖墙是
#      轴对齐的、且功能由“哪一侧有墙”编码。D7/2 + D2/0 一起破坏了这个前提：墙本身是对角镜面，功能由
#      朝向编码，不再有 side-touch / head-on 两套规则。
#    - 原 `wall_cells` 只需要知道“是不是墙”。融合后不够：solver 必须按 8 连通识别每个对角镜面组件，
#      判断它是 "/" 还是 "\"，然后再查反射表。
#    - 原代码默认墙可能是 1 像素或轴对齐 3 像素线段。融合后不成立；新 solver 新增假设：
#         * 每个有效墙都是 8 连通的 3 像素对角线段
#         * 同色墙之间不会 8 邻接粘成更大组件
#      generator 会显式保证。
#
# d) 放置合法性
#    - generator 继续 output-first：先规划每条正交折线路径和每次镜面反射，再反推出输入中的 L 箭头和
#      对角镜面。
#    - 镜面中心总在当前段末端正前方一格，因此“撞墙前一格改色、再从原地折返到垂直方向”的几何事件
#      与 solver 完全一致。
#    - 箭头、镜面、输出轨迹三者互不重叠；同色镜面之间还额外禁止 8 邻接，避免被 8 连通检测并成一个
#      非法大组件。
#    - 候选样例只有在 `main(input)` 精确复现规划输出时才会接受。
#
# e) 视觉可观察性
#    - 输入里能直接看到斜着放置的 "/" 与 "\" 镜面，而输出里新增的是水平/垂直折线，所以“斜镜面把
#      光路折 90 度”是肉眼可见的。
#    - 由于颜色 7 和 8 都会以两种朝向出现，同色镜面对不同路径弯折并不一致；这逼迫解题者关注几何朝向，
#      而不是把颜色误当成控制变量。
#    - 每个样例都覆盖水平来向和垂直来向，因此人类能看到完整的镜面规律，而不是只见到某一类特例。
#
# f) 可视化整洁度
#    - 保持 18-20 的稀疏方阵、固定 3 个箭头，减少拥挤。
#    - 每段直线长度至少为 2，尾段也至少为 2，所以每次折向都很清晰。
#    - 颜色角色分层明确：箭头初色来自非镜面色；镜面只用 7/8；反射后轨迹直接变成对应镜面色，肉眼能
#      跟踪“碰到哪面镜子就继承哪种色”。
#
# g) 变化的可推断性
#    - 颜色变化规则保持原题直观版：反射后轨迹色 = 被撞镜面色。由于镜面本体在输入里可见、输出里的新
#      线段紧接着从镜面前一格延伸，所以颜色继承有明确结构线索。
#    - 唯一真正需要归纳的新控制变量是镜面朝向。生成器过滤条件要求：
#         * 两种朝向都至少被使用 2 次
#         * 每种朝向都要被水平来向和垂直来向各撞到一次
#         * 每种镜面颜色都要出现在两种朝向上
#      从而让人类能在少量样例中看出“转向由朝向决定，颜色只负责继承”。
#
# 8) 融合行为量化指标
#    - `diff_vs_swapped_mirrors`: 把所有 "/" 和 "\" 的反射表互换后，再运行同一输入，输出与真规则的
#      像素差数量。若该值为 0，说明镜面朝向根本没起作用，样例退化。
#    - `axes_by_kind`: 每种镜面朝向是否都被水平/垂直来向击中过。
#    - `color_to_kinds`: 每种镜面颜色是否都在 "/" 与 "\" 两种朝向上重复出现。
#      generator 只接受这三项都达标的样例。
#
# 9) main() 的新隐式假设与 generator 对应保证
#    - 假设 1：箭头仍是孤立的单色 3 像素 L 形。generator 保证箭头互不重叠，且箭头色不与镜面色复用。
#    - 假设 2：有效墙都是 3 像素对角镜面。generator 只产出这种组件。
#    - 假设 3：同色镜面组件彼此不 8 邻接。generator 用同色 8 邻域检查保证。
#    - 假设 4：每次反射时，镜面中心都在正前方一格，所以 solver 的“前方撞镜再转向”模型成立。
#    - 假设 5：路径不会穿过其他箭头、镜面或已规划轨迹，也不会在网格内形成死循环。generator 用占用检查、
#      尾段可出界检查和最终 oracle 检查共同保证。

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


Coord = tuple[int, int]
Direction = tuple[int, int]

DIR_N = (-1, 0)
DIR_E = (0, 1)
DIR_S = (1, 0)
DIR_W = (0, -1)

CARDINAL_DIRS = (DIR_N, DIR_E, DIR_S, DIR_W)
HORIZONTAL_DIRS = (DIR_E, DIR_W)
VERTICAL_DIRS = (DIR_N, DIR_S)

MIRROR_SLASH = "slash"
MIRROR_BACK = "back"
MIRROR_KINDS = (MIRROR_SLASH, MIRROR_BACK)
SWAPPED_MIRRORS = {
    MIRROR_SLASH: MIRROR_BACK,
    MIRROR_BACK: MIRROR_SLASH,
}

WALL_COLORS = (7, 8)
ARROW_COLORS = (1, 2, 3, 4, 5, 6)

ARROW_OFFSETS = {
    DIR_E: [(-1, 0), (0, -1), (0, 0)],   # missing NW
    DIR_S: [(-1, 0), (0, 0), (0, 1)],    # missing NE
    DIR_W: [(0, 0), (0, 1), (1, 0)],     # missing SE
    DIR_N: [(0, -1), (0, 0), (1, 0)],    # missing SW
}

REFLECT_MAP = {
    MIRROR_SLASH: {
        DIR_N: DIR_E,
        DIR_E: DIR_N,
        DIR_S: DIR_W,
        DIR_W: DIR_S,
    },
    MIRROR_BACK: {
        DIR_N: DIR_W,
        DIR_W: DIR_N,
        DIR_S: DIR_E,
        DIR_E: DIR_S,
    },
}


def _in_bounds(H: int, W: int, r: int, c: int) -> bool:
    return 0 <= int(r) < H and 0 <= int(c) < W


def _is_arrow_pts(pts: list[Coord]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[Coord]) -> tuple[Direction, Coord] | None:
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


def _arrow_points_from_start(sr: int, sc: int, direction: Direction) -> list[Coord]:
    return [(int(sr + dr), int(sc + dc)) for dr, dc in ARROW_OFFSETS[direction]]


def _classify_diag_mirror(pts: list[Coord]) -> str | None:
    if len(pts) != 3:
        return None
    rs = sorted(r for r, _c in pts)
    cs = sorted(c for _r, c in pts)
    if not (rs == [rs[0], rs[0] + 1, rs[0] + 2] and cs == [cs[0], cs[0] + 1, cs[0] + 2]):
        return None
    r0, c0 = min(rs), min(cs)
    norm = {(int(r - r0), int(c - c0)) for r, c in pts}
    if norm == {(0, 0), (1, 1), (2, 2)}:
        return MIRROR_BACK
    if norm == {(0, 2), (1, 1), (2, 0)}:
        return MIRROR_SLASH
    return None


def _mirror_cells(center: Coord, kind: str) -> list[Coord]:
    r, c = center
    if kind == MIRROR_SLASH:
        return [(int(r - 1), int(c + 1)), (int(r), int(c)), (int(r + 1), int(c - 1))]
    return [(int(r - 1), int(c - 1)), (int(r), int(c)), (int(r + 1), int(c + 1))]


def _reflect(direction: Direction, kind: str) -> Direction:
    return REFLECT_MAP[str(kind)][direction]


def _detect_scene(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, list[Coord]]], set[Coord], dict[Coord, int], dict[int, str], dict[int, int]]:
    bg = int(get_bg(g))
    arrow_cells: set[Coord] = set()
    arrows: list[tuple[int, list[Coord]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_id_by_cell: dict[Coord, int] = {}
    wall_kind: dict[int, str] = {}
    wall_color: dict[int, int] = {}
    next_id = 0

    for col in colors_present(g, exclude={bg}):
        mask = (g == int(col)).copy()
        for r, c in arrow_cells:
            mask[r, c] = False
        if not mask.any():
            continue
        for coords in components_mask(mask, connectivity=8):
            pts = [(int(r), int(c)) for r, c in coords]
            kind = _classify_diag_mirror(pts)
            if kind is None:
                continue
            wid = next_id
            next_id += 1
            wall_kind[wid] = str(kind)
            wall_color[wid] = int(col)
            for cell in pts:
                wall_id_by_cell[cell] = wid

    return bg, arrows, arrow_cells, wall_id_by_cell, wall_kind, wall_color


def _trace_variant(
    grid: np.ndarray,
    *,
    swap_mirrors: bool = False,
    collect_events: bool = False,
):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells, wall_id_by_cell, wall_kind, wall_color = _detect_scene(g)
    out = np.array(g, copy=True)
    events: list[dict[str, object]] = []

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        direction, cur = info
        cur_col = int(start_col)
        seen: set[tuple[Coord, Direction, int]] = set()

        while True:
            state = (cur, direction, int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + direction[0])
            nc = int(cur[1] + direction[1])
            if not _in_bounds(H, W, nr, nc):
                break

            if (nr, nc) in arrow_cells:
                break

            if (nr, nc) in wall_id_by_cell:
                wid = int(wall_id_by_cell[(nr, nc)])
                actual_kind = str(wall_kind[wid])
                used_kind = SWAPPED_MIRRORS[actual_kind] if swap_mirrors else actual_kind
                wcol = int(wall_color[wid])
                next_direction = _reflect(direction, used_kind)
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                if collect_events:
                    events.append(
                        {
                            "incoming": direction,
                            "incoming_axis": "h" if direction[1] != 0 else "v",
                            "wall_color": int(wcol),
                            "wall_kind": actual_kind,
                            "used_kind": used_kind,
                            "bounce_from": cur,
                            "contact": (int(nr), int(nc)),
                            "outgoing": next_direction,
                        }
                    )
                cur_col = int(wcol)
                direction = next_direction
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

    if collect_events:
        return out, events
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace orthogonal rays from L-shaped arrows and reflect them on diagonal mirrors.

    Fused rule:
    - Each L-arrow decodes to one of four cardinal launch directions.
    - Rays advance one cell at a time horizontally or vertically.
    - Every wall is a 3-pixel diagonal mirror, either "/" or "\".
    - When a ray would enter any mirror pixel, it stays on the previous cell,
      recolors that cell to the mirror color, adopts that color, and turns by the
      physical reflection law for that mirror orientation.
    - Rays stop when they leave the grid or would step into another arrow.
    """
    return _trace_variant(grid, swap_mirrors=False, collect_events=False)


def _steps_to_edge(r: int, c: int, direction: Direction, H: int, W: int) -> int:
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
    direction: Direction,
    seg_len: int,
    blocked: set[Coord],
    H: int,
    W: int,
) -> list[Coord] | None:
    dr, dc = direction
    rr, cc = int(cur_r), int(cur_c)
    cells: list[Coord] = []
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
    direction: Direction,
    blocked: set[Coord],
    H: int,
    W: int,
) -> list[Coord] | None:
    dr, dc = direction
    rr, cc = int(cur_r), int(cur_c)
    cells: list[Coord] = []
    while True:
        rr += int(dr)
        cc += int(dc)
        if not _in_bounds(H, W, rr, cc):
            return cells
        if (rr, cc) in blocked:
            return None
        cells.append((int(rr), int(cc)))


def _same_color_clear(
    cells: list[Coord],
    global_same_color: set[Coord],
    local_same_color: set[Coord],
) -> bool:
    occupied = global_same_color | local_same_color
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (int(r + dr), int(c + dc)) in occupied:
                    return False
    return True


def _valid_start_positions(H: int, W: int, direction: Direction) -> list[Coord]:
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
    start_direction: Direction,
    bounce_plan: list[tuple[str, int]],
    occupied: set[Coord],
    wall_color_cells_global: dict[int, set[Coord]],
) -> tuple[list[Coord], list[tuple[int, int, int]], list[tuple[list[Coord], int, str]]] | None:
    start_positions = _valid_start_positions(H, W, start_direction)
    _rng.shuffle(start_positions)

    for sr, sc in start_positions[:220]:
        arrow_pts = _arrow_points_from_start(int(sr), int(sc), start_direction)
        if any((r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = int(sr), int(sc)
        direction = start_direction
        cur_col = int(color)
        ray_cells: list[tuple[int, int, int]] = []
        ray_set: set[Coord] = set()
        walls: list[tuple[list[Coord], int, str]] = []
        wall_cells_local: set[Coord] = set()
        wall_color_cells_local = {int(col): set() for col in WALL_COLORS}
        ok = True

        for mirror_kind, wall_color in bounce_plan:
            blocked = occupied | set(arrow_pts) | ray_set | wall_cells_local
            max_steps = _steps_to_edge(cur_r, cur_c, direction, H, W)
            if max_steps < 3:
                ok = False
                break

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
                wall_cells = _mirror_cells(center, str(mirror_kind))
                if not all(_in_bounds(H, W, rr, cc) for rr, cc in wall_cells):
                    continue
                if any(cell in blocked or cell in set(seg_cells) for cell in wall_cells):
                    continue
                if not _same_color_clear(
                    wall_cells,
                    wall_color_cells_global[int(wall_color)],
                    wall_color_cells_local[int(wall_color)],
                ):
                    continue

                next_direction = _reflect(direction, str(mirror_kind))
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
                walls.append((wall_cells, int(wall_color), str(mirror_kind)))
                wall_cells_local.update(wall_cells)
                wall_color_cells_local[int(wall_color)].update(wall_cells)

                lr, lc, _old = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wall_color))
                cur_r, cur_c = int(end_r), int(end_c)
                direction = next_direction
                cur_col = int(wall_color)
                committed = True
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

        return arrow_pts, ray_cells, walls

    return None


def _collect_events(grid: np.ndarray) -> list[dict[str, object]]:
    _out, events = _trace_variant(grid, swap_mirrors=False, collect_events=True)
    return events


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    out_true = main(grid)
    out_swapped = _trace_variant(grid, swap_mirrors=True, collect_events=False)
    events = _collect_events(grid)

    kind_use_counts = {
        kind: sum(int(str(e["wall_kind"]) == kind) for e in events)
        for kind in MIRROR_KINDS
    }
    color_use_counts = {
        int(col): sum(int(int(e["wall_color"]) == int(col)) for e in events)
        for col in WALL_COLORS
    }
    axes_by_kind = {
        kind: {str(e["incoming_axis"]) for e in events if str(e["wall_kind"]) == kind}
        for kind in MIRROR_KINDS
    }
    color_to_kinds = {
        int(col): {str(e["wall_kind"]) for e in events if int(e["wall_color"]) == int(col)}
        for col in WALL_COLORS
    }
    kind_to_colors = {
        kind: {int(e["wall_color"]) for e in events if str(e["wall_kind"]) == kind}
        for kind in MIRROR_KINDS
    }

    return {
        "turn_count": len(events),
        "diff_vs_swapped_mirrors": int(np.sum(out_true != out_swapped)),
        "kind_use_counts": kind_use_counts,
        "color_use_counts": color_use_counts,
        "axes_by_kind": axes_by_kind,
        "color_to_kinds": color_to_kinds,
        "kind_to_colors": kind_to_colors,
        "distinct_event_types": len(
            {
                (str(e["incoming_axis"]), str(e["wall_kind"]), int(e["wall_color"]))
                for e in events
            }
        ),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[Coord] = set()
    wall_color_cells_global = {int(col): set() for col in WALL_COLORS}

    arrow_colors = list(ARROW_COLORS)
    _rng.shuffle(arrow_colors)
    arrow_colors = arrow_colors[:3]

    dir_a = _rng.choice(HORIZONTAL_DIRS)
    dir_b = _rng.choice(VERTICAL_DIRS)
    remaining = [d for d in CARDINAL_DIRS if d not in {dir_a, dir_b}]
    dir_c = _rng.choice(remaining)
    start_directions = [dir_a, dir_b, dir_c]
    _rng.shuffle(start_directions)

    bounce_plans = [
        [(MIRROR_SLASH, 7), (MIRROR_BACK, 7)],
        [(MIRROR_SLASH, 8), (MIRROR_BACK, 8)],
        [(_rng.choice(MIRROR_KINDS), int(_rng.choice(WALL_COLORS)))],
    ]
    _rng.shuffle(bounce_plans)

    for color, start_direction, bounce_plan in zip(arrow_colors, start_directions, bounce_plans):
        planned = _plan_arrow(
            H,
            W,
            int(color),
            start_direction,
            bounce_plan,
            occupied,
            wall_color_cells_global,
        )
        if planned is None:
            return None

        arrow_pts, ray_cells, walls = planned
        for r, c in arrow_pts:
            grid[r, c] = int(color)
            out[r, c] = int(color)
        for r, c, ray_col in ray_cells:
            out[r, c] = int(ray_col)
        for wall_cells, wall_color, _kind in walls:
            for r, c in wall_cells:
                grid[r, c] = int(wall_color)
                out[r, c] = int(wall_color)
                wall_color_cells_global[int(wall_color)].add((int(r), int(c)))

        occupied |= set(arrow_pts)
        occupied |= {(int(r), int(c)) for r, c, _ray_col in ray_cells}
        occupied |= {(int(r), int(c)) for wall_cells, _wcol, _kind in walls for r, c in wall_cells}

    metrics = _behavior_metrics(grid)
    if int(metrics["turn_count"]) < 5:
        return None
    if int(metrics["diff_vs_swapped_mirrors"]) < 4:
        return None
    if any(int(metrics["kind_use_counts"][kind]) < 2 for kind in MIRROR_KINDS):
        return None
    if any(int(metrics["color_use_counts"][col]) < 2 for col in WALL_COLORS):
        return None
    if any(metrics["axes_by_kind"][kind] != {"h", "v"} for kind in MIRROR_KINDS):
        return None
    if any(metrics["color_to_kinds"][col] != set(MIRROR_KINDS) for col in WALL_COLORS):
        return None
    if any(metrics["kind_to_colors"][kind] != set(WALL_COLORS) for kind in MIRROR_KINDS):
        return None
    if int(metrics["distinct_event_types"]) < 4:
        return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(900):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 900 attempts")
