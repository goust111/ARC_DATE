# Combo variant of 142ca369 (index 7)
# combo_spec: D8/0+D10/0
# Name: 距离排序发射+墙壁反射后消失

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis
# combo_rule:
#   - 保留原题对象词汇：3 像素 L 形箭头、单色墙像素、对角线射线、侧碰/正撞反射。
#   - 融合后的关键区别不是“射线轨迹互相变成墙”，而是“箭头按距离顺序发射，并共享一套会被擦除的活墙集合”。
#   - 每次命中墙时：几何反射规则不变；射线继承墙色；被真正触碰的墙像素变成背景色。
#   - 后续箭头看到的是更新后的墙场，所以顺序依赖来自“先发射者把后发射者本来会碰到的墙擦掉了”。
# concrete_trace:
#   - 设共享墙像素 W 在接近中心的位置。
#   - 近中心箭头 A 先发射，走到 W 左侧相邻格，因侧碰 W 而反射；输出中 W 被擦成 0，A 后续线段继承 W 的颜色。
#   - 远中心箭头 B 后发射，原本它会在 W 上方相邻格再次侧碰该墙；但现在 W 已不存在，所以 B 不再转向，直接沿原对角线继续。
#   - 原题/独立处理：B 会反射，且 bounce 点会被改成墙色。
#   - 本融合版：B 不反射；W 在输出中缺失；A 的转折后线段换成墙色，B 的几何则因“前一步擦墙”而改变。
# complexity_calibration:
#   - 生成器固定为 2 个箭头，避免多箭头多墙叠加导致顺序线索被淹没。
#   - 共享墙放在网格中心附近，保证“离中心更近的箭头先发射”有明确视觉参照。
#   - 远箭头被构造成必须依赖同一墙像素，否则输出会退化成原题近似；因此每个样例都稳定体现两项改动。
#   - 仍允许 18-20 的原尺寸范围和原箭头形状，不改变任务阅读方式。
# implicit_assumption_audit:
#   - 原 `for start_col, pts in arrows:` 默认箭头彼此独立，顺序无关；融合后必须按中心距离排序。
#   - 原 `wall_cells` 是静态集合；融合后必须改为所有箭头共享的 `live_walls`，并在命中时删除像素。
#   - 原 “反射时把当前 ray cell 重涂成墙色” 的假设失效；现在 bounce cell 保持到达时的射线颜色，只有后续段继承墙色。
#   - 原去环状态 `(cur, dr, dc, cur_col)` 默认环境不变；现在同一局部状态在不同剩余墙集合下语义不同，必须把墙签名并入状态。
#   - 原代码把所有非箭头像素都当墙；本版本仍沿用，但生成器必须保证没有别的伪对象混入。
# placement_legality:
#   - 共享墙既是 A 的真实反射点，也是 B 的“本该反射但已消失”的关键控制位。
#   - 箭头、墙、两条射线的计划轨迹在输入中不重叠；输出中两条射线也不相交，避免颜色覆盖歧义。
#   - 四种对角朝向通过同一局部模板镜像得到，保证物理关系不变且始终在边界内。
# observability:
#   - 人类能直接看到：一个靠中心的箭头先在某墙处折返并留下颜色继承；同一个墙在输出中消失；另一个更远的箭头穿过本来会被该墙影响的位置却没有转向。
#   - 排序依据只用几何距离到中心，不用颜色值等不可视因素。
# cleanliness:
#   - 每个样例只有 1 个关键墙和 2 条不相交射线，局部因果关系集中，输出不会变成杂乱的多层折线。
# new_solver_assumptions_and_generator_guards:
#   - solver 假设箭头按中心距离可稳定排序；生成器让近箭头与远箭头距离差很大，避免 tie。
#   - solver 假设被擦除墙不会再次作为必要反射点；生成器刻意让 B 依赖“墙已消失”，而不会要求同一墙像素在输出中二次保留。
#   - solver 假设 later rays 不需要把 earlier ray 当障碍；生成器构造了互不相交的两条输出轨迹来匹配这一点.


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


def _center_dist2(r: int, c: int, H: int, W: int) -> int:
    return int((2 * r - (H - 1)) ** 2 + (2 * c - (W - 1)) ** 2)


def _order_key(start: tuple[int, int], H: int, W: int, color: int) -> tuple[int, int, int, int]:
    r, c = start
    return (_center_dist2(r, c, H, W), int(r), int(c), int(color))


def _arrow_points_from_start(sr: int, sc: int, dr: int, dc: int) -> list[tuple[int, int]]:
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _detect_arrows(g: np.ndarray, bg: int) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    H, W = g.shape
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[dict[str, object]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            direction, start = info
            arrow_cells |= set(pts)
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "dir": direction,
                    "start": start,
                    "order_key": _order_key(start, H, W, int(col)),
                }
            )
    return arrows, arrow_cells


def _trace_single_ray(
    out: np.ndarray,
    bg: int,
    arrow_cells: set[tuple[int, int]],
    live_walls: dict[tuple[int, int], int],
    start_col: int,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict[str, object]:
    H, W = out.shape
    cur = (int(start[0]), int(start[1]))
    dr, dc = int(direction[0]), int(direction[1])
    cur_col = int(start_col)

    erased_cells: list[tuple[int, int]] = []
    seen: set[tuple[tuple[int, int], int, int, int, tuple[tuple[int, int], ...]]] = set()

    def erase_wall(cell: tuple[int, int]) -> int:
        erased_cells.append(cell)
        wcol = int(live_walls.pop(cell))
        out[cell[0], cell[1]] = int(bg)
        return wcol

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

        nxt = (nr, nc)
        if nxt in live_walls:
            cur_col = erase_wall(nxt)
            dr, dc = -int(dr), -int(dc)
            continue

        if nxt in arrow_cells:
            break

        cur = nxt
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

    return {"erased_cells": erased_cells}


def _trace_all(grid: np.ndarray, *, shared_live_walls: bool) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells = _detect_arrows(g, bg)
    initial_walls = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()
    arrows_sorted = sorted(arrows, key=lambda a: a["order_key"])
    shared_walls = dict(initial_walls)
    per_arrow: list[dict[str, object]] = []

    for arrow in arrows_sorted:
        live_walls = shared_walls if shared_live_walls else dict(initial_walls)
        ray_info = _trace_single_ray(
            out=out,
            bg=bg,
            arrow_cells=arrow_cells,
            live_walls=live_walls,
            start_col=int(arrow["color"]),
            start=arrow["start"],
            direction=arrow["dir"],
        )
        per_arrow.append(
            {
                "color": int(arrow["color"]),
                "start": arrow["start"],
                "order_key": arrow["order_key"],
                "erased_cells": list(ray_info["erased_cells"]),
            }
        )

    meta = {
        "per_arrow": per_arrow,
        "erased_total": int(sum(len(a["erased_cells"]) for a in per_arrow)),
        "remaining_walls": int(len(shared_walls if shared_live_walls else initial_walls)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire arrows in increasing distance-to-center order.

    Reflection geometry matches the original task, but all arrows share one mutable wall set:
    when a wall pixel is touched, that pixel is erased and later arrows no longer see it.
    """
    return _trace_all(grid, shared_live_walls=True)[0]


def _shared_wall_anchor(size: int, step_sign: int) -> int:
    if step_sign == 1:
        return 10 if size <= 19 else 11
    return size - 11


def _scene_spec(H: int, W: int, dr: int, dc: int) -> dict[str, object]:
    wr = _shared_wall_anchor(H, dr)
    wc = _shared_wall_anchor(W, dc)
    wall = (int(wr), int(wc))

    a_start = (int(wr - 3 * dr), int(wc - 4 * dc))
    b_start = (int(wr - 9 * dr), int(wc - 8 * dc))

    return {
        "wall": wall,
        "a_start": a_start,
        "b_start": b_start,
        "dir": (int(dr), int(dc)),
    }


def _input_from_scene(
    H: int,
    W: int,
    a_color: int,
    b_color: int,
    wall_color: int,
    dr: int,
    dc: int,
) -> np.ndarray | None:
    spec = _scene_spec(H, W, dr, dc)
    wall = spec["wall"]
    a_start = spec["a_start"]
    b_start = spec["b_start"]

    grid = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for color, start in ((a_color, a_start), (b_color, b_start)):
        pts = _arrow_points_from_start(int(start[0]), int(start[1]), int(dr), int(dc))
        if len(set(pts)) != 3:
            return None
        for r, c in pts:
            if not (0 <= r < H and 0 <= c < W):
                return None
            if (r, c) in occupied:
                return None
        for r, c in pts:
            grid[r, c] = int(color)
            occupied.add((r, c))

    wr, wc = wall
    if not (0 <= wr < H and 0 <= wc < W):
        return None
    if wall in occupied:
        return None
    grid[wr, wc] = int(wall_color)
    occupied.add(wall)
    return grid


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    ordered_out, ordered_meta = _trace_all(grid, shared_live_walls=True)
    independent_out, _ = _trace_all(grid, shared_live_walls=False)

    diff_cells = int(np.count_nonzero(ordered_out != independent_out))
    erased_cells = int(np.count_nonzero((grid != 0) & (ordered_out == 0)))
    per_arrow = ordered_meta["per_arrow"]
    first_erased = len(per_arrow[0]["erased_cells"]) if per_arrow else 0
    second_erased = len(per_arrow[1]["erased_cells"]) if len(per_arrow) > 1 else 0
    return {
        "diff_vs_independent": int(diff_cells),
        "erased_cells": int(erased_cells),
        "first_erased": int(first_erased),
        "second_erased": int(second_erased),
    }


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    a_color, b_color, wall_color = [int(c) for c in palette[:3]]
    dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])

    grid = _input_from_scene(
        H=H,
        W=W,
        a_color=a_color,
        b_color=b_color,
        wall_color=wall_color,
        dr=int(dr),
        dc=int(dc),
    )
    if grid is None:
        return None

    solved = main(grid)
    if solved.shape != grid.shape:
        return None

    arrows, _arrow_cells = _detect_arrows(grid, bg=0)
    if len(arrows) != 2:
        return None
    arrows_sorted = sorted(arrows, key=lambda a: a["order_key"])
    if arrows_sorted[0]["start"] == arrows_sorted[1]["start"]:
        return None

    metrics = _combo_metrics(grid)
    if metrics["erased_cells"] < 1:
        return None
    if metrics["first_erased"] < 1:
        return None
    if metrics["second_erased"] != 0:
        return None
    if metrics["diff_vs_independent"] < 4:
        return None

    independent_out, _ = _trace_all(grid, shared_live_walls=False)
    if np.array_equal(independent_out, solved):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 200 attempts")
