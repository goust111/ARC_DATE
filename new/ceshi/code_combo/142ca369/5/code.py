# Combo variant of 142ca369 (index 5)
# combo_spec: D7/0+D15/0
# Name: L/T形墙壁+功能分化

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 箭头保持原题不变：每个箭头都是 2x2 包围盒里缺一角的 3 像素 L 形，
#      缺角决定射线的对角发射方向，直角顶点是射线起点。
#    - 墙不再是点/线段，而是 4 像素连通块，并且只允许两种形状：
#         L tetromino -> 反射墙
#         T tetromino -> 穿透墙
#    - 射线仍按原题逐格对角前进，但现在必须先识别“接触到的连通块是什么形状”。
#      例子：当前射线方向是 (+1,+1)，走到 (r,c) 后：
#         若 (r+1,c) 属于某个 L 形墙，则该格改成墙色并翻转 dr，后续方向变成 (-1,+1)。
#         若 (r+1,c) 属于某个 T 形墙，则该格改成墙色，但方向不变，后续仍是 (+1,+1)。
#      原题在这两种情况下都会反射；融合后同样的几何接触在 L/T 上出现了功能分化。
#    - 这里保留了 D7 的“必须看墙的形状”，也保留了 D15 的“不同墙类型功能不同”，
#      但功能区分不再靠尺寸，而是靠 L/T 轮廓本身。
#
# b) 复杂度校准
#    - 只保留两种 4 像素墙形，视觉词汇很小，解题者能从少量样例里稳定识别。
#    - 每个样例都强制同时出现至少 1 个 L 墙事件和 1 个 T 墙事件，避免退化回单规则。
#    - 生成器只产出“单侧接触”的墙：要么接触 (r+dr,c)，要么接触 (r,c+dc)。
#      这样反射方向由接触边决定，穿透现象也更容易肉眼观察。
#
# c) 对原 solver 隐式假设的审查
#    - 原 solver 把所有非箭头像素放进一个 `wall_cells` 集合，隐含“所有墙等价”。
#      这里不成立，必须按 4 邻域连通块检测墙，并按归一化轮廓分类成 L/T。
#    - 原 solver 的 side-touch 代码默认“碰墙必翻轴”。这里不成立：
#      只有 L 墙翻轴，T 墙只改色不翻轴。
#    - 原 solver 只关心墙像素是否存在，不关心它们属于哪个结构。这里不成立：
#      同一个 T/L 连通块的所有像素必须被归为同一堵墙，否则无法根据形状决定功能。
#
# d) 放置合法性
#    - 箭头、墙、输出中的射线路径都不重叠。
#    - 任意新墙都不能占用先前射线状态的“侧碰槽位”，否则会对旧路径产生隐藏触发。
#    - 新墙的下一步前进方向必须是空的，避免把本应是 side-touch 的事件变成 head-on。
#    - 同色 4 邻接并块被禁止，保证 per-color 连通块检测不会把箭头和墙粘成一个对象。
#
# e) 视觉可观察性
#    - L 墙附近：射线在接触点改色并拐弯。
#    - T 墙附近：射线在接触点改色但继续沿同一对角线前进。
#    - 两类墙本身轮廓也不同，所以解题者可以从“轮廓 + 输出运动差异”共同归纳规则。
#
# f) 可视化整洁度
#    - 只用 2-3 支箭和 3-4 堵墙，避免网格过密。
#    - 通过保留接触槽位和禁止未来路径贴墙，避免一个区域同时承载多重交互。
#
# g) 颜色变化可推断性
#    - 无论 L 还是 T，射线的新颜色都直接继承当前接触墙的颜色，没有隐藏映射表。
#    - 生成器尽量避免“墙色 == 当前射线色”，让改色现象在样例中可见。
#
# h) 融合行为量化指标
#    - 关键指标是每个样例中 T 墙触发次数、L 墙触发次数，以及
#      “把 T 错当成 L 反射墙时，输出会不会改变”。
#    - 如果 T 墙经常存在但与“全反射”反事实 solver 得到同样输出，说明生成器退化了。
#
# i) 新 solver 的隐式假设与生成器保证
#    - 假设 1：所有非箭头连通块都是大小 4 的合法 L/T 墙。生成器只产出这两类块。
#    - 假设 2：每个射线接触点至多触发一堵墙。生成器通过侧碰槽位保留保证这一点。
#    - 假设 3：箭头与墙不会同色 4 邻接。生成器在选墙色时检查相邻输入像素。
#    - 假设 4：虽然 solver 支持 T 墙 head-on 穿透，但生成器只构造 side-touch 的 T，
#      这样样例中的功能分化更直观。


Coord = tuple[int, int]


def _normalize_shape(cells: set[Coord]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(cells: set[Coord], rot: int, reflect: bool) -> set[Coord]:
    out = set(cells)
    if reflect:
        out = {(r, -c) for r, c in out}
    for _ in range(rot % 4):
        out = {(c, -r) for r, c in out}
    return set(_normalize_shape(out))


def _all_orientations(base: set[Coord], *, allow_reflect: bool) -> list[tuple[Coord, ...]]:
    variants: set[tuple[Coord, ...]] = set()
    for rot in range(4):
        variants.add(_normalize_shape(_transform_shape(base, rot, False)))
        if allow_reflect:
            variants.add(_normalize_shape(_transform_shape(base, rot, True)))
    return sorted(variants)


_BASE_L4 = {(0, 0), (1, 0), (2, 0), (2, 1)}
_BASE_T4 = {(0, 0), (0, 1), (0, 2), (1, 1)}
_WALL_VARIANTS = {
    "L": _all_orientations(_BASE_L4, allow_reflect=True),
    "T": _all_orientations(_BASE_T4, allow_reflect=False),
}
_L_SHAPES = {variant for variant in _WALL_VARIANTS["L"]}
_T_SHAPES = {variant for variant in _WALL_VARIANTS["T"]}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _max_diag_steps(H: int, W: int, r: int, c: int, dr: int, dc: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


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
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _classify_wall_shape(pts: list[Coord]) -> str | None:
    if len(pts) != 4:
        return None
    norm = _normalize_shape(set(pts))
    if norm in _L_SHAPES:
        return "L"
    if norm in _T_SHAPES:
        return "T"
    return None


def _touch_axes(wall_cells: set[Coord], nr: int, nc: int, dr: int, dc: int) -> set[str]:
    axes = set()
    if (nr + dr, nc) in wall_cells:
        axes.add("dr")
    if (nr, nc + dc) in wall_cells:
        axes.add("dc")
    return axes


def _detect_scene(g: np.ndarray, bg: int):
    arrow_cells: set[Coord] = set()
    arrow_sigs: set[tuple[Coord, ...]] = set()
    arrows: list[tuple[int, list[Coord]]] = []

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                sig = tuple(sorted(pts))
                arrows.append((int(col), pts))
                arrow_sigs.add(sig)
                arrow_cells |= set(pts)

    wall_id_by_cell: dict[Coord, int] = {}
    wall_kind: dict[int, str] = {}
    wall_color: dict[int, int] = {}
    wall_cells_by_id: dict[int, set[Coord]] = {}
    wid = 0

    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            sig = tuple(sorted(pts))
            if sig in arrow_sigs:
                continue
            kind = _classify_wall_shape(pts)
            if kind is None:
                continue
            wall_kind[wid] = kind
            wall_color[wid] = int(col)
            wall_cells = set(pts)
            wall_cells_by_id[wid] = wall_cells
            for cell in wall_cells:
                wall_id_by_cell[cell] = wid
            wid += 1

    return arrows, arrow_cells, wall_id_by_cell, wall_kind, wall_color, wall_cells_by_id


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - All walls are 4-cell L/T polyominoes.
    - L walls behave like normal reflective walls: side-touch flips the touched axis,
      head-on reverses both axes, and the ray inherits the wall color.
    - T walls are permeable: any touch/head-on only changes the ray color; direction
      stays unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells, wall_id_by_cell, wall_kind, wall_color, wall_cells_by_id = _detect_scene(g, bg)
    wall_cells = set(wall_id_by_cell)
    out = g.copy()

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in arrow_cells:
                break

            if (nr, nc) in wall_cells:
                wid = wall_id_by_cell[(nr, nc)]
                wcol = int(wall_color[wid])
                if wall_kind[wid] == "T":
                    cur = (int(nr), int(nc))
                    cur_col = int(wcol)
                    continue
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            touched: list[tuple[int, str, set[str]]] = []
            row_side = (nr + int(dr), nc)
            col_side = (nr, nc + int(dc))
            candidate_wids = []
            if row_side in wall_id_by_cell:
                candidate_wids.append(wall_id_by_cell[row_side])
            if col_side in wall_id_by_cell:
                wid2 = wall_id_by_cell[col_side]
                if wid2 not in candidate_wids:
                    candidate_wids.append(wid2)

            for wid in candidate_wids:
                axes = _touch_axes(wall_cells_by_id[wid], nr, nc, int(dr), int(dc))
                if axes:
                    touched.append((wid, wall_kind[wid], axes))

            if not touched:
                continue

            l_hits = [item for item in touched if item[1] == "L"]
            if l_hits:
                chosen_wid, _, axes = min(l_hits, key=lambda item: (-len(item[2]), item[0]))
            else:
                chosen_wid, _, axes = min(touched, key=lambda item: item[0])
                axes = set()

            wcol = int(wall_color[chosen_wid])
            if int(g[nr, nc]) == int(bg):
                out[nr, nc] = int(wcol)
            cur_col = int(wcol)
            if "dr" in axes:
                dr = -int(dr)
            if "dc" in axes:
                dc = -int(dc)

    return out


def _touch_positions_from_states(states: list[tuple[int, int, int, int]]) -> set[Coord]:
    out = set()
    for r, c, dr, dc in states:
        out.add((int(r + dr), int(c)))
        out.add((int(r), int(c + dc)))
    return out


def _relative_wall_candidates(shape_name: str, dr: int, dc: int, touch_mode: str) -> list[tuple[Coord, ...]]:
    if touch_mode == "row":
        mandatory = {(dr, 0)}
        forbidden = {(0, dc)}
    else:
        mandatory = {(0, dc)}
        forbidden = {(dr, 0)}

    opts: set[tuple[Coord, ...]] = set()
    first_req = next(iter(mandatory))
    for variant in _WALL_VARIANTS[shape_name]:
        pts = list(variant)
        for vr, vc in pts:
            shift_r = first_req[0] - vr
            shift_c = first_req[1] - vc
            shifted = tuple(sorted((r + shift_r, c + shift_c) for r, c in pts))
            shifted_set = set(shifted)
            if (0, 0) in shifted_set:
                continue
            if mandatory <= shifted_set and forbidden.isdisjoint(shifted_set):
                opts.add(shifted)
    return sorted(opts)


def _make_arrow(start: Coord, dr: int, dc: int) -> list[Coord]:
    sr, sc = start
    mr, mc = sr - dr, sc - dc
    return [(int(r), int(c)) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _pick_wall_color(
    palette: list[int],
    wall_cells: set[Coord],
    local_input_colors: dict[Coord, int],
    grid: np.ndarray,
    cur_col: int,
) -> int | None:
    forbidden_same_color: set[int] = set()
    for r, c in wall_cells:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in local_input_colors:
                forbidden_same_color.add(int(local_input_colors[(rr, cc)]))
            elif 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(grid[rr, cc]) != 0:
                forbidden_same_color.add(int(grid[rr, cc]))

    prefer = [col for col in palette if col not in forbidden_same_color and col != int(cur_col)]
    if prefer:
        return int(_rng.choice(prefer))
    fallback = [col for col in palette if col not in forbidden_same_color]
    if fallback:
        return int(_rng.choice(fallback))
    return None


def _plan_for_arrow(index: int) -> list[tuple[str, str]]:
    side = lambda: _rng.choice(["row", "col"])
    if index == 0:
        return [("T", side()), ("L", side())]
    if index == 1:
        plan = [("L", side())]
        if _rng.random() < 0.6:
            plan.append(("T", side()))
        return plan
    n_extra = _rng.choice([1, 2])
    return [(_rng.choice(["L", "T"]), side()) for _ in range(n_extra)]


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.choice([2, 3])
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied_all: set[Coord] = set()
    occupied_walls: set[Coord] = set()
    committed_ray_states: list[tuple[int, int, int, int]] = []

    used_L = False
    used_T = False

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        bounce_plan = _plan_for_arrow(ai)

        placed = False
        for _ in range(180):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue

            start = (_rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi))
            arrow_pts = _make_arrow(start, dr, dc)
            arrow_set = set(arrow_pts)
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied_all for r, c in arrow_pts):
                continue

            cur_r, cur_c, cur_dr, cur_dc, cur_col = int(start[0]), int(start[1]), int(dr), int(dc), int(color)
            ray_cells: list[tuple[int, int, int]] = []
            ray_states: list[tuple[int, int, int, int]] = []
            ray_set: set[Coord] = set()
            walls: list[tuple[set[Coord], int, str, str]] = []
            local_wall_set: set[Coord] = set()
            local_input_colors: dict[Coord, int] = {cell: color for cell in arrow_set}
            ok = True

            for seg_idx in range(len(bounce_plan) + 1):
                max_steps = _max_diag_steps(H, W, cur_r, cur_c, cur_dr, cur_dc)
                if seg_idx < len(bounce_plan):
                    min_seg = 2
                    max_seg = max_steps - 1
                    if max_seg < min_seg:
                        ok = False
                        break
                    seg_len = _rng.randint(min_seg, max_seg)
                else:
                    seg_len = max_steps

                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied_all or (nr, nc) in ray_set or (nr, nc) in arrow_set or (nr, nc) in local_wall_set:
                        ok = False
                        break
                    if (nr + cur_dr, nc) in occupied_walls or (nr + cur_dr, nc) in local_wall_set:
                        ok = False
                        break
                    if (nr, nc + cur_dc) in occupied_walls or (nr, nc + cur_dc) in local_wall_set:
                        ok = False
                        break

                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_states.append((int(nr), int(nc), int(cur_dr), int(cur_dc)))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)

                if not ok:
                    break

                if seg_idx >= len(bounce_plan):
                    continue

                shape_name, touch_mode = bounce_plan[seg_idx]
                remaining_bounces = len(bounce_plan) - seg_idx - 1
                reserved_touch = _touch_positions_from_states(committed_ray_states + ray_states[:-1])
                bounce_options: list[tuple[set[Coord], int, int]] = []

                for rel_cells in _relative_wall_candidates(shape_name, cur_dr, cur_dc, touch_mode):
                    abs_cells = {(cur_r + rr, cur_c + cc) for rr, cc in rel_cells}
                    if any(not (0 <= r < H and 0 <= c < W) for r, c in abs_cells):
                        continue
                    if abs_cells & occupied_all or abs_cells & ray_set or abs_cells & arrow_set or abs_cells & local_wall_set:
                        continue
                    if abs_cells & reserved_touch:
                        continue

                    next_dr, next_dc = int(cur_dr), int(cur_dc)
                    if shape_name == "L":
                        if touch_mode == "row":
                            next_dr = -next_dr
                        else:
                            next_dc = -next_dc

                    next_cell = (cur_r + next_dr, cur_c + next_dc)
                    if next_cell in abs_cells or next_cell in occupied_all or next_cell in arrow_set or next_cell in local_wall_set:
                        continue

                    future_steps = _max_diag_steps(H, W, cur_r, cur_c, next_dr, next_dc)
                    need = 2 if remaining_bounces > 0 else 1
                    if future_steps < need:
                        continue

                    bounce_options.append((abs_cells, next_dr, next_dc))

                if not bounce_options:
                    ok = False
                    break

                wall_cells, next_dr, next_dc = _rng.choice(bounce_options)
                wcol = _pick_wall_color(palette, wall_cells, local_input_colors, grid, cur_col)
                if wcol is None:
                    ok = False
                    break

                walls.append((wall_cells, int(wcol), shape_name, touch_mode))
                for cell in wall_cells:
                    local_wall_set.add(cell)
                    local_input_colors[cell] = int(wcol)

                lr, lc, _ = ray_cells[-1]
                ray_cells[-1] = (int(lr), int(lc), int(wcol))
                cur_col = int(wcol)
                cur_dr, cur_dc = int(next_dr), int(next_dc)

            if not ok or len(ray_cells) < 4:
                continue

            new_cells = set(arrow_set) | set(ray_set) | set(local_wall_set)
            if new_cells & occupied_all:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wall_cells, wcol, shape_name, touch_mode in walls:
                for r, c in wall_cells:
                    grid[r, c] = out[r, c] = int(wcol)
                if shape_name == "L":
                    used_L = True
                else:
                    used_T = True

            occupied_all |= new_cells
            occupied_walls |= local_wall_set
            committed_ray_states.extend(ray_states)
            placed = True
            break

        if not placed:
            return None

    if not (used_L and used_T):
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
