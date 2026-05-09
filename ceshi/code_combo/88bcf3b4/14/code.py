# Combo variant of 88bcf3b4 (index 14)
# combo_spec: D3/1+D4/0
# Name: 直线源虚线输出
#
# Design analysis
# 1. Concrete behavior:
#    - 原题输入里的 source 是一条贴着 bar terminal 的不规则细线；这里把它收紧成一条轴对齐直线，
#      但仍然从 terminal 的“侧边”接出，而不是沿 bar 的前进方向接出。
#    - 这样可保持原题最关键的 root-side 几何：若完整输出 route 为
#      [(0,1),(1,2),(2,3),(3,3),(4,2)]（局部坐标，top-edge 且 root 在右侧），
#      原题输入可能是同长度折线；融合后输入改成 [(0,1),(0,2),(0,3),(0,4),(0,5)]。
#    - solver 仍先恢复完整绕障 route，再只涂 route[0], route[2], route[4], ...，
#      所以输出变成虚线，可见像素数是 ceil(L/2)。
#
# 2. Complexity calibration:
#    - 只引入两条稳定变化：source 变直、output 变虚线；bar / obstacle / wrapping route 的核心几何保留。
#    - source 的方向不开放成任意水平/垂直，而是固定为“平行边界、垂直于 bar”，避免样例不足时出现
#      “有时侧向、有时前向”的不可归纳混杂情况。
#
# 3. Main() assumptions audit:
#    - 原 solver 用一般细线提取逻辑推断 segment 主干；融合后若继续允许任意 irregular source，
#      就会退化回原题，因此 segment 选择必须显式要求 straight-line component。
#    - 原 solver 把整条旧 segment 都当作可走区域；对直线 source 这样做会诱导 pathfinder 沿旧直线横向滑行，
#      偏离 generator 的 output-first route。因此这里把 approach path 的 walkable 收紧为 bg-only。
#    - 原 apex-tightness 断言依赖 seg_start 相对 terminal 是侧向一步；这也是为什么 generator 必须把
#      直线 source 放在 terminal 侧边，而不是正前方。
#
# 4. Placement legality:
#    - generator 仍采用 output-first：先造完整 output_route，再派生 dashed output_seg。
#    - reserve 继续覆盖完整 output_route，而不是只覆盖可见虚线像素，防止多 triple 挤进虚线空档。
#    - obstacle 生长时除了避开 bar 和 output_route，还要避开 input straight source 的 4-邻域缓冲，
#      否则 obstacle 会贴到直线源上，让输入看起来像对象粘连。
#
# 5. Visual observability:
#    - 人类能直接看到三件事反复出现：输入 source 都是实心直线；输出都变成绕障后的同色虚线；
#      bar 与 obstacle 不变。
#    - 虚线从 attachment cell 开始保留第 1、3、5... 格，因此依然明显是“同一条线段被重新安放”。
#
# 6. Visual cleanliness:
#    - straight source 比折线 source 更规整，配合 reserve 走廊，多个 triple 也不容易相互缠绕。
#    - obstacle 的输入缓冲减少 source 与 obstacle 的近距离噪声，输出虚线空隙里也不会混入其他对象。
#
# 7. Behavior validation target:
#    - 对每个 triple，应满足：输入 source component 是 straight line，输出该色总像素数等于各 straight source
#      长度的 ceil(L/2) 之和。若生成样本中 source 常常不是直线，或输出像素数仍接近 L，则说明融合退化。
import random
import re
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, paint_coords


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        cols = sorted(set(cs))
        return cols == list(range(min(cols), max(cols) + 1))
    if len(set(cs)) == 1:
        rows = sorted(set(rs))
        return rows == list(range(min(rows), max(rows) + 1))
    return False


def _bar_endpoints(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        c = cs[0]
        return (min(rs), c), (max(rs), c)
    r = rs[0]
    return (r, min(cs)), (r, max(cs))


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _bfs_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _bfs_path_with_diag_filter(
    start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray
) -> list[tuple[int, int]] | None:
    h, w = walkable.shape
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}

    def can_step(r: int, c: int, nr: int, nc: int) -> bool:
        dr = nr - r
        dc = nc - c
        if dr == 0 or dc == 0:
            return True
        mid1 = (r, c + dc)
        mid2 = (r + dr, c)
        if 0 <= mid1[0] < h and 0 <= mid1[1] < w and bool(walkable[mid1]):
            return False
        if 0 <= mid2[0] < h and 0 <= mid2[1] < w and bool(walkable[mid2]):
            return False
        return True

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in prev:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            if not can_step(r, c, nr, nc):
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _greedy_manhattan_path(start: tuple[int, int], goal: tuple[int, int], walkable: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = walkable.shape

    def md(p: tuple[int, int]) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    cur = start
    path = [cur]
    seen = {cur}
    while cur != goal:
        cur_md = md(cur)
        best = None
        for nr, nc in neighbors8(cur[0], cur[1]):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (nr, nc)
            if nxt in seen:
                continue
            if not bool(walkable[nr, nc]) and nxt != goal:
                continue
            nmd = md(nxt)
            if nmd >= cur_md:
                continue
            is_diag = 1 if (nr != cur[0] and nc != cur[1]) else 0
            key = (nmd, is_diag, nr, nc)
            if best is None or key < best:
                best = key
        if best is None:
            return None
        _, _, nr, nc = best
        cur = (nr, nc)
        path.append(cur)
        seen.add(cur)
    return path


def _dashed_points(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [path[i] for i in range(0, len(path), 2)]


def _bar_forward(comp: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        return (1, 0) if min(rs) == 0 else (-1, 0)
    return (0, 1) if min(cs) == 0 else (0, -1)


def _straight_segment_info(
    comp: list[tuple[int, int]], terminal: tuple[int, int], bar_vertical: bool
) -> tuple[int, tuple[int, int], list[tuple[int, int]]] | None:
    if len(comp) < 2 or not _is_straight_line(comp):
        return None
    e0, e1 = _bar_endpoints(comp)
    endpoints = [p for p in (e0, e1) if abs(p[0] - terminal[0]) + abs(p[1] - terminal[1]) == 1]
    if len(endpoints) != 1:
        return None
    seg_start = endpoints[0]
    off_r = seg_start[0] - terminal[0]
    off_c = seg_start[1] - terminal[1]
    if bar_vertical:
        if off_r != 0 or abs(off_c) != 1:
            return None
    else:
        if off_c != 0 or abs(off_r) != 1:
            return None
    return len(comp), seg_start, comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swing a straight side-attached source line around a nearby obstacle, then repaint only alternating route cells.

    Input structure:
    - bg, bar color (straight line perpendicular to a grid edge, one endpoint on the border), segment color (a straight horizontal/vertical source attached beside the bar terminal), obstacle color (a nearby irregular sprite).

    Rule:
    - For each valid bar, locate the straight source segment whose endpoint is 4-adjacent to the terminal on the bar's side.
    - Reconstruct the same full rope-like reroute as in the base task: approach the nearest obstacle, follow its exposed skin, then keep extending in the exit direction.
    - Do not paint that whole route solid. Paint only route cells 1, 3, 5, ... from the attachment side, producing a dashed output whose visible pixel count is ceil(L/2) when the full route length is L.

    Key insight: The source is easier to recognize because it is a pure straight line, but the output no longer preserves visible length; the solver must combine the old wrapping geometry with a new every-other-cell painting rule.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            for comp in _components(g, col, conn8=False):
                if len(comp) < 2 or not _is_straight_line(comp):
                    continue
                e0, e1 = _bar_endpoints(comp)
                touch0 = _touches_border(e0, h, w)
                touch1 = _touches_border(e1, h, w)
                if touch0 == touch1:
                    continue
                rs = [r for r, _ in comp]
                cs = [c for _, c in comp]
                vertical = len(set(cs)) == 1
                border_ep = e0 if touch0 else e1
                if vertical:
                    if border_ep[0] not in (0, h - 1):
                        continue
                else:
                    if border_ep[1] not in (0, w - 1):
                        continue
                terminal_ep = e1 if touch0 else e0
                tr, tc = terminal_ep
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal_ep

    moves = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        rs_bar = [r for r, _ in bar_comp]
        cs_bar = [c for _, c in bar_comp]
        bar_vertical = len(set(cs_bar)) == 1

        neigh = []
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg and v != bar_col:
                    neigh.append((v, (nr, nc)))
        assert neigh, f"bar terminal {terminal} must have a non-bg/non-bar neighbor"

        best_seg = None
        for cand_col, cand_start in neigh:
            for comp in _components(g, cand_col, conn8=False):
                if cand_start not in comp:
                    continue
                info = _straight_segment_info(comp, terminal, bar_vertical)
                if info is None:
                    break
                L, seg_start, seg_comp = info
                key = (L, cand_col, seg_start[0], seg_start[1])
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, seg_start, seg_comp)
                break
        assert best_seg is not None, f"no valid straight segment for bar at {terminal}"
        (L, _, _, _), seg_col, seg_start, seg_comp = best_seg

        obstacle_candidates = [c for c in colors if c not in (bar_col, seg_col)]
        assert obstacle_candidates, "must have at least one obstacle color distinct from bar and segment"

        best = None
        obstacle_set = None
        for col in obstacle_candidates:
            for comp in _components(g, col, conn8=True):
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best is None or key < best:
                    best = key
                    obstacle_set = set(comp)
        assert obstacle_set is not None, "must find at least one obstacle component"

        off_r = seg_start[0] - terminal[0]
        off_c = seg_start[1] - terminal[1]

        walkable = g == bg
        walkable[terminal] = True

        def pick_p_near() -> tuple[int, int]:
            good = []
            for pr, pc in obstacle_set:
                er, ec = pr + off_r, pc + off_c
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if (er, ec) in obstacle_set:
                    continue
                if int(g[er, ec]) != bg:
                    continue
                d = abs(pr - terminal[0]) + abs(pc - terminal[1])
                good.append((d, pr, pc))
            assert good, "no obstacle pixel has a valid translated endpoint outside obstacle"
            _, pr, pc = min(good)
            return (pr, pc)

        p_near = pick_p_near()
        walkable[p_near] = True
        path_t_to_obs = _greedy_manhattan_path(terminal, p_near, walkable) or _bfs_path(terminal, p_near, walkable)
        assert path_t_to_obs is not None, f"no path from terminal {terminal} to obstacle {p_near}"
        shifted = [(r + off_r, c + off_c) for r, c in path_t_to_obs]
        assert shifted and shifted[0] == seg_start, "shifted path must start at seg_start"
        assert all(0 <= r < h and 0 <= c < w for r, c in shifted), "shifted path must stay within grid"
        entry = shifted[-1]

        skin = set()
        for r, c in obstacle_set:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacle_set:
                    skin.add((nr, nc))

        cr = sum(r for r, _ in obstacle_set) / len(obstacle_set)
        cc = sum(c for _, c in obstacle_set) / len(obstacle_set)
        away = (cr - terminal[0], cc - terminal[1])

        def skin_ok(p: tuple[int, int]) -> bool:
            return int(g[p[0], p[1]]) == bg

        skin_candidates = [p for p in skin if skin_ok(p)] + [entry]

        def exit_key(p: tuple[int, int]) -> tuple[float, float, int, int]:
            r, c = p
            md = abs(r - terminal[0]) + abs(c - terminal[1])
            dot = (r - terminal[0]) * away[0] + (c - terminal[1]) * away[1]
            return (md, dot, -r, -c)

        exit_cell = max(skin_candidates, key=exit_key)

        skin_walkable = np.zeros((h, w), dtype=bool)
        for r, c in skin:
            if skin_ok((r, c)):
                skin_walkable[r, c] = True
        skin_walkable[entry] = True
        skin_walkable[exit_cell] = True
        boundary_path = _bfs_path_with_diag_filter(entry, exit_cell, skin_walkable) or [entry]
        assert len(boundary_path) >= 2, "boundary path must have at least 2 cells to derive extension direction"
        dr_ext = boundary_path[-1][0] - boundary_path[-2][0]
        dc_ext = boundary_path[-1][1] - boundary_path[-2][1]

        route = list(shifted)
        if boundary_path and boundary_path[0] == entry:
            route.extend(boundary_path[1:])

        cur = route[-1]
        while len(route) < L:
            nr, nc = cur[0] + dr_ext, cur[1] + dc_ext
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(g[nr, nc]) != bg:
                break
            route.append((nr, nc))
            cur = (nr, nc)

        new_pts = route[:L]
        dashed_pts = _dashed_points(new_pts)
        assert not any(p in obstacle_set for p in new_pts), "output segment passes through obstacle"

        fwd = _bar_forward(bar_comp)
        step_classes = []
        for j in range(len(new_pts) - 1):
            dr = new_pts[j + 1][0] - new_pts[j][0]
            dc = new_pts[j + 1][1] - new_pts[j][1]
            fwd_comp = dr * fwd[0] + dc * fwd[1]
            perp_comp = dr * fwd[1] + dc * (-fwd[0])
            if abs(dr) + abs(dc) == 1:
                step_classes.append("F" if fwd_comp > 0 else "X")
            elif fwd_comp > 0 and perp_comp > 0:
                step_classes.append("FR")
            elif fwd_comp > 0 and perp_comp < 0:
                step_classes.append("FL")
            else:
                step_classes.append("X")
        s = "".join(step_classes)
        assert re.fullmatch(r"(FL)*(F)*(FR)*", s) or re.fullmatch(r"(FR)*(F)*(FL)*", s), (
            f"output segment not monotone wrap: {' '.join(step_classes)}"
        )

        first_step = step_classes[0] if step_classes else ""
        if first_step in ("FL", "FR"):
            p1_end = 0
            for j2, sc in enumerate(step_classes):
                if sc == first_step:
                    p1_end = j2 + 1
                else:
                    break
            apex = new_pts[p1_end]
            inner = (-off_r, -off_c)
            apex_inner = (apex[0] + inner[0], apex[1] + inner[1])
            assert apex_inner in obstacle_set, f"obstacle not tight at V apex: apex={apex}, inner={apex_inner}"

        moves.append((seg_comp, seg_col, new_pts, dashed_pts))

    for seg_comp, _, _, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg

    for _, seg_col, _, dashed_pts in moves:
        for r, c in dashed_pts:
            if int(out[r, c]) == bg:
                out[r, c] = seg_col
    return out


# --------------- generator (combo output-first) ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 7, 27
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
PHASE_MAX = 3
P1_MIN = 2
OBS_SIZE_MIN, OBS_SIZE_MAX = 3, 9
OBS_MD_MIN, OBS_MD_MAX = 2, 6
MAX_RETRIES = 400
PLACEMENT_PAD = 1

EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _terminal_for(edge: str, n: int, bar_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (bar_len - 1, pos)
    if edge == "bottom":
        return (n - bar_len, pos)
    if edge == "left":
        return (pos, bar_len - 1)
    return (pos, n - bar_len)


def _transform(
    locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int
) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lr * fr + lc * sr, tc + lr * fc + lc * sc) for lr, lc in locals_]


def _grow_obstacle(
    start: tuple[int, int], max_size: int, forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    obstacle = {start}
    for _ in range(max_size - 1):
        cands = [
            (nr, nc)
            for r, c in obstacle
            for nr, nc in neighbors4(r, c)
            if (nr, nc) not in obstacle and (nr, nc) not in forbidden
        ]
        if not cands:
            break
        obstacle.add(random.choice(cands))
    return list(obstacle)


def _obstacle_is_observable(obstacle: list[tuple[int, int]]) -> bool:
    if len(obstacle) < 3:
        return False
    rs = {r for r, _ in obstacle}
    cs = {c for _, c in obstacle}
    return len(rs) > 1 and len(cs) > 1


def _build_straight_input_path(seg_len: int, side: int) -> list[tuple[int, int]]:
    return [(0, side * k) for k in range(1, seg_len + 1)]


def _build_output_path(
    seg_len: int, side: int, p1: int, p2: int, p3: int
) -> list[tuple[int, int]]:
    coords = [(0, side)]
    phase_steps = [(1, side), (1, 0), (1, -side)]
    for phase_len, (dr, dc) in zip([p1, p2, p3], phase_steps):
        for _ in range(phase_len):
            pr, pc = coords[-1]
            coords.append((pr + dr, pc + dc))
    return coords


def _expanded_4neighbors(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(coords)
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            out.add((nr, nc))
    return out


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _make_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None

    bar_len = random.randint(BAR_LEN_MIN, max_bar)
    max_seg = min(SEG_LEN_MAX, n - bar_len + 1, n - 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)
    side = random.choice((-1, 1))

    bar = [(-k, 0) for k in range(bar_len)]
    bar_set = set(bar)
    input_seg = _build_straight_input_path(seg_len, side)
    input_buffer = _expanded_4neighbors(input_seg)

    remaining = seg_len - 1
    max_p1 = min(PHASE_MAX, remaining - 1)
    if max_p1 < P1_MIN:
        return None
    p1 = random.randint(P1_MIN, max_p1)
    remaining -= p1
    p2 = random.randint(0, min(PHASE_MAX, remaining))
    p3 = remaining - p2

    output_route = _build_output_path(seg_len, side, p1, p2, p3)
    output_seg = _dashed_points(output_route)
    route_set = set(output_route)

    apex = output_route[p1]
    apex_inner = (apex[0], apex[1] - side)
    if apex_inner in route_set or apex_inner in bar_set or apex_inner in input_buffer:
        return None

    obs_size = random.randint(OBS_SIZE_MIN, OBS_SIZE_MAX)
    obstacle = _grow_obstacle(apex_inner, obs_size, route_set | bar_set | input_buffer)
    if not _obstacle_is_observable(obstacle):
        return None
    if any(p in input_buffer for p in obstacle):
        return None

    nearest_md = min(abs(r) + abs(c) for r, c in obstacle)
    if not (OBS_MD_MIN <= nearest_md <= OBS_MD_MAX):
        return None

    union = bar + obstacle + input_seg + output_route
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "obstacle": obstacle,
        "input_seg": input_seg,
        "output_route": output_route,
        "output_seg": output_seg,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["obstacle"] + triple["input_seg"] + triple["output_route"]
    bar_len = triple["bar_len"]

    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_route": _transform(triple["output_route"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 11:
        max_triples = 1
    elif n < 15:
        max_triples = 2
    else:
        max_triples = 3
    triple_count = random.randint(1, max_triples)

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    for edge in random.sample(list(EDGE_DATA), triple_count):
        triple = _make_triple(n)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)
        paint_coords(inp, placed["input_seg"], seg_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["obstacle"], obstacle_color)
        paint_coords(out, placed["output_seg"], seg_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    if len(placed_triples) > 1:
        for i, placed in enumerate(placed_triples):
            tr, tc = placed["terminal"]
            own = min(abs(r - tr) + abs(c - tc) for r, c in placed["obstacle"])
            other = min(
                abs(r - tr) + abs(c - tc)
                for j, other_placed in enumerate(placed_triples)
                if j != i
                for r, c in other_placed["obstacle"]
            )
            if own >= other:
                return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
